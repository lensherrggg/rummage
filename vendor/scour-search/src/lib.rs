//! # Scour
//!
//! Embeddable **hybrid search primitives** for Rust — the in-process core
//! of a retrieval pipeline, with **zero dependencies**:
//!
//! - [`bm25::Bm25Index`] — Okapi BM25 keyword search (inverted index,
//!   Porter stemming, stopword removal)
//! - [`hnsw::HnswIndex`] — HNSW approximate nearest neighbor vector index
//!   (cosine distance, soft deletes, deterministic construction)
//! - [`fuse::rrf_fuse`] — Reciprocal Rank Fusion to merge lexical and
//!   semantic rankings
//! - [`chunk::chunk_text`] — UTF-8-safe, boundary-aware text chunking for
//!   embedding pipelines
//! - [`HybridIndex`] — the four combined: one type that indexes text +
//!   vectors and serves fused hybrid queries
//!
//! ## Example
//!
//! ```
//! use scour::HybridIndex;
//!
//! let mut index = HybridIndex::new(3); // 3-dim embeddings for the demo
//! index.add("doc-a", "rust systems programming", &[1.0, 0.0, 0.0]);
//! index.add("doc-b", "gardening in spring", &[0.0, 1.0, 0.0]);
//!
//! // Hybrid query: keyword text + query embedding, fused with RRF.
//! let results = index.search("rust programming", &[0.9, 0.1, 0.0], 2);
//! assert_eq!(results[0].0, "doc-a");
//! ```
//!
//! Bring your own embeddings: Scour is deliberately model-agnostic. Any
//! `&[f32]` works — ONNX, candle, an HTTP embedding service, or test
//! fixtures.

pub mod bm25;
pub mod chunk;
pub mod corpus;
pub mod embed;
pub mod fuse;
pub mod hnsw;
pub mod text;

pub use bm25::Bm25Index;
pub use chunk::{chunk_text, chunk_text_with_overlap};
pub use embed::{embed, DEMO_DIM};
pub use fuse::{rrf_fuse, rrf_fuse_scored, DEFAULT_RRF_K};
pub use hnsw::{cosine_distance, HnswIndex, HnswParams};

/// A combined lexical + vector index serving RRF-fused hybrid queries.
///
/// Wraps a [`Bm25Index`] and an [`HnswIndex`] under one id space.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HybridIndex {
    lexical: Bm25Index,
    vector: HnswIndex,
    rrf_k: f64,
}

impl HybridIndex {
    /// Create a hybrid index for embeddings of `dimensions`.
    pub fn new(dimensions: usize) -> Self {
        Self::with_params(dimensions, HnswParams::default(), DEFAULT_RRF_K)
    }

    pub fn with_params(dimensions: usize, hnsw: HnswParams, rrf_k: f64) -> Self {
        Self {
            lexical: Bm25Index::new(),
            vector: HnswIndex::with_params(dimensions, hnsw),
            rrf_k,
        }
    }

    /// Index a document under `id` with its text and embedding.
    /// Re-adding an id replaces both representations.
    pub fn add(&mut self, id: &str, text: &str, embedding: &[f32]) {
        self.lexical.add_document(id, text);
        self.vector.insert(id, embedding);
    }

    /// Remove a document from both indexes. Returns whether the id existed
    /// in the vector index.
    pub fn remove(&mut self, id: &str) -> bool {
        self.lexical.remove_document(id);
        self.vector.remove(id)
    }

    /// Number of live documents (vector-index count).
    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vector.is_empty()
    }

    /// Add only the lexical representation (used when semantic embeddings are
    /// unavailable, or when backfilling vectors later).
    pub fn add_lexical(&mut self, id: &str, text: &str) {
        self.lexical.add_document(id, text);
    }

    /// Add only the vector representation for an id already present lexically.
    pub fn add_vector(&mut self, id: &str, embedding: &[f32]) {
        self.vector.insert(id, embedding);
    }

    /// Whether the vector index currently contains a live node with `id`.
    pub fn contains_vector(&self, id: &str) -> bool {
        self.vector.contains(id)
    }

    /// Hybrid query: BM25 over `query_text`, ANN over `query_embedding`,
    /// fused with RRF. Returns `(id, fused_score)` best-first.
    ///
    /// Each leg retrieves `k * 3` candidates (a standard over-fetch so the
    /// fusion has material to work with), and the fused list is truncated
    /// to `k`.
    pub fn search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        k: usize,
    ) -> Vec<(String, f64)> {
        if k == 0 {
            return Vec::new();
        }
        let fetch = k.saturating_mul(3);
        let lexical = self.lexical.search(query_text, fetch);
        let semantic = self.vector.search(query_embedding, fetch);

        let mut fused = rrf_fuse_scored(&lexical, &semantic, self.rrf_k);
        fused.truncate(k);
        fused
    }

    /// Lexical-only query (BM25).
    pub fn search_lexical(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        self.lexical.search(query, k)
    }

    /// Semantic-only query (HNSW, cosine distance — smaller is closer).
    pub fn search_semantic(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        self.vector.search(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_prefers_doc_matching_both_signals() {
        let mut index = HybridIndex::new(4);
        // doc-a: strong on both keyword and vector.
        index.add(
            "doc-a",
            "rust async runtime internals",
            &[1.0, 0.0, 0.0, 0.0],
        );
        // doc-b: keyword match only.
        index.add("doc-b", "rust cookbook recipes", &[0.0, 1.0, 0.0, 0.0]);
        // doc-c: vector match only.
        index.add("doc-c", "tokio scheduler design", &[0.9, 0.1, 0.0, 0.0]);

        let results = index.search("rust runtime", &[1.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(results[0].0, "doc-a");
    }

    #[test]
    fn remove_drops_from_both_legs() {
        let mut index = HybridIndex::new(2);
        index.add("x", "unique pelican words", &[1.0, 0.0]);
        index.add("y", "other content", &[0.0, 1.0]);

        assert!(index.remove("x"));
        let results = index.search("pelican", &[1.0, 0.0], 5);
        assert!(results.iter().all(|(id, _)| id != "x"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut index = HybridIndex::new(2);
        index.add("x", "abc", &[1.0, 0.0]);
        assert!(index.search("abc", &[1.0, 0.0], 0).is_empty());
    }
}
