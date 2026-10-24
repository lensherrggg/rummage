//! Okapi BM25 keyword search over an in-memory inverted index.
//!
//! Documents are analyzed with [`crate::text::tokenize`] (lowercase →
//! stopword removal → Porter stemming). Standard BM25 parameters
//! `k1 = 1.2`, `b = 0.75`.

use crate::text::tokenize;
use std::collections::{HashMap, HashSet};

const K1: f64 = 1.2;
const B: f64 = 0.75;

/// In-memory BM25 index. Cloneable; cheap to snapshot.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bm25Index {
    inverted_index: HashMap<String, Vec<(String, f32)>>,
    doc_lengths: HashMap<String, usize>,
    avg_doc_length: f64,
    doc_count: usize,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Add (or replace) a document.
    pub fn add_document(&mut self, doc_id: &str, text: &str) {
        if self.doc_lengths.contains_key(doc_id) {
            self.remove_document(doc_id);
        }

        let tokens = tokenize(text);
        let doc_length = tokens.len();
        let mut term_counts: HashMap<String, usize> = HashMap::new();

        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }

        for (term, count) in term_counts {
            self.inverted_index
                .entry(term)
                .or_default()
                .push((doc_id.to_string(), count as f32));
        }

        self.doc_lengths.insert(doc_id.to_string(), doc_length);
        self.recalculate_stats();
    }

    /// Remove a document. No-op if absent.
    pub fn remove_document(&mut self, doc_id: &str) {
        if self.doc_lengths.remove(doc_id).is_none() {
            return;
        }

        self.inverted_index.retain(|_, postings| {
            postings.retain(|(existing_doc_id, _)| existing_doc_id != doc_id);
            !postings.is_empty()
        });

        self.recalculate_stats();
    }

    /// Top-`k` documents for `query`, scored by Okapi BM25, best first.
    /// Ties break lexicographically by document id for determinism.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        if k == 0 || self.doc_count == 0 {
            return Vec::new();
        }

        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut seen_terms = HashSet::new();

        for term in query_terms {
            if !seen_terms.insert(term.clone()) {
                continue;
            }

            let Some(postings) = self.inverted_index.get(&term) else {
                continue;
            };

            let n_qi = postings.len() as f64;
            let idf = ((self.doc_count as f64 - n_qi + 0.5) / (n_qi + 0.5) + 1.0).ln();

            for (doc_id, term_freq) in postings {
                let doc_length = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f64;
                let freq = *term_freq as f64;
                let norm = K1 * (1.0 - B + B * doc_length / self.avg_doc_length.max(1.0));
                let score = idf * (freq * (K1 + 1.0)) / (freq + norm);
                *scores.entry(doc_id.clone()).or_insert(0.0) += score;
            }
        }

        let mut results: Vec<(String, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
    }

    /// Build an index from `(doc_id, text)` pairs.
    pub fn build_from_documents(docs: &[(String, String)]) -> Self {
        let mut index = Self::new();
        for (doc_id, text) in docs {
            index.add_document(doc_id, text);
        }
        index
    }

    fn recalculate_stats(&mut self) {
        self.doc_count = self.doc_lengths.len();
        let total_length: usize = self.doc_lengths.values().sum();
        self.avg_doc_length = if self.doc_count == 0 {
            0.0
        } else {
            total_length as f64 / self.doc_count as f64
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_relevant_documents_first() {
        let docs = vec![
            (
                "doc1".to_string(),
                "Rust provides memory safety and fearless concurrency for systems programming"
                    .to_string(),
            ),
            (
                "doc2".to_string(),
                "Basketball players score points with dunks and three pointers".to_string(),
            ),
            (
                "doc3".to_string(),
                "Cooking pasta requires boiling water and preparing a flavorful sauce".to_string(),
            ),
            (
                "doc4".to_string(),
                "Search engines use indexing and ranking algorithms for text retrieval".to_string(),
            ),
            (
                "doc5".to_string(),
                "Space telescopes observe distant galaxies and stellar connections".to_string(),
            ),
        ];

        let index = Bm25Index::build_from_documents(&docs);
        let results = index.search("search indexing retrieval", 3);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, "doc4");
    }

    #[test]
    fn empty_query_returns_empty_results() {
        let docs = vec![("doc1".to_string(), "Simple document text".to_string())];
        let index = Bm25Index::build_from_documents(&docs);

        assert!(index.search("", 5).is_empty());
        assert!(index.search("the and is", 5).is_empty());
    }

    #[test]
    fn remove_document_excludes_it_from_search() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "Rust language and compiler optimizations");
        index.add_document("doc2", "Gardening tips for trees and flowers");

        let before = index.search("rust compiler", 5);
        assert_eq!(before[0].0, "doc1");

        index.remove_document("doc1");
        let after = index.search("rust compiler", 5);

        assert!(after.iter().all(|(doc_id, _)| doc_id != "doc1"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn re_adding_a_document_replaces_it() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "rust compiler internals");
        index.add_document("doc1", "gardening with flowers");

        assert_eq!(index.len(), 1);
        assert!(index.search("rust", 5).is_empty());
        assert_eq!(index.search("gardening", 5)[0].0, "doc1");
    }

    #[test]
    fn deterministic_tie_break() {
        let mut index = Bm25Index::new();
        index.add_document("b", "identical content here");
        index.add_document("a", "identical content here");

        let results = index.search("identical content", 2);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");
    }
}
