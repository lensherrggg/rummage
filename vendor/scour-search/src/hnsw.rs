//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor
//! index with cosine distance and soft deletes.
//!
//! Pure Rust, no dependencies. Level assignment uses a deterministic
//! per-index PRNG, so index construction is reproducible for a given
//! insertion order.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Tunable HNSW construction/search parameters.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HnswParams {
    /// Max connections per node on layers > 0.
    pub m: usize,
    /// Max connections per node on layer 0.
    pub m_max0: usize,
    /// Candidate-list width during construction.
    pub ef_construction: usize,
    /// Candidate-list width during search (floored at `k`).
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Node {
    id: String,
    embedding: Vec<f32>,
    connections: Vec<Vec<usize>>,
    deleted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Candidate {
    distance: FloatOrd,
    node: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// In-memory approximate nearest neighbor index (cosine distance).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HnswIndex {
    dimensions: usize,
    params: HnswParams,
    level_multiplier: f32,
    rng_state: u64,
    nodes: Vec<Node>,
    id_to_index: HashMap<String, usize>,
    entry_point: Option<usize>,
    max_level: usize,
    active_len: usize,
}

impl HnswIndex {
    /// Create an index for vectors of `dimensions` with default parameters.
    pub fn new(dimensions: usize) -> Self {
        Self::with_params(dimensions, HnswParams::default())
    }

    pub fn with_params(dimensions: usize, params: HnswParams) -> Self {
        Self {
            dimensions,
            params,
            level_multiplier: 1.0 / (params.m.max(2) as f32).ln(),
            rng_state: 0x9E37_79B9_7F4A_7C15,
            nodes: Vec::new(),
            id_to_index: HashMap::new(),
            entry_point: None,
            max_level: 0,
            active_len: 0,
        }
    }

    /// Insert a vector. Re-inserting an existing id replaces it; inserting
    /// over a soft-deleted id revives it.
    ///
    /// # Panics
    /// Panics if `embedding.len() != dimensions`.
    pub fn insert(&mut self, id: &str, embedding: &[f32]) {
        assert_eq!(
            embedding.len(),
            self.dimensions,
            "embedding dimension mismatch"
        );

        if let Some(&existing) = self.id_to_index.get(id) {
            if self.nodes[existing].deleted {
                self.nodes[existing].deleted = false;
                self.nodes[existing].embedding = embedding.to_vec();
                self.active_len += 1;
                return;
            }
            self.remove(id);
        }

        let level = self.random_level();
        let node_index = self.nodes.len();
        let node = Node {
            id: id.to_string(),
            embedding: embedding.to_vec(),

            connections: vec![Vec::new(); level + 1],
            deleted: false,
        };

        if self.entry_point.is_none() {
            self.entry_point = Some(node_index);
            self.max_level = level;
            self.id_to_index.insert(id.to_string(), node_index);
            self.nodes.push(node);
            self.active_len += 1;
            return;
        }

        let mut ep = self.entry_point.unwrap();
        let mut ep_dist = cosine_distance(embedding, &self.nodes[ep].embedding);

        for layer in ((level + 1)..=self.max_level).rev() {
            let (best, best_dist) = self.greedy_search_layer(embedding, ep, ep_dist, layer);
            ep = best;
            ep_dist = best_dist;
        }

        let max_level_before = self.max_level;
        self.id_to_index.insert(id.to_string(), node_index);
        self.nodes.push(node);

        let mut current_ep = ep;
        for layer in (0..=usize::min(level, max_level_before)).rev() {
            let candidates = self.search_layer(
                embedding,
                vec![current_ep],
                self.params.ef_construction,
                layer,
            );
            let max_conn = if layer == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            let neighbors = self.select_neighbors(candidates, max_conn, Some(node_index));

            self.nodes[node_index].connections[layer] = neighbors.clone();

            for &neighbor in &neighbors {
                self.nodes[neighbor].connections[layer].push(node_index);
                if self.nodes[neighbor].connections[layer].len() > max_conn {
                    let pruned = self.prune_connections(neighbor, layer, max_conn);
                    self.nodes[neighbor].connections[layer] = pruned;
                }
            }

            if let Some(&best) = neighbors.first() {
                current_ep = best;
            }
        }

        if level > max_level_before {
            self.entry_point = Some(node_index);
            self.max_level = level;
        }

        self.active_len += 1;
    }

    /// Top-`k` nearest neighbors of `query` as `(id, cosine_distance)`,
    /// nearest first. Soft-deleted entries are excluded.
    ///
    /// # Panics
    /// Panics if `query.len() != dimensions`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if k == 0 || self.entry_point.is_none() {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dimensions, "query dimension mismatch");

        let mut ep = self.entry_point.unwrap();
        let mut ep_dist = cosine_distance(query, &self.nodes[ep].embedding);

        for layer in (1..=self.max_level).rev() {
            let (best, best_dist) = self.greedy_search_layer(query, ep, ep_dist, layer);
            ep = best;
            ep_dist = best_dist;
        }

        let mut candidates = self.search_layer(query, vec![ep], self.params.ef_search.max(k), 0);
        candidates.sort_by(|a, b| a.distance.0.total_cmp(&b.distance.0));
        candidates
            .into_iter()
            .filter(|c| !self.nodes[c.node].deleted)
            .take(k)
            .map(|c| (self.nodes[c.node].id.clone(), c.distance.0))
            .collect()
    }

    /// Soft-delete an id. Returns whether anything was deleted.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(&index) = self.id_to_index.get(id) {
            if !self.nodes[index].deleted {
                self.nodes[index].deleted = true;
                self.active_len = self.active_len.saturating_sub(1);
                return true;
            }
        }
        false
    }

    /// Number of live (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.active_len
    }

    /// Whether `id` names a live (non-deleted) node.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index
            .get(id)
            .map(|&i| !self.nodes[i].deleted)
            .unwrap_or(false)
    }

    pub fn is_empty(&self) -> bool {
        self.active_len == 0
    }

    /// Build an index from `(id, embedding)` pairs with default parameters.
    /// Dimensionality is taken from the first entry.
    pub fn build_from_embeddings(entries: &[(String, Vec<f32>)]) -> Self {
        let dimensions = entries.first().map(|(_, v)| v.len()).unwrap_or(0);
        let mut index = Self::new(dimensions);
        for (id, embedding) in entries {
            index.insert(id, embedding);
        }
        index
    }

    fn random_level(&mut self) -> usize {
        // splitmix64 step over per-index state: deterministic per insertion order.
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u = ((self.rng_state >> 11) as f64) / ((1u64 << 53) as f64);
        let u = u.max(f32::MIN_POSITIVE as f64).min(1.0 - f64::EPSILON) as f32;
        (-u.ln() * self.level_multiplier).floor() as usize
    }

    fn greedy_search_layer(
        &self,
        query: &[f32],
        entry: usize,
        entry_dist: f32,
        layer: usize,
    ) -> (usize, f32) {
        let mut current = entry;
        let mut current_dist = entry_dist;
        loop {
            let mut changed = false;
            for &neighbor in &self.nodes[current].connections[layer] {
                let dist = cosine_distance(query, &self.nodes[neighbor].embedding);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        (current, current_dist)
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut top_candidates: BinaryHeap<Candidate> = BinaryHeap::new();

        for ep in entry_points {
            if !visited.insert(ep) {
                continue;
            }
            let dist = cosine_distance(query, &self.nodes[ep].embedding);
            let candidate = Candidate {
                distance: FloatOrd(dist),
                node: ep,
            };
            candidates.push(Reverse(candidate));
            top_candidates.push(candidate);
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst = top_candidates
                .peek()
                .map(|c| c.distance.0)
                .unwrap_or(f32::INFINITY);
            if top_candidates.len() >= ef && current.distance.0 > worst {
                break;
            }

            for &neighbor in &self.nodes[current.node].connections[layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let dist = cosine_distance(query, &self.nodes[neighbor].embedding);
                let candidate = Candidate {
                    distance: FloatOrd(dist),
                    node: neighbor,
                };
                let threshold = top_candidates
                    .peek()
                    .map(|c| c.distance.0)
                    .unwrap_or(f32::INFINITY);
                if top_candidates.len() < ef || dist < threshold {
                    candidates.push(Reverse(candidate));
                    top_candidates.push(candidate);
                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }
                }
            }
        }

        top_candidates.into_sorted_vec()
    }

    fn select_neighbors(
        &self,
        mut candidates: Vec<Candidate>,
        max_neighbors: usize,
        exclude: Option<usize>,
    ) -> Vec<usize> {
        candidates.sort_by(|a, b| a.distance.0.total_cmp(&b.distance.0));
        let mut seen = HashSet::new();
        candidates
            .into_iter()
            .filter(|c| Some(c.node) != exclude)
            .filter(|c| seen.insert(c.node))
            .take(max_neighbors)
            .map(|c| c.node)
            .collect()
    }

    fn prune_connections(&self, node: usize, layer: usize, max_neighbors: usize) -> Vec<usize> {
        let embedding = &self.nodes[node].embedding;
        let candidates = self.nodes[node].connections[layer]
            .iter()
            .copied()
            .map(|neighbor| Candidate {
                distance: FloatOrd(cosine_distance(embedding, &self.nodes[neighbor].embedding)),
                node: neighbor,
            })
            .collect::<Vec<_>>();
        self.select_neighbors(candidates, max_neighbors, Some(node))
    }
}

/// Cosine distance in `[0, 2]`: `1 - cosine_similarity`. Zero vectors are
/// defined to be at distance 1 from everything.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    if a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    1.0 - similarity.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut values = Vec::with_capacity(dim);
        for _ in 0..dim {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let raw = ((x >> 40) as u32) as f32 / (u24_max() as f32);
            values.push(raw * 2.0 - 1.0);
        }
        values
    }

    const fn u24_max() -> u32 {
        (1 << 24) - 1
    }

    #[test]
    fn cosine_distance_identical_vectors_is_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_distance(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn insert_and_search_returns_correct_nearest() {
        let dim = 16;
        let mut index = HnswIndex::new(dim);
        let mut entries = Vec::new();

        for i in 0..100 {
            let v = pseudo_random_vec(dim, i as u64 + 1);
            index.insert(&format!("vec-{i}"), &v);
            entries.push((format!("vec-{i}"), v));
        }

        for q in 0..10 {
            let query = pseudo_random_vec(dim, 10_000 + q);
            let result = index.search(&query, 1);
            assert!(!result.is_empty());

            let expected = entries
                .iter()
                .map(|(id, v)| (id.clone(), cosine_distance(&query, v)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            assert_eq!(result[0].0, expected.0);
            assert!((result[0].1 - expected.1).abs() < 1e-6);
        }
    }

    #[test]
    fn remove_excludes_deleted_items_from_search() {
        let mut index = HnswIndex::new(3);
        index.insert("a", &[1.0, 0.0, 0.0]);
        index.insert("b", &[0.0, 1.0, 0.0]);
        index.insert("c", &[0.0, 0.0, 1.0]);

        assert!(index.remove("a"));
        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert!(results.iter().all(|(id, _)| id != "a"));
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn reinsert_after_remove_revives() {
        let mut index = HnswIndex::new(2);
        index.insert("a", &[1.0, 0.0]);
        index.remove("a");
        index.insert("a", &[0.0, 1.0]);

        let results = index.search(&[0.0, 1.0], 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn build_from_embeddings_matches_sequential_insert() {
        let dim = 8;
        let entries = (0..50)
            .map(|i| (format!("id-{i}"), pseudo_random_vec(dim, i as u64 + 500)))
            .collect::<Vec<_>>();

        let built = HnswIndex::build_from_embeddings(&entries);
        let mut sequential = HnswIndex::new(dim);
        for (id, emb) in &entries {
            sequential.insert(id, emb);
        }

        for q in 0..5 {
            let query = pseudo_random_vec(dim, q as u64 + 1000);
            let built_results = built.search(&query, 10);
            let sequential_results = sequential.search(&query, 10);
            assert_eq!(built_results, sequential_results);
        }
    }

    #[test]
    fn construction_is_deterministic() {
        let dim = 8;
        let entries = (0..200)
            .map(|i| (format!("id-{i}"), pseudo_random_vec(dim, i as u64 + 9)))
            .collect::<Vec<_>>();

        let a = HnswIndex::build_from_embeddings(&entries);
        let b = HnswIndex::build_from_embeddings(&entries);

        let query = pseudo_random_vec(dim, 777);
        assert_eq!(a.search(&query, 10), b.search(&query, 10));
    }

    #[test]
    fn recall_at_10_above_90_percent_on_1k_vectors() {
        let dim = 32;
        let entries: Vec<(String, Vec<f32>)> = (0..1000)
            .map(|i| (format!("v{i}"), pseudo_random_vec(dim, i as u64 + 31)))
            .collect();
        let index = HnswIndex::build_from_embeddings(&entries);

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in 0..20 {
            let query = pseudo_random_vec(dim, 50_000 + q);
            let approx: HashSet<String> = index
                .search(&query, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            let mut exact: Vec<(String, f32)> = entries
                .iter()
                .map(|(id, v)| (id.clone(), cosine_distance(&query, v)))
                .collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1));

            for (id, _) in exact.into_iter().take(10) {
                total += 1;
                if approx.contains(&id) {
                    hits += 1;
                }
            }
        }

        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.9, "recall@10 = {recall}");
    }
}
