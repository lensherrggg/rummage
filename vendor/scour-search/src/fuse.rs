//! Reciprocal Rank Fusion (RRF) for combining ranked result lists.
//!
//! RRF is the standard way to merge lexical (BM25) and semantic (vector)
//! rankings without score normalization: each list contributes
//! `1 / (k + rank)` per item, and the sums are re-ranked. `k = 60` is the
//! widely used default from Cormack et al. (2009).

use std::collections::HashMap;
use std::hash::Hash;

/// Default RRF constant.
pub const DEFAULT_RRF_K: f64 = 60.0;

/// Fuse multiple ranked lists (best-first) into one ranking.
///
/// Each input list is a sequence of ids ordered best-first. Output is
/// `(id, fused_score)` sorted by descending score; ties break by the
/// id's earliest appearance for determinism.
pub fn rrf_fuse<K>(ranked_lists: &[Vec<K>], k: f64) -> Vec<(K, f64)>
where
    K: Eq + Hash + Clone,
{
    let mut scores: HashMap<K, f64> = HashMap::new();
    let mut first_seen: HashMap<K, usize> = HashMap::new();
    let mut order = 0usize;

    for list in ranked_lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
            first_seen.entry(id.clone()).or_insert_with(|| {
                order += 1;
                order
            });
        }
    }

    let mut fused: Vec<(K, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| first_seen[&a.0].cmp(&first_seen[&b.0]))
    });
    fused
}

/// Convenience: fuse two scored lists, ignoring their native scores and
/// using only rank order (the whole point of RRF).
pub fn rrf_fuse_scored<K, S1, S2>(a: &[(K, S1)], b: &[(K, S2)], k: f64) -> Vec<(K, f64)>
where
    K: Eq + Hash + Clone,
{
    let list_a: Vec<K> = a.iter().map(|(id, _)| id.clone()).collect();
    let list_b: Vec<K> = b.iter().map(|(id, _)| id.clone()).collect();
    rrf_fuse(&[list_a, list_b], k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_in_both_lists_outranks_single_list_items() {
        let lexical = vec!["a", "b", "c"];
        let semantic = vec!["d", "b", "e"];

        let fused = rrf_fuse(&[lexical, semantic], DEFAULT_RRF_K);
        assert_eq!(fused[0].0, "b");
    }

    #[test]
    fn single_list_preserves_order() {
        let fused = rrf_fuse(&[vec!["x", "y", "z"]], DEFAULT_RRF_K);
        let ids: Vec<&str> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["x", "y", "z"]);
    }

    #[test]
    fn empty_input_is_empty() {
        let fused: Vec<(String, f64)> = rrf_fuse(&[], DEFAULT_RRF_K);
        assert!(fused.is_empty());
    }

    #[test]
    fn scored_lists_fuse_by_rank_not_score() {
        // Wildly different score scales must not matter.
        let bm25 = vec![("a", 17.3_f64), ("b", 4.0)];
        let vector = vec![("b", 0.02_f32), ("c", 0.9)];

        let fused = rrf_fuse_scored(&bm25, &vector, DEFAULT_RRF_K);
        assert_eq!(fused[0].0, "b");
    }

    #[test]
    fn deterministic_tie_break() {
        let fused = rrf_fuse(&[vec!["a"], vec!["b"]], DEFAULT_RRF_K);
        // Same score; "a" appeared first.
        assert_eq!(fused[0].0, "a");
        assert_eq!(fused[1].0, "b");
        assert!((fused[0].1 - fused[1].1).abs() < 1e-12);
    }
}
