//! Deterministic, dependency-free text embeddings for the live demo.
//!
//! Scour's library is deliberately *bring-your-own-embeddings* — it indexes
//! any `&[f32]` and never picks a model for you. But a recruiter-facing demo
//! needs a vector leg that actually means something on arbitrary typed
//! queries, with **no model download, no GPU, no SaaS, and no new crates**.
//!
//! This module is that embedder: a pure-`std` feature-hashing encoder.
//!
//! ## How it works
//!
//! Each piece of text is turned into a sparse bag of features:
//!   * stemmed token unigrams (shares Scour's own analyzer, so the vector
//!     space agrees with BM25 on vocabulary), and
//!   * character trigrams over each raw token (so morphology and typos —
//!     `colour`/`color`, `optimise`/`optimize` — land near each other even
//!     when the stems differ).
//!
//! Every feature is hashed (FNV-1a) into one of `dim` buckets with a signed
//! contribution (`+1`/`-1` from a second hash bit), then the vector is
//! L2-normalized. This is the classic *hashing trick* / *random projection*
//! used in production retrieval when you want a fixed-width, deterministic,
//! model-free embedding. Cosine distance over these vectors rewards shared
//! sub-word structure — which is exactly the signal that lets the semantic
//! leg rescue queries the keyword leg misses.
//!
//! It is not a transformer. It does not pretend to be. It is an honest,
//! auditable stand-in that makes the *architecture* — three retrieval legs
//! fused — demonstrable end to end, which is the point of the demo.

use crate::text::tokenize;

/// Embedding width used by the demo corpus and query encoder.
pub const DEMO_DIM: usize = 512;

/// Encode `text` into a deterministic L2-normalized embedding of `dim` floats.
///
/// Same input + same `dim` always yields the same vector (no RNG, no state),
/// so the demo is fully reproducible.
///
/// The feature set is layered so that **topical content words dominate** and
/// character structure only nudges:
///   * stemmed token unigrams (weight 1.0) — the primary topical signal,
///     sharing Scour's analyzer so the space agrees with BM25 on vocabulary;
///   * stemmed token bigrams (weight 0.6) — light phrase/co-occurrence signal;
///   * character trigrams (weight 0.2) — a small fuzzy assist so morphology
///     and spelling variants still pull together, without letting common
///     substrings like `ing`/`the` swamp the topical signal.
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim.max(1)];

    let toks = tokenize(text);

    // Leg 1: stemmed token unigrams — the dominant topical signal.
    for tok in &toks {
        add_feature(&mut v, dim, &format!("w:{tok}"), 1.0);
    }

    // Leg 2: stemmed token bigrams — phrase / co-occurrence signal.
    for pair in toks.windows(2) {
        add_feature(&mut v, dim, &format!("b:{}_{}", pair[0], pair[1]), 0.6);
    }

    // Leg 3: character trigrams over content tokens only — a *light* fuzzy
    // assist. Skipping stopwords and stems shorter than 4 chars keeps high
    // frequency substrings (`the`, `ing`) from dominating the space.
    for tok in &toks {
        if tok.chars().count() < 4 {
            continue;
        }
        let padded = format!("^{tok}$");
        let chars: Vec<char> = padded.chars().collect();
        for w in chars.windows(3) {
            let tri: String = w.iter().collect();
            add_feature(&mut v, dim, &format!("c:{tri}"), 0.2);
        }
    }

    l2_normalize(&mut v);
    v
}

/// Add a signed feature contribution into the hashed vector.
fn add_feature(v: &mut [f32], dim: usize, feature: &str, weight: f32) {
    let h = fnv1a(feature.as_bytes());
    let bucket = (h % dim as u64) as usize;
    // Independent sign bit from a salted hash so it is uncorrelated with the
    // bucket choice.
    let sign = if fnv1a_salted(feature.as_bytes(), 0x9E37_79B9) & 1 == 0 {
        1.0
    } else {
        -1.0
    };
    v[bucket] += sign * weight;
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// FNV-1a 64-bit hash — small, fast, dependency-free, good enough for the
/// hashing trick (we are not hashing for cryptographic strength).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv1a_salted(bytes: &[u8], salt: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ salt;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn deterministic() {
        assert_eq!(
            embed("rust memory safety", DEMO_DIM),
            embed("rust memory safety", DEMO_DIM)
        );
    }

    #[test]
    fn normalized() {
        let v = embed("hierarchical navigable small world graphs", DEMO_DIM);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm was {norm}");
    }

    #[test]
    fn related_text_is_closer_than_unrelated() {
        let q = embed("rust memory safety", DEMO_DIM);
        let related = embed("the rust borrow checker guarantees memory safety", DEMO_DIM);
        let unrelated = embed("slow simmered tomato pasta sauce", DEMO_DIM);
        assert!(
            cos(&q, &related) > cos(&q, &unrelated),
            "related {} should beat unrelated {}",
            cos(&q, &related),
            cos(&q, &unrelated)
        );
    }

    #[test]
    fn morphology_brings_word_variants_near() {
        // Shared content stems ("galax", "telescop") should pull astronomy
        // text together far more than an unrelated cooking sentence.
        let q = embed("distant galaxies and telescopes", DEMO_DIM);
        let related = embed("a telescope resolving a faraway galaxy", DEMO_DIM);
        let unrelated = embed("slow simmered tomato pasta sauce", DEMO_DIM);
        assert!(cos(&q, &related) > cos(&q, &unrelated));
    }

    #[test]
    fn empty_is_zero_vector() {
        let v = embed("", DEMO_DIM);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
