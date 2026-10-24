# Scour

[![CI](https://github.com/SA-Ark/scour/actions/workflows/ci.yml/badge.svg)](https://github.com/SA-Ark/scour/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen.svg)

**Hybrid search primitives for Rust you can embed anywhere — BM25, HNSW, reciprocal-rank fusion, and UTF-8-safe chunking. Zero dependencies.**

Scour is the in-process retrieval core pulled out of a production semantic-memory engine that serves 180,000+ documents. It gives you the four primitives every RAG or search pipeline ends up needing, in plain Rust with no dependency tree at all. Drop it into a CLI, a server, a WASM target, or a test harness without dragging in tokio, an ANN library, and a tokenizer framework to get four functions.

## ▶ Live demo — try it yourself

**[scour.chakrakali.com](https://scour.chakrakali.com)** — type a query and watch the **same corpus** ranked three ways, side by side:

| Lane | What it does | Where it shines |
|---|---|---|
| **BM25 keyword** | Okapi BM25 over an inverted index | Exact-vocabulary matches — precise, but brittle to wording |
| **Vector (HNSW)** | Approximate nearest neighbors, cosine distance | Paraphrases & synonyms the keyword lane misses |
| **Hybrid (RRF)** | Reciprocal-rank fusion of both | The safe default — recovers the best of both legs |

The hybrid lane tags each hit with a provenance badge (`both · L#2 V#1`, `vector #2`) so you can see which leg found it and at what rank — which is really the whole argument for hybrid, made visible. Try one of the example queries like *"spread incoming traffic across machines"*, a paraphrase with almost no keyword overlap: BM25 leans on the single shared word, the vector lane pulls in the distributed-systems neighbours, and RRF promotes the doc both legs agree on.

The whole demo is one self-contained binary. It indexes a ~30-document seeded corpus at startup and serves both the UI and a JSON `/api/search` endpoint — no database, no model server, no SaaS. The vector lane's embeddings come from a deterministic feature-hashing encoder written in pure Rust ([`src/embed.rs`](src/embed.rs)), because the demo had to stay as dependency-free as the library it's showing off.

### Run the demo locally

```bash
cargo run --release --bin scour-demo      # then open http://127.0.0.1:8087
# bind elsewhere:  SCOUR_ADDR=0.0.0.0:9000 cargo run --release --bin scour-demo
```

API: `GET /api/search?q=<query>&k=8` → `{ query, count, lexical[], semantic[], fused[] }`,
where each `fused[]` item carries a `from: { lexical, semantic }` rank provenance object.


```rust
use scour::HybridIndex;

let mut index = HybridIndex::new(384); // your embedding width

index.add("doc-1", "the rust borrow checker enforces memory safety", &embedding_1);
index.add("doc-2", "slow-simmered ragu builds flavor over hours",   &embedding_2);

// lexical + semantic legs, fused with Reciprocal Rank Fusion
let hits = index.search("memory safety in rust", &query_embedding, 10);
```

## Architecture

```
                              HybridIndex
        ┌──────────────────────────┴──────────────────────────┐
        │                                                     │
   Bm25Index                                             HnswIndex
   ┌─────────────────────────┐                ┌──────────────────────────────┐
   │ inverted index           │                │ hierarchical NSW graph       │
   │ Okapi BM25 (k1=1.2,b=.75)│                │ cosine distance              │
   │                          │                │ soft deletes + revive        │
   │   text::tokenize         │                │ deterministic construction   │
   │   ├─ lowercase/split     │                │ tunable M / ef params        │
   │   ├─ stopword filter     │                └──────────────┬───────────────┘
   │   └─ Porter stemmer      │                               │
   └────────────┬─────────────┘                               │
                │       ranked lexical        ranked semantic │
                └──────────────┐      ┌───────────────────────┘
                               ▼      ▼
                        fuse::rrf_fuse  (1 / (k + rank), k = 60)
                               │
                               ▼
                      fused, deterministic top-k

   chunk::chunk_text — boundary-aware (¶ → sentence → line), UTF-8-safe,
   lossless: chunks always reassemble to the original text byte-for-byte
```

## What's in the box

| Module | What it is |
|---|---|
| `bm25` | Okapi BM25 over an in-memory inverted index; deterministic tie-breaks; add/remove/replace documents |
| `hnsw` | Pure-Rust HNSW ANN index: cosine distance, soft deletes with revive-on-reinsert, reproducible builds (per-index PRNG), tunable `HnswParams` |
| `fuse` | Reciprocal Rank Fusion (Cormack et al., 2009) — merge any number of ranked lists without score normalization; generic over id type |
| `text` | Unicode-aware tokenizer + English stopwords + a full five-step Porter stemmer |
| `chunk` | Chunking for embedding pipelines: paragraph/sentence/line boundary preference, never splits a UTF-8 code point, optional overlap, lossless round-trip |
| `HybridIndex` | All of it behind one type: `add`, `remove`, `search` (fused), `search_lexical`, `search_semantic` |

## Benchmarks

`cargo run --release --example bench` — 20,000 docs, 384-dim embeddings, 1,000 queries, single thread, Intel i7-13700H. The workload is seeded, so the numbers reproduce.

| Operation | Result |
|---|---|
| HNSW build (20K × 384-dim, ef_construction=200) | ~45 s (≈ 447 inserts/s) |
| HNSW search, k=10 | **~0.9 ms/query** |
| HNSW recall@10 vs exact scan | **≥ 90%** (enforced by a unit test, not just claimed) |
| BM25 build (~100-word docs) | ≈ 22,000 docs/s |
| BM25 search, selective terms (typical) | **106 µs/query** |
| BM25 search, dense terms (worst case: every query term in ~half the corpus) | 10.5 ms/query |
| Hybrid (BM25 + HNSW + RRF), k=10 | dominated by the legs above |
| Chunking | ≈ 497 MB/s |

The worst-case BM25 row is there on purpose. Posting-list density is what drives the cost, and a benchmark that quietly drops the dense case isn't telling you anything.

## Design decisions

**Zero dependencies.** The whole crate is `std`. This isn't minimalism for its own sake — it means you can audit the thing in one sitting, compile it to WASM, and never get woken up by someone else's dependency churn.

**Determinism everywhere.** BM25 ties break lexicographically. HNSW level assignment uses a per-index seeded PRNG, so the same insertion order gives you the same graph and the same results. RRF ties break by first appearance. The difference between a debuggable pipeline and a haunted one is whether it does the same thing twice.

**Soft deletes in HNSW.** Really deleting a node from a navigable small-world graph means re-linking its neighbours; the pragmatic answer is to mark it deleted and filter at query time, and `insert` revives a deleted id in place.

**Bring your own embeddings.** Any `&[f32]` works. Scour won't pick an embedding model, an HTTP client, or a runtime for you — that's your call, not the library's.

**Lossless chunking.** `chunk_text(t, n).concat() == t` is a tested invariant. Chunkers that silently drop or duplicate bytes corrupt a RAG corpus in ways you don't notice until weeks later.

## Quickstart

```toml
[dependencies]
scour-search = { git = "https://github.com/SA-Ark/scour" } # crates.io publish pending
```

```rust
use scour::{chunk_text, Bm25Index, HnswIndex, HybridIndex, rrf_fuse};

// 1. chunk a document for your embedding pipeline
let chunks = chunk_text(&long_document, 2000);

// 2. or use the pieces standalone
let mut bm25 = Bm25Index::new();
bm25.add_document("a", "fearless concurrency in rust");
let lexical_hits = bm25.search("rust concurrency", 5);

let mut ann = HnswIndex::new(384);
ann.insert("a", &embedding);
let semantic_hits = ann.search(&query_embedding, 5);

// 3. or fuse arbitrary ranked lists from anywhere
let fused = rrf_fuse(&[list_a, list_b, list_c], scour::DEFAULT_RRF_K);
```

Run the test suite and benchmark:

```bash
cargo test
cargo run --release --example bench
```

## Provenance

This came out of the retrieval core of a production memory system — 180K+ documents, hybrid semantic + keyword recall. It got hardened on the way out: the chunker learned UTF-8 code-point safety (the original could split a multibyte character), HNSW construction became deterministic per index, and every module brought its tests along.

## License

MIT — see [LICENSE](LICENSE).
