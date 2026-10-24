//! Seeded demo corpus for the live Scour demo.
//!
//! A small, hand-curated, multi-topic corpus chosen so the three retrieval
//! legs visibly disagree — which is the whole pedagogical point of the demo:
//!
//!   * **BM25** wins when the query shares exact vocabulary with a document.
//!   * **Vector** wins when the query is a paraphrase or uses different words
//!     for the same idea (vocabulary mismatch).
//!   * **RRF fusion** is the safe default: it recovers the best of both and
//!     rarely loses to either leg alone.
//!
//! Each entry has a short human title (shown in the UI) and a body (indexed).

/// A single demo document: `(id, title, body)`.
pub type Doc = (&'static str, &'static str, &'static str);

/// The seeded corpus. ~30 documents across distinct domains.
pub const CORPUS: &[Doc] = &[
    // ---- Rust / systems ----
    (
        "rust-borrow",
        "Rust borrow checker",
        "The Rust borrow checker enforces memory safety at compile time, rejecting \
         use-after-free and data races before the program ever runs. Ownership and \
         lifetimes are checked statically, so there is no garbage collector.",
    ),
    (
        "rust-async",
        "Async Rust and futures",
        "Async Rust turns futures into state machines that an executor polls. \
         Cooperative tasks are scheduled onto a small pool of worker threads, giving \
         high concurrency without a thread per connection.",
    ),
    (
        "memory-leaks",
        "Avoiding leaks without a GC",
        "Languages without automatic garbage collection still avoid leaks through \
         deterministic destruction: when a value goes out of scope its resources are \
         released immediately. RAII ties resource lifetime to object lifetime.",
    ),
    (
        "zero-cost",
        "Zero-cost abstractions",
        "A zero-cost abstraction compiles down to the same machine code you would \
         have written by hand. Iterators, generics, and async all monomorphize away, \
         so high-level code carries no runtime penalty.",
    ),
    // ---- Search / IR ----
    (
        "bm25",
        "How BM25 ranks documents",
        "Okapi BM25 scores a document by term frequency saturated against document \
         length and weighted by inverse document frequency. Rare query terms count \
         for more; very frequent terms in long documents count for less.",
    ),
    (
        "inverted-index",
        "The inverted index",
        "A keyword search engine stores an inverted index: a map from each term to \
         the list of documents containing it. Query evaluation walks these posting \
         lists instead of scanning every document.",
    ),
    (
        "vector-search",
        "Approximate nearest neighbors",
        "Semantic search embeds text into high-dimensional vectors and finds the \
         closest ones by cosine similarity. Exact search is too slow at scale, so a \
         navigable small-world graph approximates the nearest neighbors in log time.",
    ),
    (
        "rrf",
        "Reciprocal rank fusion",
        "When you have a keyword ranking and a semantic ranking, reciprocal rank \
         fusion merges them by summing one over the rank of each result. It needs no \
         score calibration and reliably beats either ranked list on its own.",
    ),
    (
        "hybrid-why",
        "Why hybrid retrieval wins",
        "Lexical retrieval is precise but brittle to wording; dense retrieval \
         generalizes but can drift off-topic. Combining both legs covers each one's \
         blind spot, which is why modern retrieval pipelines fuse keyword and vector \
         results rather than choosing one.",
    ),
    (
        "stemming",
        "Stemming and tokenization",
        "Before indexing, an analyzer lowercases text, removes high-frequency \
         stopwords, and reduces words to their stems so that running, runs, and ran \
         all match the same term. Good tokenization is half of good recall.",
    ),
    // ---- Cooking (clear semantic neighbor cluster, different vocabulary) ----
    (
        "pasta",
        "Cooking pasta properly",
        "Cook pasta in plenty of heavily salted boiling water, then finish the \
         noodles directly in the sauce so the starch binds everything together. \
         Reserve a little cooking water to loosen the sauce.",
    ),
    (
        "ragu",
        "A slow-simmered ragu",
        "A good ragu builds flavor over hours. Brown the meat, soften the soffritto, \
         deglaze with wine, then let the sauce barely bubble for a long time so it \
         reduces and deepens.",
    ),
    (
        "bread",
        "Baking sourdough bread",
        "Sourdough rises from a wild yeast starter rather than commercial yeast. A \
         long cold fermentation develops flavor, and a hot dutch oven traps steam to \
         give the loaf a crackling crust.",
    ),
    (
        "knife-skills",
        "Sharp knives and prep",
        "Most of cooking is preparation. A sharp knife and steady cuts make even \
         dicing safer and faster, and laying out every ingredient before you turn on \
         the heat keeps a busy stovetop calm.",
    ),
    // ---- Astronomy (vocabulary-mismatch showcase) ----
    (
        "galaxies",
        "Distant galaxies",
        "Telescopes resolve galaxies whose light has traveled for billions of years. \
         Because that light is so old, looking far across space is also looking far \
         back in time toward the early universe.",
    ),
    (
        "orbits",
        "Why orbits are stable",
        "A stable orbit is a balance: gravity pulls a body inward while its sideways \
         velocity carries it past, so it keeps falling around the central mass \
         without ever hitting it.",
    ),
    (
        "black-holes",
        "What a black hole is",
        "A black hole is a region where gravity is so strong that nothing, not even \
         light, can escape past the event horizon. They form when a massive star \
         collapses at the end of its life.",
    ),
    (
        "telescope-types",
        "Refractors and reflectors",
        "Optical telescopes gather light with either a lens or a mirror. Large \
         research instruments use mirrors because a big lens sags under its own \
         weight, while a mirror can be supported from behind.",
    ),
    // ---- Climbing / outdoors ----
    (
        "belay",
        "Belaying a climber safely",
        "The belayer manages the rope so a falling climber is caught quickly and \
         gently. Keeping a brake hand on the rope at all times is the one rule that \
         is never broken.",
    ),
    (
        "knots",
        "Essential climbing knots",
        "A figure-eight follow-through ties the rope to the harness, while a clove \
         hitch lets you adjust your position at an anchor without untying. Dress and \
         set every knot before you trust it.",
    ),
    // ---- Music ----
    (
        "scales",
        "Practicing scales",
        "Scales build the muscle memory and ear that improvisation depends on. \
         Practicing slowly with a metronome and only speeding up once each note is \
         clean beats racing through mistakes.",
    ),
    (
        "chords",
        "How chords are built",
        "A chord stacks intervals on a root note: a major triad is the root, a major \
         third, and a perfect fifth. Adding a seventh or a ninth gives the richer, \
         tenser colors of jazz harmony.",
    ),
    // ---- Health / fitness ----
    (
        "sleep",
        "Why sleep matters",
        "During deep sleep the brain consolidates memories and clears metabolic \
         waste, while the body repairs tissue. Consistent timing matters more than \
         the occasional long lie-in.",
    ),
    (
        "running",
        "Building running endurance",
        "Endurance comes from a base of easy aerobic miles, not from going hard \
         every day. Most runs should be slow enough to hold a conversation, with \
         only one or two harder efforts a week.",
    ),
    // ---- Finance ----
    (
        "compound",
        "The power of compounding",
        "Compound growth means returns earn their own returns, so small consistent \
         contributions snowball over decades. Time in the market matters far more \
         than trying to time the market.",
    ),
    (
        "diversification",
        "Spreading risk",
        "Holding many uncorrelated assets smooths the ride: when one falls another \
         may rise, so the portfolio as a whole swings less than any single holding. \
         You give up the best case to avoid the worst.",
    ),
    // ---- Distributed systems (semantic neighbor of search/Rust) ----
    (
        "consensus",
        "Distributed consensus",
        "Replicated systems agree on a single order of operations using a consensus \
         protocol like Raft, so the cluster keeps working and stays consistent even \
         when some machines crash or messages are delayed.",
    ),
    (
        "caching",
        "Caching and invalidation",
        "A cache trades freshness for speed by keeping a copy of expensive results \
         close to where they are used. The hard part is invalidation: knowing when \
         the cached copy has gone stale and must be recomputed.",
    ),
    (
        "load-balancing",
        "Spreading load across servers",
        "A load balancer distributes incoming requests across a fleet of servers so \
         no single machine is overwhelmed, and routes traffic away from instances \
         that fail their health checks.",
    ),
    (
        "rate-limiting",
        "Protecting a service with rate limits",
        "Rate limiting caps how many requests a client can make in a window, \
         protecting a service from abuse and accidental overload. A token bucket \
         allows short bursts while bounding the long-run average.",
    ),
];

/// A few curated example queries the UI offers as one-click buttons, chosen to
/// make the lanes disagree in instructive ways.
pub const EXAMPLE_QUERIES: &[&str] = &[
    "how does rust prevent memory bugs",
    "simmering a flavorful tomato sauce",
    "light from the early universe",
    "merge keyword and semantic rankings",
    "keeping a cache from going stale",
    "earning interest on interest over time",
];
