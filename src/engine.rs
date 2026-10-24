use crate::config::{path_key, simplify_path, Config, RuleCache, ScopeRules};
use crate::embed::{self, Embedder};
use crate::extract::extract_text;
use crate::normalize::{normalize, normalize_query};
use crate::persist;
use crate::query::ParsedQuery;
use crate::rerank::{self, GemmaEmbedder, RerankStore};
use crate::store::{Store, StoredFile};
use crate::tokenize;
use anyhow::Result;
use scour::{chunk_text_with_overlap, rrf_fuse_scored, HybridIndex};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

pub const RESULT_LIMIT: usize = 10;
pub const EMBED_BATCH_DOCS: usize = 32;
const RRF_K: f64 = 60.0;
const CHUNK_BYTES: usize = 1024;
const CHUNK_OVERLAP: usize = 128;
const SNIPPET_CHARS: usize = 90;
const SNIPPET_CONTEXT: usize = 24;
const SNIPPET_HIGHLIGHTS: usize = 24;
const FALLBACK_DIMS: usize = 512;
const INDEX_FILE: &str = "index.bin";
const RERANK_FILE: &str = "rerank.bin";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    Disabled,
    Loading,
    Ready { dims: usize },
    Unavailable { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexPhase {
    Scan,
    Embed,
    Rerank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexProgress {
    pub phase: IndexPhase,
    pub done: usize,
    pub total: usize,
}

impl IndexProgress {
    pub fn percent(&self) -> u8 {
        if self.total == 0 {
            return 100;
        }
        (self.done.saturating_mul(100) / self.total).min(100) as u8
    }
}

#[derive(Debug, Clone)]
pub struct IndexStatus {
    pub model: ModelStatus,
    pub rerank: ModelStatus,
    pub indexed_files: usize,
    pub indexing: bool,
    /// Only present while `indexing` is true and a phase is actively working.
    pub progress: Option<IndexProgress>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileHit {
    pub path: PathBuf,
    /// Snippet taken from the best-matching chunk, pre-split into
    /// `(text, highlighted)` spans.
    pub snippet: Vec<(String, bool)>,
    pub detail: HitDetail,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HitDetail {
    /// Full best-matching chunk, whitespace collapsed and split into
    /// `(text, highlighted)` spans.
    pub spans: Vec<(String, bool)>,
    /// 1-based rank of this file among distinct files in the keyword candidates.
    pub lexical_rank: Option<usize>,
    /// 1-based rank of this file among distinct files in the semantic candidates.
    pub semantic_rank: Option<usize>,
    /// Ordering score before rerank. RRF when both legs ran, otherwise the
    /// surviving leg's native score.
    pub fused_score: f64,
    /// Present only for files that took part in rerank.
    pub rerank: Option<RerankDetail>,
    /// Whether rerank ran at all for this query.
    pub rerank_applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankDetail {
    /// Raw Gemma similarity.
    pub score: f64,
    /// Configured blend weight of the rerank score.
    pub weight: f64,
    /// 1-based position before rerank.
    pub before: usize,
    /// 1-based position after rerank.
    pub after: usize,
}

struct RerankOutcome {
    details: HashMap<i64, RerankDetail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileEventKind {
    Upsert,
    Remove,
    /// Rules may have changed (or a directory appeared/moved), so re-walk the
    /// whole subtree and reconcile it against the index.
    RescanDir,
}

#[derive(Debug, Clone)]
pub struct FileEvent {
    pub path: PathBuf,
    pub kind: FileEventKind,
}

struct FileInfo {
    path: String,
    norm_text: String,
    chunks: u32,
    embedded: bool,
}

pub struct IndexEngine {
    cfg: Config,
    roots: Vec<PathBuf>,
    index_dir: PathBuf,
    rules: ScopeRules,
    store: Store,
    hybrid: HybridIndex,
    dims: usize,
    index_model_id: String,
    model: Option<Embedder>,
    model_status: ModelStatus,
    rerank_model: Option<GemmaEmbedder>,
    rerank_status: ModelStatus,
    rerank: RerankStore,
    rerank_pending: VecDeque<i64>,
    rerank_pending_set: HashSet<i64>,
    rerank_top_n: usize,
    rerank_weight: f64,
    rerank_batch_size: usize,
    leg_top_k: usize,
    result_limit: usize,
    files: HashMap<i64, FileInfo>,
    pending: VecDeque<i64>,
    pending_set: HashSet<i64>,
    embed_done: usize,
    embed_total: usize,
    rerank_done: usize,
    rerank_total: usize,
    scan_progress: Option<IndexProgress>,
    last_error: Option<String>,
    batch_size: usize,
}

impl IndexEngine {
    pub fn new(
        cfg: Config,
        roots: Vec<PathBuf>,
        index_dir: PathBuf,
        model_dir: PathBuf,
        rerank_dir: PathBuf,
    ) -> Result<Self> {
        let index_dir = simplify_path(&index_dir);
        let roots: Vec<PathBuf> = roots.iter().map(|r| simplify_path(r)).collect();
        let rules = ScopeRules::global(&cfg);
        let store = Store::open(&index_dir)?;
        let mut last_error = None;

        let environment = if cfg.model.enabled || cfg.rerank.enabled {
            match embed::environment() {
                Ok(env) => Some(env),
                Err(err) => {
                    last_error = Some(format!("初始化 ONNX 运行环境失败：{err}"));
                    None
                }
            }
        } else {
            None
        };

        let (model, model_status) = if cfg.model.enabled {
            let loaded = environment.as_ref().map(|env| {
                Embedder::load(env, &model_dir, &cfg.model.file, cfg.model.max_length)
            });
            match loaded {
                Some(Ok(m)) => (Some(m), ModelStatus::Ready { dims: 0 }),
                Some(Err(err)) => (
                    None,
                    ModelStatus::Unavailable {
                        reason: err.to_string(),
                    },
                ),
                None => (
                    None,
                    ModelStatus::Unavailable {
                        reason: "ONNX 运行环境不可用".to_string(),
                    },
                ),
            }
        } else {
            (None, ModelStatus::Disabled)
        };
        let model_status = match (&model, model_status) {
            (Some(m), ModelStatus::Ready { .. }) => ModelStatus::Ready { dims: m.dims() },
            (_, other) => other,
        };
        let current_dims = model.as_ref().map(|m| m.dims());
        let current_id = model.as_ref().map(|m| m.id().to_string());

        let (rerank_model, rerank_status) = if cfg.rerank.enabled {
            let loaded = environment.as_ref().map(|env| {
                GemmaEmbedder::load(env, &rerank_dir, &cfg.rerank.file, cfg.rerank.max_length)
            });
            match loaded {
                Some(Ok(m)) => {
                    let dims = m.dims();
                    (Some(m), ModelStatus::Ready { dims })
                }
                Some(Err(err)) => (
                    None,
                    ModelStatus::Unavailable {
                        reason: err.to_string(),
                    },
                ),
                None => (
                    None,
                    ModelStatus::Unavailable {
                        reason: "ONNX 运行环境不可用".to_string(),
                    },
                ),
            }
        } else {
            (None, ModelStatus::Disabled)
        };

        let rerank_store = match &rerank_model {
            Some(m) => match rerank::load(&index_dir.join(RERANK_FILE)) {
                Ok(Some(s)) if s.dims == m.dims() && s.model_id == m.id() => s,
                Ok(Some(_)) => {
                    if last_error.is_none() {
                        last_error = Some("重排模型已变更，将重建重排索引".to_string());
                    }
                    RerankStore::new(m.id(), m.dims())
                }
                Ok(None) => RerankStore::new(m.id(), m.dims()),
                Err(err) => {
                    if last_error.is_none() {
                        last_error = Some(format!("重排索引损坏，将重建：{err}"));
                    }
                    RerankStore::new(m.id(), m.dims())
                }
            },
            None => RerankStore::default(),
        };
        if cfg.rerank.enabled
            && last_error.is_none()
            && matches!(rerank_status, ModelStatus::Unavailable { .. })
        {
            if let ModelStatus::Unavailable { reason } = &rerank_status {
                last_error = Some(format!("重排模型未就绪，已跳过重排：{reason}"));
            }
        }

        let snapshot = match persist::load(&index_dir.join(INDEX_FILE)) {
            Ok(s) => s,
            Err(err) => {
                if last_error.is_none() {
                    last_error = Some(format!("索引快照损坏，将重建：{err}"));
                }
                None
            }
        };
        let compatible = snapshot
            .as_ref()
            .is_some_and(|s| match (&model, current_dims) {
                (Some(_), Some(d)) => {
                    s.dims == d
                        && (s.model_id == "none" || current_id.as_deref() == Some(&s.model_id))
                }
                _ => true,
            });

        let stored = store.all_files()?;
        let files: HashMap<i64, FileInfo> = stored
            .iter()
            .map(|f| {
                (
                    f.id,
                    FileInfo {
                        path: f.path.clone(),
                        norm_text: f.norm_text.clone(),
                        chunks: f.chunks,
                        embedded: f.embedded,
                    },
                )
            })
            .collect();

        let (hybrid, dims, index_model_id, loaded) = if compatible {
            let snap = snapshot.expect("compatible snapshot");
            (snap.hybrid, snap.dims, snap.model_id, true)
        } else {
            (
                HybridIndex::new(current_dims.unwrap_or(FALLBACK_DIMS)),
                current_dims.unwrap_or(FALLBACK_DIMS),
                current_id.clone().unwrap_or_else(|| "none".to_string()),
                false,
            )
        };

        let mut engine = Self {
            batch_size: cfg.model.batch_size.max(1),
            rerank_top_n: cfg.rerank.top_n,
            rerank_weight: cfg.rerank.weight,
            rerank_batch_size: cfg.rerank.batch_size.max(1),
            leg_top_k: cfg.search.candidates.max(1),
            result_limit: cfg.search.results.max(1),
            cfg,
            roots,
            index_dir,
            rules,
            store,
            hybrid,
            dims,
            index_model_id,
            model,
            model_status,
            rerank_model,
            rerank_status,
            rerank: rerank_store,
            rerank_pending: VecDeque::new(),
            rerank_pending_set: HashSet::new(),
            files,
            pending: VecDeque::new(),
            pending_set: HashSet::new(),
            embed_done: 0,
            embed_total: 0,
            rerank_done: 0,
            rerank_total: 0,
            scan_progress: None,
            last_error,
        };

        if loaded {
            engine.reconcile_loaded_index()?;
        } else {
            engine.rebuild_lexical_index()?;
        }
        engine.queue_missing_embeddings();
        engine.queue_missing_rerank();
        Ok(engine)
    }

    pub fn status(&self, indexing: bool) -> IndexStatus {
        let progress = if !indexing {
            None
        } else if let Some(scan) = self.scan_progress {
            Some(scan)
        } else if self.has_pending_embeddings() {
            Some(IndexProgress {
                phase: IndexPhase::Embed,
                done: self.embed_done,
                total: self.embed_total,
            })
        } else if self.has_pending_rerank() {
            Some(IndexProgress {
                phase: IndexPhase::Rerank,
                done: self.rerank_done,
                total: self.rerank_total,
            })
        } else {
            None
        };
        IndexStatus {
            model: self.model_status.clone(),
            rerank: self.rerank_status.clone(),
            indexed_files: self.files.len(),
            indexing,
            progress,
            last_error: self.last_error.clone(),
        }
    }

    pub fn model_label(&self) -> &'static str {
        match &self.model_status {
            ModelStatus::Disabled => "已禁用",
            ModelStatus::Loading => "加载中",
            ModelStatus::Ready { .. } => "就绪",
            ModelStatus::Unavailable { .. } => "未就绪",
        }
    }

    pub fn rerank_label(&self) -> &'static str {
        match &self.rerank_status {
            ModelStatus::Disabled => "已关闭",
            ModelStatus::Loading => "加载中",
            ModelStatus::Ready { .. } => "就绪",
            ModelStatus::Unavailable { .. } => "未就绪",
        }
    }

    pub fn note_error(&mut self, message: impl Into<String>) {
        let message = message.into();
        log::error!("{message}");
        self.last_error = Some(message);
    }

    pub fn has_pending_embeddings(&self) -> bool {
        !self.pending.is_empty() && self.model.is_some()
    }

    pub fn has_pending_rerank(&self) -> bool {
        !self.rerank_pending.is_empty() && self.rerank_model.is_some()
    }

    pub fn has_pending_work(&self) -> bool {
        self.has_pending_embeddings() || self.has_pending_rerank()
    }

    pub fn scan_lexical(&mut self, progress: impl FnMut(&IndexStatus)) -> Result<()> {
        self.scan_lexical_cancellable(progress, || false)
    }

    /// Like [`Self::scan_lexical`], but checks `should_stop` between files so a
    /// quit during the initial scan does not block process exit. A cancelled
    /// scan skips the stale-file sweep; the next startup rescans.
    pub fn scan_lexical_cancellable(
        &mut self,
        mut progress: impl FnMut(&IndexStatus),
        mut should_stop: impl FnMut() -> bool,
    ) -> Result<()> {
        let started = Instant::now();
        let paths = self.collect_paths();
        let total = paths.len();
        let mut seen: HashSet<String> = HashSet::with_capacity(total);
        let mut processed = 0usize;
        let mut changed = 0usize;
        let mut last_emit = Instant::now();
        if total > 0 {
            self.scan_progress = Some(IndexProgress {
                phase: IndexPhase::Scan,
                done: 0,
                total,
            });
            progress(&self.status(true));
        }

        for path in &paths {
            if should_stop() {
                self.scan_progress = None;
                return Ok(());
            }
            seen.insert(path_key(path));
            match self.upsert_file_from_disk(path) {
                Ok(true) => changed += 1,
                Ok(false) => {}
                Err(err) => self.note_error(format!("读取 {} 失败：{err}", path.display())),
            }
            processed += 1;
            if processed.is_multiple_of(16) || last_emit.elapsed() >= Duration::from_millis(100) {
                self.scan_progress = Some(IndexProgress {
                    phase: IndexPhase::Scan,
                    done: processed,
                    total,
                });
                progress(&self.status(true));
                last_emit = Instant::now();
            }
        }
        if total > 0 {
            self.scan_progress = Some(IndexProgress {
                phase: IndexPhase::Scan,
                done: total,
                total,
            });
            progress(&self.status(true));
        }

        let stale: Vec<i64> = self
            .files
            .iter()
            .filter(|(_, info)| !seen.contains(&info.path.to_lowercase()))
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            self.remove_file(id)?;
        }

        self.scan_progress = None;
        self.persist()?;
        log::info!(
            "扫描完成：{total} 个文件，{changed} 个变更，耗时 {:.2}s",
            started.elapsed().as_secs_f64()
        );
        progress(&self.status(false));
        Ok(())
    }

    pub fn apply_events(&mut self, events: Vec<FileEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let mut cache = RuleCache::new(self.rules.clone(), self.roots.clone());
        let mut changed = 0usize;
        for event in events {
            let path = simplify_path(&event.path);
            if path.starts_with(&self.index_dir) {
                continue;
            }
            match event.kind {
                FileEventKind::Upsert => {
                    if !path.is_file() {
                        changed += self.remove_path(&path, &mut cache)?;
                        continue;
                    }
                    if !self.is_indexable_with(&path, &cache.for_file(&path)) {
                        continue;
                    }
                    match self.upsert_file_from_disk(&path) {
                        Ok(true) => changed += 1,
                        Ok(false) => {}
                        Err(err) => self.note_error(format!("读取 {} 失败：{err}", path.display())),
                    }
                }
                FileEventKind::Remove => {
                    changed += self.remove_path(&path, &mut cache)?;
                }
                FileEventKind::RescanDir => {
                    changed += self.rescan_dir(&path, &mut cache)?;
                }
            }
        }
        for err in cache.take_errors() {
            self.note_error(err);
        }
        if changed > 0 {
            self.persist()?;
        }
        Ok(())
    }

    pub fn next_embed_batch(&mut self) -> Option<Vec<i64>> {
        self.model.as_ref()?;
        let mut batch = Vec::with_capacity(self.batch_size);
        while batch.len() < self.batch_size {
            let Some(id) = self.pending.pop_front() else {
                break;
            };
            self.pending_set.remove(&id);
            let eligible = self
                .files
                .get(&id)
                .is_some_and(|info| !info.embedded && info.chunks > 0);
            if eligible {
                batch.push(id);
            } else {
                self.embed_done += 1;
            }
        }
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    pub fn embed_documents(&mut self, ids: &[i64]) -> Result<()> {
        self.embed_documents_cancellable(ids, || false)
    }

    /// 与 [`Self::embed_documents`] 相同，但会在每个嵌入小批开始前检查
    /// `should_stop`。触发停止时本批不写入任何向量，退出后下一轮会重新嵌入。
    pub fn embed_documents_cancellable(
        &mut self,
        ids: &[i64],
        mut should_stop: impl FnMut() -> bool,
    ) -> Result<()> {
        if self.model.is_none() {
            return Ok(());
        }

        let mut texts = Vec::new();
        let mut layout: Vec<(i64, usize)> = Vec::new();
        for id in ids {
            let Some(info) = self.files.get(id) else {
                continue;
            };
            let chunks = chunk_document(&info.norm_text);
            layout.push((*id, chunks.len()));
            texts.extend(chunks);
        }

        if texts.is_empty() {
            for (id, _) in layout {
                self.mark_embedded(id)?;
            }
            return Ok(());
        }

        let batch_size = self.batch_size.max(1);
        let mut vectors = Vec::with_capacity(texts.len());
        let mut embed_error = None;
        {
            let model = self.model.as_mut().expect("model checked");
            for batch in texts.chunks(batch_size) {
                if should_stop() {
                    return Ok(());
                }
                match model.embed(batch, batch_size) {
                    Ok(mut embedded) => vectors.append(&mut embedded),
                    Err(err) => {
                        embed_error = Some(err);
                        break;
                    }
                }
            }
        }
        if let Some(err) = embed_error {
            self.note_error(format!("生成语义向量失败，已降级为仅关键词检索：{err}"));
            self.model_status = ModelStatus::Unavailable {
                reason: err.to_string(),
            };
            self.model = None;
            return Ok(());
        }

        let mut pos = 0usize;
        for (id, count) in layout {
            for idx in 0..count {
                let cid = chunk_id(id, idx);
                self.hybrid.add_vector(&cid, &vectors[pos]);
                pos += 1;
            }
            self.mark_embedded(id)?;
        }
        if let Some(model) = &self.model {
            self.index_model_id = model.id().to_string();
        }
        Ok(())
    }

    pub fn next_rerank_batch(&mut self) -> Option<Vec<i64>> {
        self.rerank_model.as_ref()?;
        let mut batch = Vec::with_capacity(self.rerank_batch_size);
        while batch.len() < self.rerank_batch_size {
            let Some(id) = self.rerank_pending.pop_front() else {
                break;
            };
            self.rerank_pending_set.remove(&id);
            let eligible = self.files.get(&id).is_some_and(|info| {
                info.chunks > 0 && !self.rerank.is_complete(id, info.chunks)
            });
            if eligible {
                batch.push(id);
            } else {
                self.rerank_done += 1;
            }
        }
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    pub fn embed_rerank_documents(&mut self, ids: &[i64]) -> Result<()> {
        self.embed_rerank_documents_cancellable(ids, || false)
    }

    /// 与 [`Self::embed_rerank_documents`] 相同，但每个嵌入小批前检查停止信号。
    pub fn embed_rerank_documents_cancellable(
        &mut self,
        ids: &[i64],
        mut should_stop: impl FnMut() -> bool,
    ) -> Result<()> {
        if self.rerank_model.is_none() {
            return Ok(());
        }
        let mut texts = Vec::new();
        let mut layout: Vec<(i64, usize)> = Vec::new();
        for id in ids {
            let Some(info) = self.files.get(id) else {
                continue;
            };
            let chunks = chunk_document(&info.norm_text);
            layout.push((*id, chunks.len()));
            texts.extend(
                chunks
                    .into_iter()
                    .map(|chunk| format!("{}{}", rerank::DOC_PREFIX, chunk)),
            );
        }
        if texts.is_empty() {
            return Ok(());
        }

        let batch_size = self.rerank_batch_size.max(1);
        let mut vectors = Vec::with_capacity(texts.len());
        let mut embed_error = None;
        {
            let model = self.rerank_model.as_mut().expect("rerank model checked");
            for batch in texts.chunks(batch_size) {
                if should_stop() {
                    return Ok(());
                }
                match model.embed(batch, batch_size) {
                    Ok(mut embedded) => vectors.append(&mut embedded),
                    Err(err) => {
                        embed_error = Some(err);
                        break;
                    }
                }
            }
        }
        if let Some(err) = embed_error {
            self.note_error(format!("生成重排向量失败，已跳过重排：{err}"));
            self.rerank_status = ModelStatus::Unavailable {
                reason: err.to_string(),
            };
            self.rerank_model = None;
            return Ok(());
        }

        let mut pos = 0usize;
        for (id, count) in layout {
            let file_vectors = vectors[pos..pos + count].to_vec();
            pos += count;
            self.rerank.set(id, file_vectors);
            self.rerank_done += 1;
        }
        Ok(())
    }

    pub fn persist(&mut self) -> Result<()> {
        persist::save(
            &self.index_dir.join(INDEX_FILE),
            &self.hybrid,
            &self.index_model_id,
            self.dims,
        )?;
        self.store
            .set_meta("index_model_id", &self.index_model_id)?;
        self.store.set_meta("index_dims", &self.dims.to_string())?;
        if !self.rerank.model_id.is_empty() {
            rerank::save(&self.index_dir.join(RERANK_FILE), &self.rerank)?;
        }
        Ok(())
    }

    pub fn search(&mut self, parsed: &ParsedQuery) -> Result<Vec<FileHit>> {
        let ncfg = &self.cfg.normalize;
        let query_norm = normalize_query(&parsed.text, ncfg);
        let query_seg = tokenize::segment(&query_norm);
        let phrases: Vec<String> = parsed
            .exact_phrases
            .iter()
            .map(|p| normalize_query(p, ncfg).to_lowercase())
            .filter(|p| !p.is_empty())
            .collect();

        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut best_chunk: HashMap<i64, (usize, f64)> = HashMap::new();
        let mut lexical_ranks: HashMap<i64, usize> = HashMap::new();
        let mut semantic_ranks: HashMap<i64, usize> = HashMap::new();
        if !parsed.text.trim().is_empty() {
            let lexical = if query_seg.trim().is_empty() {
                Vec::new()
            } else {
                self.hybrid.search_lexical(&query_seg, self.leg_top_k)
            };
            let semantic = match self.model.as_mut() {
                Some(model) => match model.embed_one(&query_norm) {
                    Ok(vector) => Some(self.hybrid.search_semantic(&vector, self.leg_top_k)),
                    Err(err) => {
                        self.note_error(format!("查询向量生成失败，已降级为仅关键词检索：{err}"));
                        self.model_status = ModelStatus::Unavailable {
                            reason: err.to_string(),
                        };
                        self.model = None;
                        None
                    }
                },
                None => None,
            };
            lexical_ranks = file_ranks(&lexical);
            if let Some(sem) = &semantic {
                semantic_ranks = file_ranks(sem);
            }
            let ranked: Vec<(String, f64)> = match (lexical.is_empty(), semantic) {
                (false, Some(sem)) => rrf_fuse_scored(&lexical, &sem, RRF_K),
                (false, None) => lexical,
                (true, Some(sem)) => sem
                    .into_iter()
                    .map(|(id, distance)| (id, 1.0 - f64::from(distance)))
                    .collect(),
                (true, None) => Vec::new(),
            };
            for (cid, score) in ranked {
                if let Some((fid, idx)) = parse_chunk_position(&cid) {
                    scores
                        .entry(fid)
                        .and_modify(|s| {
                            if score > *s {
                                *s = score;
                            }
                        })
                        .or_insert(score);
                    best_chunk
                        .entry(fid)
                        .and_modify(|(best, best_score)| {
                            if score > *best_score {
                                *best = idx;
                                *best_score = score;
                            }
                        })
                        .or_insert((idx, score));
                }
            }
        }

        let candidates: Vec<(i64, f64)> = if parsed.text.trim().is_empty() {
            self.files.keys().map(|id| (*id, 0.0)).collect()
        } else {
            scores.into_iter().collect()
        };

        let mut hits: Vec<(i64, String, f64)> = Vec::new();
        for (id, score) in candidates {
            let Some(info) = self.files.get(&id) else {
                continue;
            };
            if !parsed
                .dir_filter
                .as_deref()
                .is_none_or(|needle| contains_path(&info.path, needle))
            {
                continue;
            }
            if !parsed.name_filter.as_deref().is_none_or(|needle| {
                file_name(&info.path)
                    .to_lowercase()
                    .contains(&needle.to_lowercase())
            }) {
                continue;
            }
            if !phrases.is_empty() {
                let haystack = info.norm_text.to_lowercase();
                if !phrases.iter().all(|p| haystack.contains(p)) {
                    continue;
                }
            }
            hits.push((id, info.path.clone(), score));
        }

        hits.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.to_lowercase().cmp(&b.1.to_lowercase()))
        });
        hits.truncate(self.result_limit);
        let rerank = if !parsed.text.trim().is_empty() {
            self.rerank_hits(&query_norm, &mut hits)
        } else {
            None
        };

        Ok(hits
            .into_iter()
            .map(|(id, path, score)| {
                let (snippet, detail_spans) = match self.files.get(&id) {
                    Some(info) => {
                        let idx = best_chunk.get(&id).map(|(idx, _)| *idx);
                        match chunk_at(&info.norm_text, idx) {
                            Some(chunk) => (
                                build_snippet(&chunk, &query_norm),
                                build_detail(&chunk, &query_norm),
                            ),
                            None => (Vec::new(), Vec::new()),
                        }
                    }
                    None => (Vec::new(), Vec::new()),
                };
                FileHit {
                    path: PathBuf::from(path),
                    snippet,
                    detail: HitDetail {
                        spans: detail_spans,
                        lexical_rank: lexical_ranks.get(&id).copied(),
                        semantic_rank: semantic_ranks.get(&id).copied(),
                        fused_score: score,
                        rerank: rerank.as_ref().and_then(|r| r.details.get(&id).copied()),
                        rerank_applied: rerank.is_some(),
                    },
                }
            })
            .collect())
    }

    fn rerank_hits(
        &mut self,
        query_norm: &str,
        hits: &mut [(i64, String, f64)],
    ) -> Option<RerankOutcome> {
        if hits.len() < 2 || query_norm.trim().is_empty() {
            return None;
        }
        if !matches!(self.rerank_status, ModelStatus::Ready { .. }) {
            return None;
        }
        let n = self.rerank_top_n.min(hits.len());
        let top: Vec<(i64, f64)> = hits[..n]
            .iter()
            .map(|(id, _, score)| (*id, *score))
            .collect();
        let complete = top.iter().all(|(id, _)| {
            self.files
                .get(id)
                .is_some_and(|info| self.rerank.is_complete(*id, info.chunks))
        });
        if !complete {
            return None;
        }

        let prompt = format!("{}{}", rerank::QUERY_PREFIX, query_norm);
        let query = match self.rerank_model.as_mut() {
            Some(model) => match model.embed_one(&prompt) {
                Ok(vector) => vector,
                Err(err) => {
                    self.note_error(format!("生成重排查询向量失败，已跳过重排：{err}"));
                    self.rerank_status = ModelStatus::Unavailable {
                        reason: err.to_string(),
                    };
                    self.rerank_model = None;
                    return None;
                }
            },
            None => return None,
        };

        let mut gemma = HashMap::with_capacity(n);
        for (id, _) in &top {
            let info = self.files.get(id)?;
            gemma.insert(*id, self.rerank.max_similarity(*id, info.chunks, &query)?);
        }
        let order = rerank::blended_top(&top, &gemma, self.rerank_weight)?;

        let mut details = HashMap::with_capacity(n);
        for (after, id) in order.iter().enumerate() {
            let before = top.iter().position(|(tid, _)| tid == id)? + 1;
            details.insert(
                *id,
                RerankDetail {
                    score: *gemma.get(id)?,
                    weight: self.rerank_weight,
                    before,
                    after: after + 1,
                },
            );
        }

        let mut by_id: HashMap<i64, (i64, String, f64)> = hits[..n]
            .iter()
            .map(|(id, path, score)| (*id, (*id, path.clone(), *score)))
            .collect();
        let reordered: Vec<(i64, String, f64)> = order
            .into_iter()
            .filter_map(|id| by_id.remove(&id))
            .collect();
        if reordered.len() != n {
            return None;
        }
        for (slot, item) in reordered.into_iter().enumerate() {
            hits[slot] = item;
        }
        Some(RerankOutcome { details })
    }

    fn collect_paths(&mut self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut walk_error = None;
        let roots = self.roots.clone();
        let mut cache = RuleCache::new(self.rules.clone(), roots.clone());
        for root in &roots {
            let (files, error) = walk_files(&self.index_dir, root, &mut cache);
            walk_error = error.or(walk_error);
            for path in files {
                if self.is_indexable_with(&path, &cache.for_file(&path)) {
                    out.push(path);
                }
            }
        }
        for err in cache.take_errors() {
            self.note_error(err);
        }
        if let Some(err) = walk_error {
            self.note_error(err);
        }
        out.sort_by_key(|p| p.to_string_lossy().to_lowercase());
        out.dedup_by(|a, b| path_key(a) == path_key(b));
        out
    }

    fn is_indexable_with(&self, path: &Path, rules: &ScopeRules) -> bool {
        if !self.roots.iter().any(|root| is_under(path, root)) {
            return false;
        }
        if path.starts_with(&self.index_dir) {
            return false;
        }
        rules.is_indexable(path, &self.roots)
    }

    /// Re-walk a subtree after its rules changed (or it was created/moved),
    /// then remove indexed files that are no longer part of it.
    fn rescan_dir(&mut self, dir: &Path, cache: &mut RuleCache) -> Result<usize> {
        if dir.starts_with(&self.index_dir) {
            return Ok(0);
        }
        if !self.roots.iter().any(|root| is_under(dir, root)) {
            return Ok(0);
        }

        let mut seen: HashSet<String> = HashSet::new();
        let mut changed = 0usize;
        if dir.is_dir() {
            let (files, walk_error) = walk_files(&self.index_dir, dir, cache);
            if let Some(err) = walk_error {
                self.note_error(err);
            }
            for path in files {
                if !self.is_indexable_with(&path, &cache.for_file(&path)) {
                    continue;
                }
                seen.insert(path_key(&path));
                match self.upsert_file_from_disk(&path) {
                    Ok(true) => changed += 1,
                    Ok(false) => {}
                    Err(err) => self.note_error(format!("读取 {} 失败：{err}", path.display())),
                }
            }
        }

        let key = path_key(dir);
        let prefixes = [format!("{key}\\"), format!("{key}/")];
        let stale: Vec<i64> = self
            .files
            .iter()
            .filter(|(_, info)| {
                let path = info.path.to_lowercase();
                (path == key || prefixes.iter().any(|prefix| path.starts_with(prefix)))
                    && !seen.contains(&path)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            self.remove_file(id)?;
            changed += 1;
        }
        Ok(changed)
    }

    fn upsert_file_from_disk(&mut self, path: &Path) -> Result<bool> {
        let meta = std::fs::metadata(path)?;
        if !meta.is_file() {
            return Ok(false);
        }
        let path_str = simplify_path(path).to_string_lossy().into_owned();
        let mtime_ns = system_time_ns(meta.modified().ok());
        let size = meta.len() as i64;

        // Fast path: unchanged mtime + size means we do not need to read or
        // hash the file at all. This is what keeps a warm startup from
        // re-reading every large PDF/OFD just to discover nothing changed.
        let old = self.store.get_by_path(&path_str)?;
        if let Some(old) = &old {
            if old.mtime_ns == mtime_ns && old.size == size {
                self.queue_missing_work(old);
                return Ok(false);
            }
        }

        let bytes = std::fs::read(path)?;
        let hash = blake3::hash(&bytes).to_hex().to_string();

        if let Some(old) = &old {
            if old.hash == hash {
                if old.mtime_ns != mtime_ns || old.size != size {
                    self.store.update_file_meta(&path_str, mtime_ns, size)?;
                }
                self.queue_missing_work(old);
                return Ok(false);
            }
        }
        let Some(text) = extract_text(path, &bytes)? else {
            // No text layer (e.g. a scanned PDF): keep it out of the index
            // rather than storing an empty document.
            if let Some(old) = self.store.get_by_path(&path_str)? {
                self.remove_file(old.id)?;
                return Ok(true);
            }
            return Ok(false);
        };
        let norm = normalize(&text, &self.cfg.normalize);
        self.index_file_lexical(&path_str, &hash, mtime_ns, size, &norm)?;
        Ok(true)
    }

    fn queue_missing_work(&mut self, old: &StoredFile) {
        if !old.embedded && old.chunks > 0 && self.model.is_some() {
            self.add_pending(old.id);
        }
        if old.chunks > 0
            && self.rerank_model.is_some()
            && !self.rerank.is_complete(old.id, old.chunks)
        {
            self.add_rerank_pending(old.id);
        }
    }

    fn index_file_lexical(
        &mut self,
        path: &str,
        hash: &str,
        mtime_ns: i64,
        size: i64,
        norm: &str,
    ) -> Result<i64> {
        let chunks = chunk_document(norm);
        let old = self.store.get_by_path(path)?;
        let old_chunks = old.as_ref().map(|f| f.chunks).unwrap_or(0);
        let embedded = chunks.is_empty();
        let id = self.store.upsert(
            path,
            hash,
            mtime_ns,
            size,
            norm,
            chunks.len() as u32,
            embedded,
        )?;

        let upper = old_chunks.max(chunks.len() as u32) as usize;
        for idx in 0..upper {
            self.hybrid.remove(&chunk_id(id, idx));
        }
        for (idx, chunk) in chunks.iter().enumerate() {
            self.hybrid
                .add_lexical(&chunk_id(id, idx), &tokenize::segment(chunk));
        }

        let info = FileInfo {
            path: path.to_string(),
            norm_text: norm.to_string(),
            chunks: chunks.len() as u32,
            embedded,
        };
        self.files.insert(id, info);
        self.pending_set.remove(&id);
        self.pending.retain(|p| *p != id);
        if !embedded && self.model.is_some() {
            self.add_pending(id);
        }
        self.rerank.remove(id);
        self.rerank_pending_set.remove(&id);
        self.rerank_pending.retain(|p| *p != id);
        if !embedded && self.rerank_model.is_some() {
            self.add_rerank_pending(id);
        }
        Ok(id)
    }

    fn reindex_lexical(&mut self, id: i64) -> Result<()> {
        let Some(info) = self.files.get(&id) else {
            return Ok(());
        };
        let norm = info.norm_text.clone();
        let old_chunks = info.chunks;
        let chunks = chunk_document(&norm);
        let upper = old_chunks.max(chunks.len() as u32) as usize;
        for idx in 0..upper {
            self.hybrid.remove(&chunk_id(id, idx));
        }
        for (idx, chunk) in chunks.iter().enumerate() {
            self.hybrid
                .add_lexical(&chunk_id(id, idx), &tokenize::segment(chunk));
        }
        let embedded = chunks.is_empty();
        self.store.update_chunks(id, chunks.len() as u32)?;
        self.store.set_embedded(id, embedded)?;
        if let Some(info) = self.files.get_mut(&id) {
            info.chunks = chunks.len() as u32;
            info.embedded = embedded;
        }
        if !embedded && self.model.is_some() {
            self.add_pending(id);
        }
        if !embedded && self.rerank_model.is_some() {
            let chunk_count = self.files.get(&id).map(|info| info.chunks).unwrap_or(0);
            if !self.rerank.is_complete(id, chunk_count) {
                self.add_rerank_pending(id);
            }
        }
        Ok(())
    }

    fn mark_embedded(&mut self, id: i64) -> Result<()> {
        self.store.set_embedded(id, true)?;
        if let Some(info) = self.files.get_mut(&id) {
            info.embedded = true;
        }
        self.embed_done += 1;
        Ok(())
    }

    fn add_pending(&mut self, id: i64) {
        if self.pending.is_empty() && self.pending_set.is_empty() {
            self.embed_done = 0;
            self.embed_total = 0;
        }
        if self.pending_set.insert(id) {
            self.pending.push_back(id);
            self.embed_total += 1;
        }
    }

    fn queue_missing_embeddings(&mut self) {
        if self.model.is_none() {
            return;
        }
        let ids: Vec<i64> = self
            .files
            .iter()
            .filter(|(_, info)| !info.embedded && info.chunks > 0)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.add_pending(id);
        }
    }

    fn add_rerank_pending(&mut self, id: i64) {
        if self.rerank_pending.is_empty() && self.rerank_pending_set.is_empty() {
            self.rerank_done = 0;
            self.rerank_total = 0;
        }
        if self.rerank_pending_set.insert(id) {
            self.rerank_pending.push_back(id);
            self.rerank_total += 1;
        }
    }

    fn queue_missing_rerank(&mut self) {
        if self.rerank_model.is_none() {
            return;
        }
        self.rerank.retain_files(|id| self.files.contains_key(&id));
        let ids: Vec<i64> = self
            .files
            .iter()
            .filter(|(id, info)| {
                info.chunks > 0 && !self.rerank.is_complete(**id, info.chunks)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.add_rerank_pending(id);
        }
    }

    fn remove_path(&mut self, target: &Path, cache: &mut RuleCache) -> Result<usize> {
        if target.is_file() {
            if self.is_indexable_with(target, &cache.for_file(target)) {
                return Ok(usize::from(self.upsert_file_from_disk(target)?));
            }
            let path = simplify_path(target).to_string_lossy().into_owned();
            if let Some(old) = self.store.get_by_path(&path)? {
                self.remove_file(old.id)?;
                return Ok(1);
            }
            return Ok(0);
        }
        if target.is_dir() {
            return Ok(0);
        }
        let key = target.to_string_lossy().to_lowercase();
        let prefix = format!("{key}\\");
        let ids: Vec<i64> = self
            .files
            .iter()
            .filter(|(_, info)| {
                let p = info.path.to_lowercase();
                p == key || p.starts_with(&prefix)
            })
            .map(|(id, _)| *id)
            .collect();
        let mut removed = 0;
        for id in ids {
            self.remove_file(id)?;
            removed += 1;
        }
        Ok(removed)
    }

    fn remove_file(&mut self, id: i64) -> Result<()> {
        if let Some(info) = self.files.remove(&id) {
            for idx in 0..info.chunks as usize {
                self.hybrid.remove(&chunk_id(id, idx));
            }
            self.store.delete_by_id(id)?;
            self.pending_set.remove(&id);
            self.pending.retain(|p| *p != id);
            self.rerank_pending_set.remove(&id);
            self.rerank_pending.retain(|p| *p != id);
            self.rerank.remove(id);
        }
        Ok(())
    }

    fn rebuild_lexical_index(&mut self) -> Result<()> {
        self.store.clear_embedded_all()?;
        let ids: Vec<i64> = self.files.keys().copied().collect();
        for id in ids {
            self.reindex_lexical(id)?;
        }
        Ok(())
    }

    fn reconcile_loaded_index(&mut self) -> Result<()> {
        let ids: Vec<i64> = self.files.keys().copied().collect();
        for id in ids {
            let (expected, embedded) = {
                let Some(info) = self.files.get(&id) else {
                    continue;
                };
                (chunk_document(&info.norm_text).len(), info.embedded)
            };
            let stored_chunks = self.files[&id].chunks as usize;
            if expected != stored_chunks {
                self.reindex_lexical(id)?;
                continue;
            }
            if embedded {
                let complete =
                    (0..expected).all(|idx| self.hybrid.contains_vector(&chunk_id(id, idx)));
                if !complete {
                    self.store.set_embedded(id, false)?;
                    if let Some(info) = self.files.get_mut(&id) {
                        info.embedded = false;
                    }
                }
            }
        }
        Ok(())
    }
}

fn chunk_document(norm: &str) -> Vec<String> {
    chunk_text_with_overlap(norm, CHUNK_BYTES, CHUNK_OVERLAP)
}

fn chunk_id(file_id: i64, idx: usize) -> String {
    format!("{file_id}#{idx}")
}

fn parse_chunk_position(id: &str) -> Option<(i64, usize)> {
    let (file, idx) = id.split_once('#')?;
    Some((file.parse().ok()?, idx.parse().ok()?))
}

fn file_ranks<S>(list: &[(String, S)]) -> HashMap<i64, usize> {
    let mut ranks: HashMap<i64, usize> = HashMap::new();
    for (cid, _) in list {
        if let Some((fid, _)) = parse_chunk_position(cid) {
            let next = ranks.len() + 1;
            ranks.entry(fid).or_insert(next);
        }
    }
    ranks
}

fn chunk_at(norm_text: &str, chunk_idx: Option<usize>) -> Option<String> {
    let chunks = chunk_document(norm_text);
    let last = chunks.len().checked_sub(1)?;
    chunks
        .into_iter()
        .nth(chunk_idx.unwrap_or(0).min(last))
}

fn build_snippet(chunk: &str, query_norm: &str) -> Vec<(String, bool)> {
    let terms = snippet_terms(query_norm);
    let lower = chunk.to_lowercase();
    let first = if lower.len() == chunk.len() {
        first_match(&lower, &terms)
    } else {
        None
    };
    let (mut text, cut_left, cut_right) = snippet_window(chunk, first);
    if cut_left {
        text.insert(0, '…');
    }
    if cut_right {
        text.push('…');
    }
    highlight_snippet(&text, &terms)
}

fn build_detail(chunk: &str, query_norm: &str) -> Vec<(String, bool)> {
    let collapsed = chunk.split_whitespace().collect::<Vec<_>>().join(" ");
    highlight_snippet(&collapsed, &snippet_terms(query_norm))
}

fn snippet_terms(query_norm: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let full = query_norm.trim().to_lowercase();
    if full.chars().count() > 1 {
        terms.push(full);
    }
    for token in tokenize::tokens(query_norm) {
        if token.chars().count() >= 2 && !terms.contains(&token) {
            terms.push(token);
        }
    }
    terms.sort_by_key(|term| std::cmp::Reverse(term.chars().count()));
    terms
}

fn first_match(lower: &str, terms: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for term in terms {
        if let Some(start) = lower.find(term.as_str()) {
            if best.is_none_or(|(best_start, _)| start < best_start) {
                best = Some((start, start + term.len()));
            }
        }
    }
    best
}

fn snippet_window(text: &str, first: Option<(usize, usize)>) -> (String, bool, bool) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let (start_char, end_char) = match first {
        Some((start, end)) if chars.len() > SNIPPET_CHARS => {
            let start_idx = chars.partition_point(|(offset, _)| *offset < start);
            let end_idx = chars.partition_point(|(offset, _)| *offset < end);
            let from = start_idx.saturating_sub(SNIPPET_CONTEXT);
            let to = (from + SNIPPET_CHARS).max(end_idx).min(chars.len());
            (from, to)
        }
        _ if chars.len() > SNIPPET_CHARS => (0, SNIPPET_CHARS),
        _ => (0, chars.len()),
    };
    let start_byte = chars[start_char].0;
    let end_byte = chars
        .get(end_char)
        .map(|(offset, _)| *offset)
        .unwrap_or(text.len());
    let window = &text[start_byte..end_byte];
    let collapsed = window.split_whitespace().collect::<Vec<_>>().join(" ");
    (collapsed, start_char > 0, end_char < chars.len())
}

fn highlight_snippet(text: &str, terms: &[String]) -> Vec<(String, bool)> {
    let lower = text.to_lowercase();
    if terms.is_empty() || lower.len() != text.len() {
        return vec![(text.to_string(), false)];
    }
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for term in terms {
        let mut from = 0;
        while ranges.len() < SNIPPET_HIGHLIGHTS {
            let Some(offset) = lower[from..].find(term.as_str()) else {
                break;
            };
            let start = from + offset;
            let end = start + term.len();
            if ranges.iter().all(|(seen, seen_end)| end <= *seen || start >= *seen_end) {
                ranges.push((start, end));
            }
            from = end.max(start + 1);
        }
    }
    if ranges.is_empty() {
        return vec![(text.to_string(), false)];
    }
    ranges.sort_unstable();
    let mut spans = Vec::with_capacity(ranges.len() * 2 + 1);
    let mut cursor = 0;
    for (start, end) in ranges {
        if start > cursor {
            spans.push((text[cursor..start].to_string(), false));
        }
        spans.push((text[start..end].to_string(), true));
        cursor = end;
    }
    if cursor < text.len() {
        spans.push((text[cursor..].to_string(), false));
    }
    spans
}

fn system_time_ns(time: Option<SystemTime>) -> i64 {
    time.and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

fn file_name(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

fn contains_path(path: &str, needle: &str) -> bool {
    let hay = path.replace('/', "\\").to_lowercase();
    let needle = needle.replace('/', "\\").to_lowercase();
    hay.contains(&needle)
}

fn walk_files(
    index_dir: &Path,
    start: &Path,
    cache: &mut RuleCache,
) -> (Vec<PathBuf>, Option<String>) {
    let mut files = Vec::new();
    let mut walk_error = None;
    let walker = WalkDir::new(start)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            let path = simplify_path(entry.path());
            if path.starts_with(index_dir) {
                return false;
            }
            if entry.file_type().is_dir() {
                if let Some(name) = path.file_name() {
                    // Whether this directory itself is ignored is decided by
                    // the rules of its parent, not by a rule file inside it.
                    if cache.for_file(&path).is_ignored_dir(name) {
                        return false;
                    }
                }
            }
            true
        });
    for entry in walker {
        match entry {
            Ok(e) if e.file_type().is_file() => files.push(simplify_path(e.path())),
            Ok(_) => {}
            Err(err) => walk_error = Some(format!("扫描目录出错：{err}")),
        }
    }
    (files, walk_error)
}

fn is_under(path: &Path, root: &Path) -> bool {
    let p = path.to_string_lossy().to_lowercase();
    let r = root.to_string_lossy().to_lowercase();
    p == r || p.starts_with(&format!("{r}\\")) || p.starts_with(&format!("{r}/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn config_without_model() -> Config {
        let mut cfg = Config::default();
        cfg.model.enabled = false;
        cfg
    }

    fn engine_for(root: &Path) -> IndexEngine {
        engine_for_cfg(root, config_without_model())
    }

    fn engine_for_cfg(root: &Path, cfg: Config) -> IndexEngine {
        IndexEngine::new(
            cfg,
            vec![root.to_path_buf()],
            root.join(".rummage"),
            root.join("models"),
            root.join("rerank-models"),
        )
        .unwrap()
    }

    fn write_doc(root: &Path, name: &str, text: &str) {
        std::fs::write(root.join(name), text).unwrap();
    }

    #[test]
    fn scans_and_searches_chinese() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "# 预处理\n\n如何合并 OCR 断行。\n");
        write_doc(dir.path(), "b.md", "页眉页脚清理方法。\n");
        write_doc(dir.path(), "c.txt", "如何合并 OCR 断行。");
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine
            .search(&crate::query::parse("合并 OCR 断行"))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("a.md"));
    }

    #[test]
    fn dir_name_and_phrase_filters() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("ocr")).unwrap();
        std::fs::create_dir(dir.path().join("pdf")).unwrap();
        write_doc(
            &dir.path().join("ocr"),
            "preprocess.md",
            "页眉页脚 处理方案",
        );
        write_doc(&dir.path().join("pdf"), "pipeline.md", "页眉页脚 处理方案");
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();

        let hits = engine
            .search(&crate::query::parse(r#"页眉页脚 dir:ocr"#))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("preprocess.md"));

        let hits = engine
            .search(&crate::query::parse(r#"页眉页脚 name:pipeline"#))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("pipeline.md"));

        let hits = engine
            .search(&crate::query::parse(r#""页眉页脚""#))
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn incremental_update_and_delete() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "旧的内容：合并断行");
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();

        write_doc(dir.path(), "a.md", "新的内容：页脚清理");
        engine
            .apply_events(vec![FileEvent {
                path: dir.path().join("a.md"),
                kind: FileEventKind::Upsert,
            }])
            .unwrap();
        assert!(engine
            .search(&crate::query::parse("合并断行"))
            .unwrap()
            .is_empty());
        assert_eq!(
            engine
                .search(&crate::query::parse("页脚清理"))
                .unwrap()
                .len(),
            1
        );

        std::fs::remove_file(dir.path().join("a.md")).unwrap();
        engine
            .apply_events(vec![FileEvent {
                path: dir.path().join("a.md"),
                kind: FileEventKind::Remove,
            }])
            .unwrap();
        assert!(engine
            .search(&crate::query::parse("页脚清理"))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn ignores_builtin_directories() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("node_modules")).unwrap();
        write_doc(&dir.path().join("node_modules"), "x.md", "合并断行");
        write_doc(dir.path(), "y.md", "合并断行");
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("y.md"));
    }

    #[test]
    fn indexes_only_configured_extensions() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "合并断行");
        write_doc(dir.path(), "b.txt", "合并断行");
        write_doc(dir.path(), "c.pdf", "合并断行");

        let mut cfg = config_without_model();
        cfg.index.extensions = Some(vec!["txt".into()]);
        let mut engine = IndexEngine::new(
            cfg,
            vec![dir.path().to_path_buf()],
            dir.path().join(".rummage"),
            dir.path().join("models"),
            dir.path().join("rerank-models"),
        )
        .unwrap();
        engine.scan_lexical(|_| {}).unwrap();

        assert_eq!(engine.status(false).indexed_files, 1);
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("b.txt"));
    }

    #[test]
    fn include_whitelist_matches_name_prefix() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        write_doc(dir.path(), "会议纪要-1.md", "合并断行");
        write_doc(&dir.path().join("sub"), "会议纪要-2.md", "合并断行");
        write_doc(dir.path(), "通知.md", "合并断行");

        let mut cfg = config_without_model();
        cfg.include.patterns = vec!["会议纪要*".into()];
        let mut engine = IndexEngine::new(
            cfg,
            vec![dir.path().to_path_buf()],
            dir.path().join(".rummage"),
            dir.path().join("models"),
            dir.path().join("rerank-models"),
        )
        .unwrap();
        engine.scan_lexical(|_| {}).unwrap();

        assert_eq!(engine.status(false).indexed_files, 2);
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .all(|h| h.path.ends_with("会议纪要-1.md") || h.path.ends_with("会议纪要-2.md")));
    }

    #[test]
    fn include_path_whitelist_and_ignore_precedence() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("2025").join("draft")).unwrap();
        write_doc(&dir.path().join("2025"), "会议纪要-1.md", "合并断行");
        write_doc(
            &dir.path().join("2025").join("draft"),
            "会议纪要-2.md",
            "合并断行",
        );
        write_doc(&dir.path().join("2025"), "通知.md", "合并断行");

        let mut cfg = config_without_model();
        cfg.include.patterns = vec!["2025/**".into()];
        cfg.ignore.dirs = vec!["draft*".into()];
        let mut engine = IndexEngine::new(
            cfg,
            vec![dir.path().to_path_buf()],
            dir.path().join(".rummage"),
            dir.path().join("models"),
            dir.path().join("rerank-models"),
        )
        .unwrap();
        engine.scan_lexical(|_| {}).unwrap();

        assert_eq!(engine.status(false).indexed_files, 2);
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .all(|h| !h.path.to_string_lossy().contains("draft")));
    }

    #[test]
    fn ignores_wildcard_directories() {
        let dir = tempdir().unwrap();
        for name in ["draft-2024", "2023", "keep"] {
            std::fs::create_dir(dir.path().join(name)).unwrap();
            write_doc(&dir.path().join(name), "x.md", "合并断行");
        }
        let mut cfg = config_without_model();
        cfg.ignore.dirs = vec!["draft*".into(), "20??".into()];
        let mut engine = IndexEngine::new(
            cfg,
            vec![dir.path().to_path_buf()],
            dir.path().join(".rummage"),
            dir.path().join("models"),
            dir.path().join("rerank-models"),
        )
        .unwrap();
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("keep\\x.md"));
    }

    #[test]
    fn indexes_text_documents_and_skips_scan_only_files() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let docx = r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body><w:p><w:r><w:t>光伏电站巡检记录：组件清洁。</w:t></w:r></w:p></w:body>
        </w:document>"#;
        std::fs::write(root.join("report.docx"), crate::extract::make_docx(docx)).unwrap();
        std::fs::write(
            root.join("patrol.pdf"),
            crate::extract::make_pdf("BT /F1 24 Tf 72 720 Td (Hello PDF) Tj ET"),
        )
        .unwrap();
        std::fs::write(root.join("scan.pdf"), crate::extract::make_pdf("q Q")).unwrap();
        std::fs::write(
            root.join("notice.ofd"),
            crate::extract::make_ofd(&[("Pages/Page_0/Content.xml", "省公司信息：树线矛盾治理。")]),
        )
        .unwrap();

        let mut engine = engine_for(root);
        engine.scan_lexical(|_| {}).unwrap();
        assert_eq!(engine.status(false).indexed_files, 3);

        let hits = engine.search(&crate::query::parse("光伏")).unwrap();
        assert!(
            hits.iter().any(|h| h.path.ends_with("report.docx")),
            "{hits:?}"
        );
        let hits = engine.search(&crate::query::parse("Hello PDF")).unwrap();
        assert!(
            hits.iter().any(|h| h.path.ends_with("patrol.pdf")),
            "{hits:?}"
        );
        let hits = engine.search(&crate::query::parse("树线矛盾")).unwrap();
        assert!(
            hits.iter().any(|h| h.path.ends_with("notice.ofd")),
            "{hits:?}"
        );
        assert!(
            hits.iter().all(|h| !h.path.ends_with("scan.pdf")),
            "{hits:?}"
        );
    }

    #[test]
    fn local_rule_file_overrides_global_in_scan() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sub = root.join("sub");
        std::fs::create_dir(&sub).unwrap();
        write_doc(root, "global.md", "合并断行");
        write_doc(root, "local.md", "合并断行");
        write_doc(&sub, "global.md", "合并断行");
        write_doc(&sub, "local.md", "合并断行");
        std::fs::write(
            sub.join(".rummage.toml"),
            "[include]\npatterns = [\"local*\"]\n",
        )
        .unwrap();

        let mut cfg = config_without_model();
        cfg.include.patterns = vec!["global*".into()];
        let mut engine = IndexEngine::new(
            cfg,
            vec![root.to_path_buf()],
            root.join(".rummage"),
            root.join("models"),
            root.join("rerank-models"),
        )
        .unwrap();
        engine.scan_lexical(|_| {}).unwrap();

        assert_eq!(engine.status(false).indexed_files, 2);
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        let paths: Vec<String> = hits
            .iter()
            .map(|h| h.path.to_string_lossy().replace('/', "\\"))
            .collect();
        assert_eq!(paths.len(), 2, "{paths:?}");
        assert!(paths.iter().any(|p| p.ends_with("global.md")));
        assert!(paths.iter().any(|p| p.ends_with("sub\\local.md")));
    }

    #[test]
    fn rescan_dir_applies_rule_changes() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_doc(root, "a.md", "合并断行");
        write_doc(root, "b.md", "合并断行");
        let mut engine = engine_for(root);
        engine.scan_lexical(|_| {}).unwrap();
        assert_eq!(engine.status(false).indexed_files, 2);

        std::fs::write(
            root.join(".rummage.toml"),
            "[include]\npatterns = [\"a*\"]\n",
        )
        .unwrap();
        engine
            .apply_events(vec![FileEvent {
                path: root.to_path_buf(),
                kind: FileEventKind::RescanDir,
            }])
            .unwrap();
        assert_eq!(engine.status(false).indexed_files, 1);
        let hits = engine.search(&crate::query::parse("合并断行")).unwrap();
        assert!(hits.iter().all(|h| h.path.ends_with("a.md")), "{hits:?}");

        std::fs::write(
            root.join(".rummage.toml"),
            "[include]\npatterns = [\"*\"]\n",
        )
        .unwrap();
        engine
            .apply_events(vec![FileEvent {
                path: root.to_path_buf(),
                kind: FileEventKind::RescanDir,
            }])
            .unwrap();
        assert_eq!(engine.status(false).indexed_files, 2);

        std::fs::remove_file(root.join(".rummage.toml")).unwrap();
        engine
            .apply_events(vec![FileEvent {
                path: root.to_path_buf(),
                kind: FileEventKind::RescanDir,
            }])
            .unwrap();
        assert_eq!(engine.status(false).indexed_files, 2);
    }

    #[test]
    fn scan_reports_progress_and_hides_it_when_done() {
        let dir = tempdir().unwrap();
        for i in 0..20 {
            write_doc(dir.path(), &format!("f{i:02}.md"), "进度测试内容");
        }
        let mut engine = engine_for(dir.path());
        let mut scans = Vec::new();
        engine
            .scan_lexical(|status| {
                if status.indexing {
                    scans.push(status.progress);
                }
            })
            .unwrap();

        assert!(!scans.is_empty());
        let first = scans[0].expect("scan progress");
        assert_eq!(first.phase, IndexPhase::Scan);
        assert_eq!(first.done, 0);
        assert_eq!(first.total, 20);
        let reported: Vec<IndexProgress> = scans.iter().flatten().copied().collect();
        assert!(reported.iter().all(|p| p.phase == IndexPhase::Scan));
        assert!(reported.iter().all(|p| p.total == 20));
        assert!(reported.iter().any(|p| p.done == 16));
        assert!(reported.iter().any(|p| p.done == 20));
        assert!(engine.status(false).progress.is_none());
    }

    #[test]
    fn progress_percent_is_clamped() {
        let percent = |done, total| {
            IndexProgress {
                phase: IndexPhase::Embed,
                done,
                total,
            }
            .percent()
        };
        assert_eq!(percent(0, 0), 100);
        assert_eq!(percent(1, 3), 33);
        assert_eq!(percent(3, 3), 100);
        assert_eq!(percent(4, 3), 100);
    }

    #[test]
    fn cancellable_scan_stops_before_indexing() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "合并断行");
        let mut engine = engine_for(dir.path());
        engine.scan_lexical_cancellable(|_| {}, || true).unwrap();
        assert_eq!(engine.status(false).indexed_files, 0);
    }

    #[test]
    fn result_limit_is_configurable() {
        let dir = tempdir().unwrap();
        for i in 0..15 {
            write_doc(dir.path(), &format!("f{i}.md"), "相同的关键词内容");
        }

        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("关键词")).unwrap();
        assert_eq!(hits.len(), RESULT_LIMIT);

        let dir = tempdir().unwrap();
        for i in 0..15 {
            write_doc(dir.path(), &format!("f{i}.md"), "相同的关键词内容");
        }
        let mut cfg = config_without_model();
        cfg.search.results = 3;
        let mut engine = engine_for_cfg(dir.path(), cfg);
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("关键词")).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn leg_candidate_count_is_configurable() {
        let dir = tempdir().unwrap();
        for i in 0..15 {
            write_doc(dir.path(), &format!("f{i}.md"), "相同的关键词内容");
        }
        let mut cfg = config_without_model();
        cfg.search.candidates = 2;
        let mut engine = engine_for_cfg(dir.path(), cfg);
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("关键词")).unwrap();
        assert!(hits.len() <= 2, "{hits:?}");
    }

    fn snippet_text(hit: &FileHit) -> String {
        hit.snippet.iter().map(|(text, _)| text.as_str()).collect()
    }

    #[test]
    fn search_snippet_contains_query_with_highlight() {
        let dir = tempdir().unwrap();
        write_doc(
            dir.path(),
            "a.md",
            "第一段内容。\n\n工程咨询公司构建全过程造价咨询体系，提升结算管理质效。\n\n第三段内容。",
        );
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("结算管理质效")).unwrap();
        assert_eq!(hits.len(), 1);
        let text = snippet_text(&hits[0]);
        assert!(text.contains("结算管理质效"), "{text}");
        assert!(
            hits[0]
                .snippet
                .iter()
                .any(|(part, highlight)| *highlight && part.contains("结算")),
            "{:?}",
            hits[0].snippet
        );
        let detail: String = hits[0]
            .detail
            .spans
            .iter()
            .map(|(text, _)| text.as_str())
            .collect();
        assert!(
            detail.contains("工程咨询公司构建全过程造价咨询体系"),
            "{detail}"
        );
        assert_eq!(hits[0].detail.lexical_rank, Some(1));
        assert_eq!(hits[0].detail.semantic_rank, None);
        assert!(hits[0].detail.fused_score > 0.0);
        assert!(!hits[0].detail.rerank_applied);
        assert!(hits[0].detail.rerank.is_none());
    }

    #[test]
    fn file_ranks_count_distinct_files_in_candidate_order() {
        let list = vec![
            ("1#0".to_string(), 3.0),
            ("2#0".to_string(), 2.0),
            ("1#1".to_string(), 1.0),
        ];
        let ranks = file_ranks(&list);
        assert_eq!(ranks.get(&1), Some(&1));
        assert_eq!(ranks.get(&2), Some(&2));
    }

    #[test]
    fn empty_query_snippet_starts_document() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "第一段内容。\n\n第二段内容。");
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("")).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(snippet_text(&hits[0]).contains("第一段内容"));
        assert_eq!(hits[0].detail.lexical_rank, None);
        assert_eq!(hits[0].detail.semantic_rank, None);
        assert_eq!(hits[0].detail.fused_score, 0.0);
        let detail: String = hits[0]
            .detail
            .spans
            .iter()
            .map(|(text, _)| text.as_str())
            .collect();
        assert!(detail.contains("第一段内容"), "{detail}");
    }

    #[test]
    fn snippet_window_centers_on_late_match() {
        let dir = tempdir().unwrap();
        let filler = "无关内容。".repeat(30);
        let tail = "尾部内容。".repeat(30);
        write_doc(
            dir.path(),
            "a.md",
            &format!("{filler}\n\n结算管理质效在此处。\n\n{tail}"),
        );
        let mut engine = engine_for(dir.path());
        engine.scan_lexical(|_| {}).unwrap();
        let hits = engine.search(&crate::query::parse("结算管理质效")).unwrap();
        assert_eq!(hits.len(), 1);
        let text = snippet_text(&hits[0]);
        assert!(text.contains("结算管理质效"), "{text}");
        assert!(text.starts_with('…'), "{text}");
        assert!(text.ends_with('…'), "{text}");
    }

    #[test]
    fn snapshot_is_reused_without_rehash() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "持久化索引测试内容");
        {
            let mut engine = engine_for(dir.path());
            engine.scan_lexical(|_| {}).unwrap();
            assert_eq!(
                engine.search(&crate::query::parse("持久化")).unwrap().len(),
                1
            );
        }
        let mut engine = engine_for(dir.path());
        assert_eq!(
            engine.search(&crate::query::parse("持久化")).unwrap().len(),
            1
        );
    }

    #[test]
    fn missing_snapshot_rebuilds_lexical_from_sqlite() {
        let dir = tempdir().unwrap();
        write_doc(dir.path(), "a.md", "重建索引测试内容");
        {
            let mut engine = engine_for(dir.path());
            engine.scan_lexical(|_| {}).unwrap();
        }
        std::fs::remove_file(dir.path().join(".rummage").join(INDEX_FILE)).unwrap();
        let mut engine = engine_for(dir.path());
        assert_eq!(
            engine.search(&crate::query::parse("重建")).unwrap().len(),
            1
        );
        engine.scan_lexical(|_| {}).unwrap();
        assert_eq!(
            engine.search(&crate::query::parse("重建")).unwrap().len(),
            1
        );
    }
}
