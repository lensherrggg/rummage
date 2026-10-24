use crate::config::Config;
use crate::engine::{FileEvent, FileHit, IndexEngine, IndexStatus, ModelStatus};
use crate::query::ParsedQuery;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub enum IndexMsg {
    Query {
        id: u64,
        parsed: ParsedQuery,
        reply: Sender<Vec<FileHit>>,
    },
    FileEvents(Vec<FileEvent>),
    /// A non-fatal problem found outside the engine (e.g. a malformed local
    /// rule file) that should still be surfaced in the UI.
    Note(String),
    Shutdown,
}

pub enum UiMsg {
    Status(IndexStatus),
    QueryResult { id: u64, hits: Vec<FileHit> },
    Error(String),
}

pub struct IndexHandle {
    pub tx: Sender<IndexMsg>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl IndexHandle {
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.tx.send(IndexMsg::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub struct ModelDirs {
    pub embedding: PathBuf,
    pub rerank: PathBuf,
}

pub fn spawn(
    cfg: Config,
    roots: Vec<PathBuf>,
    index_dir: PathBuf,
    model_dirs: ModelDirs,
    ui_tx: Sender<UiMsg>,
) -> IndexHandle {
    let (tx, rx) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    let thread = thread::spawn(move || {
        run(cfg, roots, index_dir, model_dirs, ui_tx, rx, stop_flag);
    });
    IndexHandle {
        tx,
        stop,
        thread: Some(thread),
    }
}

fn run(
    cfg: Config,
    roots: Vec<PathBuf>,
    index_dir: PathBuf,
    model_dirs: ModelDirs,
    ui_tx: Sender<UiMsg>,
    rx: Receiver<IndexMsg>,
    stop: Arc<AtomicBool>,
) {
    let ModelDirs {
        embedding: model_dir,
        rerank: rerank_dir,
    } = model_dirs;
    let rerank_loading = if cfg.rerank.enabled {
        ModelStatus::Loading
    } else {
        ModelStatus::Disabled
    };
    let loading = IndexStatus {
        model: ModelStatus::Loading,
        rerank: rerank_loading,
        indexed_files: 0,
        indexing: true,
        progress: None,
        last_error: None,
    };
    let _ = ui_tx.send(UiMsg::Status(loading));

    let mut engine = match IndexEngine::new(cfg, roots, index_dir, model_dir, rerank_dir) {
        Ok(engine) => engine,
        Err(err) => {
            let _ = ui_tx.send(UiMsg::Error(format!("初始化索引失败：{err}")));
            let _ = ui_tx.send(UiMsg::Status(IndexStatus {
                model: ModelStatus::Unavailable {
                    reason: err.to_string(),
                },
                rerank: ModelStatus::Unavailable {
                    reason: err.to_string(),
                },
                indexed_files: 0,
                indexing: false,
                progress: None,
                last_error: Some(err.to_string()),
            }));
            return;
        }
    };

    let _ = ui_tx.send(UiMsg::Status(engine.status(true)));

    if let Err(err) = engine.scan_lexical_cancellable(
        |status| {
            let _ = ui_tx.send(UiMsg::Status(status.clone()));
        },
        || stop.load(Ordering::Relaxed),
    ) {
        let msg = format!("扫描索引失败：{err}");
        engine.note_error(msg.clone());
        let _ = ui_tx.send(UiMsg::Error(msg));
    }
    let _ = ui_tx.send(UiMsg::Status(engine.status(engine.has_pending_work())));
    if stop.load(Ordering::Relaxed) {
        let _ = engine.persist();
        return;
    }

    loop {
        if stop.load(Ordering::Relaxed) {
            let _ = engine.persist();
            let _ = ui_tx.send(UiMsg::Status(engine.status(false)));
            return;
        }
        let pending = engine.has_pending_work();
        let message = if pending {
            match rx.try_recv() {
                Ok(msg) => Some(msg),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        } else {
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(msg) => Some(msg),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        };

        if let Some(msg) = message {
            match msg {
                IndexMsg::Query { id, parsed, reply } => match engine.search(&parsed) {
                    Ok(hits) => {
                        let _ = reply.send(hits.clone());
                        let _ = ui_tx.send(UiMsg::QueryResult { id, hits });
                    }
                    Err(err) => {
                        let msg = format!("检索失败：{err}");
                        engine.note_error(msg.clone());
                        let _ = reply.send(Vec::new());
                        let _ = ui_tx.send(UiMsg::Error(msg));
                    }
                },
                IndexMsg::FileEvents(events) => {
                    if let Err(err) = engine.apply_events(events) {
                        let msg = format!("更新索引失败：{err}");
                        engine.note_error(msg.clone());
                        let _ = ui_tx.send(UiMsg::Error(msg));
                    }
                    let _ = ui_tx.send(UiMsg::Status(
                        engine.status(engine.has_pending_work()),
                    ));
                }
                IndexMsg::Note(text) => {
                    engine.note_error(text.clone());
                    let _ = ui_tx.send(UiMsg::Error(text));
                }
                IndexMsg::Shutdown => {
                    let _ = engine.persist();
                    let _ = ui_tx.send(UiMsg::Status(engine.status(false)));
                    return;
                }
            }
        }

        if engine.has_pending_embeddings() {
            if let Some(batch) = engine.next_embed_batch() {
                match engine.embed_documents_cancellable(&batch, || stop.load(Ordering::Relaxed))
                {
                    Ok(()) => {
                        if stop.load(Ordering::Relaxed) {
                            continue;
                        }
                        if let Err(err) = engine.persist() {
                            let msg = format!("索引落盘失败：{err}");
                            engine.note_error(msg.clone());
                            let _ = ui_tx.send(UiMsg::Error(msg));
                        }
                    }
                    Err(err) => {
                        let msg = format!("语义索引失败：{err}");
                        engine.note_error(msg.clone());
                        let _ = ui_tx.send(UiMsg::Error(msg));
                    }
                }
                let _ = ui_tx.send(UiMsg::Status(
                    engine.status(engine.has_pending_work()),
                ));
            }
        } else if engine.has_pending_rerank() {
            if let Some(batch) = engine.next_rerank_batch() {
                if let Err(err) = engine
                    .embed_rerank_documents_cancellable(&batch, || stop.load(Ordering::Relaxed))
                {
                    let msg = format!("重排索引失败：{err}");
                    engine.note_error(msg.clone());
                    let _ = ui_tx.send(UiMsg::Error(msg));
                } else if stop.load(Ordering::Relaxed) {
                    continue;
                } else if let Err(err) = engine.persist() {
                    let msg = format!("索引落盘失败：{err}");
                    engine.note_error(msg.clone());
                    let _ = ui_tx.send(UiMsg::Error(msg));
                }
                let _ = ui_tx.send(UiMsg::Status(
                    engine.status(engine.has_pending_work()),
                ));
            }
        }
    }

    let _ = engine.persist();
    let _ = ui_tx.send(UiMsg::Status(engine.status(false)));
}
