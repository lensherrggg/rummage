use crate::config::{is_local_rule_file, simplify_path, RuleCache, ScopeRules};
use crate::engine::{FileEvent, FileEventKind};
use crate::worker::IndexMsg;
use anyhow::{Context, Result};
use notify::event::ModifyKind;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub struct WatcherHandle {
    stop_tx: Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl WatcherHandle {
    pub fn stop(mut self) {
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

pub fn spawn(
    roots: &[PathBuf],
    index_dir: &Path,
    rules: &ScopeRules,
    debounce: Duration,
    index_tx: Sender<IndexMsg>,
) -> Result<WatcherHandle> {
    let (raw_tx, raw_rx) = mpsc::channel();
    let mut watcher = RecommendedWatcher::new(
        move |result| {
            let _ = raw_tx.send(result);
        },
        notify::Config::default(),
    )
    .context("创建文件监控器失败")?;

    for root in roots {
        watcher
            .watch(root, RecursiveMode::Recursive)
            .with_context(|| format!("监控目录失败：{}", root.display()))?;
    }

    let (stop_tx, stop_rx) = mpsc::channel();
    let index_dir = simplify_path(index_dir);
    let roots: Vec<PathBuf> = roots.iter().map(|r| simplify_path(r)).collect();
    let rules = rules.clone();
    let thread = thread::spawn(move || {
        run_loop(
            &mut watcher,
            raw_rx,
            stop_rx,
            index_tx,
            index_dir,
            roots,
            rules,
            debounce.max(Duration::from_millis(50)),
        );
    });

    Ok(WatcherHandle {
        stop_tx,
        thread: Some(thread),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    _watcher: &mut RecommendedWatcher,
    raw_rx: Receiver<notify::Result<Event>>,
    stop_rx: Receiver<()>,
    index_tx: Sender<IndexMsg>,
    index_dir: PathBuf,
    roots: Vec<PathBuf>,
    rules: ScopeRules,
    debounce: Duration,
) {
    let mut cache = RuleCache::new(rules, roots.clone());
    let mut pending: HashMap<PathBuf, FileEventKind> = HashMap::new();
    let mut deadline: Option<Instant> = None;

    loop {
        if stop_rx.try_recv().is_ok() {
            return;
        }

        let timeout = match deadline {
            Some(at) => at.saturating_duration_since(Instant::now()),
            None => Duration::from_millis(100),
        };

        match raw_rx.recv_timeout(timeout) {
            Ok(Ok(event)) => {
                for (path, kind) in classify(&event, &index_dir, &roots, &mut cache) {
                    pending.insert(path, kind);
                }
                for err in cache.take_errors() {
                    if index_tx.send(IndexMsg::Note(err)).is_err() {
                        return;
                    }
                }
                if !pending.is_empty() {
                    deadline = Some(Instant::now() + debounce);
                }
            }
            Ok(Err(_)) => {}
            Err(RecvTimeoutError::Timeout) => {
                let due = deadline.map(|at| Instant::now() >= at).unwrap_or(false);
                if due && !pending.is_empty() {
                    let events: Vec<FileEvent> = pending
                        .drain()
                        .map(|(path, kind)| FileEvent { path, kind })
                        .collect();
                    deadline = None;
                    if index_tx.send(IndexMsg::FileEvents(events)).is_err() {
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn classify(
    event: &Event,
    index_dir: &Path,
    roots: &[PathBuf],
    cache: &mut RuleCache,
) -> Vec<(PathBuf, FileEventKind)> {
    let mut out = Vec::new();
    if matches!(event.kind, EventKind::Access(_)) {
        return out;
    }

    for raw in &event.paths {
        let path = simplify_path(raw);
        if path.starts_with(index_dir) || !under_roots(&path, roots) {
            continue;
        }

        if is_local_rule_file(&path) {
            if let Some(dir) = path.parent() {
                cache.invalidate(dir);
                out.push((dir.to_path_buf(), FileEventKind::RescanDir));
            }
            continue;
        }

        if path.is_dir() {
            let structural = matches!(
                event.kind,
                EventKind::Create(_)
                    | EventKind::Remove(_)
                    | EventKind::Modify(ModifyKind::Name(_))
            );
            if structural {
                cache.invalidate(&path);
                out.push((path, FileEventKind::RescanDir));
            }
        } else if path.exists() {
            if cache.for_file(&path).is_indexable(&path, roots) {
                out.push((path, FileEventKind::Upsert));
            }
        } else {
            // A removed file or directory; the engine drops the path (and the
            // whole subtree when a directory vanished) from the index.
            cache.invalidate(&path);
            out.push((path, FileEventKind::Remove));
        }
    }
    out
}

fn under_roots(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| {
        let p = path.to_string_lossy().to_lowercase();
        let r = root.to_string_lossy().to_lowercase();
        p == r || p.starts_with(&format!("{r}\\")) || p.starts_with(&format!("{r}/"))
    })
}
