use rummage::config::{Config, ScopeRules};
use rummage::engine::{FileHit, IndexStatus, ModelStatus};
use rummage::query;
use rummage::worker::{self, IndexMsg, UiMsg};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};
use tempfile::tempdir;

fn wait_status(rx: &Receiver<UiMsg>, want: usize) -> IndexStatus {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        assert!(!left.is_zero(), "等待索引文件数 {want} 超时");
        match rx.recv_timeout(left) {
            Ok(UiMsg::Status(status)) if status.indexed_files == want => return status,
            Ok(_) => {}
            Err(err) => panic!("状态通道异常：{err}"),
        }
    }
}

fn run_query(tx: &Sender<IndexMsg>, id: u64, text: &str) -> Vec<FileHit> {
    let (reply_tx, reply_rx) = mpsc::channel();
    tx.send(IndexMsg::Query {
        id,
        parsed: query::parse(text),
        reply: reply_tx,
    })
    .unwrap();
    reply_rx
        .recv_timeout(Duration::from_secs(15))
        .expect("索引线程未回复查询")
}

#[test]
fn watcher_keeps_worker_index_in_sync() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    std::fs::write(root.join("a.md"), "# 预处理\n\n如何合并 OCR 硬换行。\n").unwrap();
    std::fs::write(root.join("notes.md"), "密级标记不得收录。\n").unwrap();

    let mut cfg = Config::default();
    cfg.model.enabled = false;
    cfg.include.patterns = vec!["a*".to_string(), "b*".to_string()];
    let rules = ScopeRules::global(&cfg);
    let (ui_tx, ui_rx) = mpsc::channel();
    let index = worker::spawn(
        cfg,
        vec![root.clone()],
        root.join(".rummage"),
        worker::ModelDirs {
            embedding: root.join("models"),
            rerank: root.join("rerank-models"),
        },
        ui_tx,
    );

    let status = wait_status(&ui_rx, 1);
    assert!(matches!(status.model, ModelStatus::Disabled));

    let watcher = rummage::watcher::spawn(
        std::slice::from_ref(&root),
        &root.join(".rummage"),
        &rules,
        Duration::from_millis(100),
        index.tx.clone(),
    )
    .expect("启动监控失败");

    std::fs::write(root.join("b.md"), "光伏电站巡检记录：组件清洁。\n").unwrap();
    std::thread::sleep(Duration::from_millis(200));
    wait_status(&ui_rx, 2);

    std::fs::write(root.join("c.md"), "密级标记不得收录。\n").unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let outside = run_query(&index.tx, 99, "密级标记");
    assert!(outside.is_empty(), "{outside:?}");

    let hits = run_query(&index.tx, 1, "光伏组件");
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert!(hits[0].path.ends_with("b.md"));

    std::fs::remove_file(root.join("a.md")).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    wait_status(&ui_rx, 1);

    let hits = run_query(&index.tx, 2, "合并硬换行");
    assert!(hits.is_empty(), "{hits:?}");

    watcher.stop();
    index.shutdown();
}

#[test]
fn watcher_applies_local_rule_file_on_the_fly() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    std::fs::write(root.join("a.md"), "如何合并 OCR 硬换行。\n").unwrap();
    std::fs::write(root.join("b.md"), "光伏组件清洁。\n").unwrap();

    let mut cfg = Config::default();
    cfg.model.enabled = false;
    let rules = ScopeRules::global(&cfg);
    let (ui_tx, ui_rx) = mpsc::channel();
    let index = worker::spawn(
        cfg,
        vec![root.clone()],
        root.join(".rummage"),
        worker::ModelDirs {
            embedding: root.join("models"),
            rerank: root.join("rerank-models"),
        },
        ui_tx,
    );
    wait_status(&ui_rx, 2);

    let watcher = rummage::watcher::spawn(
        std::slice::from_ref(&root),
        &root.join(".rummage"),
        &rules,
        Duration::from_millis(100),
        index.tx.clone(),
    )
    .expect("启动监控失败");

    std::fs::write(
        root.join(".rummage.toml"),
        "[include]\npatterns = [\"a*\"]\n",
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(400));
    wait_status(&ui_rx, 1);
    assert!(
        run_query(&index.tx, 1, "光伏组件").is_empty(),
        "局部规则未生效"
    );

    std::fs::remove_file(root.join(".rummage.toml")).unwrap();
    std::thread::sleep(Duration::from_millis(400));
    wait_status(&ui_rx, 2);
    assert_eq!(run_query(&index.tx, 2, "光伏组件").len(), 1);

    watcher.stop();
    index.shutdown();
}
