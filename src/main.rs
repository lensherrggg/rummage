use anyhow::{Context, Result};
use clap::Parser;
use rummage::config::{Config, ScopeRules};
use rummage::{logging, ocr, tui, watcher, worker};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(
    name = "rummage",
    version,
    about = "本地文档混合检索 TUI（Markdown / Word / PDF / OFD，关键词 + 语义，完全离线）"
)]
struct Cli {
    /// 监控根目录，可重复指定
    #[arg(long = "dir", value_name = "DIR")]
    dirs: Vec<PathBuf>,

    /// 配置文件路径（默认读取当前目录的 rummage.toml）
    #[arg(long = "config", value_name = "FILE")]
    config: Option<PathBuf>,

    /// 额外忽略的目录/文件名模式，可重复；等价于同时写进 [ignore] dirs 和 names
    #[arg(long = "ignore", value_name = "PATTERN")]
    ignore: Vec<String>,
}

fn main() -> Result<()> {
    let log_path = logging::init();
    if log_path.is_none() {
        eprintln!("警告：没有可写的日志目录，本次运行不会记录日志文件");
    }
    let result = run(log_path.as_deref());
    if let Err(err) = &result {
        log::error!("程序异常退出：{err:#}");
    }
    result
}

fn run(log_path: Option<&Path>) -> Result<()> {
    let cli = Cli::parse();
    let loaded = Config::load(cli.config.as_deref())?;
    let mut cfg = loaded.config;
    cfg.apply_cli_ignores(&cli.ignore);

    log::info!("rummage {} 启动", env!("CARGO_PKG_VERSION"));
    if let Some(path) = log_path {
        log::info!("日志文件：{}", path.display());
    }
    if let Some(path) = &loaded.path {
        log::info!("配置文件：{}", path.display());
    }

    let roots = cfg.resolve_roots(&cli.dirs, &loaded.base)?;
    let index_dir = cfg.resolve_index_dir(&roots, &loaded.base);
    let rules = ScopeRules::global(&cfg);
    let model_dir = cfg.resolve_model_dir(&loaded.base);
    let rerank_dir = cfg.resolve_rerank_model_dir(&loaded.base);
    log::info!("监控目录：{:?}", roots);
    log::info!("索引目录：{}", index_dir.display());
    log::info!(
        "模型：{}；重排：{}；OCR：{}",
        if cfg.model.enabled { "启用" } else { "禁用" },
        if cfg.rerank.enabled { "启用" } else { "禁用" },
        if cfg.ocr.enabled { "启用" } else { "禁用" }
    );
    if cfg.ocr.enabled {
        let dir = cfg.ocr.resolve_dir(&loaded.base);
        let pdfium = cfg.ocr.resolve_pdfium(&loaded.base);
        log::info!(
            "OCR 模型：{}；PDFium：{}；推理后端：{:?}",
            dir.display(),
            pdfium.display(),
            cfg.ocr.provider
        );
        ocr::configure(
            &cfg.ocr,
            &loaded.base,
            &index_dir,
            cfg.index.textless_pdf,
        );
    }

    std::fs::create_dir_all(&index_dir)
        .with_context(|| format!("创建索引目录失败：{}", index_dir.display()))?;

    let (ui_tx, ui_rx) = mpsc::channel();
    let index = worker::spawn(
        cfg.clone(),
        roots.clone(),
        index_dir.clone(),
        worker::ModelDirs {
            embedding: model_dir,
            rerank: rerank_dir,
        },
        ui_tx,
    );

    let debounce = Duration::from_millis(cfg.watch.debounce_ms);
    let (watcher, watcher_error) =
        match watcher::spawn(&roots, &index_dir, &rules, debounce, index.tx.clone()) {
            Ok(watcher) => (Some(watcher), None),
            Err(err) => (None, Some(format!("文件监控不可用：{err}"))),
        };

    if let Some(err) = &watcher_error {
        log::error!("{err}");
    }
    let mut app = tui::App::new(watcher.is_some(), watcher_error);
    app.set_result_limit(cfg.search.results);
    let tui_result = tui::run(app, index.tx.clone(), ui_rx);

    if let Some(watcher) = watcher {
        watcher.stop();
    }
    index.shutdown();
    log::info!("退出");

    tui_result
}
