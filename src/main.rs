use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
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

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 显式下载运行所需的模型文件
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    /// 下载 bge 嵌入模型（可选 bge-base）和/或 OCR 模型到 models 目录
    Download(DownloadArgs),
}

#[derive(Debug, Args)]
struct DownloadArgs {
    /// 模型根目录（包含 bge-small-zh-v1.5 / bge-base-zh-v1.5 / ocr）；默认 exe 同目录 models
    #[arg(long = "dir", value_name = "DIR")]
    dir: Option<PathBuf>,

    /// 只下载 bge / bge-base / ocr；可逗号分隔，默认 all（bge + ocr，不含 bge-base）
    #[arg(long = "only", default_value = "all", value_name = "LIST")]
    only: String,

    /// 已存在也重新下载
    #[arg(long)]
    force: bool,

    /// 模型下载源；国内可用 https://hf-mirror.com
    #[arg(long = "hf-base", default_value = "https://huggingface.co", value_name = "URL")]
    hf_base: String,

    /// OCR 字典源；默认使用 jsDelivr 上的 PaddleOCR 文本
    #[arg(
        long = "dict-url",
        default_value = "https://fastly.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@main/ppocr/utils/ppocr_keys_v1.txt",
        value_name = "URL"
    )]
    dict_url: String,
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
    if let Some(Command::Models { action }) = &cli.command {
        return run_models(action);
    }
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

#[derive(Debug, Clone, Copy)]
struct DownloadFile {
    remote: &'static str,
    local: &'static str,
    min_bytes: u64,
}

fn run_models(action: &ModelsCommand) -> Result<()> {
    match action {
        ModelsCommand::Download(args) => download_models(args),
    }
}

fn download_models(args: &DownloadArgs) -> Result<()> {
    let only = parse_only(&args.only)?;
    let base = args
        .dir
        .clone()
        .unwrap_or_else(|| models_dir().unwrap_or_else(|_| PathBuf::from("models")));
    let ocr_dir = base.join("ocr");

    println!("模型目录：{}", base.display());
    if only.small {
        download_bge(
            &base,
            "bge-small-zh-v1.5",
            "onnx/model.onnx",
            "model.onnx",
            50 * 1024 * 1024,
            args,
        )?;
    }
    if only.base {
        download_bge(
            &base,
            "bge-base-zh-v1.5",
            "onnx/model.onnx",
            "model.onnx",
            200 * 1024 * 1024,
            args,
        )?;
    }
    if only.ocr {
        println!("下载 PP-OCRv4 → {}", ocr_dir.display());
        let files = [
            DownloadFile {
                remote: "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
                local: "det.onnx",
                min_bytes: 1_000_000,
            },
            DownloadFile {
                remote: "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx",
                local: "rec.onnx",
                min_bytes: 2_000_000,
            },
        ];
        for file in &files {
            let url = format!("{}/SWHL/RapidOCR/resolve/main/{}", args.hf_base.trim_end_matches('/'), file.remote);
            download_file(&url, &ocr_dir, file, args.force)?;
        }
        let dict = DownloadFile {
            remote: "",
            local: "dict.txt",
            min_bytes: 1_000,
        };
        download_file(&args.dict_url, &ocr_dir, &dict, args.force)?;
    }

    println!("完成。若下载源较慢，可用 --hf-base https://hf-mirror.com 重试。");
    Ok(())
}

fn download_bge(
    base: &Path,
    repo: &str,
    remote_onnx: &'static str,
    local_onnx: &'static str,
    onnx_min_bytes: u64,
    args: &DownloadArgs,
) -> Result<()> {
    let dir = base.join(repo);
    println!("下载 {repo} → {}", dir.display());
    let files = [
        DownloadFile {
            remote: "config.json",
            local: "config.json",
            min_bytes: 100,
        },
        DownloadFile {
            remote: remote_onnx,
            local: local_onnx,
            min_bytes: onnx_min_bytes,
        },
        DownloadFile {
            remote: "special_tokens_map.json",
            local: "special_tokens_map.json",
            min_bytes: 10,
        },
        DownloadFile {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            min_bytes: 100_000,
        },
        DownloadFile {
            remote: "tokenizer_config.json",
            local: "tokenizer_config.json",
            min_bytes: 100,
        },
    ];
    for file in &files {
        let url = format!(
            "{}/Xenova/{repo}/resolve/main/{}",
            args.hf_base.trim_end_matches('/'),
            file.remote
        );
        download_file(&url, &dir, file, args.force)?;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct OnlyModels {
    small: bool,
    base: bool,
    ocr: bool,
}

fn parse_only(value: &str) -> Result<OnlyModels> {
    let mut only = OnlyModels {
        small: false,
        base: false,
        ocr: false,
    };
    for part in value.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "" => {}
            "all" => {
                only.small = true;
                only.ocr = true;
            }
            "bge" | "embedding" | "semantic" => only.small = true,
            "bge-base" | "base" => only.base = true,
            "ocr" => only.ocr = true,
            other => bail!("--only 只支持 bge / bge-base / ocr / all，收到：{other}"),
        }
    }
    if !only.small && !only.base && !only.ocr {
        bail!("--only 至少要包含 bge、bge-base 或 ocr");
    }
    Ok(only)
}

fn models_dir() -> Result<PathBuf> {
    Ok(std::env::current_exe()?
        .parent()
        .context("无法定位 exe 目录")?
        .join("models"))
}

fn download_file(url: &str, dir: &Path, file: &DownloadFile, force: bool) -> Result<()> {
    let dest = dir.join(file.local);
    if !force {
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.len() >= file.min_bytes {
                println!("  已存在，跳过：{}", dest.display());
                return Ok(());
            }
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("创建模型目录失败：{}", parent.display()))?;
    }
    println!("  下载：{url}");
    let mut response = ureq::get(url)
        .call()
        .with_context(|| format!("请求失败：{url}"))?;
    let bytes = response
        .body_mut()
        .with_config()
        .limit(256 * 1024 * 1024)
        .read_to_vec()
        .with_context(|| format!("读取响应失败：{url}"))?;
    if (bytes.len() as u64) < file.min_bytes {
        bail!(
            "下载内容过小：{}（{} 字节，至少需要 {}）",
            url,
            bytes.len(),
            file.min_bytes
        );
    }
    let part = dest.with_extension("part");
    std::fs::write(&part, &bytes)
        .with_context(|| format!("写入临时文件失败：{}", part.display()))?;
    if dest.exists() {
        std::fs::remove_file(&dest)
            .with_context(|| format!("删除旧文件失败：{}", dest.display()))?;
    }
    std::fs::rename(&part, &dest)
        .with_context(|| format!("重命名失败：{}", dest.display()))?;
    println!("  完成：{}（{} 字节）", dest.display(), bytes.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_parser_accepts_aliases_and_rejects_empty() {
        let small = OnlyModels {
            small: true,
            base: false,
            ocr: false,
        };
        assert_eq!(
            parse_only("all").unwrap(),
            OnlyModels {
                small: true,
                base: false,
                ocr: true
            }
        );
        assert_eq!(parse_only("bge").unwrap(), small);
        assert_eq!(parse_only("semantic").unwrap(), small);
        assert_eq!(
            parse_only("bge-base").unwrap(),
            OnlyModels {
                small: false,
                base: true,
                ocr: false
            }
        );
        assert_eq!(
            parse_only("base,ocr").unwrap(),
            OnlyModels {
                small: false,
                base: true,
                ocr: true
            }
        );
        assert_eq!(
            parse_only("bge,bge-base,ocr").unwrap(),
            OnlyModels {
                small: true,
                base: true,
                ocr: true
            }
        );
        assert!(parse_only("").is_err());
        assert!(parse_only("docx").is_err());
    }
}
