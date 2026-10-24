//! Optional OCR for PDF pages without a usable text layer.
//!
//! Rendering is done by PDFium; recognition by RapidOCR/PP-OCRv4 ONNX models.
//! Both are optional at runtime: if the DLL or models are missing, OCR is
//! disabled with a logged warning and text-layer extraction keeps working.

use crate::config::OcrConfig;
use anyhow::{bail, Context, Result};
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};
use rapidocr_core::{
    config::{DetConfig, InferenceOptions, LimitType, PipelineConfig, RapidOcrConfig, RecConfig},
    RapidOcr,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone)]
struct OcrOptions {
    enabled: bool,
    dir: PathBuf,
    pdfium: PathBuf,
    dpi: u32,
    max_pages: usize,
    min_text_chars: usize,
}

impl OcrOptions {
    fn from_config(cfg: &OcrConfig, base: &Path, _index_dir: &Path) -> Self {
        Self {
            enabled: cfg.enabled,
            dir: cfg.resolve_dir(base),
            pdfium: cfg.resolve_pdfium(base),
            dpi: cfg.dpi,
            max_pages: cfg.max_pages,
            min_text_chars: cfg.min_text_chars,
        }
    }
}

static OPTIONS: OnceLock<OcrOptions> = OnceLock::new();
static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();
static TEXTLESS_PDF: OnceLock<bool> = OnceLock::new();
static STATUS: AtomicU8 = AtomicU8::new(0);

const STATUS_DISABLED: u8 = 0;
const STATUS_IDLE: u8 = 1;
const STATUS_LOADING: u8 = 2;
const STATUS_READY: u8 = 3;
const STATUS_UNAVAILABLE: u8 = 4;
const STATUS_BUSY: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrStatus {
    Disabled,
    Idle,
    Loading,
    Ready,
    Busy,
    Unavailable,
}

impl OcrStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "关闭",
            Self::Idle => "待加载",
            Self::Loading => "加载中",
            Self::Ready => "就绪",
            Self::Busy => "识别中",
            Self::Unavailable => "未就绪",
        }
    }
}

pub fn status() -> OcrStatus {
    match STATUS.load(Ordering::Relaxed) {
        STATUS_IDLE => OcrStatus::Idle,
        STATUS_LOADING => OcrStatus::Loading,
        STATUS_READY => OcrStatus::Ready,
        STATUS_BUSY => OcrStatus::Busy,
        STATUS_UNAVAILABLE => OcrStatus::Unavailable,
        _ => OcrStatus::Disabled,
    }
}
static RUNTIME: OnceLock<Mutex<RuntimeState>> = OnceLock::new();

enum RuntimeState {
    Uninitialized,
    Ready(Box<OcrRuntime>),
    Unavailable,
}

struct OcrRuntime {
    pdfium: Pdfium,
    ocr: RapidOcr,
    dpi: u32,
    cache_dir: PathBuf,
    cache_key: String,
}

#[derive(Default, Serialize, Deserialize)]
struct PageCache {
    pages: BTreeMap<u32, String>,
}

/// Store OCR options once, before the index worker starts.
pub fn configure(
    cfg: &OcrConfig,
    base: &Path,
    index_dir: &Path,
    textless_pdf: bool,
) -> Option<PathBuf> {
    let _ = TEXTLESS_PDF.set(textless_pdf);
    if !textless_pdf {
        STATUS.store(STATUS_DISABLED, Ordering::Relaxed);
        log::info!("无文字层 PDF：跳过（[index] textless_pdf = false）");
        return None;
    }
    if !cfg.enabled {
        STATUS.store(STATUS_DISABLED, Ordering::Relaxed);
        return None;
    }
    STATUS.store(STATUS_IDLE, Ordering::Relaxed);
    let options = OcrOptions::from_config(cfg, base, index_dir);
    log::info!(
        "OCR：启用；模型目录 {}；PDFium {}；DPI {}；最多 {} 页",
        options.dir.display(),
        options.pdfium.display(),
        options.dpi,
        if options.max_pages == 0 {
            "不限".to_string()
        } else {
            options.max_pages.to_string()
        }
    );
    let _ = CACHE_DIR.set(index_dir.join("ocr-cache"));
    let _ = OPTIONS.set(options);
    Some(index_dir.join("ocr-cache"))
}

pub fn is_enabled() -> bool {
    OPTIONS.get().is_some_and(|options| options.enabled)
}

pub fn textless_pdf_enabled() -> bool {
    TEXTLESS_PDF.get().copied().unwrap_or(true)
}

pub fn min_text_chars() -> usize {
    OPTIONS.get().map(|options| options.min_text_chars).unwrap_or(20)
}

/// OCR the requested 1-based page numbers. Returns only pages that produced
/// text. Errors are consumed here because OCR is a best-effort enhancement.
pub fn recognize_pages(bytes: &[u8], page_numbers: &[u32]) -> BTreeMap<u32, String> {
    let Some(options) = OPTIONS.get() else {
        return BTreeMap::new();
    };
    if !options.enabled || page_numbers.is_empty() {
        return BTreeMap::new();
    }

    let mut state = match RUNTIME.get_or_init(|| Mutex::new(RuntimeState::Uninitialized)).lock()
    {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };

    if matches!(*state, RuntimeState::Uninitialized) {
        STATUS.store(STATUS_LOADING, Ordering::Relaxed);
        match OcrRuntime::load(options) {
            Ok(runtime) => {
                log::info!(
                    "OCR 就绪：{}；模型 {}",
                    runtime.cache_key,
                    options.dir.display()
                );
                *state = RuntimeState::Ready(Box::new(runtime));
                STATUS.store(STATUS_READY, Ordering::Relaxed);
            }
            Err(err) => {
                log::warn!("OCR 未就绪，扫描件将跳过：{err:#}");
                *state = RuntimeState::Unavailable;
                STATUS.store(STATUS_UNAVAILABLE, Ordering::Relaxed);
            }
        }
    }

    match &mut *state {
        RuntimeState::Ready(runtime) => runtime.recognize(bytes, page_numbers, options),
        RuntimeState::Uninitialized | RuntimeState::Unavailable => BTreeMap::new(),
    }
}

impl OcrRuntime {
    fn load(options: &OcrOptions) -> Result<Self> {
        let det = options.dir.join("det.onnx");
        let rec = options.dir.join("rec.onnx");
        let dict = options.dir.join("dict.txt");
        for path in [&det, &rec, &dict] {
            if !path.is_file() {
                bail!("缺少 OCR 模型文件：{}", path.display());
            }
        }
        if !options.pdfium.is_file() {
            bail!("缺少 PDFium 动态库：{}", options.pdfium.display());
        }

        let bindings = Pdfium::bind_to_library(&options.pdfium).map_err(|err| {
            anyhow::anyhow!(
                "加载 PDFium 失败（{}）：{err}",
                options.pdfium.display()
            )
        })?;
        let pdfium = Pdfium::new(bindings);

        let env = crate::embed::environment().context("初始化 ONNX 运行环境失败")?;
        let mut cfg = RapidOcrConfig::ppocr_v6_small(&options.dir);
        cfg.pipeline = PipelineConfig::without_cls();
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_div(2).clamp(1, 8))
            .unwrap_or(4);
        cfg.inference = InferenceOptions {
            intra_threads: threads,
            ..Default::default()
        };
        cfg.text_score = 0.5;
        cfg.det = Some(DetConfig {
            model_path: det.clone(),
            limit_side_len: 736,
            limit_type: LimitType::Min,
            input_limits: Default::default(),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            thresh: 0.3,
            box_thresh: 0.5,
            max_candidates: 1000,
            unclip_ratio: 1.6,
            min_size: 3,
        });
        cfg.rec = Some(RecConfig {
            model_path: rec.clone(),
            dict_path: dict.clone(),
            image_shape: [3, 48, 320],
            batch_size: 6,
        });
        cfg.cls = None;
        let ocr = RapidOcr::new(cfg, &env).context("初始化 RapidOCR 失败")?;

        let cache_key = format!(
            "{}x{}-{}-{}",
            det.metadata()?.len(),
            rec.metadata()?.len(),
            options.dpi,
            "ppocrv4"
        );
        let cache_dir = CACHE_DIR
            .get()
            .cloned()
            .unwrap_or_else(|| PathBuf::from("ocr-cache"));
        let _ = std::fs::create_dir_all(&cache_dir);
        Ok(Self {
            pdfium,
            ocr,
            dpi: options.dpi,
            cache_dir,
            cache_key,
        })
    }

    fn recognize(
        &mut self,
        bytes: &[u8],
        page_numbers: &[u32],
        options: &OcrOptions,
    ) -> BTreeMap<u32, String> {
        let hash = blake3::hash(bytes).to_hex().to_string();
        let cache_path = self
            .cache_dir
            .join(format!("{hash}-{}.bin", self.cache_key));
        let mut cache = std::fs::read(&cache_path)
            .ok()
            .and_then(|bytes| bincode::deserialize::<PageCache>(&bytes).ok())
            .unwrap_or_default();

        let mut wanted: Vec<u32> = page_numbers.to_vec();
        if options.max_pages > 0 && wanted.len() > options.max_pages {
            log::warn!(
                "OCR 单文件页数上限 {}，本次跳过其余 {} 页",
                options.max_pages,
                wanted.len() - options.max_pages
            );
            wanted.truncate(options.max_pages);
        }
        let missing: Vec<u32> = wanted
            .iter()
            .copied()
            .filter(|page| !cache.pages.contains_key(page))
            .collect();
        if missing.is_empty() {
            return wanted
                .into_iter()
                .filter_map(|page| cache.pages.get(&page).cloned().map(|text| (page, text)))
                .collect();
        }

        let doc = match self.pdfium.load_pdf_from_byte_slice(bytes, None) {
            Ok(doc) => doc,
            Err(err) => {
                log::warn!("PDFium 打开 PDF 失败，跳过 OCR：{err}");
                STATUS.store(STATUS_READY, Ordering::Relaxed);
                return BTreeMap::new();
            }
        };
        let page_count = doc.pages().len();
        let total_missing = missing.len();
        let mut done = 0usize;
        STATUS.store(STATUS_BUSY, Ordering::Relaxed);
        for page_num in missing {
            if page_num == 0 || page_num as i32 > page_count {
                continue;
            }
            let page_index = (page_num - 1) as i32;
            let page = match doc.pages().get(page_index) {
                Ok(page) => page,
                Err(err) => {
                    log::warn!("OCR 第 {page_num} 页无法打开：{err}");
                    continue;
                }
            };
            let target_width = ((page.width().value * self.dpi as f32 / 72.0).round() as i32)
                .clamp(64, 4096);
            let config = PdfRenderConfig::new().set_target_width(target_width);
            let image = match page.render_with_config(&config) {
                Ok(bitmap) => match bitmap.as_image() {
                    Ok(image) => image.to_rgb8(),
                    Err(err) => {
                        log::warn!("OCR 第 {page_num} 页位图转换失败：{err}");
                        continue;
                    }
                },
                Err(err) => {
                    log::warn!("OCR 第 {page_num} 页渲染失败：{err}");
                    continue;
                }
            };
            let text = match self.ocr.run_image(&image) {
                Ok(output) => output
                    .lines
                    .iter()
                    .map(|line| line.text.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"),
                Err(err) => {
                    log::warn!("OCR 第 {page_num} 页识别失败：{err}");
                    continue;
                }
            };
            if !text.trim().is_empty() {
                cache.pages.insert(page_num, text);
                // Save after every page so quitting halfway through a large
                // scanned PDF does not throw away the pages already done.
                save_cache(&cache_path, &cache);
            }
            done += 1;
            if done.is_multiple_of(10) || done == total_missing {
                log::info!("OCR 进度：{done}/{total_missing} 页");
            }
        }

        save_cache(&cache_path, &cache);
        STATUS.store(STATUS_READY, Ordering::Relaxed);
        wanted
            .into_iter()
            .filter_map(|page| cache.pages.get(&page).cloned().map(|text| (page, text)))
            .collect()
    }
}

fn save_cache(path: &Path, cache: &PageCache) {
    if let Ok(bytes) = bincode::serialize(cache) {
        let _ = std::fs::write(path, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_roundtrip() {
        let mut cache = PageCache::default();
        cache.pages.insert(1, "第一页".to_string());
        let bytes = bincode::serialize(&cache).unwrap();
        let loaded: PageCache = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded.pages.get(&1).map(String::as_str), Some("第一页"));
    }
}
