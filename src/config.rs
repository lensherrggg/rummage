use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

pub const LOCAL_RULE_FILE: &str = ".rummage.toml";

pub const BUILTIN_IGNORE_DIRS: &[&str] = &[
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".obsidian",
    ".trash",
    ".idea",
    ".vscode",
    "attachments",
    "assets",
    "images",
    "附件",
    ".rummage",
];

pub const BUILTIN_IGNORE_FILES: &[&str] = &[".DS_Store", "Thumbs.db"];

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub watch: WatchConfig,
    pub index: IndexConfig,
    pub search: SearchConfig,
    pub ocr: OcrConfig,
    pub include: IncludeConfig,
    pub ignore: IgnoreConfig,
    pub normalize: NormalizeConfig,
    pub model: ModelConfig,
    pub rerank: RerankConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct WatchConfig {
    pub roots: Vec<PathBuf>,
    pub debounce_ms: u64,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            debounce_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct IndexConfig {
    pub dir: Option<PathBuf>,
    /// 参与索引的文件扩展名；省略时使用 [`DEFAULT_EXTENSIONS`]。
    pub extensions: Option<Vec<String>>,
    /// 是否处理无文字层 PDF。`false` 时连 OCR 也不做，直接跳过。
    pub textless_pdf: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dir: None,
            extensions: None,
            textless_pdf: true,
        }
    }
}

pub const DEFAULT_EXTENSIONS: &[&str] = &["md", "markdown", "docx", "pdf", "ofd"];

/// 检索候选与最终结果条数。两者都可省略，默认值与旧版本行为一致。
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SearchConfig {
    /// 关键词与语义两路各自先取多少个分块候选，再按文件聚合。
    /// 调大可提高召回，但 RRF 融合与重排的输入也更多。
    pub candidates: usize,
    /// 最终返回、在结果列表里显示的文件数上限。
    pub results: usize,
    /// RRF 的 rank constant。调大可降低头部排名的权重。
    pub rrf_k: f64,
    /// RRF 中关键词一路的权重。
    pub lexical_weight: f64,
    /// RRF 中语义一路的权重。
    pub semantic_weight: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            candidates: 50,
            results: 10,
            rrf_k: 60.0,
            lexical_weight: 1.0,
            semantic_weight: 1.0,
        }
    }
}

/// Formats whose bytes are not plain UTF-8 text. Accepting them now would index
/// binary garbage; they need a text-extraction layer first.
const UNSUPPORTED_BINARY_EXTENSIONS: &[&str] = &["doc", "xls", "xlsx", "ppt", "pptx"];

/// 可选的扫描件 OCR：PDFium 负责渲染，RapidOCR/PP-OCRv4 ONNX 负责识别。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrProvider {
    /// 优先 DirectML（需 DirectML-capable Windows/GPU），失败回退 CPU。
    #[default]
    Auto,
    Cpu,
    DirectMl,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct OcrConfig {
    /// 默认开启；模型或 PDFium 缺失时自动降级为只处理文字层。
    pub enabled: bool,
    /// OCR 推理后端：auto / cpu / directml。
    pub provider: OcrProvider,
    /// OCR 模型目录，默认是 exe 同目录的 `models\ocr`。
    pub dir: Option<PathBuf>,
    /// PDFium 动态库路径，默认是 exe 同目录的 `pdfium.dll`。
    pub pdfium: Option<PathBuf>,
    pub dpi: u32,
    /// 单个 PDF 最多 OCR 多少页；0 表示不限制。
    pub max_pages: usize,
    /// 页面文字少于该字符数时触发 OCR。
    pub min_text_chars: usize,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: OcrProvider::Auto,
            dir: None,
            pdfium: None,
            dpi: 200,
            max_pages: 100,
            min_text_chars: 20,
        }
    }
}

impl OcrConfig {
    pub fn resolve_dir(&self, base: &Path) -> PathBuf {
        match &self.dir {
            Some(dir) if dir.is_absolute() => dir.clone(),
            Some(dir) => base.join(dir),
            None => default_exe_dir().join("models").join("ocr"),
        }
    }

    pub fn resolve_pdfium(&self, base: &Path) -> PathBuf {
        match &self.pdfium {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => base.join(path),
            None => default_exe_dir().join("pdfium.dll"),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct IncludeConfig {
    /// 为空时不启用白名单。不含路径分隔符的模式匹配文件名；
    /// 含分隔符的模式匹配相对监控根目录的路径。
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct IgnoreConfig {
    pub dirs: Vec<String>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct NormalizeConfig {
    pub enabled: bool,
    pub control_chars: bool,
    pub width_fold: bool,
    pub merge_hard_breaks: bool,
    pub drop_page_number_lines: bool,
    pub drop_page_markers: bool,
    pub drop_repeated_headers: bool,
    pub collapse_blank_lines: bool,
    pub collapse_spaces: bool,
    pub strip_markdown: bool,
    pub links: bool,
    pub table_pipes: bool,
    pub front_matter: bool,
    pub page_number_max_len: usize,
    pub header_footer_max_len: usize,
    pub header_footer_min_repeat: usize,
    pub header_footer_zone_lines: usize,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            control_chars: true,
            width_fold: true,
            merge_hard_breaks: true,
            drop_page_number_lines: true,
            drop_page_markers: true,
            drop_repeated_headers: true,
            collapse_blank_lines: true,
            collapse_spaces: true,
            strip_markdown: true,
            links: true,
            table_pipes: true,
            front_matter: true,
            page_number_max_len: 20,
            header_footer_max_len: 40,
            header_footer_min_repeat: 3,
            header_footer_zone_lines: 20,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelConfig {
    pub enabled: bool,
    pub dir: Option<PathBuf>,
    pub file: String,
    pub batch_size: usize,
    pub max_length: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
            file: "model.onnx".to_string(),
            batch_size: 32,
            max_length: 512,
        }
    }
}

/// 可选的 EmbeddingGemma 2 二次重排：只调整基础排序的前 `top_n` 个文件。
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct RerankConfig {
    pub enabled: bool,
    pub dir: Option<PathBuf>,
    /// 相对 `dir` 的 ONNX 路径；HF 导出的目录结构通常是 `onnx/model_quantized.onnx`。
    pub file: String,
    pub top_n: usize,
    /// Gemma 分数在混合中的权重，0 等于不重排，1 等于完全按 Gemma 排序。
    pub weight: f64,
    pub batch_size: usize,
    pub max_length: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: None,
            file: "onnx/model_quantized.onnx".to_string(),
            top_n: 5,
            weight: 0.5,
            batch_size: 16,
            max_length: 1024,
        }
    }
}

impl Config {
    pub fn load(explicit: Option<&Path>) -> Result<LoadedConfig> {
        let cwd = std::env::current_dir().context("无法获取当前目录")?;
        let (path, required) = match explicit {
            Some(p) => (Some(p.to_path_buf()), true),
            None => {
                let candidate = cwd.join("rummage.toml");
                if candidate.is_file() {
                    (Some(candidate), false)
                } else {
                    (None, false)
                }
            }
        };

        let (config, base) = match &path {
            Some(p) => {
                if required && !p.is_file() {
                    bail!("配置文件不存在：{}", p.display());
                }
                let text = std::fs::read_to_string(p)
                    .with_context(|| format!("读取配置文件失败：{}", p.display()))?;
                let cfg: Config = toml::from_str(&text)
                    .with_context(|| format!("解析配置文件失败：{}", p.display()))?;
                let base = p
                    .parent()
                    .filter(|d| !d.as_os_str().is_empty())
                    .map(|d| cwd.join(d))
                    .unwrap_or_else(|| cwd.clone());
                (cfg, base)
            }
            None => (Config::default(), cwd.clone()),
        };

        config.validate()?;
        Ok(LoadedConfig { config, path, base })
    }

    fn validate(&self) -> Result<()> {
        if self.search.candidates == 0 {
            bail!("search.candidates 必须大于 0");
        }
        if self.search.results == 0 {
            bail!("search.results 必须大于 0");
        }
        if !self.search.rrf_k.is_finite() || self.search.rrf_k <= 0.0 {
            bail!("search.rrf_k 必须是大于 0 的有限数");
        }
        if !self.search.lexical_weight.is_finite() || self.search.lexical_weight < 0.0 {
            bail!("search.lexical_weight 必须是非负有限数");
        }
        if !self.search.semantic_weight.is_finite() || self.search.semantic_weight < 0.0 {
            bail!("search.semantic_weight 必须是非负有限数");
        }
        if self.search.lexical_weight == 0.0 && self.search.semantic_weight == 0.0 {
            bail!("search.lexical_weight 和 search.semantic_weight 不能同时为 0");
        }
        if !(72..=600).contains(&self.ocr.dpi) {
            bail!("ocr.dpi 必须在 72 和 600 之间");
        }
        if self.model.batch_size == 0 {
            bail!("model.batch_size 必须大于 0");
        }
        if self.model.max_length == 0 {
            bail!("model.max_length 必须大于 0");
        }
        if self.rerank.top_n == 0 {
            bail!("rerank.top_n 必须大于 0");
        }
        if !self.rerank.weight.is_finite() || !(0.0..=1.0).contains(&self.rerank.weight) {
            bail!("rerank.weight 必须在 0 和 1 之间");
        }
        if self.rerank.batch_size == 0 {
            bail!("rerank.batch_size 必须大于 0");
        }
        if self.rerank.max_length == 0 {
            bail!("rerank.max_length 必须大于 0");
        }
        if let Some(raw) = &self.index.extensions {
            validate_extensions(raw)?;
        }
        Ok(())
    }

    /// 命令行 `--ignore` 的补充规则：同时按目录名和文件名匹配。
    /// 与 `[ignore]` 一样，只作用于全局规则；目录局部 `.rummage.toml` 仍会整体替换全局忽略项。
    pub fn apply_cli_ignores(&mut self, patterns: &[String]) {
        for raw in patterns {
            let pattern = raw.trim();
            if pattern.is_empty() {
                continue;
            }
            self.ignore.dirs.push(pattern.to_string());
            self.ignore.names.push(pattern.to_string());
        }
    }

    pub fn resolve_roots(&self, cli_dirs: &[PathBuf], base: &Path) -> Result<Vec<PathBuf>> {
        let mut roots = Vec::new();
        let raw: Vec<PathBuf> = if cli_dirs.is_empty() {
            self.watch.roots.clone()
        } else {
            cli_dirs.to_vec()
        };
        if raw.is_empty() {
            bail!("未指定监控目录：请使用 --dir，或在 rummage.toml 的 [watch] roots 中配置");
        }
        for p in raw {
            let abs = if p.is_absolute() { p } else { base.join(p) };
            let canonical = simplify_path(
                &abs.canonicalize()
                    .with_context(|| format!("监控目录不存在或不可访问：{}", abs.display()))?,
            );
            if !canonical.is_dir() {
                bail!("监控目录不是目录：{}", canonical.display());
            }
            if !roots.iter().any(|r: &PathBuf| r == &canonical) {
                roots.push(canonical);
            }
        }
        Ok(roots)
    }

    pub fn resolve_index_dir(&self, roots: &[PathBuf], base: &Path) -> PathBuf {
        let dir = match &self.index.dir {
            Some(dir) if dir.is_absolute() => dir.clone(),
            Some(dir) => base.join(dir),
            None => roots[0].join(".rummage"),
        };
        simplify_path(&dir)
    }

    pub fn resolve_model_dir(&self, base: &Path) -> PathBuf {
        if let Some(dir) = &self.model.dir {
            return if dir.is_absolute() {
                dir.clone()
            } else {
                base.join(dir)
            };
        }
        if let Ok(dir) = std::env::var("RUMMAGE_MODEL_DIR") {
            if !dir.trim().is_empty() {
                return PathBuf::from(dir);
            }
        }
        default_model_dir("bge-small-zh-v1.5")
    }

    pub fn resolve_rerank_model_dir(&self, base: &Path) -> PathBuf {
        if let Some(dir) = &self.rerank.dir {
            return if dir.is_absolute() {
                dir.clone()
            } else {
                base.join(dir)
            };
        }
        if let Ok(dir) = std::env::var("RUMMAGE_RERANK_MODEL_DIR") {
            if !dir.trim().is_empty() {
                return PathBuf::from(dir);
            }
        }
        default_model_dir("embeddinggemma-2")
    }
}

fn default_exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn default_model_dir(name: &str) -> PathBuf {
    default_exe_dir().join("models").join(name)
}

#[derive(Debug)]
pub struct LoadedConfig {
    pub config: Config,
    pub path: Option<PathBuf>,
    pub base: PathBuf,
}

#[derive(Debug, Clone)]
pub struct IgnoreRules {
    dirs: Vec<String>,
    names: Vec<String>,
}

impl IgnoreRules {
    pub fn new(cfg: &IgnoreConfig) -> Self {
        let dirs = BUILTIN_IGNORE_DIRS
            .iter()
            .copied()
            .chain(cfg.dirs.iter().map(String::as_str))
            .filter_map(normalize_pattern)
            .collect();
        let names = BUILTIN_IGNORE_FILES
            .iter()
            .copied()
            .chain(cfg.names.iter().map(String::as_str))
            .filter_map(normalize_pattern)
            .collect();
        Self { dirs, names }
    }

    pub fn is_ignored_dir(&self, name: &OsStr) -> bool {
        matches_any(&self.dirs, name)
    }

    pub fn is_ignored_file_name(&self, name: &OsStr) -> bool {
        matches_any(&self.names, name)
    }

    pub fn has_ignored_component(&self, path: &Path) -> bool {
        path.components().any(|component| {
            let Component::Normal(name) = component else {
                return false;
            };
            matches_any(&self.dirs, name)
        })
    }
}

#[derive(Debug, Clone)]
pub struct IncludeRules {
    patterns: Vec<IncludePattern>,
}

#[derive(Debug, Clone)]
enum IncludePattern {
    Name(String),
    Path(String),
}

impl IncludeRules {
    pub fn new(cfg: &IncludeConfig) -> Self {
        let patterns = cfg
            .patterns
            .iter()
            .filter_map(|raw| normalize_include_pattern(raw))
            .map(|pattern| {
                if pattern.contains('/') {
                    IncludePattern::Path(pattern)
                } else {
                    IncludePattern::Name(pattern)
                }
            })
            .collect();
        Self { patterns }
    }

    /// 白名单为空时全部收录。文件名模式匹配文件名的任意位置；
    /// 路径模式匹配相对任一路径根的路径。
    pub fn is_included(&self, path: &Path, roots: &[PathBuf]) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase());
        let relative = relative_path(path, roots);
        self.patterns.iter().any(|pattern| match pattern {
            IncludePattern::Name(pattern) => name
                .as_deref()
                .is_some_and(|name| wildcard_match(pattern, name)),
            IncludePattern::Path(pattern) => relative
                .as_deref()
                .is_some_and(|relative| path_glob_match(pattern, relative)),
        })
    }
}

fn normalize_include_pattern(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let mut pattern = trimmed.replace('\\', "/");
    while let Some(rest) = pattern.strip_prefix("./") {
        pattern = rest.to_string();
    }
    let pattern = pattern.trim_start_matches('/').trim_end_matches('/');
    if pattern.is_empty() {
        return None;
    }
    let pattern = if trimmed.ends_with('/') || trimmed.ends_with('\\') {
        format!("{pattern}/**")
    } else {
        pattern.to_string()
    };
    Some(pattern.to_lowercase())
}

fn relative_path(path: &Path, roots: &[PathBuf]) -> Option<String> {
    let path = path.to_string_lossy().replace('\\', "/").to_lowercase();
    roots.iter().find_map(|root| {
        let root = root.to_string_lossy().replace('\\', "/").to_lowercase();
        let root = root.trim_end_matches('/');
        path.strip_prefix(&format!("{root}/")).map(str::to_string)
    })
}

enum PathToken {
    Literal(char),
    Any,
    Star,
    GlobStar,
    /// `**/`：匹配零个或多个完整目录层级，因此 `**/a.md` 也能匹配根目录下的 `a.md`。
    GlobStarDir,
    Class(CharClass),
}

fn path_glob_match(pattern: &str, path: &str) -> bool {
    let tokens = tokenize_path_glob(pattern);
    let text: Vec<char> = path.chars().collect();
    let mut dp = vec![false; text.len() + 1];
    dp[0] = true;

    for token in tokens {
        let mut next = vec![false; text.len() + 1];
        match token {
            PathToken::Literal(ch) => {
                for j in 1..=text.len() {
                    next[j] = dp[j - 1] && text[j - 1] == ch;
                }
            }
            PathToken::Any => {
                for j in 1..=text.len() {
                    next[j] = dp[j - 1] && text[j - 1] != '/';
                }
            }
            PathToken::Star => {
                next[0] = dp[0];
                for j in 1..=text.len() {
                    next[j] = dp[j] || (next[j - 1] && text[j - 1] != '/');
                }
            }
            PathToken::GlobStar => {
                next[0] = dp[0];
                for j in 1..=text.len() {
                    next[j] = dp[j] || next[j - 1];
                }
            }
            PathToken::GlobStarDir => {
                let mut any_before = dp[0];
                next[0] = dp[0];
                for j in 1..=text.len() {
                    next[j] = dp[j] || (any_before && text[j - 1] == '/');
                    any_before |= dp[j];
                }
            }
            PathToken::Class(class) => {
                for j in 1..=text.len() {
                    next[j] = dp[j - 1] && text[j - 1] != '/' && class.matches(text[j - 1]);
                }
            }
        }
        dp = next;
    }
    dp[text.len()]
}

fn tokenize_path_glob(pattern: &str) -> Vec<PathToken> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                let mut end = i;
                while chars.get(end) == Some(&'*') {
                    end += 1;
                }
                if chars.get(end) == Some(&'/') {
                    tokens.push(PathToken::GlobStarDir);
                    i = end + 1;
                } else {
                    tokens.push(if end - i > 1 {
                        PathToken::GlobStar
                    } else {
                        PathToken::Star
                    });
                    i = end;
                }
            }
            '?' => {
                tokens.push(PathToken::Any);
                i += 1;
            }
            '[' => match parse_class(&chars, i) {
                Some((class, close)) => {
                    tokens.push(PathToken::Class(class));
                    i = close + 1;
                }
                None => {
                    tokens.push(PathToken::Literal('['));
                    i += 1;
                }
            },
            ch => {
                tokens.push(PathToken::Literal(ch));
                i += 1;
            }
        }
    }
    tokens
}

#[derive(Debug, Clone)]
pub struct FileTypes {
    extensions: Vec<String>,
}

impl FileTypes {
    pub fn new(index: &IndexConfig) -> Self {
        let extensions = index
            .extensions
            .as_deref()
            .map(normalize_extensions)
            .unwrap_or_else(|| {
                DEFAULT_EXTENSIONS
                    .iter()
                    .map(|ext| ext.to_string())
                    .collect()
            });
        Self { extensions }
    }

    pub fn is_indexable(&self, path: &Path) -> bool {
        path.extension()
            .and_then(OsStr::to_str)
            .map(|ext| {
                let ext = ext.to_lowercase();
                self.extensions.contains(&ext)
            })
            .unwrap_or(false)
    }
}

fn normalize_extensions(raw: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for ext in raw {
        let ext = ext.trim().trim_start_matches('.').to_lowercase();
        if !ext.is_empty() && !out.contains(&ext) {
            out.push(ext);
        }
    }
    out
}

fn validate_extensions(raw: &[String]) -> Result<()> {
    let extensions = normalize_extensions(raw);
    if extensions.is_empty() {
        bail!(
            "index.extensions 不能为空：请至少保留一个扩展名，或删除该字段使用默认 md/markdown/docx/pdf/ofd"
        );
    }
    if let Some(ext) = extensions
        .iter()
        .find(|ext| UNSUPPORTED_BINARY_EXTENSIONS.contains(&ext.as_str()))
    {
        bail!("index.extensions 包含 {ext}：当前版本只支持纯文本、Markdown、docx、pdf 和 ofd");
    }
    Ok(())
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LocalRuleFile {
    index: LocalIndexConfig,
    include: LocalIncludeConfig,
    ignore: LocalIgnoreConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LocalIndexConfig {
    extensions: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LocalIncludeConfig {
    patterns: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LocalIgnoreConfig {
    dirs: Vec<String>,
    names: Vec<String>,
}

/// The rule set in force for one directory subtree. A local rule file replaces
/// the global user rules wholesale; builtin ignore entries always stay on.
#[derive(Debug, Clone)]
pub struct ScopeRules {
    include: IncludeRules,
    ignore: IgnoreRules,
    file_types: FileTypes,
}

impl ScopeRules {
    pub fn global(cfg: &Config) -> Self {
        Self {
            include: IncludeRules::new(&cfg.include),
            ignore: IgnoreRules::new(&cfg.ignore),
            file_types: FileTypes::new(&cfg.index),
        }
    }

    fn from_local(file: LocalRuleFile) -> Result<Self> {
        if let Some(raw) = &file.index.extensions {
            validate_extensions(raw)?;
        }
        Ok(Self {
            include: IncludeRules::new(&IncludeConfig {
                patterns: file.include.patterns,
            }),
            ignore: IgnoreRules::new(&IgnoreConfig {
                dirs: file.ignore.dirs,
                names: file.ignore.names,
            }),
            file_types: FileTypes::new(&IndexConfig {
                dir: None,
                extensions: file.index.extensions,
                textless_pdf: true,
            }),
        })
    }

    pub fn is_indexable(&self, path: &Path, roots: &[PathBuf]) -> bool {
        if is_local_rule_file(path) {
            return false;
        }
        if !self.file_types.is_indexable(path) {
            return false;
        }
        if let Some(name) = path.file_name() {
            if self.ignore.is_ignored_file_name(name) {
                return false;
            }
        }
        !self.ignore.has_ignored_component(path) && self.include.is_included(path, roots)
    }

    pub fn is_ignored_dir(&self, name: &OsStr) -> bool {
        self.ignore.is_ignored_dir(name)
    }
}

pub fn is_local_rule_file(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.eq_ignore_ascii_case(LOCAL_RULE_FILE))
}

fn load_local_rules(path: &Path) -> Result<Option<ScopeRules>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("无法读取 {}", path.display()))?;
    let file: LocalRuleFile =
        toml::from_str(&text).with_context(|| format!("无法解析 {}", path.display()))?;
    ScopeRules::from_local(file)
        .with_context(|| format!("规则无效：{}", path.display()))
        .map(Some)
}

/// Resolves the effective [`ScopeRules`] for a path by walking from its root
/// down, loading `<dir>/.rummage.toml` at each level. The nearest rule file
/// wins; directories without one inherit the parent scope.
pub struct RuleCache {
    global: ScopeRules,
    roots: Vec<PathBuf>,
    dirs: HashMap<String, Rc<ScopeRules>>,
    errors: Vec<String>,
}

impl RuleCache {
    pub fn new(global: ScopeRules, roots: Vec<PathBuf>) -> Self {
        Self {
            global,
            roots,
            dirs: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn for_file(&mut self, path: &Path) -> Rc<ScopeRules> {
        match path.parent() {
            Some(dir) => self.for_dir(dir),
            None => Rc::new(self.global.clone()),
        }
    }

    pub fn for_dir(&mut self, dir: &Path) -> Rc<ScopeRules> {
        let key = path_key(dir);
        if let Some(rules) = self.dirs.get(&key) {
            return Rc::clone(rules);
        }
        let Some(root) = self.containing_root(dir) else {
            let rules = Rc::new(self.global.clone());
            self.dirs.insert(key, Rc::clone(&rules));
            return rules;
        };

        let mut chain = Vec::new();
        let mut cur = Some(dir.to_path_buf());
        while let Some(path) = cur {
            chain.push(path.clone());
            if path_key(&path) == path_key(&root) {
                break;
            }
            cur = path.parent().map(Path::to_path_buf);
        }
        chain.reverse();

        let mut current = Rc::new(self.global.clone());
        for path in chain {
            current = self.rules_at(&path, &current);
        }
        current
    }

    fn rules_at(&mut self, dir: &Path, inherited: &Rc<ScopeRules>) -> Rc<ScopeRules> {
        let key = path_key(dir);
        if let Some(rules) = self.dirs.get(&key) {
            return Rc::clone(rules);
        }
        let file = dir.join(LOCAL_RULE_FILE);
        let rules = match load_local_rules(&file) {
            Ok(Some(rules)) => Rc::new(rules),
            Ok(None) => Rc::clone(inherited),
            Err(err) => {
                self.errors
                    .push(format!("局部规则不可用，沿用上级规则（{err}）"));
                Rc::clone(inherited)
            }
        };
        self.dirs.insert(key, Rc::clone(&rules));
        rules
    }

    /// Drop cached rules for `dir` and everything below it.
    pub fn invalidate(&mut self, dir: &Path) {
        let key = path_key(dir);
        let child_prefixes = [format!("{key}\\"), format!("{key}/")];
        self.dirs.retain(|cached, _| {
            cached != &key
                && !child_prefixes
                    .iter()
                    .any(|prefix| cached.starts_with(prefix))
        });
    }

    pub fn take_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.errors)
    }

    fn containing_root(&self, dir: &Path) -> Option<PathBuf> {
        self.roots
            .iter()
            .filter(|root| is_under(dir, root))
            .max_by_key(|root| root.as_os_str().len())
            .cloned()
    }
}

fn is_under(path: &Path, root: &Path) -> bool {
    let p = path.to_string_lossy().to_lowercase();
    let r = root.to_string_lossy().to_lowercase();
    let r = r.trim_end_matches(['\\', '/']);
    p == r || p.starts_with(&format!("{r}\\")) || p.starts_with(&format!("{r}/"))
}

fn normalize_pattern(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_lowercase())
}

fn matches_any(patterns: &[String], name: &OsStr) -> bool {
    let name = name.to_string_lossy().to_lowercase();
    patterns
        .iter()
        .any(|pattern| wildcard_match(pattern, &name))
}

/// Case-insensitive glob for a single file/directory name: `*`, `?`,
/// `[abc]` and `[!abc]` (or `[^abc]`). An unclosed `[` is a literal.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let mut dp = vec![false; text.len() + 1];
    dp[0] = true;

    let mut i = 0;
    while i < pattern.len() {
        let mut next = vec![false; text.len() + 1];
        match pattern[i] {
            '*' => {
                let mut reachable = false;
                for j in 0..=text.len() {
                    reachable |= dp[j];
                    next[j] = reachable;
                }
                i += 1;
            }
            '?' => {
                next[1..].copy_from_slice(&dp[..text.len()]);
                i += 1;
            }
            '[' => match parse_class(&pattern, i) {
                Some((class, close)) => {
                    for j in 1..=text.len() {
                        next[j] = dp[j - 1] && class.matches(text[j - 1]);
                    }
                    i = close + 1;
                }
                None => {
                    for j in 1..=text.len() {
                        next[j] = dp[j - 1] && text[j - 1] == '[';
                    }
                    i += 1;
                }
            },
            ch => {
                for j in 1..=text.len() {
                    next[j] = dp[j - 1] && text[j - 1] == ch;
                }
                i += 1;
            }
        }
        dp = next;
    }
    dp[text.len()]
}

struct CharClass {
    negated: bool,
    ranges: Vec<(char, char)>,
}

impl CharClass {
    fn matches(&self, ch: char) -> bool {
        let hit = self.ranges.iter().any(|&(lo, hi)| lo <= ch && ch <= hi);
        hit != self.negated
    }
}

fn parse_class(pattern: &[char], start: usize) -> Option<(CharClass, usize)> {
    let mut i = start + 1;
    let negated = matches!(pattern.get(i), Some('!') | Some('^'));
    if negated {
        i += 1;
    }
    let mut ranges = Vec::new();
    let mut first = true;
    while i < pattern.len() {
        if pattern[i] == ']' && !first {
            return Some((CharClass { negated, ranges }, i));
        }
        let lo = pattern[i];
        if i + 2 < pattern.len() && pattern[i + 1] == '-' && pattern[i + 2] != ']' {
            ranges.push((lo, pattern[i + 2]));
            i += 3;
        } else {
            ranges.push((lo, lo));
            i += 1;
        }
        first = false;
    }
    None
}

/// Strip the Windows verbatim path prefix (`\\?\`) so paths display and compare
/// in their familiar `D:\...` form. notify and walkdir include it after canonicalize.
pub fn simplify_path(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    let stripped = if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        s.to_string()
    };
    PathBuf::from(stripped)
}

pub fn path_key(path: &Path) -> String {
    simplify_path(path).to_string_lossy().to_lowercase()
}

pub fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(dirs: &[&str], names: &[&str]) -> IgnoreRules {
        IgnoreRules::new(&IgnoreConfig {
            dirs: dirs.iter().map(|s| s.to_string()).collect(),
            names: names.iter().map(|s| s.to_string()).collect(),
        })
    }

    #[test]
    fn exact_names_still_match() {
        let r = rules(&["归档"], &["desktop.ini"]);
        assert!(r.is_ignored_dir(OsStr::new("归档")));
        assert!(!r.is_ignored_dir(OsStr::new("归档2024")));
        assert!(r.is_ignored_file_name(OsStr::new("desktop.ini")));
    }

    #[test]
    fn star_question_and_class_patterns() {
        let r = rules(&["draft*", "20??", "临时[0-9]", "备份[!a]"], &[]);
        assert!(r.is_ignored_dir(OsStr::new("drafts")));
        assert!(r.is_ignored_dir(OsStr::new("draft-2024")));
        assert!(!r.is_ignored_dir(OsStr::new("adrafts")));
        assert!(r.is_ignored_dir(OsStr::new("2024")));
        assert!(!r.is_ignored_dir(OsStr::new("20245")));
        assert!(r.is_ignored_dir(OsStr::new("临时3")));
        assert!(!r.is_ignored_dir(OsStr::new("临时x")));
        assert!(r.is_ignored_dir(OsStr::new("备份b")));
        assert!(!r.is_ignored_dir(OsStr::new("备份a")));
    }

    #[test]
    fn patterns_are_case_insensitive_and_trimmed() {
        let r = rules(&[" Draft* "], &["*.TMP"]);
        assert!(r.is_ignored_dir(OsStr::new("DRAFTS")));
        assert!(r.is_ignored_file_name(OsStr::new("x.tmp")));
    }

    #[test]
    fn wildcard_applies_to_every_path_component() {
        let r = rules(&["draft*"], &[]);
        assert!(r.has_ignored_component(Path::new(r"D:\docs\draft-2024\a.md")));
        assert!(r.has_ignored_component(Path::new(r"D:\docs\a\drafts\a.md")));
        assert!(!r.has_ignored_component(Path::new(r"D:\docs\final\a.md")));
    }

    #[test]
    fn builtin_ignores_survive_user_patterns() {
        let r = rules(&["draft*"], &["*.tmp"]);
        assert!(r.is_ignored_dir(OsStr::new("node_modules")));
        assert!(r.is_ignored_file_name(OsStr::new("Thumbs.db")));
    }

    #[test]
    fn unclosed_class_is_literal() {
        let r = rules(&["[草稿"], &[]);
        assert!(r.is_ignored_dir(OsStr::new("[草稿")));
        assert!(!r.is_ignored_dir(OsStr::new("草稿")));
    }

    #[test]
    fn empty_patterns_match_nothing() {
        assert!(!wildcard_match("", "x"));
        assert!(wildcard_match("*", ""));
    }

    #[test]
    fn file_types_use_defaults_or_config() {
        let default = FileTypes::new(&IndexConfig::default());
        assert!(default.is_indexable(Path::new("a.md")));
        assert!(default.is_indexable(Path::new("A.MARKDOWN")));
        assert!(!default.is_indexable(Path::new("a.txt")));

        let custom = FileTypes::new(&IndexConfig {
            dir: None,
            extensions: Some(vec![".TXT".into(), "md".into()]),
            textless_pdf: true,
        });
        assert!(custom.is_indexable(Path::new("a.txt")));
        assert!(custom.is_indexable(Path::new("a.md")));
        assert!(!custom.is_indexable(Path::new("a.markdown")));
    }

    fn include(patterns: &[&str]) -> IncludeRules {
        IncludeRules::new(&IncludeConfig {
            patterns: patterns.iter().map(|p| p.to_string()).collect(),
        })
    }

    #[test]
    fn empty_include_allows_everything() {
        let roots = vec![PathBuf::from(r"D:\docs")];
        assert!(include(&[]).is_included(Path::new(r"D:\docs\a\b.md"), &roots));
        assert!(include(&["  "]).is_included(Path::new(r"D:\docs\a\b.md"), &roots));
    }

    #[test]
    fn parses_include_section() {
        let cfg: Config =
            toml::from_str("[include]\npatterns = [\"会议纪要*\", \"2025/**\"]\n").unwrap();
        assert_eq!(cfg.include.patterns, ["会议纪要*", "2025/**"]);
        assert!(cfg.ignore.dirs.is_empty());
        assert_eq!(cfg.index.extensions, None);
    }

    #[test]
    fn include_by_file_name_prefix() {
        let roots = vec![PathBuf::from(r"D:\docs")];
        let rules = include(&["会议纪要*", "QMS-?.*"]);
        assert!(rules.is_included(Path::new(r"D:\docs\2025\会议纪要-01.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\QMS-7.md"), &roots));
        assert!(!rules.is_included(Path::new(r"D:\docs\通知.md"), &roots));
    }

    #[test]
    fn include_by_relative_path() {
        let roots = vec![PathBuf::from(r"D:\docs")];
        let rules = include(&["2025/**", "制度/*.md", "**/附件/*.md"]);
        assert!(rules.is_included(Path::new(r"D:\docs\2025\a\b.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\2025\c.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\制度\办法.md"), &roots));
        assert!(!rules.is_included(Path::new(r"D:\docs\制度\sub\办法.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\x\附件\a.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\附件\a.md"), &roots));
        assert!(!rules.is_included(Path::new(r"D:\docs\2024\a.md"), &roots));
    }

    #[test]
    fn include_patterns_are_normalized() {
        let roots = vec![PathBuf::from(r"D:\docs")];
        let rules = include(&[" .\\2025\\ ", "QMS-*.MD", "./制度/"]);
        assert!(rules.is_included(Path::new(r"D:\docs\2025\a.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\qms-1.md"), &roots));
        assert!(!rules.is_included(Path::new(r"D:\docs\通知.md"), &roots));
        assert!(rules.is_included(Path::new(r"D:\docs\制度\a.md"), &roots));
    }

    #[test]
    fn include_star_does_not_cross_directories() {
        let roots = vec![PathBuf::from(r"D:\docs")];
        let rules = include(&["2025/*"]);
        assert!(rules.is_included(Path::new(r"D:\docs\2025\a.md"), &roots));
        assert!(!rules.is_included(Path::new(r"D:\docs\2025\sub\a.md"), &roots));
    }

    #[test]
    fn extensions_reject_empty_and_unsupported_formats() {
        let mut cfg = Config::default();

        cfg.index.extensions = Some(vec![" . ".into()]);
        assert!(cfg.validate().is_err());

        cfg.index.extensions = Some(vec!["doc".into()]);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("doc"), "{err}");

        cfg.index.extensions = Some(vec!["md".into(), "PDF".into(), "DOCX".into(), "OFD".into()]);
        assert!(cfg.validate().is_ok());

        cfg.index.extensions = Some(vec!["md".into(), "xlsx".into()]);
        assert!(cfg.validate().is_err());

        cfg.index.extensions = Some(vec!["md".into(), "txt".into()]);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rerank_defaults_off_and_validates_bounds() {
        let cfg = Config::default();
        assert!(!cfg.rerank.enabled);
        assert_eq!(cfg.rerank.top_n, 5);
        assert!((cfg.rerank.weight - 0.5).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());

        let cfg: Config = toml::from_str(
            "[rerank]\nenabled = true\ndir = \"models/gemma\"\ntop_n = 3\nweight = 0.7\n",
        )
        .unwrap();
        assert!(cfg.rerank.enabled);
        assert_eq!(cfg.rerank.file, "onnx/model_quantized.onnx");
        assert_eq!(cfg.rerank.top_n, 3);
        assert!(cfg.validate().is_ok());

        let mut bad = Config::default();
        bad.rerank.top_n = 0;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.rerank.weight = 1.5;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.rerank.weight = f64::NAN;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.rerank.batch_size = 0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn search_defaults_and_overrides() {
        let cfg = Config::default();
        assert_eq!(cfg.search.candidates, 50);
        assert_eq!(cfg.search.results, 10);
        assert_eq!(cfg.search.rrf_k, 60.0);
        assert_eq!(cfg.search.lexical_weight, 1.0);
        assert_eq!(cfg.search.semantic_weight, 1.0);
        assert!(cfg.validate().is_ok());

        let cfg: Config = toml::from_str(
            "[search]\ncandidates = 20\nresults = 3\nrrf_k = 10.0\nlexical_weight = 2.0\nsemantic_weight = 0.5\n",
        )
        .unwrap();
        assert_eq!(cfg.search.candidates, 20);
        assert_eq!(cfg.search.results, 3);
        assert_eq!(cfg.search.rrf_k, 10.0);
        assert_eq!(cfg.search.lexical_weight, 2.0);
        assert_eq!(cfg.search.semantic_weight, 0.5);
        assert!(cfg.validate().is_ok());

        let mut bad = Config::default();
        bad.search.candidates = 0;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.search.results = 0;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.search.rrf_k = 0.0;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.search.lexical_weight = 0.0;
        bad.search.semantic_weight = 0.0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn ocr_defaults_and_validation() {
        let cfg = Config::default();
        assert!(cfg.index.textless_pdf);
        assert!(cfg.ocr.enabled);
        assert_eq!(cfg.ocr.provider, OcrProvider::Auto);
        assert_eq!(cfg.ocr.dpi, 200);
        assert_eq!(cfg.ocr.max_pages, 100);
        assert_eq!(cfg.ocr.min_text_chars, 20);
        assert!(cfg.validate().is_ok());

        let cfg: Config = toml::from_str(
            "[index]\ntextless_pdf = false\n\n[ocr]\nenabled = false\nprovider = \"directml\"\ndpi = 150\nmax_pages = 10\nmin_text_chars = 5\n",
        )
        .unwrap();
        assert!(!cfg.index.textless_pdf);
        assert!(!cfg.ocr.enabled);
        assert_eq!(cfg.ocr.provider, OcrProvider::DirectMl);
        assert_eq!(cfg.ocr.dpi, 150);
        assert_eq!(cfg.ocr.max_pages, 10);
        assert_eq!(cfg.ocr.min_text_chars, 5);

        let mut bad = Config::default();
        bad.ocr.dpi = 0;
        assert!(bad.validate().is_err());
        bad = Config::default();
        bad.ocr.dpi = 601;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn cli_ignore_appends_to_dirs_and_names() {
        let mut cfg = Config::default();
        cfg.apply_cli_ignores(&[" drafts* ".into(), "".into(), "~$*".into()]);
        assert_eq!(cfg.ignore.dirs, vec!["drafts*", "~$*"]);
        assert_eq!(cfg.ignore.names, vec!["drafts*", "~$*"]);

        let rules = ScopeRules::global(&cfg);
        assert!(rules.is_ignored_dir(OsStr::new("Drafts-2024")));
        assert!(rules.ignore.is_ignored_file_name(OsStr::new("~$报告.docx")));
    }

    fn writable(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, "").unwrap();
    }    fn local_cache(root: &Path, cfg: &Config) -> RuleCache {
        RuleCache::new(ScopeRules::global(cfg), vec![root.to_path_buf()])
    }

    #[test]
    fn local_rule_file_replaces_global_rules() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let sub = root.join("sub");
        writable(&root.join("keep.md"));
        writable(&root.join("skip.txt"));
        writable(&sub.join(".rummage.toml"));
        std::fs::write(
            sub.join(".rummage.toml"),
            "[index]\nextensions = [\"txt\"]\n[include]\npatterns = [\"keep*\"]\n",
        )
        .unwrap();
        writable(&sub.join("keep.md"));
        writable(&sub.join("keep.txt"));
        writable(&sub.join("other.txt"));

        let mut cfg = Config::default();
        cfg.include.patterns = vec!["*.md".into()];
        cfg.index.extensions = Some(vec!["md".into()]);
        let roots = vec![root.to_path_buf()];
        let mut cache = local_cache(root, &cfg);

        let global = |cache: &mut RuleCache, path: PathBuf| {
            let rules = cache.for_file(&path);
            rules.is_indexable(&path, &roots)
        };
        assert!(global(&mut cache, root.join("keep.md")));
        assert!(!global(&mut cache, root.join("skip.txt")));

        let local = |cache: &mut RuleCache, path: PathBuf| {
            let rules = cache.for_file(&path);
            rules.is_indexable(&path, &roots)
        };
        assert!(!local(&mut cache, sub.join("keep.md")));
        assert!(local(&mut cache, sub.join("keep.txt")));
        assert!(!local(&mut cache, sub.join("other.txt")));
        assert!(!local(&mut cache, sub.join(".rummage.toml")));
        assert!(cache.take_errors().is_empty());
    }

    #[test]
    fn nearest_rule_file_wins() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let outer = root.join("a");
        let inner = outer.join("b");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(
            root.join(".rummage.toml"),
            "[index]\nextensions = [\"txt\"]\n",
        )
        .unwrap();
        std::fs::write(
            inner.join(".rummage.toml"),
            "[index]\nextensions = [\"md\"]\n",
        )
        .unwrap();
        writable(&outer.join("x.md"));
        writable(&outer.join("x.txt"));
        writable(&inner.join("y.md"));
        writable(&inner.join("y.txt"));

        let mut cache = local_cache(root, &Config::default());
        let roots = vec![root.to_path_buf()];
        let check = |cache: &mut RuleCache, path: PathBuf| {
            let rules = cache.for_file(&path);
            rules.is_indexable(&path, &roots)
        };
        assert!(!check(&mut cache, outer.join("x.md")));
        assert!(check(&mut cache, outer.join("x.txt")));
        assert!(check(&mut cache, inner.join("y.md")));
        assert!(!check(&mut cache, inner.join("y.txt")));
    }

    #[test]
    fn malformed_rule_file_falls_back_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let bad = root.join("bad");
        writable(&root.join("ok.md"));
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join(".rummage.toml"), "unknown = 1\n").unwrap();

        let mut cache = local_cache(root, &Config::default());
        let roots = vec![root.to_path_buf()];
        let path = bad.join("ok.md");
        let rules = cache.for_file(&path);
        assert!(rules.is_indexable(&path, &roots));
        let errors = cache.take_errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains(".rummage.toml"), "{errors:?}");
    }

    #[test]
    fn local_rule_file_rejects_unknown_or_unsupported_keys() {
        assert!(toml::from_str::<LocalRuleFile>("[watch]\nroots = []\n").is_err());
        assert!(toml::from_str::<LocalRuleFile>("[index]\ndir = \".rummage\"\n").is_err());
        assert!(toml::from_str::<LocalRuleFile>("[ignore]\nfiles = []\n").is_err());

        let bad = LocalRuleFile {
            index: LocalIndexConfig {
                extensions: Some(vec!["doc".into()]),
            },
            ..LocalRuleFile::default()
        };
        assert!(ScopeRules::from_local(bad).is_err());
    }

    #[test]
    fn rule_file_is_never_indexed() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join(".rummage.toml"),
            "[index]\nextensions = [\"toml\"]\n",
        )
        .unwrap();
        let rules = ScopeRules::global(&Config::default());
        let roots = vec![root.to_path_buf()];
        let mut cache = local_cache(root, &Config::default());
        let path = root.join(".rummage.toml");
        assert!(!cache.for_file(&path).is_indexable(&path, &roots));
        assert!(!rules.is_indexable(&path, &roots));
    }
}
