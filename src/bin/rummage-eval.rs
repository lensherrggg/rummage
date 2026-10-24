use anyhow::{bail, Context, Result};
use clap::Parser;
use rummage::config::Config;
use rummage::engine::{IndexEngine, SearchMode};
use rummage::eval::{evaluate_query, fusion_tax, report, LabeledQuery, ModeReport};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "rummage-eval",
    version,
    about = "Rummage 离线检索质量评估：BM25 / 语义 / RRF 三路 Recall、MRR、nDCG"
)]
struct Cli {
    /// 已建好的索引使用的 rummage.toml；省略则读当前目录
    #[arg(long = "config", value_name = "FILE")]
    config: Option<PathBuf>,

    /// JSONL 标注查询，每行一个对象
    #[arg(long = "queries", value_name = "FILE")]
    queries: PathBuf,

    /// 截断到前 K 条结果
    #[arg(long, default_value_t = 10, value_name = "K")]
    top: usize,

    /// 逗号分隔：hybrid,lexical,semantic；默认三路都跑
    #[arg(long, default_value = "hybrid,lexical,semantic", value_name = "LIST")]
    modes: String,

    /// 覆盖 [search] rrf_k（只影响 hybrid）
    #[arg(long, value_name = "FLOAT")]
    rrf_k: Option<f64>,

    /// 覆盖关键词一路 RRF 权重
    #[arg(long, value_name = "FLOAT")]
    lexical_weight: Option<f64>,

    /// 覆盖语义一路 RRF 权重
    #[arg(long, value_name = "FLOAT")]
    semantic_weight: Option<f64>,

    /// 启用 [rerank]（默认关闭，避免单路基线被重排干扰）
    #[arg(long)]
    rerank: bool,

    /// 把逐查询结果和汇总写入 JSON
    #[arg(long = "out", value_name = "FILE")]
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.top == 0 {
        bail!("--top 必须大于 0");
    }

    let loaded = Config::load(cli.config.as_deref())?;
    let mut cfg = loaded.config;
    let roots = cfg.resolve_roots(&[], &loaded.base)?;
    let index_dir = cfg.resolve_index_dir(&roots, &loaded.base);
    let model_dir = cfg.resolve_model_dir(&loaded.base);
    let rerank_dir = cfg.resolve_rerank_model_dir(&loaded.base);

    cfg.search.results = cli.top;
    cfg.search.candidates = cfg.search.candidates.max(cli.top);
    if let Some(k) = cli.rrf_k {
        cfg.search.rrf_k = k;
    }
    if let Some(w) = cli.lexical_weight {
        cfg.search.lexical_weight = w;
    }
    if let Some(w) = cli.semantic_weight {
        cfg.search.semantic_weight = w;
    }
    if !cfg.search.rrf_k.is_finite() || cfg.search.rrf_k <= 0.0 {
        bail!("--rrf-k 必须是大于 0 的有限数");
    }
    if !cfg.search.lexical_weight.is_finite() || cfg.search.lexical_weight < 0.0 {
        bail!("--lexical-weight 必须是非负有限数");
    }
    if !cfg.search.semantic_weight.is_finite() || cfg.search.semantic_weight < 0.0 {
        bail!("--semantic-weight 必须是非负有限数");
    }
    if cfg.search.lexical_weight == 0.0 && cfg.search.semantic_weight == 0.0 {
        bail!("两路权重不能同时为 0");
    }
    if !cli.rerank {
        cfg.rerank.enabled = false;
    }

    let queries = load_queries(&cli.queries)?;
    if queries.is_empty() {
        bail!("查询文件为空：{}", cli.queries.display());
    }

    let modes = parse_modes(&cli.modes)?;
    let mut engine = IndexEngine::new(cfg, roots, index_dir, model_dir, rerank_dir)
        .context("打开索引失败；请先用 rummage 建好索引")?;
    let indexed = engine.status(false).indexed_files;
    eprintln!("索引文件数：{indexed}；查询数：{}；top K：{}", queries.len(), cli.top);

    let mut reports = Vec::new();
    for mode in modes {
        let mut outcomes = Vec::with_capacity(queries.len());
        for query in &queries {
            let parsed = rummage::query::parse(&query.query);
            let hits = engine
                .search_mode(&parsed, mode)
                .with_context(|| format!("查询失败：{}", query.id))?;
            let paths: Vec<String> = hits
                .iter()
                .map(|hit| hit.path.to_string_lossy().into_owned())
                .collect();
            outcomes.push(evaluate_query(query, &paths, cli.top));
        }
        reports.push(report(mode_name(mode), outcomes));
    }

    print_summary(&reports);
    let tax = fusion_tax(&reports);
    if let Some(tax) = &tax {
        println!(
            "\n融合税：{}/{} 条（{:.1}%），至少一路命中的查询才计入",
            tax.tax_queries,
            tax.comparable_queries,
            tax.tax_rate * 100.0
        );
    }
    if let Some(path) = &cli.out {
        let payload = serde_json::json!({
            "top_k": cli.top,
            "query_file": cli.queries.display().to_string(),
            "indexed_files": indexed,
            "reports": reports,
            "fusion_tax": tax,
        });
        fs::write(path, serde_json::to_string_pretty(&payload)?)
            .with_context(|| format!("写入评估结果失败：{}", path.display()))?;
        println!("JSON 已写入：{}", path.display());
    }
    Ok(())
}

fn load_queries(path: &std::path::Path) -> Result<Vec<LabeledQuery>> {
    let file = fs::File::open(path)
        .with_context(|| format!("打开查询文件失败：{}", path.display()))?;
    let mut out = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let query: LabeledQuery = serde_json::from_str(line)
            .with_context(|| format!("查询文件第 {} 行 JSON 无效", line_no + 1))?;
        out.push(query);
    }
    Ok(out)
}

fn parse_modes(value: &str) -> Result<Vec<SearchMode>> {
    let mut modes = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mode = SearchMode::parse(part)
            .with_context(|| format!("未知检索模式：{part}（可选 hybrid/lexical/semantic）"))?;
        if !modes.contains(&mode) {
            modes.push(mode);
        }
    }
    if modes.is_empty() {
        bail!("--modes 不能为空");
    }
    Ok(modes)
}

fn mode_name(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Hybrid => "hybrid",
        SearchMode::Lexical => "lexical",
        SearchMode::Semantic => "semantic",
    }
}

fn print_summary(reports: &[ModeReport]) {
    println!(
        "{:<10} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "mode", "judged", "Recall", "MRR", "nDCG", "Success"
    );
    for report in reports {
        let s = &report.summary;
        println!(
            "{:<10} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            report.mode, s.judged_queries, s.recall_at_k, s.mrr_at_k, s.ndcg_at_k, s.success_at_k
        );
    }
    for report in reports {
        if report.by_type.len() <= 1 {
            continue;
        }
        println!("\n[{}] by type", report.mode);
        for (kind, s) in &report.by_type {
            println!(
                "  {:<12} n={:<3} Recall={:.3} MRR={:.3} nDCG={:.3}",
                kind, s.judged_queries, s.recall_at_k, s.mrr_at_k, s.ndcg_at_k
            );
        }
    }
}
