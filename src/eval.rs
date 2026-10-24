//! Offline retrieval-quality evaluation primitives.
//!
//! The evaluation harness is deliberately separate from the TUI: it consumes
//! JSONL queries with file-level relevance labels and reports Recall@K, MRR@K,
//! nDCG@K and Success@K for each search mode.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// One JSONL line in the labelled query set.
///
/// `grades` is optional; when present it takes precedence over `relevant` and
/// allows graded nDCG (for example 2 = highly relevant, 1 = partly relevant).
#[derive(Debug, Clone, Deserialize)]
pub struct LabeledQuery {
    pub id: String,
    pub query: String,
    #[serde(rename = "type", default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub relevant: Vec<String>,
    #[serde(default)]
    pub grades: HashMap<String, u8>,
}

fn default_kind() -> String {
    "default".to_string()
}

impl LabeledQuery {
    pub fn grade_map(&self) -> HashMap<String, u8> {
        let mut grades = HashMap::new();
        for path in &self.relevant {
            grades.insert(normalize_path(path), 1);
        }
        for (path, grade) in &self.grades {
            grades.insert(normalize_path(path), *grade);
        }
        grades
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TopHit {
    pub rank: usize,
    pub path: String,
    pub relevant: bool,
    pub grade: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryOutcome {
    pub id: String,
    pub query: String,
    pub kind: String,
    /// `None` when the query has no labels (excluded from aggregate metrics).
    pub first_relevant_rank: Option<usize>,
    pub recall_at_k: f64,
    pub mrr_at_k: f64,
    pub ndcg_at_k: f64,
    pub success_at_k: f64,
    pub judged: bool,
    pub top: Vec<TopHit>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Aggregate {
    pub queries: usize,
    pub judged_queries: usize,
    pub recall_at_k: f64,
    pub mrr_at_k: f64,
    pub ndcg_at_k: f64,
    pub success_at_k: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModeReport {
    pub mode: String,
    pub summary: Aggregate,
    pub by_type: BTreeMap<String, Aggregate>,
    pub queries: Vec<QueryOutcome>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FusionTaxExample {
    pub id: String,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub hybrid_rank: Option<usize>,
    pub best_leg: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FusionTax {
    /// Queries where at least one of the two single legs has a relevant hit.
    pub comparable_queries: usize,
    /// Hybrid's first relevant hit is worse than the better single leg.
    pub tax_queries: usize,
    pub tax_rate: f64,
    pub examples: Vec<FusionTaxExample>,
}

/// Normalize a path for label matching: slash direction and Windows case do not
/// matter. This is intentionally independent of the engine's path helpers so
/// externally-authored labels are easy to write.
pub fn normalize_path(path: &str) -> String {
    path.trim()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_lowercase()
}

pub fn evaluate_query(
    query: &LabeledQuery,
    result_paths: &[String],
    k: usize,
) -> QueryOutcome {
    let grades = query.grade_map();
    let relevant: HashSet<String> = grades.keys().cloned().collect();
    let judged = !relevant.is_empty();
    let k = k.max(1);

    let mut first_relevant_rank = None;
    let mut hits = 0usize;
    let mut dcg = 0.0f64;
    let mut top = Vec::new();

    for (idx, path) in result_paths.iter().take(k).enumerate() {
        let key = normalize_path(path);
        let grade = grades.get(&key).copied().unwrap_or(0);
        let is_relevant = grade > 0;
        if is_relevant {
            hits += 1;
            if first_relevant_rank.is_none() {
                first_relevant_rank = Some(idx + 1);
            }
            dcg += (2f64.powi(i32::from(grade)) - 1.0) / ((idx + 2) as f64).log2();
        }
        top.push(TopHit {
            rank: idx + 1,
            path: path.clone(),
            relevant: is_relevant,
            grade,
        });
    }

    let recall = if judged {
        hits as f64 / relevant.len() as f64
    } else {
        0.0
    };
    let mrr = first_relevant_rank
        .map(|rank| 1.0 / rank as f64)
        .unwrap_or(0.0);
    let success = if first_relevant_rank.is_some() { 1.0 } else { 0.0 };
    let ndcg = if judged {
        let mut ideal: Vec<u8> = grades.values().copied().filter(|g| *g > 0).collect();
        ideal.sort_unstable_by(|a, b| b.cmp(a));
        let idcg: f64 = ideal
            .iter()
            .take(k)
            .enumerate()
            .map(|(idx, grade)| {
                (2f64.powi(i32::from(*grade)) - 1.0) / ((idx + 2) as f64).log2()
            })
            .sum();
        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    } else {
        0.0
    };

    QueryOutcome {
        id: query.id.clone(),
        query: query.query.clone(),
        kind: query.kind.clone(),
        first_relevant_rank,
        recall_at_k: recall,
        mrr_at_k: mrr,
        ndcg_at_k: ndcg,
        success_at_k: success,
        judged,
        top,
    }
}

pub fn aggregate(outcomes: &[QueryOutcome]) -> Aggregate {
    let judged: Vec<&QueryOutcome> = outcomes.iter().filter(|o| o.judged).collect();
    let n = judged.len();
    if n == 0 {
        return Aggregate {
            queries: outcomes.len(),
            ..Default::default()
        };
    }
    let mean = |f: fn(&QueryOutcome) -> f64| judged.iter().map(|o| f(o)).sum::<f64>() / n as f64;
    Aggregate {
        queries: outcomes.len(),
        judged_queries: n,
        recall_at_k: mean(|o| o.recall_at_k),
        mrr_at_k: mean(|o| o.mrr_at_k),
        ndcg_at_k: mean(|o| o.ndcg_at_k),
        success_at_k: mean(|o| o.success_at_k),
    }
}

/// Compare hybrid's first-relevant rank with the better single leg.
/// Queries where neither single leg found a relevant file are excluded.
pub fn fusion_tax(reports: &[ModeReport]) -> Option<FusionTax> {
    let lexical = reports.iter().find(|r| r.mode == "lexical")?;
    let semantic = reports.iter().find(|r| r.mode == "semantic")?;
    let hybrid = reports.iter().find(|r| r.mode == "hybrid")?;
    let hy: HashMap<&str, Option<usize>> = hybrid
        .queries
        .iter()
        .map(|q| (q.id.as_str(), q.first_relevant_rank))
        .collect();
    let mut comparable = 0usize;
    let mut taxed = 0usize;
    let mut examples = Vec::new();
    for (lex, sem) in lexical.queries.iter().zip(semantic.queries.iter()) {
        if lex.id != sem.id {
            continue;
        }
        let Some(hybrid_rank) = hy.get(lex.id.as_str()).copied().flatten() else {
            // Hybrid has no relevant hit; still counts if a single leg did.
            if lex.first_relevant_rank.is_none() && sem.first_relevant_rank.is_none() {
                continue;
            }
            comparable += 1;
            taxed += 1;
            if examples.len() < 20 {
                examples.push(FusionTaxExample {
                    id: lex.id.clone(),
                    lexical_rank: lex.first_relevant_rank,
                    semantic_rank: sem.first_relevant_rank,
                    hybrid_rank: None,
                    best_leg: lex
                        .first_relevant_rank
                        .into_iter()
                        .chain(sem.first_relevant_rank)
                        .min()
                        .unwrap_or(0),
                });
            }
            continue;
        };
        let Some(best_leg) = lex
            .first_relevant_rank
            .into_iter()
            .chain(sem.first_relevant_rank)
            .min()
        else {
            continue;
        };
        comparable += 1;
        if hybrid_rank > best_leg {
            taxed += 1;
            if examples.len() < 20 {
                examples.push(FusionTaxExample {
                    id: lex.id.clone(),
                    lexical_rank: lex.first_relevant_rank,
                    semantic_rank: sem.first_relevant_rank,
                    hybrid_rank: Some(hybrid_rank),
                    best_leg,
                });
            }
        }
    }
    Some(FusionTax {
        comparable_queries: comparable,
        tax_queries: taxed,
        tax_rate: if comparable == 0 {
            0.0
        } else {
            taxed as f64 / comparable as f64
        },
        examples,
    })
}

pub fn report(mode: &str, outcomes: Vec<QueryOutcome>) -> ModeReport {
    let summary = aggregate(&outcomes);
    let mut by_type: BTreeMap<String, Vec<QueryOutcome>> = BTreeMap::new();
    for outcome in &outcomes {
        by_type
            .entry(outcome.kind.clone())
            .or_default()
            .push(outcome.clone());
    }
    let by_type = by_type
        .into_iter()
        .map(|(kind, rows)| (kind, aggregate(&rows)))
        .collect();
    ModeReport {
        mode: mode.to_string(),
        summary,
        by_type,
        queries: outcomes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> LabeledQuery {
        LabeledQuery {
            id: "q1".into(),
            query: "测试".into(),
            kind: "keyword".into(),
            relevant: vec![r"D:\docs\a.md".into(), r"D:\docs\b.md".into()],
            grades: HashMap::new(),
        }
    }

    #[test]
    fn computes_recall_mrr_and_ndcg() {
        let out = evaluate_query(
            &query(),
            &[
                r"d:/docs/x.md".into(),
                r"D:\docs\B.MD".into(),
                r"D:\docs\a.md".into(),
            ],
            10,
        );
        assert_eq!(out.first_relevant_rank, Some(2));
        assert!((out.mrr_at_k - 0.5).abs() < 1e-9);
        assert!((out.recall_at_k - 1.0).abs() < 1e-9);
        assert!(out.ndcg_at_k > 0.0 && out.ndcg_at_k < 1.0);
        assert_eq!(out.success_at_k, 1.0);
    }

    #[test]
    fn graded_ndcg_prefers_high_grade_first() {
        let mut q = query();
        q.relevant.clear();
        q.grades.insert(r"D:\docs\a.md".into(), 2);
        q.grades.insert(r"D:\docs\b.md".into(), 1);
        let good = evaluate_query(&q, &[r"D:\docs\a.md".into(), r"D:\docs\b.md".into()], 10);
        let bad = evaluate_query(&q, &[r"D:\docs\b.md".into(), r"D:\docs\a.md".into()], 10);
        assert!(good.ndcg_at_k > bad.ndcg_at_k);
    }

    #[test]
    fn unlabelled_queries_are_excluded_from_aggregate() {
        let mut q = query();
        q.relevant.clear();
        let out = evaluate_query(&q, &[], 10);
        assert!(!out.judged);
        let agg = aggregate(&[out]);
        assert_eq!(agg.judged_queries, 0);
    }
}
