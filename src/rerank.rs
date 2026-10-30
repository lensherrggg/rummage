use anyhow::{bail, Context, Result};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// 交叉编码器：把 `(查询, 文档块)` 一起编码，输出一个相关性分数。
pub struct CrossEncoder {
    session: Session,
    tokenizer: Tokenizer,
    output: String,
    id: String,
}

impl CrossEncoder {
    pub fn load(
        env: &Environment,
        dir: &Path,
        onnx_file: &str,
        tokenizer_dir: Option<&Path>,
        max_length: usize,
    ) -> Result<Self> {
        let onnx_path = dir.join(onnx_file);
        if !onnx_path.is_file() {
            bail!(
                "重排模型文件缺失：{}（请把 bge-reranker 的 ONNX 与 tokenizer 文件放入模型目录）",
                onnx_path.display()
            );
        }
        let tokenizer_dir = tokenizer_dir.unwrap_or(dir);
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            bail!(
                "重排模型文件缺失：{}（可用 [rerank] tokenizer_dir 单独指定 tokenizer 目录）",
                tokenizer_path.display()
            );
        }

        let tokenizer_config: Value = std::fs::read(tokenizer_dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(Value::Null);
        let pad_token = tokenizer_config["pad_token"]
            .as_str()
            .or_else(|| tokenizer_config["pad_token"]["content"].as_str())
            .unwrap_or("<pad>")
            .to_string();

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("读取重排 tokenizer 失败：{err}"))?;
        let config: Value = std::fs::read(tokenizer_dir.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(Value::Null);
        let pad_id = config["pad_token_id"]
            .as_u64()
            .map(|id| id as u32)
            .or_else(|| tokenizer.token_to_id(&pad_token))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);
        let pad_token = tokenizer.id_to_token(pad_id).unwrap_or(pad_token);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_length.max(1),
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!("设置重排分词截断失败：{err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id,
            pad_token,
            ..Default::default()
        }));

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_div(2).clamp(1, 8))
            .unwrap_or(4);
        let session = Session::builder(env)
            .context("创建重排会话失败")?
            .with_intra_threads(threads)
            .map_err(|err| anyhow::anyhow!("设置重排推理线程数失败：{err}"))?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("初始化重排 ONNX 模型失败：{}", onnx_path.display()))?;

        let output = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("重排模型没有输出")?;
        let size = std::fs::metadata(&onnx_path)?.len()
            + std::fs::metadata(format!("{}_data", onnx_path.display()))
                .map(|m| m.len())
                .unwrap_or(0);
        let mut model = Self {
            session,
            tokenizer,
            output,
            id: format!("ce:{onnx_file}:{size}"),
        };
        model.score("预热", &["模型预热".to_string()], 1)?;
        Ok(model)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// 每个文档块与查询组成一对，返回同序的相关性分数（越大越相关）。
    pub fn score(
        &mut self,
        query: &str,
        passages: &[String],
        batch_size: usize,
    ) -> Result<Vec<f64>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = batch_size.max(1);
        let mut out = Vec::with_capacity(passages.len());
        for batch in passages.chunks(batch_size) {
            let pairs: Vec<(String, String)> = batch
                .iter()
                .map(|passage| (query.to_string(), passage.clone()))
                .collect();
            out.extend(self.score_batch(&pairs)?);
        }
        Ok(out)
    }

    fn score_batch(&mut self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|err| anyhow::anyhow!("重排分词失败：{err}"))?;
        let batch = encodings.len();
        let seq = encodings
            .first()
            .map(|e| e.get_ids().len())
            .context("重排分词结果为空")?;
        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for encoding in &encodings {
            ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
            mask.extend(encoding.get_attention_mask().iter().map(|&m| m as i64));
        }

        let ids = Tensor::from_array(([batch, seq], ids)).context("构造重排 input_ids 失败")?;
        let mask_t = Tensor::from_array(([batch, seq], mask.clone()))
            .context("构造重排 attention_mask 失败")?;
        let need_type_ids = self
            .session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        let outputs = if need_type_ids {
            let type_ids = Tensor::from_array(([batch, seq], vec![0_i64; batch * seq]))
                .context("构造重排 token_type_ids 失败")?;
            self.session.run(ort::inputs![
                "input_ids" => ids,
                "attention_mask" => mask_t,
                "token_type_ids" => type_ids,
            ])?
        } else {
            self.session.run(ort::inputs![
                "input_ids" => ids,
                "attention_mask" => mask_t,
            ])?
        };

        let value = outputs
            .get(self.output.as_str())
            .with_context(|| format!("重排模型输出缺少 {}", self.output))?;
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .with_context(|| format!("读取重排模型输出 {} 失败", self.output))?;
        let total: usize = shape.iter().map(|d| *d as usize).product();
        if total != batch {
            bail!("重排模型输出形状异常：{shape:?}，期望 [{batch}, 1]");
        }
        Ok(data.iter().map(|v| f64::from(*v)).collect())
    }
}

/// 只重排基础排序的前 `top` 个文件；返回新顺序，`None` 表示保持原样。
pub fn blended_top(
    top: &[(i64, f64)],
    rerank: &HashMap<i64, f64>,
    weight: f64,
) -> Option<Vec<i64>> {
    if top.len() < 2 {
        return None;
    }
    let mut rvals = Vec::with_capacity(top.len());
    for (id, _) in top {
        rvals.push(*rerank.get(id)?);
    }
    let (rmin, rmax) = rvals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if rmax - rmin <= 1e-12 {
        return None;
    }
    let bmin = top.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
    let bmax = top
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let brange = bmax - bmin;
    let rrange = rmax - rmin;

    let mut order: Vec<usize> = (0..top.len()).collect();
    let blended: Vec<f64> = top
        .iter()
        .zip(&rvals)
        .map(|((_, base), rerank)| {
            let bn = if brange > 1e-12 {
                (base - bmin) / brange
            } else {
                0.0
            };
            let rn = (rerank - rmin) / rrange;
            (1.0 - weight) * bn + weight * rn
        })
        .collect();
    order.sort_by(|&a, &b| blended[b].total_cmp(&blended[a]));
    Some(order.into_iter().map(|i| top[i].0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(items: &[(i64, f64)]) -> HashMap<i64, f64> {
        items.iter().copied().collect()
    }

    #[test]
    fn blend_reorders_only_the_top_group() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0), (4, 7.0), (5, 6.0)];
        let rerank = scores(&[(1, 0.1), (2, 0.9), (3, 0.2), (4, 0.0), (5, 1.0)]);
        let top = &base[..3];
        let order = blended_top(top, &rerank, 0.5).unwrap();
        assert_eq!(order, vec![2, 1, 3]);
        let rest: Vec<i64> = base[3..].iter().map(|(id, _)| *id).collect();
        assert_eq!(rest, vec![4, 5]);
    }

    #[test]
    fn missing_or_flat_scores_leave_order_untouched() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0)];
        assert!(blended_top(&base, &scores(&[(1, 0.1), (2, 0.2)]), 0.5).is_none());
        assert!(blended_top(&base, &scores(&[(1, 0.5), (2, 0.5), (3, 0.5)]), 0.5).is_none());
        assert!(blended_top(&base[..1], &scores(&[(1, 0.5)]), 0.5).is_none());
    }

    #[test]
    fn blend_is_stable_on_ties() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0)];
        // 文件 1 与 3 的混合分同为 0.5，保持基础顺序。
        let rerank = scores(&[(1, 0.1), (2, 0.9), (3, 0.9)]);
        assert_eq!(blended_top(&base, &rerank, 0.5).unwrap(), vec![2, 1, 3]);
    }

    #[test]
    fn full_weight_ignores_base_order() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0)];
        let rerank = scores(&[(1, 0.0), (2, 0.5), (3, 1.0)]);
        assert_eq!(blended_top(&base, &rerank, 1.0).unwrap(), vec![3, 2, 1]);
    }
}
