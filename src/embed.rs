use anyhow::{bail, Context, Result};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// ONNX Runtime 只允许进程内存在一个环境；主模型和重排模型必须共用它。
pub fn environment() -> Result<Environment> {
    static ENV: Mutex<Option<Environment>> = Mutex::new(None);
    let mut slot = ENV.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(env) = slot.as_ref() {
        return Ok(env.clone());
    }
    let env = ort::init()
        .with_name("rummage")
        .build()
        .map_err(|err| anyhow::anyhow!("初始化 ONNX 运行环境失败：{err}"))?;
    *slot = Some(env.clone());
    Ok(env)
}

/// bge 中文句向量模型：CLS 池化、L2 归一化。
pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    output: String,
    dims: usize,
    id: String,
}

impl Embedder {
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
                "模型文件缺失：{}（请把 bge 模型的 ONNX 与 tokenizer 文件放入模型目录）",
                onnx_path.display()
            );
        }
        let tokenizer_dir = tokenizer_dir.unwrap_or(dir);
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let path = tokenizer_dir.join(name);
            if !path.is_file() {
                bail!(
                    "模型文件缺失：{}（可用 [model] tokenizer_dir 单独指定 tokenizer 目录）",
                    path.display()
                );
            }
        }

        let tokenizer_config: Value =
            serde_json::from_slice(&std::fs::read(tokenizer_dir.join("tokenizer_config.json"))?)
                .context("解析 tokenizer_config.json 失败")?;
        let model_max_length = tokenizer_config["model_max_length"]
            .as_f64()
            .map(|v| v as usize)
            .unwrap_or(max_length);
        let max_length = max_length.min(model_max_length).max(1);
        let pad_token = pad_token(&tokenizer_config).unwrap_or_else(|| "[PAD]".to_string());

        let config: Value = std::fs::read(tokenizer_dir.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(Value::Null);
        let mut tokenizer = Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))
            .map_err(|err| anyhow::anyhow!("读取 tokenizer.json 失败：{err}"))?;
        let pad_id = config["pad_token_id"]
            .as_u64()
            .map(|id| id as u32)
            .or_else(|| tokenizer.token_to_id(&pad_token))
            .unwrap_or(0);
        let pad_token = tokenizer.id_to_token(pad_id).unwrap_or(pad_token);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!("设置分词截断失败：{err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }));

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_div(2).clamp(1, 8))
            .unwrap_or(4);
        let session = Session::builder(env)
            .context("创建 ONNX 会话失败")?
            .with_intra_threads(threads)
            .map_err(|err| anyhow::anyhow!("设置推理线程数失败：{err}"))?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("初始化 ONNX 模型失败：{}", onnx_path.display()))?;
        let output = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("模型没有输出")?;

        let size = std::fs::metadata(&onnx_path)?.len()
            + std::fs::metadata(format!("{}_data", onnx_path.display()))
                .map(|m| m.len())
                .unwrap_or(0);

        let mut model = Self {
            session,
            tokenizer,
            output,
            dims: 0,
            id: String::new(),
        };
        let probe = model.embed(&["模型预热".to_string()], 1)?;
        let dims = probe.first().map(Vec::len).unwrap_or(0);
        if dims == 0 {
            bail!("模型输出维度为 0");
        }
        model.dims = dims;
        model.id = format!("{onnx_file}:{dims}:{size}");
        Ok(model)
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn embed(&mut self, texts: &[String], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = batch_size.max(1);
        // 同长度相近的块放一批，减少 padding 带来的无效计算。
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());
        let mut out = vec![Vec::new(); texts.len()];
        for batch in order.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|&i| texts[i].clone()).collect();
            let vectors = self.embed_batch(&texts)?;
            for (&index, vector) in batch.iter().zip(vectors) {
                out[index] = vector;
            }
        }
        Ok(out)
    }

    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed(&[text.to_string()], 1)?;
        out.pop().context("模型未返回查询向量")
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|err| anyhow::anyhow!("分词失败：{err}"))?;
        let batch = encodings.len();
        let seq = encodings
            .first()
            .map(|e| e.get_ids().len())
            .context("分词结果为空")?;

        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for encoding in &encodings {
            ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
            mask.extend(encoding.get_attention_mask().iter().map(|&m| m as i64));
        }

        let ids = Tensor::from_array(([batch, seq], ids)).context("构造 input_ids 失败")?;
        let mask_t =
            Tensor::from_array(([batch, seq], mask)).context("构造 attention_mask 失败")?;
        let need_type_ids = self
            .session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        let outputs = if need_type_ids {
            let type_ids = Tensor::from_array(([batch, seq], vec![0_i64; batch * seq]))
                .context("构造 token_type_ids 失败")?;
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
            .with_context(|| format!("模型输出缺少 {}", self.output))?;
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .with_context(|| format!("读取模型输出 {} 失败", self.output))?;
        if shape.len() != 3 || shape[0] as usize != batch {
            bail!("模型输出形状异常：{shape:?}，期望 [{batch}, seq, 维度]");
        }
        let dims = shape[2] as usize;
        if dims == 0 {
            bail!("模型输出维度为 0");
        }

        let mut vectors = Vec::with_capacity(batch);
        for b in 0..batch {
            let start = b * seq * dims;
            let mut vector = data[start..start + dims].to_vec();
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in &mut vector {
                *value /= norm;
            }
            vectors.push(vector);
        }
        Ok(vectors)
    }
}

fn pad_token(config: &Value) -> Option<String> {
    let token = &config["pad_token"];
    token
        .as_str()
        .or_else(|| token["content"].as_str())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_token_reads_string_and_object_forms() {
        let plain: Value = serde_json::from_str(r#"{"pad_token":"[PAD]"}"#).unwrap();
        assert_eq!(pad_token(&plain).as_deref(), Some("[PAD]"));
        let object: Value = serde_json::from_str(r#"{"pad_token":{"content":"<pad>"}}"#).unwrap();
        assert_eq!(pad_token(&object).as_deref(), Some("<pad>"));
        let missing: Value = serde_json::from_str("{}").unwrap();
        assert!(pad_token(&missing).is_none());
    }
}
