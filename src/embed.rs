use anyhow::{bail, Context, Result};
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use ort::environment::Environment;
use std::path::Path;
use std::sync::Mutex;

pub const FALLBACK_DIMS: usize = 512;

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

pub struct Embedder {
    model: TextEmbedding,
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
                "模型文件缺失：{}（请把 bge-small-zh-v1.5 的 ONNX 与 tokenizer 文件放入模型目录）",
                onnx_path.display()
            );
        }
        // tokenizer 文件默认与 ONNX 同目录；`[model] tokenizer_dir` 可把两者分开。
        let tokenizer_dir = tokenizer_dir.unwrap_or(dir);
        let required = [
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ];
        for name in required {
            let path = tokenizer_dir.join(name);
            if !path.is_file() {
                bail!(
                    "模型文件缺失：{}（可用 [model] tokenizer_dir 单独指定 tokenizer 目录）",
                    path.display()
                );
            }
        }

        let onnx = std::fs::read(&onnx_path)
            .with_context(|| format!("读取模型失败：{}", onnx_path.display()))?;
        let tokenizer_files = TokenizerFiles {
            tokenizer_file: std::fs::read(tokenizer_dir.join("tokenizer.json"))?,
            config_file: std::fs::read(tokenizer_dir.join("config.json"))?,
            special_tokens_map_file: std::fs::read(tokenizer_dir.join("special_tokens_map.json"))?,
            tokenizer_config_file: std::fs::read(tokenizer_dir.join("tokenizer_config.json"))?,
        };
        let size = onnx.len() as u64;

        let mut user_model = UserDefinedEmbeddingModel::new(onnx, tokenizer_files);
        user_model.pooling = Some(Pooling::Cls);

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_div(2).clamp(1, 8))
            .unwrap_or(4);
        let options = InitOptionsUserDefined::new()
            .with_environment(env.clone())
            .with_max_length(max_length)
            .with_intra_threads(threads);
        let mut model = TextEmbedding::try_new_from_user_defined(user_model, options)
            .with_context(|| format!("初始化 ONNX 模型失败：{}", onnx_path.display()))?;

        let probe = model.embed(["模型预热"], Some(1)).context("模型预热失败")?;
        let dims = probe.first().map(|v| v.len()).unwrap_or(0);
        if dims == 0 {
            bail!("模型输出维度为 0");
        }
        let id = format!("{}:{}:{}", onnx_file, dims, size);
        Ok(Self { model, dims, id })
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
        let out = self
            .model
            .embed(texts, Some(batch_size.max(1)))
            .context("生成语义向量失败")?;
        if out.len() != texts.len() {
            bail!(
                "模型返回向量数不符：期望 {}，实际 {}",
                texts.len(),
                out.len()
            );
        }
        Ok(out)
    }

    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed(&[text.to_string()], 1)?;
        out.pop().context("模型未返回查询向量")
    }
}
