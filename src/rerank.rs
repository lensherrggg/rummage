use anyhow::{bail, Context, Result};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

pub const QUERY_PREFIX: &str = "task: search result | query: ";
pub const DOC_PREFIX: &str = "title: none | text: ";
const OUTPUT_NAME: &str = "sentence_embedding";
const MAGIC: &[u8; 8] = b"RMRNK\x00\x00\x01";
const VERSION: u32 = 1;

pub struct GemmaEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    dims: usize,
    id: String,
}

impl GemmaEmbedder {
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
                "重排模型文件缺失：{}（请把 EmbeddingGemma 2 的 ONNX 与 tokenizer.json 放入模型目录）",
                onnx_path.display()
            );
        }
        let tokenizer_path = tokenizer_dir.unwrap_or(dir).join("tokenizer.json");
        if !tokenizer_path.is_file() {
            bail!(
                "重排模型文件缺失：{}（可用 [rerank] tokenizer_dir 单独指定 tokenizer 目录）",
                tokenizer_path.display()
            );
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("读取重排 tokenizer 失败：{err}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!("设置重排分词截断失败：{err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: 0,
            pad_token: "<pad>".to_string(),
            ..Default::default()
        }));

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_div(2).clamp(1, 8))
            .unwrap_or(4);
        let mut builder = Session::builder(env)
            .context("创建重排会话失败")?
            .with_intra_threads(threads)
            .map_err(|err| anyhow::anyhow!("设置重排推理线程数失败：{err}"))?;
        let session = builder
            .commit_from_file(&onnx_path)
            .with_context(|| format!("初始化重排 ONNX 模型失败：{}", onnx_path.display()))?;
        if session.outputs().iter().all(|o| o.name() != OUTPUT_NAME) {
            bail!(
                "重排模型缺少输出 {OUTPUT_NAME}；当前只支持带句向量的 EmbeddingGemma 导出"
            );
        }

        let size = std::fs::metadata(&onnx_path)?.len()
            + std::fs::metadata(format!("{}_data", onnx_path.display()))
                .map(|m| m.len())
                .unwrap_or(0);
        let mut model = Self {
            session,
            tokenizer,
            dims: 0,
            id: String::new(),
        };
        let probe = model.embed(&["模型预热".to_string()], 1)?;
        let dims = probe.first().map(Vec::len).unwrap_or(0);
        if dims == 0 {
            bail!("重排模型输出维度为 0");
        }
        model.dims = dims;
        model.id = format!("gemma:{}:{}:{}", onnx_file, dims, size);
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
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());
        for chunk in order.chunks(batch_size.max(1)) {
            let batch: Vec<&str> = chunk.iter().map(|&i| texts[i].as_str()).collect();
            let vectors = self.embed_batch(&batch)?;
            for (&i, vector) in chunk.iter().zip(vectors) {
                out[i] = vector;
            }
        }
        Ok(out)
    }

    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed(&[text.to_string()], 1)?;
        out.pop().context("重排模型未返回查询向量")
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
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
        let mask =
            Tensor::from_array(([batch, seq], mask)).context("构造重排 attention_mask 失败")?;
        let empty_image = Tensor::from_array(([0usize, 512usize], Vec::<f32>::new()))
            .context("构造重排 image_features 失败")?;
        let empty_video = Tensor::from_array(([0usize, 512usize], Vec::<f32>::new()))
            .context("构造重排 video_features 失败")?;
        let empty_audio = Tensor::from_array(([0usize, 512usize], Vec::<f32>::new()))
            .context("构造重排 audio_features 失败")?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids,
                "attention_mask" => mask,
                "image_features" => empty_image,
                "video_features" => empty_video,
                "audio_features" => empty_audio,
            ])
            .context("重排模型推理失败")?;
        let (shape, data) = outputs[OUTPUT_NAME]
            .try_extract_tensor::<f32>()
            .context("读取重排模型输出失败")?;
        if shape.len() != 2 || shape[0] as usize != batch {
            bail!(
                "重排模型输出形状异常：{:?}，期望 [{batch}, 维度]",
                shape
            );
        }
        let dims = shape[1] as usize;
        if dims == 0 {
            bail!("重排模型输出维度为 0");
        }
        let mut vectors = Vec::with_capacity(batch);
        for row in data.chunks(dims) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            vectors.push(row.iter().map(|v| v / norm).collect());
        }
        Ok(vectors)
    }
}

/// 每个文件的块向量。只用于候选文件内部打分，不需要全库近邻搜索。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankStore {
    pub model_id: String,
    pub dims: usize,
    pub files: BTreeMap<i64, Vec<Vec<f32>>>,
}

impl RerankStore {
    pub fn new(model_id: impl Into<String>, dims: usize) -> Self {
        Self {
            model_id: model_id.into(),
            dims,
            files: BTreeMap::new(),
        }
    }

    pub fn is_complete(&self, file_id: i64, chunks: u32) -> bool {
        chunks > 0
            && self
                .files
                .get(&file_id)
                .is_some_and(|vectors| vectors.len() == chunks as usize)
    }

    pub fn set(&mut self, file_id: i64, vectors: Vec<Vec<f32>>) {
        self.files.insert(file_id, vectors);
    }

    pub fn remove(&mut self, file_id: i64) {
        self.files.remove(&file_id);
    }

    pub fn retain_files(&mut self, mut keep: impl FnMut(i64) -> bool) {
        self.files.retain(|id, _| keep(*id));
    }

    pub fn max_similarity(&self, file_id: i64, chunks: u32, query: &[f32]) -> Option<f64> {
        if !self.is_complete(file_id, chunks) {
            return None;
        }
        let vectors = self.files.get(&file_id)?;
        let mut best = f64::NEG_INFINITY;
        for vector in vectors {
            if vector.len() != query.len() {
                return None;
            }
            let dot: f64 = vector
                .iter()
                .zip(query)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            best = best.max(dot);
        }
        (best > f64::NEG_INFINITY).then_some(best)
    }
}

/// 只重排基础排序的前 `top` 个文件；返回新顺序，`None` 表示保持原样。
pub fn blended_top(
    top: &[(i64, f64)],
    gemma: &HashMap<i64, f64>,
    weight: f64,
) -> Option<Vec<i64>> {
    if top.len() < 2 {
        return None;
    }
    let mut gvals = Vec::with_capacity(top.len());
    for (id, _) in top {
        gvals.push(*gemma.get(id)?);
    }
    let (gmin, gmax) = gvals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if gmax - gmin <= 1e-12 {
        return None;
    }
    let bmin = top.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
    let bmax = top
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let brange = bmax - bmin;
    let grange = gmax - gmin;

    let mut order: Vec<usize> = (0..top.len()).collect();
    let blended: Vec<f64> = top
        .iter()
        .zip(&gvals)
        .map(|((_, base), gemma)| {
            let bn = if brange > 1e-12 {
                (base - bmin) / brange
            } else {
                0.0
            };
            let gn = (gemma - gmin) / grange;
            (1.0 - weight) * bn + weight * gn
        })
        .collect();
    order.sort_by(|&a, &b| blended[b].total_cmp(&blended[a]));
    Some(order.into_iter().map(|i| top[i].0).collect())
}

pub fn save(path: &Path, store: &RerankStore) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp.{}",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "rerank.bin".to_string()),
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let file =
            File::create(&tmp).with_context(|| format!("创建重排索引失败：{}", tmp.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        bincode::serialize_into(&mut writer, &VERSION)?;
        bincode::serialize_into(&mut writer, store)?;
        writer.flush()?;
        let file = writer.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("替换重排索引失败：{}", path.display()));
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<Option<RerankStore>> {
    if !path.is_file() {
        return Ok(None);
    }
    let file = File::open(path).with_context(|| format!("打开重排索引失败：{}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    if reader.read_exact(&mut magic).is_err() {
        return Ok(None);
    }
    if &magic != MAGIC {
        return Ok(None);
    }
    let version: u32 = bincode::deserialize_from(&mut reader).context("重排索引版本读取失败")?;
    if version != VERSION {
        return Ok(None);
    }
    let store: RerankStore =
        bincode::deserialize_from(&mut reader).context("重排索引内容损坏")?;
    Ok(Some(store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn scores(items: &[(i64, f64)]) -> HashMap<i64, f64> {
        items.iter().copied().collect()
    }

    #[test]
    fn blend_reorders_only_the_top_group() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0), (4, 7.0), (5, 6.0)];
        let gemma = scores(&[(1, 0.1), (2, 0.9), (3, 0.2), (4, 0.0), (5, 1.0)]);
        let top = &base[..3];
        let order = blended_top(top, &gemma, 0.5).unwrap();
        assert_eq!(order, vec![2, 1, 3]);
        let rest: Vec<i64> = base[3..].iter().map(|(id, _)| *id).collect();
        assert_eq!(rest, vec![4, 5]);
    }

    #[test]
    fn missing_or_flat_scores_leave_order_untouched() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0)];
        assert!(blended_top(&base, &scores(&[(1, 0.1), (2, 0.2)]), 0.5).is_none());
        assert!(blended_top(
            &base,
            &scores(&[(1, 0.5), (2, 0.5), (3, 0.5)]),
            0.5
        )
        .is_none());
        assert!(blended_top(&base[..1], &scores(&[(1, 0.5)]), 0.5).is_none());
    }

    #[test]
    fn blend_is_stable_on_ties() {
        let base = [(1, 10.0), (2, 9.0), (3, 8.0)];
        // 文件 1 与 3 的混合分同为 0.5，保持基础顺序。
        let gemma = scores(&[(1, 0.1), (2, 0.9), (3, 0.9)]);
        assert_eq!(
            blended_top(&base, &gemma, 0.5).unwrap(),
            vec![2, 1, 3]
        );
    }

    #[test]
    fn store_roundtrip_and_completeness() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rerank.bin");
        let mut store = RerankStore::new("gemma:model:2:1", 2);
        store.set(7, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(store.is_complete(7, 2));
        assert!(!store.is_complete(7, 3));
        assert_eq!(
            store.max_similarity(7, 2, &[1.0, 0.0]),
            Some(1.0)
        );
        save(&path, &store).unwrap();
        let loaded = load(&path).unwrap().unwrap();
        assert_eq!(loaded.model_id, "gemma:model:2:1");
        assert_eq!(loaded.dims, 2);
        assert_eq!(loaded.files[&7].len(), 2);
        assert!(load(&dir.path().join("missing.bin")).unwrap().is_none());
        std::fs::write(&path, b"broken").unwrap();
        assert!(load(&path).unwrap().is_none());
    }
}
