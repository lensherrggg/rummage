use anyhow::{Context, Result};
use scour::HybridIndex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"RMIDX\x00\x00\x01";
const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    pub model_id: String,
    pub dims: usize,
    pub hybrid: HybridIndex,
}

pub fn save(path: &Path, hybrid: &HybridIndex, model_id: &str, dims: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp.{}",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "index.bin".to_string()),
        std::process::id()
    ));

    let result = (|| -> Result<()> {
        let file = File::create(&tmp)
            .with_context(|| format!("创建索引临时文件失败：{}", tmp.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        bincode::serialize_into(&mut writer, &VERSION)?;
        bincode::serialize_into(&mut writer, model_id)?;
        bincode::serialize_into(&mut writer, &dims)?;
        bincode::serialize_into(&mut writer, hybrid)?;
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
        return Err(err).with_context(|| format!("替换索引文件失败：{}", path.display()));
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<Option<Snapshot>> {
    if !path.is_file() {
        return Ok(None);
    }
    let file = File::open(path).with_context(|| format!("打开索引失败：{}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    if reader.read_exact(&mut magic).is_err() {
        return Ok(None);
    }
    if &magic != MAGIC {
        return Ok(None);
    }
    let version: u32 = bincode::deserialize_from(&mut reader).context("索引版本读取失败")?;
    if version != VERSION {
        return Ok(None);
    }
    let model_id: String =
        bincode::deserialize_from(&mut reader).context("索引模型标识读取失败")?;
    let dims: usize = bincode::deserialize_from(&mut reader).context("索引维度读取失败")?;
    let hybrid: HybridIndex = bincode::deserialize_from(&mut reader).context("索引内容损坏")?;
    Ok(Some(Snapshot {
        version,
        model_id,
        dims,
        hybrid,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn missing_file_is_none() {
        let dir = tempdir().unwrap();
        assert!(load(&dir.path().join("index.bin")).unwrap().is_none());
    }

    #[test]
    fn broken_file_does_not_parse() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.bin");
        std::fs::write(&path, b"not an index").unwrap();
        assert!(load(&path).unwrap().is_none());
    }

    #[test]
    fn snapshot_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.bin");
        let mut hybrid = HybridIndex::new(2);
        hybrid.add("x#0", "中文 分词", &[1.0, 0.0]);
        save(&path, &hybrid, "demo", 2).unwrap();
        let snap = load(&path).unwrap().unwrap();
        assert_eq!(snap.model_id, "demo");
        assert_eq!(snap.dims, 2);
        assert_eq!(snap.hybrid.len(), 1);
    }
}
