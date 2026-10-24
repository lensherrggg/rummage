use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct StoredFile {
    pub id: i64,
    pub path: String,
    pub hash: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub norm_text: String,
    pub chunks: u32,
    pub embedded: bool,
}

pub struct Store {
    conn: Connection,
    db_path: PathBuf,
}

impl Store {
    pub fn open(index_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(index_dir)
            .with_context(|| format!("创建索引目录失败：{}", index_dir.display()))?;
        let db_path = index_dir.join("index.sqlite3");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("打开元数据库失败：{}", db_path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS files (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 path TEXT NOT NULL UNIQUE COLLATE NOCASE,
                 hash TEXT NOT NULL,
                 mtime_ns INTEGER NOT NULL,
                 size INTEGER NOT NULL,
                 norm_text TEXT NOT NULL,
                 chunks INTEGER NOT NULL,
                 embedded INTEGER NOT NULL DEFAULT 0,
                 indexed_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);",
        )?;

        let store = Self { conn, db_path };
        let version = store
            .meta("schema_version")?
            .and_then(|v| v.parse::<i64>().ok());
        match version {
            Some(v) if v == SCHEMA_VERSION => {}
            Some(v) => {
                store
                    .conn
                    .execute_batch("DROP TABLE files; DELETE FROM meta;")?;
                store.reset_schema(v)?;
            }
            None => store.reset_schema(0)?,
        }
        Ok(store)
    }

    fn reset_schema(&self, _old: i64) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS files (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 path TEXT NOT NULL UNIQUE COLLATE NOCASE,
                 hash TEXT NOT NULL,
                 mtime_ns INTEGER NOT NULL,
                 size INTEGER NOT NULL,
                 norm_text TEXT NOT NULL,
                 chunks INTEGER NOT NULL,
                 embedded INTEGER NOT NULL DEFAULT 0,
                 indexed_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);",
        )?;
        self.set_meta("schema_version", &SCHEMA_VERSION.to_string())
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn all_files(&self) -> Result<Vec<StoredFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, hash, mtime_ns, size, norm_text, chunks, embedded
             FROM files ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(StoredFile {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                mtime_ns: row.get(3)?,
                size: row.get(4)?,
                norm_text: row.get(5)?,
                chunks: row.get::<_, i64>(6)? as u32,
                embedded: row.get::<_, i64>(7)? != 0,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn get_by_path(&self, path: &str) -> Result<Option<StoredFile>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, path, hash, mtime_ns, size, norm_text, chunks, embedded
             FROM files WHERE path = ?1",
        )?;
        let mut rows = stmt.query(params![path])?;
        if let Some(row) = rows.next()? {
            Ok(Some(StoredFile {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                mtime_ns: row.get(3)?,
                size: row.get(4)?,
                norm_text: row.get(5)?,
                chunks: row.get::<_, i64>(6)? as u32,
                embedded: row.get::<_, i64>(7)? != 0,
            }))
        } else {
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert(
        &self,
        path: &str,
        hash: &str,
        mtime_ns: i64,
        size: i64,
        norm_text: &str,
        chunks: u32,
        embedded: bool,
    ) -> Result<i64> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.conn.execute(
            "INSERT INTO files(path, hash, mtime_ns, size, norm_text, chunks, embedded, indexed_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(path) DO UPDATE SET
                 hash = excluded.hash,
                 mtime_ns = excluded.mtime_ns,
                 size = excluded.size,
                 norm_text = excluded.norm_text,
                 chunks = excluded.chunks,
                 embedded = excluded.embedded,
                 indexed_at = excluded.indexed_at",
            params![
                path,
                hash,
                mtime_ns,
                size,
                norm_text,
                chunks as i64,
                if embedded { 1 } else { 0 },
                now
            ],
        )?;
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id FROM files WHERE path = ?1")?;
        let id = stmt.query_row(params![path], |row| row.get(0))?;
        Ok(id)
    }

    pub fn clear_embedded_all(&self) -> Result<()> {
        self.conn.execute("UPDATE files SET embedded = 0", [])?;
        Ok(())
    }

    pub fn update_chunks(&self, id: i64, chunks: u32) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET chunks = ?1 WHERE id = ?2",
            params![chunks as i64, id],
        )?;
        Ok(())
    }

    pub fn set_embedded(&self, id: i64, embedded: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET embedded = ?1 WHERE id = ?2",
            params![if embedded { 1 } else { 0 }, id],
        )?;
        Ok(())
    }

    pub fn delete_by_id(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;
        Ok(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn upsert_and_read_back() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let id = store
            .upsert("D:\\a.md", "hash1", 1, 10, "正文", 2, false)
            .unwrap();
        let same = store
            .upsert("D:\\a.md", "hash2", 2, 20, "新正文", 3, true)
            .unwrap();
        assert_eq!(id, same);
        let f = store.get_by_path("D:\\a.md").unwrap().unwrap();
        assert_eq!(f.hash, "hash2");
        assert_eq!(f.chunks, 3);
        assert!(f.embedded);
        assert_eq!(store.count().unwrap(), 1);
        store.delete_by_id(id).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn meta_roundtrip() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        store.set_meta("model_id", "demo").unwrap();
        assert_eq!(store.meta("model_id").unwrap().as_deref(), Some("demo"));
    }
}
