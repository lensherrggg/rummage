//! Best-effort local file logging.
//!
//! The TUI owns the terminal, so anything printed to stderr is lost or garbles
//! the screen. Errors and warnings are appended to a small log file instead:
//!
//! 1. `<exe dir>\logs\rummage.log`
//! 2. `%LOCALAPPDATA%\rummage\logs\rummage.log`
//! 3. `%TEMP%\rummage-logs\rummage.log`
//!
//! The first writable location wins. The log is rotated once at startup when
//! it grows past 5 MiB; there is intentionally no background thread.

use log::{Level, LevelFilter, Log, Metadata, Record};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

static INIT: OnceLock<Option<PathBuf>> = OnceLock::new();

struct FileLogger {
    file: Mutex<File>,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let target = record.target();
        let line = if target.is_empty() {
            format!(
                "{} [{}] {}\n",
                timestamp(),
                level_name(record.level()),
                record.args()
            )
        } else {
            format!(
                "{} [{}] {}: {}\n",
                timestamp(),
                level_name(record.level()),
                target,
                record.args()
            )
        };
        let mut file = match self.file.lock() {
            Ok(file) => file,
            Err(poisoned) => poisoned.into_inner(),
        };
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Install the global file logger. Safe to call more than once.
///
/// Returns the path that was selected, or `None` when no location was
/// writable (the program keeps running without file logging).
pub fn init() -> Option<PathBuf> {
    INIT.get_or_init(init_inner).clone()
}

pub fn path() -> Option<&'static Path> {
    INIT.get().and_then(|path| path.as_deref())
}

fn init_inner() -> Option<PathBuf> {
    for path in candidate_paths() {
        let Ok(file) = open_log(&path) else {
            continue;
        };
        let logger = Box::leak(Box::new(FileLogger {
            file: Mutex::new(file),
        }));
        if log::set_logger(logger).is_err() {
            return None;
        }
        log::set_max_level(LevelFilter::Info);
        return Some(path);
    }
    None
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("logs").join("rummage.log"));
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        if !local.trim().is_empty() {
            paths.push(
                PathBuf::from(local)
                    .join("rummage")
                    .join("logs")
                    .join("rummage.log"),
            );
        }
    }
    paths.push(
        std::env::temp_dir()
            .join("rummage-logs")
            .join("rummage.log"),
    );
    paths
}

fn open_log(path: &Path) -> std::io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::metadata(path)
        .map(|meta| meta.len() > MAX_LOG_BYTES)
        .unwrap_or(false)
    {
        let rotated = path.with_file_name("rummage.log.1");
        let _ = fs::remove_file(&rotated);
        let _ = fs::rename(path, &rotated);
    }
    OpenOptions::new().create(true).append(true).open(path)
}

fn level_name(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Warn => "WARN",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    }
}

fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_timestamp(now.as_secs())
}

fn format_timestamp(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let seconds_of_day = unix_secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60,
        seconds_of_day % 60
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 -> (y, m, d).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_epoch_and_known_dates() {
        assert_eq!(format_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_timestamp(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(format_timestamp(1_704_067_199), "2023-12-31T23:59:59Z");
    }

    #[test]
    fn civil_dates_round_trip_known_values() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
