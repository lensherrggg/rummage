//! Boundary-aware, UTF-8-safe text chunking for embedding pipelines.
//!
//! Splits long text into chunks of at most `max_bytes` bytes, preferring
//! to break at paragraph (`\n\n`), sentence (`. `), or line (`\n`)
//! boundaries, and never splitting inside a UTF-8 code point.

/// Split `text` into chunks of at most `max_bytes` bytes.
///
/// Break preference order within the window: paragraph boundary (`\n\n`),
/// sentence boundary (`. `), line break (`\n`), then a hard cut at the
/// nearest character boundary. Chunks concatenate back to the original
/// text exactly (no characters are lost or duplicated).
///
/// `max_bytes == 0` is treated as 1.
pub fn chunk_text(text: &str, max_bytes: usize) -> Vec<String> {
    let max_bytes = max_bytes.max(1);
    if text.len() <= max_bytes {
        if text.is_empty() {
            return Vec::new();
        }
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let tentative_end = floor_char_boundary(text, (start + max_bytes).min(text.len()));
        let window = &text[start..tentative_end];

        let actual_end = if tentative_end < text.len() {
            window
                .rfind("\n\n")
                .map(|pos| start + pos + 2)
                .or_else(|| window.rfind(". ").map(|pos| start + pos + 2))
                .or_else(|| window.rfind('\n').map(|pos| start + pos + 1))
                .filter(|&end| end > start)
                .unwrap_or(tentative_end)
        } else {
            tentative_end
        };

        // Guard against zero-progress when no boundary fits.
        let actual_end = if actual_end <= start {
            next_char_boundary(text, start + 1)
        } else {
            actual_end
        };

        chunks.push(text[start..actual_end].to_string());
        start = actual_end;
    }

    chunks
}

/// Like [`chunk_text`], but each chunk after the first is prefixed with the
/// last `overlap_bytes` (rounded to a char boundary) of the previous chunk.
/// Overlap preserves context across chunk borders for retrieval.
pub fn chunk_text_with_overlap(text: &str, max_bytes: usize, overlap_bytes: usize) -> Vec<String> {
    let base = chunk_text(text, max_bytes);
    if base.len() <= 1 || overlap_bytes == 0 {
        return base;
    }

    let mut out = Vec::with_capacity(base.len());
    out.push(base[0].clone());
    for i in 1..base.len() {
        let prev = &base[i - 1];
        let tail_start = ceil_char_boundary(prev, prev.len().saturating_sub(overlap_bytes));
        out.push(format!("{}{}", &prev[tail_start..], base[i]));
    }
    out
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn next_char_boundary(s: &str, idx: usize) -> usize {
    ceil_char_boundary(s, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(chunk_text("hello world", 100), vec!["hello world"]);
    }

    #[test]
    fn empty_text_is_no_chunks() {
        assert!(chunk_text("", 100).is_empty());
    }

    #[test]
    fn chunks_reassemble_exactly() {
        let text = "Paragraph one is here.\n\nParagraph two follows. It has sentences. \
                    More text continues here without any breaks at all to force hard cuts."
            .repeat(20);
        let chunks = chunk_text(&text, 128);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
        assert!(chunks.iter().all(|c| c.len() <= 128));
    }

    #[test]
    fn prefers_paragraph_boundaries() {
        let text = format!("{}\n\n{}", "a".repeat(60), "b".repeat(60));
        let chunks = chunk_text(&text, 100);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].ends_with("\n\n"));
        assert!(chunks[1].starts_with('b'));
    }

    #[test]
    fn never_splits_multibyte_chars() {
        // 4-byte emoji repeated: any naive byte cut would panic or corrupt.
        let text = "🦀".repeat(100);
        let chunks = chunk_text(&text, 10);
        assert_eq!(chunks.concat(), text);
        for c in &chunks {
            assert!(c.len() <= 10);
            assert!(c.chars().all(|ch| ch == '🦀'));
        }
    }

    #[test]
    fn multibyte_text_with_boundaries() {
        let text = "こんにちは世界。\n\nRustは素晴らしい。 ".repeat(50);
        let chunks = chunk_text(&text, 64);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn overlap_prefixes_previous_tail() {
        let text = format!("{}\n\n{}", "alpha ".repeat(20), "beta ".repeat(20));
        let chunks = chunk_text_with_overlap(&text, 100, 12);
        assert!(chunks.len() >= 2);
        // Second chunk must start with the tail of the first.
        let first = chunk_text(&text, 100);
        assert!(chunks[1].starts_with(&first[0][first[0].len() - 12..]));
    }

    #[test]
    fn zero_max_does_not_loop_forever() {
        let chunks = chunk_text("abc", 0);
        assert_eq!(chunks.concat(), "abc");
    }
}
