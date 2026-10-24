use crate::config::{is_cjk, NormalizeConfig};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

static PAGE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*[-—–]?\s*(?:\d{1,4}|[ivxlcdm]{1,6})\s*[-—–]?\s*$").unwrap()
});
static PAGE_CN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*第\s*\d+\s*页\s*$").unwrap());
static PAGE_EN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*page\s*\d+\s*(?:of\s*\d+)?\s*$").unwrap());
static PAGE_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:-{3,}|={3,}|\*{3,})\s*page\s*\d+\s*(?:-{3,}|={3,}|\*{3,})\s*$").unwrap()
});
static PAGE_HTML: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*<!--\s*page.*?-->\s*$").unwrap());
static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s{0,3}#{1,6}(?:\s+|$)").unwrap());
static LIST_ITEM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s{0,3}(?:[-*+]\s+|\d{1,3}[.)]\s+)").unwrap());
static QUOTE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s{0,3}(?:>\s?)+").unwrap());
static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(?:`{3,}|~{3,})").unwrap());
static HR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(?:[-*_]\s*){3,}$").unwrap());
static TABLE_ROW: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*\|").unwrap());
static IMAGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"!\[([^\]]*)\]\([^)]*\)").unwrap());
static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\([^)]*\)").unwrap());

/// Full document normalization used for indexing. The original file is never touched.
pub fn normalize(text: &str, cfg: &NormalizeConfig) -> String {
    if !cfg.enabled {
        return text.to_string();
    }

    let mut s = text.replace("\r\n", "\n").replace('\r', "\n");

    if cfg.control_chars {
        s = clean_control_chars(&s);
    }
    if cfg.width_fold {
        s = fold_width(&s);
    }
    if cfg.front_matter {
        // Remove YAML metadata before line rules run; otherwise strip-markdown /
        // horizontal-rule handling deletes the `---` delimiters and the block
        // can no longer be recognised.
        s = strip_front_matter(&s);
    }

    let lines: Vec<String> = s.split('\n').map(str::to_string).collect();
    let lines = if cfg.merge_hard_breaks {
        merge_hard_breaks(lines, cfg)
    } else {
        lines
    };

    let mut lines = lines;
    if cfg.drop_page_number_lines {
        lines = drop_page_number_lines(lines, cfg.page_number_max_len);
    }
    if cfg.drop_page_markers {
        lines = drop_page_markers(lines);
    }
    if cfg.drop_repeated_headers {
        lines = drop_repeated_headers(lines, cfg);
    }
    if cfg.collapse_blank_lines {
        lines = collapse_blank_lines(lines);
    }

    let mut s = lines.join("\n");
    if cfg.collapse_spaces {
        s = collapse_spaces(&s);
    }
    if cfg.strip_markdown {
        s = strip_markdown(&s);
    }
    if cfg.links {
        s = IMAGE.replace_all(&s, "$1").into_owned();
        s = LINK.replace_all(&s, "$1").into_owned();
    }
    if cfg.table_pipes {
        s = s.replace('|', " ");
    }

    s.trim().to_string()
}

/// Query normalization keeps only the non-destructive rules: dropping page-number
/// lines or front matter from a short query such as `2024` / `第3页` would erase it.
pub fn normalize_query(text: &str, cfg: &NormalizeConfig) -> String {
    let mut query_cfg = cfg.clone();
    query_cfg.merge_hard_breaks = false;
    query_cfg.drop_page_number_lines = false;
    query_cfg.drop_page_markers = false;
    query_cfg.drop_repeated_headers = false;
    query_cfg.front_matter = false;
    normalize(text, &query_cfg)
}

fn clean_control_chars(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\n' | '\t' => out.push(ch),
            '\u{000C}' | '\u{2028}' | '\u{2029}' => out.push('\n'),
            '\u{00AD}' => {}
            '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{FEFF}' => {}
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

fn fold_width(text: &str) -> String {
    text.chars()
        .map(|ch| {
            let cp = ch as u32;
            // Only fullwidth letters/digits are folded; Chinese full-width
            // punctuation (，、；：？！（）《》 etc.) is kept as written.
            if (0xFF10..=0xFF19).contains(&cp)
                || (0xFF21..=0xFF3A).contains(&cp)
                || (0xFF41..=0xFF5A).contains(&cp)
            {
                char::from_u32(cp - 0xFEE0).unwrap_or(ch)
            } else if cp == 0x3000 {
                ' '
            } else {
                ch
            }
        })
        .collect()
}

fn is_sentence_end(ch: char) -> bool {
    matches!(
        ch,
        '。' | '！'
            | '？'
            | '!'
            | '?'
            | '.'
            | '；'
            | ';'
            | '：'
            | ':'
            | '…'
            | '”'
            | '’'
            | '"'
            | '\''
            | '）'
            | ')'
            | '】'
            | '》'
            | '」'
            | '』'
            | '〕'
    )
}

fn is_structural_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    HEADING.is_match(t)
        || LIST_ITEM.is_match(t)
        || QUOTE.is_match(t)
        || FENCE.is_match(t)
        || HR.is_match(t)
        || TABLE_ROW.is_match(t)
        || t == "---"
        || t == "..."
        || is_page_number_line(t, 20)
        || is_page_marker(t)
}

fn is_page_number_line(trimmed: &str, max_len: usize) -> bool {
    if trimmed.chars().count() > max_len {
        return false;
    }
    PAGE_NUMBER.is_match(trimmed) || PAGE_CN.is_match(trimmed) || PAGE_EN.is_match(trimmed)
}

fn is_page_marker(trimmed: &str) -> bool {
    PAGE_MARKER.is_match(trimmed) || PAGE_HTML.is_match(trimmed)
}

fn front_matter_end(lines: &[String]) -> Option<usize> {
    let first = lines.iter().position(|l| !l.trim().is_empty())?;
    if lines[first].trim() != "---" {
        return None;
    }
    for (i, line) in lines.iter().enumerate().skip(first + 1) {
        let t = line.trim();
        if t == "---" || t == "..." {
            return Some(i);
        }
    }
    None
}

fn code_mask(lines: &[String]) -> Vec<bool> {
    let mut mask = vec![false; lines.len()];
    let mut inside = false;
    for (i, line) in lines.iter().enumerate() {
        if FENCE.is_match(line.trim_start()) {
            mask[i] = true;
            inside = !inside;
        } else if inside {
            mask[i] = true;
        }
    }
    mask
}

fn repeated_short_lines(lines: &[String], cfg: &NormalizeConfig) -> HashSet<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for line in lines {
        let t = line.trim();
        if !t.is_empty() && t.chars().count() <= cfg.header_footer_max_len {
            *counts.entry(t.to_string()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(_, n)| *n >= cfg.header_footer_min_repeat)
        .map(|(l, _)| l)
        .collect()
}

fn merge_hard_breaks(lines: Vec<String>, cfg: &NormalizeConfig) -> Vec<String> {
    let front_end = if cfg.front_matter {
        front_matter_end(&lines)
    } else {
        None
    };
    let repeated = repeated_short_lines(&lines, cfg);
    let code = code_mask(&lines);
    let mut out = Vec::with_capacity(lines.len());
    let mut i = 0;

    while i < lines.len() {
        let mut current = lines[i].clone();
        let protected_current = front_end.map(|e| i <= e).unwrap_or(false)
            || code[i]
            || repeated.contains(current.trim())
            || is_structural_line(&current);
        let mut next = i + 1;

        if !protected_current {
            while next < lines.len() {
                let line = &lines[next];
                if front_end.map(|e| next <= e).unwrap_or(false)
                    || code[next]
                    || repeated.contains(line.trim())
                    || is_structural_line(line)
                {
                    break;
                }
                if !can_merge(&current, line) {
                    break;
                }
                current = join_break(&current, line);
                next += 1;
                if current
                    .trim_end()
                    .chars()
                    .last()
                    .map(is_sentence_end)
                    .unwrap_or(true)
                {
                    break;
                }
            }
        }

        out.push(current);
        i = next;
    }

    out
}

fn can_merge(current: &str, next: &str) -> bool {
    let cur = current.trim_end();
    let nxt = next.trim();
    if cur.is_empty() || nxt.is_empty() {
        return false;
    }
    let last = match cur.chars().last() {
        Some(c) => c,
        None => return false,
    };
    if is_sentence_end(last) {
        return false;
    }
    // Avoid gluing a word onto a heading-like or table-like next line.
    !is_structural_line(nxt)
}

fn join_break(a: &str, b: &str) -> String {
    let left = a.trim_end();
    let right = b.trim_start();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    let lc = left.chars().last().unwrap();
    let rc = right.chars().next().unwrap();
    if is_cjk(lc) || is_cjk(rc) {
        format!("{left}{right}")
    } else {
        format!("{left} {right}")
    }
}

fn drop_page_number_lines(lines: Vec<String>, max_len: usize) -> Vec<String> {
    lines
        .into_iter()
        .filter(|l| !is_page_number_line(l.trim(), max_len))
        .collect()
}

fn drop_page_markers(lines: Vec<String>) -> Vec<String> {
    lines
        .into_iter()
        .filter(|l| !is_page_marker(l.trim()))
        .collect()
}

fn drop_repeated_headers(lines: Vec<String>, cfg: &NormalizeConfig) -> Vec<String> {
    let repeated = repeated_short_lines(&lines, cfg);
    if repeated.is_empty() {
        return lines;
    }
    let n = lines.len();
    let zone = cfg.header_footer_zone_lines;
    lines
        .into_iter()
        .enumerate()
        .filter(|(i, line)| {
            if *i >= zone && i + zone < n {
                return true;
            }
            !repeated.contains(line.trim())
        })
        .map(|(_, line)| line)
        .collect()
}

fn collapse_blank_lines(lines: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(lines.len());
    let mut blank = false;
    for line in lines {
        if line.trim().is_empty() {
            if !blank {
                out.push(String::new());
            }
            blank = true;
        } else {
            out.push(line);
            blank = false;
        }
    }
    out
}

fn collapse_spaces(text: &str) -> String {
    text.split('\n')
        .map(collapse_spaces_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn collapse_spaces_line(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == ' ' || ch == '\t' {
            let mut j = i;
            while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                j += 1;
            }
            let prev = out.chars().last();
            let next = chars.get(j).copied();
            let drop = match (prev, next) {
                (Some(p), Some(n)) => is_cjk(p) && is_cjk(n),
                _ => true,
            };
            if !drop {
                out.push(' ');
            }
            i = j;
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out.trim().to_string()
}

fn strip_markdown(text: &str) -> String {
    let mut out = Vec::new();
    for line in text.split('\n') {
        let t = line.trim_start();
        if FENCE.is_match(t) || HR.is_match(t) {
            out.push(String::new());
            continue;
        }
        let mut s = line.to_string();
        if HEADING.is_match(&s) {
            s = HEADING.replace(&s, "").into_owned();
        }
        if QUOTE.is_match(&s) {
            s = QUOTE.replace(&s, "").into_owned();
        }
        if LIST_ITEM.is_match(&s) {
            s = LIST_ITEM.replace(&s, "").into_owned();
        }
        out.push(s);
    }
    out.join("\n")
}

fn strip_front_matter(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let first = match lines.iter().position(|l| !l.trim().is_empty()) {
        Some(i) if lines[i].trim() == "---" => i,
        _ => return text.to_string(),
    };
    let close = lines
        .iter()
        .enumerate()
        .skip(first + 1)
        .find(|(_, l)| {
            let t = l.trim();
            t == "---" || t == "..."
        })
        .map(|(i, _)| i);
    match close {
        Some(end) => lines[end + 1..].join("\n").trim().to_string(),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NormalizeConfig {
        NormalizeConfig::default()
    }

    #[test]
    fn control_chars_and_bom_removed() {
        let text = "\u{FEFF}abc\u{200B}def\u{0007}ghi";
        assert_eq!(normalize(text, &cfg()), "abcdefghi");
    }

    #[test]
    fn width_fold_only_ascii_range_and_keeps_chinese_punctuation() {
        let text = "１２３ＡＢＣ，测试。";
        assert_eq!(normalize(text, &cfg()), "123ABC，测试。");
    }

    #[test]
    fn merges_chinese_hard_wraps() {
        let text = "如何合并\n硬换行\n\n## 标题\n下一行";
        let out = normalize(text, &cfg());
        assert!(out.starts_with("如何合并硬换行"), "{out}");
        assert!(out.contains("标题"), "{out}");
        assert!(out.contains("下一行"), "{out}");
    }

    #[test]
    fn does_not_merge_before_list_or_heading() {
        let text = "步骤说明\n- 第一步\n标题\n# 二级";
        let out = normalize(text, &cfg());
        assert!(out.contains("步骤说明"), "{out}");
        assert!(out.contains("第一步"), "{out}");
        assert!(out.contains("二级"), "{out}");
        assert!(!out.contains("步骤说明第一步"), "{out}");
    }

    #[test]
    fn drops_page_numbers_and_markers() {
        let text =
            "正文\n\n12\n\n第 3 页\n\nPage 4\n\n--- page 5 ---\n\n<!-- page 6 -->\n\n[7]\n\n结尾";
        let out = normalize(text, &cfg());
        for gone in ["12", "第 3 页", "Page 4", "page 5", "page 6"] {
            assert!(!out.contains(gone), "{gone} survived: {out}");
        }
        assert!(out.contains("正文"));
        assert!(out.contains("结尾"));
    }

    #[test]
    fn repeated_headers_only_removed_in_edges() {
        let mut text = String::new();
        text.push_str("某某公司文档\n");
        for i in 0..30 {
            text.push_str(&format!("正文内容第{i}行，这里保持普通长度的一段文字。\n"));
            if i == 10 {
                text.push_str("某某公司文档\n");
            }
        }
        text.push_str("某某公司文档\n");
        let cfg = NormalizeConfig {
            header_footer_zone_lines: 5,
            ..NormalizeConfig::default()
        };
        let out = normalize(&text, &cfg);
        let count = out.matches("某某公司文档").count();
        assert_eq!(count, 1, "{out}");
    }

    #[test]
    fn collapses_blank_lines_and_spaces() {
        let text = "中文   中文\n\n\n\nEnglish   words\n";
        let out = normalize(text, &cfg());
        assert_eq!(out, "中文中文\n\nEnglish words");
    }

    #[test]
    fn strips_markdown_and_links() {
        let text = "# 标题\n> 引用\n- 列表项\n\n![图注](a.png) [链接文字](https://x.y)\n\n| 甲 | 乙 |\n|---|---|";
        let out = normalize(text, &cfg());
        assert!(out.contains("标题"));
        assert!(out.contains("引用"));
        assert!(out.contains("列表项"));
        assert!(out.contains("图注 链接文字"), "{out}");
        assert!(out.contains("甲") && out.contains("乙"), "{out}");
        assert!(!out.contains('|'), "{out}");
        assert!(!out.contains('#'), "{out}");
    }

    #[test]
    fn strips_yaml_front_matter() {
        let text = "---\ntitle: 测试\ntags: [a, b]\n---\n正文第一句。\n正文第二句。";
        let out = normalize(text, &cfg());
        assert_eq!(out, "正文第一句。\n正文第二句。");
    }

    #[test]
    fn query_normalization_keeps_numbers() {
        assert_eq!(normalize_query("2024", &cfg()), "2024");
        assert_eq!(normalize_query("第3页", &cfg()), "第3页");
        let full = normalize("2024", &cfg());
        assert_eq!(full, "");
    }

    #[test]
    fn rules_can_be_disabled() {
        let c = NormalizeConfig {
            collapse_spaces: false,
            drop_page_number_lines: false,
            ..NormalizeConfig::default()
        };
        let out = normalize("中文   中文\n12", &c);
        assert!(out.contains("中文   中文"));
        assert!(out.contains("12"));
    }

    #[test]
    fn front_matter_lines_are_not_merged_away() {
        let text = "---\ntitle: 标题\ndate: 2024\n---\n正文";
        let out = normalize(text, &cfg());
        assert_eq!(out, "正文");
    }

    #[test]
    fn code_fence_content_is_not_merged() {
        let text = "```\nlet a = 1;\nlet b = 2;\n```\n普通段落\n下一行";
        let out = normalize(text, &cfg());
        assert!(out.contains("let a = 1;"), "{out}");
        assert!(out.contains("let b = 2;"), "{out}");
        assert!(out.contains("普通段落下一行"), "{out}");
    }
}
