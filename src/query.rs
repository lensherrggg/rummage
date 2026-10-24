/// Single-pass query parser. Natural-language text is preserved; `dir:`, `name:`
/// and `"phrase"` take their content out of the main query.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedQuery {
    pub text: String,
    pub dir_filter: Option<String>,
    pub name_filter: Option<String>,
    pub exact_phrases: Vec<String>,
}

impl ParsedQuery {
    pub fn has_filters(&self) -> bool {
        self.dir_filter.is_some() || self.name_filter.is_some() || !self.exact_phrases.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && !self.has_filters()
    }
}

pub fn parse(input: &str) -> ParsedQuery {
    let chars: Vec<char> = input.chars().collect();
    let mut out = ParsedQuery::default();
    let mut text_parts: Vec<String> = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        if ch == '"' {
            if let Some(rel) = chars[i + 1..].iter().position(|&c| c == '"') {
                let phrase: String = chars[i + 1..i + 1 + rel].iter().collect();
                let phrase = phrase.trim();
                if !phrase.is_empty() {
                    out.exact_phrases.push(phrase.to_string());
                }
                i = i + 1 + rel + 1;
            } else {
                let rest: String = chars[i + 1..].iter().collect();
                let rest = rest.trim();
                if !rest.is_empty() {
                    text_parts.push(rest.to_string());
                }
                break;
            }
            continue;
        }

        if let Some((kind, prefix_len)) = filter_prefix(&chars[i..]) {
            let mut j = i + prefix_len;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            // `dir:` followed by another filter keyword means this one has no value.
            if j < chars.len() && filter_prefix(&chars[j..]).is_some() {
                i = j;
                continue;
            }
            let (value, next) = read_value(&chars, j);
            let value = value.trim().to_string();
            if !value.is_empty() {
                match kind {
                    FilterKind::Dir => out.dir_filter = Some(value),
                    FilterKind::Name => out.name_filter = Some(value),
                }
            }
            i = next;
            continue;
        }

        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '"' {
            i += 1;
        }
        if start == i {
            i += 1;
            continue;
        }
        let token: String = chars[start..i].iter().collect();
        text_parts.push(token);
    }

    out.text = text_parts.join(" ").trim().to_string();
    out
}

#[derive(Clone, Copy)]
enum FilterKind {
    Dir,
    Name,
}

fn filter_prefix(rest: &[char]) -> Option<(FilterKind, usize)> {
    let take = rest.len().min(5);
    let head: String = rest[..take].iter().collect::<String>().to_ascii_lowercase();
    if head.starts_with("dir:") {
        Some((FilterKind::Dir, 4))
    } else if head.starts_with("name:") {
        Some((FilterKind::Name, 5))
    } else {
        None
    }
}

fn read_value(chars: &[char], start: usize) -> (String, usize) {
    if start >= chars.len() {
        return (String::new(), start);
    }
    if chars[start] == '"' {
        if let Some(rel) = chars[start + 1..].iter().position(|&c| c == '"') {
            let value: String = chars[start + 1..start + 1 + rel].iter().collect();
            return (value, start + 1 + rel + 1);
        }
        let value: String = chars[start + 1..].iter().collect();
        return (value, chars.len());
    }
    let mut end = start;
    while end < chars.len() && !chars[end].is_whitespace() {
        end += 1;
    }
    let value: String = chars[start..end].iter().collect();
    (value, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_spec_examples() {
        let p = parse(r#"OCR预处理 dir:docs\ocr"#);
        assert_eq!(p.text, "OCR预处理");
        assert_eq!(p.dir_filter.as_deref(), Some(r"docs\ocr"));
        assert!(p.name_filter.is_none());

        let p = parse("断行修复 name:preprocess");
        assert_eq!(p.text, "断行修复");
        assert_eq!(p.name_filter.as_deref(), Some("preprocess"));

        let p = parse(r#""页眉页脚""#);
        assert_eq!(p.text, "");
        assert_eq!(p.exact_phrases, vec!["页眉页脚"]);

        let p = parse("如何合并硬换行");
        assert_eq!(p.text, "如何合并硬换行");
        assert!(p.exact_phrases.is_empty());
    }

    #[test]
    fn combines_all_parts() {
        let p = parse(r#"如何合并 "OCR 断行" dir:"docs/ocr 资料" name:setup"#);
        assert_eq!(p.text, "如何合并");
        assert_eq!(p.exact_phrases, vec!["OCR 断行"]);
        assert_eq!(p.dir_filter.as_deref(), Some("docs/ocr 资料"));
        assert_eq!(p.name_filter.as_deref(), Some("setup"));
    }

    #[test]
    fn multiple_phrases() {
        let p = parse(r#""页眉" 与 "页脚""#);
        assert_eq!(p.text, "与");
        assert_eq!(p.exact_phrases, vec!["页眉", "页脚"]);
    }

    #[test]
    fn unclosed_quote_becomes_text() {
        let p = parse(r#"断行 "修复"#);
        assert_eq!(p.text, "断行 修复");
        assert!(p.exact_phrases.is_empty());
    }

    #[test]
    fn empty_filter_value_is_dropped() {
        let p = parse("dir:  name: ");
        assert!(p.dir_filter.is_none());
        assert!(p.name_filter.is_none());
        assert!(p.text.is_empty());
    }

    #[test]
    fn last_filter_wins() {
        let p = parse("dir:a dir:b");
        assert_eq!(p.dir_filter.as_deref(), Some("b"));
    }

    #[test]
    fn colon_inside_text_is_preserved() {
        let p = parse("如何: 合并");
        assert_eq!(p.text, "如何: 合并");
    }

    #[test]
    fn is_empty_helper() {
        assert!(parse("   ").is_empty());
        assert!(!parse(r#""页眉""#).is_empty());
    }
}
