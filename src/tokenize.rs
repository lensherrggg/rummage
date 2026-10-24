use jieba_rs::Jieba;
use std::collections::HashSet;
use std::sync::OnceLock;

static JIEBA: OnceLock<Jieba> = OnceLock::new();

/// Chinese function words. Without this filter, a query like "公司今年的经营
/// 收入情况" matches every document containing "的", which lets BM25 outvote
/// the semantic ranking.
const STOPWORDS: &[&str] = &[
    "的",
    "了",
    "着",
    "过",
    "地",
    "得",
    "是",
    "在",
    "和",
    "与",
    "及",
    "或",
    "而",
    "也",
    "就",
    "都",
    "又",
    "很",
    "更",
    "最",
    "太",
    "被",
    "把",
    "对",
    "从",
    "向",
    "到",
    "为",
    "以",
    "于",
    "之",
    "其",
    "这",
    "那",
    "这个",
    "那个",
    "这些",
    "那些",
    "一个",
    "一种",
    "一些",
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "谁",
    "什么",
    "怎么",
    "怎样",
    "如何",
    "为什么",
    "是否",
    "可以",
    "能够",
    "需要",
    "应该",
    "已经",
    "正在",
    "进行",
    "通过",
    "对于",
    "关于",
    "但是",
    "因为",
    "所以",
    "如果",
    "并且",
    "或者",
    "然后",
    "还是",
    "只是",
    "不是",
    "没有",
    "不会",
    "不能",
    "可能",
    "以及",
    "还有",
    "一样",
    "这样",
    "那样",
    "其中",
    "其他",
    "等等",
    "啊",
    "吗",
    "呢",
    "吧",
    "呀",
    "哦",
    "嗯",
    "么",
];

fn stopwords() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| STOPWORDS.iter().copied().collect())
}

fn jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

/// Chinese word segmentation shared by indexing and querying. Returns tokens
/// lowercased, stripped of pure punctuation and Chinese function words.
pub fn tokens(text: &str) -> Vec<String> {
    let stops = stopwords();
    jieba()
        .cut(text, true)
        .into_iter()
        .map(|t| t.word.trim().to_lowercase())
        .filter(|t| {
            !t.is_empty() && t.chars().any(char::is_alphanumeric) && !stops.contains(t.as_str())
        })
        .collect()
}

/// Segmented text handed to scour's BM25, which splits on whitespace and then
/// leaves Chinese tokens intact (they do not match its English stopwords).
pub fn segment(text: &str) -> String {
    tokens(text).join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_chinese_into_words() {
        let seg = segment("如何合并OCR断行");
        let toks: Vec<&str> = seg.split(' ').collect();
        assert!(toks.contains(&"合并"), "{seg}");
        assert!(toks.contains(&"断行"), "{seg}");
    }

    #[test]
    fn drops_punctuation_only_tokens() {
        let toks = tokens("页眉，页脚。 123");
        assert!(toks.iter().all(|t| t.chars().any(char::is_alphanumeric)));
        assert!(toks.iter().any(|t| t == "页眉"));
        assert!(toks.iter().any(|t| t == "123"));
    }

    #[test]
    fn drops_chinese_function_words() {
        let toks = tokens("公司今年的经营收入情况");
        assert!(!toks.iter().any(|t| t == "的"), "{toks:?}");
        assert!(toks.iter().any(|t| t == "公司"), "{toks:?}");
        assert!(!segment("如何合并OCR断行").contains("如何"));
    }

    #[test]
    fn lowercases_ascii() {
        let toks = tokens("OCR Pipeline");
        assert!(toks.contains(&"ocr".to_string()));
        assert!(toks.contains(&"pipeline".to_string()));
    }
}
