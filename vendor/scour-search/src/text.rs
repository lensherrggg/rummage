//! Text analysis: Unicode-aware tokenization, English stopword removal,
//! and a Porter stemmer.
//!
//! This is the analyzer used by [`crate::bm25::Bm25Index`]; it is exposed
//! publicly so callers can pre-process queries or build their own indexes
//! on top of the same token stream.

/// Tokenize `text`: lowercase, split on non-alphanumeric boundaries,
/// drop stopwords, and Porter-stem each surviving token.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            push_token(&mut tokens, &current);
            current.clear();
        }
    }
    if !current.is_empty() {
        push_token(&mut tokens, &current);
    }

    tokens
}

fn push_token(tokens: &mut Vec<String>, raw: &str) {
    if is_stopword(raw) {
        return;
    }
    let stemmed = stem(raw);
    if !stemmed.is_empty() {
        tokens.push(stemmed);
    }
}

/// Returns true for a small, high-frequency English stopword set.
pub fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "is"
            | "at"
            | "which"
            | "on"
            | "a"
            | "an"
            | "and"
            | "or"
            | "but"
            | "in"
            | "to"
            | "of"
            | "for"
            | "with"
            | "as"
            | "by"
            | "from"
            | "that"
            | "this"
            | "it"
            | "its"
            | "be"
            | "are"
            | "was"
            | "were"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
            | "will"
            | "would"
            | "could"
            | "should"
            | "may"
            | "might"
            | "shall"
            | "can"
            | "not"
            | "no"
            | "nor"
            | "so"
            | "if"
            | "than"
            | "too"
            | "very"
            | "just"
            | "also"
    )
}

/// Porter-stem a single lowercase word.
///
/// Implements the classic five-step Porter algorithm (1980). Words of
/// length <= 2 are returned unchanged.
pub fn stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_string();
    }

    let mut stem = word.to_string();

    step_1a(&mut stem);
    step_1b(&mut stem);
    step_2(&mut stem);
    step_3(&mut stem);
    step_4(&mut stem);
    step_5(&mut stem);

    stem
}

fn step_1a(word: &mut String) {
    if word.ends_with("sses") {
        replace_suffix(word, "sses", "ss");
    } else if word.ends_with("ies") {
        replace_suffix(word, "ies", "i");
    } else if word.ends_with("ss") {
    } else if word.ends_with('s') {
        word.pop();
    }
}

fn step_1b(word: &mut String) {
    if word.ends_with("eed") {
        let stem = &word[..word.len() - 3];
        if measure(stem) > 0 {
            replace_suffix(word, "eed", "ee");
        }
        return;
    }

    let mut changed = false;
    if word.ends_with("ed") {
        let stem = &word[..word.len() - 2];
        if contains_vowel(stem) {
            word.truncate(word.len() - 2);
            changed = true;
        }
    } else if word.ends_with("ing") {
        let stem = &word[..word.len() - 3];
        if contains_vowel(stem) {
            word.truncate(word.len() - 3);
            changed = true;
        }
    }

    if !changed {
        return;
    }

    if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
        word.push('e');
    } else if ends_with_double_consonant(word)
        && !word.ends_with('l')
        && !word.ends_with('s')
        && !word.ends_with('z')
    {
        word.pop();
    } else if measure(word) == 1 && cvc(word) {
        word.push('e');
    }
}

fn step_2(word: &mut String) {
    const RULES: [(&str, &str); 20] = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];

    for (suffix, replacement) in RULES {
        if word.ends_with(suffix) {
            let stem = &word[..word.len() - suffix.len()];
            if measure(stem) > 0 {
                replace_suffix(word, suffix, replacement);
            }
            return;
        }
    }
}

fn step_3(word: &mut String) {
    const RULES: [(&str, &str); 7] = [
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];

    for (suffix, replacement) in RULES {
        if word.ends_with(suffix) {
            let stem = &word[..word.len() - suffix.len()];
            if measure(stem) > 0 {
                replace_suffix(word, suffix, replacement);
            }
            return;
        }
    }
}

fn step_4(word: &mut String) {
    const RULES: [&str; 19] = [
        "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous",
        "ive", "ize", "al", "er", "ic", "ou", "ion",
    ];

    for suffix in RULES {
        if word.ends_with(suffix) {
            let stem = &word[..word.len() - suffix.len()];
            if measure(stem) <= 1 {
                return;
            }

            if suffix == "ion" {
                if stem.ends_with('s') || stem.ends_with('t') {
                    word.truncate(word.len() - suffix.len());
                }
            } else {
                word.truncate(word.len() - suffix.len());
            }
            return;
        }
    }
}

fn step_5(word: &mut String) {
    if word.ends_with('e') {
        let stem = &word[..word.len() - 1];
        let m = measure(stem);
        if m > 1 || (m == 1 && !cvc(stem)) {
            word.pop();
        }
    }

    if word.ends_with("ll") && measure(word) > 1 {
        word.pop();
    }
}

fn replace_suffix(word: &mut String, suffix: &str, replacement: &str) {
    let new_len = word.len() - suffix.len();
    word.truncate(new_len);
    word.push_str(replacement);
}

fn is_consonant(chars: &[char], i: usize) -> bool {
    match chars[i] {
        'a' | 'e' | 'i' | 'o' | 'u' => false,
        'y' => {
            if i == 0 {
                true
            } else {
                !is_consonant(chars, i - 1)
            }
        }
        _ => true,
    }
}

/// Porter "measure": the number of vowel-consonant sequences in `word`.
fn measure(word: &str) -> usize {
    let chars: Vec<char> = word.chars().collect();
    let mut count = 0;
    let mut in_vowel_run = false;

    for i in 0..chars.len() {
        if is_consonant(&chars, i) {
            if in_vowel_run {
                count += 1;
                in_vowel_run = false;
            }
        } else {
            in_vowel_run = true;
        }
    }

    count
}

fn contains_vowel(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    chars
        .iter()
        .enumerate()
        .any(|(i, _)| !is_consonant(&chars, i))
}

fn ends_with_double_consonant(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 2 {
        return false;
    }

    let last = chars.len() - 1;
    chars[last] == chars[last - 1] && is_consonant(&chars, last)
}

fn cvc(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 3 {
        return false;
    }

    let len = chars.len();
    if !is_consonant(&chars, len - 1)
        || is_consonant(&chars, len - 2)
        || !is_consonant(&chars, len - 3)
    {
        return false;
    }

    !matches!(chars[len - 1], 'w' | 'x' | 'y')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_expected_words() {
        assert_eq!(stem("running"), "run");
        assert_eq!(stem("trees"), "tree");
        assert_eq!(stem("connections"), "connect");
        assert_eq!(stem("relational"), "relat");
        assert_eq!(stem("happiness"), "happi");
    }

    #[test]
    fn short_words_unchanged() {
        assert_eq!(stem("go"), "go");
        assert_eq!(stem("at"), "at");
    }

    #[test]
    fn tokenize_drops_stopwords_and_stems() {
        let tokens = tokenize("The runners were running through the forest");
        assert_eq!(tokens, vec!["runner", "run", "through", "forest"]);
    }

    #[test]
    fn tokenize_handles_unicode_and_punctuation() {
        let tokens = tokenize("Búsqueda híbrida: vectors + keywords!");
        assert!(tokens.contains(&"búsqueda".to_string()));
        assert!(tokens.contains(&"vector".to_string()));
    }

    #[test]
    fn tokenize_empty_input() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("the and is of").is_empty());
    }
}
