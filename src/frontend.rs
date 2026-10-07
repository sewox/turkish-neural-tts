//! Text frontend for the EMA voice: any written Turkish in, pieces of text in
//! the model's own alphabet out. Port of `ema_lightning/frontend.py` and
//! `chunker.py`, with the pronunciation pass (pronounce.rs) in front.

use crate::pronounce::{self, Hints};
use normalizer_tr::{AmbiguityPolicy, NormalizeOptions, Normalizer};
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

/// The model's 49 symbols, in id order (0 = padding, 1 = unknown).
pub const VOCAB: [&str; 49] = [
    "<pad>", "<unk>", " ", "!", "\"", "%", "&", "'", "(", ")", ",", "-", ".", "/", ":", ";", "?",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z", "ç", "ö", "ü", "ğ", "ı", "ş",
];

/// Letters per second of speech and the longest piece the model reads at once (~10 s).
const LETTERS_PER_SECOND: f32 = 18.0;
const MAX_SECONDS: f32 = 10.0;
const MAX_LETTERS: usize = 250;
pub const SENTENCE_PAUSE: f32 = 0.25;
pub const CLAUSE_PAUSE: f32 = 0.12;

pub fn symbol_id(c: char) -> i64 {
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    VOCAB
        .iter()
        .position(|v| *v == s)
        .map(|i| i as i64)
        .unwrap_or(1)
}

fn in_vocab(c: char) -> bool {
    symbol_id(c) > 1
}

fn normalizer() -> Option<&'static Normalizer> {
    static N: OnceLock<Option<Normalizer>> = OnceLock::new();
    N.get_or_init(|| Normalizer::new().ok()).as_ref()
}

/// Numbers, dates, times, money and units read aloud (normalizer-tr); the
/// text is kept as it is if the normaliser refuses it.
fn spoken(text: &str) -> String {
    let opts = NormalizeOptions {
        ambiguity_policy: AmbiguityPolicy::Fallback,
        ..Default::default()
    };
    match normalizer().map(|n| n.normalize(text, &opts)) {
        Some(Ok(r)) => r.normalized_text().to_string(),
        _ => text.to_string(),
    }
}

/// Lowercase the Turkish way, fold accents of foreign letters, drop anything
/// the model cannot read.
pub fn alphabet(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let c = match c {
            '’' | '‘' | 'ʼ' | '´' | '`' => '\'',
            '“' | '”' | '„' | '«' | '»' => '"',
            '–' | '—' | '−' => '-',
            'İ' => 'i',
            'I' => 'ı',
            other => other,
        };
        if c == '…' {
            out.push_str("...");
            continue;
        }
        for l in c.to_lowercase() {
            if in_vocab(l) {
                out.push(l);
            } else {
                // é → e, â → a; anything else becomes a space.
                let folded: String = l
                    .nfkd()
                    .filter(|x| !unicode_normalization::char::is_combining_mark(*x))
                    .collect();
                if !folded.is_empty() && folded.chars().all(in_vocab) {
                    out.push_str(&folded);
                } else {
                    out.push(' ');
                }
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Written text → model text: pronunciations, then the normaliser, then the alphabet.
pub fn prepare(text: &str, hints: &Hints) -> String {
    alphabet(&spoken(&pronounce::prepare(text, hints)))
}

fn finish(piece: &str) -> String {
    let core = piece.trim_end_matches(['"', '\'', ')']);
    if core.ends_with(['.', '!', '?']) {
        piece.to_string()
    } else {
        format!("{}.", piece.trim_end_matches([',', ';', ':', '-', ' ']))
    }
}

/// Byte offsets right after each match of `is_cut` that is followed by a space.
fn cut_points(text: &str, limit_bytes: usize, is_cut: impl Fn(char) -> bool) -> Option<usize> {
    let mut best = None;
    let mut iter = text.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let end = i + c.len_utf8();
        if end > limit_bytes {
            break;
        }
        if is_cut(c) && iter.peek().is_some_and(|(_, n)| *n == ' ') {
            best = Some(end);
        }
    }
    best
}

/// [(piece, seconds of silence after it)], each piece short enough for one
/// pass of the model and ending like a sentence.
pub fn chunk(text: &str, speed: f32) -> Vec<(String, f32)> {
    let limit = ((LETTERS_PER_SECOND * MAX_SECONDS * speed) as usize).min(MAX_LETTERS);
    let mut pieces = Vec::new();
    let mut rest = text.trim().to_string();
    while !rest.is_empty() {
        let (cut, pause) = if rest.chars().count() > limit {
            let limit_bytes = rest
                .char_indices()
                .nth(limit)
                .map(|(i, _)| i)
                .unwrap_or(rest.len());
            if let Some(c) = cut_points(&rest, limit_bytes, |c| matches!(c, '.' | '!' | '?')) {
                (c, SENTENCE_PAUSE)
            } else if let Some(c) = cut_points(&rest, limit_bytes, |c| matches!(c, ',' | ';' | ':'))
            {
                (c, CLAUSE_PAUSE)
            } else if let Some(c) = cut_points(&rest, limit_bytes, |c| c != ' ') {
                (c, CLAUSE_PAUSE)
            } else {
                (limit_bytes, CLAUSE_PAUSE)
            }
        } else {
            (rest.len(), 0.0)
        };
        let piece = rest[..cut].trim().to_string();
        rest = rest[cut..].trim().to_string();
        if piece.chars().any(char::is_alphabetic) {
            pieces.push((finish(&piece), pause));
        }
    }
    if let Some(last) = pieces.last_mut() {
        last.1 = 0.0;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphabet_lowercases_the_turkish_way() {
        assert_eq!(alphabet("İSTANBUL ve IĞDIR"), "istanbul ve ığdır");
        assert_eq!(alphabet("Café — “test”…"), "cafe - \"test\"...");
        assert_eq!(alphabet("emoji 🎉  ve\ttab"), "emoji ve tab");
    }

    #[test]
    fn symbol_ids_match_the_vocabulary() {
        assert_eq!(symbol_id(' '), 2);
        assert_eq!(symbol_id('a'), 17);
        assert_eq!(symbol_id('ş'), 48);
        assert_eq!(symbol_id('€'), 1);
    }

    #[test]
    fn full_frontend_reads_numbers_and_acronyms() {
        let t = prepare(
            "KVKK uyumu %20 arttı, saat 14:30'da API'ye bağlanıldı.",
            &Hints::default(),
        );
        assert_eq!(
            t,
            "ka ve ka ka uyumu yüzde yirmi arttı, saat on dört otuzda ey pi ayye bağlanıldı."
        );
    }

    #[test]
    fn short_text_is_one_piece() {
        assert_eq!(
            chunk("merhaba dünya", 1.0),
            [("merhaba dünya.".to_string(), 0.0)]
        );
    }

    #[test]
    fn long_text_is_cut_at_sentence_ends() {
        let sentence = "bu cümle yaklaşık altmış harf uzunluğunda bir deneme cümlesidir.";
        let text = [sentence; 5].join(" ");
        let pieces = chunk(&text, 1.0);
        assert!(pieces.len() >= 2);
        for (p, _) in &pieces {
            assert!(p.chars().count() <= 180, "{p}");
            assert!(p.ends_with('.'));
        }
        assert_eq!(pieces.last().unwrap().1, 0.0);
        assert!(pieces[..pieces.len() - 1]
            .iter()
            .all(|(_, gap)| *gap == SENTENCE_PAUSE));
    }

    #[test]
    fn text_without_sentence_ends_is_cut_at_commas_then_spaces() {
        let text = vec!["kelime"; 60].join(", ");
        let pieces = chunk(&text, 1.0);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|(p, _)| p.ends_with('.')));
        let words = vec!["kelime"; 60].join(" ");
        assert!(chunk(&words, 1.0).len() > 1);
    }
}
