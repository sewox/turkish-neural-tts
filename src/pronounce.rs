//! Spoken forms for a Turkish voice that only knows Turkish spelling.
//!
//! The neural voice reads every letter the Turkish way, so English words,
//! brand names and acronyms come out wrong ("Jira" is fine, "SonicWall" is
//! not). Before synthesis each such word is rewritten the way people say it in
//! a Turkish office:
//!
//! 1. the caller's own pronunciations ([`Hints::lexicon`]: `SonicWall` → `Sonik Vol`),
//! 2. a built-in list of common terms (Python → paytın, Docker → dakır),
//! 3. acronyms: Turkish ones with Turkish letter names (KVKK → ka ve ka ka),
//!    the rest with English letter names (API → ey pi ay),
//! 4. English words (marked foreign, CamelCase, or spelled with w/q/x)
//!    from the CMU pronouncing dictionary (deadline → dedlayn).
//!
//! A suffix after an apostrophe stays attached (API'ye → ey pi ayye) so the
//! voice never reads the apostrophe. Tokens starting with a digit are left to
//! the number normaliser.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Built-in readings, matched without case. From how the terms are said in
/// Turkish offices (collected for EchoMind, 2026-10).
const DEFAULT_LEXICON: &[(&str, &str)] = &[
    ("jira", "jira"),
    ("java", "cava"),
    ("javascript", "cava skript"),
    ("typescript", "tayp skript"),
    ("linux", "linuks"),
    ("python", "paytın"),
    ("azure", "ejır"),
    ("github", "git hab"),
    ("gitlab", "git lab"),
    ("docker", "dakır"),
    ("kubernetes", "kubernetis"),
    ("teams", "tims"),
    ("slack", "slek"),
    ("zoom", "zum"),
    ("excel", "eksel"),
    ("outlook", "autluk"),
    ("google", "gugıl"),
    ("microsoft", "maykrosoft"),
    ("notion", "noşın"),
    ("confluence", "konfluens"),
    ("sonicwall", "sonik vol"),
    ("salesforce", "seylsfors"),
    ("whatsapp", "vatsap"),
    ("linkedin", "linkdin"),
    ("youtube", "yutub"),
    ("iphone", "ayfon"),
    ("macbook", "mekbuk"),
    ("windows", "vindovs"),
];

/// Acronyms with a reading that is neither letter-by-letter rule (case-sensitive).
const ACRONYM_LEXICON: &[(&str, &str)] = &[("SQL", "si ku el")];

/// Turkish acronyms, read with Turkish letter names (KVKK → ka ve ka ka).
const TURKISH_ACRONYMS: &[&str] = &[
    "KVKK", "SGK", "SSK", "KDV", "ÖTV", "TBMM", "AB", "ABD", "TL", "İK", "TC", "TCMB", "SPK",
    "BDDK", "MEB", "YÖK", "PTT", "THY", "TRT", "AVM", "İSO", "İTO", "TOBB", "TÜİK", "BİST", "MHRS",
    "EFT", "TCKN", "VKN", "KEP", "İBB", "ABB", "AKP", "CHP", "MHP", "HDP", "TSK", "MİT", "SGM",
    "YKS", "LGS", "KPSS", "ALES", "YDS", "TMSF", "EPDK", "BTK", "GİB",
];

/// Acronyms said as a word; left as written.
const WORD_ACRONYMS: &[&str] = &[
    "TÜBİTAK",
    "ASELSAN",
    "ROKETSAN",
    "HAVELSAN",
    "KOSGEB",
    "NATO",
    "NASA",
    "IBAN",
    "MERSİS",
    "KOBİ",
    "UNESCO",
    "UNICEF",
    "FIFA",
    "UEFA",
    "OPEC",
    "SWIFT",
];

fn turkish_letter(c: char) -> &'static str {
    match c {
        'A' => "a",
        'B' => "be",
        'C' => "ce",
        'Ç' => "çe",
        'D' => "de",
        'E' => "e",
        'F' => "fe",
        'G' => "ge",
        'Ğ' => "yumuşak ge",
        'H' => "ha",
        'I' => "ı",
        'İ' => "i",
        'J' => "je",
        'K' => "ka",
        'L' => "le",
        'M' => "me",
        'N' => "ne",
        'O' => "o",
        'Ö' => "ö",
        'P' => "pe",
        'Q' => "kü",
        'R' => "re",
        'S' => "se",
        'Ş' => "şe",
        'T' => "te",
        'U' => "u",
        'Ü' => "ü",
        'V' => "ve",
        'W' => "ve",
        'X' => "iks",
        'Y' => "ye",
        'Z' => "ze",
        _ => "",
    }
}

fn english_letter(c: char) -> &'static str {
    match c {
        'A' => "ey",
        'B' => "bi",
        'C' => "si",
        'D' => "di",
        'E' => "i",
        'F' => "ef",
        'G' => "ci",
        'H' => "eyç",
        'I' => "ay",
        'J' => "cey",
        'K' => "key",
        'L' => "el",
        'M' => "em",
        'N' => "en",
        'O' => "o",
        'P' => "pi",
        'Q' => "kyu",
        'R' => "ar",
        'S' => "es",
        'T' => "ti",
        'U' => "yu",
        'V' => "vi",
        'W' => "dabılyu",
        'X' => "eks",
        'Y' => "vay",
        'Z' => "zi",
        // A Turkish letter in a "foreign" acronym: fall back to its Turkish name.
        other => turkish_letter(other),
    }
}

/// ARPAbet phoneme → Turkish spelling. AH0/ER are the reduced vowels.
fn phoneme(p: &str) -> &'static str {
    let base = p.trim_end_matches(['0', '1', '2']);
    if p == "AH0" {
        return "ı";
    }
    match base {
        "AA" => "a",
        "AE" => "e",
        "AH" => "a",
        "AO" => "o",
        "AW" => "av",
        "AY" => "ay",
        "B" => "b",
        "CH" => "ç",
        "D" => "d",
        "DH" => "d",
        "EH" => "e",
        "ER" => "ır",
        "EY" => "ey",
        "F" => "f",
        "G" => "g",
        "HH" => "h",
        "IH" => "i",
        "IY" => "i",
        "JH" => "c",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ng",
        "OW" => "o",
        "OY" => "oy",
        "P" => "p",
        "R" => "r",
        "S" => "s",
        "SH" => "ş",
        "T" => "t",
        "TH" => "t",
        "UH" => "u",
        "UW" => "u",
        "V" => "v",
        "W" => "v",
        "Y" => "y",
        "Z" => "z",
        "ZH" => "j",
        _ => "",
    }
}

fn cmu() -> &'static HashMap<&'static str, &'static str> {
    static DICT: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    DICT.get_or_init(|| {
        include_str!("../resources/cmudict.txt")
            .lines()
            .filter_map(|l| l.split_once(' '))
            .collect()
    })
}

/// Turkish spelling of an English word from the dictionary, if it is there.
fn from_cmu(word: &str) -> Option<String> {
    let lower = word.to_lowercase();
    let phones = cmu().get(lower.as_str())?;
    let o_not_a = lower.contains('o') && !lower.contains('a');
    let mut out: Vec<&str> = phones
        .split(' ')
        .map(|p| {
            // American "a" for a spelled "o" (sonic, Notion) reads as "o" in Turkey.
            if p.starts_with("AA") && o_not_a {
                "o"
            } else {
                phoneme(p)
            }
        })
        .collect();
    // A written plural "s" stays "s" (teams → tims, not timz).
    if out.last() == Some(&"z") && lower.ends_with('s') {
        *out.last_mut().unwrap() = "s";
    }
    Some(out.concat())
}

/// onboarding → on + boarding, when both halves are dictionary words.
fn split_compound(word: &str) -> Option<String> {
    let lower = word.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    (2..chars.len().saturating_sub(1)).rev().find_map(|i| {
        let (a, b): (String, String) = (chars[..i].iter().collect(), chars[i..].iter().collect());
        Some(from_cmu(&a)? + &from_cmu(&b)?)
    })
}

/// SonicWall → ["Sonic", "Wall"]; deadline → ["deadline"].
fn camel_parts(word: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    for c in word.chars() {
        let starts_new = c.is_uppercase()
            && parts
                .last()
                .is_some_and(|p| p.chars().last().is_some_and(char::is_lowercase));
        if parts.is_empty() || starts_new {
            parts.push(String::new());
        }
        parts.last_mut().unwrap().push(c);
    }
    parts
}

fn english_word(word: &str) -> Option<String> {
    camel_parts(word)
        .iter()
        .map(|p| from_cmu(p).or_else(|| split_compound(p)))
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.join(" "))
}

fn is_acronym(word: &str) -> bool {
    let n = word.chars().count();
    (2..=6).contains(&n) && word.chars().all(|c| c.is_uppercase() && c.is_alphabetic())
}

fn looks_foreign(word: &str) -> bool {
    let camel = camel_parts(word).len() > 1;
    let lower = word.to_lowercase();
    camel || lower.contains(['w', 'q', 'x'])
}

fn acronym(word: &str) -> String {
    let turkish = TURKISH_ACRONYMS.contains(&word) || word.contains(['Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']);
    word.chars()
        .map(if turkish {
            turkish_letter
        } else {
            english_letter
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A word the voice can say the English way: in the English dictionary (or
/// two dictionary words), CamelCase, spelled with w/q/x, or an acronym. Keeps
/// Turkish words a language model wrongly calls foreign out of the list.
pub fn plausibly_english(word: &str) -> bool {
    let w = word.trim();
    if w.is_empty() || w.contains(['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü'])
    {
        return false;
    }
    if is_acronym(w) || looks_foreign(w) || from_cmu(w).is_some() {
        return true;
    }
    // Two dictionary words (onboarding) only with English spelling in it:
    // Turkish words split too ("maliyet" = "mali" + "yet").
    english_spelling(w) && split_compound(w).is_some()
}

fn english_spelling(word: &str) -> bool {
    const PATTERNS: [&str; 12] = [
        "th", "sh", "ck", "ee", "oo", "oa", "ea", "ph", "gh", "tion", "ing", "ou",
    ];
    let lower = word.to_lowercase();
    PATTERNS.iter().any(|p| lower.contains(p))
}

/// The caller's pronunciations and the words to read as English (from a
/// language model, a glossary or the user).
#[derive(Debug, Default, Clone)]
pub struct Hints {
    /// The caller's own readings: term → spoken form, e.g. ("SonicWall", "Sonik Vol").
    pub lexicon: Vec<(String, String)>,
    /// Words to read as English (lowercased).
    pub foreign: HashSet<String>,
}

/// The spoken form of one word (without its suffix), or `None` to keep it.
fn spoken(base: &str, hints: &Hints) -> Option<String> {
    let lower = base.to_lowercase();
    if let Some((_, s)) = hints
        .lexicon
        .iter()
        .find(|(t, _)| t.to_lowercase() == lower)
    {
        return Some(s.clone());
    }
    if let Some((_, s)) = ACRONYM_LEXICON.iter().find(|(t, _)| *t == base) {
        return Some((*s).to_string());
    }
    if let Some((_, s)) = DEFAULT_LEXICON.iter().find(|(t, _)| *t == lower) {
        return Some((*s).to_string());
    }
    if base.contains('/') {
        let parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
        if !parts.is_empty() && parts.iter().all(|p| is_acronym(p)) {
            return Some(
                parts
                    .iter()
                    .map(|p| acronym(p))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        return None;
    }
    if WORD_ACRONYMS.contains(&base) {
        return None;
    }
    if is_acronym(base) {
        return Some(acronym(base));
    }
    if hints.foreign.contains(&lower) || looks_foreign(base) {
        return english_word(base);
    }
    None
}

fn tokens() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // A word (letters/digits, may contain "/") with an optional apostrophe suffix.
    RE.get_or_init(|| Regex::new(r"[\p{L}\p{N}][\p{L}\p{N}/]*(?:['’][\p{L}]+)?").unwrap())
}

/// Rewrites `text` so a Turkish-only voice reads it right.
pub fn prepare(text: &str, hints: &Hints) -> String {
    tokens()
        .replace_all(text, |caps: &regex::Captures| {
            let token = &caps[0];
            // Numbers, times and dates are the normaliser's job (14:30'da).
            if token.starts_with(|c: char| c.is_ascii_digit()) {
                return token.to_string();
            }
            let (base, suffix) = match token.find(['\'', '’']) {
                Some(i) => (&token[..i], token[i..].trim_start_matches(['\'', '’'])),
                None => (token, ""),
            };
            match spoken(base, hints) {
                // API'ye → "ey pi ayye": the suffix joins the last word, no apostrophe left.
                Some(s) => format!("{s}{suffix}"),
                None if suffix.is_empty() => base.to_string(),
                None => format!("{base}{suffix}"),
            }
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(text: &str) -> String {
        prepare(text, &Hints::default())
    }

    #[test]
    fn lists_are_consistent() {
        let mut seen = HashSet::new();
        for (term, reading) in DEFAULT_LEXICON {
            assert_eq!(*term, term.to_lowercase(), "{term}: keys are lowercase");
            assert_eq!(
                *reading,
                reading.to_lowercase(),
                "{term}: readings are lowercase"
            );
            assert!(seen.insert(term.to_string()), "{term} is listed twice");
        }
        let acronyms = ACRONYM_LEXICON
            .iter()
            .map(|(t, _)| *t)
            .chain(TURKISH_ACRONYMS.iter().copied())
            .chain(WORD_ACRONYMS.iter().copied());
        for term in acronyms {
            assert_eq!(
                term,
                term.to_uppercase(),
                "{term}: acronyms are in capitals"
            );
            assert!(seen.insert(term.to_lowercase()), "{term} is listed twice");
        }
    }

    #[test]
    fn turkish_acronyms_use_turkish_letter_names() {
        assert_eq!(p("KVKK uyumu"), "ka ve ka ka uyumu");
        assert_eq!(p("SGK ve KDV"), "se ge ka ve ka de ve");
        assert_eq!(p("ÖTV'yi"), "ö te veyi");
        assert_eq!(p("İK'ya"), "i kaya");
    }

    #[test]
    fn foreign_acronyms_use_english_letter_names() {
        assert_eq!(p("API'ye bağlan"), "ey pi ayye bağlan");
        assert_eq!(p("AWS ve CI/CD"), "ey dabılyu es ve si ay si di");
        assert_eq!(p("KPI, CRM, ERP"), "key pi ay, si ar em, i ar pi");
        assert_eq!(p("PDF, USB, VPN"), "pi di ef, yu es bi, vi pi en");
        assert_eq!(p("IT ve AI"), "ay ti ve ey ay");
        assert_eq!(p("SQL sorgusu"), "si ku el sorgusu");
    }

    #[test]
    fn word_acronyms_and_lowercase_words_are_left_alone() {
        assert_eq!(p("TÜBİTAK ve NATO"), "TÜBİTAK ve NATO");
        assert_eq!(p("ai ile it"), "ai ile it");
        assert_eq!(p("A"), "A");
    }

    #[test]
    fn common_terms_use_office_readings() {
        assert_eq!(p("Python, Docker ve Azure"), "paytın, dakır ve ejır");
        assert_eq!(p("Jira'da"), "jirada");
        assert_eq!(p("SonicWall yapılandırması"), "sonik vol yapılandırması");
        assert_eq!(p("Microsoft Teams'te"), "maykrosoft timste");
    }

    #[test]
    fn english_words_come_from_the_dictionary() {
        let hints = Hints {
            foreign: ["deadline", "pipeline", "feedback", "cluster", "onboarding"]
                .map(String::from)
                .into(),
            ..Default::default()
        };
        assert_eq!(
            prepare("deadline'ı, pipeline ve cluster'ı", &hints),
            "dedlaynı, payplayn ve klastırı"
        );
        assert_eq!(
            prepare("onboarding ve feedback", &hints),
            "anbording ve fidbek"
        );
        // Spelled with w/q/x or CamelCase: foreign without a hint.
        assert_eq!(p("workshop"), "vırkşop");
    }

    #[test]
    fn user_pronunciations_win() {
        let hints = Hints {
            lexicon: vec![
                ("Jira".into(), "cira".into()),
                ("SQL".into(), "es kyu el".into()),
            ],
            ..Default::default()
        };
        assert_eq!(prepare("Jira'da SQL", &hints), "cirada es kyu el");
    }

    #[test]
    fn numbers_and_plain_turkish_pass_through() {
        assert_eq!(
            p("Saat 14:30'da 15 Ekim'e kadar."),
            "Saat 14:30'da 15 Ekime kadar."
        );
        assert_eq!(p("Ayşe'nin notları"), "Ayşenin notları");
        assert_eq!(p("Toplantı bitti."), "Toplantı bitti.");
    }

    #[test]
    fn unknown_english_words_are_kept() {
        let hints = Hints {
            foreign: ["zzqxw".into()].into(),
            ..Default::default()
        };
        assert_eq!(prepare("zzqxw", &hints), "zzqxw");
    }

    #[test]
    fn english_plausibility_filters_turkish_words() {
        for w in [
            "deadline",
            "SonicWall",
            "workshop",
            "API",
            "onboarding",
            "Teams",
        ] {
            assert!(plausibly_english(w), "{w}");
        }
        for w in ["maliyet", "düşecek", "toplantı", "kalem", "", "  "] {
            assert!(!plausibly_english(w), "{w}");
        }
    }

    #[test]
    fn camel_case_splits_on_lower_to_upper() {
        assert_eq!(camel_parts("SonicWall"), ["Sonic", "Wall"]);
        assert_eq!(camel_parts("deadline"), ["deadline"]);
        assert_eq!(camel_parts("iOS"), ["i", "OS"]);
    }
}
