//! English text → Kokoro phonemes, without espeak.
//!
//! A port of the dictionary path of misaki (hexgrad/misaki, Apache-2.0),
//! Kokoro's own G2P: its gold/silver dictionaries, the special cases (a/an/
//! the/to/in), -s/-ed/-ing stems and capital-letter stress. misaki uses spaCy
//! part-of-speech tags to pick between heteronyms ("read"); here the default
//! reading is used. Words the dictionaries do not know are Turkish names in
//! Turkish workplace text far more often than not, so they are read the Turkish way
//! (Turkish spelling is phonetic); acronyms are spelled out.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

const PRIMARY: char = 'ˈ';
const SECONDARY: char = 'ˌ';
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const DIPHTHONGS: &str = "AIOQWYʤʧ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const PUNCTS: &str = ";:,.!?—…\"“”";

/// Tech words the dictionaries lack, the way English speakers say them.
const TECH: &[(&str, &str)] = &[
    ("jira", "ʤˈiɹə"),
    ("kubernetes", "kˌubəɹnˈɛtiz"),
    ("sonicwall", "sˈɑnɪkwˌɔl"),
    ("kvkk", "kˌAvˌikˌAkˈA"),
    ("github", "ɡˈɪthˌʌb"),
    ("gitlab", "ɡˈɪtlˌæb"),
    ("devops", "dˈɛvˌɑps"),
    ("saas", "sˈæs"),
    ("figma", "fˈɪɡmə"),
];

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Entry {
    Plain(String),
    ByTag(HashMap<String, Option<String>>),
}

impl Entry {
    fn default_reading(&self) -> Option<&str> {
        match self {
            Entry::Plain(s) => Some(s),
            Entry::ByTag(m) => m.get("DEFAULT").and_then(|v| v.as_deref()),
        }
    }
}

pub struct Lexicon {
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
}

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

/// misaki `grow_dictionary`: "word" also answers "Word" and the reverse.
fn grow(d: HashMap<String, Entry>) -> HashMap<String, Entry> {
    let mut extra = HashMap::new();
    for (k, v) in &d {
        if k.chars().count() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        let cap = capitalize(&lower);
        if *k == lower {
            if *k != cap {
                extra.insert(cap, v.clone());
            }
        } else if *k == cap {
            extra.insert(lower, v.clone());
        }
    }
    extra.extend(d);
    extra
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f
            .to_uppercase()
            .chain(c.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

/// misaki `apply_stress`: stress < -1 removes, -1/0 demotes, 0.5..1 adds a
/// secondary, > 1 adds a primary.
fn apply_stress(ps: &str, stress: Option<f32>) -> String {
    let Some(stress) = stress else {
        return ps.to_string();
    };
    let has_any = ps.contains(PRIMARY) || ps.contains(SECONDARY);
    let has_vowel = ps.chars().any(is_vowel);
    if stress < -1.0 {
        ps.replace([PRIMARY, SECONDARY], "")
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && ps.contains(PRIMARY)) {
        ps.replace(SECONDARY, "")
            .replace(PRIMARY, &SECONDARY.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_any {
        if has_vowel {
            restress(&format!("{SECONDARY}{ps}"))
        } else {
            ps.to_string()
        }
    } else if stress >= 1.0 && !ps.contains(PRIMARY) && ps.contains(SECONDARY) {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has_any {
        if has_vowel {
            restress(&format!("{PRIMARY}{ps}"))
        } else {
            ps.to_string()
        }
    } else {
        ps.to_string()
    }
}

/// Moves each stress mark right before the next vowel.
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut keyed: Vec<(f32, char)> = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f32, c))
        .collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == PRIMARY || c == SECONDARY {
            if let Some(j) = chars[i..].iter().position(|&v| is_vowel(v)) {
                keyed[i].0 = (i + j) as f32 - 0.5;
            }
        }
    }
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|(_, c)| c).collect()
}

fn s_suffix(stem: &str) -> String {
    match stem.chars().last() {
        Some(c) if "ptkfθ".contains(c) => format!("{stem}s"),
        Some(c) if "szʃʒʧʤ".contains(c) => format!("{stem}ᵻz"),
        _ => format!("{stem}z"),
    }
}

fn ed_suffix(stem: &str) -> String {
    let chars: Vec<char> = stem.chars().collect();
    match chars.last() {
        Some(c) if "pkfθʃsʧ".contains(*c) => format!("{stem}t"),
        Some('d') => format!("{stem}ᵻd"),
        Some(c) if *c != 't' => format!("{stem}d"),
        _ if chars.len() < 2 => format!("{stem}ɪd"),
        _ if US_TAUS.contains(chars[chars.len() - 2]) => {
            format!("{}ɾᵻd", chars[..chars.len() - 1].iter().collect::<String>())
        }
        _ => format!("{stem}ᵻd"),
    }
}

fn ing_suffix(stem: &str) -> String {
    let chars: Vec<char> = stem.chars().collect();
    if chars.len() > 1 && chars[chars.len() - 1] == 't' && US_TAUS.contains(chars[chars.len() - 2])
    {
        format!("{}ɾɪŋ", chars[..chars.len() - 1].iter().collect::<String>())
    } else {
        format!("{stem}ɪŋ")
    }
}

/// Turkish spelling → Kokoro phonemes, stress on the last syllable.
pub fn turkish(word: &str) -> String {
    let mut ps = String::new();
    for c in word
        .replace('I', "ı")
        .replace('İ', "i")
        .to_lowercase()
        .chars()
    {
        ps.push_str(match c {
            'a' => "ɑ",
            'b' => "b",
            'c' => "ʤ",
            'ç' => "ʧ",
            'd' => "d",
            'e' => "ɛ",
            'f' => "f",
            'g' => "ɡ",
            'ğ' => "",
            'h' => "h",
            'ı' => "ɯ",
            'i' => "i",
            'j' => "ʒ",
            'k' => "k",
            'l' => "l",
            'm' => "m",
            'n' => "n",
            'o' => "o",
            'ö' => "ø",
            'p' => "p",
            'r' => "ɾ",
            's' => "s",
            'ş' => "ʃ",
            't' => "t",
            'u' => "u",
            'ü' => "y",
            'v' => "v",
            'y' => "j",
            'z' => "z",
            'w' => "v",
            'q' => "k",
            'x' => "ks",
            _ => "",
        });
    }
    let chars: Vec<char> = ps.chars().collect();
    match chars.iter().rposition(|c| "ɑɛɯioøuy".contains(*c)) {
        Some(last) => {
            let mut out: String = chars[..last].iter().collect();
            out.push(PRIMARY);
            out.extend(&chars[last..]);
            out
        }
        None => ps,
    }
}

fn ones(n: u64) -> &'static str {
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ][n as usize]
}

/// Cardinal number in English words (up to the trillions).
pub fn number_words(n: u64) -> String {
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    fn below_thousand(n: u64) -> String {
        let mut parts = Vec::new();
        if n >= 100 {
            parts.push(format!("{} hundred", ones(n / 100)));
        }
        let r = n % 100;
        if r > 0 {
            parts.push(if r < 20 {
                ones(r).to_string()
            } else if r.is_multiple_of(10) {
                TENS[(r / 10) as usize].to_string()
            } else {
                format!("{} {}", TENS[(r / 10) as usize], ones(r % 10))
            });
        }
        parts.join(" ")
    }
    if n == 0 {
        return "zero".into();
    }
    let scales = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
    let mut rest = n;
    let mut parts = Vec::new();
    for (size, name) in scales {
        if rest >= size {
            parts.push(format!("{} {name}", below_thousand(rest / size)));
            rest %= size;
        }
    }
    if rest > 0 {
        parts.push(below_thousand(rest));
    }
    parts.join(" ")
}

/// Digits, decimals and percentages in a token as English words; `None` when
/// the token is not a number.
fn spell_number(token: &str) -> Option<String> {
    let (body, percent) = match token.strip_suffix('%') {
        Some(b) => (b, true),
        None => (token, false),
    };
    let body = body.replace(',', "");
    if body.is_empty()
        || !body.chars().all(|c| c.is_ascii_digit() || c == '.')
        || body.matches('.').count() > 1
    {
        return None;
    }
    let mut words = match body.split_once('.') {
        Some((int, frac)) => {
            let int = if int.is_empty() {
                "zero".to_string()
            } else {
                number_words(int.parse().ok()?)
            };
            let frac: Vec<&str> = frac
                .chars()
                .map(|d| ones(d.to_digit(10).unwrap_or(0) as u64))
                .collect();
            format!("{int} point {}", frac.join(" "))
        }
        None => number_words(body.parse().ok()?),
    };
    if percent {
        words.push_str(" percent");
    }
    Some(words)
}

impl Lexicon {
    /// Loads misaki's `us_gold.json` and `us_silver.json` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<HashMap<String, Entry>, String> {
            let text =
                std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
            serde_json::from_str(&text).map_err(|e| format!("{name}: {e}"))
        };
        Ok(Lexicon {
            golds: grow(read("us_gold.json")?),
            silvers: grow(read("us_silver.json")?),
        })
    }

    #[cfg(test)]
    fn from_maps(golds: &[(&str, &str)], silvers: &[(&str, &str)]) -> Self {
        let m = |v: &[(&str, &str)]| {
            v.iter()
                .map(|(k, p)| (k.to_string(), Entry::Plain(p.to_string())))
                .collect::<HashMap<_, _>>()
        };
        Lexicon {
            golds: grow(m(golds)),
            silvers: grow(m(silvers)),
        }
    }

    fn gold(&self, w: &str) -> Option<&str> {
        self.golds.get(w).and_then(Entry::default_reading)
    }

    fn known(&self, w: &str) -> bool {
        if self.golds.contains_key(w) || self.silvers.contains_key(w) {
            return true;
        }
        if !w
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-')
        {
            return false;
        }
        if w.chars().count() == 1 {
            return true;
        }
        if w == w.to_uppercase() && self.golds.contains_key(&w.to_lowercase()) {
            return true;
        }
        let rest: String = w.chars().skip(1).collect();
        rest == rest.to_uppercase()
    }

    /// Letters read one by one (acronyms), misaki `get_NNP`.
    fn letters(&self, word: &str) -> Option<String> {
        let mut ps = String::new();
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            ps.push_str(self.gold(&c.to_uppercase().to_string())?);
        }
        let ps = apply_stress(&ps, Some(0.0));
        Some(match ps.rfind(SECONDARY) {
            Some(i) => format!("{}{PRIMARY}{}", &ps[..i], &ps[i + SECONDARY.len_utf8()..]),
            None => ps,
        })
    }

    fn lookup(&self, word: &str, stress: Option<f32>) -> Option<String> {
        let mut w = word.to_string();
        if w == w.to_uppercase() && !self.golds.contains_key(&w) {
            w = w.to_lowercase();
        }
        let ps = self
            .golds
            .get(&w)
            .and_then(Entry::default_reading)
            .or_else(|| self.silvers.get(&w).and_then(Entry::default_reading));
        match ps {
            Some(p) => Some(apply_stress(p, stress)),
            None => self.letters(&w),
        }
    }

    fn stem_s(&self, w: &str, stress: Option<f32>) -> Option<String> {
        let n = w.chars().count();
        if n < 3 || !w.ends_with('s') {
            return None;
        }
        let stem = if !w.ends_with("ss") && self.known(&w[..w.len() - 1]) {
            w[..w.len() - 1].to_string()
        } else if (w.ends_with("'s") || (n > 4 && w.ends_with("es") && !w.ends_with("ies")))
            && self.known(&w[..w.len() - 2])
        {
            w[..w.len() - 2].to_string()
        } else if n > 4 && w.ends_with("ies") && self.known(&format!("{}y", &w[..w.len() - 3])) {
            format!("{}y", &w[..w.len() - 3])
        } else {
            return None;
        };
        Some(s_suffix(&self.lookup(&stem, stress)?))
    }

    fn stem_ed(&self, w: &str, stress: Option<f32>) -> Option<String> {
        let n = w.chars().count();
        if n < 4 || !w.ends_with('d') {
            return None;
        }
        let stem = if !w.ends_with("dd") && self.known(&w[..w.len() - 1]) {
            &w[..w.len() - 1]
        } else if n > 4 && w.ends_with("ed") && !w.ends_with("eed") && self.known(&w[..w.len() - 2])
        {
            &w[..w.len() - 2]
        } else {
            return None;
        };
        Some(ed_suffix(&self.lookup(stem, stress)?))
    }

    fn stem_ing(&self, w: &str, stress: Option<f32>) -> Option<String> {
        let n = w.chars().count();
        if n < 5 || !w.ends_with("ing") {
            return None;
        }
        let base = &w[..w.len() - 3];
        let doubled = {
            let c: Vec<char> = base.chars().collect();
            c.len() >= 2
                && c[c.len() - 1] == c[c.len() - 2]
                && "bcdgklmnprstvxz".contains(c[c.len() - 1])
        };
        let stem = if n > 5 && self.known(base) {
            base.to_string()
        } else if self.known(&format!("{base}e")) {
            format!("{base}e")
        } else if n > 5 && (doubled || w.ends_with("cking")) && self.known(&w[..w.len() - 4]) {
            w[..w.len() - 4].to_string()
        } else {
            return None;
        };
        Some(ing_suffix(&self.lookup(&stem, stress)?))
    }

    /// One word (no spaces or punctuation): dictionary, stems, acronyms,
    /// numbers; `None` when unknown. `next_vowel`: the next word starts with a vowel sound.
    fn word(&self, word: &str, next_vowel: Option<bool>) -> Option<String> {
        let lower = word.to_lowercase();
        if let Some((_, p)) = TECH.iter().find(|(w, _)| *w == lower) {
            return Some((*p).to_string());
        }
        match word {
            "a" | "A" if word == "a" => return Some("ɐ".into()),
            "an" | "An" => return Some("ɐn".into()),
            "the" | "The" => {
                return Some(
                    if next_vowel == Some(true) {
                        "ði"
                    } else {
                        "ðə"
                    }
                    .into(),
                )
            }
            "to" | "To" => {
                return Some(match next_vowel {
                    None => self.gold("to").unwrap_or("tʊ").to_string(),
                    Some(false) => "tə".into(),
                    Some(true) => "tʊ".into(),
                })
            }
            "in" | "In" => return Some(if next_vowel.is_none() { "ˈɪn" } else { "ɪn" }.into()),
            "I" => return Some("ˌI".into()),
            _ => {}
        }
        if let Some(n) = spell_number(word) {
            return Some(
                n.split(' ')
                    .filter_map(|w| self.lookup(w, None))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        let stress = if word == lower {
            None
        } else if word == word.to_uppercase() {
            Some(2.0)
        } else {
            Some(0.5)
        };
        let mut w = word.to_string();
        let is_alpha = word.chars().all(|c| c.is_ascii_alphabetic() || c == '\'');
        if word.chars().count() > 1
            && is_alpha
            && word != lower
            && !self.golds.contains_key(word)
            && !self.silvers.contains_key(word)
            && (word == word.to_uppercase() || word[1..] == word[1..].to_lowercase())
            && (self.golds.contains_key(&lower)
                || self.silvers.contains_key(&lower)
                || self.stem_s(&lower, stress).is_some()
                || self.stem_ed(&lower, stress).is_some()
                || self.stem_ing(&lower, stress).is_some())
        {
            w = lower.clone();
        }
        if self.known(&w) && is_alpha {
            return self.lookup(&w, stress);
        }
        if let Some(stem) = w
            .strip_suffix("s'")
            .filter(|s| self.known(&format!("{s}'s")))
        {
            return self.lookup(&format!("{stem}'s"), stress);
        }
        self.stem_s(&w, stress)
            .or_else(|| self.stem_ed(&w, stress))
            .or_else(|| self.stem_ing(&w, Some(stress.unwrap_or(0.5))))
    }

    /// English text → Kokoro phoneme string.
    pub fn phonemize(&self, text: &str) -> String {
        // Words with their trailing punctuation, hyphens as spaces.
        let mut tokens: Vec<(String, String, String)> = Vec::new();
        for raw in text.replace(['-', '–'], " ").split_whitespace() {
            let lead: String = raw
                .chars()
                .take_while(|c| PUNCTS.contains(*c) || *c == '(')
                .collect();
            let rest = &raw[lead.len()..];
            let core_len = rest
                .trim_end_matches(|c: char| PUNCTS.contains(c) || c == ')')
                .len();
            let (core, trail) = rest.split_at(core_len);
            tokens.push((lead, core.to_string(), trail.to_string()));
        }
        let mut out: Vec<String> = vec![String::new(); tokens.len()];
        let mut next_vowel: Option<bool> = None;
        for (i, (lead, core, trail)) in tokens.iter().enumerate().rev() {
            let core = core.replace(['‘', '’'], "'");
            let ps = if core.is_empty() {
                String::new()
            } else {
                self.word(&core, next_vowel)
                    .or_else(|| {
                        (core.chars().count() > 1 && core == core.to_uppercase())
                            .then(|| self.letters(&core))
                            .flatten()
                    })
                    .unwrap_or_else(|| turkish(&core))
            };
            let punct_after: String = trail
                .chars()
                .filter(|c| PUNCTS.contains(*c) || "()".contains(*c))
                .collect();
            if !trail.is_empty() && trail.chars().any(|c| ".!?;:,—…".contains(c)) {
                next_vowel = None;
            } else if let Some(c) = ps.chars().find(|c| is_vowel(*c) || CONSONANTS.contains(*c)) {
                next_vowel = Some(is_vowel(c));
            }
            out[i] = format!(
                "{}{}{}",
                lead.chars()
                    .filter(|c| PUNCTS.contains(*c) || *c == '(')
                    .collect::<String>(),
                ps,
                punct_after
            );
        }
        let joined = out
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        joined.replace('ɾ', "T").replace('ʔ', "t")
    }
}

/// Turkish-only weight of a phoneme string (for the chunker): diphthongs count double.
pub fn weight(ps: &str) -> usize {
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::from_maps(
            &[
                ("the", "ðə"),
                ("goal", "ɡˈOl"),
                ("meeting", "mˈiɾɪŋ"),
                ("to", "tʊ"),
                ("agree", "əɡɹˈi"),
                ("plan", "plˈæn"),
                ("ship", "ʃˈɪp"),
                ("test", "tˈɛst"),
                ("update", "ˈʌpdˌAt"),
                ("A", "ˈA"),
                ("P", "pˈi"),
                ("I", "ˈI"),
                ("one", "wˈʌn"),
                ("two", "tˈu"),
                ("five", "fˈIv"),
                ("point", "pˈYnt"),
                ("percent", "pəɹsˈɛnt"),
                ("twenty", "twˈɛnti"),
                ("hundred", "hˈʌndɹəd"),
                ("apple", "ˈæpᵊl"),
                ("decide", "dəsˈId"),
            ],
            &[("friday", "fɹˈIdˌA")],
        )
    }

    #[test]
    fn dictionary_words_and_case() {
        let l = lex();
        assert_eq!(l.phonemize("The goal"), "ðə ɡˈOl");
        assert_eq!(l.phonemize("Friday."), "fɹˈIdˌA.");
        assert_eq!(l.phonemize("PLAN"), "plˈæn");
    }

    #[test]
    fn the_and_to_follow_the_next_sound() {
        let l = lex();
        assert_eq!(l.phonemize("the apple"), "ði ˈæpᵊl");
        assert_eq!(l.phonemize("to agree"), "tʊ əɡɹˈi");
        assert_eq!(l.phonemize("to ship"), "tə ʃˈɪp");
    }

    #[test]
    fn suffixes_come_from_the_stem() {
        let l = lex();
        assert_eq!(l.phonemize("tests"), "tˈɛsts");
        assert_eq!(l.phonemize("plans"), "plˈænz");
        assert_eq!(l.phonemize("decided"), "dəsˈIdᵻd");
        assert_eq!(l.phonemize("updated"), "ˈʌpdˌAɾᵻd".replace('ɾ', "T"));
        assert_eq!(l.phonemize("shipping"), "ʃˈɪpɪŋ");
    }

    #[test]
    fn numbers_are_spoken() {
        assert_eq!(number_words(0), "zero");
        assert_eq!(number_words(21), "twenty one");
        assert_eq!(
            number_words(1_250_000),
            "one million two hundred fifty thousand"
        );
        assert_eq!(spell_number("20%").unwrap(), "twenty percent");
        assert_eq!(spell_number("1.25").unwrap(), "one point two five");
        assert_eq!(spell_number("abc"), None);
        assert_eq!(lex().phonemize("20%"), "twˈɛnti pəɹsˈɛnt");
    }

    #[test]
    fn unknown_words_are_read_the_turkish_way() {
        assert_eq!(turkish("Ayşe"), "ɑjʃˈɛ");
        assert_eq!(turkish("Mehmet"), "mɛhmˈɛt");
        assert_eq!(turkish("Kadıköy"), "kɑdɯkˈøj");
        assert_eq!(lex().phonemize("Ayşe"), "ɑjʃˈɛ");
    }

    #[test]
    fn acronyms_and_tech_terms() {
        let l = lex();
        assert_eq!(l.phonemize("AP"), "ˌApˈi");
        assert_eq!(l.phonemize("Jira"), "ʤˈiɹə");
        assert_eq!(l.phonemize("Kubernetes"), "kˌubəɹnˈɛtiz");
    }

    #[test]
    fn stress_rules() {
        assert_eq!(apply_stress("plˈæn", Some(-2.0)), "plæn");
        assert_eq!(apply_stress("plˈæn", Some(-1.0)), "plˌæn");
        assert_eq!(apply_stress("plæn", Some(2.0)), "plˈæn");
        assert_eq!(apply_stress("plæn", Some(0.5)), "plˌæn");
    }

    /// Against misaki itself: `KOKORO_LEXICON=<dir with us_gold.json, us_silver.json,
    /// g2p_reference.json> cargo test --lib misaki_parity -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn misaki_parity() {
        let dir = std::env::var("KOKORO_LEXICON").expect("set KOKORO_LEXICON");
        let lex = Lexicon::load(Path::new(&dir)).unwrap();
        let cases: Vec<serde_json::Value> = serde_json::from_str(
            &std::fs::read_to_string(Path::new(&dir).join("g2p_reference.json")).unwrap(),
        )
        .unwrap();
        let (mut same, mut total) = (0, 0);
        for c in &cases {
            let ours = lex.phonemize(c["text"].as_str().unwrap());
            let theirs = c["ps"].as_str().unwrap();
            let (a, b): (Vec<&str>, Vec<&str>) =
                (ours.split(' ').collect(), theirs.split(' ').collect());
            for (x, y) in a.iter().zip(&b) {
                total += 1;
                if x == y {
                    same += 1;
                } else {
                    eprintln!("  {x}  vs misaki  {y}");
                }
            }
            if a.len() != b.len() {
                eprintln!(
                    "LEN {} vs {}: {ours}\n            {theirs}",
                    a.len(),
                    b.len()
                );
            }
        }
        eprintln!(
            "PARITY {same}/{total} words ({:.1}%)",
            100.0 * same as f32 / total as f32
        );
    }
}
