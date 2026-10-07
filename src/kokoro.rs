//! Kokoro-82M (Apache-2.0) for English: phonemes in, 24 kHz audio out.
//!
//! One ONNX graph (the 8-bit/fp16 build, 86 MB) plus a 256-dim style vector
//! per voice, chosen by the length of the input (`voices/<name>.bin`,
//! 510 × 256 floats). Phonemes come from english.rs, no espeak.

use crate::english::Lexicon;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: u32 = 24_000;
pub const MODEL_FILE: &str = "model_q8f16.onnx";
/// The voices offered (American and British female; liked in the trial).
pub const VOICES: [&str; 2] = ["af_heart", "bf_emma"];
pub const LEXICON_FILES: [&str; 2] = ["us_gold.json", "us_silver.json"];
/// Model limit per call, without the two padding tokens.
const MAX_TOKENS: usize = 510;
const STYLE: usize = 256;
const SENTENCE_PAUSE: f32 = 0.2;

const VOCAB: [(char, i64); 115] = [
    ('$', 0),
    (';', 1),
    (':', 2),
    (',', 3),
    ('.', 4),
    ('!', 5),
    ('?', 6),
    ('—', 9),
    ('…', 10),
    ('"', 11),
    ('(', 12),
    (')', 13),
    ('“', 14),
    ('”', 15),
    (' ', 16),
    ('̃', 17),
    ('ʣ', 18),
    ('ʥ', 19),
    ('ʦ', 20),
    ('ʨ', 21),
    ('ᵝ', 22),
    ('ꭧ', 23),
    ('A', 24),
    ('I', 25),
    ('O', 31),
    ('Q', 33),
    ('S', 35),
    ('T', 36),
    ('W', 39),
    ('Y', 41),
    ('ᵊ', 42),
    ('a', 43),
    ('b', 44),
    ('c', 45),
    ('d', 46),
    ('e', 47),
    ('f', 48),
    ('h', 50),
    ('i', 51),
    ('j', 52),
    ('k', 53),
    ('l', 54),
    ('m', 55),
    ('n', 56),
    ('o', 57),
    ('p', 58),
    ('q', 59),
    ('r', 60),
    ('s', 61),
    ('t', 62),
    ('u', 63),
    ('v', 64),
    ('w', 65),
    ('x', 66),
    ('y', 67),
    ('z', 68),
    ('ɑ', 69),
    ('ɐ', 70),
    ('ɒ', 71),
    ('æ', 72),
    ('β', 75),
    ('ɔ', 76),
    ('ɕ', 77),
    ('ç', 78),
    ('ɖ', 80),
    ('ð', 81),
    ('ʤ', 82),
    ('ə', 83),
    ('ɚ', 85),
    ('ɛ', 86),
    ('ɜ', 87),
    ('ɟ', 90),
    ('ɡ', 92),
    ('ɥ', 99),
    ('ɨ', 101),
    ('ɪ', 102),
    ('ʝ', 103),
    ('ɯ', 110),
    ('ɰ', 111),
    ('ŋ', 112),
    ('ɳ', 113),
    ('ɲ', 114),
    ('ɴ', 115),
    ('ø', 116),
    ('ɸ', 118),
    ('θ', 119),
    ('œ', 120),
    ('ɹ', 123),
    ('ɾ', 125),
    ('ɻ', 126),
    ('ʁ', 128),
    ('ɽ', 129),
    ('ʂ', 130),
    ('ʃ', 131),
    ('ʈ', 132),
    ('ʧ', 133),
    ('ʊ', 135),
    ('ʋ', 136),
    ('ʌ', 138),
    ('ɣ', 139),
    ('ɤ', 140),
    ('χ', 142),
    ('ʎ', 143),
    ('ʒ', 147),
    ('ʔ', 148),
    ('ˈ', 156),
    ('ˌ', 157),
    ('ː', 158),
    ('ʰ', 162),
    ('ʲ', 164),
    ('↓', 169),
    ('→', 171),
    ('↗', 172),
    ('↘', 173),
    ('ᵻ', 177),
];

fn token_id(c: char) -> Option<i64> {
    VOCAB.iter().find(|(v, _)| *v == c).map(|(_, id)| *id)
}

/// Phonemes → model ids, dropping symbols the model does not know.
fn ids(ps: &str) -> Vec<i64> {
    ps.chars().filter_map(token_id).collect()
}

/// Splits ids at spaces/punctuation so no piece exceeds the model limit.
fn pieces(ids: &[i64]) -> Vec<Vec<i64>> {
    let space = token_id(' ').unwrap_or(16);
    let mut out = Vec::new();
    let mut rest = ids;
    while rest.len() > MAX_TOKENS {
        let cut = rest[..MAX_TOKENS]
            .iter()
            .rposition(|&t| t == space)
            .map_or(MAX_TOKENS, |i| i + 1);
        out.push(rest[..cut].to_vec());
        rest = &rest[cut..];
    }
    if !rest.is_empty() {
        out.push(rest.to_vec());
    }
    out
}

/// Sentences of `text` (cut after . ! ? and kept with their punctuation).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        if ".!?".contains(c) && chars.get(i + 1).is_none_or(|n| n.is_whitespace()) {
            out.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub struct Kokoro {
    session: Session,
    lexicon: Lexicon,
    dir: PathBuf,
    voices: HashMap<String, Vec<f32>>,
}

fn ort_err(e: impl std::fmt::Display) -> String {
    format!("Kokoro: {e}")
}

/// Model, dictionaries and at least one voice are in `dir`.
pub fn installed(dir: &Path) -> bool {
    dir.join(MODEL_FILE).is_file()
        && LEXICON_FILES.iter().all(|f| dir.join(f).is_file())
        && VOICES.iter().any(|v| voice_path(dir, v).is_file())
}

pub fn voice_path(dir: &Path, voice: &str) -> PathBuf {
    dir.join("voices").join(format!("{voice}.bin"))
}

impl Kokoro {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(8);
        let session = Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?
            .commit_from_file(dir.join(MODEL_FILE))
            .map_err(ort_err)?;
        Ok(Kokoro {
            session,
            lexicon: Lexicon::load(dir)?,
            dir: dir.to_path_buf(),
            voices: HashMap::new(),
        })
    }

    fn style(&mut self, voice: &str, len: usize) -> Result<Vec<f32>, String> {
        if !self.voices.contains_key(voice) {
            let bytes =
                std::fs::read(voice_path(&self.dir, voice)).map_err(|e| format!("{voice}: {e}"))?;
            let floats: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if floats.len() < STYLE {
                return Err(format!("{voice}: bad voice file"));
            }
            self.voices.insert(voice.to_string(), floats);
        }
        let v = &self.voices[voice];
        let rows = v.len() / STYLE;
        let row = len.min(rows - 1);
        Ok(v[row * STYLE..(row + 1) * STYLE].to_vec())
    }

    fn run(&mut self, ids: &[i64], voice: &str, speed: f32) -> Result<Vec<f32>, String> {
        let style = self.style(voice, ids.len())?;
        let mut input = Vec::with_capacity(ids.len() + 2);
        input.push(0);
        input.extend_from_slice(ids);
        input.push(0);
        let n = input.len();
        let out = self
            .session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, n), input).map_err(ort_err)?).map_err(ort_err)?,
                "style" => Tensor::from_array(Array2::from_shape_vec((1, STYLE), style).map_err(ort_err)?).map_err(ort_err)?,
                "speed" => Tensor::from_array(ndarray::Array1::from_vec(vec![speed])).map_err(ort_err)?
            ])
            .map_err(ort_err)?;
        let (_, wav) = out["waveform"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        Ok(wav.to_vec())
    }

    /// Speech for English text with `voice` (one of [`VOICES`]).
    /// `on_progress` gets 0..1 per finished sentence, weighted by its length.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        on_progress: &dyn Fn(f32),
    ) -> Result<Vec<f32>, String> {
        let speed = if speed.is_finite() {
            speed.clamp(0.5, 2.0)
        } else {
            1.0
        };
        let list = sentences(text);
        let total = list.iter().map(String::len).sum::<usize>().max(1) as f32;
        let mut done = 0;
        let mut audio = Vec::new();
        for sentence in list {
            let ps = self.lexicon.phonemize(&sentence);
            for piece in pieces(&ids(&ps)) {
                audio.extend(self.run(&piece, voice, speed)?);
            }
            audio.extend(std::iter::repeat_n(
                0.0,
                (SENTENCE_PAUSE * SAMPLE_RATE as f32) as usize,
            ));
            done += sentence.len();
            on_progress(done as f32 / total);
        }
        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_maps_phonemes() {
        assert_eq!(token_id('$'), Some(0));
        assert_eq!(token_id(' '), Some(16));
        assert_eq!(token_id('ᵻ'), Some(177));
        assert_eq!(ids("a?"), vec![token_id('a').unwrap(), 6]);
        assert_eq!(ids("€"), Vec::<i64>::new());
    }

    #[test]
    fn long_input_is_cut_at_spaces() {
        let word = ids("ab ");
        let long: Vec<i64> = word.iter().cycle().take(1200).copied().collect();
        let parts = pieces(&long);
        assert!(parts.len() >= 3);
        assert!(parts.iter().all(|p| p.len() <= MAX_TOKENS));
        assert_eq!(parts.iter().map(Vec::len).sum::<usize>(), 1200);
        assert_eq!(*parts[0].last().unwrap(), token_id(' ').unwrap());
    }

    #[test]
    fn sentences_keep_their_punctuation() {
        assert_eq!(sentences("One. Two? Three"), ["One.", "Two?", "Three"]);
        assert_eq!(sentences("Version 2.5 ships."), ["Version 2.5 ships."]);
    }

    #[test]
    fn nothing_installed_in_an_empty_dir() {
        assert!(!installed(&std::env::temp_dir().join("no-kokoro-here")));
    }

    /// Real model: `KOKORO_DIR=<dir> cargo test --lib kokoro_real -- --ignored --nocapture`
    /// (dir: model_q8f16.onnx, us_gold.json, us_silver.json, voices/af_heart.bin).
    #[test]
    #[ignore]
    fn kokoro_real_model() {
        let dir = std::env::var("KOKORO_DIR").expect("set KOKORO_DIR");
        let mut k = Kokoro::load(Path::new(&dir)).unwrap();
        k.synthesize("Warm up.", "af_heart", 1.0, &|_| {}).unwrap();
        let text = "The goal of the meeting was to agree on the release plan. Ayşe will update the Jira \
                    ticket and share the notes with Mehmet before the deadline. Revenue grew 20% this quarter.";
        let t = std::time::Instant::now();
        let audio = k.synthesize(text, "af_heart", 1.0, &|_| {}).unwrap();
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        eprintln!("{secs:.1}s audio in {:.2}s", t.elapsed().as_secs_f32());
        std::fs::write(
            format!("{dir}/rust_kokoro.wav"),
            crate::ema::wav_bytes_at(&audio, SAMPLE_RATE),
        )
        .unwrap();
        assert!(secs > 5.0);
    }
}
