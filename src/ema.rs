//! EMA Lightning (Apache-2.0, 8.6M parameters): Turkish text in, 48 kHz audio out.
//!
//! Three ONNX graphs exported from the PyTorch model (see
//! `export/ema_to_onnx.py`): the text stage (letters → features and durations),
//! the sound stage (four flow steps to 25 Hz latents) and the decoder
//! (latents → audio). The bookkeeping between them, the word/frame timeline,
//! is done here; it is a port of `ema_lightning/engine.py` and the timeline
//! part of `model.py`.

use crate::frontend::{self, symbol_id};
use crate::pronounce::Hints;
use ndarray::{Array2, Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub const SAMPLE_RATE: u32 = 48_000;
const HOP: usize = 1920;
const LATENT: usize = 64;
const STEPS: usize = 4;
const MAX_WORD_FRAMES: f32 = 250.0;
const MAX_FRAMES: usize = 3000;
pub const FILES: [&str; 3] = ["text.onnx", "sound.onnx", "decoder.onnx"];

pub struct Ema {
    text: Session,
    sound: Session,
    decoder: Session,
}

/// Word/frame timeline of one piece of text.
#[derive(Debug, Clone, PartialEq)]
struct Timeline {
    /// Word of each letter, and the first letter of that word.
    cw: Vec<usize>,
    wstart: Vec<usize>,
    /// Word of each frame and the position of the frame inside its word (0..1).
    fw: Vec<usize>,
    fp: Vec<f32>,
}

fn words(text: &[char]) -> (Vec<usize>, Vec<usize>) {
    let mut starts: Vec<usize> = (0..text.len())
        .filter(|&i| text[i] != ' ' && (i == 0 || text[i - 1] == ' '))
        .collect();
    if starts.is_empty() {
        starts.push(0);
    }
    let mut bounds = vec![0];
    bounds.extend_from_slice(&starts[1..]);
    bounds.push(text.len());
    let (mut cw, mut wstart) = (Vec::new(), Vec::new());
    for (w, pair) in bounds.windows(2).enumerate() {
        for _ in pair[0]..pair[1] {
            cw.push(w);
            wstart.push(pair[0]);
        }
    }
    (cw, wstart)
}

fn timeline(text: &[char], dur: &[f32]) -> Timeline {
    let (cw, wstart) = words(text);
    let n_words = cw.last().map_or(1, |w| w + 1);
    let mut sums = vec![0f32; n_words];
    for (l, d) in dur.iter().enumerate() {
        sums[cw[l]] += d;
    }
    let counts: Vec<usize> = sums
        .iter()
        .map(|s| s.round_ties_even().clamp(1.0, MAX_WORD_FRAMES) as usize)
        .collect();
    let frames = counts.iter().sum::<usize>().min(MAX_FRAMES);
    let (mut fw, mut fp) = (Vec::with_capacity(frames), Vec::with_capacity(frames));
    'outer: for (w, &n) in counts.iter().enumerate() {
        for k in 0..n {
            if fw.len() == frames {
                break 'outer;
            }
            fw.push(w);
            fp.push((k as f64 / n as f64) as f32);
        }
    }
    Timeline { cw, wstart, fw, fp }
}

/// Inputs of the sound stage computed from the timeline (model.py `sound_stage`
/// and the aligner's `letter_pos` positions).
fn positions(t: &Timeline, dur: &[f32]) -> (Vec<f32>, Vec<f32>, Array3<f32>) {
    let l = dur.len();
    let n_words = t.cw.last().map_or(1, |w| w + 1);
    let c: Vec<f32> = dur.iter().map(|d| d.max(1e-4)).collect();
    let mut done = vec![0f32; l];
    let mut acc = 0f32;
    for i in 0..l {
        acc += c[i];
        done[i] = acc;
    }
    let before: Vec<f32> = (0..l).map(|i| done[i] - c[i]).collect();
    let mut total = vec![0f32; n_words];
    let mut wlen = vec![0f32; n_words];
    for i in 0..l {
        total[t.cw[i]] += c[i];
        wlen[t.cw[i]] += 1.0;
    }
    let wlen: Vec<f32> = wlen.iter().map(|v| v.max(1.0)).collect();
    let mut woff = vec![0f32; n_words];
    let mut run = 0f32;
    for w in 0..n_words {
        woff[w] = run;
        run += wlen[w];
    }
    let cg: Vec<f32> = (0..l)
        .map(|i| {
            let w = t.cw[i];
            let cp =
                ((done[i] - before[t.wstart[i]] - 0.5 * c[i]) / total[w].max(1e-8)).clamp(0.0, 1.0);
            woff[w] + cp * wlen[w]
        })
        .collect();
    let fg: Vec<f32> =
        t.fw.iter()
            .zip(&t.fp)
            .map(|(&w, &p)| woff[w] + p * wlen[w])
            .collect();
    let frames = t.fw.len();
    let allow = Array3::from_shape_fn((1, frames, l), |(_, f, i)| {
        let rel = t.cw[i] as i64 - t.fw[f] as i64;
        if (-1..=1).contains(&rel) {
            1.0
        } else {
            0.0
        }
    });
    (cg, fg, allow)
}

/// Seeded standard-normal noise (xoshiro256++ and Box–Muller): the same text
/// and seed always give the same audio.
struct Noise([u64; 4]);

impl Noise {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Noise([next(), next(), next(), next()])
    }
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal_pair(&mut self) -> (f32, f32) {
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        ((r * a.cos()) as f32, (r * a.sin()) as f32)
    }
    fn tensor(&mut self, frames: usize) -> Array4<f32> {
        let n = STEPS * frames * LATENT;
        let mut v = Vec::with_capacity(n + 1);
        while v.len() < n {
            let (a, b) = self.normal_pair();
            v.push(a);
            v.push(b);
        }
        v.truncate(n);
        Array4::from_shape_vec((1, STEPS, frames, LATENT), v).expect("noise shape")
    }
}

fn ort_err(e: impl std::fmt::Display) -> String {
    format!("EMA: {e}")
}

impl Ema {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let open = |name: &str| -> Result<Session, String> {
            let threads = std::thread::available_parallelism()
                .map_or(4, |n| n.get())
                .min(8);
            Session::builder()
                .map_err(ort_err)?
                .with_intra_threads(threads)
                .map_err(ort_err)?
                .commit_from_file(dir.join(name))
                .map_err(ort_err)
        };
        Ok(Ema {
            text: open(FILES[0])?,
            sound: open(FILES[1])?,
            decoder: open(FILES[2])?,
        })
    }

    /// Audio for one piece already in the model's alphabet.
    fn piece(&mut self, text: &str, speed: f32, seed: u64) -> Result<Vec<f32>, String> {
        let chars: Vec<char> = text.chars().collect();
        let l = chars.len();
        if l == 0 {
            return Ok(Vec::new());
        }
        let ids = Array2::from_shape_vec((1, l), chars.iter().map(|&c| symbol_id(c)).collect())
            .map_err(ort_err)?;
        let out = self
            .text
            .run(ort::inputs![
                "ids" => Tensor::from_array(ids).map_err(ort_err)?,
                "mask" => Tensor::from_array(Array2::<f32>::ones((1, l))).map_err(ort_err)?
            ])
            .map_err(ort_err)?;
        let (_, h_data) = out["h"].try_extract_tensor::<f32>().map_err(ort_err)?;
        let h =
            Array3::from_shape_vec((1, l, h_data.len() / l), h_data.to_vec()).map_err(ort_err)?;
        let (_, dur) = out["dur"].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dur: Vec<f32> = dur.iter().map(|d| d / speed).collect();
        drop(out);

        let t = timeline(&chars, &dur);
        let frames = t.fw.len();
        let (cg, fg, allow) = positions(&t, &dur);
        let out = self
            .sound
            .run(ort::inputs![
                "h" => Tensor::from_array(h).map_err(ort_err)?,
                "cg" => Tensor::from_array(Array2::from_shape_vec((1, l), cg).map_err(ort_err)?).map_err(ort_err)?,
                "fg" => Tensor::from_array(Array2::from_shape_vec((1, frames), fg).map_err(ort_err)?).map_err(ort_err)?,
                "allow" => Tensor::from_array(allow).map_err(ort_err)?,
                "fmask" => Tensor::from_array(Array2::<f32>::ones((1, frames))).map_err(ort_err)?,
                "noise" => Tensor::from_array(Noise::new(seed).tensor(frames)).map_err(ort_err)?,
                "fpos" => Tensor::from_array(Array2::from_shape_fn((1, frames), |(_, i)| i as f32)).map_err(ort_err)?
            ])
            .map_err(ort_err)?;
        let (_, lat) = out["latents"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let z = Array3::from_shape_vec((1, frames, LATENT), lat.to_vec())
            .map_err(ort_err)?
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        drop(out);

        let out = self
            .decoder
            .run(ort::inputs!["z" => Tensor::from_array(z).map_err(ort_err)?])
            .map_err(ort_err)?;
        let (_, audio) = out["audio"].try_extract_tensor::<f32>().map_err(ort_err)?;
        Ok(audio.iter().take(frames * HOP).copied().collect())
    }

    /// Speech for written Turkish: pronunciations, numbers, pieces, audio.
    /// `on_progress` gets 0..1 per finished piece.
    pub fn synthesize(
        &mut self,
        text: &str,
        hints: &Hints,
        speed: f32,
        cancel: Option<&AtomicBool>,
        on_progress: &dyn Fn(f32),
    ) -> Result<Vec<f32>, String> {
        let speed = if speed.is_finite() {
            speed.clamp(0.25, 4.0)
        } else {
            1.0
        };
        let model_text = frontend::prepare(text, hints);
        let pieces = frontend::chunk(&model_text, speed);
        let mut audio = Vec::new();
        for (i, (piece, pause)) in pieces.iter().enumerate() {
            if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                return Err("cancelled".into());
            }
            audio.extend(self.piece(piece, speed, 0)?);
            audio.extend(std::iter::repeat_n(
                0.0,
                (pause * SAMPLE_RATE as f32) as usize,
            ));
            on_progress((i + 1) as f32 / pieces.len() as f32);
        }
        Ok(audio)
    }
}

/// Whether all model files are in `dir`.
pub fn installed(dir: &Path) -> bool {
    FILES.iter().all(|f| dir.join(f).is_file())
}

/// 16-bit PCM WAV of mono 48 kHz samples.
pub fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    wav_bytes_at(samples, SAMPLE_RATE)
}

/// 16-bit PCM WAV of mono samples at `rate`.
pub fn wav_bytes_at(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_follow_the_reference_engine() {
        // "ab cd": a word's trailing space belongs to it, like engine.py.
        let (cw, wstart) = words(&"ab cd".chars().collect::<Vec<_>>());
        assert_eq!(cw, [0, 0, 0, 1, 1]);
        assert_eq!(wstart, [0, 0, 0, 3, 3]);
    }

    #[test]
    fn timeline_matches_the_python_engine() {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../resources/timeline_ref.json")).unwrap();
        let text: Vec<char> = reference["text"].as_str().unwrap().chars().collect();
        let dur: Vec<f32> = reference["dur"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let t = timeline(&text, &dur);
        let ints = |k: &str| -> Vec<usize> {
            reference[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        assert_eq!(t.cw, ints("cw"));
        assert_eq!(t.wstart, ints("wstart"));
        assert_eq!(t.fw, ints("fw"));
        let fp: Vec<f32> = reference["fp"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert!(t.fp.iter().zip(&fp).all(|(a, b)| (a - b).abs() < 1e-5));
        let ids: Vec<i64> = text.iter().map(|&c| symbol_id(c)).collect();
        assert_eq!(
            ids,
            reference["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn noise_is_seeded_and_standard_normal() {
        let a = Noise::new(0).tensor(100);
        let b = Noise::new(0).tensor(100);
        assert_eq!(a, b);
        assert_ne!(a, Noise::new(1).tensor(100));
        let n = a.len() as f32;
        let mean = a.sum() / n;
        let var = a.mapv(|x| (x - mean).powi(2)).sum() / n;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "mean {mean} var {var}"
        );
    }

    #[test]
    fn wav_header_is_well_formed() {
        let w = wav_bytes(&[0.0, 1.0, -1.0]);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(w.len(), 44 + 6);
        assert_eq!(i16::from_le_bytes([w[46], w[47]]), 32767);
    }

    /// Real model: `ECHOMIND_EMA_DIR=<dir with the .onnx files> cargo test --lib ema_real -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn ema_real_model() {
        let dir = std::env::var("ECHOMIND_EMA_DIR").expect("set ECHOMIND_EMA_DIR");
        let mut ema = Ema::load(Path::new(&dir)).unwrap();
        let text = "Toplantının amacı yeni sürümün planını belirlemekti. KVKK uyumu ve SonicWall yapılandırması, \
                    API'ye geçiş ve Jira'daki deadline konuşuldu. Bütçe %20 arttı.";
        let hints = Hints {
            foreign: ["deadline".into()].into(),
            ..Default::default()
        };
        ema.synthesize("Isınma.", &hints, 1.0, None, &|_| {})
            .unwrap();
        let t = std::time::Instant::now();
        let audio = ema.synthesize(text, &hints, 1.0, None, &|_| {}).unwrap();
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        eprintln!("{secs:.1}s audio in {:.2}s", t.elapsed().as_secs_f32());
        std::fs::write(format!("{dir}/rust_sample.wav"), wav_bytes(&audio)).unwrap();
        assert!(secs > 5.0);
    }
}
