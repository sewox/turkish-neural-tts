//! Offline neural text to speech for Turkish and English, in-process on ONNX
//! Runtime. No Python, no espeak, no network once the model files are on disk.
//!
//! - [`ema`]: EMA Lightning (Turkish, 48 kHz) from three ONNX graphs; the
//!   word/frame timeline between them is computed here.
//! - [`kokoro`]: Kokoro-82M (English, 24 kHz) with a Rust port of misaki's
//!   dictionary G2P ([`english`]).
//! - [`pronounce`] and [`frontend`]: how the Turkish voice says English
//!   words, brand names and acronyms (Python → paytın, KVKK → ka ve ka ka),
//!   then normalization with normalizer-tr.
//!
//! ```no_run
//! use std::path::Path;
//! use turkish_neural_tts::{ema, pronounce::Hints};
//!
//! let mut voice = ema::Ema::load(Path::new("models/ema-lightning"))?;
//! let audio = voice.synthesize("Toplantı cuma günü 14:30'da.", &Hints::default(), 1.0, None, &|_| {})?;
//! std::fs::write("out.wav", ema::wav_bytes(&audio)).map_err(|e| e.to_string())?;
//! # Ok::<(), String>(())
//! ```

pub mod ema;
pub mod english;
pub mod frontend;
pub mod kokoro;
pub mod pronounce;
