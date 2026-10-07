//! Text to a WAV file.
//!
//! cargo run --release --example say -- tr models/ema-lightning out.wav "Toplantı cuma günü."
//! cargo run --release --example say -- en models/kokoro out.wav "The release ships on Friday."

use std::path::Path;
use turkish_neural_tts::{ema, kokoro, pronounce::Hints};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [lang, dir, out, text] = args.as_slice() else {
        return Err("usage: say <tr|en> <model dir> <out.wav> <text>".into());
    };
    let started = std::time::Instant::now();
    let wav = match lang.as_str() {
        "tr" => {
            let audio = ema::Ema::load(Path::new(dir))?.synthesize(
                text,
                &Hints::default(),
                1.0,
                None,
                &|_| {},
            )?;
            ema::wav_bytes(&audio)
        }
        "en" => {
            let audio = kokoro::Kokoro::load(Path::new(dir))?.synthesize(
                text,
                kokoro::VOICES[0],
                1.0,
                &|_| {},
            )?;
            ema::wav_bytes_at(&audio, kokoro::SAMPLE_RATE)
        }
        _ => return Err(format!("unknown language: {lang}")),
    };
    std::fs::write(out, wav).map_err(|e| e.to_string())?;
    eprintln!("{out} in {:.2}s", started.elapsed().as_secs_f32());
    Ok(())
}
