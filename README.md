# turkish-neural-tts

Offline neural text to speech for **Turkish and English**, in Rust, on ONNX
Runtime. No Python, no espeak, no network once the model files are on disk.

- **Turkish:** [EMA Lightning](https://github.com/canberk7/ema-lightning)
  (8.6M parameters, 48 kHz), exported to three ONNX graphs. The word/frame
  timeline between them is computed in Rust.
- **English:** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
  (8-bit/fp16 build, 86 MB, 24 kHz) with a Rust port of
  [misaki](https://github.com/hexgrad/misaki)'s dictionary G2P.
- **Pronunciation rules for real Turkish workplace text:** English terms,
  brand names and acronyms read the way people say them (Python → *paytın*,
  KVKK → *ka ve ka ka*, API'ye → *ey pi ayye*), and Turkish names read
  correctly by the English voice. See [docs/pronunciation.md](docs/pronunciation.md).

Built for [EchoMind](https://github.com/sewox/EchoMind), the meeting
assistant, where it reads meeting briefings aloud. Shared here so others can
use it.

## Speed

On an Apple M1 Pro, CPU only:

| Voice | Speed |
|---|---|
| EMA Lightning (Turkish) | about 20× faster than real time |
| Kokoro-82M q8f16 (English) | about 1.8× faster than real time |

## Model files

**Turkish (EMA Lightning, ONNX):** from the [v0.1.0 release](https://github.com/sewox/turkish-neural-tts/releases/tag/v0.1.0).
Put them in one folder.

| File | Size | SHA-256 |
|---|---|---|
| `text.onnx` | 4,804,278 | `b832ac6d0a54f822a80765797053b315b31d73f8cfbdaf5347dd1865ff3c47a3` |
| `sound.onnx` | 17,927,687 | `679e6c0b35c8407b4c791031aab91383fde03690873f358b1813eee9951d881c` |
| `decoder.onnx` | 12,031,945 | `396c06a8e5711fd40bfd2c3f4f33851feb2ad07a86485833844e0ae8ff637b17` |

They are exported from the EMA Lightning weights at Hugging Face revision
`7a6ba1ad216bb2f1da9863f80ac8770a6a807632` by
[`export/ema_to_onnx.py`](export/ema_to_onnx.py). The export is reproducible:
run it with [`export/requirements.txt`](export/requirements.txt) and you get
the same bytes. The script checks the weights' SHA-256 first and compares every
graph with the PyTorch model, through PyTorch and through ONNX Runtime.

**English (Kokoro-82M):** in one folder, from
[onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
and [misaki](https://github.com/hexgrad/misaki/tree/main/misaki/data):

```
model_q8f16.onnx      (onnx/model_q8f16.onnx)
voices/af_heart.bin   (American)
voices/bf_emma.bin    (British)
us_gold.json
us_silver.json
```

## Use

```toml
[dependencies]
turkish-neural-tts = { git = "https://github.com/sewox/turkish-neural-tts" }
```

```rust
use std::path::Path;
use turkish_neural_tts::{ema, kokoro, pronounce::Hints};

// Turkish
let mut tr = ema::Ema::load(Path::new("models/ema-lightning"))?;
let hints = Hints {
    // Your own readings, e.g. from a glossary.
    lexicon: vec![("SonicWall".into(), "Sonik Vol".into())],
    // Words to read as English (e.g. marked by a language model).
    foreign: ["deadline".to_string()].into(),
};
let audio = tr.synthesize("Deadline cuma, KVKK raporu API'ye eklenecek.", &hints, 1.0, None, &|p| {
    eprintln!("{:.0}%", p * 100.0)
})?;
std::fs::write("tr.wav", ema::wav_bytes(&audio)).unwrap();

// English
let mut en = kokoro::Kokoro::load(Path::new("models/kokoro"))?;
let audio = en.synthesize("Ayşe will update the Jira ticket.", "af_heart", 1.0, &|_| {})?;
std::fs::write("en.wav", ema::wav_bytes_at(&audio, kokoro::SAMPLE_RATE)).unwrap();
```

Or from the command line:

```sh
cargo run --release --example say -- tr models/ema-lightning out.wav "Toplantı cuma günü saat 14:30'da."
cargo run --release --example say -- en models/kokoro out.wav "The release ships on Friday."
```

`ort` downloads a matching ONNX Runtime build at compile time
(`download-binaries`); it is linked statically, so nothing has to be
installed on the user's machine.

## How the Turkish voice is run

`export/ema_to_onnx.py` splits EMA Lightning into graphs that only use plain
tensor ops:

| Graph | Inputs | Outputs |
|---|---|---|
| `text.onnx` | `ids[1,L]` i64, `mask[1,L]` | `h[1,L,224]`, `dur[1,L]` |
| `sound.onnx` | `h`, `cg[1,L]`, `fg[1,T]`, `allow[1,T,L]`, `fmask[1,T]`, `noise[1,4,T,64]`, `fpos[1,T]` | `latents[1,T,64]` |
| `decoder.onnx` | `z[1,64,T]` | `audio[1,T·1920]` |

Changes from the PyTorch model, all checked to give the same output:
`expm1` is written as `exp − 1`; the aligner's learned frame query is
broadcast instead of `expand`ed; frame positions come in as `fpos` instead of
a `Range` op. The word of each letter and frame, positions, the attention
window and the noise (xoshiro256++ with Box–Muller) are computed in
[`src/ema.rs`](src/ema.rs), a port of the reference engine. A test checks the
timeline against the Python engine's output.

## Türkçe

Türkçe ve İngilizce için çevrimdışı, doğal sesli metin okuma. Rust ile
yazıldı, ONNX Runtime üzerinde çalışıyor; Python ya da espeak gerektirmiyor.

- **Türkçe:** [Canberk Aslan](https://github.com/canberk7)'ın
  [EMA Lightning](https://github.com/canberk7/ema-lightning) modeli, ONNX'e
  aktarılmış hâliyle. Sayı, tarih ve semboller
  [Erdem Tuna](https://github.com/erdemtuna)'nın
  [normalizer-tr](https://github.com/erdemtuna/normalizer-tr) kütüphanesiyle okunur.
- **İngilizce:** [hexgrad](https://github.com/hexgrad)'ın
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) modeli ve
  [misaki](https://github.com/hexgrad/misaki) sözlükleri; Türkçe isimleri de doğru okur.
- **Okunuş kuralları:** iş yerinde kullanılan İngilizce terimler, marka adları
  ve kısaltmalar alışılmış biçimde okunur: Python → *paytın*, Docker → *dakır*,
  KVKK → *ka ve ka ka*, SQL → *si ku el*. Kendi okunuşlarınızı da
  ekleyebilirsiniz. Ayrıntılar: [docs/pronunciation.md](docs/pronunciation.md).

Daha iyi okunuş önerilerinizi pull request olarak bekliyoruz.

## Thanks and licenses

This repository is Apache-2.0 ([LICENSE](LICENSE)). It stands on:

- **[EMA Lightning](https://github.com/canberk7/ema-lightning)** by
  [Canberk Aslan](https://github.com/canberk7), Apache-2.0
  ([model on Hugging Face](https://huggingface.co/canberkkkkkk/ema-lightning)):
  the Turkish model; the ONNX files are an export of its weights, and the Rust
  engine ports parts of its reference engine.
- **[normalizer-tr](https://github.com/erdemtuna/normalizer-tr)** by
  [Erdem Tuna](https://github.com/erdemtuna), Apache-2.0: Turkish numbers,
  dates and symbols in spoken form.
- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by
  [hexgrad](https://github.com/hexgrad), Apache-2.0: the English model
  ([ONNX build](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
  by onnx-community).
- **[misaki](https://github.com/hexgrad/misaki)** by
  [hexgrad](https://github.com/hexgrad), Apache-2.0: the English G2P ported
  here.
- **[CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)**,
  Carnegie Mellon University, BSD 2-Clause: English words respelled for the
  Turkish voice ([resources/CMUDICT_LICENSE](resources/CMUDICT_LICENSE)).
- **[ONNX Runtime](https://onnxruntime.ai)** through the
  [`ort`](https://github.com/pykeio/ort) crate.

See [NOTICE](NOTICE).
