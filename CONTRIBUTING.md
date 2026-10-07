# Contributing

Thanks for helping! Most contributions here are **pronunciations**: a term
your team says differently, or one the voices get wrong. Code fixes and
speed-ups are welcome too.

*Türkçe özet:* Okunuş önerisi için kod yazmanız gerekmiyor;
[okunuş önerisi formunu](https://github.com/sewox/turkish-neural-tts/issues/new?template=pronunciation.yml)
doldurmanız yeterli. PR gönderecekseniz aşağıdaki adımları izleyin ve PR
şablonundaki listeyi doldurun.

## Not a coder? Open an issue

Use the [pronunciation suggestion form](https://github.com/sewox/turkish-neural-tts/issues/new?template=pronunciation.yml):
the term, how it is read now, how people say it, and in what context.

## Adding a pronunciation

### Turkish voice (`src/pronounce.rs`)

Pick the right list:

| The term is… | Add it to | Example |
|---|---|---|
| a word or brand name | `DEFAULT_LEXICON` (lowercase key) | `("docker", "dakır")` |
| an acronym with an unusual reading | `ACRONYM_LEXICON` (exact case) | `("SQL", "si ku el")` |
| a Turkish acronym (Turkish letter names) | `TURKISH_ACRONYMS` | `"KVKK"` → *ka ve ka ka* |
| an acronym said as a word | `WORD_ACRONYMS` | `"NATO"` |

Other acronyms already get English letter names (API → *ey pi ay*); add them
only if that reading is wrong.

Write the reading **in Turkish spelling, lowercase**, the way the word is
said in Turkish offices: `paytın`, not `pay-thon` or IPA.

### English voice (`src/english.rs`)

Only for words the misaki dictionaries do not know: add them to `TECH` as
Kokoro phonemes with stress marks, e.g. `("figma", "fˈɪɡmə")`. Unknown
Turkish names need nothing; they are already read the Turkish way.

### Rules for readings

- **Common use wins.** Add the reading most people use. When a term has two
  common readings (Azure: *ejır* / *azur*), say so in the PR; the other one
  can still be set per user through `Hints::lexicon`.
- **One term per line**, no duplicates across the lists (a test checks the
  lists, so search first).
- **No suffixes in the key.** Write `jira`, not `jira'da`; suffixes after an
  apostrophe are handled for you (Jira'da → *jirada*).
- **Add a test** next to the similar ones in the same file, in a short
  sentence:

  ```rust
  assert_eq!(p("Docker imajı"), "dakır imajı");
  ```

- **Update [docs/pronunciation.md](docs/pronunciation.md)** so the table
  matches the code.

## Before you open the PR

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

CI runs the same on Linux, macOS and Windows; a PR is merged only when it is
green.

If you can, listen to the result:

```sh
cargo run --release --example say -- tr models/ema-lightning out.wav "Docker imajı hazır."
```

## Code changes

- Keep changes small and focused; one topic per PR.
- Match the surrounding style: short doc comments that say *why*, plain
  names, no new dependencies without a reason in the PR.
- Changes to the ONNX graphs or `export/ema_to_onnx.py` must keep the export
  reproducible and say how the output was checked against PyTorch.
- `tests` cover behavior, not implementation: a reading, a timeline, a WAV
  header.

## Commit messages

Short, imperative summary line, in English:

```
pronounce: read "Kubernetes" as kubernetis
english: add "figma" to the tech words
```

## License

By contributing you agree that your contribution is licensed under
Apache-2.0, like the rest of the repository.
