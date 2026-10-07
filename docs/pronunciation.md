# Pronunciation rules

Both voices meet the same kind of text: Turkish workplace language full of
English terms, brand names, acronyms and Turkish names. These are the rules
that make them read it the way people say it. They live in
[`src/pronounce.rs`](../src/pronounce.rs) (Turkish voice) and
[`src/english.rs`](../src/english.rs) (English voice); pull requests with
better readings are welcome.

## Turkish voice (EMA Lightning)

EMA reads every letter the Turkish way. Before normalization, each word that
is not Turkish is rewritten in Turkish spelling, in this order:

1. **The caller's own readings** (`Hints::lexicon`), e.g. `SonicWall` → `Sonik Vol`.
   A team that says SQL as "es kyu el" adds it here.
2. **Built-in terms** (below), matched without case.
3. **Acronyms** (two or more capitals):
   - Turkish acronyms take Turkish letter names: KVKK → *ka ve ka ka*.
   - Acronyms said as a word stay as written: NATO, TÜBİTAK.
   - All others take English letter names: API → *ey pi ay*, PDF → *pi di ef*,
     KPI → *key pi ay*.
4. **English words** marked as foreign (`Hints::foreign`, e.g. by a language
   model), written in CamelCase, or spelled with w/q/x, from the CMU
   Pronouncing Dictionary respelled in Turkish: deadline → *dedlayn*.

A Turkish suffix after an apostrophe stays on the rewritten word
(API'ye → *ey pi ayye*, Jira'da → *jirada*), so the apostrophe is never read
aloud. A word is only treated as English when its spelling could be English
(`plausibly_english`), so Turkish words such as *maliyet* are never rewritten.

### Built-in terms

| Written | Said |
|---|---|
| jira | jira |
| java | cava |
| javascript | cava skript |
| typescript | tayp skript |
| linux | linuks |
| python | paytın |
| azure | ejır |
| github | git hab |
| gitlab | git lab |
| docker | dakır |
| kubernetes | kubernetis |
| teams | tims |
| slack | slek |
| zoom | zum |
| excel | eksel |
| outlook | autluk |
| google | gugıl |
| microsoft | maykrosoft |
| notion | noşın |
| confluence | konfluens |
| sonicwall | sonik vol |
| salesforce | seylsfors |
| whatsapp | vatsap |
| linkedin | linkdin |
| youtube | yutub |
| iphone | ayfon |
| macbook | mekbuk |
| windows | vindovs |
| SQL | si ku el |

### Turkish acronyms (Turkish letter names)

KVKK, SGK, SSK, KDV, ÖTV, TBMM, AB, ABD, TL, İK, TC, TCMB, SPK, BDDK, MEB, YÖK, PTT, THY, TRT, AVM, İSO, İTO, TOBB, TÜİK, BİST, MHRS, EFT, TCKN, VKN, KEP, İBB, ABB, AKP, CHP, MHP, HDP, TSK, MİT, SGM, YKS, LGS, KPSS, ALES, YDS, TMSF, EPDK, BTK, GİB

### Acronyms said as a word

TÜBİTAK, ASELSAN, ROKETSAN, HAVELSAN, KOSGEB, NATO, NASA, IBAN, MERSİS, KOBİ, UNESCO, UNICEF, FIFA, UEFA, OPEC, SWIFT

## English voice (Kokoro-82M)

English text is turned into Kokoro's phonemes by a Rust port of the
dictionary path of [misaki](https://github.com/hexgrad/misaki): its gold and
silver dictionaries, the special cases (a/an/the/to/in), -s/-ed/-ing stems,
stress on capitals and spoken numbers. It matches misaki on 241 of 241 test
words. There is no espeak fallback:

- Words the dictionaries do not know are, in Turkish workplace text, mostly
  Turkish names, so they are read with Turkish spelling (stress on the last
  syllable), e.g. *Ayşe*, *Mehmet*.
- Acronyms are spelled out with English letter names.
- Tech words missing from the dictionaries:

| Written | Phonemes |
|---|---|
| jira | `ʤˈiɹə` |
| kubernetes | `kˌubəɹnˈɛtiz` |
| sonicwall | `sˈɑnɪkwˌɔl` |
| kvkk | `kˌAvˌikˌAkˈA` |
| github | `ɡˈɪthˌʌb` |
| gitlab | `ɡˈɪtlˌæb` |
| devops | `dˈɛvˌɑps` |
| saas | `sˈæs` |
| figma | `fˈɪɡmə` |
