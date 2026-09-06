# Corpus format

`viter_io::scan(dir, &CorpusOptions)` walks a directory recursively and produces a
`Corpus { utts, speakers, phones, silence_phones, oov_words }`.

## Folder layout

Each utterance is an audio file paired with a transcript file of the same stem in the same
directory:

```
corpus/
  speaker_a/
    utt001.wav      utt001.txt
    utt002.flac     utt002.lab
  speaker_b/
    interview.mp3   interview.txt
```

- Audio: `.wav`, `.flac`, `.mp3` — decoded by symphonia, downmixed to mono, resampled to
  16 kHz. Any sample rate in; 16 kHz is what the features see.
- Transcript: `.txt` or `.lab` with the same stem. Both extensions mean the same thing; `.lab`
  is MFA's convention. Content is one line of whitespace-separated tokens.
- Utterance id = the path relative to the corpus root, without extension.
- Audio without a transcript, or a transcript without audio, is skipped with a warning.

Nesting depth is free; `scan` recurses.

## Speaker detection

`CorpusOptions.speaker_from` selects the strategy:

| `SpeakerSource` | speaker id |
|---|---|
| `ParentDir` (default) | the name of the directory containing the file — the layout above |
| `Prefix(n)` | the first `n` characters of the utterance stem, for flat `spk1_utt003.wav` layouts |
| `Single` | one speaker for the entire corpus |

Speaker identity matters: CMVN statistics are pooled per speaker, and SAT estimates one fMLLR
transform per speaker. A corpus with wrong speaker boundaries trains and aligns worse, and a
single-speaker corpus should say so rather than pretend each file is its own speaker.

## Dictionary mode

With `--dict` (`CorpusOptions.dictionary`), each transcript token is a *word* looked up in a
pronunciation dictionary. Format is MFA's, which is CMUdict-compatible:

```
HELLO   HH AH0 L OW1
WORLD   W ER1 L D
READ    R IY1 D
READ(2) R EH1 D
```

- Word, then whitespace, then its phones separated by whitespace.
- Lookup is case-insensitive.
- Variant markers `(1)`, `(2)`, … are stripped from the word, so `READ` and `READ(2)` are two
  pronunciations **of the same word**. All pronunciations listed for a word are kept, in file
  order, and the alignment graph carries them as alternatives; the Viterbi pass picks one per
  occurrence and records it in `Alignment.prons` / `WordInterval.pron`. Two entries with an
  identical phone sequence are deduplicated (the first wins).
- The phone inventory of the model is the set of phones appearing in *any* pronunciation in the
  dictionary, plus the silence and OOV phones.

### Probability columns

MFA's extended format puts optional numeric columns between the word and its phones. Viter
parses the same three shapes (`plans/kalpy/kalpy/fstext/lexicon.py`):

| columns | meaning |
|---|---|
| 0 | no probabilities; the graph uses its global defaults |
| 1 | `prob` — pronunciation probability; graph cost `\|ln p\|`, `p` floored at 0.01 |
| 2 | `prob`, `silence_after_prob` — P(silence follows this word) |
| 4 | the above plus `silence_before_correction`, `non_silence_before_correction` — multipliers on the silence / no-silence branch *before* the word |

```
aɪ	0.99	0.08	1.64	0.79	aɪ
```

is the 4-column shape, as in `data/dict/ljspeech_ipa.dict`. A field counts as a probability
column only if it matches MFA's pattern `digits.digits` — so a bare `1` or a phone named `0`
stays a phone, and three numeric columns is not a recognised shape (the third is read as a
phone). Missing columns stay `None` and each falls back to its global default rather than to a
hardcoded number.

## Phoneme-string mode (no dictionary)

Omit `--dict` and the transcript tokens are *phones* rather than words. How they group into
words depends on whether the transcript marks word boundaries:

**No separator — every token is its own one-phone word.** So

```
hh ax l ou
```

is four one-phone words. This is the original behaviour and stays the default.

**With a separator — tokens group into multi-phone words.** Either marker works, and they can
be mixed:

- a `|` token between words: `hh ax | l ou` is two words, `hh ax` and `l ou`;
- a run of **two or more spaces** (or a tab): `hh ax  l ou` is the same two words, since single
  spaces then only separate the phones *inside* a word.

A word built this way gets one pronunciation holding all of its phones, position-tagged the
usual way (`hh_B ax_E`), so a multi-phone word behaves exactly like a dictionary word — the
words tier of the output shows `hh ax` as one interval instead of two. Presence of a separator
anywhere in the transcript switches the whole utterance into grouped mode.

This is what makes viter language-agnostic: any symbol set works as long as it is used
consistently across the corpus, because the phone inventory is simply the set of distinct
tokens observed. There is no G2P, no language model, and no need for a lexicon of any kind.
Pronunciation alternatives do not exist in this mode — each word has exactly one.

Optional silence is still inserted between words and at the utterance edges, so pauses are
modelled just as in dictionary mode.

Use this mode when you have phone-level transcripts (from another aligner, from a
grapheme-to-phoneme step you ran yourself, or from a language with no MFA dictionary), and
dictionary mode when you have orthographic text.

## Position-dependent phones

On by default (`position_dependent = true`, matching MFA). Each phone of a word is suffixed
with its position in the word:

| word length | tags |
|---|---|
| 1 phone | `_S` (singleton) |
| ≥ 2 phones | `_B` (begin), `_I` (internal) …, `_E` (end) |

So `HELLO → HH AH0 L OW1` becomes `HH_B AH0_I L_I OW1_E`, and a one-phone word becomes `X_S`.
The tags are pure string suffixes on the phone symbol, but they expand the phone set, and they
feed the tree roots and the automatically-obtained questions — position variants of the same
base phone share a root but may split apart. `SymbolTable` holds the tagged symbols; the model
records `position_dependent` so alignment retags identically.

Silence and OOV phones are **never** tagged. `io::untag("AH_B") == "AH"` strips the suffix for
TextGrid and CTM output, so what you read back is the plain phone.

In ungrouped phoneme-string mode every token is a one-phone word, so tagging would make every
phone `_S` and add nothing — turn it off there. With `|` or double-space word grouping the tags
are meaningful again and worth keeping on.

## Silence and `spn`

Two special phones always exist in the symbol table, ahead of the corpus phones:

- `sil` (`CorpusOptions.silence_phone`) — optional silence. It is not written in transcripts.
  `hmm::build_graph` inserts an optional `sil` branch before the first word, between every pair
  of words, and after the last word, with cost `-log(silence_probability)` against
  `-log(1 - silence_probability)` for the direct edge (`0.5` / `0.5` by default, and
  `initial_silence_probability` at the utterance start). Silence uses the 5-state topology and
  is context-independent in the decision tree.
- `spn` (`CorpusOptions.oov_phone`) — spoken noise, the pronunciation given to out-of-vocabulary
  words. Also context-independent.

In output, silence appears as `sil` in the phones tier and as an empty interval in the words
tier, matching MFA's TextGrids.

## OOV handling

In dictionary mode, a transcript word absent from the dictionary is mapped to the single OOV
phone `spn` and counted in `Corpus.oov_words` (word → occurrence count). The CLI prints the
top OOV words at the end of a run; a large count usually means a dictionary in the wrong
alphabet or a case/normalisation mismatch, not genuinely unknown words. An OOV word gets a single pronunciation, `spn`, with no
probability columns. OOV words still get a word interval in the output — one covering their
`spn` span — so the words tier stays aligned with the transcript.

In phoneme-string mode there is no OOV: every token is a phone by definition. A typo therefore
becomes a new phone with almost no training data rather than an error, so check
`Corpus.phones` if the inventory is larger than you expect.

## Aligning against a trained model

`io::remap(&mut corpus, model_phones)` re-resolves a freshly scanned corpus's phone ids — every
phone of every pronunciation — against the symbol table stored in the `.viter` model. Phones the model has never seen are a hard error
listing every offending symbol — aligning with a mismatched dictionary or phone set fails loudly
rather than silently substituting.

`io::single(audio, transcript, opts, phones)` builds a one-utterance corpus for the
`viter align audio.wav --text "..."` path, reusing the model's symbol table directly.
