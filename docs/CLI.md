# CLI

One binary, three subcommands.

```
viter train  <corpus_dir> -o model.viter [flags]
viter align  <corpus_dir|audio.wav> <model.viter> -o out_dir [flags]
viter serve  <dir> [flags]
```

## `viter train`

Trains an acoustic model from a corpus. Runs mono → tri → LDA+MLLT → SAT by default, then
aligns the full corpus with the final model.

| flag | default | meaning |
|---|---|---|
| `<corpus_dir>` | required | corpus root, scanned recursively — see [CORPUS-FORMAT.md](CORPUS-FORMAT.md) |
| `-o, --output <model.viter>` | required | where to write the trained model |
| `--dict <dict.txt>` | none | pronunciation dictionary. Omit for phoneme-string mode |
| `--out-textgrids <DIR>` | none | also write TextGrids for the final training alignments |
| `--config <train.toml>` | none | `TrainConfig` overrides; unset fields keep MFA defaults |
| `--no-lda` | off | stop after the triphone stage (implies `--no-sat`) |
| `--no-sat` | off | stop after LDA+MLLT; no fMLLR, no `am_si` in the model |
| `--cpu` | off | force the CPU scoring path; skip GPU adapter probing |
| `--seed <N>` | from config | RNG seed for subset shuffling, Gaussian splitting, flat start |

Dropping stages trades accuracy for time. `--no-sat` is a reasonable choice for a
single-speaker corpus, where per-speaker adaptation has little to adapt to.

```bash
# Full recipe with a dictionary, on GPU
viter train ./corpus -o english.viter --dict ./english_us_arpa.dict

# Phoneme-string corpus, mono+tri only, deterministic
viter train ./phones_corpus -o quick.viter --no-lda --seed 42

# Train and immediately get TextGrids for the training data
viter train ./corpus -o m.viter --dict d.txt --out-textgrids ./aligned
```

## `viter align`

Aligns audio against an existing model. The first argument is either a corpus directory or a
single audio file.

| flag | default | meaning |
|---|---|---|
| `<corpus_dir\|audio.wav>` | required | a corpus to scan, or one audio file |
| `<model.viter>` | required | a model from `viter train` |
| `-o, --output <out_dir>` | required | where TextGrids (or CTMs) are written |
| `--dict <dict.txt>` | none | dictionary; must match the one used for training |
| `--text "..."` | none | inline transcript for the single-file form, instead of a `.txt`/`.lab` |
| `--beam <N>` | 10 | Viterbi beam |
| `--retry-beam <N>` | 40 | wider beam used only for utterances that fail at `--beam` |
| `--cpu` | off | force the CPU scoring path |
| `--ctm` | off | write Kaldi/MFA CTM (`utt 1 start dur label`) instead of TextGrids |

Output mirrors the corpus layout: an utterance at `corpus/spk/utt.wav` becomes
`out_dir/spk/utt.TextGrid`, with `words` and `phones` interval tiers. If the model was trained
with SAT (`am_si` present), alignment automatically runs two passes — speaker-independent
align, per-speaker fMLLR estimation, adapted re-align.

Raising `--beam` fixes utterances that fail to align, at a cost in time; a widespread failure
usually means a transcript/audio mismatch rather than too narrow a beam.

```bash
# Align a corpus
viter align ./corpus english.viter -o ./aligned --dict ./english_us_arpa.dict

# One file with an inline transcript
viter align talk.wav english.viter -o ./out --dict d.txt --text "hello world"

# CTM output, wider beam for a noisy corpus
viter align ./corpus m.viter -o ./out --dict d.txt --ctm --beam 20 --retry-beam 80
```

## `viter import`

Converts a Montreal Forced Aligner acoustic model into a `.viter` model, so viter can align with
MFA's own trained parameters. Accepts either the distributed `.zip` or an unpacked model directory.

| flag | default | meaning |
|---|---|---|
| `<path>` | required | MFA model `.zip`, or a directory holding `final.mdl`, `tree`, `phones.txt`, `meta.json` |
| `-o, --out <FILE>` | `model.viter` | where to write the converted model |

```bash
viter import ljspeech_ipa.zip -o ljspeech.viter
viter align ./corpus ljspeech.viter --dict ljspeech.dict -o ./out
```

The importer reads Kaldi's binary `TransitionModel`, `AmDiagGmm` and `ContextDependency` directly.
It rebuilds the transition model from the imported tree and topology and then copies MFA's
probabilities across by matching `(phone, hmm_state, forward_pdf, self_loop_pdf)` tuples, so the
transition ids are viter's own while the parameters stay MFA's; a tuple-count mismatch is reported
rather than silently accepted.

The feature pipeline is taken from `meta.json`, except that `uses_splices` is cross-checked against
the evidence: MFA exports that flag as `false` even for LDA+MLLT models, so when `lda.mat` maps
spliced MFCCs onto exactly the GMM dimension, the splice+LDA pipeline is used regardless.

## `viter serve`

Starts the web viewer over a directory of audio and TextGrids — normally an `align` output
directory sitting next to the audio, or the audio directory itself.

| flag | default | meaning |
|---|---|---|
| `<dir>` | required | directory scanned recursively for `x.TextGrid` paired with `x.wav\|flac\|mp3` |
| `--port <N>` | 7878 | listen port |
| `--no-open` | off | do not open a browser automatically |

```bash
viter serve ./aligned
viter serve ./aligned --port 9000 --no-open
```

See [VIEWER.md](VIEWER.md) for the API and keyboard shortcuts.

## Large corpora

Features are held in memory during a run: about 0.2 GB per hour of audio at peak, so
a 24 h corpus peaks near 8 GB and a 128 GB machine handles roughly 400 h in one call.
Beyond that, split the work; nothing about the model changes.

**Training** does not need the whole corpus. MFA's schedule (and viter's) trains on
subsets of 10,000 to 50,000 utterances, so train on a representative slice and align
the rest with the model:

```
viter train corpus/part-00 --dict dict.txt -o model.viter   # 20-50k utterances is plenty
```

**Aligning** is linear in audio and embarrassingly parallel across chunks. Any folder
split works because every utterance is aligned independently:

```
for part in corpus/part-*; do
  viter align "$part" model.viter --dict dict.txt -o "out/$(basename "$part")"
done
```

Speaker adaptation (fMLLR) is estimated per speaker within one `align` call, so keep a
speaker's files in the same chunk. A chunk of 50 to 100 hours keeps peak memory in
the low tens of GB; throughput is about 24 h of audio per 3 minutes on a GPU, so 1000 h
takes on the order of two hours regardless of how it is chunked.

A disk-backed feature store that removes the need to chunk is tracked in
[issue #6](https://github.com/thewh1teagle/viter/issues/6).

## Summary output

`train` and `align` print a short summary when they finish:

```
utterances     4821
aligned        4809
failed           12
oov words        37  (top: THE_QUICK 9, BROWN 5, ...)
device         gpu (Vulkan, NVIDIA GB10)
elapsed        18m 42s
RTF            0.031
```

- **failed** — utterances that produced no path even at `--retry-beam`. These get no output
  file; the ids are logged at warn level.
- **RTF** — real-time factor, wall-clock seconds per second of audio. Below 1.0 is faster than
  real time.
- **oov words** — dictionary mode only; see [CORPUS-FORMAT.md](CORPUS-FORMAT.md#oov-handling).

Progress during a run is one `indicatif` bar per stage (`mono iter 12/40`), plus a bar for the
initial feature extraction. Detailed logging goes through `tracing`; raise it with
`RUST_LOG=debug`.

## Exit codes

| code | meaning |
|---|---|
| 0 | success — including a run where some utterances failed to align, as long as at least one succeeded |
| 1 | usage error: bad flags, missing file, unreadable corpus, no utterances found |
| 2 | data error: dictionary/model phone mismatch (`io::remap` failure), unreadable model, corrupt `.viter` |
| 3 | run failure: every utterance failed to align, or a training stage could not complete |

Failures are reported once, with the offending path or symbol, not as a stack trace.
