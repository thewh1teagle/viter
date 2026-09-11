# CLI

One binary, three subcommands. Everything here is also callable from Python — see [PYTHON.md](PYTHON.md).

```
viter train  <corpus_dir> -o model.viter [flags]
viter align  <corpus_dir|audio.wav> <model.viter> -o out_dir [flags]
viter serve  <dir> [flags]
```

## `viter train`

Trains an acoustic model from a corpus. Runs mono → tri → LDA+MLLT → SAT by default.
The final full-corpus alignment runs only with `--out-textgrids`; otherwise training
ends after the model is trained. Run `viter align` later to produce training TextGrids.

| flag | default | meaning |
|---|---|---|
| `<corpus_dir>` | required | corpus root, scanned recursively — see [CORPUS-FORMAT.md](CORPUS-FORMAT.md) |
| `-o, --output <model.viter>` | required | where to write the trained model |
| `--dict <dict.txt>` | none | pronunciation dictionary. Omit for phoneme-string mode |
| `--out-textgrids <DIR>` | none | run the final full-corpus alignment and write its TextGrids |
| `--config <train.toml>` | none | `TrainConfig` overrides; unset fields keep MFA defaults |
| `--no-lda` | off | skip LDA+MLLT; SAT then adapts on delta features, as MFA 3.x exports by default |
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
| `--no-refine` | off | keep boundaries on the 10 ms frame grid (MFA's output) |
| `--cpu` | off | force the CPU scoring path |
| `--ctm` | off | write Kaldi/MFA CTM (`utt 1 start dur label`) instead of TextGrids |

Output mirrors the corpus layout: an utterance at `corpus/spk/utt.wav` becomes
`out_dir/spk/utt.TextGrid`, with `words` and `phones` interval tiers. If the model was trained
with SAT (`am_si` present), alignment automatically runs two passes — speaker-independent
align, per-speaker fMLLR estimation, adapted re-align.

Boundaries are then refined to 1 ms: a window of up to ±30 ms around each one is rescored
with 1 ms-shifted features and the two phones' core pdfs, and the boundary goes where their
likelihood ratio crosses its range at a level set by the energy step across the window, or
at the energy onset when the next segment starts with a transient (a stop burst). This is
MFA's `--fine_tune` with a wider window, steadier pdfs and an energy contour; on TIMIT it
moves phone boundaries within 10 ms of the hand labels from 50% to 68%
([TIMIT.md](TIMIT.md)). `--no-refine` gives the frame-grid alignment the
decoder produced, which is what MFA writes without `--fine_tune` and what the parity numbers
in [PARITY.md](PARITY.md) are measured on. CTM times have 3 decimals either way.

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

The base MFCCs stay in memory for the whole run: 13 floats per 10 ms frame, about
19 MB per hour of audio. Passes that need derived features (deltas, spliced+LDA,
speaker-adapted) walk chunks of about 1.2 million frames (3.3 h).

Built-in training stages retain up to **2 GiB** of derived features between passes,
reusing the same arrays while their transforms are unchanged. The cache is cleared
after each MLLT or fMLLR update, before accumulation uses the new feature space, and
released when the stage ends. Chunks beyond the cache budget are derived and dropped
as they are processed, so larger corpora still stream. Alignment passes remain
uncached. The feature working set is therefore the base store, at most 2 GiB of
cached training features, and the current chunk with its transform temporaries.

Models, decoding scores and statistics use additional memory. In particular, decision
tree statistics cover a whole training subset and can dominate training's peak.

Speaker adaptation is exact across chunks: fMLLR statistics are accumulated chunk by
chunk and solved once per speaker, so a speaker may span chunks and a single-speaker
corpus splits like any other. The chunk size is `chunk_frames` in the training config.

**Training** does not need the whole corpus either. MFA's schedule (and viter's) trains on
subsets of 10,000 to 50,000 utterances, so a representative slice is enough:

```
viter train corpus/part-00 --dict dict.txt -o model.viter   # 20-50k utterances is plenty
```

**Aligning** is linear in audio, about 24 h per 3 minutes on a GPU, so 1000 h takes on the
order of two hours. Folder splits still work for spreading across machines; every
utterance is aligned independently, but keep a speaker's files together so fMLLR sees
all of them.

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

Without `--out-textgrids`, `train` omits `aligned` and `failed` from the summary because
no final alignment was measured, and prints a hint to use `viter align` for TextGrids.

Progress during a run is a single bar covering the whole schedule. The counter under it
is in **utterance-passes** — one utterance processed by one pass (MFCC, graph build,
alignment, accumulation, tree stats, …) — and its total is computed before training starts by
enumerating every pass the schedule will run, so reaching it means the run is over. The bar
and its percentage show the elapsed share of the *predicted* total time rather than the raw
counter, because a late pass over the same utterances costs several times more than an early
one. The current stage and sub-step show as text on the bar (`SAT/fMLLR 2 · iter 12/35 ·
align 4,608/13,093 · 1,234,567/3,651,390`), and each stage prints one persistent line with
its elapsed time when it finishes. Detailed logging goes through `tracing`; raise it with
`RUST_LOG=debug`.

The ETA is measured rather than extrapolated from the counter: every pass's cost (including
the gaps around it — tree builds, model updates, fMLLR solves) is timed as it runs, a pass
that has not run yet is predicted from measured passes of the same kind, extrapolated in
model size when that kind ran in an earlier stage, and a kind that has never run is taken
from a built-in cost table scaled to this machine's measured speed. Measured on LJSpeech (13k
utterances, GPU): never more than ~20% off after the first minute, typically under 10%. On
CPU the estimate is within ~15% from the triphone stage on. On a small corpus (1k
utterances) it can read ~30% low before the SAT rounds, whose cost is not knowable until one
has been measured.

## Exit codes

| code | meaning |
|---|---|
| 0 | success — including a run where some utterances failed to align, as long as at least one succeeded |
| 1 | usage error: bad flags, missing file, unreadable corpus, no utterances found |
| 2 | data error: dictionary/model phone mismatch (`io::remap` failure), unreadable model, corrupt `.viter` |
| 3 | run failure: every utterance failed to align, or a training stage could not complete |

Failures are reported once, with the offending path or symbol, not as a stack trace.
