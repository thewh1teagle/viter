# TIMIT benchmark

Accuracy against hand-labelled phone boundaries. Parity ([PARITY.md](PARITY.md)) says viter
reproduces MFA; this says how close both are to a human. Tracking issue:
https://github.com/thewh1teagle/viter/issues/8.

## Setup

- Corpus: TIMIT, 4620 TRAIN utterances (3.9 h, 462 speakers) for training, TEST for scoring
  with the SA1/SA2 sentences excluded (1344 utterances). Speaker = TIMIT speaker directory.
- Input is the reference phone sequence (61-phone set, closures kept, `h#`/`pau`/`epi`
  dropped and modelled as optional silence), grouped into words from the `.WRD` tier.
  Phoneme-string mode, no dictionary, `--no-position-dependent`. MFA gets the same corpus
  through a one-token-per-word dictionary (`plans/timit/timit_003.py`).
- Scoring: every start (and gap-adjacent end) boundary of every non-silence phone, matched by
  sequence position; `plans/timit/timit_002.py`. The literature reports 10/25/50 ms, so the
  25 ms column is the one to compare.

```
uv run plans/timit/timit_001.py                       # data/data → data/timit
viter train data/timit/train --no-position-dependent -o data/timit.viter
viter align data/timit/test data/timit.viter -o data/timit-test-tg
uv run plans/timit/timit_002.py data/timit-test-tg
```

## Results

viter, full schedule (mono → tri → LDA+MLLT → SAT ×3 with pronunciation probabilities),
trained on TRAIN in 31 min (of which ~28 min is per-speaker fMLLR, see issue #10), aligned
TEST in 72 s. 0 failed utterances, 0 label mismatches.

TEST (1344 utterances, 50,337 phone boundaries) vs hand labels:

| system | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median | mean |
|---|---|---|---|---|---|---|
| **viter, trained on TIMIT** | **49.9%** | **80.7%** | **88.2%** | **97.7%** | 10.0 ms | 13.6 ms |
| MFA trained on TIMIT (McAuliffe 2026) | 63.6% | — | 85.3% | 97.1% | — | 12.0 ms |
| MFA `english_us_arpa` 3.0 (McAuliffe 2026) | 61.9% | — | 83.6% | 97.4% | — | 12.1 ms |
| MAUS (McAuliffe 2026) | 63.6% | — | 86.8% | 97.8% | — | 11.3 ms |
| viter, `--sat-rounds 1 --no-pron-probs` | TODO | TODO | TODO | TODO | TODO | TODO |

TRAIN (seen data, the training-time TextGrids, 3696 utterances, 139k boundaries):

| system | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median | mean |
|---|---|---|---|---|---|---|
| viter | 50.9% | 81.7% | 89.1% | 98.0% | 9.7 ms | 12.9 ms |

viter is at or above the published GMM-HMM numbers at 25 and 50 ms and ~13 points below
them at 10 ms. The published runs are on a different test subset with MFA's own silence
handling, so the 25 ms column is the comparable one; the 10 ms gap is large enough to be
real and is the thing to investigate (a systematic offset of a few ms on some boundary
types, not wrong paths — the label sequences all match and p95 is 37 ms).

Per phone class on TEST (viter):

| class | n | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median |
|---|---|---|---|---|---|---|
| vowel | 12655 | 65.1% | 87.9% | 92.4% | 98.5% | 7.2 ms |
| diphthong | 4317 | 53.5% | 82.9% | 89.1% | 97.5% | 9.0 ms |
| glide | 6026 | 49.6% | 71.4% | 79.3% | 94.9% | 10.0 ms |
| nasal | 4729 | 56.3% | 84.1% | 89.1% | 97.4% | 8.3 ms |
| fricative | 7498 | 47.0% | 82.4% | 90.1% | 98.2% | 10.7 ms |
| affricate | 623 | 63.2% | 89.2% | 93.7% | 99.0% | 7.5 ms |
| stop | 7494 | 33.6% | 71.0% | 82.9% | 97.5% | 13.9 ms |
| closure | 6995 | 35.2% | 79.6% | 90.2% | 98.5% | 12.5 ms |

Worst phones at 20 ms (39-fold): `p` 28%, `hh` 54%, `b` 57%, `dh` 61%, `q` 62%, `ng` 68%,
`w` 69%, `l` 69%. Stops and closures carry most of the 10 ms deficit: the closure→burst
boundary is where a 10 ms frame grid hurts most, and `p` is an outlier worth a look on its own.

Word boundaries on TEST (`.WRD` tier, 13,210 boundaries; 8 files skipped where the word
grouping from `.WRD` disagrees with the phone grouping), for comparison with neural word
aligners:

| system | ≤10 ms | ≤25 ms | ≤50 ms | ≤100 ms |
|---|---|---|---|---|
| **viter, words tier, official TEST** | **41.1%** | **78.2%** | **93.3%** | **98.4%** |
| MWA (Weber 2026, own split, trained on TIMIT words) | 58.0% | 81.3% | 91.6% | 97.8% |
| MFA pretrained `english_us_arpa`, orthographic input (Rousso 2024 via Weber 2026) | 41.6% | 72.8% | 89.4% | 97.4% |
| WhisperX (same) | 22.4% | 52.7% | 82.4% | 94.2% |

The MWA/MFA rows use a speaker-level 80/10/10 split rather than the official TEST, and the
MFA row is the pretrained model on orthographic text, so they are indicative only.
Issue: https://github.com/thewh1teagle/viter/issues/9.

Published reference: McAuliffe, *Montreal Forced Aligner and the state of speech-to-text
alignment in 2026*, Table 5 (https://arxiv.org/abs/2606.18466). Same input (manual phonetic
transcripts), same corpus; the test subset may differ. Weber et al., *Multilingual Word-Level
Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming*,
Interspeech 2026 (https://arxiv.org/abs/2606.10675), Table 3.

## Notes

- No failed utterances and no label-sequence mismatches on TRAIN or TEST at the default beam.
- TODO: `--sat-rounds 1 --no-pron-probs` row (462 speakers, ~10 utterances each, so fMLLR
  has little data per speaker; the short schedule may lose less here than on LJSpeech).
- TODO: the 10 ms gap. Candidates: boundary placement inside a frame (viter/MFA both snap to
  10 ms; the published MFA numbers are on the same grid, so this is not the whole story),
  transition priors on closure→burst, and `p` specifically.
