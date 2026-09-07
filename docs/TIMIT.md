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
trained on TRAIN in 31 min before the fMLLR fix of issue #10 (~5 min after), aligned TEST
in 18 s including boundary refinement. 0 failed utterances, 0 label mismatches.

TEST (1344 utterances, 50,337 phone boundaries) vs hand labels:

| system | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median | mean |
|---|---|---|---|---|---|---|
| **viter, trained on TIMIT** | **58.0%** | **84.3%** | **89.5%** | **97.7%** | 8.3 ms | 12.2 ms |
| viter, `--no-refine` (10 ms frame grid, MFA's output) | 49.9% | 80.7% | 88.2% | 97.7% | 10.0 ms | 13.6 ms |
| MFA trained on TIMIT (McAuliffe 2026) | 63.6% | — | 85.3% | 97.1% | — | 12.0 ms |
| MFA `english_us_arpa` 3.0 (McAuliffe 2026) | 61.9% | — | 83.6% | 97.4% | — | 12.1 ms |
| MAUS (McAuliffe 2026) | 63.6% | — | 86.8% | 97.8% | — | 11.3 ms |

TRAIN (seen data, the training-time TextGrids, frame grid, 3696 utterances, 139k boundaries):

| system | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median | mean |
|---|---|---|---|---|---|---|
| viter | 50.9% | 81.7% | 89.1% | 98.0% | 9.7 ms | 12.9 ms |

viter is above the published GMM-HMM numbers at 25 and 50 ms and below them at 10 ms. The
published runs are on a different test subset with MFA's own silence handling, so the 25 ms
column is the comparable one.

### The 10 ms column (issue #12)

On the frame grid the error is not noise but a bias by transition class: boundaries *into*
low-energy segments (closures, stops, silence-adjacent `hh`/`dh`) land 12-30 ms early and
boundaries out of them into vowels 2-7 ms late, while vowel-to-vowel-like boundaries are
unbiased. The decoder switches phones where the two *boundary* HMM states' likelihoods
cross, and those states were trained on exactly the transition frames. Removing each
transition class's mean bias (an oracle) takes the 10 ms figure to 68%, so that is the
whole story. `plans/timit/timit_004.py` prints the signed breakdown.

`viter align` therefore refines every boundary to 1 ms by default ([CLI.md](CLI.md)): a
window of up to ±30 ms is rescored with 1 ms-shifted features and the two phones'
middle-state pdfs, and the boundary goes where their likelihood ratio crosses the midpoint
of its own range in the window. That is MFA's `--fine_tune` idea, self-calibrating per
boundary; MFA uses the boundary-state pdfs and a ±10 ms window, which on this data gives
~57% at 10 ms against 58% here. Refined boundaries are placed at the midpoint between the
centres of the two 1 ms frames they separate, i.e. 4.5 ms after the frame index, which is
what the frame-grid convention already does at 10 ms.

What remains, from the signed breakdown on TEST after refinement:

- Stops are the worst class (38.8% at 10 ms): the closure→burst boundary is still 14 ms
  early on average. The burst is a transient that flips the ratio as soon as it enters the
  25 ms analysis window; a shorter window scored by the 25 ms-trained pdfs did not help.
- Utterance-initial boundaries (silence → first phone, 3% of boundaries) are 20-25 ms
  early and refinement cannot move them: TIMIT labels the closure of an initial stop as
  part of `h#`, while medial closures are their own phones, so the model learned the
  initial stop as closure+burst and scores the closure frames as the stop.
- Boundaries into vowels out of glides/nasals are now 5-9 ms late.

Per phone class on TEST (viter, refined):

| class | n | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median |
|---|---|---|---|---|---|---|
| vowel | 12655 | 64.4% | 86.3% | 91.0% | 98.3% | 6.8 ms |
| diphthong | 4317 | 59.6% | 82.9% | 88.4% | 97.2% | 7.6 ms |
| glide | 6026 | 47.5% | 72.1% | 79.6% | 95.1% | 10.8 ms |
| nasal | 4729 | 64.9% | 85.3% | 89.5% | 97.1% | 7.0 ms |
| fricative | 7498 | 63.9% | 88.7% | 93.1% | 98.2% | 7.4 ms |
| affricate | 623 | 64.2% | 89.6% | 93.6% | 98.2% | 7.5 ms |
| stop | 7494 | 38.8% | 79.1% | 86.5% | 97.8% | 12.8 ms |
| closure | 6995 | 63.5% | 91.8% | 95.2% | 98.6% | 7.5 ms |

Worst phones at 20 ms (39-fold): `p` 35%, `hh` 61%, `q` 66%, `l` 69%, `b` 70%.

Word boundaries on TEST (`.WRD` tier, 13,210 boundaries; 8 files skipped where the word
grouping from `.WRD` disagrees with the phone grouping), for comparison with neural word
aligners:

| system | ≤10 ms | ≤25 ms | ≤50 ms | ≤100 ms |
|---|---|---|---|---|
| **viter, words tier, official TEST** | **47.5%** | **80.2%** | **93.2%** | **98.4%** |
| viter, `--no-refine` | 41.1% | 78.2% | 93.3% | 98.4% |
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
- Not run yet: `--sat-rounds 1 --no-pron-probs`. With 462 speakers of ~10 utterances each,
  fMLLR has little data per speaker, so the short schedule may lose less here than on LJSpeech.
- Cheap loop for boundary work: `viter align data/timit/test/DR1 data/timit.viter -o out/DR1`
  (1.6 s, 88 scored files, within a point of the full set) then `timit_004.py out`.
