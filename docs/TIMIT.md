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
| **viter, trained on TIMIT** | **68.1%** | **86.5%** | **90.5%** | **97.7%** | 5.9 ms | 10.5 ms |
| viter, `--no-refine` (10 ms frame grid, MFA's output) | 49.9% | 80.7% | 88.2% | 97.7% | 10.0 ms | 13.6 ms |
| MFA trained on TIMIT (McAuliffe 2026) | 63.6% | — | 85.3% | 97.1% | — | 12.0 ms |
| MFA `english_us_arpa` 3.0 (McAuliffe 2026) | 61.9% | — | 83.6% | 97.4% | — | 12.1 ms |
| MAUS (McAuliffe 2026) | 63.6% | — | 86.8% | 97.8% | — | 11.3 ms |

TRAIN (seen data, the training-time TextGrids, frame grid, 3696 utterances, 139k boundaries):

| system | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median | mean |
|---|---|---|---|---|---|---|
| viter | 50.9% | 81.7% | 89.1% | 98.0% | 9.7 ms | 12.9 ms |

viter is above the published GMM-HMM numbers at every tolerance. The published runs are on
a different test subset with MFA's own silence handling, so the 25 ms column is the
comparable one.

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
~57% at 10 ms against 58% for the midpoint crossing alone. Refined boundaries are placed
at the midpoint between the centres of the two 1 ms frames they separate, i.e. 4.5 ms after
the frame index, which is what the frame-grid convention already does at 10 ms.

Two more corrections use a 1 ms log-energy contour of the waveform, i.e. the data rather
than phone identities (`refine.rs`):

- Boundaries *into a transient*: where some candidate in the window has a rise of at least
  10 dB between the mean log energy of the 10 ms before it and the 10 ms after, the boundary
  goes at the largest such rise. The burst flips the pdf ratio as soon as it enters the
  25 ms analysis window, so the midpoint crossing was 13 ms early for closure→stop (30%
  within 10 ms); the energy onset is within 10 ms of the hand label for 86% of them. The
  10 ms before the rise must lie inside the window, otherwise the burst before a stop→vowel
  boundary is picked up instead of the vowel. Worth +8 points at 10 ms.
- Elsewhere the crossing level moves from the midpoint towards the louder side's plateau by
  0.1 per nat of the energy step across the window (second half minus first half), clamped
  to 0.35..0.65. Measured on the curves, the midpoint crossing is ~5 ms early into a quieter
  segment (vowel→closure/fricative/nasal) and ~3 ms late into a louder one (nasal→vowel,
  fricative→vowel), saturating within about a nat either way. Worth +2 points. Boundaries
  into the model's silence phone keep the midpoint: before a pause the window lies in the
  phone's decay into silence, not between two plateaus, and the hand label is at or before
  the window, so a lower level only made those later.

Tried and not kept: a shorter analysis window (5/10/15 ms) scored by the 25 ms-trained
pdfs; a least-squares change point instead of the midpoint split; modal-state instead of
middle-state pdfs; a deadzone on the level rule (weakens it without separating the silence
case); a symmetric "offset" rule at the largest energy fall (helps vowel→closure a little,
nothing else).

What remains, from the signed breakdown on TEST after refinement:

- Glides: vowel→glide is 5 ms early and glide→vowel 5 ms late (36% and 44% within 10 ms),
  both with the refined boundary inside the vowel, and no energy feature separates them
  (the step is ~0). `l` and `hh` are the worst phones.
- Utterance-initial boundaries (silence → first phone) are still 7-17 ms early on average
  (median 1-5 ms): TIMIT labels the closure of an initial stop as part of `h#`, while
  medial closures are their own phones, so the model learned the initial stop as
  closure+burst; the onset rule now catches the burst when it is inside the window.
- Phone ends before a pause are 5 ms late (44% within 10 ms): the phone pdfs cover the
  decay into silence and the hand label is before the refinement window.

Per phone class on TEST (viter, refined):

| class | n | ≤10 ms | ≤20 ms | ≤25 ms | ≤50 ms | median |
|---|---|---|---|---|---|---|
| vowel | 12655 | 67.8% | 86.5% | 91.1% | 98.4% | 5.6 ms |
| diphthong | 4317 | 64.0% | 83.3% | 88.7% | 97.2% | 6.5 ms |
| glide | 6026 | 54.1% | 74.5% | 80.7% | 95.1% | 8.5 ms |
| nasal | 4729 | 70.0% | 86.9% | 90.8% | 97.4% | 5.8 ms |
| fricative | 7498 | 69.3% | 89.2% | 92.4% | 98.0% | 6.4 ms |
| affricate | 623 | 79.1% | 90.0% | 92.6% | 97.8% | 3.5 ms |
| stop | 7494 | 76.5% | 89.4% | 91.8% | 97.7% | 3.7 ms |
| closure | 6995 | 70.4% | 92.3% | 95.0% | 98.6% | 6.4 ms |

Worst phones at 20 ms (39-fold): `hh` 65%, `q` 66%, `l` 69%, `p` 75%, `uw` 76%.

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
