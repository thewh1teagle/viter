# Parity

Parity with Kaldi/MFA numerics is the definition of correct. viter is not "an aligner
inspired by MFA" — it is meant to produce the same numbers, and every stage is checkable
against the reference implementation sitting in `plans/`.

## Current status (2026-09-07, LJSpeech)

Measured against the TextGrids of a real MFA 3.4 `train` run on the same corpus, dictionary
and machine, with `plans/parity/parity_001.py` (boundary diff), `parity_002_breakdown.py`
(silence-adjacent vs internal), `parity_003_silences.py` (silence intervals) and
`parity_004_feats.py` (features vs Kaldi binaries).

| | viter vs MFA | viter vs viter (other seed, before 2026-09-07) |
|---|---|---|
| phone label sequences | identical on all 13,088 files | identical |
| phone boundaries within 10 ms | 87% (83% before 2026-09-07) | 77% |
| within 20 ms | 97% (96%) | 95% |
| median / p95 difference | 0 ms / 10 ms (20 ms) | 0 ms |
| silence intervals | 56.3k (MFA 59.9k) | |

The 2026-09-07 training changes (graphs re-costed from the transition model each
realignment, deterministic flat start; [TRAINING.md](TRAINING.md)) moved every row towards
MFA; the earlier numbers are kept for the record.

MFA's TextGrids are as close to viter's as one viter run is to another: parity. The
configuration is MFA's own for this corpus, `--no-position-dependent` with LDA and the
default schedule (mono → tri → LDA+MLLT → SAT → SAT 2 → pronunciation probabilities → SAT 3),
about 10 minutes on an NVIDIA DGX Spark. The short schedule (`--sat-rounds 1 --no-pron-probs`)
trains in about 2 minutes at roughly 70% within 20 ms.

What it took (all found by auditing training against the Kaldi/MFA source and by diffing
MFA's model structure through `viter import`): MFA's real phone topology (state 0 skips to
state 1, 2 or the exit and only state 1 loops), one pdf per silence state, MFA's Gaussian
schedule details, SAT rounds initialised from statistics, composed fMLLR transforms,
silence-free LDA/MLLT statistics, and MFA's exact pronunciation-probability counting.
Aligning with an imported MFA model reproduces MFA's own TextGrids at 99.8% within 20 ms.
Tracking issue: https://github.com/thewh1teagle/viter/issues/5.

**TIMIT.** The same check against hand labels rather than against MFA: MFA 3.4.0 trained on
the TIMIT corpus of [TIMIT.md](TIMIT.md) (same audio, phone sets, dictionary and machine;
`plans/timit/timit_003.py`, artefacts in `data/timit-mfa/`) in 19 min, viter in 3.6 min.
TEST phone boundaries within 10/20/25/50 ms of the hand labels on the frame grid: MFA
57.9 / 85.7 / 91.1 / 97.9%, viter (`--no-refine`) 56.4 / 85.4 / 90.9 / 97.9%; with MFA's
`compare_alignments` metric (`timit_002.py --mfa-metric`) 61.1 / 91.2 / 98.1% vs
58.6 / 91.2 / 98.1% within 10/25/50 ms. Two MFA 3.4.0 bugs turned up on the way, both worked
around locally for the reference run (`data/timit-mfa/mfa-3.4.0-local-patches.diff`): the
as-shipped `mfa align` resets `uses_speaker_adaptation` before `_align()` and so skips the
fMLLR second pass even with a speaker-adapted model (57.5% at 10 ms as shipped, 57.9% with
the pass restored; the training-time TextGrids do use it), and `--fine_tune` is a no-op
because since 31d0f552 it runs inside `_align()` before the phone intervals it adjusts have
been collected (and once moved after collection, its job also needs the words archive and
mis-keys its alignment reader; patched to run, it lowers the grid to 56.5% at 10 ms).
viter's `align` does the fMLLR pass and refines by default.

These are frame-grid numbers: training-time TextGrids, and `viter align --no-refine`. By
default `align` refines every boundary to 1 ms afterwards ([CLI.md](CLI.md), issue #12),
which is deliberately *not* what MFA writes; compare against MFA with `--no-refine`.

## Method

MFA and kalpy are Python wrappers over Kaldi. That makes the reference runnable: a `uv`
script imports `kalpy` / `montreal_forced_aligner`, runs one stage on a small fixed corpus,
dumps intermediate tensors to `.npy`, and compares them against the same tensors dumped by
viter. Per `AGENTS.md`, validation scripts are standalone `uv` scripts at
`plans/<name>/<name>_NNN.py` with a matching `.md` describing what they check.

The whole-pipeline check that exists today is `plans/parity/parity_001.py`: it needs no MFA
install, it diffs two TextGrid folders (viter output vs MFA output) boundary by boundary.
Per-stage tensor checks against kalpy are the planned next level. Shape:

```python
# /// script
# dependencies = ["numpy", "montreal-forced-aligner", "kalpy"]
# ///
```

It runs, per stage, both implementations over `data/` (a handful of utterances, committed, small
enough to be fast and fixed enough to be comparable), and reports max absolute error, max
relative error, and a pass/fail against the tolerance for that stage. viter side: a hidden
`--dump-stage <dir>` debug path writes the same arrays as `.npy`. One script, one exit code,
one table of stage-by-stage results.

## Per-stage checks

| stage | compared against | quantity | tolerance |
|---|---|---|---|
| MFCC | `kalpy.feat.mfcc.MfccComputer` | the `[frames, 13]` matrix, per utterance | **1e-4** absolute |
| CMVN | Kaldi `ApplyCmvn` | normalized features | 1e-4 |
| deltas / splice | `ComputeDeltas`, `SpliceFrames` | the 39- and 91-dim matrices | 1e-5 (pure reordering and fixed-coefficient sums) |
| GMM scoring | `AmDiagGmm::LogLikelihood` | per `(frame, pdf)` log-likelihood | 1e-4 relative — also the CPU-vs-GPU contract in `device` |
| monophone | MFA's mono training log | **total log-likelihood per iteration** | within 0.1% at every iteration; the curve shape must match, not just the endpoint |
| tree | Kaldi `BuildTree` | **leaf count**, then the leaf assignment of every context window | leaf count exact; assignments ≥ 99% identical |
| transition model | `TransitionModel` | number of transition ids, and the tuple table | exact |
| LDA / MLLT | `LdaEstimate::Estimate`, `MlltAccs::Update` | the transform matrices | 1e-3 (eigenvector sign and ordering normalized before comparison) |
| fMLLR | `ComputeFmllrMatrixDiagGmm` | per-speaker transform, and objective improvement | 1e-3 |
| alignment | MFA's final TextGrids | **median phone-boundary difference < 10 ms**, i.e. one frame | median < 10 ms; also report p95 and max |
| TextGrid | praatio output | long-format text | byte-for-byte for the long format |

The alignment check is the one that matters to users and the one with the loosest tolerance,
because a one-frame boundary difference is inaudible and MFA itself is not stable to the frame
across reruns with different Gaussian-split RNG. Report the full distribution, not just the
median: a good result is a median at 0 ms with a thin tail, and a median at 0 ms with a fat
tail means some utterances are aligning to a different path entirely.

Earlier stages have tighter tolerances precisely because errors compound. An MFCC that is off
by 1e-3 will not produce a matching tree, so debugging always starts at the top of the table
and moves down: the first row that fails is the bug.

## Known sources of drift

These are expected and bounded. They are the reason tolerances are not zero.

**Tree tie-breaking.** `SplitDecisionTree` picks splits from a priority queue ordered by
objective improvement. When two candidate questions score equally — common with small stats
or symmetric phone sets — the winner depends on queue ordering and on floating-point
comparison of near-identical doubles. A different tie break changes the leaf assignment of a
handful of context windows, which changes which Gaussians see which frames, which perturbs
everything downstream. The leaf *count* is stable (`round_num_leaves = true`); individual
assignments are not perfectly so. Read `build-tree-utils.cc` line by line and match Kaldi's
comparison and insertion order exactly, which removes most of it.

**GPU float ordering.** The GMM scoring GEMM sums over `1 + 2d` terms. A tiled GPU kernel
accumulates them in a different order than a CPU dot product, and f32 addition is not
associative, so results differ in the last bits. This is bounded at 1e-4 relative per
`(frame, pdf)` — the `device` contract — and does not change Viterbi paths except where two
paths are within that of each other, which is rare and produces boundary shifts of at most one
frame. Parity runs should use `--cpu` when comparing against MFA, and compare GPU against CPU
separately.

**Pronunciation variants.** v1 keeps only the first pronunciation of each dictionary word,
while MFA's lexicon FST offers all of them and lets the decoder choose. For words where MFA
picks a non-first variant, the phone sequence itself differs and boundary comparison is
meaningless. The parity script must either use a single-pronunciation dictionary, or exclude
utterances containing multi-pronunciation words from the boundary statistics and report how
many were excluded. Do not average this away silently.

**RNG-dependent steps.** Gaussian splitting perturbs means with random noise, `EqualAlign`
draws the optional silences at random (viter's walk is deterministic,
[TRAINING.md](TRAINING.md)), and LDA/MLLT accumulation prunes randomly (`random_prune = 4.0`). viter
uses a seeded `Xoshiro256PlusPlus`, so viter is reproducible against itself, but its stream
is not Kaldi's. Total log-likelihood and leaf counts are robust to this; individual Gaussian
means are not, so never compare those directly.

**Accumulator precision.** Kaldi accumulates GMM, tree, LDA, MLLT and fMLLR statistics in
`double` and stores models in `float`. viter follows exactly (`f32` in features and models,
`f64` accumulators). Deviating from this — accumulating in f32 to save memory — shows up
first as a monophone log-likelihood curve that drifts a fraction of a percent low, which is
why that check is on the curve and not just the final value.
