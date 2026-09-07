# Bounded training optimization validation

Final retained changes preserve exact replay outputs while substantially reducing
LJSpeech adaptation time. No full-corpus training was used for these measurements.
All measurements below ran on the NVIDIA GB10, with 20 Rayon threads.

## Final measured results

| Workload | Original fMLLR accumulation | Final | Speedup |
| --- | ---: | ---: | ---: |
| LJSpeech, 128 utterances / 83,223 frames | 90.897 ms | 22.719 ms | 4.001x |
| LJSpeech, 256 utterances / 166,367 frames | 160.087 ms | 42.766 ms | 3.743x |
| TIMIT SAT, 16 speakers × 8 utterances / 38,864 frames | 130.526 ms | 131.053 ms | 0.996x |

The final dispatch selects separately compiled cached/uncached fMLLR kernels at
4,096 live frames per GPU batch. This retains the large-batch improvement without
the measured small-batch regression of using the cached kernel everywhere.
Original GMM accumulation remains unchanged. The unused-triangle fMLLR GEMM skip
also preserves the original summation order.

Scoring against the earliest pre-change executable improved 45.604 → 40.941 ms
for 128 utterances and 91.320 → 82.333 ms for 256 utterances (about 1.11x).
Corresponding production alignment was 51.851 → 44.436 ms and 103.741 →
94.165 ms. These comparisons span separate sessions; the larger adaptation gain
was repeatable across both sample sizes. Allocation-heavy accumulation timings
varied between otherwise idle runs, so small changes in those timings should not
be treated as demonstrated gains.

The inverse-only fMLLR solver change reduced the 40-dimensional single-speaker
solve from approximately 38 to 25 ms. Final fMLLR comparisons below use an
intermediate baseline that already contains this exact-output solver change and
the interleaved scoring layout. The feature cache is exercised by training-loop
tests and its separate benchmark, not by this frozen replay.

## Exactness evidence

All three final comparisons passed with **zero absolute and relative tolerance**:

- LJSpeech 128: 8,111,862 GMM/transition statistics and 70,521 fMLLR statistics and transform values.
- LJSpeech 256: 8,111,862 GMM/transition statistics and 70,521 fMLLR statistics and transform values.
- TIMIT SAT 16×8: 5,622,573 GMM/transition statistics and 1,128,336 fMLLR statistics and transform values.

Every transition path, emitted word and pronunciation choice matched exactly.
The GMM artifacts include occupancy, means, variances, transitions, likelihood and
frame totals. fMLLR artifacts include every count, K/G matrix element and solved
transform. Both LJSpeech fixtures had zero alignment failures.

Final artifact pairs:

| Baseline prefix | Final prefix |
| --- | --- |
| `/tmp/replay-ab-base128` | `/tmp/replay-final128` |
| `/tmp/replay-ab-base256` | `/tmp/replay-final256` |
| `/tmp/replay-sat-base16x8` | `/tmp/replay-final-timit` |

Preserved binaries are `/tmp/viter-perf-replay-baseline` (original hot phases),
`/tmp/viter-perf-replay-extra-dump-baseline` (original adaptation kernels, improved
solver/interleaved scoring), and `/tmp/viter-replay-dispatch` (final candidate).

## Reproduction

`crates/train/examples/perf_replay.rs` scans the corpus manifest, selects at most
256 evenly spaced utterances across sorted corpus order, and computes audio
features only for those utterances. It never runs a training schedule. For SAT
models it uses the speaker-independent acoustic model and LDA features, matching
the initial adaptation pass. Round zero warms kernels; reported phase comparisons
use medians of later repetitions. Run one benchmark at a time without concurrent
compilation, GPU tests, or training.

```sh
cargo build --release -p viter-train --example perf_replay
VITER_REPLAY_EXTRA=1 target/release/examples/perf_replay data/ljfull data/ljfull-v12.viter data/dict/ljspeech_ipa_noprobs.dict 128 5 64 /tmp/replay-candidate128 > /tmp/replay-candidate128.log
uv run plans/optimization/optimization_001.py /tmp/replay-ab-base128 /tmp/replay-candidate128
```

Save a pre-change executable before production edits, or build it from an isolated
checkout, to establish a baseline. `VITER_REPLAY_CPU=1` explicitly selects the CPU;
otherwise a missing GPU is an error.

Without `VITER_REPLAY_EXTRA`, phases cover model load, manifest scan, base/derived
features, graph construction, isolated scoring, isolated Viterbi, production
`align_batch`, and GMM accumulation on frozen alignments. With EXTRA enabled,
additional phases measure MLE on a fresh disposable model clone, fMLLR input
preparation, GPU accumulation and transform solve. MLE removal and splitting are
disabled because subset occupancy does not represent a full corpus. Updated
models and transforms never feed into subsequent repetitions.

An output prefix produces `.log` (redirected stdout), `.align`, `.stats`, and,
with EXTRA enabled, `.extra`. The standalone validator defaults to exact values;
explicit tolerances support investigating experimental kernels, but the final
retained candidate passes without them.

The TIMIT fixture `/tmp/viter-perf-timit16x8` contains symlinks to 8 original
utterances from each of 16 evenly spaced speakers. Its generated dictionary
`/tmp/viter-perf-timit16x8.dict` gives each original pipe-delimited phone group a
word key, preserving the phone sequence. It uses the actual adapted model
`data/timit.viter` (69,318 Gaussians, 3,624 PDFs, 40 dimensions). The LJSpeech SI
model has 100,041 Gaussians, 4,056 PDFs and 40 dimensions.

## Bounded training check

A separate complete training-schedule check on only 64 real utterances took
7.451 seconds for the original executable and 6.740 seconds for the candidate
(1.105x, process wall time). All 64 final TextGrids were byte-identical and there
were no alignment failures. The retained artifacts are under
`/tmp/viter-small-training-final`; the companion optimization_002 report records
the training check. This verifies an end-to-end bounded case, not a full-corpus
speedup estimate.

## Scope

Frozen replay verifies numerical outputs and hot-phase performance. It does not
measure full training convergence, initial monophone retry-heavy decoding, or the
training feature cache. It cannot establish a 2–3 minute full-corpus training time.
Those claims require separate evidence; full-corpus training was intentionally
excluded from this task's iteration loop.
