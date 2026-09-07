# Bounded training validation

`optimization_002.py` trains both supplied binaries on 64 evenly spaced LJSpeech
utterances, with the full default stage schedule and TextGrid export. It never
trains on the full corpus: the script requires a source corpus larger than its
sample and caps the sample at 128 utterances. All output lives in a new directory
under `/tmp`; existing models and datasets are untouched.

```sh
uv run plans/optimization/optimization_002.py /tmp/viter-opt-baseline /tmp/viter-opt-candidate
```

The gate requires identical successful utterances and byte-identical TextGrids.
Wall times are printed, but small-corpus timing does not establish full-LJSpeech
runtime: the fitted trees and Gaussian counts are much smaller. Frozen full-size
models in `perf_replay` provide complementary hot-path measurements.

## Result (2026-09-07, NVIDIA GB10)

Baseline 7.451 seconds, candidate 6.740 seconds (1.106x). Both completed the
full default schedule on the same 64 utterances with zero final alignment
failures; all 64 TextGrid files were byte-identical. Logs, models and stage
checkpoints are in `/tmp/viter-small-training-final`.
