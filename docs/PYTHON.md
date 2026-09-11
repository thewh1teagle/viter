# Python

The same aligner, in-process. One wheel per platform holds the whole engine — no Kaldi, no
conda, no separate binary download. Installing it also puts the `viter` command on PATH.

## Install

```
uv pip install viter
```

```
pip install viter
```

Or run the CLI without installing anything:

```
uvx viter --help
```

Wheels are `cp39-abi3`: Python 3.9 and newer, for linux x86_64/aarch64, macOS arm64/x86_64 and
Windows x86_64. The only runtime dependency is numpy.

Runnable end-to-end scripts live in [crates/python/examples/](../crates/python/examples/) — standalone `uv`
scripts, each one `uv run`-able as is.

## Quick start

```python
import viter

model = viter.train("corpus/", dict="dict.txt")
alignment = model.align("audio.wav", "the quick brown fox")
for word in alignment.words:
    print(f"{word.start:.3f} {word.end:.3f} {word.label}")
alignment.to_textgrid("audio.TextGrid")
```

## `train`

```python
viter.train(corpus_dir, out=None, *, dict=None, config=None, cpu=False, seed=None,
            no_tri=False, no_lda=False, no_sat=False, no_pron_probs=False,
            sat_rounds=None, no_subset=False, position_dependent=True,
            work_dir=None, progress=None, quiet=False) -> Model
```

Trains mono → tri → LDA+MLLT → SAT on `corpus_dir` and returns the model; writes it to `out`
too if given. It skips the final full-corpus alignment; use `model.align_corpus` afterwards
to write TextGrids. Every keyword mirrors the flag of the same name on `viter train` — see
[CLI.md](CLI.md) for what each one costs and buys. `dict` omitted means phoneme-string mode:
every token in a transcript is a phone.

`config` takes either a path to a `train.toml` or a dict with the same keys; unset fields keep
the MFA defaults.

```python
model = viter.train(
    "corpus/",
    "model.viter",
    dict="dict.txt",
    config={"mono": {"num_iterations": 20}},
    seed=1234,
)
```

Short schedule, CPU, no speaker adaptation — useful for a single-speaker corpus or a test:

```python
model = viter.train("corpus/", cpu=True, no_sat=True, sat_rounds=1, no_pron_probs=True)
```

`progress` is a callable taking one dict with the keys `done`, `total`, `fraction`,
`elapsed` (seconds), `eta` (seconds, or `None` until the first pass finishes), `stage`,
`step` and `mismatches` (a plan-vs-run disagreement counter, normally 0). `done`/`total`
count utterance-passes — one utterance processed by one pass — over the whole run, so
`done == total` (exactly once, at the end) means the run is over.
`fraction` is the elapsed share of the predicted total time, the same number the terminal bar
shows, and it never decreases. The callback fires at most ~10 times a second and on every
stage change; exceptions it raises are reported as unraisable and do not stop training.
`quiet=True` suppresses viter's own terminal bar while still calling `progress`.

```python
from tqdm import tqdm

bar = tqdm(total=1000, unit="permille")

def on_progress(info):
    bar.n = int(1000 * info["fraction"])
    bar.set_description(info["stage"])
    bar.refresh()

viter.train("corpus/", "model.viter", progress=on_progress, quiet=True)
```

The ETA is measured, not extrapolated from the raw counter: the plan of every pass the
schedule will run is fixed before training starts, each pass's cost (including the gaps
around it) is measured as it runs, a pass that has not run yet is predicted from measured
passes of the same kind — extrapolated in model size when the same kind ran in earlier
stages — and a kind that has never run from a built-in cost table scaled to this machine.
Measured on LJSpeech (13k utterances, GPU): never more than ~20% off after the first
minute, typically under 10%. On CPU the estimate is within ~15% from the triphone stage on.
On a small corpus (1k utterances) it can read ~30% low before the SAT rounds, whose cost is
not knowable until one has been measured.

Training releases the GIL, so other threads keep running, and Ctrl-C interrupts between stages.

## `Model`

```python
viter.Model(path, *, dict=None, cpu=False)
```

Loads a `.viter` model and probes the compute device once. Pass `dict=` to align with a
different pronunciation dictionary than the one training used; `cpu=True` skips GPU probing.

| attribute | |
|---|---|
| `phones` | `list[str]`, the phone inventory |
| `feature_dim` | `int`, feature dimension of the acoustic model |
| `speaker_adapted` | `bool`, whether the model has the SAT (fMLLR) stage |

```python
model = viter.Model("model.viter")
model.save("copy.viter")
```

### `Model.align`

```python
model.align(audio, text, *, sample_rate=None, beam=None, retry_beam=None,
            refine=True, speaker=None) -> Alignment
```

`audio` is a path, or a 1-D numpy array of float32/float64 (or a plain list) in `[-1, 1]`, in
which case `sample_rate` is required. Any sample rate works; it is resampled to 16 kHz
internally. `text` is the transcript, or a path to a `.txt`/`.lab` file.

```python
import numpy as np, soundfile as sf

samples, sr = sf.read("audio.wav", dtype="float32")
a = model.align(samples, "the quick brown fox", sample_rate=sr)
```

`refine=False` returns raw frame-grid boundaries (10 ms) instead of the 1 ms refined ones.
`beam` / `retry_beam` widen the search for hard utterances. Raises `ViterError` if the
utterance does not align.

### `Model.align_many`

```python
model.align_many(items, *, speakers=None, beam=None, retry_beam=None,
                 refine=True) -> list[Alignment | None]
```

`items` is a list of `(audio, text)` or `(audio, text, sample_rate)`. Returns one entry per
item, `None` where alignment failed. Utterances are aligned in parallel across cores.

```python
out = model.align_many([
    ("a.wav", "hello world"),
    ("b.wav", "goodbye world"),
])
```

**Speakers.** A single `align` call is speaker-independent: with a SAT model there is one
utterance to estimate fMLLR from, so adaptation has almost nothing to work with, and with a
non-SAT model there is no adaptation at all. To get the adaptation the CLI gets, batch the
utterances of one speaker together and label them:

```python
out = model.align_many(items, speakers=["spk1", "spk1", "spk2"])
```

Items sharing a speaker share one fMLLR transform. Without `speakers` every item is treated as
the same single speaker.

### `Model.align_corpus`

```python
model.align_corpus(corpus_dir, out_dir, *, ctm=False, beam=None, retry_beam=None,
                   refine=True) -> AlignSummary
```

What `viter align` does: walks the corpus, aligns every utterance with per-speaker adaptation,
and writes TextGrids (or CTM with `ctm=True`) into `out_dir`.

```python
s = viter.Model("model.viter").align_corpus("corpus/", "out/")
print(s.aligned, "/", s.utterances, "aligned;", len(s.failed), "failed")
for word, n in sorted(s.oov_words.items(), key=lambda kv: -kv[1])[:10]:
    print(n, word)
```

`AlignSummary` carries `utterances: int`, `aligned: int`, `failed: list[str]` (utterance ids)
and `oov_words: dict[str, int]`.

## `Alignment`, `Word`, `Phone`

```python
alignment.words     # list[Word]
alignment.phones    # list[Phone]
alignment.duration  # float, seconds
```

`Word` has `label`, `start`, `end` and `phones: list[Phone]`; `Phone` has `label`, `start`,
`end`. Times are seconds. Phone labels are untagged — the `_B`/`_I`/`_E`/`_S` position suffixes
are stripped, exactly as in the TextGrids.

```python
alignment.to_textgrid("out.TextGrid")   # write the long-format TextGrid
text = alignment.textgrid()             # the same thing as a string
```

## Plotting

```
uv pip install "viter[plot]"
```

```python
alignment.plot(audio, sample_rate=None, *, ax=None, tiers=("words", "phones"),
               spectrogram=True, zoom=None, n_mels=80, cmap="magma")
viter.plot(alignment, audio, ...)   # the same function, as a module-level call
```

`audio` is either a wav path or the 1-D float array you aligned (then `sample_rate` is
required). The top panel is a log-mel spectrogram (`spectrogram=False` draws the waveform
instead), with one labelled strip per tier below it, all sharing one time axis;
`zoom=(start, end)` limits the range in seconds. Returns the main axes.

```python
a = model.align("talk.wav", "hello world")
a.plot("talk.wav", zoom=(0.0, 2.0))
```

It is plain matplotlib, so in a notebook the figure renders inline; outside one, call
`matplotlib.pyplot.show()` or `savefig()` on the returned axes' figure. Importing `viter`
without matplotlib installed is fine — the error only comes when you call `plot`.

## `import_mfa`

```python
viter.import_mfa(path, out=None) -> Model
```

Converts an MFA acoustic model (a `.zip`) into a viter model, saving it to `out` if given.

```python
model = viter.import_mfa("english_mfa.zip", "english.viter")
```

## `serve`

```python
viter.serve(dir, *, port=7878, open=False, audio=None) -> None
```

Blocks, serving the viewer over the TextGrids in `dir`. `audio=` points at the corpus the audio
lives in when it is not beside the TextGrids. Ctrl-C returns from the call.

```python
viter.serve("out/", port=8000, audio="corpus/")
```

## `main`

```python
viter.main(argv=None) -> int
```

Runs the CLI in-process; `argv` defaults to `sys.argv`. This is what the installed `viter`
command calls.

```python
viter.main(["viter", "align", "corpus/", "model.viter", "-o", "out/"])
```

## Errors, the GIL and Ctrl-C

Every failure raises `viter.ViterError`. Long calls — `train`, every `align*`, `serve` — release
the GIL for their duration, so threads keep running and the whole thing parallelises across
utterances internally. Ctrl-C interrupts `serve` immediately and `train` between stages; a
single in-flight `align` finishes first.
