<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-dark.svg">
    <img src="assets/logo/logo-light.svg" width="72" alt="viter">
  </picture>
</p>

<h1 align="center">viter</h1>

<p align="center">Forced alignment in one binary. The Montreal Forced Aligner recipe, on your GPU.</p>

<p align="center">
  <a href="https://pypi.org/project/viter/"><img src="https://img.shields.io/pypi/v/viter" alt="PyPI"></a>
  <a href="https://github.com/thewh1teagle/viter/releases"><img src="https://img.shields.io/github/v/release/thewh1teagle/viter" alt="release"></a>
</p>

## Install

```
uv pip install viter
```

The wheel ships the `viter` command and the Python API. For the binary alone,
`cargo binstall --git https://github.com/thewh1teagle/viter viter` or the
[releases page](https://github.com/thewh1teagle/viter/releases).

## Use

```
viter train corpus/ --dict dict.txt -o model.viter
viter align corpus/ model.viter -o out/
viter serve out/
```

`corpus/` holds audio files with a same-named `.txt` transcript beside each one. Without a dictionary every token is a phoneme, so any language works. An existing MFA model loads with `viter import`.

Pretrained English models and dictionaries: [models-v1.0](https://github.com/thewh1teagle/viter/releases/tag/models-v1.0).

## Python

```python
import viter

model = viter.train("corpus/", dict="dict.txt")
alignment = model.align("audio.wav", "the quick brown fox")
alignment.to_textgrid("audio.TextGrid")
```

Examples in [crates/python/examples/](crates/python/examples/).

## Documentation

[docs/](docs/) — CLI flags, corpus format, the Python API, the training recipe, and how the numbers were measured.

MIT.
