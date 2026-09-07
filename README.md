<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-dark.svg">
    <img src="assets/logo/logo-light.svg" width="72" alt="viter">
  </picture>
</p>

<h1 align="center">viter</h1>

<p align="center">Forced alignment in one binary. The Montreal Forced Aligner recipe, on your GPU.</p>

|  | Montreal Forced Aligner | viter |
|---|---|---|
| Setup | conda, Kaldi, Python | one static binary |
| Train on LJSpeech, 23.9 h, same recipe | 42 min | **10 min** |
| Same, short schedule | — | **2 min** |
| Agreement with MFA on LJSpeech, phone boundaries within 20 ms | reference | **95.7%** |
| Phone boundaries within 25 ms of TIMIT hand labels, trained on TIMIT | 85.3% ([published](docs/TIMIT.md)) | **88.2%** |
| Output | Praat TextGrids | the same TextGrids |
| Viewer | Praat | `viter serve` |

## Install

```
cargo binstall --git https://github.com/thewh1teagle/viter viter
```

Or grab a binary from the [releases page](https://github.com/thewh1teagle/viter/releases).

## Use

```
viter train corpus/ --dict dict.txt -o model.viter
viter align corpus/ model.viter -o out/
viter serve out/
```

`corpus/` holds audio files with a same-named `.txt` transcript beside each one. Without a dictionary every token is a phoneme, so any language works. An existing MFA model loads with `viter import`.

## Documentation

| | |
|---|---|
| [CLI](docs/CLI.md) | Every flag for `train`, `align`, `import` and `serve` |
| [Corpus format](docs/CORPUS-FORMAT.md) | Layout, dictionaries, phoneme mode |
| [Training](docs/TRAINING.md) | The recipe, stage by stage |
| [Architecture](docs/ARCHITECTURE.md) | Crates, GPU path, no FSTs |
| [Viewer](docs/VIEWER.md) | The `serve` app and its shortcuts |
| [Parity](docs/PARITY.md) | Agreement with MFA, measured |
| [TIMIT](docs/TIMIT.md) | Accuracy against hand-labelled phone boundaries |
| [Development](docs/DEVELOPMENT.md) | Workspace, tests, rules |

MIT.
