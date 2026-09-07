<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-dark.svg">
    <img src="assets/logo/logo-light.svg" width="72" alt="viter">
  </picture>
</p>

<h1 align="center">viter</h1>

<p align="center">A forced aligner in one Rust binary. The Montreal Forced Aligner recipe, on your GPU.</p>

viter trains an acoustic model on your corpus and aligns it to the phone, writing Praat TextGrids that drop straight into an MFA workflow. It is one static binary: no Python, no conda, no Kaldi, no CUDA toolkit. The GPU path is wgpu (Vulkan, Metal, DX12), and the CPU fallback always works.

## Install

```
cargo binstall viter        # prebuilt binary
cargo install --path .      # or build from source
```

## Quick start

```
viter train corpus/ --dict dict.txt -o model.viter
viter align corpus/ model.viter -o out/
viter serve out/
```

A corpus is audio with transcripts of the same stem beside it:

```
corpus/
  speaker1/
    utt1.wav
    utt1.txt
    utt2.flac
    utt2.txt
```

`.wav`, `.flac` and `.mp3` pair with `.txt` or `.lab`. Already have a Montreal Forced Aligner model? `viter import` brings it in.

## What you get

|  | Montreal Forced Aligner | viter |
|---|---|---|
| Setup | conda, Kaldi, Python | one static binary |
| Train LJSpeech (13,093 utts, 23.9 h) | 42 min | **3 min** |
| Peak memory | | 6.5 GB |
| Align | | RTF 0.002 |

- **MFA-compatible output.** Praat TextGrids with `words` and `phones` tiers. On the full LJSpeech run, phone labels are identical to MFA's on every file.
- **A viewer in the binary.** `viter serve` opens a local page: waveform, tiers on the same time scale, click a phone to hear it, keyboard driven.
- **Any language.** Use a pronunciation dictionary, or skip it and give phone strings directly, with `|` marking word boundaries.
- **Runs you can watch.** Cargo-style progress with an overall ETA, and `--log FILE` when you want the detail on disk.

## How it works

- **Monophone.** Flat start from equal alignment, then Viterbi realignment with growing Gaussian counts.
- **Triphone.** Decision-tree state tying over phone context.
- **LDA+MLLT.** Spliced features projected to 40 dimensions with maximum-likelihood linear transforms.
- **SAT/fMLLR.** Speaker-adaptive training, plus a speaker-independent model for the first alignment pass.

Kaldi-exact MFCC, diagonal-covariance GMMs, transition models, HMM topologies, the tree builder and the beam Viterbi decoder are ported from Kaldi's C++ rather than reinvented. See [docs/TRAINING.md](docs/TRAINING.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Alignment quality is measured against MFA continuously: [docs/PARITY.md](docs/PARITY.md).

## Documentation

| | |
|---|---|
| [CLI](docs/CLI.md) | Every flag for `train`, `align` and `serve`, with exit codes |
| [Corpus format](docs/CORPUS-FORMAT.md) | Layout, dictionaries, phoneme-string mode, silence and OOV |
| [Training](docs/TRAINING.md) | The recipe stage by stage, hyperparameters, the `.viter` file |
| [Viewer](docs/VIEWER.md) | The `serve` app, its API and keyboard shortcuts |
| [Parity](docs/PARITY.md) | How numeric agreement with MFA is verified |
| [Development](docs/DEVELOPMENT.md) | Workspace layout, tests, working rules |

## License

MIT
