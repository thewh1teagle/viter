<p align="center"><img src="assets/logo/logo.svg" width="72"></p>

<h1 align="center">viter</h1>

<p align="center">A forced aligner in one Rust binary. The Montreal Forced Aligner recipe, on your GPU.</p>

|  | Montreal Forced Aligner | viter |
|---|---|---|
| Setup | conda, Kaldi, Python | one static binary |
| LJSpeech (13,093 utts / 23.9 h), same box | 42 min | **4 min 36 s** |
| Viewer | open Praat yourself | `viter serve` |

Measured on an NVIDIA DGX Spark, same corpus and dictionary for both. viter: 4 min 36 s wall clock, 6.7 GB peak RSS, GPU via wgpu Vulkan. MFA 3.x: 2,502 s. Aligning with a trained model runs at RTF ≈ 0.002, 24 hours of audio in about three minutes.

## Quick start

```
cargo binstall viter      # prebuilt binary from the GitHub release
cargo install --path .       # or build from source
```

```
viter train corpus/ --dict dict.txt -o model.viter
viter align corpus/ model.viter -o out/
viter serve out/
```

`serve` opens a local page: waveform on top, `words` and `phones` tiers below on the same time scale, click a phone to hear it, keyboard driven.

A corpus is audio files with transcripts of the same stem next to them:

```
corpus/
  speaker1/
    utt1.wav
    utt1.txt
    utt2.flac
    utt2.txt
```

`.wav`, `.flac`, `.mp3` pair with `.txt` or `.lab`. With a dictionary, transcripts are words. Without one, every whitespace token is a phoneme, so viter works for any language given phone-level transcripts.

## Why

MFA works, but it costs an environment to install and most of an hour to train, and the output is a folder of TextGrids you have to open somewhere else. viter is a single binary with no Python, no conda, no Kaldi, and no CUDA toolkit: the GPU path is wgpu (Vulkan / Metal / DX12), and the CPU fallback always works. Output is Praat TextGrids with `words` and `phones` tiers, MFA-compatible. On the full LJSpeech check the phone label sequences matched MFA's exactly on all 13,087 files.

## How it works

- **monophone**. Flat start from equal alignment, then Viterbi realignment with growing Gaussian counts.
- **triphone**. Decision-tree state tying over phone context.
- **LDA+MLLT**. Spliced features projected to 40 dimensions with maximum-likelihood linear transforms.
- **SAT/fMLLR**. Speaker-adaptive training, plus a speaker-independent model for the first alignment pass.

Kaldi-exact MFCC, diagonal-covariance GMMs, transition models, HMM topologies, the tree builder and the beam Viterbi decoder are ported from Kaldi's C++ rather than reinvented. Parity with MFA is the definition of done and is tracked in [docs/PARITY.md](docs/PARITY.md).

## Status

Pre-alpha. The full recipe trains and aligns end to end and the viewer works. Still to come: pronunciation variants, silence and pronunciation probabilities, a word separator for phoneme mode, and GPU-side statistics accumulation. The CLI may still change.

Prebuilt binaries for Linux x86_64/aarch64, macOS arm64/x86_64 and Windows x86_64 are built by `.github/workflows/release.yml`; releases coming.

## Docs

[docs/README.md](docs/README.md) is the index. Start with [CLI.md](docs/CLI.md), [CORPUS-FORMAT.md](docs/CORPUS-FORMAT.md) and [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
