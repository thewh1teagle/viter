# viter docs

viter is a Rust forced aligner: a single binary that reimplements Montreal Forced
Aligner's GMM-HMM (Kaldi) training and alignment recipe, with GPU/CPU acceleration and
a built-in web viewer for the results.

| doc | what it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Crate and module map, wav → TextGrid data flow, where the GPU is used, why there are no FSTs, per-stage feature pipelines. |
| [TRAINING.md](TRAINING.md) | The MFA recipe stage by stage (mono → tri → LDA+MLLT → SAT), hyperparameter table, subset logic, `.viter` model contents. |
| [CORPUS-FORMAT.md](CORPUS-FORMAT.md) | Corpus folder layout, audio/transcript pairing, dictionary format, dictionary-free phoneme-string mode, speakers, position-dependent phones, silence and OOV. |
| [CLI.md](CLI.md) | Full flag reference for `train`, `align`, `serve`, examples, summary output and exit codes. |
| [PARITY.md](PARITY.md) | How numeric parity with MFA/Kaldi is verified per stage, the `plans/parity/` uv scripts, known sources of drift. |
| [VIEWER.md](VIEWER.md) | The `serve` web app: API endpoints, keyboard shortcuts, dev and build workflow. |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Workspace layout, AGENTS.md working rules, shadcn components, tests, where the reference sources live. |

Authoritative sources, in precedence order:

1. `plans/CONTRACTS.md` — every public signature and CLI surface. Signatures there are the law.
2. `plans/research/01_kaldi_surface.md` — the exact Kaldi/MFA surface being reimplemented, with file:line citations.
3. `plans/research/02_openfst_path.md` — why the FST layer is replaced by direct HMM-chain construction.
4. `plans/research/03_rust_stack.md` — the chosen crates and what was rejected.
5. `AGENTS.md` — working rules (file size, tooling, parallel agents).

These docs describe the design; where a doc and `plans/CONTRACTS.md` disagree, `plans/CONTRACTS.md` wins.
