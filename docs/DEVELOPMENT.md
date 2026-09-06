# Development

## Workspace layout

```
viter/
  Cargo.toml            cargo workspace root
  src/main.rs           the clap CLI binary (train / align / serve)
  crates/
    kaldi/              viter_kaldi — the Kaldi port
    io/                 viter_io — corpus, dictionary, TextGrid, CTM
    train/              viter_train — config, feature store, the four stages
    serve/              viter_serve — axum API + embedded web app
  web/                  Vite + React viewer, built to web/dist
  data/                 small committed fixtures for tests and parity runs
  plans/                reference sources and validation scripts (see below)
  AGENTS.md             working rules
  plans/CONTRACTS.md          every public signature and the CLI surface
  docs/                 you are here
```

## Rules from AGENTS.md

**Tooling.** JavaScript is pnpm only. Python is standalone `uv` scripts (`uv run script.py`)
with inline dependency metadata — never a shared virtualenv or a requirements file. Tasks go
through `chore` (`chore list`).

**File size: 700 lines maximum.** On hitting the limit, ask before splitting. Split by
responsibility — find where the file does two jobs and move one out whole. Not a line-count cut
at an arbitrary boundary, and not a `utils.rs` skim. Keep the public API where callers expect
it, and move tests along with the code they test. A module may become a directory
(`feat/mod.rs`, `feat/mfcc.rs`, …). `lib.rs` already declares the top-level modules — do not
edit it.

**Working in parallel.** Large work splits across ~4–5 subagents, one disjoint module group
each, no shared files. Only the coordinator touches shared and root configuration (`Cargo.toml`,
`lib.rs`, `plans/CONTRACTS.md`). Contracts — interfaces and design docs — are fixed *before* the
agents start, which is why `plans/CONTRACTS.md` exists and why its signatures are the law: add
private helpers freely, but do not rename or change a public item. If a signature turns out to
be impossible, implement the closest thing and leave a `// CONTRACT-DEVIATION:` comment on it
so the coordinator can reconcile.

**ETAs in minutes, never days.** Single tasks 1–10 minutes; multi-agent work tens of minutes.

## Code conventions

From `plans/CONTRACTS.md`, and they are not negotiable because parity depends on them:

- `f32` in features and models; `f64` accumulators everywhere Kaldi uses `double` (GMM stats,
  tree stats, LDA/MLLT/fMLLR accumulators).
- Feature matrices are `ndarray::Array2<f32>`, `[frames, dim]`, row-major.
- Shared small types live in `viter_kaldi::types`; big structs live in their own module.
- Errors: a `thiserror` enum per module in library crates; `anyhow::Result` in `train` and
  `serve`.
- Logging via `tracing`. No `println!` in library crates — the CLI owns stdout.
- rayon parallelises **across utterances**, never across frames within one utterance.
- Randomness is `rand_xoshiro::Xoshiro256PlusPlus`, seeded from config. Deterministic runs are
  a requirement, not a nicety.

## Tests

Unit tests live in-file behind `#[cfg(test)]`, next to the code they cover.

```bash
cargo test                      # whole workspace
cargo test -p viter_kaldi    # one crate
cargo test -p viter_kaldi feat::mfcc   # one module
cargo clippy --all-targets
cargo fmt
```

Numeric parity is checked separately, by the `uv` scripts under `plans/parity/` — see
[PARITY.md](PARITY.md). Unit tests catch structural mistakes; parity scripts catch the numeric
ones, and only the parity scripts can tell you whether the port is actually correct.

Frontend: `cd web && pnpm lint` (oxlint) and `pnpm build` (which type-checks via `tsc -b`).

## Adding a shadcn component

shadcn/ui components are vendored into `web/src/components/ui`, not imported from a package,
so they can be edited freely.

```bash
cd web
pnpm dlx shadcn@latest add dialog
```

The component lands in `web/src/components/ui/dialog.tsx`; `web/components.json` holds the
aliases and style configuration it reads. Commit the generated file — it is now project source.
Prefer an existing component over a new dependency, and edit the vendored copy rather than
wrapping it in layers.

## Reference sources

Everything viter is ported from lives under `plans/`, checked out in full so it can be read
at `file:line`:

| path | what |
|---|---|
| `plans/kaldi/src` | Kaldi C++ — the ground truth for every algorithm |
| `plans/mfa` | Montreal Forced Aligner python — the recipe, schedules and hyperparameters |
| `plans/kalpy` | kalpy — MFA's pybind11 bindings over Kaldi; the clearest map from python call to C++ entry point |
| `plans/praatio` | praatio — reference TextGrid formatting |
| `plans/openfst` | OpenFst — read to confirm it is not needed, see research 02 |
| `plans/research/01_kaldi_surface.md` | the exact surface being reimplemented, with citations |
| `plans/research/02_openfst_path.md` | why the FST layer is replaced by direct HMM construction |
| `plans/research/03_rust_stack.md` | the chosen crates, and what was rejected and why |
| `plans/<name>/<name>_NNN.py` + `.md` | validation scripts, standalone `uv` |

When implementing anything numeric, open the Kaldi source and read it line by line. The
research notes say *where* to look and *what* the parameters are; they are a map, not a
substitute for the source. Comments in viter should cite the `file:line` they came from —
that is what makes a divergence findable a month later.
