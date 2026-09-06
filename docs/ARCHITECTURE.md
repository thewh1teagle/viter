# Architecture

## Crates

| crate | path | role |
|---|---|---|
| `viter_kaldi` | `crates/kaldi` | The Kaldi port: audio, features, GMMs, HMM/topology/transition model, decision tree, transforms, decoder, device (CPU/GPU), model file. No I/O policy, no orchestration. |
| `viter_io` | `crates/io` | Corpus scanning, dictionary, TextGrid read/write, CTM export. |
| `viter_train` | `crates/train` | Orchestration: config, feature store, the four training stages, corpus alignment. |
| `viter_serve` | `crates/serve` | axum server for the viewer; serves the JSON API and the embedded web app. |
| binary | `src/main.rs` | clap CLI: `train`, `align`, `serve`. |
| web | `web/` | Vite + React + TS viewer, built to `web/dist`, embedded into the binary. |

`viter_kaldi` depends on nothing else in the workspace; `io` on `kaldi`; `train` on
`kaldi` + `io`; `serve` on `io`; the binary on all of them.

## Module map (`viter_kaldi`)

| module | contents |
|---|---|
| `types` | `Feats`, `PhoneId`, `PdfId`, `TransitionId`, `WordId`, `SymbolTable`, `Utterance`, `Alignment`, `PhoneInterval` / `WordInterval` / `IntervalAlignment`. Every shared small type lives here. |
| `audio` | symphonia decode (wav/flac/mp3), mono downmix, rubato resample to 16 kHz, wav write, slice. Samples stay in `-1..1`. |
| `feat` | Kaldi-exact MFCC (`MfccComputer`, `MfccOptions`), `CmvnStats` / `apply_cmvn`, `add_deltas`, `splice`, `apply_transform`. Multiplies by 32768 internally to match Kaldi's int16 scale. |
| `gmm` | `DiagGmm`, `AmDiagGmm` (+ `packed()` for the device), `AccumDiagGmm`, `AccumAmDiagGmm`, MLE updates, split/merge by count, `boost_silence`. |
| `hmm` | `HmmTopology`, `ContextDependency`, `TransitionModel`, `Graph`/`GraphState`/`Arc`, `build_graph`, `split_to_phones`, `convert_alignment`, `to_intervals`. |
| `tree` | `EventMap`, `GaussClusterable`, tree stats accumulation, `automatically_obtain_questions`, `build_tree`, `mfa_roots`, stats splitting helpers. |
| `transform` | `LdaEstimate`, `MlltAccs`, `FmllrDiagGmmAccs`, `transform_means`, `compose_transforms`. |
| `align` | Beam Viterbi (`align`) over `hmm::Graph`, `graph_pdfs`, `equal_align` (flat start). |
| `device` | `Device::auto/cpu/gpu`, `score`, `score_batch`, `score_components`, `gemm_abt`. CPU via faer, GPU via wgpu WGSL. |
| `model` | `AcousticModel` — the `.viter` file: save/load, postcard + `"VITR"` magic + version. |

`viter_train` splits as `config.rs`, `pipeline.rs` (feature store + shared stage helpers), and `mono.rs` / `tri.rs` / `lda.rs` / `sat.rs`, each exposing one `run(ctx, cfg)`.

## Data flow: wav → TextGrid

Alignment with a trained model (`viter align`):

```
corpus dir ──io::scan──> Corpus { utts, speakers, phones: SymbolTable, silence_phones }
                              │  transcript ─(dictionary | phoneme-string)─> Vec<Vec<PhoneId>>
                              ▼
audio file ──audio::read_16k──> Vec<f32> ──feat::MfccComputer::compute──> [frames, 13]
                              ▼
                  CmvnStats per speaker ──apply_cmvn──> normalized MFCC
                              ▼
        stage features: deltas (39)  |  splice(3,3) → LDA/MLLT (40) → fMLLR (40)
                              ▼
hmm::build_graph(words_phones, tm, ctx, opts) ──> Graph (HMM chain + optional-sil branches)
                              ▼
align::graph_pdfs(graph) ──> pdf list ──Device::score(feats, am, pdfs)──> [frames, pdfs]
                              ▼
align::align(graph, tm, scores, ...) ──> Alignment { tids, words, loglike }
                              ▼
hmm::to_intervals(tm, ali, words_phones, frame_shift) ──> IntervalAlignment
                              ▼
io::textgrid::from_alignment ──> TextGrid { "words", "phones" } ──> x.TextGrid
                                          (or io::ctm::write_ctm with --ctm)
```

With a SAT model (`am_si` present) alignment is two passes: align on the speaker-independent
model, estimate a per-speaker fMLLR transform from those alignments, then re-align on the
adapted features with `am`. This mirrors MFA.

Training follows the same path per utterance but loops: align → accumulate → update, with
the graph rebuilt only when the tree or transition model changes.

## Where the GPU is used

Exactly one place: GMM log-likelihood scoring, in `viter_kaldi::device`. It is the hot
loop — every training iteration scores every frame of every utterance against every pdf its
graph touches.

The diagonal-covariance Gaussian log-likelihood

```
ll = gconst_g + Σ_d mean_g[d]·invvar_g[d]·x[d] − 0.5·Σ_d invvar_g[d]·x[d]²
```

is rearranged into a single GEMM: pack each Gaussian as a row `[gconst, mean·invvar…, −0.5·invvar…]`
(that is `AmDiagGmm::packed()`, cached and invalidated by `am.version()`), pack each frame as
`[1, x…, x²…]`, and compute `[frames, 1+2d] × [gauss, 1+2d]ᵀ`. A second segmented-logsumexp
pass reduces per-Gaussian scores to per-pdf scores. Log weights are folded into `gconst`, as
Kaldi does in `ComputeGconsts`.

On GPU this is one tiled WGSL compute shader (16×16 tiles) plus one segmented-logsumexp
shader, with the packed GMM matrix kept resident across calls and features uploaded per
call; buffers are capped at 256 MB and frames chunked past that. `wgpu` picks Vulkan, Metal or DX12 at
runtime, so there is no CUDA or vendor SDK build dependency. `Device::auto()` tries for a
high-performance adapter and falls back to the faer + rayon CPU path, logging which it
chose. GPU and CPU must agree to 1e-4 relative per `(frame, pdf)`.

Everything else — tree building, MLE updates, LDA/MLLT/fMLLR estimation, Viterbi — stays on
the CPU, parallelised with rayon **across utterances only**, never across frames within one
utterance (frames are sequentially dependent in Viterbi and reordering them changes float sums).

## Why no FSTs

MFA builds a lexicon FST in pynini and composes `H ∘ C ∘ L ∘ G` per utterance, then runs a
plain `FasterDecoder` over it. The decoder only ever asks for the arcs out of a state, an
input label (transition-id), an output label (word), an arc weight and a final weight;
determinization and minimization are size optimizations, nothing more.

viter takes a per-word list of candidate pronunciations, not a word lattice, and for that
input every FST op drops out:

- **No `L` composition** — the lexicon lookup already happened in `io`; each word arrives as its
  set of phone sequences, which the graph unions into parallel branches.
- **No determinization/minimization** — a chain of small pronunciation unions with a binary
  optional-silence branch at each junction stays small by construction, and there are no
  disambiguation symbols.
- **No `C` (context) transducer** — triphone context is known in closed form from neighbours.
  The one subtlety is optional-silence junctions, where the neighbour depends on which branch
  is taken; that is handled by a small local fan-out (two context variants per neighbour of an
  optional-silence slot).
- **No `H` transducer** — the 3-state Bakis / 5-state silence topology is instantiated directly
  from `HmmTopology` per phone.
- **No `AddSelfLoops` / `AddTransitionProbs` passes** — `transition_scale` and `self_loop_scale`
  are applied while emitting each arc.

So `hmm::build_graph` constructs `[opt-sil] p1 [opt-sil] p2 … pN [opt-sil]` directly, with
pdf-ids from `ContextDependency` on the triphone triple, and `align::align` is a beam-pruned
token-passing Viterbi with backpointers over that sparse graph, retrying at `retry_beam` on
failure. Graph size is O(N phones), so the size optimizations were never needed.

What is given up: lattice word-alignment tooling and phonological rule FSTs. Word boundaries are
recovered by tagging the first emitting arc of each pronunciation's first phone with its
`WordId` *and* the index of that pronunciation, as MFA's `_create_word_fst` places word labels;
the Viterbi backtrace collects both, so `Alignment.prons` says which variant was chosen and
`to_intervals` copies it into `WordInterval.pron`.

The lexicon costs follow `_create_word_fst` / `create_fsts`
(`plans/kalpy/kalpy/fstext/lexicon.py:349-600`) arc for arc:

- **Pronunciation probability** — `|ln p|` with `p` floored at 0.01, on the arc entering the
  pronunciation.
- **Optional silence around a word** — MFA's three-state hub (start / non-silence / silence)
  means the junction between word `i` and word `i+1` combines word `i`'s *after* costs with word
  `i+1`'s *before* corrections: the silence path costs `-ln(silence_after_prob_i)` plus
  `-ln(silence_before_correction_{i+1})`, and the direct path `-ln(1 - silence_after_prob_i)`
  plus `-ln(non_silence_before_correction_{i+1})`. A pronunciation with no
  `silence_after_probability` falls back to the global `silence_probability`; absent corrections
  cost 0.
- **Ends** — `initial_silence_probability` at the start, and `final_silence_correction` /
  `final_non_silence_correction` as final weights on the two hub branches.

`rustfst` exists but lacks the Kaldi-specific ops (`DeterminizeStar`, `TableCompose`,
`InverseContextFst`, `GetHTransducer`, `MinimizeEncoded`, `AddSelfLoops`, `LatticeWeight`), so
hand-building the HMM graph is less code than reimplementing them. Revisit only if word-level
input with pronunciation variants is added — and even then a bespoke alternatives-graph
builder beats a general FST library.

## Feature pipelines per stage

All stages start from the same base: 16 kHz mono → 13-dim MFCC → per-speaker CMVN.

| stage | pipeline | dim |
|---|---|---|
| monophone | MFCC → CMVN → deltas (order 2, window 2) | 39 |
| triphone | MFCC → CMVN → deltas | 39 |
| LDA+MLLT | MFCC → CMVN → splice(3, 3) → LDA+MLLT transform | 40 |
| SAT | MFCC → CMVN → splice(3, 3) → LDA+MLLT → per-speaker fMLLR | 40 |

`FeatureStore` holds the base MFCC+CMVN for the whole corpus in RAM as f32 plus per-speaker
CMVN stats and (once estimated) per-speaker fMLLR transforms, and derives the stage-specific
features lazily via `FeatureKind::{Deltas, SpliceLda, SpliceLdaFmllr}`. The `.viter` model
records which pipeline it was trained with (`deltas`, `splice`, `lda`, `fmllr`) so
`align_corpus` reconstructs it exactly.
