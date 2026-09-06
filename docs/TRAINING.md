# Training

viter reimplements MFA's four-stage GMM-HMM recipe. Every hyperparameter below is MFA's default, cited to `plans/research/01_kaldi_surface.md` and the MFA python it summarizes (`plans/mfa/montreal_forced_aligner/acoustic_modeling/{monophone,triphone,lda,sat}.py`). `TrainConfig::default()` carries these values, each with a `file:line` comment.

## Prerequisites: topology and graph

Built once from the corpus symbol table, before stage 1.

- **Topology** (`HmmTopology::mfa_default`, from MFA `dictionary/mixins.py:669` `_write_topo`):
  - Silence phones: 5 states. State 0 fans out uniformly (1/(N−1)) to states 0..N−2; states
    1..N−2 fan out to 1..N−1; the final emitting state self-loops 0.75 / exits 0.25.
  - Non-silence phones: 3 states, Bakis left-to-right, self-loop 0.5 / forward 0.5, last
    emitting state forward 1.0.
- **Graph** (`hmm::build_graph`, see [ARCHITECTURE.md](ARCHITECTURE.md#why-no-fsts)):
  `transition_scale = 1.0`, `self_loop_scale = 0.1`, `silence_probability = 0.5`,
  `initial_silence_probability = 0.5`, plus the final silence/non-silence corrections.

## Stage 1 — Monophone

No context: one `ContextDependency` with N=1, P=0, built by `monophone_shared` over the phone sets, so all position variants of a phone share pdfs.

1. **Init** (Kaldi `gmm_init_mono`): compute a global single-Gaussian mean and variance from
   the first ~10 utterances; every pdf starts as a copy of that one Gaussian. Build the
   `TransitionModel` from `(ctx, topo)`.
2. **Flat start** (`align::equal_align`, Kaldi `EqualAlign`): a random path through the graph
   with roughly equal per-phone durations, giving the first alignment without a trained model.
3. **Iteration 0**: accumulate GMM + transition stats from the flat-start alignment, MLE update
   with `min_gaussian_occupancy = 3.0` (only this iteration; the kalpy default of 10.0 applies
   afterwards).
4. **Iterations 1..39**: on the realign schedule, re-align the subset with silence boosting;
   otherwise reuse the previous alignment. Accumulate, MLE update, and grow the Gaussian count
   linearly from `initial_gaussians` toward `max_gaussians` (MFA's `_increment_gaussians`) via
   `AmDiagGmm::split_by_count` with `power = 0.25`.

Realign iterations: `[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29,32,35,38]`.

**Produces**: a monophone `TransitionModel` + `AmDiagGmm`, and subset alignments that become the triphone tree statistics.

## Stage 2 — Triphone

1. **Tree stats** (`tree::accumulate_tree_stats`, Kaldi `AccumulateTreeStats`): from the
   monophone alignments and delta features, accumulate `GaussClusterable` stats keyed by
   `(context window, pdf_class)` with N=3, P=1. Silence and OOV phones are
   context-independent symbols.
2. **Questions** (`automatically_obtain_questions`, Kaldi `build-tree.cc:615`): cluster phones
   by their central-state (pdf-class 1) stats into a binary tree and emit the phone sets. MFA
   then dedups them and appends its `extra_questions` mapping.
3. **Tree** (`tree::build_tree`, Kaldi `BuildTree`): questions for context positions 0..N−1
   plus pdf-class questions `[[0],[0,1]]`; `thresh = 300.0`, `max_leaves = num_leaves`,
   `cluster_thresh = -1`, `P = 1`, `round_num_leaves = true`. Roots come from `tree::mfa_roots`:
   silence phones share one unsplit root; each non-silence phone (with all its position
   variants) gets a shared, splittable root.
4. **Model init** (Kaldi `gmm_init_model`): split tree stats by leaf, sum per leaf, build one
   Gaussian per leaf with `var_floor = 0.01`, `min_count = 20.0`, `power = 0.2`, then mix up to
   `initial_gaussians`.
5. **Alignment conversion** (`hmm::convert_alignment`): map the monophone alignments onto the
   new tree and transition model rather than re-aligning from scratch.
6. **Iterate** like monophone for 35 iterations, realigning every 10.

**Produces**: a decision tree (`EventMap`), triphone `TransitionModel` + `AmDiagGmm`, and triphone alignments.

## Stage 3 — LDA + MLLT

Feature pipeline switches to spliced MFCC: `splice(3, 3)` on 13-dim CMVN'd MFCC → 91 dims, projected to 40.

1. **LDA stats** (`transform::LdaEstimate`) from the triphone alignments on spliced features, with `random_prune = 4.0` and silence phones as a restricted set.
2. **Estimate** `LdaEstimate::estimate` → a `[40, 91(+1)]` transform (Kaldi
   `transform/lda-estimate.cc:85`), via Cholesky whitening plus a self-adjoint eigen-solve.
3. **Retrain** the triphone model on the LDA features (new tree, `num_leaves = 2500`).
4. **MLLT** at iterations `[2, 4, 6, 12]`: accumulate `MlltAccs`, `update()` the matrix,
   `compose_transforms` it onto the running LDA matrix, and `transform_means` on the model so
   the existing Gaussians stay valid under the new projection.

**Produces**: a composed 40-dim LDA+MLLT matrix, a new tree, and an LDA-space model.

## Stage 4 — SAT (speaker-adaptive training with fMLLR)

1. **Retrain** on LDA features (`num_leaves = 2500`, `power = 0.2`).
2. **fMLLR** at iterations `[2, 4, 6, 12]`: per speaker, accumulate `FmllrDiagGmmAccs` from the
   current alignments (silence posteriors down-weighted), then
   `ComputeFmllrMatrixDiagGmmFull` — `update_type = full`, `min_count = 500.0`,
   `num_iters = 40`. Subsequent iterations train on the adapted features.
3. **Realign** at iterations `[10, 15]`.
4. **SI alignment model** (`am_si`, MFA's `final.alimdl`): a two-feature accumulation pass
   (Kaldi `gmm-acc-stats-twofeats`) taking posteriors from the *adapted* features but stats
   from the *speaker-independent* features, then an MLE update with
   `remove_low_count_gaussians = false`. This is what the first alignment pass uses before any
   fMLLR transform exists for a new speaker.

**Produces**: the final adapted model `am`, the SI model `am_si`, the fMLLR options, and the tree/transform needed to reproduce the pipeline at alignment time.

## Final pass

The full corpus (not a subset) is aligned with the final model — two passes when `am_si` is
present — and those alignments are returned in `Trained.alignments`, written as TextGrids if
`--out-textgrids` was given.

## Hyperparameters

| | mono | tri | lda+mllt | sat |
|---|---|---|---|---|
| subset (utts) | 2000 | 5000 | 10000 | 10000 |
| iterations | 40 | 35 | 35 | 35 |
| num_leaves | — | 1000 | 2500 | 2500 |
| initial_gaussians | 135 | = num_leaves | = num_leaves | max_gaussians / 2 |
| max_gaussians | 1000 | 10000 | 15000 | 15000 |
| power | 0.25 | 0.25 | 0.25 | 0.2 |
| boost_silence | 1.25 | 1.25 | 1.0 | 1.0 |
| cluster_threshold | — | −1 | −1 | −1 |
| realign iterations | see schedule above | every 10 | as tri | 10, 15 |
| final gaussian iter | — | iters − 10 | iters − 10 | iters − 5 |
| initial_beam | 6 | — | — | — |
| min_gaussian_occupancy | 3.0 (iter 0) | 10.0 | 10.0 | 10.0 |
| feature dim | 39 | 39 | 40 | 40 |
| stage-specific | — | — | lda_dim 40, splice ±3, random_prune 4.0, mllt iters [2,4,6,12] | fmllr iters [2,4,6,12], full update, min_count 500, 40 inner iters |

Alignment-time defaults (`AlignOptions`, MFA `alignment/mixins.py:69-91`): `beam = 10`,
`retry_beam = 40`, `acoustic_scale = 0.1`, `transition_scale = 1.0`, `self_loop_scale = 0.1`,
`boost_silence = 1.0`, `careful = false`. Transition MLE update: `floor = 0.01`,
`mincount = 5.0`.

## Subset logic

Early stages train on a subset; the sizes above are MFA's. The subset is the first N utterances after a deterministic shuffle seeded from `TrainConfig.seed`, so runs are reproducible. When the corpus is smaller than a stage's subset size, the whole corpus is used.
`TrainConfig.subset = false` trains every stage on everything — slower, and it diverges from
MFA, so parity runs must leave subsetting on.

## The `.viter` model file

One postcard-serialized file prefixed with magic bytes `"VITR"` and a `u32` format version
(currently 1). It holds everything `align` needs and nothing it does not:

| field | purpose |
|---|---|
| `version` | format version |
| `phones`, `silence_phones`, `position_dependent` | symbol table (position-tagged if enabled) and which phones are silence |
| `mfcc` | the exact `MfccOptions` used, so features are reproduced bit-for-bit |
| `deltas` | `Some` for mono/tri models |
| `splice`, `lda` | `Some` for LDA/SAT models: the ±3 splice and the composed LDA+MLLT matrix |
| `topo`, `ctx`, `tm` | topology, decision tree / context dependency, transition model |
| `am` | the acoustic model (speaker-adapted for SAT) |
| `am_si` | the SI alignment model, present only for SAT — its presence is what triggers two-pass fMLLR alignment |
| `fmllr` | fMLLR options for the adaptation pass |
| `graph_opts` | silence probabilities and scales, so graphs are rebuilt identically |
| `meta` | free-form strings: corpus trained on, date, viter version, utterance count |

With an `out_dir`, one `.viter` is written per completed stage alongside a `train.log`, so a run can be inspected or a later stage restarted without repeating earlier ones.
