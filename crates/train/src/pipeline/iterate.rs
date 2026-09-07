//! The per-iteration GMM training loop shared by mono, tri, lda and sat.
//!
//! MFA's `train_iteration` (`base.py:367-381`): realign when the iteration is in the
//! realignment list, run any stage-specific work (MLLT, fMLLR), accumulate, update,
//! then increment the gaussian target.
//!
//! Every pass over the stage's utterances walks speaker-aligned chunks (see
//! [`super::chunk`]) and derives each chunk's feature view only for as long as that
//! chunk is being processed. Built-in stages retain at most 2 GiB of unchanged
//! derived chunks between passes; larger corpora continue streaming the rest.
//! The public custom-hook entry point retains the uncached behavior.

use anyhow::Result;
use std::{borrow::Cow, time::Instant};
use viter_kaldi::types::{Alignment, Feats};

use super::{
    GraphSet, IterationSummary, StageCtx, Stats, UpdateOptions, align, chunk, progress, stats,
};
use crate::config::{GaussianSchedule, Stage};

/// Retained derived data is bounded independently of corpus size. Chunks that do
/// not fit keep their original streaming lifetime, so peak extra memory is this
/// budget plus one currently derived chunk (and its temporary transform inputs).
const DERIVED_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;

struct DerivedCache {
    chunks: Vec<Option<Vec<Feats>>>,
    budget: usize,
    retained: usize,
    full: bool,
}

impl DerivedCache {
    fn new(chunks: usize, budget: usize) -> Self {
        Self {
            chunks: (0..chunks).map(|_| None).collect(),
            budget,
            retained: 0,
            full: false,
        }
    }

    fn clear(&mut self) {
        for chunk in &mut self.chunks {
            *chunk = None;
        }
        self.retained = 0;
        self.full = false;
    }

    fn get(&mut self, index: usize, derive: impl FnOnce() -> Vec<Feats>) -> Cow<'_, [Feats]> {
        if self.chunks[index].is_none() {
            let feats = derive();
            let bytes = feats
                .capacity()
                .saturating_mul(size_of::<Feats>())
                .saturating_add(
                    feats
                        .iter()
                        .map(|f| f.len().saturating_mul(size_of::<f32>()))
                        .fold(0usize, usize::saturating_add),
                );
            if self.budget == 0 || self.full || bytes > self.budget.saturating_sub(self.retained) {
                self.full = true;
                return Cow::Owned(feats);
            }
            self.retained += bytes;
            self.chunks[index] = Some(feats);
        }
        // Never evict a useful chunk to admit a later one: cyclic full-corpus
        // scans would otherwise thrash whenever the corpus exceeds the budget.
        Cow::Borrowed(self.chunks[index].as_deref().expect("cached above"))
    }
}

/// The per-iteration loop shared by mono, tri, lda and sat.
///
/// A stage supplies its schedule and two hooks, and this drives MFA's
/// `train_iteration` (`base.py:367-381`): realign when the iteration is in the
/// realignment list, run any stage-specific work (MLLT, fMLLR), accumulate, update,
/// then increment the gaussian target.
pub struct IterationPlan<'p> {
    pub stage: Stage,
    pub num_iterations: usize,
    pub realignment_iterations: Vec<usize>,
    pub gaussians: GaussianSchedule,
    pub power: f32,
    pub boost_silence: f32,
    /// Beam override for the first iteration (monophone only).
    pub initial_beam: Option<f32>,
    /// min_gaussian_occupancy for iteration updates (10.0 except mono iteration 0).
    pub min_gaussian_occupancy: f64,
    pub utts: &'p [usize],
}

/// Stage-specific work that runs inside an iteration.
///
/// An iteration walks the stage's utterances one speaker-aligned chunk at a time
/// (see [`chunk`]), deriving that chunk's features, aligning, running the hooks and
/// accumulating before advancing to the next chunk. Built-in stages can retain
/// a bounded cache of unchanged features. Hooks see three parts: `begin_iteration`
/// once before the chunks, `accumulate_chunk` per chunk with that chunk's
/// utterances, alignments and features, and `end_iteration` once after them.
pub trait IterationHooks {
    /// Called once at the start of an iteration, before any chunk is processed.
    /// Returns true when this iteration needs per-chunk work (`accumulate_chunk` and
    /// `end_iteration` are then called); false skips both.
    fn begin_iteration(&mut self, _ctx: &mut StageCtx<'_>, _iteration: usize) -> Result<bool> {
        Ok(false)
    }

    /// Stage-specific accumulation over one chunk. `utts` are the chunk's corpus
    /// indices and `alignments`/`feats` are indexed alongside them.
    fn accumulate_chunk(
        &mut self,
        _ctx: &StageCtx<'_>,
        _iteration: usize,
        _utts: &[usize],
        _alignments: &[Option<Alignment>],
        _feats: &[Feats],
    ) -> Result<()> {
        Ok(())
    }

    /// Called once after the last chunk of an iteration; this is where a hook that
    /// accumulated across chunks performs its update (MLLT re-estimation, fMLLR
    /// transform installation). Any feature-space change it makes is picked up
    /// automatically because the next chunk/iteration derives features afresh.
    fn end_iteration(&mut self, _ctx: &mut StageCtx<'_>, _iteration: usize) -> Result<()> {
        Ok(())
    }
}

/// A hooks implementation that does nothing (mono, tri).
pub struct NoHooks;
impl IterationHooks for NoHooks {}

/// Run a stage's training iterations.
///
/// Every pass over `plan.utts` — realignment, the hooks, and accumulation — walks
/// speaker-aligned chunks (see [`chunk`]) and derives each chunk's feature view
/// (deltas, splice+LDA, +fMLLR) only for as long as that chunk is being processed.
/// Per-utterance work and utterance order are unchanged, so the result is the same
/// as a single whole-subset pass; only the peak memory differs. The graphs are built
/// once for the whole subset (they are small) and indexed per chunk.
pub fn run_iterations(
    ctx: &mut StageCtx<'_>,
    plan: &mut IterationPlan<'_>,
    hooks: &mut dyn IterationHooks,
    feats_kind: &dyn Fn(&StageCtx<'_>, &[usize]) -> Vec<Feats>,
    graphs: &mut GraphSet,
    initial_alignments: Vec<Option<Alignment>>,
) -> Result<Vec<Option<Alignment>>> {
    run_iterations_impl(ctx, plan, hooks, feats_kind, graphs, initial_alignments, 0)
}

/// Built-in stage entry point. Its feature closures depend only on the base
/// features and transforms, and its hooks install transforms in end_iteration.
/// Custom callers retain the uncached public entry point above.
pub(crate) fn run_iterations_cached(
    ctx: &mut StageCtx<'_>,
    plan: &mut IterationPlan<'_>,
    hooks: &mut dyn IterationHooks,
    feats_kind: &dyn Fn(&StageCtx<'_>, &[usize]) -> Vec<Feats>,
    graphs: &mut GraphSet,
    initial_alignments: Vec<Option<Alignment>>,
) -> Result<Vec<Option<Alignment>>> {
    run_iterations_impl(
        ctx,
        plan,
        hooks,
        feats_kind,
        graphs,
        initial_alignments,
        DERIVED_CACHE_BYTES,
    )
}

fn run_iterations_impl(
    ctx: &mut StageCtx<'_>,
    plan: &mut IterationPlan<'_>,
    hooks: &mut dyn IterationHooks,
    feats_kind: &dyn Fn(&StageCtx<'_>, &[usize]) -> Vec<Feats>,
    graphs: &mut GraphSet,
    initial_alignments: Vec<Option<Alignment>>,
    cache_bytes: usize,
) -> Result<Vec<Option<Alignment>>> {
    let stage_name = plan.stage.name();
    let mut alignments = initial_alignments;
    let opts = ctx.cfg.align.clone();
    let batch = ctx.cfg.batch_utts;

    // Chunks of positions into `plan.utts` (and therefore into `graphs` and
    // `alignments`), so the whole-subset graph set can be sliced per chunk.
    let chunks = chunk::by_frames(&ctx.feats, plan.utts, chunk::frames_for(ctx.cfg));
    let pos_of: std::collections::HashMap<usize, usize> =
        plan.utts.iter().enumerate().map(|(p, &u)| (u, p)).collect();
    let chunk_pos: Vec<Vec<usize>> = chunks
        .iter()
        .map(|c| c.iter().map(|u| pos_of[u]).collect())
        .collect();

    let mut feature_cache = DerivedCache::new(chunks.len(), cache_bytes);
    for iteration in 1..=plan.num_iterations {
        let started = Instant::now();
        let mut failed = alignments.iter().filter(|a| a.is_none()).count();

        let realign = plan.realignment_iterations.contains(&iteration);
        let beam = if iteration == 1 {
            plan.initial_beam
        } else {
            None
        };
        let iter_opts = align::iteration_align_options(&opts, beam);
        if realign {
            let m = ctx.model();
            // Kaldi compiles training graphs without transition probabilities and adds
            // the current model's on every alignment pass; do the same.
            graphs.apply_transition_probs(&m.tm, &ctx.graph);
            failed = 0;
        }

        // A hook (MLLT, fMLLR) changes the feature space mid-iteration: it reads the
        // pre-hook features and the accumulation that follows reads the post-hook
        // ones, exactly as the unchunked loop did (hook, rebuild, accumulate). So an
        // iteration with an active hook walks the chunks twice — once for the
        // realignment and the hook's own statistics, once to accumulate — and an
        // iteration without one can reuse cached features across both passes.
        // An active hook invalidates the cache before the accumulation pass.
        let wants_hook = hooks.begin_iteration(ctx, iteration)?;

        let iter_phase = |what: &'static str| {
            progress::IterPhase::new(
                ctx.progress,
                stage_name,
                iteration,
                plan.num_iterations,
                what,
                plan.utts.len() as u64,
            )
        };

        if realign || wants_hook {
            let mut align_bar = realign.then(|| iter_phase("align"));
            for (index, (c, positions)) in chunks.iter().zip(&chunk_pos).enumerate() {
                let feats = feature_cache.get(index, || feats_kind(ctx, c));

                if let Some(phase) = align_bar.as_mut() {
                    let silence_pdfs = ctx.silence_pdfs();
                    // Positions inside a chunk are consecutive (the chunker keeps
                    // `plan.utts` order), so the chunk is a borrowed window on the
                    // whole-subset graph set.
                    let sub = graphs.slice(positions[0], positions[positions.len() - 1] + 1);
                    let b = phase.bar();
                    let outcome = {
                        let m = ctx.model();
                        align::align_boosted(
                            sub,
                            &m.tm,
                            &m.am,
                            &silence_pdfs,
                            plan.boost_silence,
                            ctx.device,
                            &feats,
                            &iter_opts,
                            batch,
                            &b,
                        )
                    };
                    b.finish();
                    phase.done(c.len() as u64);
                    failed += outcome.failed;
                    for (&p, a) in positions.iter().zip(outcome.alignments) {
                        alignments[p] = a;
                    }
                }

                if wants_hook {
                    let chunk_alis: Vec<Option<Alignment>> =
                        positions.iter().map(|&p| alignments[p].clone()).collect();
                    hooks.accumulate_chunk(ctx, iteration, c, &chunk_alis, &feats)?;
                }
            }
            if wants_hook {
                hooks.end_iteration(ctx, iteration)?;
                // MLLT changes LDA; fMLLR installs speaker transforms. Drop every
                // pre-hook view before deriving any post-hook accumulation data.
                feature_cache.clear();
            }
        }

        let mut accum_bar = iter_phase("accumulate");
        let mut st: Option<Stats> = None;
        for (index, (c, positions)) in chunks.iter().zip(&chunk_pos).enumerate() {
            let feats = feature_cache.get(index, || feats_kind(ctx, c));
            let chunk_alis: Vec<Option<Alignment>> =
                positions.iter().map(|&p| alignments[p].clone()).collect();
            let b = accum_bar.bar();
            let part = {
                let m = ctx.model();
                stats::accumulate(ctx.device, &m.am, &m.tm, &chunk_alis, &feats, &b)
            };
            b.finish();
            accum_bar.done(c.len() as u64);
            match &mut st {
                Some(acc) => acc.merge(part),
                None => st = Some(part),
            }
        }

        let st = match st {
            Some(s) => s,
            None => {
                let m = ctx.model();
                Stats::empty(&m.am, &m.tm)
            }
        };

        let update_opts = UpdateOptions {
            mixup: plan.gaussians.current(),
            power: plan.power,
            min_gaussian_occupancy: plan.min_gaussian_occupancy,
            ..UpdateOptions::default()
        };
        let mut rng = ctx.rng.clone();
        let result = {
            let m = ctx.model_mut();
            stats::update_model(&st, &mut m.tm, &mut m.am, &update_opts, &mut rng)
        };
        ctx.rng = rng;

        plan.gaussians.step(iteration);

        ctx.progress.iteration_summary(&IterationSummary {
            stage: stage_name,
            iteration,
            num_iterations: plan.num_iterations,
            loglike_per_frame: st.loglike_per_frame(),
            gaussians: result.num_gauss,
            failed,
            elapsed: started.elapsed(),
        });
    }

    Ok(alignments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn features(value: f32) -> Vec<Feats> {
        vec![Feats::from_elem((5, 2), value)]
    }

    #[test]
    fn cached_chunk_is_borrowed_until_transform_invalidation() {
        let mut cache = DerivedCache::new(1, 1024);
        let original = cache.get(0, || features(1.0))[0].as_ptr();
        let hit = cache.get(0, || panic!("unchanged features must not be derived"));
        assert!(matches!(hit, Cow::Borrowed(_)));
        assert_eq!(hit[0].as_ptr(), original, "cache hits must not deep-clone");
        cache.clear();
        assert_eq!(cache.retained, 0);
        assert_eq!(cache.get(0, || features(2.0))[0][[0, 0]], 2.0);
    }

    #[test]
    fn full_cache_keeps_prefix_while_tail_chunks_stream() {
        let budget = size_of::<Feats>() + 10 * size_of::<f32>();
        let mut cache = DerivedCache::new(3, budget);
        for _ in 0..3 {
            assert!(matches!(cache.get(0, || features(0.0)), Cow::Borrowed(_)));
            for chunk in 1..3 {
                assert!(matches!(
                    cache.get(chunk, || features(chunk as f32)),
                    Cow::Owned(_)
                ));
            }
            assert_eq!(cache.retained, budget);
            assert!(cache.chunks[0].is_some());
            assert!(cache.chunks[1..].iter().all(Option::is_none));
        }
        cache.clear();
        assert!(!cache.full);
        assert!(matches!(cache.get(0, || features(3.0)), Cow::Borrowed(_)));
    }

    #[test]
    fn oversized_chunks_and_disabled_cache_do_not_retain_data() {
        for budget in [0, 1] {
            let mut cache = DerivedCache::new(1, budget);
            for value in [1.0, 2.0] {
                let fresh = cache.get(0, || features(value));
                assert!(matches!(fresh, Cow::Owned(_)));
                assert_eq!(fresh[0][[0, 0]], value);
            }
            assert_eq!(cache.retained, 0);
            assert!(cache.chunks[0].is_none());
        }
    }

    /// Exercise the actual iteration driver, including a mid-iteration feature
    /// change. Cached and streaming runs must produce identical GMM parameters,
    /// transition probabilities and alignments, while deriving half as often.
    #[test]
    fn iteration_cache_preserves_updates_across_transform_hook() {
        use crate::{
            config::TrainConfig,
            pipeline::{FeatureStore, ModelState, Progress},
        };
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use viter_io::corpus::Corpus;
        use viter_kaldi::{
            audio::Audio,
            device::Device,
            gmm::{AmDiagGmm, DiagGmm},
            hmm::{ContextDependency, GraphOptions, HmmTopology, TransitionModel},
            types::{Pronunciation, Utterance},
        };

        struct ChangeFeatures;
        impl IterationHooks for ChangeFeatures {
            fn begin_iteration(&mut self, _: &mut StageCtx<'_>, iteration: usize) -> Result<bool> {
                Ok(iteration == 2)
            }
            fn end_iteration(&mut self, ctx: &mut StageCtx<'_>, _: usize) -> Result<()> {
                ctx.model_mut().lda = Some(Feats::from_elem((1, 1), 3.0));
                Ok(())
            }
        }
        let corpus = Corpus {
            utts: (0..4)
                .map(|i| Utterance {
                    id: format!("u{i}"),
                    speaker: "speaker".into(),
                    audio: "/dev/null".into(),
                    text: "word".into(),
                    words: vec!["word".into()],
                    prons: vec![vec![Pronunciation::plain(vec![2])]],
                })
                .collect(),
            speakers: vec!["speaker".into()],
            phones: Default::default(),
            silence_phones: vec![1],
            oov_words: Default::default(),
        };
        let cfg = TrainConfig {
            chunk_frames: 1,
            ..Default::default()
        };
        let progress = Progress::hidden();
        let device = Device::cpu();
        let utts: Vec<_> = (0..4).collect();
        let run = |budget| {
            let mut mfcc = viter_kaldi::feat::MfccOptions::default();
            mfcc.dither = 0.0;
            let store = FeatureStore::build_with_audio(
                &corpus,
                &mfcc,
                &Default::default(),
                3,
                3,
                &progress,
                &|_, _| {
                    Ok(Audio {
                        samples: vec![0.0; 4000],
                        sample_rate: 16000,
                    })
                },
            )
            .unwrap();
            let topo = HmmTopology::mfa_default(&[1], &[2], 5, 3);
            let dependency = ContextDependency::monophone_shared(&[vec![1], vec![2]], &|p| {
                topo.num_pdf_classes(p)
            });
            let tm = TransitionModel::new(&dependency, &topo);
            let am = AmDiagGmm::init(
                &DiagGmm::from_single_gaussian(&[0.0, 0.0], &[1.0, 1.0]),
                tm.num_pdfs(),
            );
            let mut ctx = StageCtx {
                corpus: &corpus,
                feats: store,
                device: &device,
                cfg: &cfg,
                progress: &progress,
                rng: Xoshiro256PlusPlus::seed_from_u64(19),
                silence_phones: vec![1],
                stage_subset: 0,
                lexicon_probs: None,
                graph: GraphOptions {
                    silence_phone: 1,
                    silence_prob: 0.0,
                    initial_silence_prob: 0.0,
                    ..Default::default()
                },
                model: Some(ModelState {
                    topo,
                    ctx: dependency,
                    tm,
                    am,
                    lda: None,
                    am_si: None,
                }),
                alignments: Vec::new(),
            };
            let bar = progress.bar("graphs", utts.len() as u64);
            let m = ctx.model();
            let mut graphs = GraphSet::build(
                utts.len(),
                |i| corpus.utts[i].prons.clone(),
                &m.tm,
                &m.ctx,
                &ctx.graph,
                &bar,
            );
            let initial = utts
                .iter()
                .map(|&u| {
                    viter_kaldi::align::equal_align(
                        graphs.graph(u),
                        &m.tm,
                        ctx.feats.num_frames(u),
                        &mut Xoshiro256PlusPlus::seed_from_u64(u as u64),
                    )
                })
                .collect();
            let mut plan = IterationPlan {
                stage: Stage::Mono,
                num_iterations: 3,
                realignment_iterations: vec![],
                gaussians: GaussianSchedule::new(m.am.num_gauss(), m.am.num_gauss(), 3),
                power: 0.25,
                boost_silence: 1.0,
                initial_beam: None,
                min_gaussian_occupancy: 0.0,
                utts: &utts,
            };
            let calls = Cell::new(0);
            let derive = |ctx: &StageCtx<'_>, us: &[usize]| {
                calls.set(calls.get() + 1);
                let offset = ctx.model().lda.as_ref().map_or(0.0, |m| m[[0, 0]]);
                us.iter()
                    .map(|&u| {
                        Feats::from_shape_fn((ctx.feats.num_frames(u), 2), |(r, c)| {
                            offset + (u + c) as f32 * 0.25 + r as f32 * 0.01
                        })
                    })
                    .collect()
            };
            let aligned = run_iterations_impl(
                &mut ctx,
                &mut plan,
                &mut ChangeFeatures,
                &derive,
                &mut graphs,
                initial,
                budget,
            )
            .unwrap();
            let m = ctx.model();
            let mut values = Vec::new();
            for p in 0..m.am.num_pdfs() {
                let g = m.am.pdf(p as u32);
                values.extend(
                    g.weights
                        .iter()
                        .chain(g.gconsts.iter())
                        .chain(g.means_invvars.iter())
                        .chain(g.inv_vars.iter())
                        .map(|x| x.to_bits()),
                );
            }
            values.extend(
                (1..=m.tm.num_transition_ids())
                    .map(|t| m.tm.get_transition_log_prob(t as u32).to_bits()),
            );
            let tids: Vec<_> = aligned
                .into_iter()
                .map(|a| a.expect("valid initial alignment").tids)
                .collect();
            (values, tids, calls.get())
        };
        let streamed = run(0);
        let cached = run(1024 * 1024);
        assert_eq!(
            cached.0, streamed.0,
            "model or transition probabilities changed"
        );
        assert_eq!(cached.1, streamed.1, "alignments changed");
        assert_eq!(
            cached.2 * 2,
            streamed.2,
            "unchanged passes should reuse features"
        );
    }
}
