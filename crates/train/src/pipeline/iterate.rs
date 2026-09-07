//! The per-iteration GMM training loop shared by mono, tri, lda and sat.
//!
//! MFA's `train_iteration` (`base.py:367-381`): realign when the iteration is in the
//! realignment list, run any stage-specific work (MLLT, fMLLR), accumulate, update,
//! then increment the gaussian target.
//!
//! Every pass over the stage's utterances walks speaker-aligned chunks (see
//! [`super::chunk`]) and derives each chunk's feature view only for as long as that
//! chunk is being processed, so peak memory is the base feature store plus one
//! chunk's derived features rather than the whole subset's.

use anyhow::Result;
use std::time::Instant;
use viter_kaldi::types::{Alignment, Feats};

use super::{
    GraphSet, IterationSummary, StageCtx, Stats, UpdateOptions, align, chunk, progress, stats,
};
use crate::config::{GaussianSchedule, Stage};

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
/// accumulating before dropping them, so only one chunk of derived features is live
/// at once. A hook therefore sees the iteration in three parts: `begin_iteration`
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
        // iteration without one walks them once, deriving each chunk's features
        // afresh both times.
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
            for (c, positions) in chunks.iter().zip(&chunk_pos) {
                let feats = feats_kind(ctx, c);

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
            }
        }

        let mut accum_bar = iter_phase("accumulate");
        let mut st: Option<Stats> = None;
        for (c, positions) in chunks.iter().zip(&chunk_pos) {
            let feats = feats_kind(ctx, c);
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
