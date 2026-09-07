//! Speaker adapted training (SAT) with fMLLR.
//!
//! MFA: `acoustic_modeling/sat.py`. Rebuild the tree from the previous stage's
//! alignments, then train like triphone but estimate a per-speaker fMLLR transform at
//! `fmllr_iterations` (`sat.py:285-303`), which changes the feature space for every
//! subsequent iteration. When training finishes, build the speaker-independent
//! alignment model (`final.alimdl`) with two-feature statistics: posteriors from the
//! adapted features, statistics from the unadapted ones
//! (`sat.py:313-376 create_align_model`, kalpy `gmm/train.py:161`).

use anyhow::Result;
use rayon::prelude::*;
use viter_kaldi::gmm::AmDiagGmm;
use viter_kaldi::hmm::TransitionModel;
use viter_kaldi::transform::{FmllrDiagGmmAccs, Mat};
use viter_kaldi::types::{Alignment, Feats};

use crate::config::{GaussianSchedule, SatConfig, Stage};
use crate::pipeline::{
    FeatureKind, GraphSet, IterationHooks, IterationPlan, StageCtx, StageOutput, UpdateOptions,
    run_iterations, stats,
};
use crate::tri::{self, TreeSetup};

pub fn run(ctx: &mut StageCtx<'_>, cfg: &SatConfig) -> Result<StageOutput> {
    let utts = ctx.subset_for(Stage::Sat);
    ctx.progress.stage(
        "sat",
        &format!(
            "{} utterances, {} iterations, {} leaves",
            utts.len(),
            cfg.num_iterations,
            cfg.num_leaves
        ),
    );

    // Align this stage's (larger) subset with the previous stage's model, like MFA.
    let prev_alignments: Vec<Option<Alignment>> =
        crate::pipeline::align_subset_with_previous(ctx, &utts)?;

    // SAT runs on the LDA feature space when the LDA stage ran, otherwise on deltas.
    let kind = current_kind(ctx);
    let feats = ctx.feats.feats_for_many(&utts, kind);

    // Rebuild the tree. `sat.py` inherits `_setup_tree`; when a previous model exists
    // it initializes from it (`gmm_init_model_from_previous`) with
    // mix_up = mix_down = initial_gaussians (`triphone.py:441-461`).
    let setup = tri::build_tree_stage(
        ctx,
        &utts,
        &prev_alignments,
        &feats,
        &TreeSetup {
            num_leaves: cfg.num_leaves,
            thresh: cfg.tree_thresh,
            cluster_thresh: cfg.cluster_threshold,
            var_floor: cfg.tree_var_floor,
            mixup: cfg.initial_gaussians(),
            from_previous: true,
        },
    )?;

    let mut plan = IterationPlan {
        stage: Stage::Sat,
        num_iterations: cfg.num_iterations,
        realignment_iterations: cfg.realignment_iterations(),
        gaussians: GaussianSchedule::new(
            cfg.initial_gaussians(),
            cfg.max_gaussians,
            cfg.final_gaussian_iteration(),
        ),
        power: cfg.power,
        boost_silence: cfg.boost_silence,
        initial_beam: None,
        min_gaussian_occupancy: UpdateOptions::default().min_gaussian_occupancy,
        utts: &utts,
    };

    // Convert alignments and install the new model (shared with triphone), but run
    // the iterations here so the fMLLR hook is in play.
    let converted = install(ctx, setup, prev_alignments, &utts)?;

    let bar = ctx.progress.bar("graphs", utts.len() as u64);
    let graphs = {
        let m = ctx.model();
        GraphSet::build(
            utts.len(),
            |i| ctx.words_of(utts[i]),
            &m.tm,
            &m.ctx,
            &ctx.cfg.graph,
            &bar,
        )
    };
    bar.finish();

    let mut hooks = FmllrHooks { cfg: cfg.clone() };
    let rebuild = |c: &StageCtx<'_>, u: &[usize]| {
        let kind = current_kind(c);
        c.feats.feats_for_many(u, kind)
    };
    let feats = ctx.feats.feats_for_many(&utts, current_kind(ctx));

    let alignments = run_iterations(
        ctx, &mut plan, &mut hooks, feats, &rebuild, &graphs, converted,
    )?;

    // Speaker-independent alignment model from two-feature statistics.
    build_align_model(ctx, cfg, &utts, &alignments, &mut plan)?;

    ctx.progress.stage_done(
        "sat",
        &format!(
            "{} gaussians, {} speakers adapted",
            ctx.model().am.num_gauss(),
            (0..ctx.feats.num_speakers())
                .filter(|&s| ctx.feats.fmllr_for(s).is_some())
                .count()
        ),
    );
    Ok(StageOutput { utts, alignments })
}

/// The feature view SAT uses: LDA(+fMLLR) when an LDA transform exists, else deltas.
fn current_kind<'a>(ctx: &'a StageCtx<'_>) -> FeatureKind<'a> {
    match ctx.model().lda.as_ref() {
        Some(lda) => FeatureKind::SpliceLdaFmllr(lda),
        None => FeatureKind::Deltas,
    }
}

/// Convert the previous alignments onto the new tree and install the new model.
fn install(
    ctx: &mut StageCtx<'_>,
    setup: tri::TreeStageSetup,
    prev_alignments: Vec<Option<Alignment>>,
    utts: &[usize],
) -> Result<Vec<Option<Alignment>>> {
    let bar = ctx.progress.bar("convert alignments", utts.len() as u64);
    let converted: Vec<Option<Alignment>> = {
        let old = ctx.model();
        prev_alignments
            .par_iter()
            .map(|a| {
                let out = a.as_ref().and_then(|ali| {
                    viter_kaldi::hmm::convert_alignment(
                        &old.tm,
                        &setup.tm,
                        &setup.ctx_dep,
                        tri::CONTEXT_WIDTH,
                        &ali.tids,
                    )
                    .map(|tids| Alignment {
                        utt: ali.utt.clone(),
                        tids,
                        words: ali.words.clone(),
                        prons: ali.prons.clone(),
                        loglike: ali.loglike,
                    })
                });
                bar.inc(1);
                out
            })
            .collect()
    };
    bar.finish();

    let m = ctx.model_mut();
    m.ctx = setup.ctx_dep;
    m.tm = setup.tm;
    m.am = setup.am;
    Ok(converted)
}

/// Estimate one fMLLR transform per speaker.
///
/// Silence posteriors are scaled by `silence_weight` (MFA `features.py:635` = 0.0, so
/// silence frames contribute nothing) before accumulating
/// (`FmllrDiagGmmAccs::AccumulateFromPosteriors`). A speaker whose statistics fall
/// below `fmllr.min_count` keeps no transform, matching Kaldi's behaviour of leaving
/// such speakers unadapted.
///
/// The transition and acoustic models are passed explicitly rather than read from
/// `ctx.model()`, so this is callable from `align_corpus`, where the context carries
/// no in-training model.
pub fn estimate_fmllr(
    ctx: &StageCtx<'_>,
    tm: &TransitionModel,
    am: &AmDiagGmm,
    utts: &[usize],
    alignments: &[Option<Alignment>],
    feats: &[Feats],
    cfg: &SatConfig,
) -> Result<Vec<Option<Mat>>> {
    let dim = am.dim();
    let num_speakers = ctx.feats.num_speakers();
    let silence_pdfs: std::collections::HashSet<u32> =
        tm.silence_pdfs(&ctx.silence_phones).into_iter().collect();

    // Group this list's entries by speaker.
    let mut by_speaker: Vec<Vec<usize>> = vec![Vec::new(); num_speakers];
    for (i, &u) in utts.iter().enumerate() {
        by_speaker[ctx.feats.speaker_of(u)].push(i);
    }

    let bar = ctx.progress.bar("fmllr", num_speakers as u64);
    let silence_weight = cfg.silence_weight;
    let transforms: Vec<Option<Mat>> = by_speaker
        .par_iter()
        .enumerate()
        .map(|(spk, entries)| {
            // A speaker with thousands of utterances (single-speaker TTS corpora)
            // must not serialize on one core: accumulate per utterance chunk in
            // parallel and reduce, the sums are exact up to f64 rounding order.
            let accumulate = |chunk: &[usize]| -> FmllrDiagGmmAccs {
                let mut accs = FmllrDiagGmmAccs::new(dim);
                let mut posteriors: Vec<f32> = Vec::new();
                for &i in chunk {
                    let Some(ali) = &alignments[i] else { continue };
                    let f = &feats[i];
                    let frames = ali.tids.len().min(f.nrows());
                    for t in 0..frames {
                        let pdf = tm.transition_id_to_pdf(ali.tids[t]);
                        let row = f.row(t);
                        let x = row.as_slice().expect("feature rows are contiguous");
                        let gmm = am.pdf(pdf);
                        gmm.component_posteriors(x, &mut posteriors);
                        if silence_pdfs.contains(&pdf) {
                            if silence_weight == 0.0 {
                                continue;
                            }
                            for p in posteriors.iter_mut() {
                                *p *= silence_weight;
                            }
                        }
                        accs.accumulate_from_posteriors(gmm, x, &posteriors);
                    }
                }
                accs
            };
            let chunk = (entries.len() / (rayon::current_num_threads() * 4)).max(8);
            let accs = entries.par_chunks(chunk).map(accumulate).reduce(
                || FmllrDiagGmmAccs::new(dim),
                |mut a, b| {
                    a.add(&b);
                    a
                },
            );
            bar.inc(1);
            if accs.count() < cfg.fmllr.min_count {
                // Too little data to adapt this speaker reliably.
                return None;
            }
            // Start from the speaker's existing transform so estimation is incremental
            // across fMLLR iterations, as Kaldi's gmm-est-fmllr does.
            let prior = ctx.feats.fmllr_for(spk);
            let (mat, _objf, _count) = accs.update(&cfg.fmllr, prior);
            Some(mat)
        })
        .collect();
    bar.finish();

    let adapted = transforms.iter().filter(|t| t.is_some()).count();
    tracing::info!(
        speakers = num_speakers,
        adapted,
        "estimated fMLLR transforms"
    );
    Ok(transforms)
}

/// fMLLR estimation as an iteration hook.
struct FmllrHooks {
    cfg: SatConfig,
}

impl IterationHooks for FmllrHooks {
    fn before_accumulate(
        &mut self,
        ctx: &mut StageCtx<'_>,
        iteration: usize,
        utts: &[usize],
        alignments: &[Option<Alignment>],
        feats: &[Feats],
    ) -> Result<bool> {
        if !self.cfg.fmllr_iterations().contains(&iteration) {
            return Ok(false);
        }
        let transforms = {
            let m = ctx.model();
            estimate_fmllr(ctx, &m.tm, &m.am, utts, alignments, feats, &self.cfg)?
        };
        for (spk, t) in transforms.into_iter().enumerate() {
            if let Some(mat) = t {
                ctx.feats.set_fmllr(spk, mat);
            }
        }
        // Adapted feature space: the cached view must be rebuilt.
        Ok(true)
    }
}

/// Build the speaker-independent alignment model (`final.alimdl`).
///
/// `gmm-acc-stats-twofeats`: posteriors come from the adapted features that the SAT
/// model was trained on, statistics from the unadapted features that alignment will
/// see on a fresh speaker. Updated with `remove_low_count_gaussians = False`
/// (`sat.py:358-363`).
fn build_align_model(
    ctx: &mut StageCtx<'_>,
    cfg: &SatConfig,
    utts: &[usize],
    alignments: &[Option<Alignment>],
    plan: &mut IterationPlan<'_>,
) -> Result<()> {
    if !ctx.feats.has_any_fmllr() {
        // Nothing was adapted, so the SAT model is already speaker independent.
        return Ok(());
    }
    ctx.progress
        .stage("sat", "building speaker-independent alignment model");

    let adapted = ctx.feats.feats_for_many(utts, current_kind(ctx));
    let unadapted = match ctx.model().lda.as_ref() {
        Some(lda) => ctx.feats.feats_for_many(utts, FeatureKind::SpliceLda(lda)),
        None => ctx.feats.feats_for_many(utts, FeatureKind::Deltas),
    };

    let bar = ctx.progress.bar("two-feats stats", utts.len() as u64);
    let st = {
        let m = ctx.model();
        stats::accumulate_with(&m.am, &m.tm, alignments, &adapted, &unadapted, &bar)
    };
    bar.finish();

    // Update a copy of the model: the SAT model itself must not change.
    let mut si_am = ctx.model().am.clone();
    let mut si_tm = ctx.model().tm.clone();
    let mut rng = ctx.rng.clone();
    let result = stats::update_model(
        &st,
        &mut si_tm,
        &mut si_am,
        &UpdateOptions {
            mixup: plan.gaussians.current(),
            power: cfg.power,
            remove_low_count_gaussians: false,
            ..UpdateOptions::default()
        },
        &mut rng,
    );
    ctx.rng = rng;

    tracing::info!(
        gaussians = result.num_gauss,
        loglike_per_frame = st.loglike_per_frame(),
        "built speaker-independent alignment model"
    );
    ctx.model_mut().am_si = Some(si_am);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmllr_schedule_defaults() {
        let cfg = SatConfig::default();
        assert_eq!(cfg.fmllr_iterations(), vec![2, 4, 6, 12]);
        assert_eq!(cfg.silence_weight, 0.0);
        assert_eq!(cfg.fmllr.min_count, 500.0);
        assert_eq!(cfg.fmllr.num_iters, 40);
    }
}
