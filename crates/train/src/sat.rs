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
use viter_kaldi::device::DeviceKind;
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

/// Run one SAT round. `key` is the progress key ("sat", "sat_2", ...); repeated
/// rounds each rebuild the tree from the *previous* round's model
/// (`gmm_init_model_from_previous`) and re-estimate fMLLR from scratch, so the
/// speaker-independent alignment model always comes from the last round.
pub fn run(ctx: &mut StageCtx<'_>, cfg: &SatConfig, key: &str) -> Result<StageOutput> {
    let utts = ctx.subset_for();
    ctx.progress.stage(
        key,
        &format!(
            "{} utterances, {} iterations, {} leaves",
            utts.len(),
            cfg.num_iterations,
            cfg.num_leaves
        ),
    );

    // A new SAT round starts from unadapted features: MFA re-aligns the subset with
    // the previous stage's speaker-independent model before rebuilding the tree, and
    // re-estimates fMLLR from scratch during the round's fmllr_iterations.
    ctx.feats.clear_fmllr();

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
            // `sat.py:254` `_setup_tree(init_from_previous=self.quick,
            // initial_mix_up=self.quick)`: a normal SAT round re-estimates one
            // gaussian per new leaf from its statistics and grows from num_leaves;
            // only MFA's optional `quick` round copies the previous model and mixes up.
            mixup: if cfg.quick {
                cfg.initial_gaussians()
            } else {
                0
            },
            from_previous: cfg.quick,
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
    let mut graphs = {
        let m = ctx.model();
        GraphSet::build(
            utts.len(),
            |i| ctx.words_of(utts[i]),
            &m.tm,
            &m.ctx,
            &ctx.graph,
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
        ctx,
        &mut plan,
        &mut hooks,
        feats,
        &rebuild,
        &mut graphs,
        converted,
    )?;

    // Speaker-independent alignment model from two-feature statistics.
    build_align_model(ctx, cfg, &utts, &alignments, &mut plan)?;

    ctx.progress.stage_done(
        key,
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

    // Per-frame aligned pdf and weight for every utterance, shared by both paths.
    // Silence frames are scaled by `silence_weight` (MFA uses 0.0, so they drop out).
    let per_utt: Vec<Option<(Vec<u32>, Vec<f32>)>> = (0..utts.len())
        .into_par_iter()
        .map(|i| {
            let ali = alignments[i].as_ref()?;
            let frames = ali.tids.len().min(feats[i].nrows());
            let mut pdfs = Vec::with_capacity(frames);
            let mut ws = Vec::with_capacity(frames);
            for t in 0..frames {
                let pdf = tm.transition_id_to_pdf(ali.tids[t]);
                pdfs.push(pdf);
                ws.push(if silence_pdfs.contains(&pdf) {
                    silence_weight
                } else {
                    1.0
                });
            }
            Some((pdfs, ws))
        })
        .collect();

    // On the GPU the whole speaker is one batched accumulation (two compute kernels
    // over the concatenated frames); the device serializes speakers itself, so that
    // loop stays sequential and only the solves run in parallel. On the CPU, keep
    // the rayon-over-chunks path.
    let estimate = |spk: usize, accs: FmllrDiagGmmAccs| -> Option<Mat> {
        bar.inc(1);
        if accs.count() < cfg.fmllr.min_count {
            // Too little data to adapt this speaker reliably.
            return None;
        }
        // kalpy (`transform.cpp:596-607`) starts from the identity on the already
        // adapted features and then composes the new transform with the speaker's
        // previous one (`feat/fmllr.py:226-229`), so adaptation accumulates.
        let (mat, _objf, _count) = accs.update(&cfg.fmllr, None);
        Some(match ctx.feats.fmllr_for(spk) {
            Some(prev) => viter_kaldi::transform::compose_transforms(&mat, prev, true),
            None => mat,
        })
    };

    let transforms: Vec<Option<Mat>> = if ctx.device.kind() == DeviceKind::Gpu {
        // Accumulate every speaker first, then solve the per-speaker transforms in
        // parallel: a solve is ~40 ms of dense linear algebra (40 row updates, each
        // inverting the transform), which on 462 TIMIT speakers dwarfed the kernels.
        let t0 = std::time::Instant::now();
        let accs: Vec<FmllrDiagGmmAccs> = by_speaker
            .iter()
            .map(|entries| {
                let live: Vec<usize> = entries
                    .iter()
                    .copied()
                    .filter(|&i| per_utt[i].is_some())
                    .collect();
                let f: Vec<&Feats> = live.iter().map(|&i| &feats[i]).collect();
                let p: Vec<&[u32]> = live
                    .iter()
                    .map(|&i| per_utt[i].as_ref().expect("filtered").0.as_slice())
                    .collect();
                let w: Vec<&[f32]> = live
                    .iter()
                    .map(|&i| per_utt[i].as_ref().expect("filtered").1.as_slice())
                    .collect();
                let mut accs = FmllrDiagGmmAccs::new(dim);
                ctx.device.fmllr_accumulate_batch(&f, &p, &w, am, &mut accs);
                accs
            })
            .collect();
        let t_acc = t0.elapsed();
        let t0 = std::time::Instant::now();
        let out: Vec<Option<Mat>> = accs
            .into_par_iter()
            .enumerate()
            .map(|(spk, accs)| estimate(spk, accs))
            .collect();
        tracing::debug!(
            accumulate_ms = t_acc.as_millis(),
            solve_ms = t0.elapsed().as_millis(),
            "fmllr phases"
        );
        out
    } else {
        by_speaker
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
                        let Some((pdfs, ws)) = &per_utt[i] else {
                            continue;
                        };
                        let f = &feats[i];
                        for t in 0..pdfs.len() {
                            if ws[t] == 0.0 {
                                continue;
                            }
                            let row = f.row(t);
                            let x = row.as_slice().expect("feature rows are contiguous");
                            let gmm = am.pdf(pdfs[t]);
                            gmm.component_posteriors(x, &mut posteriors);
                            if ws[t] != 1.0 {
                                for pp in posteriors.iter_mut() {
                                    *pp *= ws[t];
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
                estimate(spk, accs)
            })
            .collect()
    };
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
