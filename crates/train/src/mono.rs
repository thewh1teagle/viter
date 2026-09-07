//! Monophone training.
//!
//! MFA: `acoustic_modeling/monophone.py`. Initialization is `gmm_init_mono`
//! (`plans/kalpy/extensions/gmm/gmm.cpp:2213`): a monophone shared-root context
//! dependency, one Gaussian per pdf initialized to the global mean/variance taken
//! from the first ~10 utterances, and a transition model from that tree plus the
//! topology. Then a flat start by equal alignment, an iteration-0 update with
//! `min_gaussian_occupancy = 3.0`, and `num_iterations` regular iterations.

use anyhow::{Result, anyhow};
use viter_kaldi::gmm::{AmDiagGmm, DiagGmm};
use viter_kaldi::hmm::{ContextDependency, HmmTopology, TransitionModel};
use viter_kaldi::types::{Feats, PhoneId};
use rayon::prelude::*;

use crate::config::{GaussianSchedule, MonoConfig, Stage};
use crate::pipeline::{
    FeatureKind, GraphSet, IterationPlan, IterationSummary, ModelState, NoHooks, StageCtx,
    StageOutput, UpdateOptions, align, run_iterations, stats,
};

/// Number of leading utterances whose features seed the global Gaussian.
/// `monophone.py:320-331`: MFA collects feature matrices until it has more than 10.
const INIT_UTTERANCES: usize = 10;

pub fn run(ctx: &mut StageCtx<'_>, cfg: &MonoConfig) -> Result<StageOutput> {
    let utts = ctx.subset_for(Stage::Mono);
    ctx.progress.stage(
        "mono",
        &format!("{} utterances, {} iterations", utts.len(), cfg.num_iterations),
    );

    // Topology and monophone tree. Silence phones get `num_sil_states`, everything
    // else `num_nonsil_states` (`dictionary/mixins.py:669 _write_topo`).
    let all_phones: Vec<PhoneId> = ctx.corpus.phones.phone_ids().collect();
    let silence: Vec<PhoneId> = ctx.silence_phones.clone();
    let nonsilence: Vec<PhoneId> =
        all_phones.iter().copied().filter(|p| !silence.contains(p)).collect();

    let topo = HmmTopology::mfa_default(
        &silence,
        &nonsilence,
        ctx.cfg.num_sil_states,
        ctx.cfg.num_nonsil_states,
    );

    // Monophone shared roots: MFA shares roots across the position variants of a
    // phone (`shared_phones_set_symbols`), which for monophone means one shared root
    // per base phone. Each phone is its own set when no grouping is available.
    let phone_sets = shared_phone_sets(ctx, &all_phones);
    let topo_for_pdfs = topo.clone();
    let num_pdf_classes = move |p: PhoneId| topo_for_pdfs.num_pdf_classes(p);
    let ctx_dep = ContextDependency::monophone_shared(&phone_sets, &num_pdf_classes);
    let tm = TransitionModel::new(&ctx_dep, &topo);

    // Global single Gaussian from the first utterances of the subset, replicated to
    // every pdf (gmm.cpp:2213).
    let feats_bar = ctx.progress.spinner("mono init");
    let seed_utts: Vec<usize> = utts.iter().copied().take(INIT_UTTERANCES + 1).collect();
    let seed_feats: Vec<Feats> = ctx.feats.feats_for_many(&seed_utts, FeatureKind::Deltas);
    let (mean, var) = global_mean_var(&seed_feats)?;
    let proto = DiagGmm::from_single_gaussian(&mean, &var);
    let am = AmDiagGmm::init(&proto, ctx_dep.num_pdfs());
    feats_bar.finish();

    tracing::info!(
        pdfs = am.num_pdfs(),
        gauss = am.num_gauss(),
        transition_ids = tm.num_transition_ids(),
        "initialized monophone model"
    );

    ctx.model = Some(ModelState { topo, ctx: ctx_dep, tm, am, lda: None, am_si: None });

    // Stage features: cmvn'd MFCC + deltas, computed once and reused every iteration.
    let feats = ctx.feats.feats_for_many(&utts, FeatureKind::Deltas);

    // Graphs are built once: the monophone tree never changes during this stage.
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

    // Flat start: equal alignment, then the iteration-0 update
    // (`monophone.py:240-307 mono_align_equal`).
    let bar = ctx.progress.bar("equal align", utts.len() as u64);
    let num_frames: Vec<usize> = utts.iter().map(|&u| ctx.feats.num_frames(u)).collect();
    let outcome = {
        let m = ctx.model();
        align::equal_align_all(&graphs, &m.tm, &num_frames, ctx.cfg.seed, &bar)
    };
    bar.finish();
    if outcome.ok_count() == 0 {
        return Err(anyhow!(
            "equal alignment failed for every utterance; check the lexicon and audio lengths"
        ));
    }

    let started = std::time::Instant::now();
    let bar = ctx.progress.bar("mono iter 0 · accumulate", utts.len() as u64);
    let st = {
        let m = ctx.model();
        stats::accumulate(&m.am, &m.tm, &outcome.alignments, &feats, &bar)
    };
    bar.finish();

    // `gmm_init_mono` gives one gaussian per pdf; MFA's `initial_gaussians` is the
    // mixup target of this first update (`monophone.py:291-296`).
    let mut rng = ctx.rng.clone();
    let result = {
        let m = ctx.model_mut();
        stats::update_model(
            &st,
            &mut m.tm,
            &mut m.am,
            &UpdateOptions {
                mixup: cfg.initial_gaussians,
                power: cfg.power,
                min_gaussian_occupancy: cfg.min_gaussian_occupancy,
                ..UpdateOptions::default()
            },
            &mut rng,
        )
    };
    ctx.rng = rng;

    ctx.progress.iteration_summary(&IterationSummary {
        stage: "mono",
        iteration: 0,
        num_iterations: cfg.num_iterations,
        loglike_per_frame: st.loglike_per_frame(),
        gaussians: result.num_gauss,
        failed: outcome.failed,
        elapsed: started.elapsed(),
    });

    // MFA resets the schedule to whatever the model actually has after init
    // (`monophone.py:351-352`, `base.py:222`).
    let initial = result.num_gauss.max(cfg.initial_gaussians);
    let mut plan = IterationPlan {
        stage: Stage::Mono,
        num_iterations: cfg.num_iterations,
        realignment_iterations: cfg.realignment_iterations(),
        gaussians: GaussianSchedule::new(
            initial,
            cfg.max_gaussians,
            cfg.final_gaussian_iteration(),
        ),
        power: cfg.power,
        boost_silence: cfg.boost_silence,
        initial_beam: Some(cfg.initial_beam),
        min_gaussian_occupancy: UpdateOptions::default().min_gaussian_occupancy,
        utts: &utts,
    };

    let rebuild = |c: &StageCtx<'_>, u: &[usize]| c.feats.feats_for_many(u, FeatureKind::Deltas);
    let alignments = run_iterations(
        ctx,
        &mut plan,
        &mut NoHooks,
        feats,
        &rebuild,
        &graphs,
        outcome.alignments,
    )?;

    ctx.progress.stage_done("mono", "");
    Ok(StageOutput { utts, alignments })
}

/// Phone sets sharing a tree root. Position-dependent variants of the same base phone
/// share (MFA `shared_phones_set_symbols`); otherwise each phone stands alone.
fn shared_phone_sets(ctx: &StageCtx<'_>, phones: &[PhoneId]) -> Vec<Vec<PhoneId>> {
    use std::collections::BTreeMap;
    if !ctx.cfg.position_dependent {
        return phones.iter().map(|&p| vec![p]).collect();
    }
    let mut groups: BTreeMap<String, Vec<PhoneId>> = BTreeMap::new();
    for &p in phones {
        let sym = ctx.corpus.phones.sym(p);
        let base = viter_kaldi::types::untag_phone(sym).to_string();
        groups.entry(base).or_default().push(p);
    }
    groups.into_values().collect()
}

/// Global mean and variance over the seed utterances, the initialization Kaldi's
/// `gmm-init-mono` performs before replicating the Gaussian to every pdf.
fn global_mean_var(feats: &[Feats]) -> Result<(Vec<f32>, Vec<f32>)> {
    let dim = feats
        .iter()
        .find(|f| f.nrows() > 0)
        .map(|f| f.ncols())
        .ok_or_else(|| anyhow!("no features available to initialize the monophone model"))?;

    let (sum, sumsq, count) = feats
        .par_iter()
        .map(|f| {
            let mut sum = vec![0.0f64; dim];
            let mut sumsq = vec![0.0f64; dim];
            let mut count = 0.0f64;
            for row in f.rows() {
                for (d, &x) in row.iter().enumerate() {
                    sum[d] += x as f64;
                    sumsq[d] += (x as f64) * (x as f64);
                }
                count += 1.0;
            }
            (sum, sumsq, count)
        })
        .reduce(
            || (vec![0.0f64; dim], vec![0.0f64; dim], 0.0f64),
            |mut a, b| {
                for d in 0..dim {
                    a.0[d] += b.0[d];
                    a.1[d] += b.1[d];
                }
                a.2 += b.2;
                a
            },
        );

    if count <= 0.0 {
        return Err(anyhow!("seed utterances contained no frames"));
    }

    let mut mean = vec![0.0f32; dim];
    let mut var = vec![0.0f32; dim];
    for d in 0..dim {
        let m = sum[d] / count;
        // Kaldi floors the initial variance at a small positive value so the gconsts
        // are finite even for a constant dimension.
        let v = (sumsq[d] / count - m * m).max(1.0e-4);
        mean[d] = m as f32;
        var[d] = v as f32;
    }
    Ok((mean, var))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn global_mean_var_matches_direct_computation() {
        let f: Feats = array![[0.0f32, 2.0], [2.0, 4.0], [4.0, 6.0]];
        let (mean, var) = global_mean_var(&[f]).unwrap();
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);
        // variance of {0,2,4} about the mean is 8/3
        assert!((var[0] - 8.0 / 3.0).abs() < 1e-5);
        assert!((var[1] - 8.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn global_mean_var_floors_zero_variance() {
        let f: Feats = array![[1.0f32], [1.0], [1.0]];
        let (_, var) = global_mean_var(&[f]).unwrap();
        assert!(var[0] > 0.0);
    }

    #[test]
    fn global_mean_var_rejects_empty() {
        assert!(global_mean_var(&[]).is_err());
    }
}
