//! LDA + MLLT training.
//!
//! MFA: `acoustic_modeling/lda.py`. Accumulate LDA statistics from the triphone
//! alignments on spliced features, estimate a `lda_dimension`-row transform, rebuild
//! the tree on the transformed features, then train like triphone with MLLT
//! re-estimation at `mllt_iterations` (`lda.py:463-481`): each MLLT update transforms
//! the model means (`gmm-transform-means`) and composes the new matrix onto the
//! running LDA transform (`compose-transforms`, `lda.py:455-461`).

use anyhow::{Result, anyhow};
use rand::SeedableRng;
use rayon::prelude::*;
use viter_kaldi::transform::{self, LdaEstimate, LdaEstimateOptions, Mat, MlltAccs};
use viter_kaldi::types::{Alignment, Feats};

use crate::config::{GaussianSchedule, LdaConfig, Stage};
use crate::pipeline::{
    FeatureKind, GraphSet, IterationHooks, IterationPlan, StageCtx, StageOutput, UpdateOptions,
    run_iterations,
};
use crate::tri::{self, TreeSetup};

pub fn run(ctx: &mut StageCtx<'_>, cfg: &LdaConfig) -> Result<StageOutput> {
    let utts = ctx.subset_for();
    ctx.progress.stage(
        "lda",
        &format!(
            "{} utterances, {} iterations, {} leaves, dim {}",
            utts.len(),
            cfg.num_iterations,
            cfg.num_leaves,
            cfg.lda_dimension
        ),
    );

    // Align this stage's (larger) subset with the previous stage's model, like MFA.
    let prev_alignments: Vec<Option<Alignment>> =
        crate::pipeline::align_subset_with_previous(ctx, &utts)?;

    // 1. LDA statistics on spliced features, classes = pdfs of the current model
    // (`lda.py:20-33 LdaStatsAccumulator`, Kaldi acc-lda).
    let lda_mat = estimate_lda(ctx, cfg, &utts, &prev_alignments)?;
    tracing::info!(
        rows = lda_mat.nrows(),
        cols = lda_mat.ncols(),
        "estimated LDA transform"
    );

    // 2. Rebuild the tree on LDA features. `lda.py:380-386` calls `_setup_tree` with
    // `initial_mix_up=False`, so the model starts at one gaussian per leaf.
    let feats = ctx
        .feats
        .feats_for_many(&utts, FeatureKind::SpliceLda(&lda_mat));
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
            mixup: 0,
            from_previous: false,
        },
    )?;

    // Record the transform on the model so feature derivation switches to the LDA path.
    ctx.model_mut().lda = Some(lda_mat);

    // 3. Convert alignments, rebuild graphs, iterate with MLLT.
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

    {
        let m = ctx.model_mut();
        m.ctx = setup.ctx_dep;
        m.tm = setup.tm;
        m.am = setup.am;
    }

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

    let mut plan = IterationPlan {
        stage: Stage::Lda,
        num_iterations: cfg.num_iterations,
        realignment_iterations: cfg.realignment_iterations(),
        gaussians: GaussianSchedule::new(
            cfg.num_leaves,
            cfg.max_gaussians,
            cfg.final_gaussian_iteration(),
        ),
        power: cfg.power,
        boost_silence: cfg.boost_silence,
        initial_beam: None,
        min_gaussian_occupancy: UpdateOptions::default().min_gaussian_occupancy,
        utts: &utts,
    };

    let mut hooks = MlltHooks { cfg: cfg.clone() };
    let rebuild = |c: &StageCtx<'_>, u: &[usize]| {
        let lda = c
            .model()
            .lda
            .clone()
            .expect("lda stage always has a transform");
        c.feats.feats_for_many(u, FeatureKind::SpliceLda(&lda))
    };
    // Features are recomputed from the current transform each iteration only when the
    // hook signals an MLLT update, so the initial view is the one built above.
    let feats = ctx.feats.feats_for_many(
        &utts,
        FeatureKind::SpliceLda(ctx.model().lda.as_ref().unwrap()),
    );

    let alignments = run_iterations(
        ctx,
        &mut plan,
        &mut hooks,
        feats,
        &rebuild,
        &mut graphs,
        converted,
    )?;

    ctx.progress.stage_done("lda", "");
    Ok(StageOutput { utts, alignments })
}

/// Accumulate LDA statistics and estimate the transform.
///
/// Classes are the pdf ids of the current (triphone) model; each frame contributes its
/// aligned pdf. Kaldi's `acc-lda` weights every frame 1.0 for a Viterbi alignment.
fn estimate_lda(
    ctx: &StageCtx<'_>,
    cfg: &LdaConfig,
    utts: &[usize],
    alignments: &[Option<Alignment>],
) -> Result<Mat> {
    let m = ctx.model();
    let num_classes = m.tm.num_pdfs();
    let dim = ctx.feats.spliced_dim();

    // kalpy `LdaStatsAccumulator` (feat/lda.py:44-60) passes the silence phones with
    // `silence_weight = 0.0`: silence frames contribute nothing to the scatter.
    let silence_pdfs: std::collections::HashSet<u32> = ctx.silence_pdfs().into_iter().collect();
    let bar = ctx.progress.bar("lda stats", utts.len() as u64);
    let acc = (0..utts.len())
        .into_par_iter()
        .fold(
            || LdaEstimate::new(num_classes, dim),
            |mut acc, i| {
                if let Some(ali) = &alignments[i] {
                    let spliced = ctx.feats.spliced(utts[i]);
                    let frames = ali.tids.len().min(spliced.nrows());
                    for t in 0..frames {
                        let pdf = m.tm.transition_id_to_pdf(ali.tids[t]);
                        if silence_pdfs.contains(&pdf) {
                            continue;
                        }
                        let row = spliced.row(t);
                        let x = row.as_slice().expect("feature rows are contiguous");
                        acc.accumulate(x, pdf as usize, 1.0);
                    }
                }
                bar.inc(1);
                acc
            },
        )
        .reduce(
            || LdaEstimate::new(num_classes, dim),
            |mut a, b| {
                a.add(&b);
                a
            },
        );
    bar.finish();

    let opts = LdaEstimateOptions {
        dim: cfg.lda_dimension,
        ..LdaEstimateOptions::default()
    };
    let (lda_mat, _full) = acc.estimate(&opts);
    if lda_mat.nrows() != cfg.lda_dimension {
        return Err(anyhow!(
            "LDA estimation returned {} rows, expected {}",
            lda_mat.nrows(),
            cfg.lda_dimension
        ));
    }
    Ok(lda_mat)
}

/// MLLT re-estimation at the configured iterations.
struct MlltHooks {
    cfg: LdaConfig,
}

impl IterationHooks for MlltHooks {
    fn before_accumulate(
        &mut self,
        ctx: &mut StageCtx<'_>,
        iteration: usize,
        utts: &[usize],
        alignments: &[Option<Alignment>],
        feats: &[Feats],
    ) -> Result<bool> {
        if !self.cfg.mllt_iterations.contains(&iteration) {
            return Ok(false);
        }

        let dim = ctx.model().am.dim();
        let bar = ctx.progress.iter_bar(
            "lda",
            iteration,
            self.cfg.num_iterations,
            "mllt",
            utts.len() as u64,
        );

        let random_prune = self.cfg.random_prune;
        let seed = ctx.cfg.seed;
        let mllt_silence: std::collections::HashSet<u32> = ctx.silence_pdfs().into_iter().collect();
        let mllt_silence = &mllt_silence;
        let accs = {
            let m = ctx.model();
            (0..utts.len())
                .into_par_iter()
                .fold(
                    || MlltAccs::new(dim, random_prune),
                    |mut acc, i| {
                        if let Some(ali) = &alignments[i] {
                            // Deterministic per-utterance stream: MLLT prunes randomly.
                            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
                                seed ^ ((iteration as u64) << 32)
                                    ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            );
                            let f = &feats[i];
                            let frames = ali.tids.len().min(f.nrows());
                            let mut posteriors: Vec<f32> = Vec::new();
                            for t in 0..frames {
                                let pdf = m.tm.transition_id_to_pdf(ali.tids[t]);
                                // kalpy MlltStatsAccumulator: silence_weight 0.0
                                // (feat/lda.py:120-140, mllt.cc:162-169 scales the
                                // posteriors by the weight).
                                if mllt_silence.contains(&pdf) {
                                    continue;
                                }
                                let row = f.row(t);
                                let x = row.as_slice().expect("rows are contiguous");
                                let gmm = m.am.pdf(pdf);
                                gmm.component_posteriors(x, &mut posteriors);
                                acc.accumulate_from_posteriors(gmm, x, &posteriors, &mut rng);
                            }
                        }
                        bar.inc(1);
                        acc
                    },
                )
                .reduce(
                    || MlltAccs::new(dim, random_prune),
                    |mut a, b| {
                        a.add(&b);
                        a
                    },
                )
        };
        bar.finish();

        // Update: new [dim, dim] matrix, applied to the model means and composed onto
        // the running LDA transform (`lda.py:434-461`).
        let mut mat: Mat = identity(dim);
        let (objf, count) = accs.update(&mut mat);
        tracing::info!(
            iteration,
            objf_per_frame = if count > 0.0 { objf / count } else { 0.0 },
            "MLLT update"
        );

        let m = ctx.model_mut();
        transform::transform_means(&mut m.am, &mat);
        let prev = m.lda.take().expect("lda stage always has a transform");
        m.lda = Some(transform::compose_transforms(&mat, &prev, false));

        // Feature space changed: the caller must rebuild its cached view.
        Ok(true)
    }
}

fn identity(dim: usize) -> Mat {
    let mut m: Mat = ndarray::Array2::zeros((dim, dim));
    for i in 0..dim {
        m[[i, i]] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matrix_is_square_and_unit() {
        let m = identity(4);
        assert_eq!(m.shape(), &[4, 4]);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[[i, j]], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn mllt_only_fires_on_scheduled_iterations() {
        let cfg = LdaConfig::default();
        assert!(cfg.mllt_iterations.contains(&2));
        assert!(cfg.mllt_iterations.contains(&12));
        assert!(!cfg.mllt_iterations.contains(&3));
    }
}
