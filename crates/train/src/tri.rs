//! Triphone training, and the tree-building shared with the LDA and SAT stages.
//!
//! MFA: `acoustic_modeling/triphone.py`. Accumulate tree statistics from the previous
//! stage's alignments, derive questions automatically, build the decision tree,
//! initialize a model from the per-leaf statistics (`gmm_init_model`,
//! `plans/kalpy/extensions/gmm/gmm.cpp:2307`) or from the previous model
//! (`gmm_init_model_from_previous`, gmm.cpp:2411), convert the old alignments onto
//! the new tree, then iterate exactly as monophone does.

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashMap;
use viter_kaldi::gmm::AmDiagGmm;
use viter_kaldi::hmm::{self, ContextDependency, TransitionModel};
use viter_kaldi::tree::{
    self, AccumulateTreeStatsOptions, BuildTreeStats, EventType, GaussClusterable,
};
use viter_kaldi::types::{Alignment, Feats, PhoneId};

use crate::config::{GaussianSchedule, Stage, TriConfig};
use crate::pipeline::{
    FeatureKind, GraphSet, IterationPlan, NoHooks, StageCtx, StageOutput, UpdateOptions,
    run_iterations,
};

/// Kaldi triphone context: N = 3, P = 1.
pub const CONTEXT_WIDTH: usize = 3;
pub const CENTRAL_POSITION: usize = 1;

pub fn run(ctx: &mut StageCtx<'_>, cfg: &TriConfig) -> Result<StageOutput> {
    let utts = ctx.subset_for();
    ctx.progress.stage(
        "tri",
        &format!(
            "{} utterances, {} iterations, {} leaves",
            utts.len(),
            cfg.num_iterations,
            cfg.num_leaves
        ),
    );

    let feats = ctx.feats.feats_for_many(&utts, FeatureKind::Deltas);

    // Alignments from the previous stage, restricted to this stage's utterances.
    // Align this stage's (larger) subset with the previous stage's model, like MFA.
    let prev_alignments: Vec<Option<Alignment>> =
        crate::pipeline::align_subset_with_previous(ctx, &utts)?;

    let setup = build_tree_stage(
        ctx,
        &utts,
        &prev_alignments,
        &feats,
        &TreeSetup {
            num_leaves: cfg.num_leaves,
            thresh: cfg.tree_thresh,
            cluster_thresh: cfg.cluster_threshold,
            var_floor: cfg.tree_var_floor,
            // `triphone.py:463-468`: a fresh triphone model mixes up to
            // initial_gaussians (= num_leaves) at init.
            mixup: cfg.num_leaves,
            from_previous: false,
        },
    )?;

    let alignments = install_and_train(
        ctx,
        setup,
        prev_alignments,
        &utts,
        feats,
        &mut IterationPlan {
            stage: Stage::Tri,
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
        },
    )?;

    ctx.progress.stage_done("tri", "");
    Ok(StageOutput { utts, alignments })
}

/// Parameters for one stage's tree build.
pub struct TreeSetup {
    pub num_leaves: usize,
    pub thresh: f64,
    pub cluster_thresh: f64,
    pub var_floor: f64,
    /// Mixup target passed to model initialization; 0 means "one gaussian per leaf".
    pub mixup: usize,
    /// True for LDA/SAT, which initialize from the previous model
    /// (`gmm_init_model_from_previous`) rather than from the tree statistics alone.
    pub from_previous: bool,
}

/// A freshly built tree and the model initialized on it.
pub struct TreeStageSetup {
    pub ctx_dep: ContextDependency,
    pub tm: TransitionModel,
    pub am: AmDiagGmm,
}

/// Accumulate tree stats, build the tree, and initialize the acoustic model.
///
/// Shared by tri, lda and sat: `lda.py:381` and `sat.py` both call MFA's
/// `_setup_tree`, which is `triphone.py:338-469`.
pub fn build_tree_stage(
    ctx: &StageCtx<'_>,
    utts: &[usize],
    alignments: &[Option<Alignment>],
    feats: &[Feats],
    setup: &TreeSetup,
) -> Result<TreeStageSetup> {
    let m = ctx.model();

    // Context-independent phones: silence and the OOV phone (`triphone.py`,
    // `AccumulateTreeStats` ci_phones).
    let opts = AccumulateTreeStatsOptions {
        var_floor: setup.var_floor,
        context_width: CONTEXT_WIDTH,
        central_position: CENTRAL_POSITION,
        ci_phones: ctx.silence_phones.clone(),
    };

    let bar = ctx.progress.bar("tree stats", utts.len() as u64);
    let merged: HashMap<EventType, GaussClusterable> = (0..utts.len())
        .into_par_iter()
        .fold(HashMap::new, |mut acc, i| {
            if let Some(ali) = &alignments[i] {
                tree::accumulate_tree_stats(&opts, &m.tm, &ali.tids, &feats[i], &mut acc);
            }
            bar.inc(1);
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            tree::merge_tree_stats(&mut a, b);
            a
        });
    bar.finish();

    if merged.is_empty() {
        return Err(anyhow!(
            "no tree statistics were accumulated; the previous stage produced no alignments"
        ));
    }
    // Kaldi builds tree stats from a std::map keyed by event (tree-accu.cc:39), i.e.
    // sorted; summation order and split tie-breaks then match run to run.
    let stats: BuildTreeStats = tree::stats_map_to_vec(merged);

    // Roots: silence phones share one non-split root, every other phone (all its
    // position variants together) gets a shared, splittable root
    // (`tree::mfa_roots`, MFA's roots file).
    let nonsil_groups = nonsilence_groups(ctx);
    let (phone_sets, share_roots, do_split) = tree::mfa_roots(&nonsil_groups, &ctx.silence_phones);

    // Questions: automatically obtained from the central-state statistics
    // (`triphone.py:403`, Kaldi `AutomaticallyObtainQuestions` with pdf-class [1], P=1).
    let spinner = ctx.progress.spinner("tree questions");
    let phone_questions =
        tree::automatically_obtain_questions(&stats, &phone_sets, &[1], CENTRAL_POSITION);
    // MFA dedups the returned sets (`triphone.py:406-417`).
    let phone_questions = dedup_questions(phone_questions);
    // kalpy derives the pdf-class questions from the topology's largest state
    // count (tree.cpp:1169): with MFA's 5-state silence that is [0],[0,1],[0,1,2],
    // [0,1,2,3], so silence can keep one pdf per state. Capping at 3 (the
    // non-silence count) collapsed silence onto 3 pdfs and starved the model.
    let max_pdf_classes = m
        .topo
        .phones()
        .iter()
        .map(|&p| m.topo.num_pdf_classes(p))
        .max()
        .unwrap_or(3);
    let questions = tree::make_questions_with(&phone_questions, CONTEXT_WIDTH, max_pdf_classes, 0);
    spinner.finish();

    let spinner = ctx.progress.spinner("build tree");
    let phone2num_pdf_classes = m.topo.phone_to_num_pdf_classes();
    let (event_map, num_leaves) = tree::build_tree(
        &questions,
        &phone_sets,
        &phone2num_pdf_classes,
        &share_roots,
        &do_split,
        &stats,
        setup.thresh,
        setup.num_leaves,
        setup.cluster_thresh,
        CENTRAL_POSITION,
        true,
    );
    spinner.finish();

    let ctx_dep = ContextDependency {
        n: CONTEXT_WIDTH,
        p: CENTRAL_POSITION,
        map: event_map,
    };
    let tm = TransitionModel::new(&ctx_dep, &m.topo);
    tracing::info!(
        leaves = num_leaves,
        pdfs = ctx_dep.num_pdfs(),
        "built decision tree"
    );

    // Initialize the acoustic model on the new tree.
    let spinner = ctx.progress.spinner("init model");
    let am = if setup.from_previous {
        init_from_previous(&stats, &ctx_dep, &m.ctx, &m.am, setup)?
    } else {
        init_from_stats(&stats, &ctx_dep, setup)?
    };
    spinner.finish();

    Ok(TreeStageSetup { ctx_dep, tm, am })
}

/// `gmm_init_model` (gmm.cpp:2307): one Gaussian per leaf from the summed statistics
/// of the leaf, then mix up to the target by occupancy.
fn init_from_stats(
    stats: &BuildTreeStats,
    ctx_dep: &ContextDependency,
    setup: &TreeSetup,
) -> Result<AmDiagGmm> {
    let mut per_leaf = tree::split_stats_by_map(stats, &ctx_dep.map);
    per_leaf.resize(ctx_dep.num_pdfs(), Vec::new());
    // Kaldi gmm-init-model.cc:69-74: a leaf with no statistics (a phone unseen in
    // training that shares no root) gets the average of all stats, with a warning.
    let avg = tree::sum_stats(stats).ok_or_else(|| anyhow!("no tree statistics at all"))?;
    let mut am = AmDiagGmm::new();
    let mut occs: Vec<f64> = Vec::with_capacity(per_leaf.len());
    for (leaf, leaf_stats) in per_leaf.iter().enumerate() {
        let summed = match tree::sum_stats(leaf_stats) {
            Some(s) => s,
            None => {
                tracing::debug!(
                    pdf = leaf,
                    "tree leaf has no stats; using the global average"
                );
                avg.clone()
            }
        };
        occs.push(summed.count());
        let mean: Vec<f32> = summed.mean().into_iter().map(|x| x as f32).collect();
        // Kaldi `DiagGmm(GaussClusterable, var_floor)` (diag-gmm.cc:954) floors the
        // variance at var_floor (0.01, kalpy gmm.cpp:2314) before inverting.
        let var: Vec<f32> = summed
            .var()
            .into_iter()
            .map(|x| x.max(setup.var_floor) as f32)
            .collect();
        am.add_pdf(viter_kaldi::gmm::DiagGmm::from_single_gaussian(&mean, &var));
    }
    if setup.mixup > 0 {
        // Kaldi gmm-init-model uses power 0.2 and min_count 20.0 for this initial
        // allocation (`plans/research/01_kaldi_surface.md` section 3).
        let mut rng = <rand_xoshiro::Xoshiro256PlusPlus as rand::SeedableRng>::from_seed([0u8; 32]);
        am.split_by_count(&occs, setup.mixup, 0.01, 0.2, 20.0, &mut rng);
    }
    am.compute_gconsts();
    Ok(am)
}

/// `gmm_init_model_from_previous` (gmm.cpp:2411): map each new leaf onto the old
/// model's distribution for the same context, then mix up/down to the target.
fn init_from_previous(
    stats: &BuildTreeStats,
    ctx_dep: &ContextDependency,
    old_ctx: &ContextDependency,
    old_am: &AmDiagGmm,
    setup: &TreeSetup,
) -> Result<AmDiagGmm> {
    let mut per_leaf = tree::split_stats_by_map(stats, &ctx_dep.map);
    per_leaf.resize(ctx_dep.num_pdfs(), Vec::new());
    let avg = tree::sum_stats(stats).ok_or_else(|| anyhow!("no tree statistics at all"))?;
    let mut am = AmDiagGmm::new();
    let mut occs: Vec<f64> = Vec::with_capacity(per_leaf.len());

    for leaf_stats in &per_leaf {
        let summed = tree::sum_stats(leaf_stats);
        occs.push(summed.as_ref().map(|s| s.count()).unwrap_or(0.0));

        // Find the old pdf that the majority of this leaf's statistics mapped to, and
        // copy its GMM — the "from previous" initialization.
        let mut votes: HashMap<u32, f64> = HashMap::new();
        for (event, c) in leaf_stats.iter() {
            if let Some(old_pdf) = old_ctx.map.map(event) {
                *votes.entry(old_pdf).or_insert(0.0) += c.count();
            }
        }
        let best = votes
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(pdf, _)| pdf);

        match best {
            Some(pdf) => am.add_pdf(old_am.pdf(pdf).clone()),
            // No old pdf covers this leaf: fall back to its own statistics.
            None => {
                let s = summed.or_else(|| {
                    tracing::debug!("tree leaf has neither statistics nor a previous pdf; using the global average");
                    Some(avg.clone())
                }).ok_or_else(|| {
                    anyhow!("unreachable: global average stats missing")
                })?;
                let mean: Vec<f32> = s.mean().into_iter().map(|x| x as f32).collect();
                let var: Vec<f32> = s.var().into_iter().map(|x| x as f32).collect();
                am.add_pdf(viter_kaldi::gmm::DiagGmm::from_single_gaussian(&mean, &var));
            }
        }
    }

    // `triphone.py:441-461`: mix_up = mix_down = initial_gaussians.
    if setup.mixup > 0 {
        let mut rng = <rand_xoshiro::Xoshiro256PlusPlus as rand::SeedableRng>::from_seed([0u8; 32]);
        if am.num_gauss() > setup.mixup {
            am.merge_by_count(&occs, setup.mixup, 0.2, 20.0);
        }
        if am.num_gauss() < setup.mixup {
            am.split_by_count(&occs, setup.mixup, 0.01, 0.2, 20.0, &mut rng);
        }
    }
    am.compute_gconsts();
    Ok(am)
}

/// Install a new tree/model, convert the previous alignments onto it, rebuild the
/// graphs, and run the stage's iterations.
fn install_and_train(
    ctx: &mut StageCtx<'_>,
    setup: TreeStageSetup,
    prev_alignments: Vec<Option<Alignment>>,
    utts: &[usize],
    feats: Vec<Feats>,
    plan: &mut IterationPlan<'_>,
) -> Result<Vec<Option<Alignment>>> {
    // Convert alignments before replacing the model: the conversion needs both.
    let bar = ctx.progress.bar("convert alignments", utts.len() as u64);
    let converted: Vec<Option<Alignment>> = {
        let old = ctx.model();
        prev_alignments
            .par_iter()
            .map(|a| {
                let out = a.as_ref().and_then(|ali| {
                    hmm::convert_alignment(
                        &old.tm,
                        &setup.tm,
                        &setup.ctx_dep,
                        CONTEXT_WIDTH,
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

    let lost = converted.iter().filter(|a| a.is_none()).count()
        - prev_alignments.iter().filter(|a| a.is_none()).count();
    if lost > 0 {
        ctx.progress.warn(format!(
            "{lost} alignments could not be converted to the new tree"
        ));
    }

    {
        let m = ctx.model_mut();
        m.ctx = setup.ctx_dep;
        m.tm = setup.tm;
        m.am = setup.am;
    }

    // Graphs must be rebuilt: the tree changed, so pdf ids and transition ids did too.
    let bar = ctx.progress.bar("graphs", utts.len() as u64);
    let graphs = {
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

    let rebuild = |c: &StageCtx<'_>, u: &[usize]| c.feats.feats_for_many(u, FeatureKind::Deltas);
    run_iterations(ctx, plan, &mut NoHooks, feats, &rebuild, &graphs, converted)
}

/// Groups of phones sharing a tree root: all position variants of one base phone.
pub fn nonsilence_groups(ctx: &StageCtx<'_>) -> Vec<Vec<PhoneId>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<PhoneId>> = BTreeMap::new();
    for p in ctx.corpus.phones.phone_ids() {
        if ctx.silence_phones.contains(&p) {
            continue;
        }
        let base = viter_kaldi::types::untag_phone(ctx.corpus.phones.sym(p)).to_string();
        groups.entry(base).or_default().push(p);
    }
    groups.into_values().collect()
}

/// `triphone.py:406-417`: keep the first occurrence of each distinct question set.
fn dedup_questions(questions: Vec<Vec<PhoneId>>) -> Vec<Vec<PhoneId>> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(questions.len());
    for q in questions {
        if seen.insert(q.clone()) {
            out.push(q);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_are_deduplicated_preserving_order() {
        let qs = vec![vec![1, 2], vec![3], vec![1, 2], vec![2, 1]];
        let out = dedup_questions(qs);
        assert_eq!(out, vec![vec![1, 2], vec![3], vec![2, 1]]);
    }

    #[test]
    fn triphone_context_is_kaldi_standard() {
        assert_eq!(CONTEXT_WIDTH, 3);
        assert_eq!(CENTRAL_POSITION, 1);
    }
}
