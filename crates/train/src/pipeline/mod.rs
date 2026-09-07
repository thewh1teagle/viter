//! Training orchestration: feature store, stage context, the shared GMM iteration
//! loop, and the two public entry points `train` and `align_corpus`.
//!
//! `pipeline` is a directory module (plans/CONTRACTS.md allows this: "a module may become a
//! directory"). `progress` lives inside it rather than as a top-level `progress.rs`
//! because `lib.rs` is owned by the coordinator and declares only the contract's
//! modules.

pub mod align;
pub mod features;
pub mod progress;
pub mod stats;

use anyhow::{Context, Result, anyhow};
use viter_kaldi::align::AlignOptions;
use viter_kaldi::device::Device;
use viter_kaldi::gmm::AmDiagGmm;
use viter_kaldi::hmm::{self, ContextDependency, HmmTopology, TransitionModel};
use viter_kaldi::model::AcousticModel;
use viter_kaldi::transform::Mat;
use viter_kaldi::types::{Alignment, Feats, IntervalAlignment, PdfId, PhoneId, Pronunciation};
use viter_io::corpus::Corpus;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

pub use align::{AlignOutcome, GraphSet};
pub use features::{FeatureKind, FeatureStore};
pub use progress::{IterationSummary, Progress};
pub use stats::{Stats, UpdateOptions};

use crate::config::{GaussianSchedule, Stage, TrainConfig};
use crate::{lda, mono, sat, tri};

/// Result of a training run.
pub struct Trained {
    pub model: AcousticModel,
    /// Final alignments over the full corpus, in `corpus.utts` order.
    pub alignments: Vec<Alignment>,
}

/// The model pieces a stage reads and writes.
pub struct ModelState {
    pub topo: HmmTopology,
    pub ctx: ContextDependency,
    pub tm: TransitionModel,
    pub am: AmDiagGmm,
    /// Composed LDA+MLLT transform once the LDA stage has run.
    pub lda: Option<Mat>,
    /// Speaker-independent alignment model, produced by SAT.
    pub am_si: Option<AmDiagGmm>,
}

/// Everything a stage needs. Stages mutate `model`, `alignments` and the feature
/// store's fMLLR transforms; everything else is read-only.
pub struct StageCtx<'a> {
    pub corpus: &'a Corpus,
    pub feats: FeatureStore,
    pub device: &'a Device,
    pub cfg: &'a TrainConfig,
    pub progress: &'a Progress,
    pub rng: Xoshiro256PlusPlus,
    /// Silence phone ids (from the corpus symbol table).
    pub silence_phones: Vec<PhoneId>,
    /// Current model, `None` before monophone initialization.
    pub model: Option<ModelState>,
    /// Alignments for the utterances of the stage that just finished, indexed by
    /// corpus utterance index. `None` where alignment failed or the utterance was
    /// not in that stage's subset.
    pub alignments: Vec<Option<Alignment>>,
}

impl<'a> StageCtx<'a> {
    pub fn model(&self) -> &ModelState {
        self.model.as_ref().expect("stage ran before monophone initialization")
    }
    pub fn model_mut(&mut self) -> &mut ModelState {
        self.model.as_mut().expect("stage ran before monophone initialization")
    }

    /// Silence pdfs of the current model, for boosting.
    pub fn silence_pdfs(&self) -> Vec<PdfId> {
        self.model().tm.silence_pdfs(&self.silence_phones)
    }

    /// Utterance indices this stage trains on: MFA's subset, shortest-first, or all.
    ///
    /// `corpus/base.py:2735-2745` orders candidates by duration ascending (preferring
    /// manually aligned utterances, which we do not have) and takes the subset from
    /// that pool. We take the `subset` shortest utterances, which is the same
    /// selection without MFA's random tie-break inside the larger pool — shortest
    /// utterances align fastest and most reliably from a flat start.
    pub fn subset_for(&self, stage: Stage) -> Vec<usize> {
        let n = self.corpus.utts.len();
        let want = self.cfg.subset_for(stage, n);
        if want == 0 {
            return (0..n).collect();
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (self.feats.num_frames(i), i));
        order.truncate(want);
        order.sort_unstable();
        order
    }

    /// Candidate pronunciations per word for one utterance, as `hmm::build_graph` wants them.
    pub fn words_of(&self, utt: usize) -> Vec<Vec<Pronunciation>> {
        self.corpus.utts[utt].prons.clone()
    }
}


/// MFA aligns every stage's subset with the *previous* stage's final model before
/// training starts (`trainer.py:592-604`: `self.current_aligner = previous; self.align()`).
/// Subsets grow (2000 -> 5000 -> 10000), so utterances new to this stage have no
/// alignment yet; utterances already aligned are re-aligned too, exactly like MFA.
pub fn align_subset_with_previous(
    ctx: &StageCtx<'_>,
    utts: &[usize],
) -> Result<Vec<Option<Alignment>>> {
    let m = ctx.model();
    let kind = match m.lda.as_ref() {
        Some(lda) => FeatureKind::SpliceLda(lda),
        None => FeatureKind::Deltas,
    };
    let feats = ctx.feats.feats_for_many(utts, kind);
    let bar = ctx.progress.bar("graphs", utts.len() as u64);
    let graphs = GraphSet::build(
        utts.len(),
        |i| ctx.words_of(utts[i]),
        &m.tm,
        &m.ctx,
        &ctx.cfg.graph,
        &bar,
    );
    bar.finish();
    let bar = ctx.progress.bar("align (previous model)", utts.len() as u64);
    // Alignment workflows use MFA's align defaults: boost_silence 1.0 (`alignment/mixins.py:69-91`).
    let outcome = align::align_batch(
        &graphs,
        &m.tm,
        &m.am,
        ctx.device,
        &feats,
        &ctx.cfg.align,
        ctx.cfg.batch_utts,
        &bar,
    );
    bar.finish();
    if outcome.failed > 0 {
        ctx.progress.warn(format!(
            "{} of {} utterances failed to align with the previous model",
            outcome.failed,
            utts.len()
        ));
    }
    Ok(outcome.alignments)
}

/// Output every stage returns.
pub struct StageOutput {
    /// Utterance indices this stage trained on.
    pub utts: Vec<usize>,
    /// Alignments for those utterances, in `utts` order.
    pub alignments: Vec<Option<Alignment>>,
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

/// Stage-specific work that runs inside an iteration, before accumulation.
pub trait IterationHooks {
    /// Called at the start of each iteration; may change the feature view (LDA/MLLT
    /// re-estimation, fMLLR estimation). Returns true when the cached stage features
    /// must be rebuilt.
    fn before_accumulate(
        &mut self,
        _ctx: &mut StageCtx<'_>,
        _iteration: usize,
        _utts: &[usize],
        _alignments: &[Option<Alignment>],
        _feats: &[Feats],
    ) -> Result<bool> {
        Ok(false)
    }
}

/// A hooks implementation that does nothing (mono, tri).
pub struct NoHooks;
impl IterationHooks for NoHooks {}

/// Run a stage's training iterations.
///
/// `feats` is the stage's cached feature view for `plan.utts`; it is rebuilt whenever
/// a hook says the feature space changed.
pub fn run_iterations(
    ctx: &mut StageCtx<'_>,
    plan: &mut IterationPlan<'_>,
    hooks: &mut dyn IterationHooks,
    mut feats: Vec<Feats>,
    rebuild_feats: &dyn Fn(&StageCtx<'_>, &[usize]) -> Vec<Feats>,
    graphs: &GraphSet,
    initial_alignments: Vec<Option<Alignment>>,
) -> Result<Vec<Option<Alignment>>> {
    let stage_name = plan.stage.name();
    let mut alignments = initial_alignments;
    let opts = ctx.cfg.align.clone();
    let batch = ctx.cfg.batch_utts;

    for iteration in 1..=plan.num_iterations {
        let started = Instant::now();
        let mut failed = alignments.iter().filter(|a| a.is_none()).count();

        if plan.realignment_iterations.contains(&iteration) {
            let beam = if iteration == 1 { plan.initial_beam } else { None };
            let iter_opts = align::iteration_align_options(&opts, beam);
            let silence_pdfs = ctx.silence_pdfs();
            let bar = ctx.progress.iter_bar(
                stage_name,
                iteration,
                plan.num_iterations,
                "align",
                plan.utts.len() as u64,
            );
            let m = ctx.model();
            let outcome = align::align_boosted(
                graphs,
                &m.tm,
                &m.am,
                &silence_pdfs,
                plan.boost_silence,
                ctx.device,
                &feats,
                &iter_opts,
                batch,
                &bar,
            );
            bar.finish();
            failed = outcome.failed;
            alignments = outcome.alignments;
        }

        if hooks.before_accumulate(ctx, iteration, plan.utts, &alignments, &feats)? {
            feats = rebuild_feats(ctx, plan.utts);
        }

        let bar = ctx.progress.iter_bar(
            stage_name,
            iteration,
            plan.num_iterations,
            "accumulate",
            plan.utts.len() as u64,
        );
        let st = {
            let m = ctx.model();
            stats::accumulate(&m.am, &m.tm, &alignments, &feats, &bar)
        };
        bar.finish();

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

/// Scatter a stage's alignments back into a corpus-indexed vector.
pub fn scatter(n: usize, utts: &[usize], alignments: Vec<Option<Alignment>>) -> Vec<Option<Alignment>> {
    let mut out = vec![None; n];
    for (slot, ali) in utts.iter().zip(alignments) {
        out[*slot] = ali;
    }
    out
}

/// Train an acoustic model on a corpus.
pub fn train(
    corpus: &Corpus,
    cfg: &TrainConfig,
    device: &Device,
    out_dir: Option<&Path>,
) -> Result<Trained> {
    if corpus.utts.is_empty() {
        return Err(anyhow!("cannot train on an empty corpus"));
    }
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating output directory {}", dir.display()))?;
    }

    let progress = Progress::new();
    let mut plan = vec!["features", "mono"];
    if cfg.stages.tri {
        plan.push("tri");
    }
    if cfg.stages.lda {
        plan.push("lda");
    }
    if cfg.stages.sat {
        plan.push("sat");
    }
    plan.push("final");
    progress.plan(&plan);
    progress.stage(
        "features",
        &format!("{} utterances, {} speakers", corpus.utts.len(), corpus.speakers.len()),
    );

    let feats = FeatureStore::build_with(
        corpus,
        &cfg.mfcc,
        &cfg.deltas,
        cfg.lda.splice_left,
        cfg.lda.splice_right,
        &progress,
    )?;

    let mut ctx = StageCtx {
        corpus,
        feats,
        device,
        cfg,
        progress: &progress,
        rng: Xoshiro256PlusPlus::seed_from_u64(cfg.seed),
        silence_phones: corpus.silence_phones.clone(),
        model: None,
        alignments: vec![None; corpus.utts.len()],
    };

    let out = mono::run(&mut ctx, &cfg.mono)?;
    ctx.alignments = scatter(corpus.utts.len(), &out.utts, out.alignments);
    save_stage(&ctx, out_dir, "mono")?;

    if cfg.stages.tri {
        let out = tri::run(&mut ctx, &cfg.tri)?;
        ctx.alignments = scatter(corpus.utts.len(), &out.utts, out.alignments);
        save_stage(&ctx, out_dir, "tri")?;
    }
    if cfg.stages.lda {
        let out = lda::run(&mut ctx, &cfg.lda)?;
        ctx.alignments = scatter(corpus.utts.len(), &out.utts, out.alignments);
        save_stage(&ctx, out_dir, "lda")?;
    }
    if cfg.stages.sat {
        let out = sat::run(&mut ctx, &cfg.sat)?;
        ctx.alignments = scatter(corpus.utts.len(), &out.utts, out.alignments);
        save_stage(&ctx, out_dir, "sat")?;
    }

    let model = build_model(&ctx)?;

    // Final pass: align the whole corpus with the finished model.
    progress.stage("final", &format!("aligning {} utterances", corpus.utts.len()));
    let final_alignments = final_alignment(&ctx, &model)?;
    let failed = final_alignments.iter().filter(|a| a.is_none()).count();
    if failed > 0 {
        progress.warn(format!("{failed} utterances failed to align in the final pass"));
    }
    let alignments: Vec<Alignment> = final_alignments
        .into_iter()
        .enumerate()
        .filter_map(|(i, a)| {
            a.map(|mut a| {
                a.utt = corpus.utts[i].id.clone();
                a
            })
        })
        .collect();

    progress.stage_done(
        "training",
        &format!("{} utterances aligned", alignments.len()),
    );
    progress.finish();

    Ok(Trained { model, alignments })
}

/// Final full-corpus alignment with the trained model, including the fMLLR two-pass
/// when the model is speaker-adapted.
fn final_alignment(ctx: &StageCtx<'_>, model: &AcousticModel) -> Result<Vec<Option<Alignment>>> {
    let utts: Vec<usize> = (0..ctx.corpus.utts.len()).collect();
    let bar = ctx.progress.bar("graphs", utts.len() as u64);
    let graphs = GraphSet::build(
        utts.len(),
        |i| ctx.words_of(utts[i]),
        &model.tm,
        &model.ctx,
        &ctx.cfg.graph,
        &bar,
    );
    bar.finish();

    let mut store_feats = FeatureView::for_model(ctx, model, &utts);
    let bar = ctx.progress.bar("final align", utts.len() as u64);
    let outcome = align::align_batch(
        &graphs,
        &model.tm,
        model.am_si.as_ref().unwrap_or(&model.am),
        ctx.device,
        &store_feats.feats,
        &ctx.cfg.align,
        ctx.cfg.batch_utts,
        &bar,
    );
    bar.finish();

    // Speaker-independent model present means SAT: estimate fMLLR from the first pass
    // and realign on adapted features.
    if model.am_si.is_some() {
        let transforms = sat::estimate_fmllr(
            ctx,
            &model.tm,
            &model.am,
            &utts,
            &outcome.alignments,
            &store_feats.feats,
            &ctx.cfg.sat,
        )?;
        store_feats.apply_fmllr(ctx, &utts, &transforms);
        let bar = ctx.progress.bar("final align (fmllr)", utts.len() as u64);
        let adapted = align::align_batch(
            &graphs,
            &model.tm,
            &model.am,
            ctx.device,
            &store_feats.feats,
            &ctx.cfg.align,
            ctx.cfg.batch_utts,
            &bar,
        );
        bar.finish();
        return Ok(adapted.alignments);
    }

    Ok(outcome.alignments)
}

/// Features for a set of utterances matching what a model expects.
struct FeatureView {
    feats: Vec<Feats>,
}

impl FeatureView {
    fn for_model(ctx: &StageCtx<'_>, model: &AcousticModel, utts: &[usize]) -> Self {
        let feats = match &model.lda {
            Some(lda) => ctx.feats.feats_for_many(utts, FeatureKind::SpliceLda(lda)),
            None => ctx.feats.feats_for_many(utts, FeatureKind::Deltas),
        };
        Self { feats }
    }

    /// Apply per-speaker fMLLR transforms to the current view.
    fn apply_fmllr(&mut self, ctx: &StageCtx<'_>, utts: &[usize], transforms: &[Option<Mat>]) {
        self.feats
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, f)| {
                let spk = ctx.feats.speaker_of(utts[i]);
                if let Some(x) = &transforms[spk] {
                    *f = viter_kaldi::feat::apply_transform(f, x);
                }
            });
    }
}

/// Assemble the serializable model from the current stage state.
fn build_model(ctx: &StageCtx<'_>) -> Result<AcousticModel> {
    let m = ctx.model();
    let mut meta = std::collections::BTreeMap::new();
    meta.insert("viter_version".into(), env!("CARGO_PKG_VERSION").into());
    meta.insert("num_utterances".into(), ctx.corpus.utts.len().to_string());
    meta.insert("num_speakers".into(), ctx.corpus.speakers.len().to_string());
    meta.insert("num_gaussians".into(), m.am.num_gauss().to_string());
    meta.insert("num_pdfs".into(), m.am.num_pdfs().to_string());

    Ok(AcousticModel {
        version: 1,
        phones: ctx.corpus.phones.clone(),
        silence_phones: ctx.silence_phones.clone(),
        position_dependent: ctx.cfg.position_dependent,
        mfcc: ctx.cfg.mfcc.clone(),
        deltas: if m.lda.is_none() { Some(ctx.cfg.deltas.clone()) } else { None },
        splice: m
            .lda
            .as_ref()
            .map(|_| (ctx.cfg.lda.splice_left, ctx.cfg.lda.splice_right)),
        lda: m.lda.clone(),
        topo: m.topo.clone(),
        ctx: m.ctx.clone(),
        tm: m.tm.clone(),
        am: m.am.clone(),
        am_si: m.am_si.clone(),
        fmllr: m.am_si.as_ref().map(|_| ctx.cfg.sat.fmllr.clone()),
        graph_opts: ctx.cfg.graph.clone(),
        meta,
    })
}

/// Write `<out_dir>/<name>.viter` after a stage completes.
fn save_stage(ctx: &StageCtx<'_>, out_dir: Option<&Path>, name: &str) -> Result<()> {
    let Some(dir) = out_dir else { return Ok(()) };
    let model = build_model(ctx)?;
    let path = dir.join(format!("{name}.viter"));
    model
        .save(&path)
        .with_context(|| format!("writing intermediate model {}", path.display()))?;
    tracing::info!(stage = name, path = %path.display(), "wrote intermediate model");
    Ok(())
}

/// Align a corpus with an already-trained model.
///
/// With a speaker-adapted model (`am_si` present) this is MFA's two pass procedure:
/// align with the speaker-independent model, estimate per-speaker fMLLR from those
/// alignments, then realign on the adapted features with the SAT model.
pub fn align_corpus(
    corpus: &Corpus,
    model: &AcousticModel,
    device: &Device,
    opts: Option<&AlignOptions>,
) -> Result<Vec<Option<IntervalAlignment>>> {
    if corpus.utts.is_empty() {
        return Ok(Vec::new());
    }
    let progress = Progress::new();
    let align_opts = opts.cloned().unwrap_or_default();

    progress.stage("align", &format!("{} utterances", corpus.utts.len()));

    let cfg = TrainConfig {
        mfcc: model.mfcc.clone(),
        deltas: model.deltas.clone().unwrap_or_default(),
        align: align_opts.clone(),
        graph: model.graph_opts.clone(),
        ..TrainConfig::default()
    };
    let (sl, sr) = model.splice.unwrap_or((cfg.lda.splice_left, cfg.lda.splice_right));

    let feats =
        FeatureStore::build_with(corpus, &model.mfcc, &cfg.deltas, sl, sr, &progress)?;
    let frame_shift_s = feats.frame_shift_s();

    let ctx = StageCtx {
        corpus,
        feats,
        device,
        cfg: &cfg,
        progress: &progress,
        rng: Xoshiro256PlusPlus::seed_from_u64(cfg.seed),
        silence_phones: model.silence_phones.clone(),
        model: None,
        alignments: Vec::new(),
    };

    let utts: Vec<usize> = (0..corpus.utts.len()).collect();
    let _tg = Instant::now();
    let bar = progress.bar("graphs", utts.len() as u64);
    let graphs = GraphSet::build(
        utts.len(),
        |i| corpus.utts[utts[i]].prons.clone(),
        &model.tm,
        &model.ctx,
        &cfg.graph,
        &bar,
    );
    bar.finish();

    tracing::debug!(graphs_ms = _tg.elapsed().as_millis(), "graphs built");
    let _tf = Instant::now();
    let mut view = FeatureView::for_model(&ctx, model, &utts);
    tracing::debug!(feats_ms = _tf.elapsed().as_millis(), "stage features derived");

    // Pass 1: speaker-independent model if we have one, else the single model.
    let first_model = model.am_si.as_ref().unwrap_or(&model.am);
    let bar = progress.bar("align", utts.len() as u64);
    let mut outcome = align::align_batch(
        &graphs,
        &model.tm,
        first_model,
        device,
        &view.feats,
        &align_opts,
        cfg.batch_utts,
        &bar,
    );
    bar.finish();

    // Pass 2: fMLLR-adapted realignment.
    if model.am_si.is_some() {
        let fmllr_cfg = crate::config::SatConfig {
            fmllr: model.fmllr.clone().unwrap_or_else(|| cfg.sat.fmllr.clone()),
            ..cfg.sat.clone()
        };
        let transforms = sat::estimate_fmllr(
            &ctx,
            &model.tm,
            &model.am,
            &utts,
            &outcome.alignments,
            &view.feats,
            &fmllr_cfg,
        )?;
        view.apply_fmllr(&ctx, &utts, &transforms);
        let bar = progress.bar("align (fmllr)", utts.len() as u64);
        outcome = align::align_batch(
            &graphs,
            &model.tm,
            &model.am,
            device,
            &view.feats,
            &align_opts,
            cfg.batch_utts,
            &bar,
        );
        bar.finish();
    }

    if outcome.failed > 0 {
        progress.warn(format!(
            "{} of {} utterances failed to align",
            outcome.failed,
            utts.len()
        ));
    }

    let intervals: Vec<Option<IntervalAlignment>> = outcome
        .alignments
        .par_iter()
        .enumerate()
        .map(|(i, a)| {
            a.as_ref().map(|ali| {
                let mut iv =
                    hmm::to_intervals(&model.tm, ali, &corpus.utts[i].prons, frame_shift_s);
                iv.utt = corpus.utts[i].id.clone();
                iv
            })
        })
        .collect();

    progress.stage_done(
        "align",
        &format!("{} aligned, {} failed", utts.len() - outcome.failed, outcome.failed),
    );
    progress.finish();
    Ok(intervals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_places_alignments_at_corpus_indices() {
        let utts = vec![1usize, 3];
        let alis = vec![
            Some(Alignment { utt: "a".into(), tids: vec![1], words: vec![], prons: vec![], loglike: 0.0 }),
            None,
        ];
        let out = scatter(5, &utts, alis);
        assert_eq!(out.len(), 5);
        assert!(out[0].is_none());
        assert_eq!(out[1].as_ref().unwrap().utt, "a");
        assert!(out[3].is_none());
    }
}
