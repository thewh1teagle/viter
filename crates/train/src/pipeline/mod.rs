//! Training orchestration: feature store, stage context, the shared GMM iteration
//! loop, and the two public entry points `train` and `align_corpus`.
//!
//! `pipeline` is a directory module (plans/CONTRACTS.md allows this: "a module may become a
//! directory"). `progress` lives inside it rather than as a top-level `progress.rs`
//! because `lib.rs` is owned by the coordinator and declares only the contract's
//! modules.

pub mod align;
pub mod chunk;
pub mod features;
pub mod full_pass;
pub mod inmem;
pub mod iterate;
pub mod progress;
pub mod refine;
pub mod stats;

use anyhow::{Context, Result, anyhow};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::path::Path;
use viter_io::corpus::Corpus;
use viter_kaldi::device::Device;
use viter_kaldi::gmm::AmDiagGmm;
use viter_kaldi::hmm::{ContextDependency, HmmTopology, TransitionModel};
use viter_kaldi::model::AcousticModel;
use viter_kaldi::transform::Mat;
use viter_kaldi::types::{Alignment, PdfId, PhoneId, Pronunciation};

pub use align::{AlignOutcome, GraphSet};
pub use features::{FeatureKind, FeatureStore};
pub use full_pass::{AlignOverrides, align_corpus, align_corpus_with, align_subset_with_previous};
pub use inmem::align_corpus_with_audio;
pub use iterate::{IterationHooks, IterationPlan, NoHooks, run_iterations};
pub use progress::{IterationSummary, Progress};
pub use stats::{Stats, UpdateOptions};

use crate::config::{StageSpec, TrainConfig};
use crate::{lda, mono, pronprob, sat, tri};

/// Result of a training run.
pub struct Trained {
    pub model: AcousticModel,
    /// Successful final alignments in `corpus.utts` order; empty when the pass is skipped.
    pub alignments: Vec<Alignment>,
}

/// Output options for a training run, separate from the training recipe.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrainOptions {
    /// Align the full corpus with the finished model. Defaults to false.
    pub final_alignment: bool,
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
    /// Subset size for the stage currently running (0 = whole corpus).
    pub stage_subset: usize,
    /// Learned pronunciation probabilities, once a `PronProbs` round has run.
    pub lexicon_probs: Option<crate::pronprob::LexiconProbs>,
    /// Graph options for the stage currently running; the pron-prob round updates
    /// the global silence probabilities here (`trainer.py` -> dictionary silence probs).
    pub graph: viter_kaldi::hmm::GraphOptions,
    /// Current model, `None` before monophone initialization.
    pub model: Option<ModelState>,
    /// Alignments for the utterances of the stage that just finished, indexed by
    /// corpus utterance index. `None` where alignment failed or the utterance was
    /// not in that stage's subset.
    pub alignments: Vec<Option<Alignment>>,
}

impl<'a> StageCtx<'a> {
    pub fn model(&self) -> &ModelState {
        self.model
            .as_ref()
            .expect("stage ran before monophone initialization")
    }
    pub fn model_mut(&mut self) -> &mut ModelState {
        self.model
            .as_mut()
            .expect("stage ran before monophone initialization")
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
    pub fn subset_for(&self) -> Vec<usize> {
        let n = self.corpus.utts.len();
        let want = self.cfg.subset_size(self.stage_subset, n);
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
        let u = &self.corpus.utts[utt];
        match &self.lexicon_probs {
            None => u.prons.clone(),
            Some(probs) => u
                .words
                .iter()
                .zip(&u.prons)
                .map(|(w, prons)| crate::pronprob::apply(probs, w, prons))
                .collect(),
        }
    }
}

/// Output every stage returns.
pub struct StageOutput {
    /// Utterance indices this stage trained on.
    pub utts: Vec<usize>,
    /// Alignments for those utterances, in `utts` order.
    pub alignments: Vec<Option<Alignment>>,
}

/// Scatter a stage's alignments back into a corpus-indexed vector.
pub fn scatter(
    n: usize,
    utts: &[usize],
    alignments: Vec<Option<Alignment>>,
) -> Vec<Option<Alignment>> {
    let mut out = vec![None; n];
    for (slot, ali) in utts.iter().zip(alignments) {
        out[*slot] = ali;
    }
    out
}

/// Train an acoustic model and align the full corpus with it.
/// Use `train_with` to skip the final alignment when only the model is needed.
pub fn train(
    corpus: &Corpus,
    cfg: &TrainConfig,
    device: &Device,
    out_dir: Option<&Path>,
) -> Result<Trained> {
    train_with(
        corpus,
        cfg,
        device,
        out_dir,
        &TrainOptions {
            final_alignment: true,
        },
    )
}

/// Train an acoustic model, optionally aligning the full corpus afterwards.
pub fn train_with(
    corpus: &Corpus,
    cfg: &TrainConfig,
    device: &Device,
    out_dir: Option<&Path>,
    opts: &TrainOptions,
) -> Result<Trained> {
    if corpus.utts.is_empty() {
        return Err(anyhow!("cannot train on an empty corpus"));
    }
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating output directory {}", dir.display()))?;
    }

    let progress = Progress::new();
    let n = corpus.utts.len();
    let schedule = cfg.effective_schedule(n);

    // Relative cost per stage drives the overall % and ETA; features and the final
    // two-pass alignment count as a few passes over the whole corpus.
    let mut plan: Vec<(String, f64)> = vec![("features".to_string(), 2.0 * n as f64)];
    for spec in &schedule {
        plan.push((spec.key(), spec.cost(n, cfg)));
    }
    if opts.final_alignment {
        plan.push(("final".to_string(), 4.0 * n as f64));
    }
    let plan_refs: Vec<(&str, f64)> = plan.iter().map(|(k, w)| (k.as_str(), *w)).collect();
    progress.plan(&plan_refs);
    progress.stage(
        "features",
        &format!(
            "{} utterances, {} speakers",
            corpus.utts.len(),
            corpus.speakers.len()
        ),
    );

    let feats = FeatureStore::build_with(
        corpus,
        &cfg.mfcc,
        &cfg.deltas,
        cfg.lda.splice_left,
        cfg.lda.splice_right,
        &progress,
    )?;
    let total_frames: usize = (0..corpus.utts.len()).map(|i| feats.num_frames(i)).sum();
    progress.stage_done(
        "features",
        &format!(
            "{} utterances · {:.1} h · {} speaker{} · {} dims",
            corpus.utts.len(),
            total_frames as f64 * feats.frame_shift_s() as f64 / 3600.0,
            corpus.speakers.len(),
            if corpus.speakers.len() == 1 { "" } else { "s" },
            cfg.mfcc.num_ceps
        ),
    );

    let mut ctx = StageCtx {
        corpus,
        feats,
        device,
        cfg,
        progress: &progress,
        rng: Xoshiro256PlusPlus::seed_from_u64(cfg.seed),
        silence_phones: corpus.silence_phones.clone(),
        stage_subset: 0,
        lexicon_probs: None,
        graph: cfg.graph.clone(),
        model: None,
        alignments: vec![None; corpus.utts.len()],
    };

    for spec in &schedule {
        let key = spec.key();
        ctx.stage_subset = spec.subset();
        let out = match spec {
            StageSpec::Mono { .. } => Some(mono::run(&mut ctx, &cfg.mono)?),
            StageSpec::Tri {
                num_leaves,
                max_gaussians,
                ..
            } => Some(tri::run(
                &mut ctx,
                &crate::config::TriConfig {
                    num_leaves: *num_leaves,
                    max_gaussians: *max_gaussians,
                    ..cfg.tri.clone()
                },
            )?),
            StageSpec::Lda {
                num_leaves,
                max_gaussians,
                ..
            } => Some(lda::run(
                &mut ctx,
                &crate::config::LdaConfig {
                    num_leaves: *num_leaves,
                    max_gaussians: *max_gaussians,
                    ..cfg.lda.clone()
                },
            )?),
            StageSpec::Sat {
                num_leaves,
                max_gaussians,
                num_iterations,
                quick,
                ..
            } => Some(sat::run(
                &mut ctx,
                &crate::config::SatConfig {
                    num_leaves: *num_leaves,
                    max_gaussians: *max_gaussians,
                    num_iterations: *num_iterations,
                    quick: *quick,
                    ..cfg.sat.clone()
                },
                &key,
            )?),
            StageSpec::PronProbs { .. } => {
                run_pron_probs(&mut ctx, &key)?;
                None
            }
        };
        if let Some(out) = out {
            ctx.alignments = scatter(corpus.utts.len(), &out.utts, out.alignments);
        }
        save_stage(&ctx, out_dir, &key)?;
    }

    let model = build_model(&ctx)?;

    if !opts.final_alignment {
        progress.stage_done("training", "model trained");
        progress.finish();
        return Ok(Trained {
            model,
            alignments: Vec::new(),
        });
    }

    // Final pass: align the whole corpus with the finished model.
    progress.stage(
        "final",
        &format!("aligning {} utterances", corpus.utts.len()),
    );
    let final_alignments = full_pass::final_alignment(&ctx, &model)?;
    let failed = final_alignments.iter().filter(|a| a.is_none()).count();
    if failed > 0 {
        progress.warn(format!(
            "{failed} utterances failed to align in the final pass"
        ));
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

/// MFA's `pronunciation_probabilities` round (`trainer.py:215,219`): align the
/// round's subset with the model just trained (two-pass fMLLR when it is a SAT
/// model), estimate per-pronunciation probabilities and the global silence
/// probabilities from those alignments, and use them for every graph built after.
fn run_pron_probs(ctx: &mut StageCtx<'_>, key: &str) -> Result<()> {
    let utts = ctx.subset_for();
    ctx.progress
        .stage(key, &format!("{} utterances", utts.len()));

    let alignments = align_for_pron_probs(ctx, &utts)?;

    let probs = {
        let m = ctx.model();
        pronprob::estimate(
            ctx.corpus,
            &m.tm,
            ctx.graph.silence_phone,
            &utts,
            &alignments,
        )
    };

    // The learned global silence probabilities replace the graph defaults for every
    // later stage, exactly as MFA writes them back onto the dictionary.
    ctx.graph.silence_prob = probs.silence_prob;
    ctx.graph.initial_silence_prob = probs.initial_silence_prob;
    ctx.graph.final_silence_correction = probs.final_silence_correction;
    ctx.graph.final_non_silence_correction = probs.final_non_silence_correction;

    let words = probs.words.len();
    ctx.lexicon_probs = Some(probs);
    ctx.alignments = scatter(ctx.corpus.utts.len(), &utts, alignments);

    ctx.progress.stage_done(
        key,
        &format!(
            "{words} words · silence p={:.3} initial={:.3}",
            ctx.graph.silence_prob, ctx.graph.initial_silence_prob
        ),
    );
    Ok(())
}

/// Align a subset with the current model, including the fMLLR second pass when the
/// model is speaker adapted. Used by the pronunciation-probability round.
fn align_for_pron_probs(ctx: &mut StageCtx<'_>, utts: &[usize]) -> Result<Vec<Option<Alignment>>> {
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

    let lda = ctx.model().lda.clone();
    let mut feats = match lda.as_ref() {
        Some(l) => ctx.feats.feats_for_many(utts, FeatureKind::SpliceLda(l)),
        None => ctx.feats.feats_for_many(utts, FeatureKind::Deltas),
    };

    let bar = ctx.progress.bar("align", utts.len() as u64);
    let mut outcome = {
        let m = ctx.model();
        align::align_batch(
            &graphs,
            &m.tm,
            m.am_si.as_ref().unwrap_or(&m.am),
            ctx.device,
            &feats,
            &ctx.cfg.align,
            ctx.cfg.batch_utts,
            &bar,
        )
    };
    bar.finish();

    // MFA's pronunciation-probability stage counts from the *speaker-independent*
    // single-pass alignment of the previous stage (`trainer.py:600` -> `align()` with
    // `uses_speaker_adaptation = False`, `alignment/base.py:285`), never from an
    // fMLLR-adapted second pass, so the second pass below is disabled.
    if false && ctx.model().am_si.is_some() {
        let transforms = {
            let m = ctx.model();
            sat::estimate_fmllr(
                ctx,
                &m.tm,
                &m.am,
                utts,
                &outcome.alignments,
                &feats,
                &ctx.cfg.sat,
            )?
        };
        for (spk, t) in transforms.into_iter().enumerate() {
            if let Some(mat) = t {
                ctx.feats.set_fmllr(spk, mat);
            }
        }
        let adapted_kind = match lda.as_ref() {
            Some(l) => FeatureKind::SpliceLdaFmllr(l),
            None => FeatureKind::Deltas,
        };
        feats = ctx.feats.feats_for_many(utts, adapted_kind);
        let bar = ctx.progress.bar("align (fmllr)", utts.len() as u64);
        outcome = {
            let m = ctx.model();
            align::align_batch(
                &graphs,
                &m.tm,
                &m.am,
                ctx.device,
                &feats,
                &ctx.cfg.align,
                ctx.cfg.batch_utts,
                &bar,
            )
        };
        bar.finish();
    }
    Ok(outcome.alignments)
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
        deltas: if m.lda.is_none() {
            Some(ctx.cfg.deltas.clone())
        } else {
            None
        },
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
        graph_opts: ctx.graph.clone(),
        lexicon_probs: ctx.lexicon_probs.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_places_alignments_at_corpus_indices() {
        let utts = vec![1usize, 3];
        let alis = vec![
            Some(Alignment {
                utt: "a".into(),
                tids: vec![1],
                words: vec![],
                prons: vec![],
                loglike: 0.0,
            }),
            None,
        ];
        let out = scatter(5, &utts, alis);
        assert_eq!(out.len(), 5);
        assert!(out[0].is_none());
        assert_eq!(out[1].as_ref().unwrap().utt, "a");
        assert!(out[3].is_none());
    }
}
