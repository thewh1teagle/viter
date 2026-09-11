//! Full-corpus alignment passes: training's `final_alignment` and the `viter align`
//! entry points.
//!
//! Both do the same thing — build graphs, derive features, align, optionally
//! estimate fMLLR and realign on the adapted features — over every utterance in the
//! corpus. Both do it one speaker-aligned chunk at a time (see [`super::chunk`]):
//! the base MFCCs stay resident but the derived views a pass consumes (39-dim
//! deltas, 40-dim splice+LDA, ~0.2 GB per hour of audio) are built for one chunk and
//! dropped before the next, so peak memory is the store plus one chunk. Every
//! per-utterance computation is unchanged; a chunk may close mid-speaker, so the
//! per-speaker fMLLR estimation is accumulate-then-solve
//! ([`sat::FmllrEstimator`]): pass 1 walks every chunk accumulating the speaker
//! statistics, one solve produces the transforms, and pass 2 walks the chunks again
//! on the adapted features. The pass-1 alignments for the whole corpus are kept
//! between the walks (one `u32` per frame, far smaller than the feature views).

use anyhow::{Context, Result};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::time::Instant;
use viter_io::corpus::Corpus;
use viter_kaldi::align::AlignOptions;
use viter_kaldi::device::Device;
use viter_kaldi::hmm;
use viter_kaldi::model::AcousticModel;
use viter_kaldi::transform::Mat;
use viter_kaldi::types::{Alignment, Feats, IntervalAlignment};

use super::progress::{Phase, Progress};
use super::{FeatureKind, FeatureStore, GraphSet, StageCtx, align, chunk, refine, sat};
use crate::config::TrainConfig;

/// Final full-corpus alignment with the trained model, including the fMLLR two-pass
/// when the model is speaker-adapted.
pub(super) fn final_alignment(
    ctx: &StageCtx<'_>,
    model: &AcousticModel,
    sat_ran: bool,
) -> Result<Vec<Option<Alignment>>> {
    let utts: Vec<usize> = (0..ctx.corpus.utts.len()).collect();

    // MFA's trainer aligns the corpus at the end with `boost_silence = 1.5`
    // (`trainer.py:187`, carried into both fMLLR passes via `align_options`);
    // a standalone `mfa align` uses 1.0, which is what `viter align` does.
    const TRAINER_FINAL_BOOST: f32 = 1.5;
    let silence_pdfs = model.tm.silence_pdfs(&model.silence_phones);

    let mut out: Vec<Option<Alignment>> = vec![None; utts.len()];
    let chunks = chunk::by_frames(&ctx.feats, &utts, chunk::frames_for(ctx.cfg));

    // Graphs once for the whole corpus, in chunk order, so each chunk is a window
    // on the set (as the training stages do). They are small, and pass 2 needs the
    // same graphs again. Chunk `k` owns `offsets[k]..offsets[k + 1]`.
    let order: Vec<usize> = chunks.iter().flatten().copied().collect();
    let mut offsets = Vec::with_capacity(chunks.len() + 1);
    offsets.push(0usize);
    for c in &chunks {
        offsets.push(offsets.last().unwrap() + c.len());
    }
    let bar = ctx.progress.bar("graphs", utts.len() as u64);
    let graphs = GraphSet::build(
        order.len(),
        |i| ctx.words_of(order[i]),
        &model.tm,
        &model.ctx,
        &ctx.graph,
        &bar,
    );
    bar.finish();

    // Speaker-independent model present means SAT: estimate fMLLR from the first
    // pass and realign on adapted features.
    let mut estimator = model
        .am_si
        .is_some()
        .then(|| sat::FmllrEstimator::new(ctx, &model.am));

    // Pass 1: align every chunk with the speaker-independent model, accumulating the
    // fMLLR statistics as we go. The alignments are kept for the whole corpus.
    let mut align_bar = Phase::new(ctx.progress, "final align", utts.len() as u64);
    for (k, c) in chunks.iter().enumerate() {
        let sub = graphs.slice(offsets[k], offsets[k + 1]);
        let store_feats = FeatureView::for_model(ctx, model, c);
        let bar = align_bar.bar();
        let outcome = align::align_boosted(
            sub,
            &model.tm,
            model.am_si.as_ref().unwrap_or(&model.am),
            &silence_pdfs,
            TRAINER_FINAL_BOOST,
            ctx.device,
            &store_feats.feats,
            &ctx.cfg.align,
            ctx.cfg.batch_utts,
            &bar,
        );
        bar.finish();
        align_bar.done(c.len() as u64);

        if let Some(est) = estimator.as_mut() {
            est.accumulate(
                ctx,
                &model.tm,
                &model.am,
                c,
                &outcome.alignments,
                &store_feats.feats,
                &ctx.cfg.sat,
            )?;
        }

        for (&u, a) in c.iter().zip(outcome.alignments) {
            out[u] = a;
        }
    }

    // Pass 2: one solve over all speakers, then realign every chunk on the adapted
    // features with the same graphs.
    if let Some(est) = estimator {
        let transforms = est.solve(ctx, &ctx.cfg.sat)?;
        let mut adapt_bar = Phase::new(ctx.progress, "final align (fmllr)", utts.len() as u64);
        for (k, c) in chunks.iter().enumerate() {
            let sub = graphs.slice(offsets[k], offsets[k + 1]);
            let mut store_feats = FeatureView::for_model(ctx, model, c);
            store_feats.apply_fmllr(ctx, c, &transforms);
            let bar = adapt_bar.bar();
            let adapted = align::align_boosted(
                sub,
                &model.tm,
                &model.am,
                &silence_pdfs,
                TRAINER_FINAL_BOOST,
                ctx.device,
                &store_feats.feats,
                &ctx.cfg.align,
                ctx.cfg.batch_utts,
                &bar,
            );
            bar.finish();
            adapt_bar.done(c.len() as u64);
            for (&u, a) in c.iter().zip(adapted.alignments) {
                out[u] = a;
            }
        }
    } else if sat_ran {
        // The plan counts the "final align (fmllr)" pass whenever the schedule
        // contained a SAT stage; this model has no `am_si`, so credit it.
        ctx.progress.skip(utts.len() as u64);
    }

    Ok(out)
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
        self.feats.par_iter_mut().enumerate().for_each(|(i, f)| {
            let spk = ctx.feats.speaker_of(utts[i]);
            if let Some(x) = &transforms[spk] {
                *f = viter_kaldi::feat::apply_transform(f, x);
            }
        });
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
        &ctx.graph,
        &bar,
    );
    bar.finish();
    let bar = ctx
        .progress
        .bar("align (previous model)", utts.len() as u64);
    // Alignment workflows use MFA's align defaults: boost_silence 1.0 (`alignment/mixins.py:69-91`).
    // With a SAT model the speaker-independent alignment model is the one that works
    // on unadapted features (MFA aligns with `final.alimdl`).
    let outcome = align::align_batch(
        &graphs,
        &m.tm,
        m.am_si.as_ref().unwrap_or(&m.am),
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
    align_corpus_with(corpus, model, device, opts, &AlignOverrides::default())
}

/// Align-time knobs that are not part of the model: MFA's global silence
/// probabilities (`silence_probability`, `initial_silence_probability`,
/// `final_silence_correction`, `final_non_silence_correction`, learned during
/// its pronunciation-probability stage and stored in the model meta) and the
/// silence boost applied to the acoustic model during alignment.
#[derive(Clone, Debug, Default)]
pub struct AlignOverrides {
    pub silence_prob: Option<f32>,
    pub initial_silence_prob: Option<f32>,
    pub final_silence_correction: Option<f32>,
    pub final_non_silence_correction: Option<f32>,
    /// 1.0 = no boost (MFA's align default).
    pub boost_silence: Option<f32>,
    /// Refine phone boundaries to 1 ms after alignment (see [`refine`]).
    pub refine: Option<refine::RefineOptions>,
}

pub fn align_corpus_with(
    corpus: &Corpus,
    model: &AcousticModel,
    device: &Device,
    opts: Option<&AlignOptions>,
    over: &AlignOverrides,
) -> Result<Vec<Option<IntervalAlignment>>> {
    align_corpus_reading(corpus, model, device, opts, over, &|_, u| {
        viter_kaldi::audio::read_16k(&u.audio)
            .with_context(|| format!("reading audio for utterance {}", u.id))
    })
}

/// Shared body of [`align_corpus_with`] and [`super::align_corpus_with_audio`]:
/// `audio_of` supplies the 16 kHz waveform of each utterance (for MFCC and for
/// refinement).
pub(crate) fn align_corpus_reading(
    corpus: &Corpus,
    model: &AcousticModel,
    device: &Device,
    opts: Option<&AlignOptions>,
    over: &AlignOverrides,
    audio_of: &(
         dyn Fn(usize, &viter_kaldi::types::Utterance) -> Result<viter_kaldi::audio::Audio> + Sync
     ),
) -> Result<Vec<Option<IntervalAlignment>>> {
    if corpus.utts.is_empty() {
        return Ok(Vec::new());
    }
    let progress = Progress::new();
    let align_opts = opts.cloned().unwrap_or_default();

    progress.stage("align", &format!("{} utterances", corpus.utts.len()));

    let mut graph = model.graph_opts.clone();
    // A model that learned pronunciation probabilities also learned the global
    // silence probabilities; use them unless the caller overrides.
    if let Some(lp) = &model.lexicon_probs {
        graph.silence_prob = lp.silence_prob;
        graph.initial_silence_prob = lp.initial_silence_prob;
        graph.final_silence_correction = lp.final_silence_correction;
        graph.final_non_silence_correction = lp.final_non_silence_correction;
    }
    if let Some(v) = over.silence_prob {
        graph.silence_prob = v;
    }
    if let Some(v) = over.initial_silence_prob {
        graph.initial_silence_prob = v;
    }
    if let Some(v) = over.final_silence_correction {
        graph.final_silence_correction = v;
    }
    if let Some(v) = over.final_non_silence_correction {
        graph.final_non_silence_correction = v;
    }
    let boost = over.boost_silence.unwrap_or(1.0);
    let cfg = TrainConfig {
        mfcc: model.mfcc.clone(),
        deltas: model.deltas.clone().unwrap_or_default(),
        align: align_opts.clone(),
        graph,
        ..TrainConfig::default()
    };
    let (sl, sr) = model
        .splice
        .unwrap_or((cfg.lda.splice_left, cfg.lda.splice_right));

    let feats = FeatureStore::build_with_audio(
        corpus,
        &model.mfcc,
        &cfg.deltas,
        sl,
        sr,
        &progress,
        audio_of,
    )?;
    let frame_shift_s = feats.frame_shift_s();

    let ctx = StageCtx {
        corpus,
        feats,
        device,
        cfg: &cfg,
        progress: &progress,
        rng: Xoshiro256PlusPlus::seed_from_u64(cfg.seed),
        silence_phones: model.silence_phones.clone(),
        stage_subset: 0,
        lexicon_probs: model.lexicon_probs.clone(),
        graph: cfg.graph.clone(),
        model: None,
        alignments: Vec::new(),
    };

    let utts: Vec<usize> = (0..corpus.utts.len()).collect();

    // Pass 1: speaker-independent model if we have one, else the single model.
    let first_model = model.am_si.as_ref().unwrap_or(&model.am);
    let silence_pdfs = model.tm.silence_pdfs(&model.silence_phones);

    let mut alignments: Vec<Option<Alignment>> = vec![None; utts.len()];
    let mut transforms: Vec<Option<Mat>> = vec![None; ctx.feats.num_speakers()];
    let mut failed = 0usize;

    let chunks = chunk::by_frames(&ctx.feats, &utts, chunk::frames_for(&cfg));
    let mut graph_bar = Phase::new(&progress, "graphs", utts.len() as u64);
    let mut align_bar = Phase::new(&progress, "align", utts.len() as u64);

    let fmllr_cfg = crate::config::SatConfig {
        fmllr: model.fmllr.clone().unwrap_or_else(|| cfg.sat.fmllr.clone()),
        ..cfg.sat.clone()
    };
    let mut estimator = model
        .am_si
        .is_some()
        .then(|| sat::FmllrEstimator::new(&ctx, &model.am));

    // Pass 1 over every chunk: align with the speaker-independent model and fold the
    // chunk's fMLLR statistics into the per-speaker accumulators. Chunks may close
    // mid-speaker, so the solve waits until every chunk has contributed.
    for c in &chunks {
        let _tg = Instant::now();
        let bar = graph_bar.bar();
        let graphs = GraphSet::build(
            c.len(),
            |i| ctx.words_of(c[i]),
            &model.tm,
            &model.ctx,
            &ctx.graph,
            &bar,
        );
        bar.finish();
        graph_bar.done(c.len() as u64);
        tracing::debug!(graphs_ms = _tg.elapsed().as_millis(), "graphs built");

        let _tf = Instant::now();
        let view = FeatureView::for_model(&ctx, model, c);
        tracing::debug!(
            feats_ms = _tf.elapsed().as_millis(),
            "stage features derived"
        );

        let bar = align_bar.bar();
        let outcome = align::align_boosted(
            &graphs,
            &model.tm,
            first_model,
            &silence_pdfs,
            boost,
            device,
            &view.feats,
            &align_opts,
            cfg.batch_utts,
            &bar,
        );
        bar.finish();
        align_bar.done(c.len() as u64);

        if let Some(est) = estimator.as_mut() {
            est.accumulate(
                &ctx,
                &model.tm,
                &model.am,
                c,
                &outcome.alignments,
                &view.feats,
                &fmllr_cfg,
            )?;
        } else {
            failed += outcome.failed;
        }
        for (&u, a) in c.iter().zip(outcome.alignments) {
            alignments[u] = a;
        }
    }

    // Pass 2: solve the per-speaker transforms once, then realign every chunk on the
    // adapted features.
    if let Some(est) = estimator {
        transforms = est.solve(&ctx, &fmllr_cfg)?;
        let mut adapt_bar = Phase::new(&progress, "align (fmllr)", utts.len() as u64);
        let mut graph2_bar = Phase::new(&progress, "graphs", utts.len() as u64);
        for c in &chunks {
            let bar = graph2_bar.bar();
            let graphs = GraphSet::build(
                c.len(),
                |i| ctx.words_of(c[i]),
                &model.tm,
                &model.ctx,
                &ctx.graph,
                &bar,
            );
            bar.finish();
            graph2_bar.done(c.len() as u64);

            let mut view = FeatureView::for_model(&ctx, model, c);
            view.apply_fmllr(&ctx, c, &transforms);
            let bar = adapt_bar.bar();
            let outcome = align::align_boosted(
                &graphs,
                &model.tm,
                &model.am,
                &silence_pdfs,
                boost,
                device,
                &view.feats,
                &align_opts,
                cfg.batch_utts,
                &bar,
            );
            bar.finish();
            adapt_bar.done(c.len() as u64);
            failed += outcome.failed;
            for (&u, a) in c.iter().zip(outcome.alignments) {
                alignments[u] = a;
            }
        }
    }

    if failed > 0 {
        progress.warn(format!(
            "{} of {} utterances failed to align",
            failed,
            utts.len()
        ));
    }

    let mut intervals: Vec<Option<IntervalAlignment>> = alignments
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

    if let Some(ropts) = &over.refine {
        let bar = progress.bar("refine", utts.len() as u64);
        let feats = &ctx.feats;
        let lda = model.lda.as_ref().map(|m| ((sl, sr), m));
        let alignments = &alignments;
        intervals = intervals
            .par_iter()
            .enumerate()
            .map(|(i, iv)| {
                bar.inc(1);
                let iv = iv.as_ref()?;
                let ali = alignments[i].as_ref()?;
                let audio = audio_of(i, &corpus.utts[i]).ok()?;
                let setup = refine::UttFeatureSetup {
                    mfcc: &model.mfcc,
                    cmvn: feats.cmvn_stats(feats.speaker_of(i)),
                    deltas: &cfg.deltas,
                    lda,
                    fmllr: transforms[feats.speaker_of(i)].as_ref(),
                };
                Some(refine::refine_utterance(
                    &audio.samples,
                    &setup,
                    model,
                    ali,
                    iv,
                    ropts,
                ))
            })
            .collect();
        bar.finish();
    }

    progress.stage_done(
        "align",
        &format!("{} aligned, {} failed", utts.len() - failed, failed),
    );
    progress.finish();
    Ok(intervals)
}

#[cfg(test)]
mod tests {

    #[test]
    fn scatter_into_corpus_order_is_independent_of_chunking() {
        // What the chunk loop does with each chunk's results: whatever grouping the
        // chunker produces, writing `out[u]` puts every utterance back at its corpus
        // index, so the output order never depends on the chunk boundaries.
        let n = 6usize;
        let whole = vec![(0..n).collect::<Vec<_>>()];
        let split = vec![vec![0usize, 1], vec![2, 3, 4], vec![5]];
        let value = |u: usize| format!("u{u}");

        let run = |chunks: &[Vec<usize>]| {
            let mut out: Vec<Option<String>> = vec![None; n];
            for c in chunks {
                let results: Vec<Option<String>> = c.iter().map(|&u| Some(value(u))).collect();
                for (&u, a) in c.iter().zip(results) {
                    out[u] = a;
                }
            }
            out
        };
        assert_eq!(run(&whole), run(&split));
        assert_eq!(run(&split)[3].as_deref(), Some("u3"));
    }

    #[test]
    fn transforms_merge_keeps_one_entry_per_speaker() {
        // `estimate_fmllr` returns a full-length vector with `Some` only for the
        // speakers in the chunk; merging chunk results must fill in exactly those.
        let num_speakers = 4;
        let mut global: Vec<Option<u32>> = vec![None; num_speakers];
        for (chunk_of, present) in [(10u32, vec![0usize, 1]), (20, vec![3])] {
            let mut ret: Vec<Option<u32>> = vec![None; num_speakers];
            for &s in &present {
                ret[s] = Some(chunk_of + s as u32);
            }
            for (spk, t) in ret.into_iter().enumerate() {
                if let Some(v) = t {
                    global[spk] = Some(v);
                }
            }
        }
        assert_eq!(global, vec![Some(10), Some(11), None, Some(23)]);
    }
}
