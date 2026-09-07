//! Shared statistics accumulation and model update.
//!
//! One iteration of every GMM stage is: accumulate GMM + transition stats over the
//! stage's utterances from the current alignments, MLE-update the transition model,
//! then MLE-update the acoustic model with the iteration's mixup target. This mirrors
//! MFA `acoustic_modeling/base.py:281-349 acc_stats` exactly, including the ordering
//! (transitions first, then GMM) and the fact that stats are accumulated with the
//! unboosted model.

use rayon::prelude::*;
use viter_kaldi::device::Device;
use viter_kaldi::gmm::{self, AccumAmDiagGmm, AmDiagGmm, GmmFlags, MleDiagGmmOptions};
use viter_kaldi::hmm::{MleTransitionUpdateConfig, TransitionAccs, TransitionModel};
use viter_kaldi::types::{Alignment, Feats, PdfId};

use super::progress::Bar;

/// GMM + transition statistics for one iteration.
pub struct Stats {
    pub gmm: AccumAmDiagGmm,
    pub transitions: TransitionAccs,
}

impl Stats {
    /// Average acoustic log-likelihood per frame, the number MFA logs each iteration
    /// (`base.py:342-348`).
    pub fn loglike_per_frame(&self) -> f64 {
        if self.gmm.total_frames > 0.0 {
            self.gmm.total_loglike / self.gmm.total_frames
        } else {
            0.0
        }
    }
    pub fn total_frames(&self) -> f64 {
        self.gmm.total_frames
    }

    /// Zero statistics shaped for a model, the identity of [`merge`](Self::merge).
    /// A stage with no utterances at all produces this.
    pub fn empty(am: &AmDiagGmm, tm: &TransitionModel) -> Self {
        Self {
            gmm: AccumAmDiagGmm::new(am, GmmFlags::ALL),
            transitions: TransitionAccs(vec![0.0; tm.num_transition_ids() + 1]),
        }
    }

    /// `self += other`, the same sum [`accumulate_with`] performs when rayon joins two
    /// partial accumulators. Chunked passes accumulate one chunk at a time and merge,
    /// which is the same set of f64 adds in a split order rayon could itself produce.
    pub fn merge(&mut self, other: Stats) {
        // `AccumAmDiagGmm::add` sums the per-pdf accumulators and both running totals.
        self.gmm.add(&other.gmm, 1.0);
        for (x, y) in self
            .transitions
            .0
            .iter_mut()
            .zip(other.transitions.0.iter())
        {
            *x += *y;
        }
    }
}

/// Accumulate over every aligned utterance.
///
/// GMM statistics go through [`Device::accumulate_batch`], which on the GPU runs the
/// per-frame posterior and the per-gaussian reduction as two compute kernels against
/// the already-resident packed model. Transition statistics stay on the CPU: they are
/// one counter bump per frame and cost nothing next to the GMM work.
///
/// Utterances whose alignment is `None` are skipped.
pub fn accumulate(
    device: &Device,
    am: &AmDiagGmm,
    tm: &TransitionModel,
    alignments: &[Option<Alignment>],
    feats: &[Feats],
    bar: &Bar,
) -> Stats {
    let mut out = Stats {
        gmm: AccumAmDiagGmm::new(am, GmmFlags::ALL),
        transitions: TransitionAccs(vec![0.0; tm.num_transition_ids() + 1]),
    };

    // Frames aligned to each pdf, plus the transition counts, in one CPU pass.
    // Utterances are handled in batches so the f32 GPU sums stay small (see the
    // precision note in `viter_kaldi::device::accum`).
    let ready: Vec<usize> = (0..alignments.len())
        .filter(|&i| alignments[i].is_some())
        .collect();

    for chunk in ready.chunks(ACCUM_BATCH) {
        let prepared: Vec<(usize, Vec<PdfId>)> = chunk
            .par_iter()
            .map(|&i| {
                let ali = alignments[i].as_ref().expect("filtered to Some");
                let n = ali.tids.len().min(feats[i].nrows());
                let pdfs = (0..n)
                    .map(|t| tm.transition_id_to_pdf(ali.tids[t]))
                    .collect();
                (i, pdfs)
            })
            .collect();

        for (i, _) in &prepared {
            let ali = alignments[*i].as_ref().expect("filtered to Some");
            let n = ali.tids.len().min(feats[*i].nrows());
            for t in 0..n {
                tm.accumulate(&mut out.transitions, ali.tids[t], 1.0);
            }
        }

        let _t = std::time::Instant::now();
        let f: Vec<&Feats> = prepared.iter().map(|(i, _)| &feats[*i]).collect();
        let p: Vec<&[PdfId]> = prepared.iter().map(|(_, v)| v.as_slice()).collect();
        device.accumulate_batch(&f, &p, None, am, &mut out.gmm);
        tracing::debug!(
            us = _t.elapsed().as_micros() as u64,
            utts = chunk.len(),
            "accumulate_batch"
        );
        bar.inc(chunk.len() as u64);
    }
    bar.inc((alignments.len() - ready.len()) as u64);
    out
}

/// Utterances per device accumulation batch. Large enough to fill the GPU, small
/// enough that the f32 per-batch sums round negligibly against the f64 totals.
const ACCUM_BATCH: usize = 256;

/// As `accumulate`, but posteriors come from `post_feats` while the statistics are
/// gathered on `stats_feats`.
///
/// With adapted features for posteriors and speaker-independent features for stats
/// this is `gmm-acc-stats-twofeats`, which SAT uses to build the SI alignment model
/// (`acoustic_modeling/sat.py:313-376`, kalpy `gmm/train.py:161 TwoFeatsStatsAccumulator`).
/// Two feature streams do not fit the single-stream device kernel, so this stays on
/// the CPU.
pub fn accumulate_with(
    am: &AmDiagGmm,
    tm: &TransitionModel,
    alignments: &[Option<Alignment>],
    post_feats: &[Feats],
    stats_feats: &[Feats],
    bar: &Bar,
) -> Stats {
    let identity = || Stats {
        gmm: AccumAmDiagGmm::new(am, GmmFlags::ALL),
        transitions: TransitionAccs(vec![0.0; tm.num_transition_ids() + 1]),
    };

    (0..alignments.len())
        .into_par_iter()
        .fold(identity, |mut acc, i| {
            if let Some(ali) = &alignments[i] {
                accumulate_one(am, tm, ali, &post_feats[i], Some(&stats_feats[i]), &mut acc);
            }
            bar.inc(1);
            acc
        })
        .reduce(identity, |mut a, b| {
            a.gmm.add(&b.gmm, 1.0);
            for (x, y) in a.transitions.0.iter_mut().zip(b.transitions.0.iter()) {
                *x += *y;
            }
            a
        })
}

/// One utterance's contribution. Viterbi alignment means the posterior is 1.0 on the
/// aligned transition id for each frame (kalpy `gmm/train.py` GmmStatsAccumulator).
fn accumulate_one(
    am: &AmDiagGmm,
    tm: &TransitionModel,
    ali: &Alignment,
    post_feats: &Feats,
    stats_feats: Option<&Feats>,
    out: &mut Stats,
) {
    let mut frames = ali.tids.len().min(post_feats.nrows());
    if let Some(sf) = stats_feats {
        frames = frames.min(sf.nrows());
    }
    if frames == 0 {
        return;
    }

    let mut posteriors: Vec<f32> = Vec::new();
    let mut tot_like = 0.0f64;
    for t in 0..frames {
        let tid = ali.tids[t];
        tm.accumulate(&mut out.transitions, tid, 1.0);
        let pdf = tm.transition_id_to_pdf(tid);

        let post_row = post_feats.row(t);
        let post_x = post_row.as_slice().expect("feature rows are contiguous");

        match stats_feats {
            // Posteriors from one feature space, stats gathered in another.
            Some(sf) => {
                let like = am.pdf(pdf).component_posteriors(post_x, &mut posteriors);
                tot_like += like as f64;
                let stats_row = sf.row(t);
                let stats_x = stats_row.as_slice().expect("feature rows are contiguous");
                out.gmm
                    .accumulate_from_posteriors(pdf, stats_x, &posteriors);
            }
            None => {
                let like = out.gmm.accumulate_for_gmm(am, pdf, post_x, 1.0);
                tot_like += like as f64;
            }
        }
    }
    out.gmm.total_frames += frames as f64;
    out.gmm.total_loglike += tot_like;
}

/// Options for the acoustic model update of one iteration.
pub struct UpdateOptions {
    /// Mixup target: `current_gaussians` for this iteration (`base.py:336`).
    pub mixup: usize,
    /// `power` for the split-target allocation (`base.py:336`).
    pub power: f32,
    /// Kaldi `MleDiagGmmOptions::min_gaussian_occupancy`, 10.0 except for monophone's
    /// iteration-0 update where MFA passes 3.0 (`monophone.py:293`).
    pub min_gaussian_occupancy: f64,
    /// `remove_low_count_gaussians`; MFA passes False when building the SI alignment
    /// model (`sat.py:362`).
    pub remove_low_count_gaussians: bool,
    pub perturb_factor: f32,
    /// Kaldi `SplitByCount` min_count: kalpy `mle_update` passes its own default
    /// 20.0 (`gmm.cpp:296-304`), never `min_gaussian_occupancy`.
    pub split_min_count: f32,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            mixup: 0,
            power: 0.25,
            // Kaldi MleDiagGmmOptions default.
            min_gaussian_occupancy: 10.0,
            remove_low_count_gaussians: true,
            // Kaldi gmm-est default perturb factor for splitting.
            perturb_factor: 0.01,
            split_min_count: 20.0,
        }
    }
}

/// Result of one iteration's model update.
pub struct UpdateResult {
    /// GMM objective function improvement per frame.
    pub gmm_objf_per_frame: f64,
    /// Transition model log-likelihood improvement per frame.
    pub transition_objf_per_frame: f64,
    pub num_gauss: usize,
}

/// Update the transition model then the acoustic model, mixing up to the target.
///
/// Order and semantics follow `base.py:330-341`: transitions first, then
/// `mle_update(gmm_accs, mixup=current_gaussians, power=power)`, where kalpy's
/// `mle_update` is MleAmDiagGmmUpdate followed by SplitByCount to the mixup target
/// (`plans/kalpy/extensions/gmm/gmm.cpp:258`).
pub fn update_model(
    stats: &Stats,
    tm: &mut TransitionModel,
    am: &mut AmDiagGmm,
    opts: &UpdateOptions,
    rng: &mut impl rand::Rng,
) -> UpdateResult {
    let tcfg = MleTransitionUpdateConfig::default();
    let (t_objf, t_count) = tm.mle_update(&stats.transitions, &tcfg);

    let gopts = MleDiagGmmOptions {
        min_gaussian_occupancy: opts.min_gaussian_occupancy,
        remove_low_count_gaussians: opts.remove_low_count_gaussians,
        ..MleDiagGmmOptions::default()
    };
    let (g_objf, g_count) = gmm::mle_am_diag_gmm_update(&gopts, &stats.gmm, GmmFlags::ALL, am);

    // Mix up to this iteration's target, allocating gaussians across pdfs by occupancy
    // (Kaldi AmDiagGmm::SplitByCount / GetSplitTargets with `power`).
    // kalpy `mle_update` (gmm.cpp:291-294) calls SplitByCount whenever mixup != 0;
    // GetSplitTargets allocates per pdf from occupancy^power, so pdfs below their
    // own target keep splitting even when the model total already meets `mixup`.
    if opts.mixup > 0 {
        let occs = stats.gmm.pdf_occupancies();
        am.split_by_count(
            &occs,
            opts.mixup,
            opts.perturb_factor,
            opts.power,
            // kalpy mle_update passes a separate split `min_count`, default 20.0
            // (`gmm.cpp:296-304`), never `min_gaussian_occupancy`.
            opts.split_min_count,
            rng,
        );
    }

    UpdateResult {
        gmm_objf_per_frame: if g_count > 0.0 { g_objf / g_count } else { 0.0 },
        transition_objf_per_frame: if t_count > 0.0 { t_objf / t_count } else { 0.0 },
        num_gauss: am.num_gauss(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_options_defaults_match_kaldi() {
        let o = UpdateOptions::default();
        assert_eq!(o.min_gaussian_occupancy, 10.0);
        assert!(o.remove_low_count_gaussians);
    }

    /// Stand-in for one "chunk" of accumulation: the same per-frame adds
    /// `accumulate_one` performs, without needing a trained model.
    fn accumulate_range(range: std::ops::Range<usize>) -> Stats {
        let mut s = Stats {
            gmm: AccumAmDiagGmm {
                accs: vec![viter_kaldi::gmm::AccumDiagGmm::new(2, 3, GmmFlags::ALL)],
                total_frames: 0.0,
                total_loglike: 0.0,
            },
            transitions: TransitionAccs(vec![0.0; 4]),
        };
        for t in range {
            let x = [t as f32 * 0.5, 1.0 - t as f32, 0.25];
            let post = [0.75f32, 0.25];
            s.gmm.accumulate_from_posteriors(0, &x, &post);
            s.gmm.total_loglike += -(t as f64) * 0.125;
            s.transitions.0[t % 4] += 1.0;
        }
        s.gmm.total_frames += 0.0; // frames come from the posteriors above
        s
    }

    #[test]
    fn merging_halves_equals_accumulating_the_whole() {
        let whole = accumulate_range(0..64);
        let mut merged = accumulate_range(0..32);
        merged.merge(accumulate_range(32..64));

        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
        assert!(rel(whole.gmm.total_frames, merged.gmm.total_frames) < 1e-9);
        assert!(rel(whole.gmm.total_loglike, merged.gmm.total_loglike) < 1e-9);
        for (a, b) in whole.transitions.0.iter().zip(merged.transitions.0.iter()) {
            assert_eq!(a, b);
        }
        let (wa, ma) = (&whole.gmm.accs[0], &merged.gmm.accs[0]);
        for (a, b) in wa.occupancy.iter().zip(ma.occupancy.iter()) {
            assert!(rel(*a, *b) < 1e-9);
        }
        for (a, b) in wa.mean_accum.iter().zip(ma.mean_accum.iter()) {
            assert!(rel(*a, *b) < 1e-9);
        }
        for (a, b) in wa.var_accum.iter().zip(ma.var_accum.iter()) {
            assert!(rel(*a, *b) < 1e-9);
        }
    }

    #[test]
    fn loglike_per_frame_handles_empty() {
        let s = Stats {
            gmm: AccumAmDiagGmm {
                accs: Vec::new(),
                total_frames: 0.0,
                total_loglike: -5.0,
            },
            transitions: TransitionAccs(Vec::new()),
        };
        assert_eq!(s.loglike_per_frame(), 0.0);
    }
}
