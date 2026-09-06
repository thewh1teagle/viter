//! Shared statistics accumulation and model update.
//!
//! One iteration of every GMM stage is: accumulate GMM + transition stats over the
//! stage's utterances from the current alignments, MLE-update the transition model,
//! then MLE-update the acoustic model with the iteration's mixup target. This mirrors
//! MFA `acoustic_modeling/base.py:281-349 acc_stats` exactly, including the ordering
//! (transitions first, then GMM) and the fact that stats are accumulated with the
//! unboosted model.

use viter_kaldi::gmm::{
    self, AccumAmDiagGmm, AmDiagGmm, GmmFlags, MleDiagGmmOptions,
};
use viter_kaldi::hmm::{MleTransitionUpdateConfig, TransitionAccs, TransitionModel};
use viter_kaldi::types::{Alignment, Feats};
use rayon::prelude::*;

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
}

/// Accumulate over every aligned utterance in parallel.
///
/// Each rayon task builds its own accumulators and they are reduced with `add`, which
/// is exactly how MFA sums the per-job accumulators (`base.py:309-321`).
/// Utterances whose alignment is `None` are skipped.
pub fn accumulate(
    am: &AmDiagGmm,
    tm: &TransitionModel,
    alignments: &[Option<Alignment>],
    feats: &[Feats],
    bar: &Bar,
) -> Stats {
    accumulate_inner(am, tm, alignments, feats, None, bar)
}

/// As `accumulate`, but posteriors come from `post_feats` while the statistics are
/// gathered on `stats_feats`.
///
/// With the same slice for both this is plain `gmm-acc-stats-ali`. With adapted
/// features for posteriors and speaker-independent features for stats it is
/// `gmm-acc-stats-twofeats`, which SAT uses to build the SI alignment model
/// (`acoustic_modeling/sat.py:313-376`, kalpy `gmm/train.py:161 TwoFeatsStatsAccumulator`).
pub fn accumulate_with(
    am: &AmDiagGmm,
    tm: &TransitionModel,
    alignments: &[Option<Alignment>],
    post_feats: &[Feats],
    stats_feats: &[Feats],
    bar: &Bar,
) -> Stats {
    accumulate_inner(am, tm, alignments, post_feats, Some(stats_feats), bar)
}

/// `stats_feats = None` means single-feature accumulation.
fn accumulate_inner(
    am: &AmDiagGmm,
    tm: &TransitionModel,
    alignments: &[Option<Alignment>],
    post_feats: &[Feats],
    stats_feats: Option<&[Feats]>,
    bar: &Bar,
) -> Stats {
    let identity = || Stats {
        gmm: AccumAmDiagGmm::new(am, GmmFlags::ALL),
        transitions: TransitionAccs(vec![0.0; tm.num_transition_ids() + 1]),
    };

    let reduced = (0..alignments.len())
        .into_par_iter()
        .fold(identity, |mut acc, i| {
            if let Some(ali) = &alignments[i] {
                accumulate_one(
                    am,
                    tm,
                    ali,
                    &post_feats[i],
                    stats_feats.map(|s| &s[i]),
                    &mut acc,
                );
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
        });
    reduced
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
                out.gmm.accumulate_from_posteriors(pdf, stats_x, &posteriors);
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
    if opts.mixup > 0 && opts.mixup > am.num_gauss() {
        let occs = stats.gmm.pdf_occupancies();
        am.split_by_count(&occs, opts.mixup, opts.perturb_factor, opts.power, gopts.min_gaussian_occupancy as f32, rng);
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

    #[test]
    fn loglike_per_frame_handles_empty() {
        let s = Stats {
            gmm: AccumAmDiagGmm { accs: Vec::new(), total_frames: 0.0, total_loglike: -5.0 },
            transitions: TransitionAccs(Vec::new()),
        };
        assert_eq!(s.loglike_per_frame(), 0.0);
    }
}
