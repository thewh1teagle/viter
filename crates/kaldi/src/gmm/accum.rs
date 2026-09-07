//! Sufficient statistics for ML re-estimation.
//!
//! Port of the accumulator halves of `gmm/mle-diag-gmm.{h,cc}` and
//! `gmm/mle-am-diag-gmm.{h,cc}`. All accumulators are `f64`, matching Kaldi's
//! `double` stats: a training pass sums millions of frames and `f32` would drift.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::am::AmDiagGmm;
use super::diag::DiagGmm;
use crate::types::PdfId;

/// Which parameters an accumulator carries stats for.
///
/// Kaldi's `GmmFlagsType` (`gmm/model-common.h`): `m` means, `v` variances,
/// `w` weights. Kaldi's transition flag `t` lives in the transition model, not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GmmFlags(pub u8);

impl GmmFlags {
    pub const MEANS: GmmFlags = GmmFlags(0x001);
    pub const VARIANCES: GmmFlags = GmmFlags(0x002);
    pub const WEIGHTS: GmmFlags = GmmFlags(0x004);
    pub const ALL: GmmFlags = GmmFlags(0x001 | 0x002 | 0x004);

    pub fn contains(self, other: GmmFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn intersects(self, other: GmmFlags) -> bool {
        (self.0 & other.0) != 0
    }

    pub fn union(self, other: GmmFlags) -> GmmFlags {
        GmmFlags(self.0 | other.0)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `AugmentGmmFlags` (`gmm/model-common.cc:53`): variances imply means, means
    /// imply weights, and weights are always added, since stats with no weights
    /// would break dimension checks downstream.
    pub fn augment(self) -> GmmFlags {
        debug_assert_eq!(self.0 & !GmmFlags::ALL.0, 0, "invalid GmmFlags bits");
        let mut f = self;
        if f.contains(GmmFlags::VARIANCES) {
            f = f.union(GmmFlags::MEANS);
        }
        if f.contains(GmmFlags::MEANS) {
            f = f.union(GmmFlags::WEIGHTS);
        }
        if !f.contains(GmmFlags::WEIGHTS) {
            tracing::warn!("adding kGmmWeights to empty flags");
            f = f.union(GmmFlags::WEIGHTS);
        }
        f
    }
}

impl std::ops::BitOr for GmmFlags {
    type Output = GmmFlags;
    fn bitor(self, rhs: GmmFlags) -> GmmFlags {
        GmmFlags(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for GmmFlags {
    type Output = GmmFlags;
    fn bitand(self, rhs: GmmFlags) -> GmmFlags {
        GmmFlags(self.0 & rhs.0)
    }
}

impl std::fmt::Display for GmmFlags {
    /// `GmmFlagsToString` (`gmm/model-common.cc:43`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.contains(GmmFlags::MEANS) {
            write!(f, "m")?;
        }
        if self.contains(GmmFlags::VARIANCES) {
            write!(f, "v")?;
        }
        if self.contains(GmmFlags::WEIGHTS) {
            write!(f, "w")?;
        }
        Ok(())
    }
}

/// Zeroth, first and second order stats for one `DiagGmm`.
///
/// `occupancy[g]` is the soft count of frames assigned to component `g`,
/// `mean_accum[g]` the posterior-weighted sum of `x`, and `var_accum[g]` the
/// posterior-weighted sum of `x^2`. `mean_accum`/`var_accum` are `[0, 0]` when the
/// corresponding flag is off, as in Kaldi.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccumDiagGmm {
    pub occupancy: Vec<f64>,
    pub mean_accum: Array2<f64>,
    pub var_accum: Array2<f64>,
    pub flags: GmmFlags,
}

impl AccumDiagGmm {
    /// `AccumDiagGmm::Resize` (`mle-diag-gmm.cc:106`), zero-initialised.
    pub fn new(num_gauss: usize, dim: usize, flags: GmmFlags) -> Self {
        assert!(
            num_gauss > 0 && dim > 0,
            "AccumDiagGmm::new needs positive sizes"
        );
        let flags = flags.augment();
        AccumDiagGmm {
            occupancy: vec![0.0; num_gauss],
            mean_accum: if flags.contains(GmmFlags::MEANS) {
                Array2::zeros((num_gauss, dim))
            } else {
                Array2::zeros((0, 0))
            },
            var_accum: if flags.contains(GmmFlags::VARIANCES) {
                Array2::zeros((num_gauss, dim))
            } else {
                Array2::zeros((0, 0))
            },
            flags,
        }
    }

    /// Accumulator shaped for `gmm`.
    pub fn from_gmm(gmm: &DiagGmm, flags: GmmFlags) -> Self {
        Self::new(gmm.num_gauss(), gmm.dim(), flags)
    }

    pub fn num_gauss(&self) -> usize {
        self.occupancy.len()
    }

    /// Feature dimension; zero when no mean stats are kept.
    pub fn dim(&self) -> usize {
        if self.mean_accum.nrows() > 0 {
            self.mean_accum.ncols()
        } else if self.var_accum.nrows() > 0 {
            self.var_accum.ncols()
        } else {
            0
        }
    }

    /// Zero the stats selected by `flags` (`AccumDiagGmm::SetZero`).
    pub fn set_zero(&mut self, flags: GmmFlags) {
        assert!(
            self.flags.contains(flags),
            "flags in argument do not match the active accumulators"
        );
        if flags.contains(GmmFlags::WEIGHTS) {
            self.occupancy.iter_mut().for_each(|o| *o = 0.0);
        }
        if flags.contains(GmmFlags::MEANS) {
            self.mean_accum.fill(0.0);
        }
        if flags.contains(GmmFlags::VARIANCES) {
            self.var_accum.fill(0.0);
        }
    }

    /// Add `x` weighted by per-component `posteriors`
    /// (`AccumDiagGmm::AccumulateFromPosteriors`, `mle-diag-gmm.cc:171`).
    pub fn accumulate_from_posteriors(&mut self, x: &[f32], posteriors: &[f32]) {
        assert_eq!(
            posteriors.len(),
            self.num_gauss(),
            "AccumDiagGmm::accumulate_from_posteriors: posterior count mismatch"
        );
        let want_means = self.flags.contains(GmmFlags::MEANS);
        let want_vars = self.flags.contains(GmmFlags::VARIANCES);
        if want_means {
            assert_eq!(
                x.len(),
                self.dim(),
                "AccumDiagGmm::accumulate_from_posteriors: dimension mismatch"
            );
        }
        for g in 0..self.num_gauss() {
            let p = posteriors[g] as f64;
            self.occupancy[g] += p;
            if want_means && p != 0.0 {
                for d in 0..x.len() {
                    let xd = x[d] as f64;
                    self.mean_accum[[g, d]] += p * xd;
                    if want_vars {
                        self.var_accum[[g, d]] += p * xd * xd;
                    }
                }
            }
        }
    }

    /// Compute posteriors under `gmm`, scale them by `weight`, and accumulate.
    /// Returns the *unweighted* frame log-likelihood, like Kaldi's
    /// `AccumulateFromDiag` (`mle-diag-gmm.cc:191`).
    pub fn accumulate_from_diag(&mut self, gmm: &DiagGmm, x: &[f32], weight: f32) -> f32 {
        assert_eq!(
            gmm.num_gauss(),
            self.num_gauss(),
            "AccumDiagGmm::accumulate_from_diag: component count mismatch"
        );
        let mut posteriors = Vec::with_capacity(self.num_gauss());
        let log_like = gmm.component_posteriors(x, &mut posteriors);
        for p in posteriors.iter_mut() {
            *p *= weight;
        }
        self.accumulate_from_posteriors(x, &posteriors);
        log_like
    }

    /// Add stats for a single component directly (`AddStatsForComponent`).
    /// Used by the tree-stats path, which already has pooled `x`/`x^2` sums.
    pub fn add_stats_for_component(
        &mut self,
        g: usize,
        occ: f64,
        x_stats: &[f64],
        x2_stats: &[f64],
    ) {
        assert!(g < self.num_gauss(), "component index out of range");
        self.occupancy[g] += occ;
        if self.flags.contains(GmmFlags::MEANS) {
            for d in 0..x_stats.len() {
                self.mean_accum[[g, d]] += x_stats[d];
            }
        }
        if self.flags.contains(GmmFlags::VARIANCES) {
            for d in 0..x2_stats.len() {
                self.var_accum[[g, d]] += x2_stats[d];
            }
        }
    }

    /// `self += scale * o` (`AccumDiagGmm::Add`, `mle-diag-gmm.cc:399`).
    pub fn add(&mut self, o: &Self, scale: f64) {
        assert_eq!(
            self.num_gauss(),
            o.num_gauss(),
            "AccumDiagGmm::add: component count mismatch"
        );
        for g in 0..self.num_gauss() {
            self.occupancy[g] += scale * o.occupancy[g];
        }
        if self.flags.contains(GmmFlags::MEANS) {
            self.mean_accum
                .zip_mut_with(&o.mean_accum, |a, b| *a += scale * *b);
        }
        if self.flags.contains(GmmFlags::VARIANCES) {
            self.var_accum
                .zip_mut_with(&o.var_accum, |a, b| *a += scale * *b);
        }
    }

    /// Scale every active accumulator (`AccumDiagGmm::Scale`).
    pub fn scale(&mut self, s: f64) {
        self.occupancy.iter_mut().for_each(|o| *o *= s);
        if self.flags.contains(GmmFlags::MEANS) {
            self.mean_accum *= s;
        }
        if self.flags.contains(GmmFlags::VARIANCES) {
            self.var_accum *= s;
        }
    }

    /// Total soft count over all components.
    pub fn occupancy_sum(&self) -> f64 {
        self.occupancy.iter().sum()
    }

    /// I-smoothing: add `tau` virtual counts drawn from `gmm` itself
    /// (`AccumDiagGmm::SmoothWithModel`, `mle-diag-gmm.cc:240`; Kaldi's
    /// `IsmoothStatsDiagGmm`).
    ///
    /// This pulls the update back towards the current model when a component has
    /// little data. Not valid for updating weights, since it adds counts without
    /// corresponding evidence — Kaldi carries the same caveat.
    pub fn smooth_with_model(&mut self, tau: f64, gmm: &DiagGmm) {
        assert_eq!(
            gmm.num_gauss(),
            self.num_gauss(),
            "smooth_with_model: component count mismatch"
        );
        assert_eq!(
            gmm.dim(),
            self.dim(),
            "smooth_with_model: dimension mismatch"
        );
        let means = gmm.means();
        let vars = gmm.vars();
        for g in 0..self.num_gauss() {
            for d in 0..self.dim() {
                let m = means[[g, d]] as f64;
                let v = vars[[g, d]] as f64;
                self.mean_accum[[g, d]] += tau * m;
                // Second-order stats are var + mean^2.
                self.var_accum[[g, d]] += tau * (v + m * m);
            }
            self.occupancy[g] += tau;
        }
    }

    /// Scale stats so each component has `tau` extra counts of its own average
    /// (`AccumDiagGmm::SmoothStats`, `mle-diag-gmm.cc:211`).
    pub fn smooth_stats(&mut self, tau: f64) {
        for g in 0..self.num_gauss() {
            let occ = self.occupancy[g];
            if occ == 0.0 {
                continue;
            }
            // (tau + occ) / occ
            let s = tau / occ + 1.0;
            for d in 0..self.dim() {
                self.mean_accum[[g, d]] *= s;
                self.var_accum[[g, d]] *= s;
            }
            self.occupancy[g] += tau;
        }
    }
}

/// Per-pdf accumulators plus the running totals used for reporting objective
/// values. Port of `AccumAmDiagGmm` (`gmm/mle-am-diag-gmm.{h,cc}`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccumAmDiagGmm {
    pub accs: Vec<AccumDiagGmm>,
    pub total_frames: f64,
    pub total_loglike: f64,
}

impl AccumAmDiagGmm {
    /// One accumulator per pdf, shaped to that pdf's current Gaussian count
    /// (`AccumAmDiagGmm::Init`, `mle-am-diag-gmm.cc:40`).
    pub fn new(am: &AmDiagGmm, flags: GmmFlags) -> Self {
        let accs = (0..am.num_pdfs())
            .map(|i| AccumDiagGmm::from_gmm(am.pdf(i as PdfId), flags))
            .collect();
        AccumAmDiagGmm {
            accs,
            total_frames: 0.0,
            total_loglike: 0.0,
        }
    }

    pub fn num_accs(&self) -> usize {
        self.accs.len()
    }

    pub fn dim(&self) -> usize {
        self.accs.first().map(|a| a.dim()).unwrap_or(0)
    }

    /// Accumulate frame `x` against pdf `pdf` with frame weight `weight`; returns
    /// the frame log-likelihood (`AccumulateForGmm`, `mle-am-diag-gmm.cc:68`).
    ///
    /// Note the totals use `log_like * weight` while the return value is
    /// unweighted, exactly as Kaldi does.
    pub fn accumulate_for_gmm(
        &mut self,
        am: &AmDiagGmm,
        pdf: PdfId,
        x: &[f32],
        weight: f32,
    ) -> f32 {
        assert!(
            (pdf as usize) < self.accs.len(),
            "AccumAmDiagGmm::accumulate_for_gmm: pdf out of range"
        );
        let log_like = self.accs[pdf as usize].accumulate_from_diag(am.pdf(pdf), x, weight);
        self.total_loglike += log_like as f64 * weight as f64;
        self.total_frames += weight as f64;
        log_like
    }

    /// Posteriors from one feature stream, stats from another
    /// (`AccumulateForGmmTwofeats`, `mle-am-diag-gmm.cc:81`).
    ///
    /// This is `gmm-acc-stats-twofeats`, used to build the speaker-independent
    /// alignment model in the SAT stage: posteriors come from the fMLLR-adapted
    /// features, the stats from the unadapted ones.
    pub fn accumulate_for_gmm_twofeats(
        &mut self,
        am: &AmDiagGmm,
        pdf: PdfId,
        x_posterior: &[f32],
        x_stats: &[f32],
        weight: f32,
    ) -> f32 {
        assert!(
            (pdf as usize) < self.accs.len(),
            "AccumAmDiagGmm::accumulate_for_gmm_twofeats: pdf out of range"
        );
        let gmm = am.pdf(pdf);
        let mut posteriors = Vec::new();
        let log_like = gmm.component_posteriors(x_posterior, &mut posteriors);
        for p in posteriors.iter_mut() {
            *p *= weight;
        }
        self.accs[pdf as usize].accumulate_from_posteriors(x_stats, &posteriors);
        self.total_loglike += log_like as f64 * weight as f64;
        self.total_frames += weight as f64;
        log_like
    }

    /// Accumulate with externally supplied posteriors
    /// (`AccumAmDiagGmm::AccumulateFromPosteriors`, `mle-am-diag-gmm.cc:100`).
    pub fn accumulate_from_posteriors(&mut self, pdf: PdfId, x: &[f32], posteriors: &[f32]) {
        assert!(
            (pdf as usize) < self.accs.len(),
            "AccumAmDiagGmm::accumulate_from_posteriors: pdf out of range"
        );
        self.accs[pdf as usize].accumulate_from_posteriors(x, posteriors);
        self.total_frames += posteriors.iter().map(|&p| p as f64).sum::<f64>();
    }

    /// `self += scale * o`, including the running totals.
    pub fn add(&mut self, o: &Self, scale: f64) {
        assert_eq!(
            self.accs.len(),
            o.accs.len(),
            "AccumAmDiagGmm::add: pdf count mismatch"
        );
        for (a, b) in self.accs.iter_mut().zip(o.accs.iter()) {
            a.add(b, scale);
        }
        self.total_frames += scale * o.total_frames;
        self.total_loglike += scale * o.total_loglike;
    }

    /// Scale every accumulator and the running totals.
    pub fn scale(&mut self, s: f64) {
        for a in self.accs.iter_mut() {
            a.scale(s);
        }
        self.total_frames *= s;
        self.total_loglike *= s;
    }

    /// Total occupancy per pdf, the `state_occs` input to `split_by_count` and
    /// `merge_by_count` (`AccumAmDiagGmm::GetStateOccupancies`).
    pub fn pdf_occupancies(&self) -> Vec<f64> {
        self.accs.iter().map(|a| a.occupancy_sum()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn gmm() -> DiagGmm {
        let mut g = DiagGmm::new(2, 2);
        g.set_means_and_vars(
            &array![[0.0f32, 0.0], [4.0, 4.0]],
            &array![[1.0f32, 1.0], [1.0, 1.0]],
        );
        g.weights = vec![0.5, 0.5];
        g.compute_gconsts();
        g
    }

    #[test]
    fn flags_augment_like_kaldi() {
        assert!(GmmFlags::VARIANCES.augment().contains(GmmFlags::MEANS));
        assert!(GmmFlags::VARIANCES.augment().contains(GmmFlags::WEIGHTS));
        assert!(GmmFlags::MEANS.augment().contains(GmmFlags::WEIGHTS));
        assert_eq!(GmmFlags::ALL.augment(), GmmFlags::ALL);
        // Empty flags still gain weights.
        assert_eq!(GmmFlags(0).augment(), GmmFlags::WEIGHTS);
    }

    #[test]
    fn flags_display() {
        assert_eq!(GmmFlags::ALL.to_string(), "mvw");
        assert_eq!(GmmFlags::MEANS.to_string(), "m");
    }

    #[test]
    fn accumulate_from_posteriors_sums_moments() {
        let mut acc = AccumDiagGmm::new(2, 2, GmmFlags::ALL);
        acc.accumulate_from_posteriors(&[1.0, 2.0], &[0.25, 0.75]);
        acc.accumulate_from_posteriors(&[3.0, 4.0], &[0.5, 0.5]);
        assert!((acc.occupancy[0] - 0.75).abs() < 1e-12);
        assert!((acc.occupancy[1] - 1.25).abs() < 1e-12);
        // First-order: 0.25*1 + 0.5*3 = 1.75
        assert!((acc.mean_accum[[0, 0]] - 1.75).abs() < 1e-12);
        // Second-order: 0.25*1 + 0.5*9 = 4.75
        assert!((acc.var_accum[[0, 0]] - 4.75).abs() < 1e-12);
        assert!((acc.occupancy_sum() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn means_only_flags_skip_variance_storage() {
        let acc = AccumDiagGmm::new(3, 4, GmmFlags::MEANS);
        assert_eq!(acc.mean_accum.dim(), (3, 4));
        assert_eq!(acc.var_accum.dim(), (0, 0));
        assert_eq!(acc.dim(), 4);
    }

    #[test]
    fn accumulate_from_diag_returns_loglike_and_weights_posteriors() {
        let g = gmm();
        let mut acc = AccumDiagGmm::new(2, 2, GmmFlags::ALL);
        let x = [0.0f32, 0.0];
        let ll = acc.accumulate_from_diag(&g, &x, 2.0);
        assert!((ll - g.log_likelihood(&x)).abs() < 1e-5);
        // Weight 2.0 means the total occupancy is 2, not 1.
        assert!((acc.occupancy_sum() - 2.0).abs() < 1e-4);
        // x sits on component 0's mean, so it takes nearly all the mass.
        assert!(acc.occupancy[0] > acc.occupancy[1]);
    }

    #[test]
    fn add_and_scale_are_linear() {
        let mut a = AccumDiagGmm::new(2, 2, GmmFlags::ALL);
        a.accumulate_from_posteriors(&[1.0, 1.0], &[1.0, 0.0]);
        let b = a.clone();
        a.add(&b, 2.0);
        assert!((a.occupancy[0] - 3.0).abs() < 1e-12);
        assert!((a.mean_accum[[0, 0]] - 3.0).abs() < 1e-12);
        a.scale(0.5);
        assert!((a.occupancy[0] - 1.5).abs() < 1e-12);
        assert!((a.mean_accum[[0, 0]] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn set_zero_clears_selected_stats() {
        let mut a = AccumDiagGmm::new(2, 2, GmmFlags::ALL);
        a.accumulate_from_posteriors(&[1.0, 1.0], &[1.0, 1.0]);
        a.set_zero(GmmFlags::VARIANCES);
        assert_eq!(a.var_accum[[0, 0]], 0.0);
        // Occupancy and means survive.
        assert!(a.occupancy[0] > 0.0);
        assert!(a.mean_accum[[0, 0]] > 0.0);
    }

    #[test]
    fn smooth_with_model_adds_tau_counts_matching_the_model() {
        let g = gmm();
        let mut acc = AccumDiagGmm::new(2, 2, GmmFlags::ALL);
        acc.smooth_with_model(10.0, &g);
        // With zero real data, the smoothed stats reproduce the model exactly.
        for c in 0..2 {
            assert!((acc.occupancy[c] - 10.0).abs() < 1e-9);
            let mean = acc.mean_accum[[c, 0]] / acc.occupancy[c];
            let var = acc.var_accum[[c, 0]] / acc.occupancy[c] - mean * mean;
            let want_mean = g.means()[[c, 0]] as f64;
            assert!(
                (mean - want_mean).abs() < 1e-5,
                "mean {mean} vs {want_mean}"
            );
            assert!((var - 1.0).abs() < 1e-4, "var {var}");
        }
    }

    #[test]
    fn smooth_stats_scales_to_tau_plus_occ() {
        let mut acc = AccumDiagGmm::new(1, 1, GmmFlags::ALL);
        acc.accumulate_from_posteriors(&[2.0], &[4.0]); // occ 4, mean stat 8
        acc.smooth_stats(4.0);
        assert!((acc.occupancy[0] - 8.0).abs() < 1e-12);
        // The mean is unchanged: 16/8 == 8/4 == 2.
        assert!((acc.mean_accum[[0, 0]] / acc.occupancy[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn am_accum_tracks_totals_and_pdf_occupancies() {
        let am = AmDiagGmm::init(&gmm(), 3);
        let mut acc = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        assert_eq!(acc.num_accs(), 3);
        acc.accumulate_for_gmm(&am, 0, &[0.0, 0.0], 1.0);
        acc.accumulate_for_gmm(&am, 2, &[4.0, 4.0], 1.0);
        let occs = acc.pdf_occupancies();
        assert!((occs[0] - 1.0).abs() < 1e-4);
        assert!((occs[1]).abs() < 1e-12);
        assert!((occs[2] - 1.0).abs() < 1e-4);
        assert!((acc.total_frames - 2.0).abs() < 1e-12);
        assert!(acc.total_loglike < 0.0);
    }

    #[test]
    fn am_accum_add_is_linear() {
        let am = AmDiagGmm::init(&gmm(), 2);
        let mut a = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        a.accumulate_for_gmm(&am, 0, &[0.0, 0.0], 1.0);
        let b = a.clone();
        a.add(&b, 1.0);
        assert!((a.total_frames - 2.0).abs() < 1e-12);
        assert!((a.pdf_occupancies()[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn twofeats_takes_posteriors_from_first_stream_stats_from_second() {
        let am = AmDiagGmm::init(&gmm(), 1);
        let mut acc = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        // Posteriors from a point on component 0; stats from a different vector.
        acc.accumulate_for_gmm_twofeats(&am, 0, &[0.0, 0.0], &[7.0, 7.0], 1.0);
        let a = &acc.accs[0];
        // Nearly all mass on component 0, and the recorded first-order stat is 7,
        // proving the stats came from the second stream.
        assert!(a.occupancy[0] > 0.99);
        assert!((a.mean_accum[[0, 0]] / a.occupancy[0] - 7.0).abs() < 1e-3);
    }
}
