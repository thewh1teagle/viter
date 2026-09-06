//! Diagonal-covariance GMMs and their maximum-likelihood estimation.
//!
//! Line-by-line port of Kaldi's `gmm/diag-gmm.{h,cc}`, `gmm/am-diag-gmm.{h,cc}`,
//! `gmm/mle-diag-gmm.{h,cc}`, `gmm/mle-am-diag-gmm.{h,cc}` and
//! `gmm/model-common.{h,cc}`, plus `boost_silence` from
//! `kalpy/extensions/gmm/gmm.cpp:203`.
//!
//! Storage follows Kaldi exactly: a `DiagGmm` keeps `means_invvars = mean * invvar`
//! and `inv_vars = 1/var` rather than means and variances, and `gconsts` folds the
//! log weight, the `-0.5*d*log(2*pi)` constant, the log-determinant and the
//! `mean^2/var` term into one per-component scalar so that the log-likelihood of a
//! frame is an affine function of `[x, x^2]`.
//!
//! Accumulators use `f64` like Kaldi's `double` stats; model parameters are `f32`
//! (Kaldi's `BaseFloat`).

mod accum;
mod am;
mod diag;
mod merge;
mod mle;

pub use accum::{AccumAmDiagGmm, AccumDiagGmm, GmmFlags};
pub use am::{AmDiagGmm, PackedGmm};
pub use diag::DiagGmm;
pub use mle::{MleDiagGmmOptions, mle_am_diag_gmm_update, mle_diag_gmm_update};

/// `log(2*pi)`, Kaldi's `M_LOG_2PI` (`base/kaldi-math.h`).
pub(crate) const M_LOG_2PI: f64 = 1.837_877_066_409_345_5;

/// Errors produced by the GMM module.
#[derive(Debug, thiserror::Error)]
pub enum GmmError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("cannot split from {from} to {to} components")]
    BadSplit { from: usize, to: usize },
    #[error("invalid target number of Gaussians ({target}), #Gauss = {num_gauss}")]
    BadMerge { target: usize, num_gauss: usize },
}

/// One sample from the standard normal, matching Kaldi's `RandGauss`
/// (`base/kaldi-math.h:155`): `sqrt(-2 log u1) * cos(2 pi u2)`, the cosine branch
/// of the Box-Muller transform.
///
/// Kaldi draws `u` from `(0, 1)` exclusive via `(Rand()+1)/(RAND_MAX+2)`; we draw
/// from `[0, 1)` and shift zero away so `log(u)` is always finite.
pub(crate) fn rand_gauss<R: rand::Rng + ?Sized>(rng: &mut R) -> f32 {
    use rand::RngExt;
    let mut u1: f64 = rng.random();
    if u1 <= 0.0 {
        u1 = f64::MIN_POSITIVE;
    }
    let u2: f64 = rng.random();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// `log(sum_i exp(x_i))`, computed with the max shifted out.
///
/// Mirrors `VectorBase::LogSumExp` with no pruning: an empty slice gives `-inf`,
/// and an all `-inf` slice gives `-inf` rather than NaN.
pub(crate) fn log_sum_exp(x: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        if v > max {
            max = v;
        }
    }
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if max == f32::INFINITY {
        return f32::INFINITY;
    }
    let max_d = max as f64;
    let mut sum = 0.0f64;
    for &v in x {
        sum += ((v as f64) - max_d).exp();
    }
    (max_d + sum.ln()) as f32
}

/// In-place softmax returning `log(sum_i exp(x_i))` before normalisation.
///
/// This is Kaldi's `VectorBase::ApplySoftMax` (`matrix/kaldi-vector.cc`), used by
/// `DiagGmm::ComponentPosteriors`.
pub(crate) fn apply_soft_max(x: &mut [f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        // All components impossible; leave as-is and report -inf like Kaldi would.
        return f32::NEG_INFINITY;
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        let e = ((*v as f64) - max as f64).exp();
        *v = e as f32;
        sum += e;
    }
    for v in x.iter_mut() {
        *v = ((*v as f64) / sum) as f32;
    }
    (max as f64 + sum.ln()) as f32
}

/// Kaldi's `GetSplitTargets` (`gmm/model-common.cc:117`): allocate
/// `target_components` Gaussians over the pdfs in proportion to
/// `state_occs[i]^power`, with a floor of one Gaussian per pdf and a `min_count`
/// rule that stops a pdf growing once `(num_components+1) * min_count >= occ`.
///
/// Kaldi drives this with a max-heap ordered by `occupancy/(num_components+1e-10)`;
/// we reproduce it with an explicit `BinaryHeap` over the same key. Ties inside the
/// heap may resolve differently than in C++'s `std::priority_queue`, which is
/// unspecified there too.
pub fn get_split_targets(
    state_occs: &[f64],
    target_components: usize,
    power: f32,
    min_count: f32,
) -> Vec<usize> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    /// Kaldi's `CountStats`, ordered by `occupancy / (num_components + 1e-10)`.
    struct CountStats {
        pdf_index: usize,
        num_components: usize,
        occupancy: f64,
    }
    impl CountStats {
        fn key(&self) -> f64 {
            self.occupancy / (self.num_components as f64 + 1.0e-10)
        }
    }
    impl PartialEq for CountStats {
        fn eq(&self, other: &Self) -> bool {
            self.key() == other.key()
        }
    }
    impl Eq for CountStats {}
    impl PartialOrd for CountStats {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for CountStats {
        fn cmp(&self, other: &Self) -> Ordering {
            // BinaryHeap is a max-heap, like std::priority_queue with operator<.
            self.key()
                .partial_cmp(&other.key())
                .unwrap_or(Ordering::Equal)
                // Break ties deterministically; C++ leaves this unspecified.
                .then_with(|| other.pdf_index.cmp(&self.pdf_index))
        }
    }

    let num_pdfs = state_occs.len();
    let mut split_queue: BinaryHeap<CountStats> = BinaryHeap::with_capacity(num_pdfs);
    for (pdf_index, &occ) in state_occs.iter().enumerate() {
        // pow(occ, power); initialize with one Gaussian per PDF, to floor #Gauss at 1.
        let occ = (occ as f32).powf(power) as f64;
        split_queue.push(CountStats {
            pdf_index,
            num_components: 1,
            occupancy: occ,
        });
    }

    let mut num_gauss = num_pdfs;
    while num_gauss < target_components {
        let Some(mut state_to_split) = split_queue.pop() else {
            break;
        };
        if state_to_split.occupancy == 0.0 {
            tracing::warn!(
                target_components,
                min_count,
                "could not split up to target due to min-count (or no counts at all)"
            );
            split_queue.push(state_to_split);
            break;
        }
        let orig_occ = state_occs[state_to_split.pdf_index];
        if (state_to_split.num_components as f64 + 1.0) * min_count as f64 >= orig_occ {
            // min-count active -> disallow splitting this state any more.
            state_to_split.occupancy = 0.0;
        } else {
            state_to_split.num_components += 1;
            num_gauss += 1;
        }
        split_queue.push(state_to_split);
    }

    let mut targets = vec![0usize; num_pdfs];
    for cs in split_queue.into_iter() {
        targets[cs.pdf_index] = cs.num_components;
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::SeedableRng;

    #[test]
    fn log_sum_exp_matches_naive() {
        let x = [-1.0f32, 0.5, 2.25, -3.0];
        let naive = x.iter().map(|&v| (v as f64).exp()).sum::<f64>().ln() as f32;
        assert!((log_sum_exp(&x) - naive).abs() < 1e-5);
    }

    #[test]
    fn log_sum_exp_handles_all_neg_inf() {
        assert_eq!(log_sum_exp(&[f32::NEG_INFINITY; 3]), f32::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn soft_max_normalises_and_returns_logsumexp() {
        let mut x = [1.0f32, 2.0, 3.0];
        let expected = log_sum_exp(&x);
        let got = apply_soft_max(&mut x);
        assert!((got - expected).abs() < 1e-5);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // Ordering is preserved and the largest logit gets the largest posterior.
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn rand_gauss_is_roughly_standard_normal() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let n = 20_000;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for _ in 0..n {
            let g = rand_gauss(&mut rng) as f64;
            assert!(g.is_finite());
            sum += g;
            sumsq += g * g;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn split_targets_floor_of_one_per_pdf() {
        // With target below num_pdfs, every pdf still gets exactly one Gaussian.
        let occs = [10.0, 20.0, 30.0];
        let t = get_split_targets(&occs, 2, 0.25, 0.0);
        assert_eq!(t, vec![1, 1, 1]);
    }

    #[test]
    fn split_targets_sum_to_target_and_favour_high_occupancy() {
        let occs = [1.0, 10.0, 100.0];
        let t = get_split_targets(&occs, 12, 1.0, 0.0);
        assert_eq!(t.iter().sum::<usize>(), 12);
        // Higher occupancy pdfs get at least as many Gaussians.
        assert!(t[2] >= t[1] && t[1] >= t[0]);
    }

    #[test]
    fn split_targets_respect_min_count() {
        // min_count=50 means a pdf with occ=100 can hold at most 1 Gaussian:
        // (1+1)*50 >= 100 blocks the first split.
        let occs = [100.0, 100.0];
        let t = get_split_targets(&occs, 10, 1.0, 50.0);
        assert_eq!(t, vec![1, 1]);
    }

    #[test]
    fn split_targets_power_zero_spreads_evenly() {
        let occs = [1.0, 1000.0, 1_000_000.0];
        let t = get_split_targets(&occs, 6, 0.0, 0.0);
        assert_eq!(t, vec![2, 2, 2]);
    }
}
