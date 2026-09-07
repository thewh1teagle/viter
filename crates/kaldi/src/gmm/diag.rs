//! `DiagGmm`: a diagonal-covariance Gaussian mixture. Port of `gmm/diag-gmm.{h,cc}`.

use ndarray::{Array2, s};
use serde::{Deserialize, Serialize};

use super::{M_LOG_2PI, apply_soft_max, log_sum_exp, rand_gauss};

/// A diagonal-covariance GMM in Kaldi's "exponential" storage.
///
/// `means_invvars[g][d] = mean[g][d] * inv_vars[g][d]` and `inv_vars[g][d] = 1/var`.
/// `gconsts[g] = log(w_g) - 0.5*d*log(2*pi) + 0.5*sum_d log(invvar) - 0.5*sum_d mean^2*invvar`,
/// so `loglike(x, g) = gconsts[g] + <means_invvars[g], x> - 0.5 * <inv_vars[g], x^2>`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiagGmm {
    /// Mixture weights (not log), `[g]`.
    pub weights: Vec<f32>,
    /// `mean * invvar`, `[g, d]`.
    pub means_invvars: Array2<f32>,
    /// `1 / var`, `[g, d]`.
    pub inv_vars: Array2<f32>,
    /// Per-component constant including the log weight, `[g]`.
    pub gconsts: Vec<f32>,
}

impl DiagGmm {
    /// `num_gauss` components of dimension `dim`, with unit variances, zero means
    /// and uniform weights. Mirrors `DiagGmm::Resize` followed by a uniform
    /// `SetWeights`; `inv_vars` starts at 1 exactly as Kaldi does so that a later
    /// `set_means_and_vars` behaves.
    pub fn new(num_gauss: usize, dim: usize) -> Self {
        assert!(
            num_gauss > 0 && dim > 0,
            "DiagGmm::new needs positive sizes"
        );
        let mut g = DiagGmm {
            weights: vec![1.0 / num_gauss as f32; num_gauss],
            means_invvars: Array2::zeros((num_gauss, dim)),
            inv_vars: Array2::ones((num_gauss, dim)),
            gconsts: vec![0.0; num_gauss],
        };
        g.compute_gconsts();
        g
    }

    /// A single-component GMM with the given mean and variance, weight 1.
    ///
    /// This is the shape used by `gmm-init-mono` (global mean/var over the first
    /// utterances) and by `DiagGmm(GaussClusterable, var_floor)`.
    pub fn from_single_gaussian(mean: &[f32], var: &[f32]) -> Self {
        assert_eq!(mean.len(), var.len(), "mean/var dimension mismatch");
        let dim = mean.len();
        assert!(dim > 0, "DiagGmm::from_single_gaussian needs dim > 0");
        let mut inv_vars = Array2::zeros((1, dim));
        let mut means_invvars = Array2::zeros((1, dim));
        for d in 0..dim {
            assert!(var[d] > 0.0, "variance must be positive, got {}", var[d]);
            let iv = 1.0 / var[d];
            inv_vars[[0, d]] = iv;
            means_invvars[[0, d]] = mean[d] * iv;
        }
        let mut g = DiagGmm {
            weights: vec![1.0],
            means_invvars,
            inv_vars,
            gconsts: vec![0.0],
        };
        g.compute_gconsts();
        g
    }

    pub fn num_gauss(&self) -> usize {
        self.weights.len()
    }

    pub fn dim(&self) -> usize {
        self.inv_vars.ncols()
    }

    /// Recompute `gconsts`; returns the number of "bad" (infinite) gconsts.
    ///
    /// `gmm/diag-gmm.cc:114`. A zero weight legitimately yields `-inf`; a positive
    /// infinity is flipped to negative infinity so the eventual likelihood is `-inf`
    /// rather than NaN. NaN is a hard error in Kaldi and panics here.
    pub fn compute_gconsts(&mut self) -> usize {
        let num_mix = self.num_gauss();
        let dim = self.dim();
        let offset = -0.5 * M_LOG_2PI * dim as f64; // constant term in gconst.
        let mut num_bad = 0usize;

        // Resize if Gaussians have been removed during update.
        if self.gconsts.len() != num_mix {
            self.gconsts.resize(num_mix, 0.0);
        }

        for mix in 0..num_mix {
            assert!(
                self.weights[mix] >= 0.0,
                "DiagGmm::compute_gconsts: negative weight at component {mix}"
            );
            // May be -inf if weight == 0.
            let mut gc = (self.weights[mix] as f64).ln() + offset;
            for d in 0..dim {
                let mi = self.means_invvars[[mix, d]] as f64;
                let iv = self.inv_vars[[mix, d]] as f64;
                gc += 0.5 * iv.ln() - 0.5 * mi * mi / iv;
            }
            // Sign of the logdet is flipped because the variance is inverted, and
            // means_invvars^2 / inv_vars is mean^2 * invvar. So gc is the
            // log-likelihood at a zero feature value.
            assert!(
                !gc.is_nan(),
                "DiagGmm::compute_gconsts: NaN at component {mix}"
            );
            let mut gc = gc as f32;
            if gc.is_infinite() {
                num_bad += 1;
                // If positive infinity, make it negative infinity, so that the
                // answer becomes -inf in the end rather than NaN.
                if gc > 0.0 {
                    gc = -gc;
                }
            }
            self.gconsts[mix] = gc;
        }
        num_bad
    }

    /// Set means and variances together, then recompute gconsts.
    ///
    /// Kaldi's `SetInvVarsAndMeans` takes inverse variances; this contract takes
    /// plain variances, so we invert here. Setting both at once avoids the
    /// numerical asymmetry of `SetMeans` after `SetInvVars`.
    pub fn set_means_and_vars(&mut self, means: &Array2<f32>, vars: &Array2<f32>) {
        assert_eq!(
            means.dim(),
            (self.num_gauss(), self.dim()),
            "means shape mismatch"
        );
        assert_eq!(
            vars.dim(),
            (self.num_gauss(), self.dim()),
            "vars shape mismatch"
        );
        for g in 0..self.num_gauss() {
            for d in 0..self.dim() {
                let v = vars[[g, d]];
                assert!(v > 0.0, "variance must be positive at ({g},{d}), got {v}");
                let iv = 1.0 / v;
                self.inv_vars[[g, d]] = iv;
                self.means_invvars[[g, d]] = means[[g, d]] * iv;
            }
        }
        self.compute_gconsts();
    }

    /// Means in the natural parameterisation, `means_invvars / inv_vars`
    /// (`GetMeans`, `diag-gmm-inl.h:123`).
    pub fn means(&self) -> Array2<f32> {
        let mut m = self.means_invvars.clone();
        m.zip_mut_with(&self.inv_vars, |mi, iv| *mi /= *iv);
        m
    }

    /// Variances, `1 / inv_vars` (`GetVars`, `diag-gmm-inl.h:115`).
    pub fn vars(&self) -> Array2<f32> {
        self.inv_vars.mapv(|v| 1.0 / v)
    }

    /// Total log-likelihood of `x` under the mixture (`diag-gmm.cc:517`).
    pub fn log_likelihood(&self, x: &[f32]) -> f32 {
        let mut loglikes = Vec::new();
        self.component_log_likes(x, &mut loglikes);
        let log_sum = log_sum_exp(&loglikes);
        assert!(
            !log_sum.is_nan(),
            "DiagGmm::log_likelihood: invalid answer (overflow or invalid variances/features?)"
        );
        log_sum
    }

    /// Per-component log-likelihoods including the log weight (`diag-gmm.cc:528`).
    ///
    /// `out` is resized to `num_gauss()`.
    pub fn component_log_likes(&self, x: &[f32], out: &mut Vec<f32>) {
        assert_eq!(
            x.len(),
            self.dim(),
            "DiagGmm::component_log_likes: dimension mismatch"
        );
        out.clear();
        out.extend_from_slice(&self.gconsts);
        for g in 0..self.num_gauss() {
            let mi = self.means_invvars.row(g);
            let iv = self.inv_vars.row(g);
            // loglikes += means_invvars . x  -  0.5 * inv_vars . x^2
            let mut acc = 0.0f32;
            for d in 0..x.len() {
                let xd = x[d];
                acc += mi[d] * xd - 0.5 * iv[d] * xd * xd;
            }
            out[g] += acc;
        }
    }

    /// Posteriors of each component given `x`; returns the total log-likelihood.
    ///
    /// `diag-gmm.cc:601`: log-likelihoods are softmaxed in place, so `out` is the
    /// posterior vector and the return value is the log-sum-exp before normalising.
    pub fn component_posteriors(&self, x: &[f32], out: &mut Vec<f32>) -> f32 {
        self.component_log_likes(x, out);
        let log_sum = apply_soft_max(out);
        assert!(
            !log_sum.is_nan() && !log_sum.is_infinite(),
            "DiagGmm::component_posteriors: invalid answer (overflow or invalid variances/features?)"
        );
        log_sum
    }

    /// Split components until there are `target` of them (`diag-gmm.cc:154`).
    ///
    /// Repeatedly takes the heaviest component, halves its weight, copies it, and
    /// pushes the two copies apart along a random direction scaled by
    /// `perturb_factor`. The perturbation is applied to `means_invvars` using
    /// `sqrt(inv_var)` — which looks wrong but is right, because `means_invvars`
    /// carries the units of an inverse standard deviation.
    pub fn split(&mut self, target: usize, perturb_factor: f32, rng: &mut impl rand::Rng) {
        let current = self.num_gauss();
        assert!(
            target >= current && current != 0,
            "DiagGmm::split: cannot split from {current} to {target} components"
        );
        if target == current {
            tracing::warn!(
                target,
                "already have the target # of Gaussians; doing nothing"
            );
            return;
        }

        let dim = self.dim();
        // Resize, keeping the existing components in the leading rows.
        self.weights.resize(target, 0.0);
        self.gconsts.resize(target, 0.0);
        let mut new_mi = Array2::zeros((target, dim));
        new_mi
            .slice_mut(s![..current, ..])
            .assign(&self.means_invvars);
        self.means_invvars = new_mi;
        let mut new_iv = Array2::zeros((target, dim));
        new_iv.slice_mut(s![..current, ..]).assign(&self.inv_vars);
        self.inv_vars = new_iv;

        let mut current_components = current;
        let mut rand_vec = vec![0.0f32; dim];
        while current_components < target {
            // Find the heaviest component.
            let mut max_weight = self.weights[0];
            let mut max_idx = 0usize;
            for i in 1..current_components {
                if self.weights[i] > max_weight {
                    max_weight = self.weights[i];
                    max_idx = i;
                }
            }

            self.weights[max_idx] /= 2.0;
            self.weights[current_components] = self.weights[max_idx];

            for (i, r) in rand_vec.iter_mut().enumerate() {
                *r = rand_gauss(rng) * self.inv_vars[[max_idx, i]].sqrt();
            }

            let new = current_components;
            for d in 0..dim {
                self.inv_vars[[new, d]] = self.inv_vars[[max_idx, d]];
                self.means_invvars[[new, d]] =
                    self.means_invvars[[max_idx, d]] + perturb_factor * rand_vec[d];
                self.means_invvars[[max_idx, d]] -= perturb_factor * rand_vec[d];
            }
            current_components += 1;
        }
        self.compute_gconsts();
    }

    /// Keep only the listed components (ascending, unique), renormalising weights.
    ///
    /// This is `RemoveComponents(.., renorm_weights = true)` (`diag-gmm.cc:632`)
    /// expressed as a retain, which avoids the repeated index fixups Kaldi needs.
    pub(crate) fn retain_components(&mut self, keep: &[usize]) {
        assert!(!keep.is_empty(), "cannot remove every component");
        if keep.len() == self.num_gauss() {
            return;
        }
        let dim = self.dim();
        let mut weights = Vec::with_capacity(keep.len());
        let mut means_invvars = Array2::zeros((keep.len(), dim));
        let mut inv_vars = Array2::zeros((keep.len(), dim));
        for (new_i, &old_i) in keep.iter().enumerate() {
            weights.push(self.weights[old_i]);
            means_invvars
                .row_mut(new_i)
                .assign(&self.means_invvars.row(old_i));
            inv_vars.row_mut(new_i).assign(&self.inv_vars.row(old_i));
        }
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
        self.weights = weights;
        self.means_invvars = means_invvars;
        self.inv_vars = inv_vars;
        self.gconsts.truncate(keep.len());
    }

    /// Scale all mixture weights by `factor` and recompute gconsts.
    ///
    /// Used by silence boosting; note this deliberately does *not* renormalise, so
    /// the weights no longer sum to one — exactly what Kaldi's `boost_silence` does.
    pub fn scale_weights(&mut self, factor: f32) {
        for w in self.weights.iter_mut() {
            *w *= factor;
        }
        self.compute_gconsts();
    }
}

impl DiagGmm {
    /// Sum of the weights, used in tests and by weight renormalisation checks.
    pub(crate) fn weight_sum(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// Mean of component `g` as an owned vector.
    pub(crate) fn component_mean(&self, g: usize) -> Vec<f32> {
        (0..self.dim())
            .map(|d| self.means_invvars[[g, d]] / self.inv_vars[[g, d]])
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::SeedableRng;

    fn two_component() -> DiagGmm {
        let mut g = DiagGmm::new(2, 3);
        let means = array![[0.0f32, 1.0, 2.0], [3.0, -1.0, 0.5]];
        let vars = array![[1.0f32, 2.0, 0.5], [0.25, 1.0, 4.0]];
        g.set_means_and_vars(&means, &vars);
        g.weights = vec![0.3, 0.7];
        g.compute_gconsts();
        g
    }

    /// Reference log-likelihood computed straight from the textbook formula.
    fn naive_loglike(g: &DiagGmm, x: &[f32]) -> f64 {
        let means = g.means();
        let vars = g.vars();
        let mut total = 0.0f64;
        for i in 0..g.num_gauss() {
            let mut lp = (g.weights[i] as f64).ln();
            for d in 0..g.dim() {
                let v = vars[[i, d]] as f64;
                let diff = x[d] as f64 - means[[i, d]] as f64;
                lp += -0.5 * (2.0 * std::f64::consts::PI * v).ln() - 0.5 * diff * diff / v;
            }
            total += lp.exp();
        }
        total.ln()
    }

    #[test]
    fn roundtrip_means_and_vars() {
        let g = two_component();
        let means = g.means();
        let vars = g.vars();
        assert!((means[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((means[[1, 0]] - 3.0).abs() < 1e-6);
        assert!((vars[[0, 2]] - 0.5).abs() < 1e-6);
        assert!((vars[[1, 2]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn loglike_matches_textbook_formula() {
        let g = two_component();
        for x in [
            vec![0.0f32, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            vec![-2.5, 0.75, 1.25],
        ] {
            let got = g.log_likelihood(&x) as f64;
            let want = naive_loglike(&g, &x);
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn gconst_is_loglike_at_zero() {
        // By construction gconst[g] is the component log-likelihood at x = 0.
        let g = two_component();
        let mut lls = Vec::new();
        g.component_log_likes(&[0.0, 0.0, 0.0], &mut lls);
        for i in 0..g.num_gauss() {
            assert!((lls[i] - g.gconsts[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_weight_gives_bad_gconst() {
        let mut g = two_component();
        g.weights[0] = 0.0;
        let num_bad = g.compute_gconsts();
        assert_eq!(num_bad, 1);
        assert_eq!(g.gconsts[0], f32::NEG_INFINITY);
        // The mixture still scores finitely thanks to the surviving component.
        assert!(g.log_likelihood(&[0.0, 0.0, 0.0]).is_finite());
    }

    #[test]
    fn posteriors_sum_to_one_and_return_loglike() {
        let g = two_component();
        let x = [0.5f32, 0.25, -1.0];
        let mut post = Vec::new();
        let ll = g.component_posteriors(&x, &mut post);
        assert_eq!(post.len(), 2);
        assert!((post.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((ll - g.log_likelihood(&x)).abs() < 1e-5);
    }

    #[test]
    fn single_gaussian_posterior_is_one() {
        let g = DiagGmm::from_single_gaussian(&[1.0, 2.0], &[1.0, 3.0]);
        let mut post = Vec::new();
        g.component_posteriors(&[0.0, 0.0], &mut post);
        assert!((post[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn split_doubles_components_and_preserves_weight_mass() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let mut g = two_component();
        let before = g.weight_sum();
        g.split(4, 0.01, &mut rng);
        assert_eq!(g.num_gauss(), 4);
        assert!((g.weight_sum() - before).abs() < 1e-5);
        // Variances are copied unchanged by a split.
        let vars = g.vars();
        assert!(
            (vars[[0, 0]] - vars[[2, 0]]).abs() < 1e-5
                || (vars[[1, 0]] - vars[[2, 0]]).abs() < 1e-5
        );
    }

    #[test]
    fn split_with_zero_perturb_gives_identical_twins() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let mut g = two_component();
        g.split(3, 0.0, &mut rng);
        assert_eq!(g.num_gauss(), 3);
        // The heaviest component (index 1, weight .7) was halved and duplicated.
        assert!((g.weights[1] - 0.35).abs() < 1e-6);
        assert!((g.weights[2] - 0.35).abs() < 1e-6);
        let m1 = g.component_mean(1);
        let m2 = g.component_mean(2);
        for d in 0..3 {
            assert!((m1[d] - m2[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn split_to_same_size_is_a_noop() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut g = two_component();
        let w = g.weights.clone();
        g.split(2, 0.1, &mut rng);
        assert_eq!(g.weights, w);
    }

    #[test]
    fn scale_weights_shifts_gconsts_by_log_factor() {
        let mut g = two_component();
        let before = g.gconsts.clone();
        g.scale_weights(2.0);
        let shift = 2.0f32.ln();
        for i in 0..g.num_gauss() {
            assert!((g.gconsts[i] - (before[i] + shift)).abs() < 1e-5);
        }
        // Weights are deliberately left unnormalised.
        assert!((g.weight_sum() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn serde_roundtrip() {
        let g = two_component();
        let json = serde_json::to_string(&g).unwrap();
        let back: DiagGmm = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_gauss(), g.num_gauss());
        let x = [0.3f32, -0.2, 1.1];
        assert!((back.log_likelihood(&x) - g.log_likelihood(&x)).abs() < 1e-6);
    }
}
