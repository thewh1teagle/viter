//! Merging components of a `DiagGmm`.
//!
//! Port of `DiagGmm::Merge` and `DiagGmm::merged_components_logdet`
//! (`gmm/diag-gmm.cc:295` and `:470`). Kept apart from `diag.rs` because this is a
//! self-contained clustering algorithm rather than part of the distribution itself.

use ndarray::{Array2, ArrayView1};

use super::diag::DiagGmm;

impl DiagGmm {
    /// Merge components down to `target` (`diag-gmm.cc:295`).
    ///
    /// `target == 1` collapses to the weighted global mean and variance. Otherwise
    /// this is greedy hierarchical clustering: at each step it merges the pair whose
    /// merge costs the least log-likelihood, where the cost of merging `i` and `j` is
    /// `(w_i+w_j) * merged_logdet - w_i*logdet_i - w_j*logdet_j`.
    pub fn merge(&mut self, target: usize) {
        let num_comp = self.num_gauss();
        assert!(
            target > 0 && num_comp >= target,
            "DiagGmm::merge: invalid target number of Gaussians (={target}), #Gauss = {num_comp}"
        );
        if num_comp == target {
            return; // Nothing to do.
        }
        let dim = self.dim();

        if target == 1 {
            self.merge_to_one();
            return;
        }

        // logdet for each component; +0.5 because the variance is inverted.
        let mut logdet = vec![0.0f32; num_comp];
        for i in 0..num_comp {
            let mut acc = 0.0f64;
            for d in 0..dim {
                acc += 0.5 * (self.inv_vars[[i, d]] as f64).ln();
            }
            logdet[i] = acc as f32;
        }
        let mut discarded = vec![false; num_comp];

        // Undo the inversion: natural means, and second-order stats (var + mean^2),
        // both normalised by the zero-order stats.
        let mut vars = self.vars();
        let mut means = self.means();
        for i in 0..num_comp {
            for d in 0..dim {
                let m = means[[i, d]];
                vars[[i, d]] += m * m;
            }
        }

        // delta_like(i, j) for j < i: the (negative) change in likelihood on merging.
        let mut delta_like = Array2::<f32>::zeros((num_comp, num_comp));
        for i in 0..num_comp {
            for j in 0..i {
                let (w1, w2) = (self.weights[i], self.weights[j]);
                let w_sum = w1 + w2;
                let merged_logdet = merged_components_logdet(
                    w1,
                    w2,
                    means.row(i),
                    means.row(j),
                    vars.row(i),
                    vars.row(j),
                );
                delta_like[[i, j]] = w_sum * merged_logdet - w1 * logdet[i] - w2 * logdet[j];
            }
        }

        // Merge the pairs with the smallest impact on the log-likelihood.
        for _removed in 0..(num_comp - target) {
            // Search for the least significant change (max of the negative deltas).
            let mut max_delta_like = f32::MIN;
            let mut max_i: isize = -1;
            let mut max_j: isize = -1;
            for i in 0..num_comp {
                if discarded[i] {
                    continue;
                }
                for j in 0..i {
                    if discarded[j] {
                        continue;
                    }
                    if delta_like[[i, j]] > max_delta_like {
                        max_delta_like = delta_like[[i, j]];
                        max_i = i as isize;
                        max_j = j as isize;
                    }
                }
            }
            assert!(
                max_i != max_j && max_i != -1 && max_j != -1,
                "DiagGmm::merge: failed to find a pair to merge"
            );
            let (max_i, max_j) = (max_i as usize, max_j as usize);

            // Merge means, vars and weights of max_j into max_i.
            let (w1, w2) = (self.weights[max_i], self.weights[max_j]);
            let w_sum = w1 + w2;
            for d in 0..dim {
                means[[max_i, d]] =
                    (means[[max_i, d]] + (w2 / w1) * means[[max_j, d]]) * (w1 / w_sum);
                vars[[max_i, d]] = (vars[[max_i, d]] + (w2 / w1) * vars[[max_j, d]]) * (w1 / w_sum);
            }
            self.weights[max_i] = w_sum;

            // Update the model for the merged component: centralise the second-order
            // stats, invert, and re-form means_invvars.
            for d in 0..dim {
                let m = means[[max_i, d]];
                let iv = 1.0 / (vars[[max_i, d]] - m * m);
                self.inv_vars[[max_i, d]] = iv;
                self.means_invvars[[max_i, d]] = m * iv;
            }

            // Update logdet for the merged component.
            let mut acc = 0.0f64;
            for d in 0..dim {
                acc += 0.5 * (self.inv_vars[[max_i, d]] as f64).ln();
            }
            logdet[max_i] = acc as f32;

            discarded[max_j] = true;

            // Update delta_like against the merged component. Kaldi writes into a
            // SpMatrix, which silently swaps indices to stay lower-triangular; we do
            // the swap explicitly.
            for j in 0..num_comp {
                if j == max_i || discarded[j] {
                    continue;
                }
                let (w1, w2) = (self.weights[max_i], self.weights[j]);
                let w_sum = w1 + w2;
                let merged_logdet = merged_components_logdet(
                    w1,
                    w2,
                    means.row(max_i),
                    means.row(j),
                    vars.row(max_i),
                    vars.row(j),
                );
                let d = w_sum * merged_logdet - w1 * logdet[max_i] - w2 * logdet[j];
                if max_i > j {
                    delta_like[[max_i, j]] = d;
                } else {
                    delta_like[[j, max_i]] = d;
                }
            }
        }

        // Remove the consumed components.
        let keep: Vec<usize> = (0..num_comp).filter(|&i| !discarded[i]).collect();
        self.retain_components(&keep);
        self.compute_gconsts();
    }

    /// The `target == 1` branch of `Merge`: collapse to a single Gaussian carrying
    /// the weighted global mean and variance.
    fn merge_to_one(&mut self) {
        let num_comp = self.num_gauss();
        let dim = self.dim();
        let weights = self.weights.clone();
        let mut vars = self.vars();
        let means = self.means();
        // Add mean^2 to the variances to get second-order stats.
        for i in 0..num_comp {
            for d in 0..dim {
                let m = means[[i, d]];
                vars[[i, d]] += m * m;
            }
        }

        let mut w0 = 0.0f32;
        let mut mi0 = vec![0.0f32; dim];
        let mut iv0 = vec![0.0f32; dim];
        for i in 0..num_comp {
            w0 += weights[i];
            for d in 0..dim {
                mi0[d] += weights[i] * means[[i, d]];
                iv0[d] += weights[i] * vars[[i, d]];
            }
        }
        if (w0 - 1.0).abs() > 1e-6 * (1.0f32).max(w0.abs()) {
            tracing::warn!(
                sum = w0,
                "DiagGmm::merge: weights do not sum to 1; rescaling"
            );
            for d in 0..dim {
                mi0[d] *= w0;
                iv0[d] *= w0;
            }
            w0 = 1.0;
        }

        self.weights = vec![w0];
        self.gconsts = vec![0.0];
        self.means_invvars = Array2::zeros((1, dim));
        self.inv_vars = Array2::zeros((1, dim));
        for d in 0..dim {
            // Centralise, invert, then re-form means_invvars.
            let iv = 1.0 / (iv0[d] - mi0[d] * mi0[d]);
            self.inv_vars[[0, d]] = iv;
            self.means_invvars[[0, d]] = mi0[d] * iv;
        }
        self.compute_gconsts();
    }
}

/// Log-determinant of the Gaussian formed by merging two components
/// (`DiagGmm::merged_components_logdet`, `diag-gmm.cc:470`).
///
/// `f1`/`f2` are first-order stats (means) and `s1`/`s2` second-order stats, both
/// normalised by the zero-order stats.
fn merged_components_logdet(
    w1: f32,
    w2: f32,
    f1: ArrayView1<f32>,
    f2: ArrayView1<f32>,
    s1: ArrayView1<f32>,
    s2: ArrayView1<f32>,
) -> f32 {
    let dim = f1.len();
    let w_sum = w1 + w2;
    let mut merged_logdet = 0.0f64;
    for d in 0..dim {
        let tmp_mean = (f1[d] + (w2 / w1) * f2[d]) * (w1 / w_sum);
        let tmp_var = (s1[d] + (w2 / w1) * s2[d]) * (w1 / w_sum) - tmp_mean * tmp_mean;
        // -0.5 because the variance is not inverted here.
        merged_logdet -= 0.5 * (tmp_var as f64).ln();
    }
    merged_logdet as f32
}

#[cfg(test)]
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

    /// Weight-normalised mean over all components, the quantity a merge to one
    /// component must preserve.
    fn global_mean(g: &DiagGmm) -> Vec<f32> {
        let means = g.means();
        let mut out = vec![0.0f32; g.dim()];
        for i in 0..g.num_gauss() {
            for d in 0..g.dim() {
                out[d] += g.weights[i] * means[[i, d]];
            }
        }
        let wsum: f32 = g.weight_sum();
        for v in out.iter_mut() {
            *v /= wsum;
        }
        out
    }

    #[test]
    fn merge_to_one_preserves_global_mean_and_variance() {
        let g = two_component();
        let want_mean = global_mean(&g);
        // Global second moment, weights already sum to 1.
        let means = g.means();
        let vars = g.vars();
        let mut want_second = vec![0.0f32; g.dim()];
        for i in 0..g.num_gauss() {
            for d in 0..g.dim() {
                let m = means[[i, d]];
                want_second[d] += g.weights[i] * (vars[[i, d]] + m * m);
            }
        }

        let mut merged = g.clone();
        merged.merge(1);
        assert_eq!(merged.num_gauss(), 1);
        let got_mean = merged.component_mean(0);
        let got_var = merged.vars();
        for d in 0..3 {
            assert!((got_mean[d] - want_mean[d]).abs() < 1e-4, "mean dim {d}");
            let want_var = want_second[d] - want_mean[d] * want_mean[d];
            assert!((got_var[[0, d]] - want_var).abs() < 1e-3, "var dim {d}");
        }
    }

    #[test]
    fn merge_reduces_count_and_conserves_weight() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut g = two_component();
        g.split(8, 0.05, &mut rng);
        assert_eq!(g.num_gauss(), 8);
        let before = g.weight_sum();
        g.merge(3);
        assert_eq!(g.num_gauss(), 3);
        // retain_components renormalises, so the sum is 1 after merging.
        assert!((g.weight_sum() - 1.0).abs() < 1e-4);
        assert!((before - 1.0).abs() < 1e-4);
    }

    #[test]
    fn merge_prefers_merging_near_identical_components() {
        // Three components: 0 and 1 are nearly identical, 2 is far away.
        // Merging down to 2 should fuse 0 and 1 and leave 2 alone.
        let mut g = DiagGmm::new(3, 1);
        let means = array![[0.0f32], [0.001], [50.0]];
        let vars = array![[1.0f32], [1.0], [1.0]];
        g.set_means_and_vars(&means, &vars);
        g.weights = vec![1.0 / 3.0; 3];
        g.compute_gconsts();
        g.merge(2);
        assert_eq!(g.num_gauss(), 2);
        let mut ms: Vec<f32> = (0..2).map(|i| g.component_mean(i)[0]).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(ms[0].abs() < 0.1, "fused component near 0, got {}", ms[0]);
        assert!(
            (ms[1] - 50.0).abs() < 0.1,
            "isolated component, got {}",
            ms[1]
        );
    }

    #[test]
    fn merge_to_same_size_is_a_noop() {
        let mut g = two_component();
        let w = g.weights.clone();
        g.merge(2);
        assert_eq!(g.weights, w);
    }
}
