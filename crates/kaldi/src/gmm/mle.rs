//! Maximum-likelihood re-estimation.
//!
//! Port of `MleDiagGmmUpdate` (`gmm/mle-diag-gmm.cc:275`) and `MleAmDiagGmmUpdate`
//! (`gmm/mle-am-diag-gmm.cc:187`), including variance flooring, the weight and
//! occupancy floors, and low-count Gaussian removal.

use super::accum::{AccumAmDiagGmm, AccumDiagGmm, GmmFlags};
use super::am::AmDiagGmm;
use super::diag::DiagGmm;
use crate::types::PdfId;

/// Options for ML re-estimation (`MleDiagGmmOptions`, `gmm/mle-diag-gmm.h`).
///
/// Defaults match Kaldi's. Note MFA overrides `min_gaussian_occupancy` to 3.0 for
/// the first monophone iteration, and sets `remove_low_count_gaussians = false`
/// when estimating the SAT alignment model.
#[derive(Clone, Debug)]
pub struct MleDiagGmmOptions {
    /// Minimum weight below which a Gaussian is considered starved.
    pub min_gaussian_weight: f64,
    /// Minimum soft count below which a Gaussian is considered starved.
    pub min_gaussian_occupancy: f64,
    /// Floor applied to every variance element.
    pub min_variance: f64,
    /// Whether starved Gaussians are deleted rather than merely floored.
    pub remove_low_count_gaussians: bool,
    /// Optional per-dimension variance floor; takes precedence over `min_variance`
    /// when non-empty (Kaldi's `variance_floor_vector`).
    pub variance_floor_vector: Vec<f64>,
}

impl Default for MleDiagGmmOptions {
    fn default() -> Self {
        MleDiagGmmOptions {
            min_gaussian_weight: 1.0e-5,
            min_gaussian_occupancy: 10.0,
            min_variance: 0.001,
            remove_low_count_gaussians: true,
            variance_floor_vector: Vec::new(),
        }
    }
}

/// Detailed counters from one update, for logging.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    pub objf_change: f64,
    pub count: f64,
    pub floored_elements: usize,
    pub floored_gaussians: usize,
    pub removed_gaussians: usize,
}

/// The ML auxiliary function value of `gmm` given `acc`
/// (`MlObjective`, `mle-diag-gmm.cc:261`).
///
/// Because `gconsts` already contains the log weight and the normalising terms,
/// the objective is affine in the accumulated stats:
/// `sum_g occ_g*gconst_g + <mean_accum, means_invvars> - 0.5*<var_accum, inv_vars>`.
fn ml_objective(gmm: &DiagGmm, acc: &AccumDiagGmm) -> f64 {
    let mut obj = 0.0f64;
    for g in 0..gmm.num_gauss() {
        obj += acc.occupancy[g] * gmm.gconsts[g] as f64;
    }
    if acc.flags.contains(GmmFlags::MEANS) {
        for g in 0..gmm.num_gauss() {
            for d in 0..gmm.dim() {
                obj += acc.mean_accum[[g, d]] * gmm.means_invvars[[g, d]] as f64;
            }
        }
    }
    if acc.flags.contains(GmmFlags::VARIANCES) {
        for g in 0..gmm.num_gauss() {
            for d in 0..gmm.dim() {
                obj -= 0.5 * acc.var_accum[[g, d]] * gmm.inv_vars[[g, d]] as f64;
            }
        }
    }
    obj
}

/// ML update of one `DiagGmm` from its stats; returns `(objf_change, count)`.
///
/// `flags` selects which parameters are actually written back — they must be a
/// subset of the flags the accumulator was built with.
pub fn mle_diag_gmm_update(
    opts: &MleDiagGmmOptions,
    acc: &AccumDiagGmm,
    flags: GmmFlags,
    gmm: &mut DiagGmm,
) -> (f64, f64) {
    let s = mle_diag_gmm_update_detailed(opts, acc, flags, gmm);
    (s.objf_change, s.count)
}

/// As [`mle_diag_gmm_update`], but reporting flooring and removal counts too.
pub fn mle_diag_gmm_update_detailed(
    opts: &MleDiagGmmOptions,
    acc: &AccumDiagGmm,
    flags: GmmFlags,
    gmm: &mut DiagGmm,
) -> UpdateStats {
    assert!(
        acc.flags.contains(flags),
        "flags in argument do not match the active accumulators"
    );
    assert_eq!(
        acc.num_gauss(),
        gmm.num_gauss(),
        "mle_diag_gmm_update: component count mismatch"
    );
    assert_eq!(
        acc.dim(),
        gmm.dim(),
        "mle_diag_gmm_update: dimension mismatch"
    );

    let num_gauss = gmm.num_gauss();
    let dim = gmm.dim();
    let occ_sum: f64 = acc.occupancy_sum();
    let mut elements_floored = 0usize;
    let mut gauss_floored = 0usize;

    // Remember the old objective value.
    gmm.compute_gconsts();
    let obj_old = ml_objective(gmm, acc);

    // Work in the "normal" parameterisation (means and variances), like Kaldi's
    // DiagGmmNormal, then convert back at the end.
    let mut weights = gmm.weights.clone();
    let mut means = gmm.means();
    let mut vars = gmm.vars();

    let mut to_remove: Vec<usize> = Vec::new();
    for i in 0..num_gauss {
        let occ = acc.occupancy[i];
        let prob = if occ_sum > 0.0 {
            occ / occ_sum
        } else {
            1.0 / num_gauss as f64
        };

        if occ > opts.min_gaussian_occupancy && prob > opts.min_gaussian_weight {
            weights[i] = prob as f32;

            // Keep the old mean; needed to compensate a variance-only update.
            let old_mean: Vec<f64> = (0..dim).map(|d| means[[i, d]] as f64).collect();

            if acc.flags.intersects(GmmFlags::MEANS | GmmFlags::VARIANCES) {
                for d in 0..dim {
                    means[[i, d]] = (acc.mean_accum[[i, d]] / occ) as f32;
                }
            }

            if acc.flags.contains(GmmFlags::VARIANCES) {
                debug_assert!(acc.flags.contains(GmmFlags::MEANS));
                let mut floored_this = 0usize;
                for d in 0..dim {
                    let m = means[[i, d]] as f64;
                    // E[x^2] - mean^2
                    let mut var = acc.var_accum[[i, d]] / occ - m * m;

                    // If only variances are being updated, compensate with the
                    // difference between the new and old mean.
                    if !flags.contains(GmmFlags::MEANS) {
                        let diff = old_mean[d] - m;
                        var += diff * diff;
                    }

                    let floor = if opts.variance_floor_vector.is_empty() {
                        opts.min_variance
                    } else {
                        opts.variance_floor_vector[d]
                    };
                    if var < floor {
                        var = floor;
                        floored_this += 1;
                    }
                    vars[[i, d]] = var as f32;
                }
                if floored_this != 0 {
                    elements_floored += floored_this;
                    gauss_floored += 1;
                }
            }
        } else {
            // Insufficient occupancy.
            if opts.remove_low_count_gaussians && to_remove.len() < num_gauss - 1 {
                // Remove the component, unless it is the last one.
                tracing::debug!(
                    weight = prob,
                    occupancy = occ,
                    dim,
                    "too little data - removing Gaussian"
                );
                to_remove.push(i);
            } else {
                tracing::debug!(
                    component = i,
                    occupancy = occ,
                    weight = prob,
                    last_gaussian = opts.remove_low_count_gaussians,
                    "Gaussian has too little data but is not being removed"
                );
                weights[i] = prob.max(opts.min_gaussian_weight) as f32;
            }
        }
    }

    // Copy back only what `flags` selects.
    if flags.contains(GmmFlags::WEIGHTS) {
        gmm.weights = weights;
    }
    if flags.intersects(GmmFlags::MEANS | GmmFlags::VARIANCES) {
        // means_invvars and inv_vars are coupled, so both are rewritten from the
        // (possibly partly unchanged) natural parameters.
        let new_means = if flags.contains(GmmFlags::MEANS) {
            means
        } else {
            gmm.means()
        };
        let new_vars = if flags.contains(GmmFlags::VARIANCES) {
            vars
        } else {
            gmm.vars()
        };
        gmm.set_means_and_vars(&new_means, &new_vars);
    }

    gmm.compute_gconsts(); // or ml_objective will fail.
    let obj_new = ml_objective(gmm, acc);

    let removed_gaussians = to_remove.len();
    if removed_gaussians > 0 {
        let keep: Vec<usize> = (0..num_gauss).filter(|i| !to_remove.contains(i)).collect();
        gmm.retain_components(&keep); // renormalises the weights
        gmm.compute_gconsts();
    }

    if gauss_floored > 0 {
        tracing::debug!(elements_floored, gauss_floored, "variance elements floored");
    }

    UpdateStats {
        objf_change: obj_new - obj_old,
        count: occ_sum,
        floored_elements: elements_floored,
        floored_gaussians: gauss_floored,
        removed_gaussians,
    }
}

/// ML update of every pdf in an acoustic model; returns `(objf_change, count)`
/// summed over pdfs (`MleAmDiagGmmUpdate`, `mle-am-diag-gmm.cc:187`).
pub fn mle_am_diag_gmm_update(
    opts: &MleDiagGmmOptions,
    acc: &AccumAmDiagGmm,
    flags: GmmFlags,
    am: &mut AmDiagGmm,
) -> (f64, f64) {
    assert_eq!(
        acc.num_accs(),
        am.num_pdfs(),
        "mle_am_diag_gmm_update: accumulator/model pdf count mismatch"
    );

    let mut tot_obj_change = 0.0f64;
    let mut tot_count = 0.0f64;
    let mut tot_elems_floored = 0usize;
    let mut tot_gauss_floored = 0usize;
    let mut tot_gauss_removed = 0usize;

    for i in 0..acc.num_accs() {
        let s = mle_diag_gmm_update_detailed(opts, &acc.accs[i], flags, am.pdf_mut(i as PdfId));
        tot_obj_change += s.objf_change;
        tot_count += s.count;
        tot_elems_floored += s.floored_elements;
        tot_gauss_floored += s.floored_gaussians;
        tot_gauss_removed += s.removed_gaussians;
    }

    tracing::info!(
        elements_floored = tot_elems_floored,
        gaussians_floored = tot_gauss_floored,
        num_gauss = am.num_gauss(),
        "variance elements floored"
    );
    if opts.remove_low_count_gaussians {
        tracing::info!(
            removed = tot_gauss_removed,
            min_gaussian_occupancy = opts.min_gaussian_occupancy,
            "removed Gaussians due to low counts"
        );
    }

    (tot_obj_change, tot_count)
}

/// I-smooth every pdf's stats towards the current model
/// (`IsmoothStatsAmDiagGmmFromModel`, `kalpy/extensions/gmm/gmm.cpp:1009`).
pub fn ismooth_stats_am_diag_gmm_from_model(am: &AmDiagGmm, tau: f64, acc: &mut AccumAmDiagGmm) {
    assert_eq!(
        acc.num_accs(),
        am.num_pdfs(),
        "ismooth_stats_am_diag_gmm_from_model: pdf count mismatch"
    );
    for i in 0..acc.num_accs() {
        acc.accs[i].smooth_with_model(tau, am.pdf(i as PdfId));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::RngExt;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::SeedableRng;

    /// A two-component 1-D GMM, deliberately wrong, to be re-estimated.
    fn start_gmm() -> DiagGmm {
        let mut g = DiagGmm::new(2, 1);
        g.set_means_and_vars(&array![[-1.0f32], [1.0]], &array![[1.0f32], [1.0]]);
        g.weights = vec![0.5, 0.5];
        g.compute_gconsts();
        g
    }

    fn opts() -> MleDiagGmmOptions {
        MleDiagGmmOptions {
            min_gaussian_occupancy: 1.0,
            min_variance: 1e-6,
            ..Default::default()
        }
    }

    #[test]
    fn update_recovers_the_generating_gaussian() {
        // One component, data from N(5, 4): the ML estimate must find it.
        let mut g = DiagGmm::from_single_gaussian(&[0.0], &[1.0]);
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(21);
        let n = 4000;
        let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let x = 5.0 + 2.0 * super::super::rand_gauss(&mut rng);
            sum += x as f64;
            sumsq += (x as f64) * (x as f64);
            acc.accumulate_from_posteriors(&[x], &[1.0]);
        }
        let (_objf, count) = mle_diag_gmm_update(&opts(), &acc, GmmFlags::ALL, &mut g);
        assert!((count - n as f64).abs() < 1e-6);

        let want_mean = sum / n as f64;
        let want_var = sumsq / n as f64 - want_mean * want_mean;
        assert!((g.means()[[0, 0]] as f64 - want_mean).abs() < 1e-3);
        assert!((g.vars()[[0, 0]] as f64 - want_var).abs() < 1e-2);
        assert!((g.weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn update_increases_the_objective() {
        let mut g = start_gmm();
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(33);
        for _ in 0..2000 {
            // Bimodal data at -4 and +4; the start model is far too narrow.
            let centre = if rng.random_bool(0.5) { -4.0 } else { 4.0 };
            let x = centre + super::super::rand_gauss(&mut rng);
            acc.accumulate_from_diag(&g, &[x], 1.0);
        }
        let (objf_change, count) = mle_diag_gmm_update(&opts(), &acc, GmmFlags::ALL, &mut g);
        assert!(count > 1999.0 && count < 2001.0);
        assert!(
            objf_change > 0.0,
            "objf change {objf_change} should be positive"
        );
        // Components have moved towards the two real modes.
        let mut ms = [g.means()[[0, 0]], g.means()[[1, 0]]];
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ms[0] + 4.0).abs() < 0.3, "{ms:?}");
        assert!((ms[1] - 4.0).abs() < 0.3, "{ms:?}");
    }

    #[test]
    fn weights_become_normalised_occupancies() {
        let mut g = start_gmm();
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        // 30 counts on component 0, 10 on component 1.
        for _ in 0..30 {
            acc.accumulate_from_posteriors(&[-1.0], &[1.0, 0.0]);
        }
        for _ in 0..10 {
            acc.accumulate_from_posteriors(&[1.0], &[0.0, 1.0]);
        }
        mle_diag_gmm_update(&opts(), &acc, GmmFlags::ALL, &mut g);
        assert!((g.weights[0] - 0.75).abs() < 1e-5);
        assert!((g.weights[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn variance_floor_is_applied() {
        // All data at one point: the ML variance is 0 and must be floored.
        let mut g = DiagGmm::from_single_gaussian(&[0.0], &[1.0]);
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        for _ in 0..100 {
            acc.accumulate_from_posteriors(&[2.0], &[1.0]);
        }
        let o = MleDiagGmmOptions {
            min_gaussian_occupancy: 1.0,
            min_variance: 0.01,
            ..Default::default()
        };
        let s = mle_diag_gmm_update_detailed(&o, &acc, GmmFlags::ALL, &mut g);
        assert_eq!(s.floored_elements, 1);
        assert_eq!(s.floored_gaussians, 1);
        assert!((g.vars()[[0, 0]] - 0.01).abs() < 1e-6);
        assert!((g.means()[[0, 0]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn low_count_gaussian_is_removed() {
        let mut g = start_gmm();
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        // Component 1 gets almost nothing.
        for _ in 0..100 {
            acc.accumulate_from_posteriors(&[-1.0], &[1.0, 0.0]);
        }
        acc.accumulate_from_posteriors(&[1.0], &[0.0, 0.001]);
        let s = mle_diag_gmm_update_detailed(&opts(), &acc, GmmFlags::ALL, &mut g);
        assert_eq!(s.removed_gaussians, 1);
        assert_eq!(g.num_gauss(), 1);
        // The surviving weight is renormalised back to 1.
        assert!((g.weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn low_count_gaussian_is_kept_when_removal_disabled() {
        let mut g = start_gmm();
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        for _ in 0..100 {
            acc.accumulate_from_posteriors(&[-1.0], &[1.0, 0.0]);
        }
        acc.accumulate_from_posteriors(&[1.0], &[0.0, 0.001]);
        let o = MleDiagGmmOptions {
            min_gaussian_occupancy: 1.0,
            min_variance: 1e-6,
            remove_low_count_gaussians: false,
            ..Default::default()
        };
        let s = mle_diag_gmm_update_detailed(&o, &acc, GmmFlags::ALL, &mut g);
        assert_eq!(s.removed_gaussians, 0);
        assert_eq!(g.num_gauss(), 2);
        // The starved component is floored to min_gaussian_weight, not deleted.
        assert!(g.weights[1] >= o.min_gaussian_weight as f32);
    }

    #[test]
    fn last_gaussian_is_never_removed() {
        let mut g = DiagGmm::from_single_gaussian(&[0.0], &[1.0]);
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        acc.accumulate_from_posteriors(&[1.0], &[0.0001]);
        let s = mle_diag_gmm_update_detailed(&opts(), &acc, GmmFlags::ALL, &mut g);
        assert_eq!(s.removed_gaussians, 0);
        assert_eq!(g.num_gauss(), 1);
    }

    #[test]
    fn variance_only_update_leaves_means_alone() {
        let mut g = DiagGmm::from_single_gaussian(&[0.0], &[1.0]);
        let mut acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(8);
        for _ in 0..2000 {
            let x = 5.0 + 2.0 * super::super::rand_gauss(&mut rng);
            acc.accumulate_from_posteriors(&[x], &[1.0]);
        }
        mle_diag_gmm_update(
            &opts(),
            &acc,
            GmmFlags::VARIANCES | GmmFlags::WEIGHTS,
            &mut g,
        );
        // Mean is untouched...
        assert!(g.means()[[0, 0]].abs() < 1e-5);
        // ...and the variance absorbs the mean offset: about 4 + 25.
        let v = g.vars()[[0, 0]] as f64;
        assert!((v - 29.0).abs() < 1.5, "variance {v}");
    }

    #[test]
    fn am_update_sums_over_pdfs() {
        let am_proto = start_gmm();
        let mut am = AmDiagGmm::init(&am_proto, 3);
        let mut acc = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12);
        for p in 0..3u32 {
            let centre = p as f32 * 6.0;
            for _ in 0..500 {
                let x = centre + super::super::rand_gauss(&mut rng);
                acc.accumulate_for_gmm(&am, p, &[x], 1.0);
            }
        }
        let (objf, count) = mle_am_diag_gmm_update(&opts(), &acc, GmmFlags::ALL, &mut am);
        assert!((count - 1500.0).abs() < 1.0);
        assert!(objf > 0.0);
        // Each pdf has migrated to its own data centre.
        for p in 0..3u32 {
            let means = am.pdf(p).means();
            let centre = p as f32 * 6.0;
            let close = (0..am.pdf(p).num_gauss()).any(|g| (means[[g, 0]] - centre).abs() < 1.0);
            assert!(close, "pdf {p} did not move to {centre}");
        }
        // The model version was bumped by the update.
        assert!(am.version() > 0);
    }

    #[test]
    fn ismoothing_pulls_a_starved_pdf_back_to_the_model() {
        let am = AmDiagGmm::init(&DiagGmm::from_single_gaussian(&[3.0], &[2.0]), 2);
        let mut acc = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        // pdf 1 sees a single frame far from the model.
        acc.accumulate_for_gmm(&am, 1, &[-20.0], 1.0);
        ismooth_stats_am_diag_gmm_from_model(&am, 100.0, &mut acc);

        // pdf 0 has only virtual counts, so its stats reproduce the model exactly.
        let a0 = &acc.accs[0];
        assert!((a0.occupancy[0] - 100.0).abs() < 1e-9);
        assert!((a0.mean_accum[[0, 0]] / a0.occupancy[0] - 3.0).abs() < 1e-4);

        // pdf 1's mean is dominated by the 100 virtual counts, not the one outlier.
        let a1 = &acc.accs[1];
        let mean1 = a1.mean_accum[[0, 0]] / a1.occupancy[0];
        assert!(mean1 > 2.5 && mean1 < 3.0, "smoothed mean {mean1}");
    }

    #[test]
    fn update_with_no_data_uses_uniform_prob() {
        // occ_sum == 0: prob falls back to 1/num_gauss, which is above the weight
        // floor, but occupancy is still zero so the component is starved.
        let mut g = start_gmm();
        let acc = AccumDiagGmm::from_gmm(&g, GmmFlags::ALL);
        let s = mle_diag_gmm_update_detailed(&opts(), &acc, GmmFlags::ALL, &mut g);
        assert_eq!(s.count, 0.0);
        // One removed, one kept because it is the last.
        assert_eq!(g.num_gauss(), 1);
    }
}
