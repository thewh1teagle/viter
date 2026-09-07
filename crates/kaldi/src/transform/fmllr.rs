//! Port of `plans/kaldi/src/transform/fmllr-diag-gmm.{h,cc}` and the
//! `AffineXformStats` half of `transform-common.{h,cc}`.
//!
//! Feature-space MLLR (a.k.a. CMLLR) for diagonal-covariance GMMs. The stats
//! are Kaldi's `AffineXformStats`: a count `beta`, a `[dim, dim+1]` linear
//! term `K`, and `dim` symmetric `[dim+1, dim+1]` matrices `G`. The transform
//! is a `[dim, dim+1]` affine matrix `W = [A; b]` applied as
//! `x' = A x + b`.

use super::Mat;
use super::linalg::SpMat;
use crate::gmm::DiagGmm;

/// Kaldi's fMLLR update variants (`fmllr-update-type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FmllrUpdateType {
    /// Full affine transform (MFA default).
    #[default]
    Full,
    /// Diagonal `A` plus offset.
    Diag,
    /// Offset only; `A` stays the identity.
    Offset,
    /// No update at all.
    None,
}

/// Kaldi `FmllrOptions`. Defaults match Kaldi and MFA: full update,
/// `min_count = 500`, `num_iters = 40`.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct FmllrOptions {
    pub update_type: FmllrUpdateType,
    pub min_count: f64,
    pub num_iters: usize,
}

impl Default for FmllrOptions {
    fn default() -> Self {
        Self {
            update_type: FmllrUpdateType::Full,
            min_count: 500.0,
            num_iters: 40,
        }
    }
}

/// Kaldi `identity` affine transform: `[dim, dim+1]`, `A = I`, `b = 0`.
pub fn identity_affine(dim: usize) -> Mat {
    ndarray::Array2::from_shape_fn((dim, dim + 1), |(i, j)| if i == j { 1.0 } else { 0.0 })
}

/// Per-frame scratch stats (Kaldi `FmllrDiagGmmAccs::SingleFrameStats`).
///
/// Kaldi batches the contributions of all pdfs that fire on one frame before
/// committing them, which keeps the `[dim+1, dim+1]` outer product to one per
/// frame instead of one per pdf. We replicate that exactly, including the
/// "data changed" check that triggers a commit.
#[derive(Clone, Debug)]
struct SingleFrameStats {
    x: Vec<f32>,
    /// Linear term in the per-frame auxf, model-dim.
    a: Vec<f64>,
    /// Quadratic term in the per-frame auxf, model-dim.
    b: Vec<f64>,
    count: f64,
}

impl SingleFrameStats {
    fn new(dim: usize) -> Self {
        Self {
            x: vec![0.0; dim],
            a: vec![0.0; dim],
            b: vec![0.0; dim],
            count: 0.0,
        }
    }
    fn reset(&mut self) {
        self.count = 0.0;
        self.a.iter_mut().for_each(|v| *v = 0.0);
        self.b.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Kaldi `FmllrDiagGmmAccs` (which is an `AffineXformStats`).
#[derive(Clone, Debug)]
pub struct FmllrDiagGmmAccs {
    pub(super) dim: usize,
    /// `beta_`: total count.
    pub(super) beta: f64,
    /// `K_`: `[dim, dim+1]`, row-major f64.
    pub(super) k: Vec<f64>,
    /// `G_`: `dim` symmetric matrices of size `dim+1`.
    pub(super) g: Vec<SpMat>,
    single: SingleFrameStats,
    /// Whether `single` currently holds a valid frame (Kaldi tracks this via
    /// the count plus an approximate-equality check on `x`).
    single_valid: bool,
    /// Limits which parts of `G` we bother filling in, like Kaldi's `opts_`.
    update_type: FmllrUpdateType,
}

impl FmllrDiagGmmAccs {
    /// Kaldi `FmllrDiagGmmAccs::Init(dim)`; accumulates the full stats needed
    /// for a `Full` update.
    pub fn new(dim: usize) -> Self {
        Self::with_update_type(dim, FmllrUpdateType::Full)
    }

    /// As `new`, but only accumulate the elements a limited update needs.
    pub fn with_update_type(dim: usize, update_type: FmllrUpdateType) -> Self {
        assert!(dim > 0, "fMLLR dim must be positive");
        Self {
            dim,
            beta: 0.0,
            k: vec![0.0; dim * (dim + 1)],
            g: (0..dim).map(|_| SpMat::zeros(dim + 1)).collect(),
            single: SingleFrameStats::new(dim),
            single_valid: false,
            update_type,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total count (`beta_`). Includes any not-yet-committed frame.
    pub fn count(&self) -> f64 {
        self.beta + self.single.count
    }

    #[inline]
    pub(super) fn k_at(&self, i: usize, j: usize) -> f64 {
        self.k[i * (self.dim + 1) + j]
    }

    /// Kaldi `FmllrDiagGmmAccs::DataHasChanged(data)`.
    fn data_has_changed(&self, x: &[f32]) -> bool {
        !self.single_valid || self.single.x != x
    }

    fn init_single_frame_stats(&mut self, x: &[f32]) {
        self.single.x.copy_from_slice(x);
        self.single.reset();
        self.single_valid = true;
    }

    /// Kaldi `FmllrDiagGmmAccs::CommitSingleFrameStats()`.
    pub(super) fn commit_single_frame_stats(&mut self) {
        let dim = self.dim;
        if self.single.count == 0.0 {
            return;
        }
        // x^+ = [x; 1]
        let mut xplus = vec![0.0f64; dim + 1];
        for d in 0..dim {
            xplus[d] = self.single.x[d] as f64;
        }
        xplus[dim] = 1.0;

        self.beta += self.single.count;
        // K += a (x^+)^T
        for i in 0..dim {
            let ai = self.single.a[i];
            if ai == 0.0 {
                continue;
            }
            for j in 0..=dim {
                self.k[i * (dim + 1) + j] += ai * xplus[j];
            }
        }

        if self.update_type == FmllrUpdateType::Full {
            let mut scatter = SpMat::zeros(dim + 1);
            scatter.add_vec2(1.0, &xplus);
            for i in 0..dim {
                let bi = self.single.b[i];
                if bi != 0.0 {
                    self.g[i].add_sp(bi, &scatter);
                }
            }
        } else {
            // Only the elements a diag/offset update reads.
            for i in 0..dim {
                let scale = self.single.b[i];
                let x_i = xplus[i];
                self.g[i].m[(i, i)] += scale * x_i * x_i;
                let v = scale * x_i;
                self.g[i].m[(dim, i)] += v;
                if dim != i {
                    self.g[i].m[(i, dim)] += v;
                }
                self.g[i].m[(dim, dim)] += scale;
            }
        }

        self.single.reset();
    }

    /// Kaldi `FmllrDiagGmmAccs::AccumulateFromPosteriors(gmm, data, posterior)`.
    pub fn accumulate_from_posteriors(&mut self, gmm: &DiagGmm, x: &[f32], posteriors: &[f32]) {
        assert_eq!(x.len(), self.dim, "fMLLR: data dim mismatch");
        assert_eq!(gmm.dim(), self.dim, "fMLLR: gmm dim mismatch");
        assert_eq!(
            posteriors.len(),
            gmm.num_gauss(),
            "fMLLR: posterior count mismatch"
        );

        if self.data_has_changed(x) {
            self.commit_single_frame_stats();
            self.init_single_frame_stats(x);
        }
        let dim = self.dim;
        self.single.count += posteriors.iter().map(|&p| p as f64).sum::<f64>();
        // a += means_invvars^T * posterior ; b += inv_vars^T * posterior
        for (gidx, &p) in posteriors.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            let pd = p as f64;
            for d in 0..dim {
                self.single.a[d] += pd * gmm.means_invvars[[gidx, d]] as f64;
                self.single.b[d] += pd * gmm.inv_vars[[gidx, d]] as f64;
            }
        }
    }

    /// Kaldi `FmllrDiagGmmAccs::AccumulateForGmm(gmm, data, weight)`; returns
    /// the frame log-likelihood.
    pub fn accumulate_for_gmm(&mut self, gmm: &DiagGmm, x: &[f32], weight: f32) -> f32 {
        let mut posteriors = Vec::new();
        let loglike = gmm.component_posteriors(x, &mut posteriors);
        for p in posteriors.iter_mut() {
            *p *= weight;
        }
        self.accumulate_from_posteriors(gmm, x, &posteriors);
        loglike
    }

    /// Kaldi `AffineXformStats::Add(other)`. Both sides' pending single-frame
    /// stats are committed first.
    pub fn add(&mut self, o: &Self) {
        assert_eq!(self.dim, o.dim, "fMLLR: adding stats of different dim");
        self.commit_single_frame_stats();
        // `o` is borrowed immutably, so fold in its pending frame by hand.
        let mut other_beta = o.beta;
        let mut other_k = o.k.clone();
        let mut other_g: Vec<SpMat> = o.g.clone();
        if o.single.count != 0.0 {
            let dim = o.dim;
            let mut xplus = vec![0.0f64; dim + 1];
            for d in 0..dim {
                xplus[d] = o.single.x[d] as f64;
            }
            xplus[dim] = 1.0;
            other_beta += o.single.count;
            for i in 0..dim {
                let ai = o.single.a[i];
                for j in 0..=dim {
                    other_k[i * (dim + 1) + j] += ai * xplus[j];
                }
            }
            if o.update_type == FmllrUpdateType::Full {
                let mut scatter = SpMat::zeros(dim + 1);
                scatter.add_vec2(1.0, &xplus);
                for i in 0..dim {
                    other_g[i].add_sp(o.single.b[i], &scatter);
                }
            } else {
                for i in 0..dim {
                    let scale = o.single.b[i];
                    let x_i = xplus[i];
                    other_g[i].m[(i, i)] += scale * x_i * x_i;
                    let v = scale * x_i;
                    other_g[i].m[(dim, i)] += v;
                    if dim != i {
                        other_g[i].m[(i, dim)] += v;
                    }
                    other_g[i].m[(dim, dim)] += scale;
                }
            }
        }

        self.beta += other_beta;
        for (a, b) in self.k.iter_mut().zip(other_k.iter()) {
            *a += *b;
        }
        for (a, b) in self.g.iter_mut().zip(other_g.iter()) {
            a.add_sp(1.0, b);
        }
    }

    /// Kaldi `FmllrDiagGmmAccs::Update` / `ComputeFmllrMatrixDiagGmm`.
    ///
    /// Starts from `xform` (or the identity when `None`) and returns
    /// `(new xform [dim, dim+1], objf improvement, count)`. Below `min_count`
    /// the input transform is returned unchanged with zero improvement, as in
    /// Kaldi.
    pub fn update(&self, opts: &FmllrOptions, xform: Option<&Mat>) -> (Mat, f64, f64) {
        let dim = self.dim;
        // Fold any uncommitted frame in on a scratch copy so `update` is &self.
        let mut stats = self.clone();
        stats.commit_single_frame_stats();

        let in_xform = match xform {
            Some(m) => {
                assert_eq!(m.nrows(), dim, "fMLLR: xform has wrong number of rows");
                assert_eq!(
                    m.ncols(),
                    dim + 1,
                    "fMLLR: xform has wrong number of columns"
                );
                m.clone()
            }
            None => identity_affine(dim),
        };
        assert!(
            in_xform.iter().any(|&v| v != 0.0),
            "fMLLR: initial transform must be non-singular (e.g. the identity)"
        );

        if opts.update_type == FmllrUpdateType::Full && stats.update_type != FmllrUpdateType::Full {
            panic!("fMLLR: requesting a full update but stats were accumulated for a limited type");
        }

        if stats.beta <= opts.min_count {
            tracing::warn!(
                count = stats.beta,
                min_count = opts.min_count,
                "not updating fMLLR since below min-count"
            );
            return (in_xform, 0.0, stats.beta);
        }

        let (out, impr) = match opts.update_type {
            FmllrUpdateType::Full => {
                super::fmllr_update::compute_fmllr_matrix_full(&in_xform, &stats, opts.num_iters)
            }
            FmllrUpdateType::Diag => {
                super::fmllr_update::compute_fmllr_matrix_diagonal(&in_xform, &stats)
            }
            FmllrUpdateType::Offset => {
                super::fmllr_update::compute_fmllr_matrix_offset(&in_xform, &stats)
            }
            FmllrUpdateType::None => (in_xform.clone(), 0.0),
        };
        (out, impr, stats.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::fmllr_update::{
        fmllr_aux_func, fmllr_aux_func_diag_gmm, fmllr_inner_update,
    };

    fn make_gmm(num_gauss: usize, dim: usize) -> DiagGmm {
        let mut means = ndarray::Array2::<f32>::zeros((num_gauss, dim));
        let mut vars = ndarray::Array2::<f32>::zeros((num_gauss, dim));
        for g in 0..num_gauss {
            for d in 0..dim {
                means[[g, d]] = (g as f32) * 2.0 + (d as f32) * 0.5 - 1.0;
                vars[[g, d]] = 0.5 + 0.1 * ((g * 3 + d) % 4) as f32;
            }
        }
        let mut gmm = DiagGmm::new(num_gauss, dim);
        gmm.set_means_and_vars(&means, &vars);
        for w in gmm.weights.iter_mut() {
            *w = 1.0 / num_gauss as f32;
        }
        gmm.compute_gconsts();
        gmm
    }

    /// Data generated by applying a known affine warp to samples drawn from the
    /// model; fMLLR should recover something close to the inverse warp and must
    /// increase the auxiliary function.
    fn accumulate_shifted(dim: usize, shift: f32, n: usize) -> (DiagGmm, FmllrDiagGmmAccs) {
        let gmm = make_gmm(3, dim);
        let mut accs = FmllrDiagGmmAccs::new(dim);
        for t in 0..n {
            let x: Vec<f32> = (0..dim)
                .map(|d| ((t * 7 + d * 5) % 13) as f32 * 0.4 - 2.0 + shift)
                .collect();
            accs.accumulate_for_gmm(&gmm, &x, 1.0);
        }
        (gmm, accs)
    }

    #[test]
    fn identity_affine_shape_and_values() {
        let m = identity_affine(3);
        assert_eq!(m.shape(), &[3, 4]);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[1, 1]], 1.0);
        assert_eq!(m[[2, 3]], 0.0);
    }

    #[test]
    fn count_matches_frames() {
        let (_g, accs) = accumulate_shifted(4, 0.0, 120);
        assert!(
            (accs.count() - 120.0).abs() < 1e-2,
            "count = {}",
            accs.count()
        );
    }

    #[test]
    fn full_update_increases_objective() {
        let dim = 4;
        let (_g, accs) = accumulate_shifted(dim, 3.0, 900);
        let opts = FmllrOptions {
            min_count: 10.0,
            ..Default::default()
        };
        let start = identity_affine(dim);
        let before = fmllr_aux_func_diag_gmm(&start, &accs);
        let (out, impr, count) = accs.update(&opts, None);
        assert_eq!(out.shape(), &[dim, dim + 1]);
        assert!(count > 800.0);
        assert!(impr > 0.0, "expected improvement, got {impr}");
        let after = fmllr_aux_func_diag_gmm(&out, &accs);
        assert!((after - before - impr).abs() < 1e-3 * impr.abs().max(1.0));
        // The transform should undo part of the shift: negative offsets.
        assert!(
            out.column(dim).iter().any(|&v| v < 0.0),
            "offsets: {:?}",
            out.column(dim)
        );
    }

    #[test]
    fn below_min_count_returns_input_unchanged() {
        let dim = 3;
        let (_g, accs) = accumulate_shifted(dim, 1.0, 20);
        let opts = FmllrOptions::default(); // min_count = 500
        let (out, impr, count) = accs.update(&opts, None);
        assert_eq!(impr, 0.0);
        assert!(count < 500.0);
        assert_eq!(out, identity_affine(dim));
    }

    #[test]
    fn none_update_is_identity_passthrough() {
        let dim = 3;
        let (_g, accs) = accumulate_shifted(dim, 1.0, 900);
        let opts = FmllrOptions {
            update_type: FmllrUpdateType::None,
            min_count: 1.0,
            ..Default::default()
        };
        let (out, impr, _c) = accs.update(&opts, None);
        assert_eq!(impr, 0.0);
        assert_eq!(out, identity_affine(dim));
    }

    #[test]
    fn diagonal_update_keeps_matrix_diagonal_and_improves() {
        let dim = 4;
        let (_g, accs) = accumulate_shifted(dim, 2.5, 900);
        let opts = FmllrOptions {
            update_type: FmllrUpdateType::Diag,
            min_count: 10.0,
            ..Default::default()
        };
        let (out, impr, _c) = accs.update(&opts, None);
        assert!(impr > 0.0, "diag improvement {impr}");
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    assert_eq!(out[[i, j]], 0.0);
                }
            }
            assert!(out[[i, i]] > 0.0);
        }
    }

    #[test]
    fn offset_update_touches_only_the_last_column() {
        let dim = 4;
        let (_g, accs) = accumulate_shifted(dim, 2.5, 900);
        let opts = FmllrOptions {
            update_type: FmllrUpdateType::Offset,
            min_count: 10.0,
            ..Default::default()
        };
        let (out, impr, _c) = accs.update(&opts, None);
        assert!(impr > 0.0, "offset improvement {impr}");
        for i in 0..dim {
            for j in 0..dim {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(out[[i, j]], want);
            }
        }
    }

    #[test]
    fn add_sums_stats_including_pending_frame() {
        let dim = 3;
        let (_g1, a) = accumulate_shifted(dim, 0.0, 50);
        let (_g2, b) = accumulate_shifted(dim, 0.0, 50);
        let mut c = a.clone();
        c.add(&b);
        assert!((c.count() - (a.count() + b.count())).abs() < 1e-6);
        // K entries add too.
        assert!((c.k_at(0, 0) - 2.0 * a.k_at(0, 0)).abs() < 1e-6 * a.k_at(0, 0).abs().max(1.0));
    }

    #[test]
    fn inner_update_improves_one_row() {
        let dim = 3;
        let (_g, accs0) = accumulate_shifted(dim, 2.0, 900);
        let mut accs = accs0.clone();
        accs.commit_single_frame_stats();
        let mut flat: Vec<f64> = (0..dim)
            .flat_map(|i| (0..=dim).map(move |j| (i, j)))
            .map(|(i, j)| if i == j { 1.0 } else { 0.0 })
            .collect();
        let before = fmllr_aux_func(&flat, &accs);
        let inv_g = accs.g[0].inverted();
        let k_row: Vec<f64> = (0..=dim).map(|j| accs.k_at(0, j)).collect();
        fmllr_inner_update(&inv_g, &k_row, accs.beta, 0, &mut flat, dim);
        let after = fmllr_aux_func(&flat, &accs);
        assert!(after > before, "{after} !> {before}");
    }

    #[test]
    fn aux_func_matches_definition_on_identity() {
        let dim = 3;
        let (_g, accs0) = accumulate_shifted(dim, 0.5, 200);
        let mut accs = accs0.clone();
        accs.commit_single_frame_stats();
        let w = identity_affine(dim);
        let got = fmllr_aux_func_diag_gmm(&w, &accs);
        // beta*log|I| = 0; tr(W^T K) = sum of K's diagonal square part;
        // minus 0.5 * sum_d G_d(d,d).
        let mut want = 0.0;
        for d in 0..dim {
            want += accs.k_at(d, d);
            want -= 0.5 * accs.g[d].get(d, d);
        }
        assert!(
            (got - want).abs() < 1e-6 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}
