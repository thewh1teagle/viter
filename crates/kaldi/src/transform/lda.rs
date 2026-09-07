//! Port of `plans/kaldi/src/transform/lda-estimate.{h,cc}`.
//!
//! `LdaEstimate` accumulates per-class (per-pdf) zeroth and first order stats
//! plus a single global second-order scatter, all in `f64` exactly as Kaldi
//! does. `estimate` solves the generalized eigenproblem
//! `B v = lambda W v` by Cholesky-whitening with `W = L L^T` and taking the
//! self-adjoint eigendecomposition of `L^{-1} B L^{-T}` — this is precisely
//! what `LdaEstimate::Estimate` computes via `Matrix::Svd` + `SortSvd` on a
//! symmetric PSD matrix.

use super::Mat;
use super::linalg::{SpMat, cholesky_lower, invert, sym_eig_descending};

/// Kaldi `LdaEstimateOptions`.
#[derive(Clone, Copy, Debug)]
pub struct LdaEstimateOptions {
    /// Dimension to project to with LDA. Kaldi default 40 (MFA `lda_dimension`).
    pub dim: usize,
    /// If true, output an affine transform whose projected data mean is zero.
    /// Kaldi default `false`.
    pub remove_offset: bool,
    /// Deprecated in Kaldi; 1.0 gives conventional LDA (unit within-class
    /// variance in the projected space).
    pub within_class_factor: f64,
    /// Allow an LDA dimension larger than the number of classes.
    pub allow_large_dim: bool,
}

impl Default for LdaEstimateOptions {
    fn default() -> Self {
        Self {
            dim: 40,
            remove_offset: false,
            within_class_factor: 1.0,
            allow_large_dim: false,
        }
    }
}

/// Kaldi `LdaEstimate`: per-class count/first-order stats and a global
/// second-order scatter.
#[derive(Clone, Debug)]
pub struct LdaEstimate {
    /// `zero_acc_`: per-class total weight, `[num_classes]`.
    zero_acc: Vec<f64>,
    /// `first_acc_`: per-class weighted sum of features, `[num_classes, dim]`.
    first_acc: ndarray::Array2<f64>,
    /// `total_second_acc_`: global weighted `sum_t w_t x_t x_t^T`.
    total_second_acc: SpMat,
}

impl LdaEstimate {
    /// Kaldi `LdaEstimate::Init(num_classes, dimension)`.
    pub fn new(num_classes: usize, dim: usize) -> Self {
        Self {
            zero_acc: vec![0.0; num_classes],
            first_acc: ndarray::Array2::zeros((num_classes, dim)),
            total_second_acc: SpMat::zeros(dim),
        }
    }

    pub fn num_classes(&self) -> usize {
        self.zero_acc.len()
    }

    pub fn dim(&self) -> usize {
        self.first_acc.ncols()
    }

    /// Kaldi `LdaEstimate::TotCount()`.
    pub fn tot_count(&self) -> f64 {
        self.zero_acc.iter().sum()
    }

    /// Kaldi `LdaEstimate::ZeroAccumulators()`.
    pub fn zero_accumulators(&mut self) {
        self.zero_acc.iter_mut().for_each(|v| *v = 0.0);
        self.first_acc.fill(0.0);
        self.total_second_acc.set_zero();
    }

    /// Kaldi `LdaEstimate::Scale(f)`.
    pub fn scale(&mut self, f: f64) {
        self.zero_acc.iter_mut().for_each(|v| *v *= f);
        self.first_acc.mapv_inplace(|v| v * f);
        self.total_second_acc.scale(f);
    }

    /// Kaldi `LdaEstimate::Accumulate(data, class_id, weight)`.
    pub fn accumulate(&mut self, x: &[f32], class_id: usize, weight: f64) {
        assert!(class_id < self.num_classes(), "LDA class id out of range");
        assert_eq!(x.len(), self.dim(), "LDA feature dim mismatch");
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        self.zero_acc[class_id] += weight;
        let mut row = self.first_acc.row_mut(class_id);
        for (r, &v) in row.iter_mut().zip(xd.iter()) {
            *r += weight * v;
        }
        self.total_second_acc.add_vec2(weight, &xd);
    }

    /// Sum another accumulator into this one (Kaldi's `Read(..., add=true)`).
    pub fn add(&mut self, o: &Self) {
        assert_eq!(self.num_classes(), o.num_classes());
        assert_eq!(self.dim(), o.dim());
        for (a, b) in self.zero_acc.iter_mut().zip(o.zero_acc.iter()) {
            *a += *b;
        }
        self.first_acc += &o.first_acc;
        self.total_second_acc.add_sp(1.0, &o.total_second_acc);
    }

    /// Kaldi `LdaEstimate::GetStats`: total covariance, between-class
    /// covariance, total mean, and total count.
    fn get_stats(&self) -> (SpMat, SpMat, Vec<f64>, f64) {
        let dim = self.dim();
        let num_class = self.num_classes();
        let sum: f64 = self.tot_count();

        let mut total_covar = self.total_second_acc.clone();
        let mut total_mean = vec![0.0f64; dim];
        for c in 0..num_class {
            let row = self.first_acc.row(c);
            for d in 0..dim {
                total_mean[d] += row[d];
            }
        }
        for v in total_mean.iter_mut() {
            *v /= sum;
        }
        total_covar.scale(1.0 / sum);
        total_covar.add_vec2(-1.0, &total_mean);

        let mut between_covar = SpMat::zeros(dim);
        let mut class_mean = vec![0.0f64; dim];
        for c in 0..num_class {
            if self.zero_acc[c] != 0.0 {
                let row = self.first_acc.row(c);
                let inv = 1.0 / self.zero_acc[c];
                for d in 0..dim {
                    class_mean[d] = row[d] * inv;
                }
                between_covar.add_vec2(self.zero_acc[c] / sum, &class_mean);
            }
        }
        between_covar.add_vec2(-1.0, &total_mean);

        (total_covar, between_covar, total_mean, sum)
    }

    /// Kaldi `LdaEstimate::Estimate(opts, M, Mfull)`.
    ///
    /// Returns `(m, mfull)`: the `[dim, in]` (or `[dim, in+1]` when
    /// `remove_offset`) projection, and the corresponding full-rank
    /// `[in, in]` / `[in, in+1]` matrix.
    pub fn estimate(&self, opts: &LdaEstimateOptions) -> (Mat, Mat) {
        let dim = self.dim();
        let target_dim = opts.dim;
        assert!(target_dim > 0, "LDA target dim must be positive");
        assert!(target_dim <= dim, "LDA target dim exceeds feature dim");
        assert!(
            target_dim < self.num_classes() || opts.allow_large_dim,
            "LDA target dim >= num classes; set allow_large_dim"
        );

        let (total_covar, bc_covar, total_mean, count) = self.get_stats();

        // within-class covariance = total - between
        let mut wc_covar = total_covar;
        wc_covar.add_sp(-1.0, &bc_covar);

        let wc_covar_sqrt = match cholesky_lower(&wc_covar) {
            Some(l) => l,
            None => {
                // Kaldi: add 1e-3 * trace/dim to the diagonal and retry.
                let smooth = 1.0e-3 * wc_covar.trace() / wc_covar.dim() as f64;
                for i in 0..dim {
                    wc_covar.m[(i, i)] += smooth;
                }
                cholesky_lower(&wc_covar)
                    .expect("LDA within-class covariance not positive definite after smoothing")
            }
        };
        // Kaldi copies the triangular Cholesky factor into a full Matrix and
        // inverts it: wc_covar_sqrt_mat = L^{-1}.
        let wc_covar_sqrt_mat = invert(wc_covar_sqrt.as_ref());

        // tmp_sp = L^{-1} B L^{-T}   (Kaldi's AddMat2Sp(1.0, M, kNoTrans, B, 0.0))
        let mut tmp = SpMat::zeros(dim);
        {
            // First T = L^{-1} B, then tmp = T L^{-T}.
            let mut t = faer::Mat::<f64>::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let mut s = 0.0;
                    for k in 0..dim {
                        s += wc_covar_sqrt_mat[(i, k)] * bc_covar.m[(k, j)];
                    }
                    t[(i, j)] = s;
                }
            }
            for i in 0..dim {
                for j in 0..dim {
                    let mut s = 0.0;
                    for k in 0..dim {
                        s += t[(i, k)] * wc_covar_sqrt_mat[(j, k)];
                    }
                    tmp.m[(i, j)] = s;
                }
            }
            // Symmetrize exactly, as Kaldi's SpMatrix storage does.
            for i in 0..dim {
                for j in (i + 1)..dim {
                    let v = 0.5 * (tmp.m[(i, j)] + tmp.m[(j, i)]);
                    tmp.m[(i, j)] = v;
                    tmp.m[(j, i)] = v;
                }
            }
        }

        // Kaldi: Svd + SortSvd on the symmetric PSD tmp == descending eigen.
        let (svd_d, svd_u) = sym_eig_descending(&tmp);
        tracing::info!(
            count,
            sum_all = svd_d.iter().sum::<f64>(),
            sum_selected = svd_d[..target_dim].iter().sum::<f64>(),
            "LDA singular values"
        );

        // lda_mat = svd_u^T * wc_covar_sqrt_mat
        let mut lda_mat = vec![0.0f64; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += svd_u[(k, i)] * wc_covar_sqrt_mat[(k, j)];
                }
                lda_mat[i * dim + j] = s;
            }
        }

        // within_class_factor row scaling (not the normal code path).
        if opts.within_class_factor != 1.0 {
            for i in 0..dim {
                let old_var = 1.0 + svd_d[i];
                let new_var = opts.within_class_factor + svd_d[i];
                let scale = (new_var / old_var).sqrt();
                for j in 0..dim {
                    lda_mat[i * dim + j] *= scale;
                }
            }
        }

        let mut m: Mat =
            ndarray::Array2::from_shape_fn((target_dim, dim), |(i, j)| lda_mat[i * dim + j] as f32);
        let mut mfull: Mat =
            ndarray::Array2::from_shape_fn((dim, dim), |(i, j)| lda_mat[i * dim + j] as f32);

        if opts.remove_offset {
            m = add_mean_offset(&total_mean, &m);
            mfull = add_mean_offset(&total_mean, &mfull);
        }

        (m, mfull)
    }
}

/// Kaldi `LdaEstimate::AddMeanOffset`: append a column equal to
/// `-projection * mean`, so the projected data mean becomes zero.
fn add_mean_offset(mean_dbl: &[f64], projection: &Mat) -> Mat {
    let rows = projection.nrows();
    let cols = projection.ncols();
    debug_assert_eq!(cols, mean_dbl.len());
    let mut out: Mat = ndarray::Array2::zeros((rows, cols + 1));
    for i in 0..rows {
        let mut acc = 0.0f32;
        for j in 0..cols {
            let v = projection[[i, j]];
            out[[i, j]] = v;
            acc -= v * mean_dbl[j] as f32;
        }
        out[[i, cols]] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated classes along dim 0, noise along dim 1: the leading
    /// LDA direction must be dominated by dim 0.
    #[test]
    fn lda_finds_discriminative_direction() {
        let mut est = LdaEstimate::new(2, 2);
        for i in 0..50 {
            let jitter = ((i % 7) as f32 - 3.0) * 0.05;
            est.accumulate(&[-5.0 + jitter, ((i % 11) as f32 - 5.0)], 0, 1.0);
            est.accumulate(&[5.0 + jitter, ((i % 13) as f32 - 6.0)], 1, 1.0);
        }
        let opts = LdaEstimateOptions {
            dim: 1,
            allow_large_dim: false,
            ..Default::default()
        };
        let (m, mfull) = est.estimate(&opts);
        assert_eq!(m.shape(), &[1, 2]);
        assert_eq!(mfull.shape(), &[2, 2]);
        assert!(
            m[[0, 0]].abs() > 5.0 * m[[0, 1]].abs(),
            "leading direction should be dim 0: {m:?}"
        );
        // The full matrix's first row is the same as the reduced matrix's.
        assert!((m[[0, 0]] - mfull[[0, 0]]).abs() < 1e-6);
    }

    /// With `within_class_factor == 1.0`, the projection whitens the
    /// within-class covariance: `M W M^T == I`.
    #[test]
    fn lda_whitens_within_class_covariance() {
        let mut est = LdaEstimate::new(3, 3);
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32
        };
        let centers = [[0.0f32, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 4.0, 2.0]];
        for c in 0..3 {
            for _ in 0..200 {
                let x = [
                    centers[c][0] + next() * 2.0,
                    centers[c][1] + next() * 1.0,
                    centers[c][2] + next() * 0.5,
                ];
                est.accumulate(&x, c, 1.0);
            }
        }
        let opts = LdaEstimateOptions {
            dim: 2,
            ..Default::default()
        };
        let (m, _full) = est.estimate(&opts);

        let (total, between, _mean, _n) = est.get_stats();
        let mut within = total;
        within.add_sp(-1.0, &between);
        for a in 0..2 {
            for b in 0..2 {
                let mut acc = 0.0f64;
                for i in 0..3 {
                    for j in 0..3 {
                        acc += m[[a, i]] as f64 * within.get(i, j) * m[[b, j]] as f64;
                    }
                }
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-4, "M W M^T[{a},{b}] = {acc}");
            }
        }
    }

    #[test]
    fn remove_offset_zeroes_projected_mean() {
        // Non-degenerate within-class scatter: a purely collinear cloud makes
        // the within-class covariance singular, and the smoothed Cholesky then
        // yields a projection of magnitude ~1e7 whose f32 storage alone loses
        // several units of the projected mean. Kaldi has the same f32 output.
        let mut est = LdaEstimate::new(2, 2);
        for i in 0..40 {
            let j = (i % 5) as f32;
            let k = (i % 7) as f32;
            est.accumulate(&[10.0 + j, 20.0 - k], 0, 1.0);
            est.accumulate(&[14.0 + k, 24.0 - j], 1, 1.0);
        }
        let opts = LdaEstimateOptions {
            dim: 1,
            remove_offset: true,
            ..Default::default()
        };
        let (m, mfull) = est.estimate(&opts);
        assert_eq!(m.shape(), &[1, 3]);
        assert_eq!(mfull.shape(), &[2, 3]);
        let (_t, _b, mean, _n) = est.get_stats();
        // projected mean + offset column should be ~0
        let v = m[[0, 0]] as f64 * mean[0] + m[[0, 1]] as f64 * mean[1] + m[[0, 2]] as f64;
        assert!(v.abs() < 1e-3, "projected mean {v}");
    }

    #[test]
    fn add_and_scale_are_linear() {
        let mut a = LdaEstimate::new(2, 2);
        let mut b = LdaEstimate::new(2, 2);
        a.accumulate(&[1.0, 2.0], 0, 1.0);
        b.accumulate(&[3.0, 4.0], 1, 2.0);
        a.add(&b);
        assert_eq!(a.tot_count(), 3.0);
        a.scale(0.5);
        assert!((a.tot_count() - 1.5).abs() < 1e-12);
        a.zero_accumulators();
        assert_eq!(a.tot_count(), 0.0);
    }
}
