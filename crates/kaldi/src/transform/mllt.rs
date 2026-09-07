//! Port of `plans/kaldi/src/transform/mllt.{h,cc}`.
//!
//! Maximum Likelihood Linear Transform (global Semi-Tied Covariance). The
//! resulting transform left-multiplies the feature vector.

use super::Mat;
use super::linalg::{SpMat, invert, nd_to_faer, rand_prune};
use crate::gmm::DiagGmm;

/// Kaldi's fixed inner iteration count in `MlltAccs::Update`.
const MLLT_NUM_ITERS: usize = 200;

/// Kaldi `MlltAccs`: a count `beta` and `dim` symmetric `dim x dim` matrices.
#[derive(Clone, Debug)]
pub struct MlltAccs {
    /// `rand_prune_`: randomized-pruning threshold (MFA `random_prune`, 4.0).
    rand_prune: f64,
    /// `beta_`: total data count.
    beta: f64,
    /// `G_`: `dim` matrices of size `dim x dim`.
    g: Vec<SpMat>,
}

impl MlltAccs {
    /// Kaldi `MlltAccs::Init(dim, rand_prune)`.
    pub fn new(dim: usize, rand_prune: f64) -> Self {
        assert!(dim > 0, "MLLT dim must be positive");
        assert!(rand_prune >= 0.0, "MLLT rand_prune must be non-negative");
        Self {
            rand_prune,
            beta: 0.0,
            g: (0..dim).map(|_| SpMat::zeros(dim)).collect(),
        }
    }

    pub fn dim(&self) -> usize {
        self.g.len()
    }

    /// Total accumulated count (`beta_`).
    pub fn count(&self) -> f64 {
        self.beta
    }

    /// Kaldi `MlltAccs::AccumulateFromPosteriors(gmm, data, posteriors)`.
    ///
    /// For each mixture component with a (randomly pruned) posterior, the
    /// offset `mu_i - x` contributes `inv_var(i, j) * posterior` to `G[j]`.
    pub fn accumulate_from_posteriors(
        &mut self,
        gmm: &DiagGmm,
        x: &[f32],
        posteriors: &[f32],
        rng: &mut impl rand::Rng,
    ) {
        let dim = x.len();
        assert_eq!(dim, gmm.dim(), "MLLT: data dim != gmm dim");
        assert_eq!(dim, self.dim(), "MLLT: data dim != accumulator dim");
        assert_eq!(
            posteriors.len(),
            gmm.num_gauss(),
            "MLLT: posterior count mismatch"
        );

        let means_invvars = &gmm.means_invvars;
        let inv_vars = &gmm.inv_vars;
        let mut offset = vec![0.0f64; dim];
        let mut tmp = SpMat::zeros(dim);
        let mut this_beta = 0.0f64;

        for i in 0..posteriors.len() {
            let posterior = rand_prune(posteriors[i] as f64, self.rand_prune, rng);
            if posterior == 0.0 {
                continue;
            }
            // mean = mean_invvar / inv_var, then offset = mean - data.
            for j in 0..dim {
                let mean = (means_invvars[[i, j]] / inv_vars[[i, j]]) as f64;
                offset[j] = mean - x[j] as f64;
            }
            tmp.set_zero();
            tmp.add_vec2(1.0, &offset);
            for j in 0..dim {
                let scale = inv_vars[[i, j]] as f64 * posterior;
                self.g[j].add_sp(scale, &tmp);
            }
            this_beta += posterior;
        }
        self.beta += this_beta;
    }

    /// Kaldi `MlltAccs::AccumulateFromGmm(gmm, data, weight)`; returns the GMM
    /// log-likelihood of the frame.
    pub fn accumulate_from_gmm(
        &mut self,
        gmm: &DiagGmm,
        x: &[f32],
        weight: f32,
        rng: &mut impl rand::Rng,
    ) -> f32 {
        let mut posteriors = Vec::new();
        let loglike = gmm.component_posteriors(x, &mut posteriors);
        for p in posteriors.iter_mut() {
            *p *= weight;
        }
        self.accumulate_from_posteriors(gmm, x, &posteriors, rng);
        loglike
    }

    /// Sum another accumulator into this one.
    pub fn add(&mut self, o: &Self) {
        assert_eq!(self.dim(), o.dim(), "MLLT: summing accs of different size");
        self.beta += o.beta;
        for (a, b) in self.g.iter_mut().zip(o.g.iter()) {
            a.add_sp(1.0, b);
        }
    }

    /// Kaldi `MlltAccs::Update(M, objf_impr, count)`.
    ///
    /// `mat` must be `[dim, dim]` and non-singular on entry (typically the
    /// unit matrix, or the previous MLLT estimate). Returns
    /// `(objf improvement, count)`.
    pub fn update(&self, mat: &mut Mat) -> (f64, f64) {
        mllt_update(self.beta, &self.g, mat)
    }
}

/// Static form of Kaldi `MlltAccs::Update(beta, G, M, ...)`.
///
/// Row-by-row coordinate ascent: for each row `i`, the objective is
/// `beta log|row . cofactor| - 0.5 row^T G_i row`, maximized in closed form by
/// `row = G_i^{-1} c sqrt(beta / (c^T G_i^{-1} c))` (Gales, eq. 22).
pub(crate) fn mllt_update(beta: f64, g: &[SpMat], mat: &mut Mat) -> (f64, f64) {
    let dim = g.len();
    assert!(dim != 0, "MLLT update: no stats");
    assert_eq!(mat.nrows(), dim, "MLLT update: M has wrong number of rows");
    assert_eq!(
        mat.ncols(),
        dim,
        "MLLT update: M has wrong number of columns"
    );

    if beta < 10.0 * dim as f64 {
        if beta > 2.0 * dim as f64 {
            tracing::warn!(beta, "Mllt::update, very small count");
        } else {
            tracing::warn!(beta, "Mllt::update, insufficient count");
        }
    }

    // M as a plain row-major f64 buffer; rows are updated in place.
    let mut m = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            m[i * dim + j] = mat[[i, j]] as f64;
        }
    }

    let ginv: Vec<SpMat> = g.iter().map(|s| s.inverted()).collect();

    let mut tot_objf_impr = 0.0f64;
    let mut cofactor = vec![0.0f64; dim];
    let mut ginv_c = vec![0.0f64; dim];
    let mut row_buf = vec![0.0f64; dim];

    for p in 0..MLLT_NUM_ITERS {
        for i in 0..dim {
            // cofactor row i = row i of (M^{-1})^T, i.e. column i of M^{-1}.
            // (Kaldi: Minv = M; Minv.Invert(); Minv.Transpose(); take row i.)
            let m_faer = faer::Mat::from_fn(dim, dim, |r, c| m[r * dim + c]);
            let minv = invert(m_faer.as_ref());
            for j in 0..dim {
                cofactor[j] = minv[(j, i)];
            }

            row_buf.copy_from_slice(&m[i * dim..(i + 1) * dim]);
            let dot_before: f64 = row_buf
                .iter()
                .zip(cofactor.iter())
                .map(|(a, b)| a * b)
                .sum();
            let objf_before = beta * dot_before.abs().ln() - 0.5 * g[i].quad_form(&row_buf);

            // row = sqrt(beta / (c^T Ginv_i c)) * Ginv_i c
            ginv[i].mul_vec(&cofactor, &mut ginv_c);
            let denom: f64 = ginv_c.iter().zip(cofactor.iter()).map(|(a, b)| a * b).sum();
            let scale = (beta / denom).sqrt();
            for j in 0..dim {
                row_buf[j] = scale * ginv_c[j];
            }
            m[i * dim..(i + 1) * dim].copy_from_slice(&row_buf);

            let dot_after: f64 = row_buf
                .iter()
                .zip(cofactor.iter())
                .map(|(a, b)| a * b)
                .sum();
            let objf_after = beta * dot_after.abs().ln() - 0.5 * g[i].quad_form(&row_buf);
            if objf_after < objf_before - objf_before.abs() * 0.00001 {
                // CONTRACT-DEVIATION: Kaldi calls KALDI_ERR (fatal) here; we
                // log an error and keep going, since the aligner must not abort
                // a whole training run on one bad MLLT row.
                tracing::error!(
                    objf_before,
                    objf_after,
                    row = i,
                    "objective decrease in MLLT update"
                );
            }
            tot_objf_impr += objf_after - objf_before;
        }
        if p < 10 || p % 10 == 0 {
            tracing::debug!(
                iter = p,
                per_frame = tot_objf_impr / beta,
                beta,
                "MLLT objective improvement"
            );
        }
    }

    for i in 0..dim {
        for j in 0..dim {
            mat[[i, j]] = m[i * dim + j] as f32;
        }
    }
    (tot_objf_impr, beta)
}

/// Kaldi `AmDiagGmm` mean transformation (`gmm-transform-means`): for every
/// Gaussian of every pdf, `mean' = M * mean` (or `M * [mean; 1]` when `M` is
/// affine), then recompute the gconsts.
pub fn transform_means(am: &mut crate::gmm::AmDiagGmm, mat: &Mat) {
    let dim = am.dim();
    let rows = mat.nrows();
    let cols = mat.ncols();
    assert_eq!(
        rows, dim,
        "transform_means: transform has wrong number of rows"
    );
    assert!(
        cols == dim || cols == dim + 1,
        "transform_means: transform must be [dim, dim] or [dim, dim+1]"
    );
    let m = nd_to_faer(mat);

    for p in 0..am.num_pdfs() {
        let gmm = am.pdf_mut(p as crate::types::PdfId);
        let means = gmm.means();
        let vars = gmm.vars();
        let num_gauss = means.nrows();
        let mut new_means = ndarray::Array2::<f32>::zeros((num_gauss, dim));
        for gidx in 0..num_gauss {
            for i in 0..dim {
                let mut acc = if cols == dim + 1 { m[(i, dim)] } else { 0.0 };
                for j in 0..dim {
                    acc += m[(i, j)] * means[[gidx, j]] as f64;
                }
                new_means[[gidx, i]] = acc as f32;
            }
        }
        gmm.set_means_and_vars(&new_means, &vars);
    }
    am.compute_gconsts();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn eye(dim: usize) -> Mat {
        ndarray::Array2::from_shape_fn((dim, dim), |(i, j)| if i == j { 1.0 } else { 0.0 })
    }

    fn make_gmm(num_gauss: usize, dim: usize) -> DiagGmm {
        let mut means = ndarray::Array2::<f32>::zeros((num_gauss, dim));
        let mut vars = ndarray::Array2::<f32>::zeros((num_gauss, dim));
        for g in 0..num_gauss {
            for d in 0..dim {
                means[[g, d]] = (g as f32) * 1.5 - (d as f32) * 0.4;
                vars[[g, d]] = 0.6 + 0.2 * ((g + d) % 3) as f32;
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

    #[test]
    fn accumulate_is_positive_definite_and_counts() {
        let dim = 4;
        let gmm = make_gmm(3, dim);
        let mut accs = MlltAccs::new(dim, 0.0); // exact, no pruning
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
        let mut post = Vec::new();
        for t in 0..60 {
            let x: Vec<f32> = (0..dim)
                .map(|d| ((t * 7 + d * 3) % 11) as f32 * 0.3)
                .collect();
            gmm.component_posteriors(&x, &mut post);
            accs.accumulate_from_posteriors(&gmm, &x, &post, &mut rng);
        }
        // posteriors sum to 1 per frame, so beta == num frames.
        assert!(
            (accs.count() - 60.0).abs() < 1e-3,
            "beta = {}",
            accs.count()
        );
        // each G is PSD
        for j in 0..dim {
            let v: Vec<f64> = (0..dim).map(|k| (k as f64 + 1.0) * 0.3).collect();
            assert!(accs.g[j].quad_form(&v) >= 0.0);
        }
    }

    #[test]
    fn update_increases_objective_and_is_deterministic() {
        let dim = 3;
        let gmm = make_gmm(4, dim);
        let mut accs = MlltAccs::new(dim, 0.0);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
        let mut post = Vec::new();
        for t in 0..500 {
            let x: Vec<f32> = (0..dim)
                .map(|d| (((t * 13 + d * 5) % 17) as f32 - 8.0) * 0.4)
                .collect();
            gmm.component_posteriors(&x, &mut post);
            accs.accumulate_from_posteriors(&gmm, &x, &post, &mut rng);
        }
        let mut m = eye(dim);
        let (impr, count) = accs.update(&mut m);
        assert!((count - accs.count()).abs() < 1e-9);
        assert!(impr >= -1e-6, "objective must not decrease overall: {impr}");
        // Deterministic: same stats, same start => same result.
        let mut m2 = eye(dim);
        let (impr2, _) = accs.update(&mut m2);
        assert!((impr - impr2).abs() < 1e-9);
        for i in 0..dim {
            for j in 0..dim {
                assert!((m[[i, j]] - m2[[i, j]]).abs() < 1e-6);
            }
        }
        // A converged MLLT matrix is non-singular.
        let f = nd_to_faer(&m);
        assert!(super::super::linalg::log_abs_det(f.as_ref()).is_finite());
    }

    #[test]
    fn add_sums_stats() {
        let dim = 2;
        let gmm = make_gmm(2, dim);
        let mut a = MlltAccs::new(dim, 0.0);
        let mut b = MlltAccs::new(dim, 0.0);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
        let mut post = Vec::new();
        for t in 0..10 {
            let x = [t as f32 * 0.2, 1.0 - t as f32 * 0.1];
            gmm.component_posteriors(&x, &mut post);
            a.accumulate_from_posteriors(&gmm, &x, &post, &mut rng);
            b.accumulate_from_posteriors(&gmm, &x, &post, &mut rng);
        }
        let before = a.g[0].get(0, 0);
        let bg = b.g[0].get(0, 0);
        a.add(&b);
        assert!((a.g[0].get(0, 0) - (before + bg)).abs() < 1e-9);
        assert!((a.count() - 20.0).abs() < 1e-3);
    }

    #[test]
    fn transform_means_applies_linear_and_affine() {
        let dim = 3;
        let proto = make_gmm(2, dim);
        let mut am = crate::gmm::AmDiagGmm::init(&proto, 2);
        let before = am.pdf(0).means();

        // Linear: scale by 2.
        let mut lin = eye(dim);
        for i in 0..dim {
            lin[[i, i]] = 2.0;
        }
        transform_means(&mut am, &lin);
        let after = am.pdf(0).means();
        for g in 0..before.nrows() {
            for d in 0..dim {
                assert!((after[[g, d]] - 2.0 * before[[g, d]]).abs() < 1e-4);
            }
        }
        // Variances are untouched by a mean transform.
        for (a, b) in am.pdf(0).vars().iter().zip(proto.vars().iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        // Affine: identity plus an offset of 1 in every dim.
        let mut aff: Mat = ndarray::Array2::zeros((dim, dim + 1));
        for i in 0..dim {
            aff[[i, i]] = 1.0;
            aff[[i, dim]] = 1.0;
        }
        let pre = am.pdf(1).means();
        transform_means(&mut am, &aff);
        let post = am.pdf(1).means();
        for g in 0..pre.nrows() {
            for d in 0..dim {
                assert!((post[[g, d]] - (pre[[g, d]] + 1.0)).abs() < 1e-4);
            }
        }
    }
}
