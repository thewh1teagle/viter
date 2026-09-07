//! Small dense-linear-algebra helpers shared by the LDA / MLLT / fMLLR ports.
//!
//! Kaldi does all of this work in `double`; so do we. Everything here operates
//! on `faer::Mat<f64>` (column-major) or plain row-major `Vec<f64>` buffers,
//! and only the final transforms are narrowed to `f32`.
//!
//! Kaldi's `SpMatrix<double>` is a packed symmetric matrix. We represent the
//! same thing as a dense `Mat<f64>` that we keep symmetric by construction;
//! that costs a little memory but keeps the code honest and lets faer's
//! self-adjoint routines be used directly.

use std::sync::Once;

use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, MatRef, Side};
use rand::RngExt;

/// Run faer's solvers single-threaded.
///
/// Every matrix here is tiny (the feature dim, ≤ ~120), and faer's high-level
/// solvers otherwise fan each factorization out over rayon: on a 40×40 inverse
/// the thread hand-off costs more than the arithmetic, and fMLLR does 1,600 of
/// them per speaker. The one large product, the CPU scoring matmul, passes its
/// own `Par` explicitly and is unaffected.
fn sequential() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| faer::set_global_parallelism(faer::Par::Seq));
}

/// `ndarray` (row-major, `[r, c]`) -> `faer` `Mat<f64>`.
pub(crate) fn nd_to_faer(a: &ndarray::Array2<f32>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]] as f64)
}

/// `faer` `Mat<f64>` -> `ndarray::Array2<f32>` (row-major).
pub(crate) fn faer_to_nd(a: MatRef<'_, f64>) -> ndarray::Array2<f32> {
    ndarray::Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| a[(i, j)] as f32)
}

/// `faer` `Mat<f64>` -> `ndarray::Array2<f64>` (row-major).
#[cfg(test)]
pub(crate) fn faer_to_nd64(a: MatRef<'_, f64>) -> ndarray::Array2<f64> {
    ndarray::Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| a[(i, j)])
}

/// A symmetric accumulator held as a full dense matrix.
///
/// Mirrors Kaldi's `SpMatrix<double>` for the operations the transform code
/// needs: `AddVec2`, `AddSp`, `Scale`, `Invert`, and quadratic forms.
#[derive(Clone, Debug)]
pub(crate) struct SpMat {
    pub(crate) m: Mat<f64>,
}

impl SpMat {
    pub(crate) fn zeros(dim: usize) -> Self {
        Self {
            m: Mat::zeros(dim, dim),
        }
    }

    pub(crate) fn dim(&self) -> usize {
        self.m.nrows()
    }

    #[inline]
    pub(crate) fn get(&self, i: usize, j: usize) -> f64 {
        self.m[(i, j)]
    }

    #[inline]
    pub(crate) fn add(&mut self, i: usize, j: usize, v: f64) {
        self.m[(i, j)] += v;
        if i != j {
            self.m[(j, i)] += v;
        }
    }

    /// Kaldi `SpMatrix::AddVec2(alpha, v)`: `self += alpha * v v^T`.
    pub(crate) fn add_vec2(&mut self, alpha: f64, v: &[f64]) {
        let d = self.dim();
        debug_assert_eq!(v.len(), d);
        for i in 0..d {
            let ai = alpha * v[i];
            if ai == 0.0 {
                continue;
            }
            for j in 0..d {
                self.m[(i, j)] += ai * v[j];
            }
        }
    }

    /// Kaldi `SpMatrix::AddSp(alpha, other)`.
    pub(crate) fn add_sp(&mut self, alpha: f64, other: &SpMat) {
        debug_assert_eq!(self.dim(), other.dim());
        let d = self.dim();
        for i in 0..d {
            for j in 0..d {
                self.m[(i, j)] += alpha * other.m[(i, j)];
            }
        }
    }

    pub(crate) fn scale(&mut self, alpha: f64) {
        let d = self.dim();
        for i in 0..d {
            for j in 0..d {
                self.m[(i, j)] *= alpha;
            }
        }
    }

    pub(crate) fn set_zero(&mut self) {
        self.scale(0.0);
    }

    pub(crate) fn trace(&self) -> f64 {
        (0..self.dim()).map(|i| self.m[(i, i)]).sum()
    }

    /// `out = self * v` (symmetric matrix-vector product).
    pub(crate) fn mul_vec(&self, v: &[f64], out: &mut [f64]) {
        let d = self.dim();
        debug_assert_eq!(v.len(), d);
        debug_assert_eq!(out.len(), d);
        for i in 0..d {
            let mut s = 0.0;
            for j in 0..d {
                s += self.m[(i, j)] * v[j];
            }
            out[i] = s;
        }
    }

    /// `v^T self v`.
    pub(crate) fn quad_form(&self, v: &[f64]) -> f64 {
        let d = self.dim();
        debug_assert_eq!(v.len(), d);
        let mut total = 0.0;
        for i in 0..d {
            let mut s = 0.0;
            for j in 0..d {
                s += self.m[(i, j)] * v[j];
            }
            total += v[i] * s;
        }
        total
    }

    /// Kaldi `SpMatrix::Invert()`. Symmetric inverse; falls back to a general
    /// LU inverse when the matrix is not positive definite (Kaldi uses an
    /// LDL^T-style inverse that also handles indefinite matrices).
    pub(crate) fn inverted(&self) -> SpMat {
        sequential();
        let inv = match self.m.llt(Side::Lower) {
            Ok(llt) => llt.inverse(),
            Err(_) => self.m.partial_piv_lu().inverse(),
        };
        // Re-symmetrize: numerically the two triangles can drift apart, and
        // Kaldi's packed storage makes them identical by construction.
        let d = self.dim();
        let mut out = Mat::<f64>::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                out[(i, j)] = 0.5 * (inv[(i, j)] + inv[(j, i)]);
            }
        }
        SpMat { m: out }
    }
}

/// Kaldi `TpMatrix::Cholesky(sp)`: lower-triangular `L` with `L L^T == a`.
///
/// Returns `None` when the matrix is not positive definite, so the caller can
/// apply Kaldi's diagonal-smoothing retry.
pub(crate) fn cholesky_lower(a: &SpMat) -> Option<Mat<f64>> {
    sequential();
    let llt = a.m.llt(Side::Lower).ok()?;
    let l = llt.L();
    let d = a.dim();
    // faer's `L()` view may carry garbage above the diagonal; zero it.
    Some(Mat::from_fn(
        d,
        d,
        |i, j| if j <= i { l[(i, j)] } else { 0.0 },
    ))
}

/// General square-matrix inverse (Kaldi `Matrix::Invert()` without logdet).
pub(crate) fn invert(a: MatRef<'_, f64>) -> Mat<f64> {
    sequential();
    a.partial_piv_lu().inverse()
}

/// Kaldi `Matrix::Invert(&logdet)`: returns `(inverse, log|det|)`.
#[cfg(test)]
pub(crate) fn invert_with_logdet(a: MatRef<'_, f64>) -> (Mat<f64>, f64) {
    sequential();
    let lu = a.partial_piv_lu();
    let inv = lu.inverse();
    (inv, log_abs_det(a))
}

/// `log |det(a)|` via an LU factorization, matching Kaldi's `Matrix::LogDet`
/// magnitude (Kaldi returns the sign separately; every caller here only uses
/// the magnitude, since fMLLR/MLLT objectives take `log|det|`).
pub(crate) fn log_abs_det(a: MatRef<'_, f64>) -> f64 {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    // Plain partial-pivoting Gaussian elimination on a scratch copy: this keeps
    // the accumulation in log-space and avoids overflow for large dims.
    let mut w = a.to_owned();
    let mut logdet = 0.0f64;
    for k in 0..n {
        // pivot
        let mut piv = k;
        let mut best = w[(k, k)].abs();
        for i in (k + 1)..n {
            let v = w[(i, k)].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best == 0.0 {
            return f64::NEG_INFINITY;
        }
        if piv != k {
            for j in 0..n {
                let t = w[(k, j)];
                w[(k, j)] = w[(piv, j)];
                w[(piv, j)] = t;
            }
        }
        let pivot = w[(k, k)];
        logdet += pivot.abs().ln();
        for i in (k + 1)..n {
            let f = w[(i, k)] / pivot;
            if f == 0.0 {
                continue;
            }
            for j in k..n {
                w[(i, j)] -= f * w[(k, j)];
            }
        }
    }
    logdet
}

/// Self-adjoint eigendecomposition, eigenvalues sorted **descending** with the
/// matching eigenvector columns permuted alongside.
///
/// This is what Kaldi's `Matrix::Svd` + `SortSvd` produce when the input is
/// symmetric positive semi-definite (as it always is at the LDA call site:
/// `L^{-1} B L^{-T}`), where the singular values coincide with the eigenvalues
/// and `U` with the eigenvectors.
///
/// Returns `(eigenvalues[dim], eigenvectors as columns [dim, dim])`.
pub(crate) fn sym_eig_descending(a: &SpMat) -> (Vec<f64>, Mat<f64>) {
    let d = a.dim();
    sequential();
    let eig =
        a.m.self_adjoint_eigen(Side::Lower)
            .expect("self-adjoint eigendecomposition failed");
    let s = eig.S();
    let u = eig.U();
    let mut order: Vec<usize> = (0..d).collect();
    // faer returns nondecreasing order; Kaldi's SortSvd is descending. Sort by
    // value descending with a stable tie-break on the original index so the
    // result is deterministic.
    order.sort_by(|&x, &y| {
        s[y].partial_cmp(&s[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });
    let vals: Vec<f64> = order.iter().map(|&k| s[k]).collect();
    let vecs = Mat::from_fn(d, d, |i, j| u[(i, order[j])]);
    (vals, vecs)
}

/// Kaldi's `RandPrune(post, prune_thresh)`: an expectation-preserving
/// randomized pruning of small posteriors.
pub(crate) fn rand_prune(post: f64, prune_thresh: f64, rng: &mut impl rand::Rng) -> f64 {
    debug_assert!(prune_thresh >= 0.0);
    if post == 0.0 || post.abs() >= prune_thresh {
        return post;
    }
    let sign = if post >= 0.0 { 1.0 } else { -1.0 };
    // Kaldi: RandUniform() is in (0, 1); keep with probability |post|/thresh.
    let u: f64 = rng.random::<f64>();
    if u <= post.abs() / prune_thresh {
        sign * prune_thresh
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn sp_add_vec2_and_quad_form() {
        let mut s = SpMat::zeros(3);
        s.add_vec2(2.0, &[1.0, 2.0, 3.0]);
        approx(s.get(0, 0), 2.0, 1e-12);
        approx(s.get(0, 2), 6.0, 1e-12);
        approx(s.get(2, 2), 18.0, 1e-12);
        // v^T (2 w w^T) v = 2 (w.v)^2
        let v = [1.0, 1.0, 1.0];
        approx(s.quad_form(&v), 2.0 * 36.0, 1e-9);
        approx(s.trace(), 2.0 + 8.0 + 18.0, 1e-12);
    }

    #[test]
    fn sp_invert_roundtrip() {
        let mut s = SpMat::zeros(3);
        s.add_vec2(1.0, &[1.0, 0.5, 0.2]);
        s.add_vec2(1.0, &[0.1, 1.0, -0.3]);
        s.add_vec2(1.0, &[0.0, 0.2, 1.0]);
        for i in 0..3 {
            s.add(i, i, 0.5);
        }
        let inv = s.inverted();
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += s.get(i, k) * inv.get(k, j);
                }
                approx(acc, if i == j { 1.0 } else { 0.0 }, 1e-9);
            }
        }
    }

    #[test]
    fn cholesky_reconstructs() {
        let mut s = SpMat::zeros(4);
        for k in 0..6 {
            let v: Vec<f64> = (0..4)
                .map(|i| ((i * 7 + k * 3) % 5) as f64 * 0.3 + 0.1)
                .collect();
            s.add_vec2(1.0, &v);
        }
        for i in 0..4 {
            s.add(i, i, 1.0);
        }
        let l = cholesky_lower(&s).expect("pos def");
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += l[(i, k)] * l[(j, k)];
                }
                approx(acc, s.get(i, j), 1e-9);
            }
            for j in (i + 1)..4 {
                approx(l[(i, j)], 0.0, 0.0);
            }
        }
    }

    #[test]
    fn logdet_matches_explicit_2x2_and_3x3() {
        let a = Mat::from_fn(2, 2, |i, j| [[3.0, 1.0], [2.0, 4.0]][i][j]);
        approx(log_abs_det(a.as_ref()), 10.0f64.ln(), 1e-10);
        let b = Mat::from_fn(3, 3, |i, j| {
            [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]][i][j]
        });
        // det = 2*(6) - 0 + 1*(-3) = 9
        approx(log_abs_det(b.as_ref()), 9.0f64.ln(), 1e-10);
    }

    #[test]
    fn invert_with_logdet_agrees() {
        let a = Mat::from_fn(3, 3, |i, j| {
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]][i][j]
        });
        let (inv, ld) = invert_with_logdet(a.as_ref());
        approx(ld, log_abs_det(a.as_ref()), 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += a[(i, k)] * inv[(k, j)];
                }
                approx(acc, if i == j { 1.0 } else { 0.0 }, 1e-9);
            }
        }
    }

    #[test]
    fn sym_eig_is_descending_and_orthonormal() {
        let mut s = SpMat::zeros(3);
        s.add_vec2(4.0, &[1.0, 0.0, 0.0]);
        s.add_vec2(9.0, &[0.0, 1.0, 0.0]);
        s.add_vec2(1.0, &[0.0, 0.0, 1.0]);
        let (vals, vecs) = sym_eig_descending(&s);
        approx(vals[0], 9.0, 1e-9);
        approx(vals[1], 4.0, 1e-9);
        approx(vals[2], 1.0, 1e-9);
        // columns orthonormal
        for j in 0..3 {
            let mut n = 0.0;
            for i in 0..3 {
                n += vecs[(i, j)] * vecs[(i, j)];
            }
            approx(n, 1.0, 1e-9);
        }
    }

    #[test]
    fn rand_prune_preserves_expectation() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let post = 0.5;
        let thresh = 4.0;
        let n = 40000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += rand_prune(post, thresh, &mut rng);
        }
        approx(sum / n as f64, post, 0.05);
        // above threshold is untouched
        approx(rand_prune(5.0, 4.0, &mut rng), 5.0, 0.0);
        approx(rand_prune(0.0, 4.0, &mut rng), 0.0, 0.0);
    }

    #[test]
    fn nd_faer_roundtrip() {
        let a = ndarray::Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f32);
        let f = nd_to_faer(&a);
        let back = faer_to_nd(f.as_ref());
        assert_eq!(a, back);
        assert_eq!(faer_to_nd64(f.as_ref())[[1, 2]], 5.0);
    }
}
