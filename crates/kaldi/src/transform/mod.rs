//! Feature-space transforms: LDA, MLLT (global STC), and fMLLR/CMLLR.
//!
//! Ports of `plans/kaldi/src/transform/lda-estimate.{h,cc}`, `mllt.{h,cc}`,
//! `fmllr-diag-gmm.{h,cc}` and the transform-composition half of
//! `transform-common.{h,cc}`.
//!
//! All accumulators keep their state in `f64`, exactly as Kaldi does, and the
//! transforms they produce are `f32` matrices in the shape the rest of the
//! crate expects: `[out, in]` for a linear transform, `[out, in + 1]` for an
//! affine one whose last column is the offset.

mod fmllr;
mod fmllr_update;
mod lda;
pub(crate) mod linalg;
mod mllt;

/// A transform matrix: `[out, in]` (linear) or `[out, in + 1]` (affine, with
/// the last column holding the offset).
pub type Mat = ndarray::Array2<f32>;

pub use fmllr::{FmllrDiagGmmAccs, FmllrOptions, FmllrUpdateType, identity_affine};
pub use fmllr_update::fmllr_aux_func_diag_gmm;
pub use lda::{LdaEstimate, LdaEstimateOptions};
pub use mllt::{MlltAccs, transform_means};

/// Kaldi `ComposeTransforms(a, b, b_is_affine, c)` -> `c = a * b`.
///
/// Three cases, exactly as in Kaldi:
/// * `a.cols == b.rows`: a plain product.
/// * `a` is affine (`a.cols == b.rows + 1`) and `b` is affine: `b` is extended
///   with a final row `0 .. 0 1`, so `a`'s offset column lands on `b`'s offset.
/// * `a` is affine and `b` is linear: `b` is extended by one row *and* one
///   column, with a 1 in the new corner, producing an affine result.
///
/// Panics on a genuine dimension mismatch (Kaldi's `KALDI_ERR`, which is fatal).
pub fn compose_transforms(a: &Mat, b: &Mat, b_is_affine: bool) -> Mat {
    assert!(
        b.nrows() != 0 && a.ncols() != 0,
        "compose_transforms: empty matrix"
    );
    if a.ncols() == b.nrows() {
        return a.dot(b);
    }
    if a.ncols() == b.nrows() + 1 {
        if b_is_affine {
            // Append the row 0 0 .. 0 1 to b.
            let mut b_ext: Mat = ndarray::Array2::zeros((b.nrows() + 1, b.ncols()));
            b_ext.slice_mut(ndarray::s![..b.nrows(), ..]).assign(b);
            b_ext[[b.nrows(), b.ncols() - 1]] = 1.0;
            return a.dot(&b_ext);
        }
        // Extend b by one row and one column, with a 1 in the corner.
        let mut b_ext: Mat = ndarray::Array2::zeros((b.nrows() + 1, b.ncols() + 1));
        b_ext.slice_mut(ndarray::s![..b.nrows(), ..b.ncols()]).assign(b);
        b_ext[[b.nrows(), b.ncols()]] = 1.0;
        return a.dot(&b_ext);
    }
    panic!(
        "compose_transforms: mismatched dimensions, a has {} columns and b has {} rows",
        a.ncols(),
        b.nrows()
    );
}

/// Kaldi `ApplyAffineTransform(xform, vec)`: `vec <- xform * [vec; 1]` in place
/// for a `[dim, dim + 1]` transform.
pub fn apply_affine_transform(xform: &Mat, v: &mut [f32]) {
    let dim = xform.nrows();
    assert!(dim > 0 && xform.ncols() == dim + 1 && v.len() == dim, "apply_affine_transform: bad dims");
    let src: Vec<f32> = v.to_vec();
    for i in 0..dim {
        let mut acc = xform[[i, dim]];
        for j in 0..dim {
            acc += xform[[i, j]] * src[j];
        }
        v[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, vals: &[f32]) -> Mat {
        ndarray::Array2::from_shape_vec((rows, cols), vals.to_vec()).unwrap()
    }

    #[test]
    fn compose_linear_times_linear() {
        let a = mat(2, 3, &[1.0, 0.0, 2.0, 0.0, 1.0, 3.0]);
        let b = mat(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = compose_transforms(&a, &b, false);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[[0, 0]], 3.0); // 1*1 + 0*0 + 2*1
        assert_eq!(c[[0, 1]], 2.0);
        assert_eq!(c[[1, 0]], 3.0);
        assert_eq!(c[[1, 1]], 4.0);
    }

    #[test]
    fn compose_affine_times_affine() {
        // a: 1-D affine, doubles and adds 5. b: 1-D affine, triples and adds 1.
        let a = mat(1, 2, &[2.0, 5.0]);
        let b = mat(1, 2, &[3.0, 1.0]);
        let c = compose_transforms(&a, &b, true);
        assert_eq!(c.shape(), &[1, 2]);
        // x -> 3x + 1 -> 2(3x+1) + 5 = 6x + 7
        assert_eq!(c[[0, 0]], 6.0);
        assert_eq!(c[[0, 1]], 7.0);
        let mut v = [4.0f32];
        apply_affine_transform(&c, &mut v);
        assert_eq!(v[0], 31.0);
    }

    #[test]
    fn compose_affine_times_linear_yields_affine() {
        // a is affine [1, 2]; b is a linear [1, 1] scaling by 3.
        let a = mat(1, 2, &[2.0, 5.0]);
        let b = mat(1, 1, &[3.0]);
        let c = compose_transforms(&a, &b, false);
        assert_eq!(c.shape(), &[1, 2]);
        // x -> 3x, then 2*(3x) + 5 = 6x + 5
        assert_eq!(c[[0, 0]], 6.0);
        assert_eq!(c[[0, 1]], 5.0);
    }

    /// The LDA+MLLT composition path: a [40, 118] LDA-shaped matrix composed
    /// with a square MLLT matrix on the left keeps the affine column.
    #[test]
    fn compose_mllt_with_lda_shape() {
        let mllt: Mat = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| {
            if i == j { 1.0 } else { 0.1 }
        });
        let lda: Mat = ndarray::Array2::from_shape_fn((4, 7), |(i, j)| (i + j) as f32 * 0.1);
        let c = compose_transforms(&mllt, &lda, true);
        assert_eq!(c.shape(), &[4, 7]);
        // Row 0 of c should be mllt row 0 dotted with lda's columns.
        for j in 0..7 {
            let mut want = 0.0f32;
            for k in 0..4 {
                want += mllt[[0, k]] * lda[[k, j]];
            }
            assert!((c[[0, j]] - want).abs() < 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "mismatched dimensions")]
    fn compose_rejects_mismatch() {
        let a = mat(2, 5, &[0.0; 10]);
        let b = mat(3, 2, &[0.0; 6]);
        let _ = compose_transforms(&a, &b, false);
    }

    #[test]
    fn apply_affine_matches_manual() {
        let x = mat(2, 3, &[2.0, 0.0, 1.0, 0.0, 3.0, -1.0]);
        let mut v = [5.0f32, 7.0];
        apply_affine_transform(&x, &mut v);
        assert_eq!(v, [11.0, 20.0]);
    }
}
