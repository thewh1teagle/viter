//! The fMLLR objective function and its three update algorithms, split out of
//! `fmllr.rs` to keep both files under the project's 700-line limit.
//!
//! Ports of `ComputeFmllrMatrixDiagGmmFull` / `...Diagonal` / `...Offset`,
//! `FmllrInnerUpdate` and `FmllrAuxFuncDiagGmm` from
//! `plans/kaldi/src/transform/fmllr-diag-gmm.cc`.

use super::Mat;
use super::fmllr::FmllrDiagGmmAccs;
use super::linalg::{SpMat, invert_with_logdet, log_abs_det};

/// Kaldi `FmllrAuxFuncDiagGmm(xform, stats)`:
/// `beta log|A| + tr(W^T K) - 0.5 sum_d w_d^T G_d w_d`.
pub(super) fn fmllr_aux_func(xform: &[f64], stats: &FmllrDiagGmmAccs) -> f64 {
    let dim = stats.dim;
    let a = faer::Mat::from_fn(dim, dim, |i, j| xform[i * (dim + 1) + j]);
    let mut obj = stats.beta * log_abs_det(a.as_ref());
    // tr(xform^T K) == elementwise dot product.
    for i in 0..dim {
        for j in 0..=dim {
            obj += xform[i * (dim + 1) + j] * stats.k_at(i, j);
        }
    }
    let mut row_g = vec![0.0f64; dim + 1];
    for d in 0..dim {
        let row = &xform[d * (dim + 1)..(d + 1) * (dim + 1)];
        stats.g[d].mul_vec(row, &mut row_g);
        let q: f64 = row_g.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        obj -= 0.5 * q;
    }
    obj
}

/// Public form of Kaldi's `FmllrAuxFuncDiagGmm`.
///
/// Any pending single-frame stats are folded in first (on a scratch copy), so
/// this agrees with `count()` and with `update()`, both of which commit.
pub fn fmllr_aux_func_diag_gmm(xform: &Mat, stats: &FmllrDiagGmmAccs) -> f64 {
    let dim = stats.dim;
    assert_eq!(xform.nrows(), dim);
    assert_eq!(xform.ncols(), dim + 1);
    let mut stats_c = stats.clone();
    stats_c.commit_single_frame_stats();
    let stats = &stats_c;
    let flat: Vec<f64> = (0..dim)
        .flat_map(|i| (0..=dim).map(move |j| (i, j)))
        .map(|(i, j)| xform[[i, j]] as f64)
        .collect();
    fmllr_aux_func(&flat, stats)
}

/// Kaldi `FmllrInnerUpdate(inv_G, k, beta, row, transform)`.
///
/// Re-estimates one row of the transform in closed form: the auxf along the
/// direction of the row's cofactor is quadratic-plus-log, so the optimal step
/// solves `e1 alpha^2 + e2 alpha - beta = 0`; both roots are evaluated and the
/// better one is taken.
pub(super) fn fmllr_inner_update(
    inv_g: &SpMat,
    k_row: &[f64],
    beta: f64,
    row: usize,
    transform: &mut [f64],
    dim: usize,
) {
    debug_assert!(row < dim);
    debug_assert_eq!(k_row.len(), dim + 1);

    // Matrix of cofactors = transpose of the adjugate: invert A^T.
    let at = faer::Mat::from_fn(dim, dim, |i, j| transform[j * (dim + 1) + i]);
    let (cofact_mat, _logdet) = invert_with_logdet(at.as_ref());

    // Extended cofactor vector for this row: [cofact_mat.row(row); 0].
    let mut cofact_row = vec![0.0f64; dim + 1];
    for j in 0..dim {
        cofact_row[j] = cofact_mat[(row, j)];
    }
    cofact_row[dim] = 0.0;

    let mut cofact_row_invg = vec![0.0f64; dim + 1];
    inv_g.mul_vec(&cofact_row, &mut cofact_row_invg);

    // Quadratic for the step size.
    let e1: f64 = cofact_row_invg
        .iter()
        .zip(cofact_row.iter())
        .map(|(a, b)| a * b)
        .sum();
    let e2: f64 = cofact_row_invg
        .iter()
        .zip(k_row.iter())
        .map(|(a, b)| a * b)
        .sum();
    let discr = (e2 * e2 + 4.0 * e1 * beta).sqrt();
    let alpha1 = (-e2 + discr) / (2.0 * e1);
    let alpha2 = (-e2 - discr) / (2.0 * e1);
    let auxf1 = beta * (alpha1 * e1 + e2).abs().ln() - 0.5 * alpha1 * alpha1 * e1;
    let auxf2 = beta * (alpha2 * e1 + e2).abs().ln() - 0.5 * alpha2 * alpha2 * e1;
    let alpha = if auxf1 > auxf2 { alpha1 } else { alpha2 };

    // w_d = G_d^{-1} (alpha * cofact_d + k_d)
    for j in 0..=dim {
        cofact_row[j] = cofact_row[j] * alpha + k_row[j];
    }
    let mut out = vec![0.0f64; dim + 1];
    inv_g.mul_vec(&cofact_row, &mut out);
    transform[row * (dim + 1)..(row + 1) * (dim + 1)].copy_from_slice(&out);
}

/// Kaldi `ComputeFmllrMatrixDiagGmmFull`.
pub(super) fn compute_fmllr_matrix_full(
    in_xform: &Mat,
    stats: &FmllrDiagGmmAccs,
    num_iters: usize,
) -> (Mat, f64) {
    let dim = stats.dim;
    let inv_g: Vec<SpMat> = stats.g.iter().map(|s| s.inverted()).collect();

    let old_xform: Vec<f64> = (0..dim)
        .flat_map(|i| (0..=dim).map(move |j| (i, j)))
        .map(|(i, j)| in_xform[[i, j]] as f64)
        .collect();
    let mut new_xform = old_xform.clone();
    let old_objf = fmllr_aux_func(&old_xform, stats);

    let mut k_row = vec![0.0f64; dim + 1];
    for _iter in 0..num_iters {
        for d in 0..dim {
            for j in 0..=dim {
                k_row[j] = stats.k_at(d, j);
            }
            fmllr_inner_update(&inv_g[d], &k_row, stats.beta, d, &mut new_xform, dim);
        }
    }

    let new_objf = fmllr_aux_func(&new_xform, stats);
    let objf_improvement = new_objf - old_objf;
    tracing::debug!(
        per_frame = objf_improvement / (stats.beta + 1.0e-10),
        beta = stats.beta,
        "fMLLR objf improvement"
    );
    if objf_improvement < 0.0 && !approx_equal(new_objf, old_objf) {
        tracing::warn!("not applying fMLLR transform change: objective did not increase");
        return (in_xform.clone(), 0.0);
    }
    (flat_to_mat(&new_xform, dim), objf_improvement)
}

/// Kaldi `ComputeFmllrMatrixDiagGmmDiagonal`.
///
/// For each row `i`, with `s` the scale and `o` the offset, the auxf reduces
/// to `a s^2 + b s + beta = 0` after eliminating `o`; take the root that keeps
/// `s > 0` (`a` is negative, so the negative branch of the quadratic formula).
pub(super) fn compute_fmllr_matrix_diagonal(
    in_xform: &Mat,
    stats: &FmllrDiagGmmAccs,
) -> (Mat, f64) {
    let dim = stats.dim;
    let beta = stats.beta;
    let mut out: Vec<f64> = (0..dim)
        .flat_map(|i| (0..=dim).map(move |j| (i, j)))
        .map(|(i, j)| in_xform[[i, j]] as f64)
        .collect();
    if beta == 0.0 {
        tracing::warn!("computing diagonal fMLLR matrix: no stats [using original transform]");
        return (in_xform.clone(), 0.0);
    }
    let old_obj = fmllr_aux_func(&out, stats);
    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                assert!(
                    out[i * (dim + 1) + j] == 0.0,
                    "diagonal fMLLR: original transform must be diagonal"
                );
            }
        }
    }
    for i in 0..dim {
        let k_ii = stats.k_at(i, i);
        let k_id = stats.k_at(i, dim);
        let g_iii = stats.g[i].get(i, i);
        let g_idd = stats.g[i].get(dim, dim);
        let g_idi = stats.g[i].get(dim, i);
        let a = g_idi * g_idi / g_idd - g_iii;
        let b = k_ii - g_idi * k_id / g_idd;
        let c = beta;
        let s = (-b - (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
        assert!(s > 0.0, "diagonal fMLLR: non-positive scale");
        let o = (k_id - s * g_idi) / g_idd;
        out[i * (dim + 1) + i] = s;
        out[i * (dim + 1) + dim] = o;
    }
    let new_obj = fmllr_aux_func(&out, stats);
    (flat_to_mat(&out, dim), new_obj - old_obj)
}

/// Kaldi `ComputeFmllrMatrixDiagGmmOffset`.
pub(super) fn compute_fmllr_matrix_offset(in_xform: &Mat, stats: &FmllrDiagGmmAccs) -> (Mat, f64) {
    let dim = stats.dim;
    assert_eq!(in_xform.nrows(), dim);
    assert_eq!(in_xform.ncols(), dim + 1);
    for i in 0..dim {
        for j in 0..dim {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                in_xform[[i, j]] == want,
                "offset fMLLR: square part of the input transform must be the identity"
            );
        }
    }
    let mut out: Vec<f64> = (0..dim)
        .flat_map(|i| (0..=dim).map(move |j| (i, j)))
        .map(|(i, j)| in_xform[[i, j]] as f64)
        .collect();
    let mut objf_impr = 0.0f64;
    for i in 0..dim {
        // auxf(b_i) = -0.5 b_i^2 G_i(dim,dim) - b_i G_i(i,dim) + b_i K(i,dim)
        let g_dd = stats.g[i].get(dim, dim);
        let g_id = stats.g[i].get(i, dim);
        let k_id = stats.k_at(i, dim);
        let b_before = out[i * (dim + 1) + dim];
        let objf_before = -0.5 * b_before * b_before * g_dd - b_before * g_id + b_before * k_id;
        let b_i = (k_id - g_id) / g_dd;
        out[i * (dim + 1) + dim] = b_i;
        let objf_after = -0.5 * b_i * b_i * g_dd - b_i * g_id + b_i * k_id;
        if objf_after < objf_before {
            tracing::warn!(
                objf_before,
                objf_after,
                "objf decrease in fMLLR offset estimation"
            );
        }
        objf_impr += objf_after - objf_before;
    }
    (flat_to_mat(&out, dim), objf_impr)
}

fn flat_to_mat(flat: &[f64], dim: usize) -> Mat {
    ndarray::Array2::from_shape_fn((dim, dim + 1), |(i, j)| flat[i * (dim + 1) + j] as f32)
}

/// Kaldi `ApproxEqual` with its default relative tolerance.
fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= 0.01 * (a.abs() + b.abs()).max(1e-20)
}
