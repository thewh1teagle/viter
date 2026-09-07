//! CPU scoring backend: `faer` f32 GEMM plus a rayon-parallel segmented
//! log-sum-exp.
//!
//! The math (see plans/CONTRACTS.md, `viter_kaldi::device`) is
//! `ll = gconst + sum_d mean_invvar[d]*x[d] - 0.5*sum_d invvar[d]*x[d]^2`,
//! which is a single dot product between the frame row `[1, x, x*x]` and the
//! packed Gaussian row `[gconst, mean*invvar, -0.5*invvar]`.

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use ndarray::Array2;
use rayon::prelude::*;

use crate::types::PdfId;

/// Log of `f32::MIN` sentinel used for "no components".
pub(crate) const NEG_INF: f32 = f32::NEG_INFINITY;

/// Build the expanded feature matrix `[frames, 1 + 2*dim]` whose rows are
/// `[1, x, x*x]`.
pub(crate) fn expand_feats(feats: &Array2<f32>) -> Vec<f32> {
    let (frames, dim) = feats.dim();
    let width = 1 + 2 * dim;
    let mut out = vec![0.0f32; frames * width];
    let rows = feats.rows();
    for (t, row) in rows.into_iter().enumerate() {
        let base = t * width;
        out[base] = 1.0;
        for d in 0..dim {
            let v = row[d];
            out[base + 1 + d] = v;
            out[base + 1 + dim + d] = v * v;
        }
    }
    out
}

/// Gather the packed Gaussian rows of the requested pdfs into a compact matrix,
/// returning `(rows, width, segment_bounds)` where `segment_bounds[j]` is the
/// `(start, end)` row range of `pdfs[j]` inside the gathered matrix.
pub(crate) fn gather_pdfs(
    packed_rows: &Array2<f32>,
    offsets: &[u32],
    pdfs: &[PdfId],
) -> (Vec<f32>, usize, Vec<(u32, u32)>) {
    let width = packed_rows.ncols();
    let mut total = 0usize;
    let mut bounds = Vec::with_capacity(pdfs.len());
    for &p in pdfs {
        let p = p as usize;
        let (lo, hi) = (offsets[p] as usize, offsets[p + 1] as usize);
        let n = hi - lo;
        bounds.push((total as u32, (total + n) as u32));
        total += n;
    }

    let mut out = vec![0.0f32; total * width];
    // `PackedGmm::rows` is normally standard-layout; fall back to a row-wise
    // copy if it ever arrives as a non-contiguous view.
    let contiguous = packed_rows.as_slice();
    let mut cursor = 0usize;
    for &p in pdfs {
        let p = p as usize;
        let (lo, hi) = (offsets[p] as usize, offsets[p + 1] as usize);
        let n = hi - lo;
        if n > 0 {
            let dst = &mut out[cursor * width..(cursor + n) * width];
            match contiguous {
                Some(src) => dst.copy_from_slice(&src[lo * width..hi * width]),
                None => {
                    for (i, r) in (lo..hi).enumerate() {
                        for c in 0..width {
                            dst[i * width + c] = packed_rows[[r, c]];
                        }
                    }
                }
            }
        }
        cursor += n;
    }
    (out, width, bounds)
}

/// `C = A * B^T` where `a` is `[m, k]` row-major and `b` is `[n, k]` row-major.
/// Returns `[m, n]` row-major.
pub(crate) fn gemm_abt_raw(a: &[f32], m: usize, b: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    if k == 0 {
        return c;
    }
    let a_ref: MatRef<'_, f32> = MatRef::from_row_major_slice(a, m, k);
    let b_ref: MatRef<'_, f32> = MatRef::from_row_major_slice(b, n, k);
    let c_mut: MatMut<'_, f32> = MatMut::from_row_major_slice_mut(&mut c, m, n);
    matmul(
        c_mut,
        Accum::Replace,
        a_ref,
        b_ref.transpose(),
        1.0f32,
        par(),
    );
    c
}

/// faer's `rayon` feature is on by its default feature set (the workspace takes
/// `faer = "0.24"` with defaults), so the global rayon pool is always available.
fn par() -> Par {
    match std::num::NonZeroUsize::new(rayon::current_num_threads()) {
        Some(nz) => Par::Rayon(nz),
        None => Par::Seq,
    }
}

/// Numerically stable log-sum-exp of a slice.
pub(crate) fn logsumexp(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return NEG_INF;
    }
    let mut mx = NEG_INF;
    for &x in xs {
        if x > mx {
            mx = x;
        }
    }
    if !mx.is_finite() {
        return mx;
    }
    let mut sum = 0.0f32;
    for &x in xs {
        sum += (x - mx).exp();
    }
    mx + sum.ln()
}

/// Reduce a `[frames, total_gauss]` component matrix into `[frames, segments]`
/// by log-sum-exp over each segment's column range. Parallel over frame chunks.
pub(crate) fn segmented_logsumexp(
    comp: &[f32],
    frames: usize,
    total_gauss: usize,
    bounds: &[(u32, u32)],
) -> Array2<f32> {
    let nseg = bounds.len();
    let mut out = vec![0.0f32; frames * nseg];
    if frames == 0 || nseg == 0 {
        return Array2::from_shape_vec((frames, nseg), out).expect("shape matches allocation");
    }

    out.par_chunks_mut(nseg)
        .enumerate()
        .for_each(|(t, out_row)| {
            let row = &comp[t * total_gauss..(t + 1) * total_gauss];
            for (j, &(lo, hi)) in bounds.iter().enumerate() {
                out_row[j] = logsumexp(&row[lo as usize..hi as usize]);
            }
        });

    Array2::from_shape_vec((frames, nseg), out).expect("shape matches allocation")
}

/// Full CPU score: expanded feats x gathered packed rows, then segmented LSE.
pub(crate) fn score_cpu(
    feats: &Array2<f32>,
    packed_rows: &Array2<f32>,
    offsets: &[u32],
    pdfs: &[PdfId],
) -> Array2<f32> {
    let frames = feats.nrows();
    let (gathered, width, bounds) = gather_pdfs(packed_rows, offsets, pdfs);
    let total_gauss = if width == 0 {
        0
    } else {
        gathered.len() / width
    };
    if frames == 0 || pdfs.is_empty() {
        return Array2::zeros((frames, pdfs.len()));
    }
    let a = expand_feats(feats);
    let comp = gemm_abt_raw(&a, frames, &gathered, total_gauss, width);
    segmented_logsumexp(&comp, frames, total_gauss, &bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn expand_builds_one_x_xsq() {
        let f = arr2(&[[2.0f32, 3.0]]);
        let e = expand_feats(&f);
        assert_eq!(e, vec![1.0, 2.0, 3.0, 4.0, 9.0]);
    }

    #[test]
    fn gemm_abt_matches_naive() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 0.0, -1.0, 2.0, 2.0, 2.0]; // 2x3
        let c = gemm_abt_raw(&a, 2, &b, 2, 3);
        // row0 . b0 = 1 - 3 = -2 ; row0 . b1 = 2+4+6 = 12
        // row1 . b0 = 4 - 6 = -2 ; row1 . b1 = 8+10+12 = 30
        assert!((c[0] + 2.0).abs() < 1e-6);
        assert!((c[1] - 12.0).abs() < 1e-6);
        assert!((c[2] + 2.0).abs() < 1e-6);
        assert!((c[3] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn logsumexp_stable_for_large_values() {
        let v = [1000.0f32, 1000.0];
        let r = logsumexp(&v);
        assert!((r - (1000.0 + std::f32::consts::LN_2)).abs() < 1e-2);
    }

    #[test]
    fn logsumexp_empty_is_neg_inf() {
        assert_eq!(logsumexp(&[]), NEG_INF);
    }

    #[test]
    fn segmented_reduces_ranges() {
        // 2 frames, 3 components, segments [0,2) and [2,3)
        let comp = vec![0.0f32, 0.0, 5.0, 1.0, 1.0, 2.0];
        let out = segmented_logsumexp(&comp, 2, 3, &[(0, 2), (2, 3)]);
        assert_eq!(out.dim(), (2, 2));
        assert!((out[[0, 0]] - std::f32::consts::LN_2).abs() < 1e-5);
        assert!((out[[0, 1]] - 5.0).abs() < 1e-5);
        assert!((out[[1, 0]] - (1.0 + std::f32::consts::LN_2)).abs() < 1e-5);
        assert!((out[[1, 1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn gather_compacts_requested_pdfs_only() {
        let rows = arr2(&[[1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]);
        let offsets = vec![0u32, 1, 3, 4];
        let (g, w, b) = gather_pdfs(&rows, &offsets, &[2, 0]);
        assert_eq!(w, 2);
        assert_eq!(b, vec![(0, 1), (1, 2)]);
        assert_eq!(g[0], 4.0);
        assert_eq!(g[2], 1.0);
    }
}
