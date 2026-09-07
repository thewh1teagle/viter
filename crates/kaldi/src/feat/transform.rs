//! Post-MFCC feature transforms: CMVN, deltas, splicing and affine transforms.
//!
//! Ported from `plans/kaldi/src/transform/cmvn.cc` (`AccCmvnStats`, `ApplyCmvn`)
//! and `plans/kaldi/src/feat/feature-functions.cc` (`DeltaFeatures`,
//! `ComputeDeltas`, `SpliceFrames`).

use ndarray::Array2;

use crate::types::Feats;

/// Kaldi's CMVN statistics, stored as Kaldi's `2 x (dim+1)` double matrix:
/// row 0 is the sum plus the count in its last column, row 1 the sum of squares.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CmvnStats {
    pub sum: Vec<f64>,
    pub sumsq: Vec<f64>,
    pub count: f64,
}

impl CmvnStats {
    pub fn new(dim: usize) -> Self {
        Self {
            sum: vec![0.0; dim],
            sumsq: vec![0.0; dim],
            count: 0.0,
        }
    }

    pub fn dim(&self) -> usize {
        self.sum.len()
    }

    /// `AccCmvnStats` (cmvn.cc:30) with unit weights, over every frame.
    pub fn accumulate(&mut self, f: &Feats) {
        let dim = f.ncols();
        if self.sum.is_empty() && self.count == 0.0 {
            self.sum = vec![0.0; dim];
            self.sumsq = vec![0.0; dim];
        }
        assert_eq!(
            dim,
            self.sum.len(),
            "CMVN dim mismatch: stats {} vs feats {dim}",
            self.sum.len()
        );
        for row in f.rows() {
            self.count += 1.0;
            for (d, &v) in row.iter().enumerate() {
                let v = v as f64;
                self.sum[d] += v;
                self.sumsq[d] += v * v;
            }
        }
    }

    /// Accumulate one frame with an explicit weight, as Kaldi's weighted
    /// `AccCmvnStats` overload does.
    pub fn accumulate_frame(&mut self, frame: &[f32], weight: f64) {
        if weight == 0.0 {
            return;
        }
        assert_eq!(frame.len(), self.sum.len(), "CMVN dim mismatch");
        self.count += weight;
        for (d, &v) in frame.iter().enumerate() {
            let v = v as f64;
            self.sum[d] += v * weight;
            self.sumsq[d] += v * v * weight;
        }
    }

    /// Add another speaker's/utterance's stats into this one.
    pub fn merge(&mut self, o: &Self) {
        if self.sum.is_empty() && self.count == 0.0 {
            self.sum = vec![0.0; o.sum.len()];
            self.sumsq = vec![0.0; o.sumsq.len()];
        }
        assert_eq!(self.sum.len(), o.sum.len(), "CMVN dim mismatch in merge");
        for d in 0..self.sum.len() {
            self.sum[d] += o.sum[d];
            self.sumsq[d] += o.sumsq[d];
        }
        self.count += o.count;
    }

    /// Per-dimension mean. Panics if no frames were accumulated.
    pub fn mean(&self) -> Vec<f64> {
        assert!(self.count > 0.0, "CMVN stats are empty");
        self.sum.iter().map(|s| s / self.count).collect()
    }
}

/// `ApplyCmvn` (cmvn.cc:64).
///
/// With `norm_vars = false` (MFA's setting, since `CmvnComputer` never asks for
/// variance normalisation) this only subtracts the mean.
pub fn apply_cmvn(f: &mut Feats, stats: &CmvnStats, norm_vars: bool) {
    let dim = f.ncols();
    assert_eq!(
        dim,
        stats.sum.len(),
        "CMVN dim mismatch: stats {} vs feats {dim}",
        stats.sum.len()
    );
    // cmvn.cc:80: Kaldi errors out below a count of 1.
    assert!(
        stats.count >= 1.0,
        "insufficient stats for cepstral mean normalization: count = {}",
        stats.count
    );

    if !norm_vars {
        let offset: Vec<f32> = stats
            .sum
            .iter()
            .map(|s| (-s / stats.count) as f32)
            .collect();
        for mut row in f.rows_mut() {
            for (v, o) in row.iter_mut().zip(offset.iter()) {
                *v += o;
            }
        }
        return;
    }

    // cmvn.cc:94-115: x(d) <- x(d)*scale + offset.
    let mut scale = vec![0.0f32; dim];
    let mut offset = vec![0.0f32; dim];
    for d in 0..dim {
        let mean = stats.sum[d] / stats.count;
        let mut var = stats.sumsq[d] / stats.count - mean * mean;
        let floor = 1.0e-20f64;
        if var < floor {
            tracing::warn!("flooring cepstral variance from {var} to {floor}");
            var = floor;
        }
        let s = 1.0 / var.sqrt();
        assert!(
            s.is_finite() && s != 0.0,
            "NaN or infinity in cepstral mean/variance computation"
        );
        scale[d] = s as f32;
        offset[d] = (-(mean * s)) as f32;
    }
    for mut row in f.rows_mut() {
        for d in 0..dim {
            row[d] = row[d] * scale[d] + offset[d];
        }
    }
}

/// `DeltaFeaturesOptions` (feature-functions.h:48).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeltaOptions {
    pub order: u32,
    /// Actual window width is `2*window + 1`.
    pub window: u32,
}

impl Default for DeltaOptions {
    /// Kaldi's defaults, which MFA uses unchanged: order 2, window 2.
    fn default() -> Self {
        Self {
            order: 2,
            window: 2,
        }
    }
}

/// `DeltaFeatures::DeltaFeatures` (feature-functions.cc:54): the regression
/// filter for each delta order, built by repeatedly convolving with the
/// first-order filter.
fn delta_scales(o: &DeltaOptions) -> Vec<Vec<f32>> {
    assert!(o.order < 1000, "implausible delta order");
    assert!(o.window > 0 && o.window < 1000, "implausible delta window");
    let mut scales: Vec<Vec<f32>> = Vec::with_capacity(o.order as usize + 1);
    scales.push(vec![1.0f32]);

    for i in 1..=o.order as usize {
        let window = o.window as i32;
        let prev = &scales[i - 1];
        let prev_offset = (prev.len() as i32 - 1) / 2;
        let cur_offset = prev_offset + window;
        let mut cur = vec![0.0f32; prev.len() + 2 * window as usize];

        let mut normalizer = 0.0f32;
        for j in -window..=window {
            normalizer += (j * j) as f32;
            for k in -prev_offset..=prev_offset {
                cur[(j + k + cur_offset) as usize] += j as f32 * prev[(k + prev_offset) as usize];
            }
        }
        for v in cur.iter_mut() {
            *v *= 1.0 / normalizer;
        }
        scales.push(cur);
    }
    scales
}

/// `ComputeDeltas` (feature-functions.cc:160). Output dim is
/// `dim * (order + 1)`; frames outside the signal are replicated from the edges.
pub fn add_deltas(f: &Feats, o: &DeltaOptions) -> Feats {
    let num_frames = f.nrows();
    let feat_dim = f.ncols();
    let out_dim = feat_dim * (o.order as usize + 1);
    let mut out = Array2::<f32>::zeros((num_frames, out_dim));
    if num_frames == 0 || feat_dim == 0 {
        return out;
    }
    let scales = delta_scales(o);

    for frame in 0..num_frames {
        for (i, sc) in scales.iter().enumerate() {
            let max_offset = (sc.len() as i32 - 1) / 2;
            let base = i * feat_dim;
            for j in -max_offset..=max_offset {
                let scale = sc[(j + max_offset) as usize];
                if scale == 0.0 {
                    continue;
                }
                let mut off = frame as i32 + j;
                if off < 0 {
                    off = 0;
                } else if off >= num_frames as i32 {
                    off = num_frames as i32 - 1;
                }
                let src = f.row(off as usize);
                for d in 0..feat_dim {
                    out[[frame, base + d]] += scale * src[d];
                }
            }
        }
    }
    out
}

/// `SpliceFrames` (feature-functions.cc:205). Output dim is
/// `dim * (1 + left + right)`; edge frames are repeated.
pub fn splice(f: &Feats, left: usize, right: usize) -> Feats {
    let t = f.nrows();
    let d = f.ncols();
    assert!(t > 0 && d > 0, "SpliceFrames: empty input");
    let n = 1 + left + right;
    let mut out = Array2::<f32>::zeros((t, d * n));
    for row in 0..t {
        for j in 0..n {
            let mut t2 = row as i64 + j as i64 - left as i64;
            if t2 < 0 {
                t2 = 0;
            }
            if t2 >= t as i64 {
                t2 = t as i64 - 1;
            }
            let src = f.row(t2 as usize);
            for k in 0..d {
                out[[row, j * d + k]] = src[k];
            }
        }
    }
    out
}

/// Apply a Kaldi affine feature transform (LDA+MLLT, fMLLR).
///
/// `mat` is `[out, in]` for a linear transform, or `[out, in+1]` where the last
/// column is the offset applied after the linear part — the same convention as
/// Kaldi's `transform-feats`.
pub fn apply_transform(f: &Feats, mat: &Array2<f32>) -> Feats {
    let t = f.nrows();
    let in_dim = f.ncols();
    let out_dim = mat.nrows();
    let cols = mat.ncols();
    assert!(
        cols == in_dim || cols == in_dim + 1,
        "transform is [{out_dim}, {cols}] but features have dim {in_dim}"
    );
    let affine = cols == in_dim + 1;

    let mut out = Array2::<f32>::zeros((t, out_dim));
    for row in 0..t {
        let src = f.row(row);
        for o in 0..out_dim {
            let m = mat.row(o);
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += m[i] * src[i];
            }
            if affine {
                acc += m[in_dim];
            }
            out[[row, o]] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn feats() -> Feats {
        array![
            [1.0f32, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    }

    #[test]
    fn cmvn_stats_accumulate_and_merge() {
        let f = feats();
        let mut a = CmvnStats::new(3);
        a.accumulate(&f);
        assert_eq!(a.count, 4.0);
        assert_eq!(a.sum[0], 10.0);
        assert_eq!(a.sumsq[0], 1.0 + 4.0 + 9.0 + 16.0);
        assert_eq!(a.mean()[1], 5.0);

        let mut b = CmvnStats::new(3);
        b.accumulate(&f);
        a.merge(&b);
        assert_eq!(a.count, 8.0);
        assert_eq!(a.sum[0], 20.0);
        assert_eq!(a.mean()[0], 2.5);
    }

    #[test]
    fn cmvn_mean_only_zeroes_the_mean() {
        let mut f = feats();
        let mut s = CmvnStats::new(3);
        s.accumulate(&f);
        apply_cmvn(&mut f, &s, false);
        for d in 0..3 {
            let col_sum: f32 = f.column(d).iter().sum();
            assert!(col_sum.abs() < 1e-5, "dim {d}: {col_sum}");
        }
        // Shape is unchanged and the spread is preserved.
        assert_eq!(f.shape(), &[4, 3]);
        assert!((f[[0, 0]] - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn cmvn_with_variance_normalises_to_unit_variance() {
        let mut f = feats();
        let mut s = CmvnStats::new(3);
        s.accumulate(&f);
        apply_cmvn(&mut f, &s, true);
        for d in 0..3 {
            let col: Vec<f32> = f.column(d).to_vec();
            let mean: f32 = col.iter().sum::<f32>() / col.len() as f32;
            let var: f32 =
                col.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / col.len() as f32;
            assert!(mean.abs() < 1e-5, "dim {d} mean {mean}");
            assert!((var - 1.0).abs() < 1e-4, "dim {d} var {var}");
        }
    }

    #[test]
    fn cmvn_weighted_frame_accumulation() {
        let mut s = CmvnStats::new(2);
        s.accumulate_frame(&[1.0, 2.0], 2.0);
        s.accumulate_frame(&[3.0, 4.0], 0.0); // zero weight is a no-op
        assert_eq!(s.count, 2.0);
        assert_eq!(s.sum, vec![2.0, 4.0]);
        assert_eq!(s.mean(), vec![1.0, 2.0]);
    }

    #[test]
    fn delta_scales_are_kaldis_regression_filters() {
        let s = delta_scales(&DeltaOptions::default());
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], vec![1.0]);
        // First order: j / sum(j^2) over j in -2..2, i.e. j/10.
        assert_eq!(s[1].len(), 5);
        for (i, v) in s[1].iter().enumerate() {
            let j = i as f32 - 2.0;
            assert!((v - j / 10.0).abs() < 1e-6, "{i}: {v}");
        }
        // Second order is the first-order filter convolved with itself: 9 taps,
        // summing to zero and symmetric.
        assert_eq!(s[2].len(), 9);
        let sum: f32 = s[2].iter().sum();
        assert!(sum.abs() < 1e-6);
        for i in 0..4 {
            assert!((s[2][i] - s[2][8 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn deltas_of_a_linear_ramp_are_constant() {
        // x(t) = t: first delta is 1 everywhere away from the edges, second is 0.
        let n = 30;
        let f = Array2::from_shape_fn((n, 1), |(t, _)| t as f32);
        let d = add_deltas(&f, &DeltaOptions::default());
        assert_eq!(d.shape(), &[n, 3]);
        for t in 6..n - 6 {
            assert!((d[[t, 0]] - t as f32).abs() < 1e-4);
            assert!((d[[t, 1]] - 1.0).abs() < 1e-4, "t={t}: {}", d[[t, 1]]);
            assert!(d[[t, 2]].abs() < 1e-4, "t={t}: {}", d[[t, 2]]);
        }
    }

    #[test]
    fn deltas_replicate_edges() {
        // A constant signal has zero deltas even at the boundaries, because
        // out-of-range frames are clamped to the first/last frame.
        let f = Array2::from_elem((5, 2), 7.0f32);
        let d = add_deltas(&f, &DeltaOptions::default());
        assert_eq!(d.shape(), &[5, 6]);
        for t in 0..5 {
            for k in 2..6 {
                assert!(d[[t, k]].abs() < 1e-5, "t={t} k={k}: {}", d[[t, k]]);
            }
            assert!((d[[t, 0]] - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn deltas_order_zero_is_a_copy() {
        let f = feats();
        let d = add_deltas(
            &f,
            &DeltaOptions {
                order: 0,
                window: 2,
            },
        );
        assert_eq!(d, f);
    }

    #[test]
    fn deltas_on_empty_input() {
        let f = Array2::<f32>::zeros((0, 13));
        let d = add_deltas(&f, &DeltaOptions::default());
        assert_eq!(d.shape(), &[0, 39]);
    }

    #[test]
    fn mfcc_deltas_give_39_dims() {
        let f = Array2::<f32>::zeros((20, 13));
        let d = add_deltas(&f, &DeltaOptions::default());
        assert_eq!(d.ncols(), 39);
    }

    #[test]
    fn splice_repeats_edge_frames() {
        let f = feats();
        let s = splice(&f, 1, 1);
        assert_eq!(s.shape(), &[4, 9]);
        // Frame 0: [frame0, frame0, frame1].
        assert_eq!(
            s.row(0).to_vec(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
        );
        // Frame 3 (last): [frame2, frame3, frame3].
        assert_eq!(
            s.row(3).to_vec(),
            vec![3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 4.0, 8.0, 12.0]
        );
        // Interior frame 1: [frame0, frame1, frame2].
        assert_eq!(
            s.row(1).to_vec(),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]
        );
    }

    #[test]
    fn splice_3_3_gives_the_lda_input_dim() {
        let f = Array2::<f32>::zeros((10, 13));
        let s = splice(&f, 3, 3);
        assert_eq!(s.shape(), &[10, 91]); // 13 * 7
    }

    #[test]
    fn splice_zero_context_is_a_copy() {
        let f = feats();
        assert_eq!(splice(&f, 0, 0), f);
    }

    #[test]
    fn apply_linear_transform() {
        let f = feats();
        // Pick out dim 1 and the sum of dims 0 and 2.
        let m = array![[0.0f32, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let out = apply_transform(&f, &m);
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(out.row(0).to_vec(), vec![2.0, 4.0]);
        assert_eq!(out.row(3).to_vec(), vec![8.0, 16.0]);
    }

    #[test]
    fn apply_affine_transform_uses_the_last_column_as_offset() {
        let f = feats();
        let m = array![[1.0f32, 0.0, 0.0, 10.0], [0.0, 0.0, 1.0, -1.0]];
        let out = apply_transform(&f, &m);
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(out.row(0).to_vec(), vec![11.0, 2.0]);
        assert_eq!(out.row(1).to_vec(), vec![12.0, 5.0]);
    }

    #[test]
    fn apply_identity_transform_is_a_copy() {
        let f = feats();
        let mut m = Array2::<f32>::zeros((3, 3));
        for i in 0..3 {
            m[[i, i]] = 1.0;
        }
        assert_eq!(apply_transform(&f, &m), f);
    }
}
