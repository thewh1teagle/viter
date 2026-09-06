//! Port of `plans/kaldi/src/tree/clusterable-classes.{h,cc}` (`GaussClusterable`)
//! and the generic `Clusterable` methods in `itf/clusterable-itf.h`
//! (`ObjfPlus`, `ObjfMinus`, `Distance`).
//!
//! `SumClusterable` from `cluster-utils.cc` lives here too since it is just an
//! Option-returning fold over `GaussClusterable`.

use serde::{Deserialize, Serialize};

const M_LOG_2PI: f64 = 1.836_593_418_756_099_2;

/// Kaldi `GaussClusterable`: diagonal-Gaussian sufficient statistics.
/// `stats` is `[2, dim]`: row 0 = sum x, row 1 = sum x^2. All in f64 as Kaldi.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussClusterable {
    pub count: f64,
    pub stats: ndarray::Array2<f64>,
    pub var_floor: f64,
}

impl GaussClusterable {
    /// Kaldi `GaussClusterable(dim, var_floor)`.
    pub fn new(dim: usize, var_floor: f64) -> Self {
        Self {
            count: 0.0,
            stats: ndarray::Array2::zeros((2, dim)),
            var_floor,
        }
    }

    /// Kaldi `GaussClusterable(x_stats, x2_stats, var_floor, count)`.
    pub fn from_stats(x: &[f64], x2: &[f64], var_floor: f64, count: f64) -> Self {
        assert_eq!(x.len(), x2.len());
        let dim = x.len();
        let mut stats = ndarray::Array2::zeros((2, dim));
        for d in 0..dim {
            stats[[0, d]] = x[d];
            stats[[1, d]] = x2[d];
        }
        Self {
            count,
            stats,
            var_floor,
        }
    }

    pub fn dim(&self) -> usize {
        self.stats.ncols()
    }

    /// Kaldi `AddStats(vec, weight)`.
    pub fn add_stats(&mut self, x: &[f32], weight: f64) {
        debug_assert_eq!(x.len(), self.dim());
        self.count += weight;
        for (d, &xi) in x.iter().enumerate() {
            let xi = xi as f64;
            self.stats[[0, d]] += weight * xi;
            self.stats[[1, d]] += weight * xi * xi;
        }
    }

    /// Kaldi `SetZero()`.
    pub fn set_zero(&mut self) {
        self.count = 0.0;
        self.stats.fill(0.0);
    }

    /// Kaldi `Add(other)`.
    pub fn add(&mut self, o: &Self) {
        self.count += o.count;
        self.stats += &o.stats;
    }

    /// Kaldi `Sub(other)`.
    pub fn sub(&mut self, o: &Self) {
        self.count -= o.count;
        self.stats -= &o.stats;
    }

    /// Kaldi `Scale(f)`.
    pub fn scale(&mut self, f: f64) {
        assert!(f >= 0.0);
        self.count *= f;
        self.stats *= f;
    }

    /// Kaldi `Normalizer()` == count.
    pub fn normalizer(&self) -> f64 {
        self.count
    }

    pub fn count(&self) -> f64 {
        self.count
    }

    /// Kaldi `GaussClusterable::Objf()`, including the variance floor.
    pub fn objf(&self) -> f64 {
        if self.count <= 0.0 {
            return 0.0;
        }
        let dim = self.dim();
        let mut sum_log = 0.0;
        let mut objf_per_frame = 0.0;
        for d in 0..dim {
            let mean = self.stats[[0, d]] / self.count;
            let var = self.stats[[1, d]] / self.count - mean * mean;
            let floored = var.max(self.var_floor);
            sum_log += floored.ln();
            objf_per_frame += -0.5 * var / floored;
        }
        objf_per_frame += -0.5 * (sum_log + M_LOG_2PI * dim as f64);
        if objf_per_frame.is_nan() {
            return 0.0;
        }
        objf_per_frame * self.count
    }

    /// Kaldi `Clusterable::ObjfPlus`.
    pub fn objf_plus(&self, o: &Self) -> f64 {
        let mut c = self.clone();
        c.add(o);
        c.objf()
    }

    /// Kaldi `Clusterable::ObjfMinus`.
    pub fn objf_minus(&self, o: &Self) -> f64 {
        let mut c = self.clone();
        c.sub(o);
        c.objf()
    }

    /// Kaldi `Clusterable::Distance`: the (non-negative) objf loss from merging.
    pub fn distance(&self, o: &Self) -> f64 {
        let mut c = self.clone();
        c.add(o);
        let mut ans = self.objf() + o.objf() - c.objf();
        if ans < 0.0 {
            ans = 0.0;
        }
        ans
    }

    /// Per-dimension mean (undefined for zero count; returns zeros).
    pub fn mean(&self) -> Vec<f64> {
        if self.count == 0.0 {
            return vec![0.0; self.dim()];
        }
        (0..self.dim())
            .map(|d| self.stats[[0, d]] / self.count)
            .collect()
    }

    /// Per-dimension variance (unfloored), zeros for zero count.
    pub fn var(&self) -> Vec<f64> {
        if self.count == 0.0 {
            return vec![0.0; self.dim()];
        }
        (0..self.dim())
            .map(|d| {
                let m = self.stats[[0, d]] / self.count;
                self.stats[[1, d]] / self.count - m * m
            })
            .collect()
    }

    /// Row 0 of the stats (sum of x).
    pub fn x_stats(&self) -> ndarray::ArrayView1<'_, f64> {
        self.stats.row(0)
    }
    /// Row 1 of the stats (sum of x^2).
    pub fn x2_stats(&self) -> ndarray::ArrayView1<'_, f64> {
        self.stats.row(1)
    }
}

/// Kaldi `SumClusterable`: sum a vector of optional stats. `None` if all empty.
pub fn sum_clusterable(v: &[Option<GaussClusterable>]) -> Option<GaussClusterable> {
    let mut ans: Option<GaussClusterable> = None;
    for c in v.iter().flatten() {
        match &mut ans {
            None => ans = Some(c.clone()),
            Some(a) => a.add(c),
        }
    }
    ans
}

/// Kaldi `SumClusterableObjf`.
pub fn sum_clusterable_objf(v: &[Option<GaussClusterable>]) -> f64 {
    v.iter().flatten().map(|c| c.objf()).sum()
}

/// Kaldi `SumClusterableNormalizer`.
pub fn sum_clusterable_normalizer(v: &[Option<GaussClusterable>]) -> f64 {
    v.iter().flatten().map(|c| c.normalizer()).sum()
}

/// Kaldi `EnsureClusterableVectorNotNull`: replace `None`s with zeroed stats
/// copied from the first non-`None` element.
pub fn ensure_not_null(v: &mut [Option<GaussClusterable>]) {
    let example = match v.iter().flatten().next() {
        None => return,
        Some(c) => {
            let mut e = c.clone();
            e.set_zero();
            e
        }
    };
    for slot in v.iter_mut() {
        if slot.is_none() {
            *slot = Some(example.clone());
        }
    }
}

/// Kaldi `AddToClusters`: sum `stats` into `clusters` by `assignments`,
/// extending `clusters` with `None` as needed.
pub fn add_to_clusters(
    stats: &[Option<GaussClusterable>],
    assignments: &[usize],
    clusters: &mut Vec<Option<GaussClusterable>>,
) {
    assert_eq!(stats.len(), assignments.len());
    if stats.is_empty() {
        return;
    }
    let max_assignment = *assignments.iter().max().unwrap();
    if clusters.len() <= max_assignment {
        clusters.resize(max_assignment + 1, None);
    }
    for (i, s) in stats.iter().enumerate() {
        if let Some(s) = s {
            match &mut clusters[assignments[i]] {
                None => clusters[assignments[i]] = Some(s.clone()),
                Some(c) => c.add(s),
            }
        }
    }
}

/// Kaldi `AddToClustersOptimized`: same result as `add_to_clusters`, but it
/// starts the largest cluster from `total` and subtracts the rest. Reproduced
/// exactly since the floating-point result differs from the naive sum.
pub fn add_to_clusters_optimized(
    stats: &[Option<GaussClusterable>],
    assignments: &[usize],
    total: &GaussClusterable,
    clusters: &mut Vec<Option<GaussClusterable>>,
) {
    assert_eq!(stats.len(), assignments.len());
    if stats.is_empty() {
        return;
    }
    let num_clust = 1 + *assignments.iter().max().unwrap();
    if clusters.len() < num_clust {
        clusters.resize(num_clust, None);
    }
    let mut num_stats_for_cluster = vec![0i32; num_clust];
    let mut num_total_stats = 0i32;
    for (i, s) in stats.iter().enumerate() {
        if s.is_some() {
            num_total_stats += 1;
            num_stats_for_cluster[assignments[i]] += 1;
        }
    }
    if num_total_stats == 0 {
        return;
    }
    let mut subtract_index: i32 = -1;
    for c in 0..num_clust {
        if num_stats_for_cluster[c] > num_total_stats - num_stats_for_cluster[c] {
            subtract_index = c as i32;
            match &mut clusters[c] {
                None => clusters[c] = Some(total.clone()),
                Some(x) => x.add(total),
            }
            break;
        }
    }
    for (i, s) in stats.iter().enumerate() {
        if let Some(s) = s {
            let a = assignments[i];
            if a as i32 != subtract_index {
                match &mut clusters[a] {
                    None => clusters[a] = Some(s.clone()),
                    Some(x) => x.add(s),
                }
                if subtract_index != -1 {
                    clusters[subtract_index as usize]
                        .as_mut()
                        .expect("subtract cluster set above")
                        .sub(s);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gc(vals: &[&[f32]], var_floor: f64) -> GaussClusterable {
        let mut c = GaussClusterable::new(vals[0].len(), var_floor);
        for v in vals {
            c.add_stats(v, 1.0);
        }
        c
    }

    #[test]
    fn objf_matches_closed_form() {
        // Two points 0 and 2 in 1-D: mean 1, var 1, no flooring at 0.01.
        let c = gc(&[&[0.0], &[2.0]], 0.01);
        assert!((c.count - 2.0).abs() < 1e-12);
        let expect = 2.0 * (-0.5 * 1.0 + -0.5 * (1.0f64.ln() + M_LOG_2PI));
        assert!((c.objf() - expect).abs() < 1e-9);
    }

    #[test]
    fn var_floor_applies() {
        // All points identical -> var 0, floored to var_floor.
        let c = gc(&[&[1.0], &[1.0], &[1.0]], 0.5);
        let expect = 3.0 * (-0.0 + -0.5 * (0.5f64.ln() + M_LOG_2PI));
        assert!((c.objf() - expect).abs() < 1e-9);
    }

    #[test]
    fn zero_count_objf_is_zero() {
        let c = GaussClusterable::new(3, 0.01);
        assert_eq!(c.objf(), 0.0);
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = gc(&[&[0.0], &[2.0]], 0.01);
        let b = gc(&[&[5.0]], 0.01);
        let mut m = a.clone();
        m.add(&b);
        assert!((m.count - 3.0).abs() < 1e-12);
        m.sub(&b);
        assert!((m.count - 2.0).abs() < 1e-12);
        assert!((m.stats[[0, 0]] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn distance_is_nonnegative_and_symmetric() {
        let a = gc(&[&[0.0], &[0.2]], 0.01);
        let b = gc(&[&[5.0], &[5.2]], 0.01);
        let d = a.distance(&b);
        assert!(d > 0.0);
        assert!((d - b.distance(&a)).abs() < 1e-6);
    }

    #[test]
    fn optimized_add_matches_plain() {
        let s: Vec<Option<GaussClusterable>> = vec![
            Some(gc(&[&[0.0]], 0.01)),
            Some(gc(&[&[1.0]], 0.01)),
            Some(gc(&[&[2.0]], 0.01)),
            Some(gc(&[&[3.0]], 0.01)),
        ];
        let assignments = vec![0usize, 1, 1, 1];
        let total = sum_clusterable(&s).unwrap();
        let mut c1 = Vec::new();
        add_to_clusters(&s, &assignments, &mut c1);
        let mut c2 = Vec::new();
        add_to_clusters_optimized(&s, &assignments, &total, &mut c2);
        for i in 0..2 {
            let a = c1[i].as_ref().unwrap();
            let b = c2[i].as_ref().unwrap();
            assert!((a.count - b.count).abs() < 1e-9);
            assert!((a.stats[[0, 0]] - b.stats[[0, 0]]).abs() < 1e-9);
        }
    }
}
