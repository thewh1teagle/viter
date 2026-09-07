//! `AmDiagGmm`: one `DiagGmm` per pdf. Port of `gmm/am-diag-gmm.{h,cc}`.

use std::sync::{Arc, Mutex};

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::diag::DiagGmm;
use super::get_split_targets;
use crate::types::PdfId;

/// The acoustic model: a diagonal GMM for every pdf (tied HMM state).
///
/// `version` is bumped on every mutation so downstream caches — notably the
/// packed device matrix and the GPU-resident copy of it — can tell when they are
/// stale. Any code taking `&mut` access to a pdf must bump it; `pdf_mut` does so
/// eagerly for that reason.
#[derive(Debug, Serialize, Deserialize)]
pub struct AmDiagGmm {
    pdfs: Vec<DiagGmm>,
    /// Never serialized: a loaded model gets a fresh unique number.
    #[serde(skip, default = "next_version")]
    version: u64,
    /// Cache of `packed()`, tagged with the version it was built from. Not part of
    /// the serialised model: it is pure derived data.
    #[serde(skip)]
    packed_cache: Mutex<Option<(u64, Arc<PackedGmm>)>>,
}

/// The acoustic model flattened for a single batched GEMM.
///
/// `rows` is `[total_gauss, 1 + 2*dim]`; row `g` is
/// `[gconst, mean*invvar (dim), -0.5*invvar (dim)]`. Paired with a frame row
/// `[1, x (dim), x^2 (dim)]`, the dot product is exactly that component's
/// log-likelihood, so scoring a whole utterance is one `[frames, 1+2d] x [gauss, 1+2d]^T`
/// matrix product followed by a segmented log-sum-exp.
///
/// `offsets[p]..offsets[p+1]` is the row range belonging to pdf `p`; `offsets` has
/// `num_pdfs + 1` entries.
#[derive(Clone, Debug)]
pub struct PackedGmm {
    pub rows: Array2<f32>,
    pub offsets: Vec<u32>,
}

impl Default for AmDiagGmm {
    fn default() -> Self {
        Self::new()
    }
}

/// Cloning carries the model and its version but shares no cache: the clone will
/// rebuild its packed matrix on first use. `Mutex` is not `Clone`, so this is
/// written by hand rather than derived.
/// Process-wide version counter: every model instance and every mutation gets a
/// number no other instance has ever had, so caches keyed by version (the GPU's
/// resident packed model) can never confuse two models.
fn next_version() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Clone for AmDiagGmm {
    fn clone(&self) -> Self {
        AmDiagGmm {
            pdfs: self.pdfs.clone(),
            // A clone is a different model as far as device caches are concerned:
            // it must never share a version with its source.
            version: next_version(),
            packed_cache: Mutex::new(None),
        }
    }
}

impl AmDiagGmm {
    pub fn new() -> Self {
        AmDiagGmm {
            pdfs: Vec::new(),
            version: next_version(),
            packed_cache: Mutex::new(None),
        }
    }

    /// `num_pdfs` copies of `proto` (`AmDiagGmm::Init`, `am-diag-gmm.cc`).
    /// This is how `gmm-init-mono` builds a flat-start monophone model from one
    /// global Gaussian.
    pub fn init(proto: &DiagGmm, num_pdfs: usize) -> Self {
        assert!(num_pdfs > 0, "AmDiagGmm::init needs at least one pdf");
        AmDiagGmm {
            pdfs: vec![proto.clone(); num_pdfs],
            version: next_version(),
            packed_cache: Mutex::new(None),
        }
    }

    pub fn add_pdf(&mut self, g: DiagGmm) {
        if let Some(first) = self.pdfs.first() {
            assert_eq!(
                first.dim(),
                g.dim(),
                "AmDiagGmm::add_pdf: dimension mismatch"
            );
        }
        self.pdfs.push(g);
        self.bump();
    }

    pub fn num_pdfs(&self) -> usize {
        self.pdfs.len()
    }

    pub fn dim(&self) -> usize {
        self.pdfs.first().map(|g| g.dim()).unwrap_or(0)
    }

    pub fn pdf(&self, i: PdfId) -> &DiagGmm {
        &self.pdfs[i as usize]
    }

    /// Mutable access to a pdf. Bumps the version immediately, since the caller may
    /// change anything through the returned reference.
    pub fn pdf_mut(&mut self, i: PdfId) -> &mut DiagGmm {
        self.bump();
        &mut self.pdfs[i as usize]
    }

    /// Total Gaussians across all pdfs.
    pub fn num_gauss(&self) -> usize {
        self.pdfs.iter().map(|g| g.num_gauss()).sum()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// Log-likelihood of `x` under pdf `pdf`.
    pub fn log_likelihood(&self, pdf: PdfId, x: &[f32]) -> f32 {
        self.pdfs[pdf as usize].log_likelihood(x)
    }

    /// Increment the version and drop the packed cache.
    fn bump(&mut self) {
        self.version = next_version();
        // `&mut self` means no other reader can hold the lock.
        if let Ok(mut guard) = self.packed_cache.lock() {
            *guard = None;
        }
    }

    /// Split Gaussians across pdfs up to `target_components` total
    /// (`AmDiagGmm::SplitByCount`, `am-diag-gmm.cc:102`).
    ///
    /// `occs` are the per-pdf occupancy counts from the last accumulation pass;
    /// `get_split_targets` turns them into a per-pdf Gaussian budget.
    pub fn split_by_count(
        &mut self,
        occs: &[f64],
        target_components: usize,
        perturb_factor: f32,
        power: f32,
        min_count: f32,
        rng: &mut impl rand::Rng,
    ) {
        assert_eq!(
            occs.len(),
            self.num_pdfs(),
            "AmDiagGmm::split_by_count: one occupancy per pdf required"
        );
        let gauss_at_start = self.num_gauss();
        let targets = get_split_targets(occs, target_components, power, min_count);

        for i in 0..self.pdfs.len() {
            if self.pdfs[i].num_gauss() < targets[i] {
                self.pdfs[i].split(targets[i], perturb_factor, rng);
            }
        }
        self.bump();
        tracing::info!(
            num_pdfs = self.num_pdfs(),
            target_components,
            power,
            perturb_factor,
            min_count,
            from = gauss_at_start,
            to = self.num_gauss(),
            "split states"
        );
    }

    /// Merge Gaussians down to `target_components` total
    /// (`AmDiagGmm::MergeByCount`, `am-diag-gmm.cc:125`).
    pub fn merge_by_count(
        &mut self,
        occs: &[f64],
        target_components: usize,
        power: f32,
        min_count: f32,
    ) {
        assert_eq!(
            occs.len(),
            self.num_pdfs(),
            "AmDiagGmm::merge_by_count: one occupancy per pdf required"
        );
        let gauss_at_start = self.num_gauss();
        let mut targets = get_split_targets(occs, target_components, power, min_count);

        for i in 0..self.pdfs.len() {
            if targets[i] == 0 {
                targets[i] = 1; // can't merge below 1.
            }
            if self.pdfs[i].num_gauss() > targets[i] {
                self.pdfs[i].merge(targets[i]);
            }
        }
        self.bump();
        tracing::info!(
            num_pdfs = self.num_pdfs(),
            target_components,
            power,
            min_count,
            from = gauss_at_start,
            to = self.num_gauss(),
            "merged states"
        );
    }

    /// Recompute every pdf's gconsts; returns the total number of bad ones
    /// (`AmDiagGmm::ComputeGconsts`, `am-diag-gmm.cc:90`).
    pub fn compute_gconsts(&mut self) -> usize {
        let mut num_bad = 0;
        for g in self.pdfs.iter_mut() {
            num_bad += g.compute_gconsts();
        }
        if num_bad > 0 {
            tracing::warn!(num_bad, "found bad Gaussian components");
        }
        self.bump();
        num_bad
    }

    /// Scale the mixture weights of the listed pdfs by `boost` and recompute
    /// gconsts (`kalpy/extensions/gmm/gmm.cpp:203`).
    ///
    /// Weights are left unnormalised on purpose: boosting silence makes silence
    /// states score higher relative to everything else, and MFA undoes it by
    /// calling again with `1/boost`. The caller resolves silence phones to pdfs
    /// (Kaldi's `GetPdfsForPhones`) since that needs the transition model.
    pub fn boost_silence(&mut self, silence_pdfs: &[PdfId], boost: f32) {
        for &pdf in silence_pdfs {
            self.pdfs[pdf as usize].scale_weights(boost);
        }
        self.bump();
    }

    /// The packed `[total_gauss, 1+2*dim]` matrix for batched device scoring.
    ///
    /// Cached internally and rebuilt only when `version()` changes, so the hot
    /// alignment loop can call this per utterance for free.
    pub fn packed(&self) -> PackedGmm {
        (*self.packed_arc()).clone()
    }

    /// Shared handle to the cached packed matrix, avoiding a copy of what can be a
    /// very large matrix. The device layer keeps this alive across calls.
    pub fn packed_arc(&self) -> Arc<PackedGmm> {
        let mut guard = self.packed_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((v, ref cached)) = *guard
            && v == self.version
        {
            return Arc::clone(cached);
        }
        let built = Arc::new(self.build_packed());
        *guard = Some((self.version, Arc::clone(&built)));
        built
    }

    fn build_packed(&self) -> PackedGmm {
        let dim = self.dim();
        let total_gauss = self.num_gauss();
        let mut rows = Array2::<f32>::zeros((total_gauss, 1 + 2 * dim));
        let mut offsets = Vec::with_capacity(self.pdfs.len() + 1);
        offsets.push(0u32);

        let mut r = 0usize;
        for g in self.pdfs.iter() {
            for c in 0..g.num_gauss() {
                rows[[r, 0]] = g.gconsts[c];
                for d in 0..dim {
                    rows[[r, 1 + d]] = g.means_invvars[[c, d]];
                    rows[[r, 1 + dim + d]] = -0.5 * g.inv_vars[[c, d]];
                }
                r += 1;
            }
            offsets.push(r as u32);
        }
        debug_assert_eq!(r, total_gauss);
        PackedGmm { rows, offsets }
    }
}

impl PackedGmm {
    /// Number of pdfs described by `offsets`.
    pub fn num_pdfs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Row range `[start, end)` of pdf `p`.
    pub fn pdf_range(&self, p: PdfId) -> (usize, usize) {
        let p = p as usize;
        (self.offsets[p] as usize, self.offsets[p + 1] as usize)
    }

    /// The feature-side row for one frame: `[1, x, x^2]`, matching the layout of
    /// `rows` so that `dot(frame_row, rows[g])` is component `g`'s log-likelihood.
    pub fn frame_row(x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(1 + 2 * x.len());
        out.push(1.0);
        out.extend_from_slice(x);
        out.extend(x.iter().map(|&v| v * v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmm::log_sum_exp;
    use ndarray::array;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::SeedableRng;

    fn proto() -> DiagGmm {
        let mut g = DiagGmm::new(2, 3);
        g.set_means_and_vars(
            &array![[0.0f32, 1.0, 2.0], [3.0, -1.0, 0.5]],
            &array![[1.0f32, 2.0, 0.5], [0.25, 1.0, 4.0]],
        );
        g.weights = vec![0.4, 0.6];
        g.compute_gconsts();
        g
    }

    #[test]
    fn init_replicates_proto() {
        let am = AmDiagGmm::init(&proto(), 5);
        assert_eq!(am.num_pdfs(), 5);
        assert_eq!(am.dim(), 3);
        assert_eq!(am.num_gauss(), 10);
        let x = [0.5f32, 0.5, 0.5];
        let ll0 = am.log_likelihood(0, &x);
        for p in 1..5 {
            assert!((am.log_likelihood(p, &x) - ll0).abs() < 1e-6);
        }
    }

    #[test]
    fn version_bumps_on_mutation() {
        let mut am = AmDiagGmm::init(&proto(), 2);
        let v0 = am.version();
        am.add_pdf(proto());
        assert_ne!(am.version(), v0);
        let v1 = am.version();
        let _ = am.pdf_mut(0);
        assert_ne!(am.version(), v1);
        let v2 = am.version();
        am.compute_gconsts();
        assert_ne!(am.version(), v2);
    }

    #[test]
    fn packed_layout_reproduces_component_loglikes() {
        let am = AmDiagGmm::init(&proto(), 3);
        let packed = am.packed();
        assert_eq!(packed.rows.dim(), (6, 7));
        assert_eq!(packed.offsets, vec![0, 2, 4, 6]);

        let x = [0.25f32, -1.5, 2.0];
        let mut frame = Vec::new();
        PackedGmm::frame_row(&x, &mut frame);
        assert_eq!(frame.len(), 7);

        for p in 0..3u32 {
            let (start, end) = packed.pdf_range(p);
            let mut lls = Vec::new();
            for r in start..end {
                let mut acc = 0.0f32;
                for k in 0..7 {
                    acc += frame[k] * packed.rows[[r, k]];
                }
                lls.push(acc);
            }
            // Matches the per-component path...
            let mut want = Vec::new();
            am.pdf(p).component_log_likes(&x, &mut want);
            for (a, b) in lls.iter().zip(want.iter()) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
            // ...and the segmented log-sum-exp matches the mixture likelihood.
            let got = log_sum_exp(&lls);
            assert!((got - am.log_likelihood(p, &x)).abs() < 1e-4);
        }
    }

    #[test]
    fn packed_is_cached_until_version_changes() {
        let mut am = AmDiagGmm::init(&proto(), 2);
        let a = am.packed_arc();
        let b = am.packed_arc();
        assert!(Arc::ptr_eq(&a, &b), "cache should be reused");
        am.compute_gconsts();
        let c = am.packed_arc();
        assert!(!Arc::ptr_eq(&a, &c), "cache should be invalidated");
    }

    #[test]
    fn packed_tracks_ragged_pdfs() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let mut am = AmDiagGmm::init(&proto(), 3);
        am.pdf_mut(1).split(5, 0.01, &mut rng);
        let packed = am.packed();
        assert_eq!(packed.offsets, vec![0, 2, 7, 9]);
        assert_eq!(packed.rows.nrows(), 9);
        assert_eq!(packed.num_pdfs(), 3);
    }

    #[test]
    fn boost_silence_raises_silence_likelihood_by_log_boost() {
        let mut am = AmDiagGmm::init(&proto(), 3);
        let x = [0.1f32, 0.2, 0.3];
        let before = am.log_likelihood(0, &x);
        let other_before = am.log_likelihood(1, &x);
        am.boost_silence(&[0], 1.25);
        let after = am.log_likelihood(0, &x);
        assert!((after - (before + 1.25f32.ln())).abs() < 1e-4);
        // Non-silence pdfs are untouched.
        assert!((am.log_likelihood(1, &x) - other_before).abs() < 1e-6);
        // And it is exactly undone by boosting with the reciprocal.
        am.boost_silence(&[0], 1.0 / 1.25);
        assert!((am.log_likelihood(0, &x) - before).abs() < 1e-4);
    }

    #[test]
    fn split_by_count_hits_the_target() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut am = AmDiagGmm::init(&DiagGmm::from_single_gaussian(&[0.0; 3], &[1.0; 3]), 4);
        assert_eq!(am.num_gauss(), 4);
        let occs = [100.0, 200.0, 400.0, 800.0];
        am.split_by_count(&occs, 16, 0.01, 0.25, 0.0, &mut rng);
        assert_eq!(am.num_gauss(), 16);
        // The busiest pdf ends up with the most Gaussians.
        assert!(am.pdf(3).num_gauss() >= am.pdf(0).num_gauss());
    }

    #[test]
    fn merge_by_count_reduces_to_target() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let mut am = AmDiagGmm::init(&DiagGmm::from_single_gaussian(&[0.0; 3], &[1.0; 3]), 4);
        let occs = [100.0, 200.0, 400.0, 800.0];
        am.split_by_count(&occs, 32, 0.05, 0.25, 0.0, &mut rng);
        assert_eq!(am.num_gauss(), 32);
        am.merge_by_count(&occs, 12, 0.25, 0.0);
        assert_eq!(am.num_gauss(), 12);
    }

    #[test]
    fn merge_by_count_never_goes_below_one_per_pdf() {
        let mut am = AmDiagGmm::init(&proto(), 4);
        // Target far below num_pdfs: every pdf keeps exactly one Gaussian.
        am.merge_by_count(&[1.0, 1.0, 1.0, 1.0], 1, 0.25, 0.0);
        assert_eq!(am.num_gauss(), 4);
        for p in 0..4 {
            assert_eq!(am.pdf(p).num_gauss(), 1);
        }
    }

    #[test]
    fn serde_roundtrip_drops_cache_but_keeps_model() {
        let am = AmDiagGmm::init(&proto(), 2);
        let _ = am.packed_arc();
        let json = serde_json::to_string(&am).unwrap();
        let back: AmDiagGmm = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_pdfs(), 2);
        assert_eq!(back.num_gauss(), 4);
        let x = [1.0f32, 0.0, -1.0];
        assert!((back.log_likelihood(0, &x) - am.log_likelihood(0, &x)).abs() < 1e-6);
        // The cache rebuilds transparently after deserialisation.
        assert_eq!(back.packed().rows.dim(), (4, 7));
    }
}
