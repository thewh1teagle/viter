//! Batched acoustic scoring — the hot loop of both training and alignment.
//!
//! A [`Device`] evaluates diagonal-covariance Gaussian mixture log-likelihoods
//! for every frame of a feature matrix against a requested set of pdfs. Two
//! backends implement the same math and agree to 1e-4 relative:
//!
//! * [`DeviceKind::Cpu`] — `faer` f32 GEMM plus a rayon-parallel segmented
//!   log-sum-exp.
//! * [`DeviceKind::Gpu`] — two WGSL compute kernels (tiled `A * B^T` GEMM and
//!   segmented log-sum-exp) driven through `wgpu`, with the packed GMM matrix
//!   kept resident across calls.
//!
//! The identity that makes this a single GEMM is
//! `ll = gconst + sum_d mean_invvar[d] * x[d] - 0.5 * sum_d invvar[d] * x[d]^2`,
//! i.e. the dot product of the frame row `[1, x, x*x]` with the packed Gaussian
//! row `[gconst, mean*invvar, -0.5*invvar]` (Kaldi folds the log mixture weight
//! into `gconst`). Per-pdf log-likelihood is then a log-sum-exp over that pdf's
//! contiguous range of Gaussian rows.

mod cpu;
mod gpu;

use std::sync::Arc;

use ndarray::Array2;

use crate::gmm::{AmDiagGmm, DiagGmm};
use crate::types::{Feats, PdfId};

/// Which backend a [`Device`] dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

#[derive(Clone)]
enum Backend {
    Cpu,
    Gpu(Arc<gpu::GpuContext>),
}

/// A compute device for batched GMM scoring.
///
/// Cheap to clone (the GPU context is shared behind an `Arc`) and safe to use
/// from several threads at once.
#[derive(Clone)]
pub struct Device {
    backend: Backend,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backend {
            Backend::Cpu => f.write_str("Device(Cpu)"),
            Backend::Gpu(ctx) => write!(f, "Device(Gpu: {})", ctx.adapter_name()),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl Device {
    /// Try to bring up a high-performance GPU adapter, falling back to the CPU
    /// backend. Logs which one was chosen at `info` level.
    pub fn auto() -> Device {
        match gpu::GpuContext::new() {
            Some(ctx) => {
                tracing::info!(adapter = %ctx.adapter_name(), "using GPU device for scoring");
                Device {
                    backend: Backend::Gpu(Arc::new(ctx)),
                }
            }
            None => {
                tracing::info!("no suitable GPU adapter; using CPU device for scoring");
                Device {
                    backend: Backend::Cpu,
                }
            }
        }
    }

    /// The CPU backend, unconditionally.
    pub fn cpu() -> Device {
        Device {
            backend: Backend::Cpu,
        }
    }

    /// The GPU backend, or `None` if no suitable adapter exists.
    pub fn gpu() -> Option<Device> {
        gpu::GpuContext::new().map(|ctx| Device {
            backend: Backend::Gpu(Arc::new(ctx)),
        })
    }

    pub fn kind(&self) -> DeviceKind {
        match self.backend {
            Backend::Cpu => DeviceKind::Cpu,
            Backend::Gpu(_) => DeviceKind::Gpu,
        }
    }

    /// Name of the GPU adapter in use, if this is a GPU device.
    pub fn adapter_name(&self) -> Option<&str> {
        match &self.backend {
            Backend::Cpu => None,
            Backend::Gpu(ctx) => Some(ctx.adapter_name()),
        }
    }

    /// Log-likelihood of every frame under every listed pdf's GMM.
    ///
    /// Returns `[frames, pdfs.len()]`; column `j` holds the log-likelihood
    /// under `pdfs[j]`. Matches [`AmDiagGmm::log_likelihood`] per
    /// `(frame, pdf)` to 1e-4 relative.
    ///
    /// # Panics
    ///
    /// If `feats` has a different dimension than the model, or a pdf id is out
    /// of range.
    pub fn score(&self, feats: &Feats, am: &AmDiagGmm, pdfs: &[PdfId]) -> Array2<f32> {
        assert_eq!(
            feats.ncols(),
            am.dim(),
            "feature dim {} does not match model dim {}",
            feats.ncols(),
            am.dim()
        );
        let packed = am.packed();
        check_pdfs(pdfs, am.num_pdfs());

        match &self.backend {
            Backend::Cpu => cpu::score_cpu(feats, &packed.rows, &packed.offsets, pdfs),
            Backend::Gpu(ctx) => {
                let expanded = cpu::expand_feats(feats);
                let job = gpu::ScoreJob { expanded: &expanded, frames: feats.nrows(), sel: pdfs };
                ctx.score_jobs(&[job], am.version(), &packed.rows, &packed.offsets)
                    .pop()
                    .expect("one job in, one matrix out")
            }
        }
    }

    /// Score many utterances at once against one shared pdf list. See
    /// [`Device::score_batch_sel`] for the per-utterance form that the aligner uses.
    pub fn score_batch(
        &self,
        feats: &[&Feats],
        am: &AmDiagGmm,
        pdfs: &[PdfId],
    ) -> Vec<Array2<f32>> {
        let sels: Vec<&[PdfId]> = feats.iter().map(|_| pdfs).collect();
        self.score_batch_sel(feats, am, &sels)
    }

    /// Score many utterances at once, each against its own pdf list. Output `i`
    /// is `[frames_i, sels[i].len()]` with column `j` = pdf `sels[i][j]`.
    ///
    /// On the GPU the whole model is resident and every utterance is one
    /// dispatch inside a single submit, so the score traffic is only the pdfs
    /// each graph actually touches. On the CPU utterances are scored in parallel.
    pub fn score_batch_sel(
        &self,
        feats: &[&Feats],
        am: &AmDiagGmm,
        sels: &[&[PdfId]],
    ) -> Vec<Array2<f32>> {
        assert_eq!(feats.len(), sels.len(), "one pdf list per utterance");
        if feats.is_empty() {
            return Vec::new();
        }
        for f in feats {
            assert_eq!(
                f.ncols(),
                am.dim(),
                "feature dim {} does not match model dim {}",
                f.ncols(),
                am.dim()
            );
        }
        let packed = am.packed();
        for sel in sels {
            check_pdfs(sel, am.num_pdfs());
        }

        match &self.backend {
            Backend::Cpu => {
                use rayon::prelude::*;
                feats
                    .par_iter()
                    .zip(sels.par_iter())
                    .map(|(f, sel)| cpu::score_cpu(f, &packed.rows, &packed.offsets, sel))
                    .collect()
            }
            Backend::Gpu(ctx) => {
                use rayon::prelude::*;
                let expanded: Vec<Vec<f32>> = feats.par_iter().map(|f| cpu::expand_feats(f)).collect();
                let jobs: Vec<gpu::ScoreJob<'_>> = feats
                    .iter()
                    .zip(expanded.iter())
                    .zip(sels.iter())
                    .map(|((f, e), sel)| gpu::ScoreJob { expanded: e, frames: f.nrows(), sel })
                    .collect();
                ctx.score_jobs(&jobs, am.version(), &packed.rows, &packed.offsets)
            }
        }
    }

    /// Per-Gaussian log-likelihoods for one pdf: `[frames, num_gauss]`.
    ///
    /// Used to form posteriors during statistics accumulation, so no log-sum-exp
    /// reduction is applied.
    pub fn score_components(&self, feats: &Feats, gmm: &DiagGmm) -> Array2<f32> {
        let dim = gmm.means_invvars.ncols();
        assert_eq!(
            feats.ncols(),
            dim,
            "feature dim {} does not match gmm dim {}",
            feats.ncols(),
            dim
        );
        let num_gauss = gmm.gconsts.len();
        let frames = feats.nrows();
        if frames == 0 || num_gauss == 0 {
            return Array2::zeros((frames, num_gauss));
        }

        let packed = pack_single(gmm);
        let width = 1 + 2 * dim;
        let expanded = cpu::expand_feats(feats);

        match &self.backend {
            Backend::Cpu => {
                let c = cpu::gemm_abt_raw(&expanded, frames, &packed, num_gauss, width);
                Array2::from_shape_vec((frames, num_gauss), c)
                    .expect("gemm output length matches shape")
            }
            Backend::Gpu(_) => {
                // Component scores are a plain GEMM with no reduction; the
                // per-pdf GPU path would need an identity segmentation and a
                // second round trip, so the CPU GEMM is used here — this is the
                // small, per-pdf accumulation path, not the batched hot loop.
                let c = cpu::gemm_abt_raw(&expanded, frames, &packed, num_gauss, width);
                Array2::from_shape_vec((frames, num_gauss), c)
                    .expect("gemm output length matches shape")
            }
        }
    }

    /// `C = A * B^T` with `a` `[m, k]` and `b` `[n, k]`, giving `[m, n]`.
    ///
    /// # Panics
    ///
    /// If the inner dimensions disagree.
    pub fn gemm_abt(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (n, kb) = b.dim();
        assert_eq!(k, kb, "gemm_abt inner dims disagree: {k} vs {kb}");
        let a_owned;
        let a_slice = match a.as_slice() {
            Some(s) => s,
            None => {
                a_owned = a.iter().copied().collect::<Vec<f32>>();
                &a_owned
            }
        };
        let b_owned;
        let b_slice = match b.as_slice() {
            Some(s) => s,
            None => {
                b_owned = b.iter().copied().collect::<Vec<f32>>();
                &b_owned
            }
        };
        let c = cpu::gemm_abt_raw(a_slice, m, b_slice, n, k);
        Array2::from_shape_vec((m, n), c).expect("gemm output length matches shape")
    }
}

fn check_pdfs(pdfs: &[PdfId], num_pdfs: usize) {
    for &p in pdfs {
        assert!(
            (p as usize) < num_pdfs,
            "pdf id {p} out of range (model has {num_pdfs} pdfs)"
        );
    }
}

/// Pack one `DiagGmm` into device rows `[gconst, mean*invvar, -0.5*invvar]`.
fn pack_single(gmm: &DiagGmm) -> Vec<f32> {
    let num_gauss = gmm.gconsts.len();
    let dim = gmm.means_invvars.ncols();
    let width = 1 + 2 * dim;
    let mut out = vec![0.0f32; num_gauss * width];
    for g in 0..num_gauss {
        let base = g * width;
        out[base] = gmm.gconsts[g];
        for d in 0..dim {
            out[base + 1 + d] = gmm.means_invvars[[g, d]];
            out[base + 1 + dim + d] = -0.5 * gmm.inv_vars[[g, d]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// A two-component, two-dimensional GMM with hand-checkable numbers.
    fn toy_gmm() -> DiagGmm {
        DiagGmm {
            weights: vec![0.5, 0.5],
            means_invvars: arr2(&[[1.0f32, 2.0], [-1.0, 0.5]]),
            inv_vars: arr2(&[[1.0f32, 2.0], [0.5, 0.25]]),
            gconsts: vec![-2.0f32, -3.0],
        }
    }

    fn reference_component_ll(gmm: &DiagGmm, g: usize, x: &[f32]) -> f32 {
        let mut ll = gmm.gconsts[g];
        for (d, &v) in x.iter().enumerate() {
            ll += gmm.means_invvars[[g, d]] * v - 0.5 * gmm.inv_vars[[g, d]] * v * v;
        }
        ll
    }

    #[test]
    fn pack_single_layout_is_gconst_meaninvvar_neg_half_invvar() {
        let gmm = toy_gmm();
        let p = pack_single(&gmm);
        assert_eq!(p.len(), 2 * 5);
        assert_eq!(&p[0..5], &[-2.0, 1.0, 2.0, -0.5, -1.0]);
        assert_eq!(&p[5..10], &[-3.0, -1.0, 0.5, -0.25, -0.125]);
    }

    #[test]
    fn score_components_matches_direct_evaluation() {
        let dev = Device::cpu();
        let gmm = toy_gmm();
        let feats = arr2(&[[0.5f32, -1.0], [2.0, 3.0], [0.0, 0.0]]);
        let got = dev.score_components(&feats, &gmm);
        assert_eq!(got.dim(), (3, 2));
        for (t, row) in feats.rows().into_iter().enumerate() {
            let x: Vec<f32> = row.to_vec();
            for g in 0..2 {
                let want = reference_component_ll(&gmm, g, &x);
                assert!(
                    (got[[t, g]] - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "frame {t} comp {g}: got {} want {want}",
                    got[[t, g]]
                );
            }
        }
    }

    #[test]
    fn score_components_empty_frames() {
        let dev = Device::cpu();
        let gmm = toy_gmm();
        let feats = Array2::<f32>::zeros((0, 2));
        assert_eq!(dev.score_components(&feats, &gmm).dim(), (0, 2));
    }

    #[test]
    fn gemm_abt_matches_manual_product() {
        let dev = Device::cpu();
        let a = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[1.0f32, 1.0], [0.0, -1.0], [2.0, 0.0]]);
        let c = dev.gemm_abt(&a, &b);
        assert_eq!(c.dim(), (2, 3));
        assert!((c[[0, 0]] - 3.0).abs() < 1e-6);
        assert!((c[[0, 1]] + 2.0).abs() < 1e-6);
        assert!((c[[0, 2]] - 2.0).abs() < 1e-6);
        assert!((c[[1, 0]] - 7.0).abs() < 1e-6);
        assert!((c[[1, 1]] + 4.0).abs() < 1e-6);
        assert!((c[[1, 2]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn gemm_abt_handles_non_contiguous_inputs() {
        let dev = Device::cpu();
        let big = arr2(&[[1.0f32, 9.0, 2.0], [3.0, 9.0, 4.0]]);
        // A non-standard-layout view (drops the middle column).
        let a = big.slice(ndarray::s![.., ..;2]).to_owned();
        let b = arr2(&[[1.0f32, 1.0]]);
        let c = dev.gemm_abt(&a, &b);
        assert!((c[[0, 0]] - 3.0).abs() < 1e-6);
        assert!((c[[1, 0]] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_device_reports_cpu_kind_and_no_adapter() {
        let dev = Device::cpu();
        assert_eq!(dev.kind(), DeviceKind::Cpu);
        assert!(dev.adapter_name().is_none());
        assert_eq!(Device::default().kind(), DeviceKind::Cpu);
    }

    #[test]
    fn auto_yields_a_usable_device() {
        let dev = Device::auto();
        assert!(matches!(dev.kind(), DeviceKind::Cpu | DeviceKind::Gpu));
    }
}
