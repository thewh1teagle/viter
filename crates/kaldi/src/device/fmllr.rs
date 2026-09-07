//! GPU fMLLR statistics accumulation.
//!
//! [`super::Device::fmllr_accumulate_batch`] takes a batch of utterances of one
//! speaker — features, the aligned pdf per frame, and a per-frame weight (MFA's
//! silence weighting, zero for skipped frames) — and adds their contribution to a
//! [`FmllrDiagGmmAccs`] in one GPU round trip.
//!
//! Two kernels, both against the packed model already resident for scoring:
//!
//! 1. `fmllr.wgsl::fmllr_ab`, one invocation per frame: posteriors over the aligned
//!    pdf's gaussians, then `a[i] = sum_g p_g invvar[i] mean[i]`,
//!    `b[i] = sum_g p_g invvar[i]`, `count = sum_g p_g` — Kaldi's `SingleFrameStats`
//!    without the batching-by-identical-x trick, which the GPU does not need
//!    because every frame gets its own outer product anyway.
//! 2. `fmllr_gemm.wgsl::fmllr_gemm`: `K = A^T X+` and `G[i] = X+^T diag(B[:,i]) X+`.
//!
//! # Rounding
//!
//! Both kernels accumulate in f32 while the CPU path uses f64. The frame loop of
//! kernel 2 is a flat sequential f32 sum of `T` products, so the worst-case relative
//! error on any `K`/`G` element is bounded by `T * eps` with `eps = 2^-24`, i.e.
//! about `6e-8 * T`; the posteriors of kernel 1 add another `~n_gauss * eps`. For a
//! speaker batch of a few hundred thousand frames that is order 1e-2 relative in the
//! worst case and far smaller in practice (the terms are same-signed for `G`, whose
//! diagonal dominates the fMLLR solve). Batches are capped at
//! [`MAX_BATCH_FRAMES`] frames, and each batch's f32 sums are added into the f64
//! accumulator, so the error does not grow with the speaker's total frame count.
//! The estimated transform is compared against the CPU path to 1e-4 relative in the
//! unit tests and end to end by alignment parity.

use wgpu::util::DeviceExt;

use super::gpu::{GpuContext, ResidentModel};
use crate::gmm::AmDiagGmm;
use crate::transform::FmllrDiagGmmAccs;
use crate::types::{Feats, PdfId};

/// Frames per GPU batch. Keeps the f32 sums short (see the rounding note) and the
/// per-batch buffers small; larger batches buy nothing once the GPU is saturated.
pub(super) const MAX_BATCH_FRAMES: usize = 262_144;

const TILE: u32 = 8;
const PARAM_BYTES: u64 = 32;

/// Everything the two fMLLR kernels need, built once per [`GpuContext`].
pub(super) struct FmllrPipelines {
    ab_layout: wgpu::BindGroupLayout,
    ab: wgpu::ComputePipeline,
    gemm_layout: wgpu::BindGroupLayout,
    gemm: wgpu::ComputePipeline,
}

impl FmllrPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let ab_layout = layout(device, "fmllr-ab-layout", &[false; 6], 3);
        let gemm_layout = layout(device, "fmllr-gemm-layout", &[false; 3], 2);
        Self {
            ab: pipeline(
                device,
                "fmllr_ab",
                include_str!("fmllr.wgsl"),
                &ab_layout,
                "fmllr_ab",
            ),
            gemm: pipeline(
                device,
                "fmllr_gemm",
                include_str!("fmllr_gemm.wgsl"),
                &gemm_layout,
                "fmllr_gemm",
            ),
            ab_layout,
            gemm_layout,
        }
    }
}

/// `read_only` flags for the storage bindings after the uniform at binding 0:
/// `reads.len()` read-only buffers then `writes` read-write ones.
fn layout(
    device: &wgpu::Device,
    label: &str,
    reads: &[bool],
    writes: usize,
) -> wgpu::BindGroupLayout {
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(PARAM_BYTES),
        },
        count: None,
    }];
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut b = 1u32;
    for _ in reads {
        entries.push(storage(b, true));
        b += 1;
    }
    for _ in 0..writes {
        entries.push(storage(b, false));
        b += 1;
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    bgl: &wgpu::BindGroupLayout,
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// One flattened batch of frames: expanded feature rows for kernel 1, `xplus` rows
/// for kernel 2, the aligned pdf and the weight per frame.
struct Batch {
    expanded: Vec<f32>,
    xplus: Vec<f32>,
    pdfs: Vec<u32>,
    weights: Vec<f32>,
    frames: usize,
}

/// Flatten a speaker's utterances into batches of at most `MAX_BATCH_FRAMES` frames,
/// dropping zero-weight frames entirely (they contribute nothing and would only cost
/// bandwidth; MFA's `silence_weight = 0` makes that most of the silence).
fn build_batches(
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    weights: &[&[f32]],
    dim: usize,
    padded: usize,
    xw: usize,
) -> Vec<Batch> {
    let mut out = Vec::new();
    let mut cur = Batch {
        expanded: Vec::new(),
        xplus: Vec::new(),
        pdfs: Vec::new(),
        weights: Vec::new(),
        frames: 0,
    };
    for ((f, pdfs), ws) in feats.iter().zip(pdf_per_frame).zip(weights) {
        let n = f.nrows().min(pdfs.len()).min(ws.len());
        for t in 0..n {
            if ws[t] == 0.0 {
                continue;
            }
            let row = f.row(t);
            // Expanded scoring row [1, x, x*x], zero padded to `padded`.
            cur.expanded.push(1.0);
            cur.expanded.extend(row.iter().copied());
            cur.expanded.extend(row.iter().map(|v| v * v));
            cur.expanded
                .resize(cur.expanded.len() + padded - (1 + 2 * dim), 0.0);
            // xplus row [x, 1], zero padded to `xw`.
            cur.xplus.extend(row.iter().copied());
            cur.xplus.push(1.0);
            cur.xplus.resize(cur.xplus.len() + xw - (dim + 1), 0.0);
            cur.pdfs.push(pdfs[t]);
            cur.weights.push(ws[t]);
            cur.frames += 1;
            if cur.frames == MAX_BATCH_FRAMES {
                out.push(std::mem::replace(
                    &mut cur,
                    Batch {
                        expanded: Vec::new(),
                        xplus: Vec::new(),
                        pdfs: Vec::new(),
                        weights: Vec::new(),
                        frames: 0,
                    },
                ));
            }
        }
    }
    if cur.frames > 0 {
        out.push(cur);
    }
    out
}

/// Accumulate fMLLR statistics for a batch of utterances on the GPU.
pub(super) fn accumulate(
    ctx: &GpuContext,
    pipes: &FmllrPipelines,
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    weights: &[&[f32]],
    am: &AmDiagGmm,
    accs: &mut FmllrDiagGmmAccs,
) {
    let dim = am.dim();
    let dim1 = dim + 1;
    let xw = dim1.div_ceil(4) * 4;
    let packed = am.packed_arc();
    ctx.with_model(
        am.version(),
        &packed.rows,
        &packed.offsets,
        |model: &ResidentModel| {
            let padded = model.width() as usize;
            for batch in build_batches(feats, pdf_per_frame, weights, dim, padded, xw) {
                let (beta, k, g) = run_batch(ctx, pipes, model, &batch, dim, dim1, xw);
                accs.add_batch_sums(beta, &k, &g);
            }
        },
    );
}

/// Run both kernels over one batch, returning `(beta, K [dim, dim1], G [dim, dim1,
/// dim1] with both triangles filled)` as f64.
fn run_batch(
    ctx: &GpuContext,
    pipes: &FmllrPipelines,
    model: &ResidentModel,
    batch: &Batch,
    dim: usize,
    dim1: usize,
    xw: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let device = ctx.wgpu_device();
    let queue = ctx.wgpu_queue();
    let t = batch.frames;
    let w4 = model.width() / 4;

    let store = |label: &str, bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let uniform = |label: &str, p: [u32; 8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };
    let zeros = |label: &str, len: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };

    let feats_buf = store("fmllr-feats", bytemuck::cast_slice(&batch.expanded));
    let xplus_buf = store("fmllr-xplus", bytemuck::cast_slice(&batch.xplus));
    let pdf_buf = store("fmllr-pdf", bytemuck::cast_slice(&batch.pdfs));
    let w_buf = store("fmllr-weight", bytemuck::cast_slice(&batch.weights));
    let a_buf = zeros("fmllr-a", t * dim);
    let b_buf = zeros("fmllr-b", t * dim);
    let cnt_buf = zeros("fmllr-cnt", t);
    let k_buf = zeros("fmllr-k", dim * dim1);
    let g_buf = zeros("fmllr-g", dim * dim1 * dim1);

    let ab_par = uniform(
        "fmllr-ab-par",
        [t as u32, dim as u32, dim1 as u32, w4, 0, 0, 0, 0],
    );
    let gemm_par = uniform(
        "fmllr-gemm-par",
        [t as u32, dim as u32, dim1 as u32, xw as u32, 0, 0, 0, 0],
    );

    let bind =
        |label: &str, l: &wgpu::BindGroupLayout, par: &wgpu::Buffer, bufs: &[&wgpu::Buffer]| {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: par.as_entire_binding(),
            }];
            for (i, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: b.as_entire_binding(),
                });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: l,
                entries: &entries,
            })
        };
    let ab_bg = bind(
        "fmllr-ab-bg",
        &pipes.ab_layout,
        &ab_par,
        &[
            &feats_buf,
            model.rows(),
            model.seg_start(),
            model.seg_end(),
            &pdf_buf,
            &w_buf,
            &a_buf,
            &b_buf,
            &cnt_buf,
        ],
    );
    let gemm_bg = bind(
        "fmllr-gemm-bg",
        &pipes.gemm_layout,
        &gemm_par,
        &[&xplus_buf, &a_buf, &b_buf, &k_buf, &g_buf],
    );

    let cnt_bytes = (t * 4).max(4) as u64;
    let k_bytes = (dim * dim1 * 4) as u64;
    let g_bytes = (dim * dim1 * dim1 * 4) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fmllr-readback"),
        size: cnt_bytes + k_bytes + g_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fmllr-encoder"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fmllr-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipes.ab);
        pass.set_bind_group(0, &ab_bg, &[]);
        pass.dispatch_workgroups((t as u32).div_ceil(64).max(1), 1, 1);
        pass.set_pipeline(&pipes.gemm);
        pass.set_bind_group(0, &gemm_bg, &[]);
        let tiles = (dim1 as u32).div_ceil(TILE);
        pass.dispatch_workgroups(tiles, tiles, dim as u32 + 1);
    }
    enc.copy_buffer_to_buffer(&cnt_buf, 0, &readback, 0, cnt_bytes);
    enc.copy_buffer_to_buffer(&k_buf, 0, &readback, cnt_bytes, k_bytes);
    enc.copy_buffer_to_buffer(&g_buf, 0, &readback, cnt_bytes + k_bytes, g_bytes);
    queue.submit(Some(enc.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("gpu poll failed");
    rx.recv()
        .expect("map callback dropped")
        .expect("buffer map failed");

    let (beta, k, mut g) = {
        let view = readback.slice(..).get_mapped_range().expect("mapped range");
        let all: &[f32] = bytemuck::cast_slice(&view[..]);
        let cnt = &all[..t];
        let ks = &all[t..t + dim * dim1];
        let gs = &all[t + dim * dim1..t + dim * dim1 + dim * dim1 * dim1];
        (
            cnt.iter().map(|&v| v as f64).sum::<f64>(),
            ks.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            gs.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        )
    };
    readback.unmap();

    // The kernel writes only `r <= c`; mirror into the lower triangle.
    for i in 0..dim {
        for r in 0..dim1 {
            for c in 0..r {
                g[(i * dim1 + r) * dim1 + c] = g[(i * dim1 + c) * dim1 + r];
            }
        }
    }
    (beta, k, g)
}

/// CPU reference for the same batch, used by the parity test and as the fallback
/// when the device has no GPU.
pub(super) fn accumulate_cpu(
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    weights: &[&[f32]],
    am: &AmDiagGmm,
    accs: &mut FmllrDiagGmmAccs,
) {
    let mut posteriors: Vec<f32> = Vec::new();
    for ((f, pdfs), ws) in feats.iter().zip(pdf_per_frame).zip(weights) {
        let n = f.nrows().min(pdfs.len()).min(ws.len());
        for t in 0..n {
            if ws[t] == 0.0 {
                continue;
            }
            let row = f.row(t);
            let x = row.as_slice().expect("feature rows are contiguous");
            let gmm = am.pdf(pdfs[t]);
            gmm.component_posteriors(x, &mut posteriors);
            if ws[t] != 1.0 {
                for p in posteriors.iter_mut() {
                    *p *= ws[t];
                }
            }
            accs.accumulate_from_posteriors(gmm, x, &posteriors);
        }
    }
    accs.commit_pending();
}

impl super::Device {
    /// Add one batch of utterances' fMLLR statistics into `accs`.
    ///
    /// `feats[u]` is `[frames, dim]`, `pdf_per_frame[u]` the aligned pdf of each
    /// frame and `weights[u]` its weight (MFA's silence weighting; a zero-weight
    /// frame contributes nothing and is skipped). Shorter of the three lengths wins,
    /// so a truncated alignment is handled the same way the CPU loop handles it.
    ///
    /// On the GPU this is two compute kernels per batch against the already resident
    /// packed model; on the CPU it is the per-frame
    /// [`FmllrDiagGmmAccs::accumulate_from_posteriors`] path. The two agree to 1e-4
    /// relative on `beta`, `K` and `G` (see the module's rounding note).
    ///
    /// # Panics
    ///
    /// If the three slices have different lengths, or a feature dim or accumulator
    /// dim disagrees with the model.
    pub fn fmllr_accumulate_batch(
        &self,
        feats: &[&Feats],
        pdf_per_frame: &[&[PdfId]],
        weights: &[&[f32]],
        am: &AmDiagGmm,
        accs: &mut FmllrDiagGmmAccs,
    ) {
        assert_eq!(
            feats.len(),
            pdf_per_frame.len(),
            "one pdf list per utterance"
        );
        assert_eq!(feats.len(), weights.len(), "one weight list per utterance");
        assert_eq!(
            accs.dim(),
            am.dim(),
            "fMLLR accs dim does not match the model"
        );
        for f in feats {
            assert_eq!(
                f.ncols(),
                am.dim(),
                "feature dim {} does not match model dim {}",
                f.ncols(),
                am.dim()
            );
        }
        if feats.is_empty() {
            return;
        }
        match &self.backend {
            super::Backend::Gpu(ctx) => {
                let pipes = ctx.fmllr_pipelines();
                accumulate(ctx, pipes, feats, pdf_per_frame, weights, am, accs)
            }
            super::Backend::Cpu => accumulate_cpu(feats, pdf_per_frame, weights, am, accs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmm::DiagGmm;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// A small random acoustic model plus random frames and alignments.
    fn fixture(
        dim: usize,
        num_pdfs: usize,
        frames: usize,
    ) -> (AmDiagGmm, Feats, Vec<PdfId>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(0xf17ab);
        let mut am = AmDiagGmm::new();
        for p in 0..num_pdfs {
            let num_gauss = 2 + p % 3;
            let mut g = DiagGmm::new(num_gauss, dim);
            for i in 0..num_gauss {
                for d in 0..dim {
                    let mean: f32 = rng.random_range(-1.0..1.0);
                    let var: f32 = rng.random_range(0.5..2.0);
                    g.inv_vars[[i, d]] = 1.0 / var;
                    g.means_invvars[[i, d]] = mean / var;
                }
                g.weights[i] = 1.0 / num_gauss as f32;
            }
            g.compute_gconsts();
            am.add_pdf(g);
        }
        let feats = Feats::from_shape_fn((frames, dim), |_| rng.random_range(-2.0f32..2.0));
        let pdfs: Vec<PdfId> = (0..frames)
            .map(|_| rng.random_range(0..num_pdfs) as PdfId)
            .collect();
        // A mix of skipped, weighted and full-weight frames.
        let weights: Vec<f32> = (0..frames)
            .map(|t| match t % 5 {
                0 => 0.0,
                1 => 0.25,
                _ => 1.0,
            })
            .collect();
        (am, feats, pdfs, weights)
    }

    fn close(name: &str, got: f64, want: f64) {
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{name}: gpu {got} vs cpu {want} (tol {tol})"
        );
    }

    #[test]
    fn gpu_matches_cpu_accs() {
        let Some(dev) = super::super::Device::gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (dim, num_pdfs, frames) = (8usize, 6usize, 401usize);
        let (am, feats, pdfs, weights) = fixture(dim, num_pdfs, frames);

        let mut want = FmllrDiagGmmAccs::new(dim);
        accumulate_cpu(&[&feats], &[&pdfs], &[&weights[..]], &am, &mut want);

        let mut got = FmllrDiagGmmAccs::new(dim);
        dev.fmllr_accumulate_batch(&[&feats], &[&pdfs], &[&weights[..]], &am, &mut got);

        close("beta", got.count(), want.count());
        for i in 0..dim {
            for j in 0..=dim {
                close(&format!("K[{i}][{j}]"), got.k_at(i, j), want.k_at(i, j));
            }
        }
        for i in 0..dim {
            for r in 0..=dim {
                for c in 0..=dim {
                    close(
                        &format!("G[{i}][{r}][{c}]"),
                        got.g_at(i, r, c),
                        want.g_at(i, r, c),
                    );
                }
            }
        }
    }

    #[test]
    fn cpu_device_takes_the_fallback_path() {
        let (dim, num_pdfs, frames) = (4usize, 3usize, 37usize);
        let (am, feats, pdfs, weights) = fixture(dim, num_pdfs, frames);
        let mut a = FmllrDiagGmmAccs::new(dim);
        super::super::Device::cpu().fmllr_accumulate_batch(
            &[&feats],
            &[&pdfs],
            &[&weights[..]],
            &am,
            &mut a,
        );
        let mut b = FmllrDiagGmmAccs::new(dim);
        accumulate_cpu(&[&feats], &[&pdfs], &[&weights[..]], &am, &mut b);
        assert_eq!(a.count(), b.count());
    }
}
