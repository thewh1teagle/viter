//! Batched GMM statistics accumulation on the device.
//!
//! One training iteration accumulates, for every frame of every utterance, the
//! posteriors of the aligned pdf's gaussians and the weighted `x` / `x^2` sums
//! they imply. On the CPU that is a per-frame log-sum-exp plus two `dim`-long
//! axpys — the largest remaining cost of a training iteration. This module
//! moves it to the GPU with two WGSL kernels (`accum.wgsl`):
//!
//! 1. `posteriors` — one invocation per frame: dot the expanded row
//!    `[1, x, x*x]` against the aligned pdf's gaussian rows of the *resident*
//!    packed model (the same rows and the same math the scoring kernel uses),
//!    log-sum-exp, and write the weight-scaled posteriors plus the frame
//!    log-likelihood.
//! 2. `reduce` — one workgroup per (active gaussian, 64-dimension slice), one
//!    lane per output dimension: sum that gaussian's posterior, `p*x` and
//!    `p*x*x` over the frames of its pdf. The frames are counting-sorted by pdf
//!    on the CPU beforehand, so each workgroup owns a contiguous, private range
//!    and every lane writes its own output slot exactly once — no barriers, no
//!    atomics and no cross-workgroup races.
//!
//! ## Precision
//!
//! The GPU sums in f32 and the result is added into the f64 CPU accumulators
//! once per batch. A gaussian's error is bounded by roughly `n * eps * S`,
//! with `n` the frames it sees in one batch, `eps = 2^-24` and `S` the sum of
//! magnitudes. A 256-utterance batch is ~2e5 frames spread over thousands of
//! gaussians, so `n` is at most ~1e4 per gaussian in the worst (monophone,
//! silence) case: a relative error under 1e-3 for that gaussian's batch
//! contribution, and far less once batches are summed in f64. Measured against
//! the CPU path the whole-corpus accumulators agree to better than 1e-4
//! relative, which the unit test below pins.

use std::sync::Arc;

use super::{Backend, Device, gpu};
use crate::gmm::{AccumAmDiagGmm, AmDiagGmm};
use crate::types::{Feats, PdfId};

/// Bytes per `Params` uniform record in `accum.wgsl`.
const PARAMS_BYTES: u64 = 32;
const POST_THREADS: u32 = 64;
/// Lanes per reduce workgroup; must match `RED_THREADS` in `accum.wgsl`.
const REDUCE_THREADS: u32 = 64;

impl Device {
    /// Accumulate GMM statistics for a batch of utterances into `acc`.
    ///
    /// `pdf_per_frame[i][t]` is the pdf frame `t` of utterance `i` is aligned to
    /// (the caller maps transition ids through the transition model), and
    /// `weights`, when given, is a per-frame weight; `None` means 1.0
    /// throughout. Returns the summed *weighted* frame log-likelihood, the
    /// number the iteration summary divides by the frame count.
    ///
    /// `acc.total_frames` and `acc.total_loglike` are updated exactly as the
    /// per-frame [`AccumAmDiagGmm::accumulate_for_gmm`] path would.
    ///
    /// # Panics
    ///
    /// If the slice lengths disagree, a feature dimension does not match the
    /// model, or a pdf id is out of range.
    pub fn accumulate_batch(
        &self,
        feats: &[&Feats],
        pdf_per_frame: &[&[PdfId]],
        weights: Option<&[&[f32]]>,
        am: &AmDiagGmm,
        acc: &mut AccumAmDiagGmm,
    ) -> f64 {
        assert_eq!(
            feats.len(),
            pdf_per_frame.len(),
            "one pdf list per utterance"
        );
        if let Some(w) = weights {
            assert_eq!(feats.len(), w.len(), "one weight list per utterance");
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
        match &self.backend {
            Backend::Cpu => accumulate_cpu(feats, pdf_per_frame, weights, am, acc),
            Backend::Gpu(ctx) => accumulate_gpu(ctx, feats, pdf_per_frame, weights, am, acc),
        }
    }
}

/// The reference path: the per-frame CPU accumulation, parallel over utterances.
fn accumulate_cpu(
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    weights: Option<&[&[f32]]>,
    am: &AmDiagGmm,
    acc: &mut AccumAmDiagGmm,
) -> f64 {
    use rayon::prelude::*;
    let flags = acc
        .accs
        .first()
        .map(|a| a.flags)
        .unwrap_or(crate::gmm::GmmFlags::ALL);
    let partial = (0..feats.len())
        .into_par_iter()
        .fold(
            || AccumAmDiagGmm::new(am, flags),
            |mut local, i| {
                let f = feats[i];
                let pdfs = pdf_per_frame[i];
                let n = pdfs.len().min(f.nrows());
                for t in 0..n {
                    let row = f.row(t);
                    let x = row.as_slice().expect("feature rows are contiguous");
                    let w = weights.map(|w| w[i][t]).unwrap_or(1.0);
                    local.accumulate_for_gmm(am, pdfs[t], x, w);
                }
                local
            },
        )
        .reduce(
            || AccumAmDiagGmm::new(am, flags),
            |mut a, b| {
                a.add(&b, 1.0);
                a
            },
        );
    let loglike = partial.total_loglike;
    acc.add(&partial, 1.0);
    loglike
}

/// Everything the GPU pass needs for one batch, laid out flat.
struct Plan {
    /// Raw feature rows `x`, padded to `xpad` floats per frame (squares are
    /// formed on the GPU, so the upload is a third of the expanded row).
    feats: Vec<f32>,
    xpad: usize,
    frame_pdf: Vec<u32>,
    frame_weight: Vec<f32>,
    /// Start of each frame's posterior slice in `post`.
    post_off: Vec<u32>,
    post_len: usize,
    /// Frame indices sorted (counting sort) by pdf.
    bucket: Vec<u32>,
    /// Per active gaussian: bucket range, component index within its pdf.
    g_bstart: Vec<u32>,
    g_bend: Vec<u32>,
    g_comp: Vec<u32>,
    /// Per active gaussian: the pdf it belongs to, for scattering the result.
    g_pdf: Vec<u32>,
    frames: usize,
}

/// Counting-sort the batch's frames by pdf and build the gaussian work list.
fn plan(
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    weights: Option<&[&[f32]]>,
    offsets: &[u32],
    num_pdfs: usize,
    padded: usize,
    dim: usize,
) -> Plan {
    let total: usize = feats
        .iter()
        .zip(pdf_per_frame.iter())
        .map(|(f, p)| p.len().min(f.nrows()))
        .sum();

    // Expand and pad each utterance's rows straight into one buffer, in parallel:
    // utterance i owns the slice starting at its frame offset.
    use rayon::prelude::*;
    let mut starts = Vec::with_capacity(feats.len() + 1);
    let mut acc_frames = 0usize;
    for (f, p) in feats.iter().zip(pdf_per_frame.iter()) {
        starts.push(acc_frames);
        acc_frames += p.len().min(f.nrows());
    }
    let _ = padded;
    let xpad = dim.div_ceil(4) * 4;
    let mut fl = vec![0.0f32; total * xpad];
    {
        // Disjoint mutable slices per utterance.
        let mut slices: Vec<&mut [f32]> = Vec::with_capacity(feats.len());
        let mut rest: &mut [f32] = &mut fl;
        for (f, p) in feats.iter().zip(pdf_per_frame.iter()) {
            let n = p.len().min(f.nrows());
            let (head, tail) = rest.split_at_mut(n * xpad);
            slices.push(head);
            rest = tail;
        }
        slices
            .par_iter_mut()
            .zip(feats.par_iter())
            .zip(pdf_per_frame.par_iter())
            .for_each(|((dst, f), pdfs)| {
                let n = pdfs.len().min(f.nrows());
                for t in 0..n {
                    let row = f.row(t);
                    let o = t * xpad;
                    for d in 0..dim {
                        dst[o + d] = row[d];
                    }
                }
            });
    }
    let _ = starts;
    let mut frame_pdf = Vec::with_capacity(total);
    let mut frame_weight = Vec::with_capacity(total);
    let mut counts = vec![0u32; num_pdfs + 1];
    for (i, (f, pdfs)) in feats.iter().zip(pdf_per_frame.iter()).enumerate() {
        let n = pdfs.len().min(f.nrows());
        for t in 0..n {
            let p = pdfs[t];
            assert!(
                (p as usize) < num_pdfs,
                "pdf id {p} out of range (model has {num_pdfs} pdfs)"
            );
            frame_pdf.push(p as u32);
            frame_weight.push(weights.map(|w| w[i][t]).unwrap_or(1.0));
            counts[p as usize + 1] += 1;
        }
    }

    // Prefix sums: bucket bounds per pdf, and each frame's posterior offset.
    for p in 0..num_pdfs {
        counts[p + 1] += counts[p];
    }
    let bstart = counts.clone();
    let mut cursor = counts;
    let mut bucket = vec![0u32; total];
    let mut post_off = vec![0u32; total];
    let mut post_len = 0usize;
    for (t, &p) in frame_pdf.iter().enumerate() {
        let slot = &mut cursor[p as usize];
        bucket[*slot as usize] = t as u32;
        *slot += 1;
        post_off[t] = post_len as u32;
        post_len += (offsets[p as usize + 1] - offsets[p as usize]) as usize;
    }

    // One workgroup per gaussian of every pdf the batch actually touched.
    let (mut g_bstart, mut g_bend, mut g_comp, mut g_pdf) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for p in 0..num_pdfs {
        let (lo, hi) = (bstart[p], bstart[p + 1]);
        if lo == hi {
            continue;
        }
        for c in 0..(offsets[p + 1] - offsets[p]) {
            g_bstart.push(lo);
            g_bend.push(hi);
            g_comp.push(c);
            g_pdf.push(p as u32);
        }
    }

    Plan {
        feats: fl,
        xpad,
        frame_pdf,
        frame_weight,
        post_off,
        post_len,
        bucket,
        g_bstart,
        g_bend,
        g_comp,
        g_pdf,
        frames: total,
    }
}

/// Split the batch so that no single GPU allocation exceeds the device cap.
fn chunk_batch(
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    padded: usize,
    cap: u64,
) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let (mut start, mut bytes) = (0usize, 0u64);
    for i in 0..feats.len() {
        let n = pdf_per_frame[i].len().min(feats[i].nrows());
        // Features dominate; posteriors and the output are much smaller.
        let b = (n * padded * 4) as u64; // conservative: the expanded width bounds the raw one
        if i > start && bytes + b > cap / 2 {
            groups.push((start, i));
            start = i;
            bytes = 0;
        }
        bytes += b;
    }
    if start < feats.len() {
        groups.push((start, feats.len()));
    }
    groups
}

fn accumulate_gpu(
    ctx: &Arc<gpu::GpuContext>,
    feats: &[&Feats],
    pdf_per_frame: &[&[PdfId]],
    weights: Option<&[&[f32]]>,
    am: &AmDiagGmm,
    acc: &mut AccumAmDiagGmm,
) -> f64 {
    if feats.is_empty() {
        return 0.0;
    }
    let dim = am.dim();
    let width = 1 + 2 * dim;
    if width > gpu::MAX_WIDTH {
        return accumulate_cpu(feats, pdf_per_frame, weights, am, acc);
    }
    let packed = am.packed_arc();
    let pipes = AccumPipelines::get(ctx);

    ctx.with_model(am.version(), &packed.rows, &packed.offsets, |model| {
        let padded = model.width() as usize;
        let num_pdfs = model.num_pdfs();
        let mut total_loglike = 0.0f64;
        let mut total_frames = 0.0f64;
        for (s, e) in chunk_batch(feats, pdf_per_frame, padded, ctx.max_buffer_bytes()) {
            let _tp = std::time::Instant::now();
            let p = plan(
                &feats[s..e],
                &pdf_per_frame[s..e],
                weights.map(|w| &w[s..e]),
                &packed.offsets,
                num_pdfs,
                padded,
                dim,
            );
            tracing::debug!(plan_us = _tp.elapsed().as_micros() as u64, "accum plan");
            if p.frames == 0 {
                continue;
            }
            let _t = std::time::Instant::now();
            let (out, loglike) = run(ctx, &pipes, model, &p, dim);
            tracing::debug!(
                run_us = _t.elapsed().as_micros() as u64,
                frames = p.frames,
                active = p.g_bstart.len(),
                "accum gpu run"
            );
            // Scatter the f32 sums into the f64 accumulators.
            let ow = width;
            for (a, &pdf) in p.g_pdf.iter().enumerate() {
                let acc_pdf = &mut acc.accs[pdf as usize];
                let comp = p.g_comp[a] as usize;
                let row = &out[a * ow..(a + 1) * ow];
                acc_pdf.occupancy[comp] += row[0] as f64;
                if acc_pdf.flags.contains(crate::gmm::GmmFlags::MEANS) {
                    for d in 0..dim {
                        acc_pdf.mean_accum[[comp, d]] += row[1 + d] as f64;
                    }
                }
                if acc_pdf.flags.contains(crate::gmm::GmmFlags::VARIANCES) {
                    for d in 0..dim {
                        acc_pdf.var_accum[[comp, d]] += row[1 + dim + d] as f64;
                    }
                }
            }
            for (t, &ll) in loglike.iter().enumerate() {
                let w = p.frame_weight[t] as f64;
                total_loglike += ll as f64 * w;
                total_frames += w;
            }
        }
        acc.total_loglike += total_loglike;
        acc.total_frames += total_frames;
        total_loglike
    })
}

/// The two compute pipelines and their shared bind group layout, built once per
/// GPU context and cached for the process.
struct AccumPipelines {
    layout: wgpu::BindGroupLayout,
    posteriors: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    /// Persistent upload / output / readback buffers, grown on demand so a
    /// steady-state batch allocates nothing.
    scratch: std::sync::Mutex<Scratch>,
}

/// One persistent buffer per binding, keyed by slot.
#[derive(Default)]
struct Scratch {
    bufs: Vec<Option<(wgpu::Buffer, u64)>>,
    params: Option<wgpu::Buffer>,
    readback: Option<(wgpu::Buffer, u64)>,
}

impl Scratch {
    /// Buffer for `slot` with at least `bytes` capacity and `usage`; grows by 1.5x.
    fn get(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        bytes: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> &wgpu::Buffer {
        if self.bufs.len() <= slot {
            self.bufs.resize_with(slot + 1, || None);
        }
        let need = bytes.max(4);
        let fits = self.bufs[slot]
            .as_ref()
            .is_some_and(|(_, cap)| *cap >= need);
        if !fits {
            // Grow by 1.5x, keep sizes 256-byte aligned (binding sizes must be multiples of 4).
            let cap = (need.max(need / 2 * 3)).div_ceil(256) * 256;
            self.bufs[slot] = Some((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: cap,
                    usage,
                    mapped_at_creation: false,
                }),
                cap,
            ));
        }
        &self.bufs[slot].as_ref().expect("just ensured").0
    }
}

impl AccumPipelines {
    fn get(ctx: &Arc<gpu::GpuContext>) -> Arc<AccumPipelines> {
        use std::sync::{Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<Vec<(usize, Arc<AccumPipelines>)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        let mut g = cache.lock().expect("accum pipeline cache poisoned");
        if let Some((_, p)) = g.iter().find(|(k, _)| *k == key) {
            return Arc::clone(p);
        }
        let built = Arc::new(AccumPipelines::build(ctx.wgpu_device()));
        g.push((key, Arc::clone(&built)));
        built
    }

    fn build(device: &wgpu::Device) -> AccumPipelines {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viter-accum-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("accum.wgsl").into()),
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(PARAMS_BYTES),
            },
            count: None,
        }];
        // 1..14: storage buffers; 8, 9 and 14 are written by the kernels.
        for b in 1..=14u32 {
            entries.push(storage_entry(b, !matches!(b, 8 | 9 | 14)));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accum-layout"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("accum-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        AccumPipelines {
            posteriors: mk("posteriors"),
            reduce: mk("reduce"),
            layout,
            scratch: std::sync::Mutex::new(Scratch::default()),
        }
    }
}

/// Upload one planned chunk, run both kernels in one submit, read back the
/// per-gaussian sums and the per-frame log-likelihoods.
fn run(
    ctx: &Arc<gpu::GpuContext>,
    pipes: &AccumPipelines,
    model: &gpu::ResidentModel,
    p: &Plan,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let device = ctx.wgpu_device();
    let queue = ctx.wgpu_queue();
    let width = 1 + 2 * dim;
    let num_active = p.g_bstart.len();
    let out_len = num_active * width;
    let out_bytes = (out_len * 4).max(4) as u64;
    let ll_bytes = (p.frames * 4).max(4) as u64;

    let params: [u32; 8] = [
        p.frames as u32,
        p.xpad as u32,
        dim as u32,
        num_active as u32,
        width as u32,
        model.width(),
        0,
        0,
    ];

    use wgpu::BufferUsages as U;
    let mut sc = pipes.scratch.lock().expect("accum scratch poisoned");
    let params_buf = sc.params.get_or_insert_with(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum-params"),
            size: PARAMS_BYTES,
            usage: U::UNIFORM | U::COPY_DST,
            mapped_at_creation: false,
        })
    });
    queue.write_buffer(params_buf, 0, bytemuck::cast_slice(&params));
    let params_buf = params_buf.clone();

    let up = U::STORAGE | U::COPY_DST;
    let rw = U::STORAGE | U::COPY_SRC;
    let _b0 = std::time::Instant::now();
    // Inputs: upload through persistent buffers.
    let inputs: [(usize, &[u8], &str); 8] = [
        (1, bytemuck::cast_slice(&p.feats), "accum-feats"),
        (5, bytemuck::cast_slice(&p.frame_pdf), "accum-frame-pdf"),
        (
            6,
            bytemuck::cast_slice(&p.frame_weight),
            "accum-frame-weight",
        ),
        (7, bytemuck::cast_slice(&p.post_off), "accum-post-off"),
        (10, bytemuck::cast_slice(&p.bucket), "accum-bucket"),
        (11, bytemuck::cast_slice(&p.g_bstart), "accum-g-start"),
        (12, bytemuck::cast_slice(&p.g_bend), "accum-g-end"),
        (13, bytemuck::cast_slice(&p.g_comp), "accum-g-comp"),
    ];
    for (slot, bytes, label) in inputs {
        let b = sc.get(device, slot, bytes.len() as u64, up, label);
        if !bytes.is_empty() {
            queue.write_buffer(b, 0, bytes);
        }
    }
    sc.get(device, 8, (p.post_len * 4) as u64, rw, "accum-post");
    sc.get(device, 9, ll_bytes, rw, "accum-loglike");
    sc.get(device, 14, out_bytes, rw, "accum-out");
    let rb_need = out_bytes + ll_bytes;
    if !sc.readback.as_ref().is_some_and(|(_, cap)| *cap >= rb_need) {
        let cap = (rb_need.max(rb_need / 2 * 3)).div_ceil(256) * 256;
        sc.readback = Some((
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("accum-readback"),
                size: cap,
                usage: U::MAP_READ | U::COPY_DST,
                mapped_at_creation: false,
            }),
            cap,
        ));
    }
    let buf = |slot: usize| -> &wgpu::Buffer { &sc.bufs[slot].as_ref().expect("bound").0 };
    let readback = &sc.readback.as_ref().expect("readback").0;

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("accum-bg"),
        layout: &pipes.layout,
        entries: &[
            bind(0, &params_buf),
            bind(1, buf(1)),
            bind(2, model.rows()),
            bind(3, model.seg_start()),
            bind(4, model.seg_end()),
            bind(5, buf(5)),
            bind(6, buf(6)),
            bind(7, buf(7)),
            bind(8, buf(8)),
            bind(9, buf(9)),
            bind(10, buf(10)),
            bind(11, buf(11)),
            bind(12, buf(12)),
            bind(13, buf(13)),
            bind(14, buf(14)),
        ],
    });

    let _b1 = std::time::Instant::now();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("accum-encoder"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("accum-pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.set_pipeline(&pipes.posteriors);
        pass.dispatch_workgroups((p.frames as u32).div_ceil(POST_THREADS).max(1), 1, 1);
        pass.set_pipeline(&pipes.reduce);
        pass.dispatch_workgroups(
            (num_active as u32).max(1),
            (dim as u32).div_ceil(REDUCE_THREADS).max(1),
            1,
        );
    }
    enc.copy_buffer_to_buffer(buf(14), 0, readback, 0, out_bytes);
    enc.copy_buffer_to_buffer(buf(9), 0, readback, out_bytes, ll_bytes);
    queue.submit(Some(enc.finish()));
    let _b2 = std::time::Instant::now();

    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..rb_need)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("gpu poll failed");
    rx.recv()
        .expect("map callback dropped")
        .expect("buffer map failed");
    let _b3 = std::time::Instant::now();
    tracing::debug!(
        upload_us = (_b1 - _b0).as_micros() as u64,
        encode_us = (_b2 - _b1).as_micros() as u64,
        wait_us = (_b3 - _b2).as_micros() as u64,
        "accum run breakdown"
    );
    let (out, ll) = {
        let view = readback
            .slice(..rb_need)
            .get_mapped_range()
            .expect("mapped range");
        let all: &[f32] = bytemuck::cast_slice(&view[..]);
        let split = (out_bytes / 4) as usize;
        (
            all[..out_len].to_vec(),
            all[split..split + p.frames].to_vec(),
        )
    };
    readback.unmap();
    (out, ll)
}

fn bind<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmm::{DiagGmm, GmmFlags};
    use ndarray::Array2;
    use rand::prelude::*;

    fn random_am(pdfs: usize, dim: usize, rng: &mut impl Rng) -> AmDiagGmm {
        let gmms: Vec<DiagGmm> = (0..pdfs)
            .map(|p| {
                let n = 1 + p % 4;
                let mut g = DiagGmm {
                    weights: (0..n).map(|_| rng.random_range(0.2..1.0)).collect(),
                    means_invvars: Array2::from_shape_fn((n, dim), |_| {
                        rng.random_range(-1.0..1.0f32)
                    }),
                    inv_vars: Array2::from_shape_fn((n, dim), |_| rng.random_range(0.5..2.0f32)),
                    gconsts: vec![0.0; n],
                };
                let s: f32 = g.weights.iter().sum();
                for w in g.weights.iter_mut() {
                    *w /= s;
                }
                g.compute_gconsts();
                g
            })
            .collect();
        let mut am = AmDiagGmm::new();
        for g in gmms {
            am.add_pdf(g);
        }
        am
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn gpu_accumulators_match_cpu() {
        let Some(gpu_dev) = Device::gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut rng = StdRng::seed_from_u64(7);
        let (dim, pdfs) = (13usize, 20usize);
        let am = random_am(pdfs, dim, &mut rng);

        let utts: Vec<Feats> = (0..6)
            .map(|_| {
                let n = rng.random_range(20..120);
                Array2::from_shape_fn((n, dim), |_| rng.random_range(-2.0..2.0f32))
            })
            .collect();
        let alis: Vec<Vec<PdfId>> = utts
            .iter()
            .map(|f| {
                (0..f.nrows())
                    .map(|_| rng.random_range(0..pdfs) as PdfId)
                    .collect()
            })
            .collect();
        let fref: Vec<&Feats> = utts.iter().collect();
        let aref: Vec<&[PdfId]> = alis.iter().map(|a| a.as_slice()).collect();

        let mut want = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        let ll_cpu = Device::cpu().accumulate_batch(&fref, &aref, None, &am, &mut want);
        let mut got = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        let ll_gpu = gpu_dev.accumulate_batch(&fref, &aref, None, &am, &mut got);

        assert!(close(ll_cpu, ll_gpu, 1e-4), "loglike {ll_cpu} vs {ll_gpu}");
        assert!(close(want.total_frames, got.total_frames, 1e-9));
        for p in 0..pdfs {
            let (a, b) = (&want.accs[p], &got.accs[p]);
            for g in 0..a.num_gauss() {
                assert!(
                    close(a.occupancy[g], b.occupancy[g], 1e-4),
                    "pdf {p} gauss {g} occ {} vs {}",
                    a.occupancy[g],
                    b.occupancy[g]
                );
                for d in 0..dim {
                    assert!(
                        close(a.mean_accum[[g, d]], b.mean_accum[[g, d]], 1e-4),
                        "pdf {p} gauss {g} dim {d} mean {} vs {}",
                        a.mean_accum[[g, d]],
                        b.mean_accum[[g, d]]
                    );
                    assert!(
                        close(a.var_accum[[g, d]], b.var_accum[[g, d]], 1e-4),
                        "pdf {p} gauss {g} dim {d} var {} vs {}",
                        a.var_accum[[g, d]],
                        b.var_accum[[g, d]]
                    );
                }
            }
        }
    }

    #[test]
    fn weights_scale_the_stats() {
        let mut rng = StdRng::seed_from_u64(11);
        let (dim, pdfs) = (5usize, 4usize);
        let am = random_am(pdfs, dim, &mut rng);
        let f: Feats = Array2::from_shape_fn((10, dim), |_| rng.random_range(-1.0..1.0f32));
        let ali: Vec<PdfId> = (0..10).map(|i| (i % pdfs) as PdfId).collect();
        let w = vec![0.5f32; 10];
        let dev = Device::cpu();
        let mut a = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        dev.accumulate_batch(&[&f], &[&ali[..]], Some(&[&w[..]]), &am, &mut a);
        let mut b = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        dev.accumulate_batch(&[&f], &[&ali[..]], None, &am, &mut b);
        for p in 0..pdfs {
            for g in 0..a.accs[p].num_gauss() {
                assert!(close(
                    a.accs[p].occupancy[g] * 2.0,
                    b.accs[p].occupancy[g],
                    1e-6
                ));
            }
        }
        assert!(close(a.total_frames * 2.0, b.total_frames, 1e-9));
    }

    #[test]
    fn empty_batch_is_a_noop() {
        let mut rng = StdRng::seed_from_u64(3);
        let am = random_am(3, 4, &mut rng);
        let mut a = AccumAmDiagGmm::new(&am, GmmFlags::ALL);
        assert_eq!(
            Device::cpu().accumulate_batch(&[], &[], None, &am, &mut a),
            0.0
        );
        assert_eq!(a.total_frames, 0.0);
    }
}
