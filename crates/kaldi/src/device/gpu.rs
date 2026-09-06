//! wgpu compute backend for batched diagonal-GMM scoring.
//!
//! One fused WGSL kernel (`shaders.wgsl::score_pdfs`) computes per-(frame, pdf)
//! log-likelihoods directly: the dot product against each of the pdf's gaussian
//! rows and an online log-sum-exp, with no per-gaussian intermediate. The whole
//! packed model stays resident on the GPU, keyed by `am_version`; each job
//! (utterance) scores only the pdfs it lists, and a batch of jobs is one submit
//! through dynamic uniform offsets into a job table. Feature uploads, pdf lists
//! and score readbacks go through persistent buffers that grow on demand, so a
//! steady-state call allocates nothing.

use std::sync::Mutex;

use ndarray::Array2;
use wgpu::util::DeviceExt;

use crate::types::PdfId;

/// Hard cap on any single buffer we allocate. Jobs are grouped so the feature
/// upload and the score matrix of a group stay under this.
pub(crate) const MAX_BUFFER_BYTES: u64 = 256 * 1024 * 1024;

/// Must match the shader constants.
const FRAMES_PER_WG: u32 = 32;
const FRAME_THREADS: u32 = 8;
const PDFS_PER_WG: u32 = 16;
/// Widest feature row the shader tile holds (`1 + 2*dim`, padded to 4), i.e. dim <= 47.
pub(crate) const MAX_WIDTH: usize = 96;
const WORKGROUP_STORAGE_BYTES: u32 = FRAMES_PER_WG * MAX_WIDTH as u32 * 4;
/// Bytes per `Job` uniform record in the shader.
const JOB_BYTES: u64 = 32;

/// One utterance to score: expanded rows `[frames, width]` (unpadded) and the
/// pdfs it needs, in output-column order.
pub(crate) struct ScoreJob<'a> {
    pub expanded: &'a [f32],
    pub frames: usize,
    pub sel: &'a [PdfId],
}

/// Round a row width up to a multiple of 4 so rows can be read as vec4s.
fn padded_width(width: usize) -> usize {
    width.div_ceil(4) * 4
}

/// Append `[rows, width]` re-laid as `[rows, padded]` with zero fill.
fn pad_rows_into(out: &mut Vec<f32>, src: &[f32], rows: usize, width: usize, padded: usize) {
    if width == padded {
        out.extend_from_slice(src);
        return;
    }
    for r in 0..rows {
        out.extend_from_slice(&src[r * width..(r + 1) * width]);
        out.resize(out.len() + (padded - width), 0.0);
    }
}

/// The packed model resident in GPU memory.
struct ResidentModel {
    version: u64,
    rows: wgpu::Buffer,
    seg_start: wgpu::Buffer,
    seg_end: wgpu::Buffer,
    /// Padded row width in floats.
    width: u32,
    num_pdfs: usize,
}

/// Persistent per-submit buffers, grown when a group needs more.
struct Scratch {
    feats: wgpu::Buffer,
    feats_cap: u64,
    sel: wgpu::Buffer,
    sel_cap: u64,
    jobs: wgpu::Buffer,
    jobs_cap: u64,
    scores: wgpu::Buffer,
    readback: wgpu::Buffer,
    scores_cap: u64,
}

pub(crate) struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    max_buffer_bytes: u64,
    max_workgroups: u32,
    job_stride: u64,
    model: Mutex<Option<ResidentModel>>,
    scratch: Mutex<Option<Scratch>>,
}

impl GpuContext {
    /// Try to bring up a high-performance adapter. Returns `None` if no
    /// suitable adapter or device is available.
    pub(crate) fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;

        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        let adapter_limits = adapter.limits();
        if adapter_limits.max_compute_workgroup_size_x < FRAME_THREADS
            || adapter_limits.max_compute_workgroup_size_y < PDFS_PER_WG
            || adapter_limits.max_compute_invocations_per_workgroup < FRAME_THREADS * PDFS_PER_WG
            || adapter_limits.max_compute_workgroup_storage_size < WORKGROUP_STORAGE_BYTES
        {
            return None;
        }
        let limits = adapter_limits.clone();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("viter-kaldi-device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viter-kaldi-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("score-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(JOB_BYTES),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("score-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("score-pdfs"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("score_pdfs"),
            compilation_options: Default::default(),
            cache: None,
        });

        let max_buffer_bytes = MAX_BUFFER_BYTES
            .min(adapter_limits.max_buffer_size)
            .min(adapter_limits.max_storage_buffer_binding_size as u64);
        let job_stride = (adapter_limits.min_uniform_buffer_offset_alignment as u64).max(JOB_BYTES);

        Some(Self {
            device,
            queue,
            adapter_name,
            pipeline,
            layout,
            max_buffer_bytes,
            max_workgroups: adapter_limits.max_compute_workgroups_per_dimension,
            job_stride,
            model: Mutex::new(None),
            scratch: Mutex::new(None),
        })
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Score every job against the model `(am_version, packed_rows, offsets)`,
    /// returning one `[frames, sel.len()]` matrix per job in job order.
    pub(crate) fn score_jobs(
        &self,
        jobs: &[ScoreJob<'_>],
        am_version: u64,
        packed_rows: &Array2<f32>,
        offsets: &[u32],
    ) -> Vec<Array2<f32>> {
        let width = packed_rows.ncols();
        assert!(
            width <= MAX_WIDTH,
            "feature width {width} exceeds the GPU kernel tile ({MAX_WIDTH}); use the CPU device"
        );
        let mut model = self.model.lock().expect("gpu model mutex poisoned");
        if model.as_ref().map(|m| m.version) != Some(am_version) {
            *model = Some(self.upload_model(am_version, packed_rows, offsets));
        }
        let resident = model.as_ref().expect("just populated");
        let padded = resident.width as usize;

        let mut scratch = self.scratch.lock().expect("gpu scratch mutex poisoned");
        let mut out: Vec<Array2<f32>> = Vec::with_capacity(jobs.len());
        for (start, end) in self.job_groups(jobs, padded) {
            out.extend(self.run_group(&mut scratch, resident, &jobs[start..end]));
        }
        out
    }

    /// Split jobs into contiguous groups whose feature upload and score output
    /// each fit under the buffer cap. A single oversized job is still one group.
    fn job_groups(&self, jobs: &[ScoreJob<'_>], padded: usize) -> Vec<(usize, usize)> {
        let cap = self.max_buffer_bytes as usize;
        let max_frames = (self.max_workgroups as usize) * FRAMES_PER_WG as usize;
        let mut groups = Vec::new();
        let (mut start, mut fb, mut sb) = (0usize, 0usize, 0usize);
        for (i, j) in jobs.iter().enumerate() {
            let f = j.frames * padded * 4;
            let s = j.frames * j.sel.len() * 4;
            assert!(j.frames <= max_frames, "utterance too long for one dispatch");
            if i > start && (fb + f > cap || sb + s > cap) {
                groups.push((start, i));
                start = i;
                fb = 0;
                sb = 0;
            }
            fb += f;
            sb += s;
        }
        if start < jobs.len() {
            groups.push((start, jobs.len()));
        }
        groups
    }

    fn upload_model(&self, version: u64, packed_rows: &Array2<f32>, offsets: &[u32]) -> ResidentModel {
        let width = packed_rows.ncols();
        let padded = padded_width(width);
        let rows = packed_rows.nrows();
        let src = packed_rows.as_standard_layout();
        let src = src.as_slice().expect("standard layout");
        let mut data = Vec::with_capacity(rows * padded);
        pad_rows_into(&mut data, src, rows, width, padded);
        let num_pdfs = offsets.len().saturating_sub(1);
        let starts: Vec<u32> = offsets[..num_pdfs].to_vec();
        let ends: Vec<u32> = offsets[1..].to_vec();
        ResidentModel {
            version,
            rows: self.storage_init("packed-rows", bytemuck::cast_slice(&pad_f32(&data))),
            seg_start: self.storage_init("seg-start", bytemuck::cast_slice(&pad_u32(&starts))),
            seg_end: self.storage_init("seg-end", bytemuck::cast_slice(&pad_u32(&ends))),
            width: padded as u32,
            num_pdfs,
        }
    }

    /// Make sure the persistent buffers can hold this group.
    fn ensure_scratch(&self, scratch: &mut Option<Scratch>, feats: u64, sel: u64, jobs: u64, scores: u64) {
        let fits = scratch.as_ref().is_some_and(|s| {
            s.feats_cap >= feats && s.sel_cap >= sel && s.jobs_cap >= jobs && s.scores_cap >= scores
        });
        if fits {
            return;
        }
        let grow = |old: Option<u64>, need: u64| old.unwrap_or(0).max(need).max(4);
        let feats_cap = grow(scratch.as_ref().map(|s| s.feats_cap), feats);
        let sel_cap = grow(scratch.as_ref().map(|s| s.sel_cap), sel);
        let jobs_cap = grow(scratch.as_ref().map(|s| s.jobs_cap), jobs);
        let scores_cap = grow(scratch.as_ref().map(|s| s.scores_cap), scores);
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        *scratch = Some(Scratch {
            feats: mk("feats", feats_cap, U::STORAGE | U::COPY_DST),
            feats_cap,
            sel: mk("sel", sel_cap, U::STORAGE | U::COPY_DST),
            sel_cap,
            jobs: mk("jobs", jobs_cap, U::UNIFORM | U::COPY_DST),
            jobs_cap,
            scores: mk("scores", scores_cap, U::STORAGE | U::COPY_SRC),
            readback: mk("scores-readback", scores_cap, U::MAP_READ | U::COPY_DST),
            scores_cap,
        });
    }

    /// One submit for a group of jobs: upload features, pdf lists and the job
    /// table, one dispatch per job, one readback.
    fn run_group(
        &self,
        scratch: &mut Option<Scratch>,
        model: &ResidentModel,
        jobs: &[ScoreJob<'_>],
    ) -> Vec<Array2<f32>> {
        let padded = model.width as usize;
        let w4 = model.width / 4;
        let _t0 = std::time::Instant::now();

        let total_feat: usize = jobs.iter().map(|j| j.frames * padded).sum();
        let total_sel: usize = jobs.iter().map(|j| j.sel.len()).sum();
        let total_out: usize = jobs.iter().map(|j| j.frames * j.sel.len()).sum();
        let mut feats = Vec::with_capacity(total_feat);
        let mut sel: Vec<u32> = Vec::with_capacity(total_sel);
        let stride = self.job_stride as usize;
        let mut table = vec![0u8; jobs.len() * stride];
        let mut out_off = 0usize;
        for (i, j) in jobs.iter().enumerate() {
            let width = j.expanded.len() / j.frames.max(1);
            let rec = [
                j.frames as u32,
                w4,
                j.sel.len() as u32,
                sel.len() as u32,
                (feats.len() / 4) as u32,
                out_off as u32,
                0,
                0,
            ];
            table[i * stride..i * stride + JOB_BYTES as usize]
                .copy_from_slice(bytemuck::cast_slice(&rec));
            pad_rows_into(&mut feats, j.expanded, j.frames, width, padded);
            for &p in j.sel {
                debug_assert!((p as usize) < model.num_pdfs, "pdf {p} out of range");
                sel.push(p);
            }
            out_off += j.frames * j.sel.len();
        }

        let feats_bytes = (feats.len() * 4).max(4) as u64;
        let sel_bytes = (sel.len() * 4).max(4) as u64;
        let jobs_bytes = table.len() as u64;
        let scores_bytes = (total_out * 4).max(4) as u64;
        self.ensure_scratch(scratch, feats_bytes, sel_bytes, jobs_bytes, scores_bytes);
        let s = scratch.as_ref().expect("scratch allocated");

        let _t1 = std::time::Instant::now();
        self.queue.write_buffer(&s.feats, 0, bytemuck::cast_slice(&pad_f32(&feats)));
        self.queue.write_buffer(&s.sel, 0, bytemuck::cast_slice(&pad_u32(&sel)));
        self.queue.write_buffer(&s.jobs, 0, &table);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("score-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &s.jobs,
                        offset: 0,
                        size: wgpu::BufferSize::new(JOB_BYTES),
                    }),
                },
                bind(1, &s.feats),
                bind(2, &model.rows),
                bind(3, &model.seg_start),
                bind(4, &model.seg_end),
                bind(5, &s.sel),
                bind(6, &s.scores),
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("score-encoder") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("score-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (i, j) in jobs.iter().enumerate() {
                if j.frames == 0 || j.sel.is_empty() {
                    continue;
                }
                pass.set_bind_group(0, &bg, &[(i * stride) as u32]);
                pass.dispatch_workgroups(
                    div_ceil(j.frames as u32, FRAMES_PER_WG),
                    div_ceil(j.sel.len() as u32, PDFS_PER_WG),
                    1,
                );
            }
        }
        enc.copy_buffer_to_buffer(&s.scores, 0, &s.readback, 0, scores_bytes);
        let _t2 = std::time::Instant::now();
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        s.readback.slice(..scores_bytes).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        rx.recv().expect("map callback dropped").expect("buffer map failed");
        let _t3 = std::time::Instant::now();

        let result = {
            let view = s.readback.slice(..scores_bytes).get_mapped_range().expect("mapped range");
            let all: &[f32] = bytemuck::cast_slice(&view[..]);
            let mut off = 0usize;
            jobs.iter()
                .map(|j| {
                    let n = j.frames * j.sel.len();
                    let m = Array2::from_shape_vec((j.frames, j.sel.len()), all[off..off + n].to_vec())
                        .expect("job output shape");
                    off += n;
                    m
                })
                .collect::<Vec<_>>()
        };
        s.readback.unmap();
        tracing::debug!(
            jobs = jobs.len(),
            frames = total_feat / padded.max(1),
            stage_us = (_t1 - _t0).as_micros(),
            upload_encode_us = (_t2 - _t1).as_micros(),
            gpu_wait_us = (_t3 - _t2).as_micros(),
            readback_us = _t3.elapsed().as_micros(),
            "gpu score group"
        );
        result
    }

    fn storage_init(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }
}

fn bind<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
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

/// Storage buffers may not be zero-sized; substitute a single element.
fn pad_f32(v: &[f32]) -> Vec<f32> {
    if v.is_empty() { vec![0.0] } else { v.to_vec() }
}

fn pad_u32(v: &[u32]) -> Vec<u32> {
    if v.is_empty() { vec![0] } else { v.to_vec() }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_ceil_rounds_up_and_never_zero() {
        assert_eq!(div_ceil(0, 16), 1);
        assert_eq!(div_ceil(1, 16), 1);
        assert_eq!(div_ceil(16, 16), 1);
        assert_eq!(div_ceil(17, 16), 2);
    }

    #[test]
    fn padding_never_yields_empty() {
        assert_eq!(pad_f32(&[]).len(), 1);
        assert_eq!(pad_u32(&[]).len(), 1);
        assert_eq!(pad_f32(&[1.0, 2.0]).len(), 2);
    }

    #[test]
    fn pad_rows_zero_fills_to_multiple_of_four() {
        let mut out = Vec::new();
        pad_rows_into(&mut out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
        assert_eq!(padded_width(81), 84);
        assert_eq!(padded_width(80), 80);
    }
}
