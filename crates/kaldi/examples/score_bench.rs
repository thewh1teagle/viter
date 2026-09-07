//! Time the GPU scoring kernel alone on a real model:
//! `cargo run --release -p viter-kaldi --example score_bench model.viter [jobs] [frames] [sel]`.
//!
//! Random frames, one random pdf list per job; reports GFLOPS on the useful
//! arithmetic (2 flops per padded weight per gaussian per frame) and the max
//! deviation from the CPU path.
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use viter_kaldi::device::Device;
use viter_kaldi::model::AcousticModel;
use viter_kaldi::types::{Feats, PdfId};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("model path");
    let arg = |a: Option<String>, d: usize| a.map(|s| s.parse().expect("int")).unwrap_or(d);
    let jobs = arg(args.next(), 64);
    let frames = arg(args.next(), 650);
    let nsel = arg(args.next(), 400);

    let m = AcousticModel::load(std::path::Path::new(&path)).expect("load");
    let am = &m.am;
    let dim = am.dim();
    let num_pdfs = am.num_pdfs();
    let mut rng = StdRng::seed_from_u64(7);
    let all: Vec<PdfId> = (0..num_pdfs as PdfId).collect();
    let feats: Vec<Feats> = (0..jobs)
        .map(|_| Feats::from_shape_fn((frames, dim), |_| rng.random_range(-3.0f32..3.0)))
        .collect();
    let sels: Vec<Vec<PdfId>> = (0..jobs)
        .map(|_| {
            let mut s = all.clone();
            s.shuffle(&mut rng);
            s.truncate(nsel.min(num_pdfs));
            s.sort_unstable();
            s
        })
        .collect();
    let refs: Vec<&Feats> = feats.iter().collect();
    let sel_refs: Vec<&[PdfId]> = sels.iter().map(|s| s.as_slice()).collect();

    let gauss: f64 = sels
        .iter()
        .flat_map(|s| s.iter().map(|&p| am.pdf(p).num_gauss() as f64))
        .sum();
    let padded = (1 + 2 * dim).div_ceil(4) * 4;
    let flops = gauss * frames as f64 * 2.0 * padded as f64;

    let gpu = Device::gpu().expect("gpu");
    println!(
        "{}  pdfs {}  dim {}  jobs {} x {} frames x {} sel  ({:.1} gauss/pdf)",
        gpu.adapter_name().unwrap_or("?"),
        num_pdfs,
        dim,
        jobs,
        frames,
        nsel,
        gauss / (jobs * nsel) as f64
    );
    // Warm up: model upload + pipeline.
    let out = gpu.score_batch_sel(&refs, am, &sel_refs);
    let runs = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        std::hint::black_box(gpu.score_batch_sel(&refs, am, &sel_refs));
    }
    let per = t0.elapsed().as_secs_f64() / runs as f64;
    println!(
        "gpu  {:.1} ms/batch  {:.0} GFLOPS",
        per * 1e3,
        flops / per / 1e9
    );

    if std::env::var("SKIP_CPU").is_err() {
        let cpu = Device::cpu();
        let t0 = std::time::Instant::now();
        let reference = cpu.score_batch_sel(&refs, am, &sel_refs);
        let per = t0.elapsed().as_secs_f64();
        println!(
            "cpu  {:.1} ms/batch  {:.0} GFLOPS",
            per * 1e3,
            flops / per / 1e9
        );
        let mut worst = 0f32;
        for (a, b) in out.iter().zip(reference.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                let rel = (x - y).abs() / y.abs().max(1.0);
                worst = worst.max(rel);
            }
        }
        println!("max rel diff vs cpu {worst:.2e}");
    }
}
