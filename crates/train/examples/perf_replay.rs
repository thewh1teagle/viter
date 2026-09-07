//! Bounded real-corpus replay; never trains or updates the frozen acoustic model.
//! cargo run --release -p viter-train --example perf_replay --
//!   data/ljfull data/ljfull-v12.viter data/dict/ljspeech_ipa_noprobs.dict 128 3 64 /tmp/replay
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use viter_kaldi::{align, device::Device, model::AcousticModel, types::Alignment};
use viter_train::pipeline::{FeatureKind, FeatureStore, GraphSet, Progress, stats};

fn phase(name: &str, repeat: usize, start: Instant) {
    println!(
        "phase={name} repeat={repeat} ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

fn dump(prefix: &str, alignments: &[Option<Alignment>], stats: &stats::Stats) -> Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.align"))?);
    for ali in alignments {
        match ali {
            None => writeln!(file, "failed")?,
            Some(a) => writeln!(file, "{:?} {:?} {:?}", a.tids, a.words, a.prons)?,
        }
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.stats"))?);
    let mut value = |x: f64| file.write_all(&x.to_le_bytes());
    value(stats.gmm.total_frames)?;
    value(stats.gmm.total_loglike)?;
    for x in &stats.transitions.0 {
        value(*x)?;
    }
    for pdf in &stats.gmm.accs {
        for x in pdf
            .occupancy
            .iter()
            .chain(pdf.mean_accum.iter())
            .chain(pdf.var_accum.iter())
        {
            value(*x)?;
        }
    }
    Ok(())
}

/// Optional diagnostics isolate update and adaptation cost, discarding their outputs.
/// Every repetition starts from the same model/statistics; there is no training loop.
fn extra_phases(
    (repeat, output): (usize, Option<&str>),
    model: &AcousticModel,
    device: &Device,
    store: &FeatureStore,
    feats: &[viter_kaldi::types::Feats],
    alignments: &[Option<Alignment>],
    accum: &stats::Stats,
) -> Result<()> {
    use rand::SeedableRng;
    use viter_kaldi::transform::FmllrDiagGmmAccs;
    let am = model.am_si.as_ref().unwrap_or(&model.am);
    // Cloning is fixture setup, outside measured MLE work.
    let mut copied_am = am.clone();
    let mut copied_tm = model.tm.clone();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
    let t = Instant::now();
    // No removal/splitting: the bounded subset is not enough to estimate corpus-wide
    // occupancy thresholds, and this diagnostic should retain the model's shape.
    let updated = stats::update_model(
        accum,
        &mut copied_tm,
        &mut copied_am,
        &stats::UpdateOptions {
            remove_low_count_gaussians: false,
            ..Default::default()
        },
        &mut rng,
    );
    phase("mle_update", repeat, t);
    std::hint::black_box(updated);
    let t = Instant::now();
    let silence = model.tm.silence_pdfs(&model.silence_phones);
    let mut by_speaker = vec![Vec::new(); store.num_speakers()];
    let mut pdfs = Vec::with_capacity(feats.len());
    let mut weights = Vec::with_capacity(feats.len());
    for (i, ali) in alignments.iter().enumerate() {
        let p: Vec<u32> = ali
            .as_ref()
            .map(|a| {
                a.tids
                    .iter()
                    .take(feats[i].nrows())
                    .map(|&tid| model.tm.transition_id_to_pdf(tid))
                    .collect()
            })
            .unwrap_or_default();
        let w: Vec<f32> = p
            .iter()
            .map(|p| if silence.contains(p) { 0.0 } else { 1.0 })
            .collect();
        if !p.is_empty() {
            by_speaker[store.speaker_of(i)].push(i);
        }
        pdfs.push(p);
        weights.push(w);
    }
    phase("fmllr_prepare", repeat, t);
    let t = Instant::now();
    let mut stats = Vec::new();
    for entries in &by_speaker {
        if entries.is_empty() {
            continue;
        }
        let f: Vec<_> = entries.iter().map(|&i| &feats[i]).collect();
        let p: Vec<_> = entries.iter().map(|&i| pdfs[i].as_slice()).collect();
        let w: Vec<_> = entries.iter().map(|&i| weights[i].as_slice()).collect();
        let mut acc = FmllrDiagGmmAccs::new(feats[0].ncols());
        device.fmllr_accumulate_batch(&f, &p, &w, am, &mut acc);
        stats.push(acc);
    }
    phase("fmllr_accumulate", repeat, t);
    let t = Instant::now();
    let opts = model.fmllr.unwrap_or_default();
    let transforms: Vec<_> = stats.par_iter().map(|s| s.update(&opts, None).0).collect();
    phase("fmllr_solve", repeat, t);
    if let Some(prefix) = output {
        let mut file = std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.extra"))?);
        for (acc, transform) in stats.iter().zip(&transforms) {
            file.write_all(&acc.count().to_le_bytes())?;
            let d = acc.dim();
            for i in 0..d {
                for j in 0..=d {
                    file.write_all(&acc.k_at(i, j).to_le_bytes())?;
                }
                for r in 0..=d {
                    for c in 0..=d {
                        file.write_all(&acc.g_at(i, r, c).to_le_bytes())?;
                    }
                }
            }
            for &x in transform {
                file.write_all(&(x as f64).to_le_bytes())?;
            }
        }
    }
    std::hint::black_box(transforms);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 3,
        "usage: perf_replay CORPUS MODEL DICT [UTTS=128] [REPEATS=3] [BATCH=64] [OUTPUT_PREFIX]"
    );
    let n = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(128usize);
    let repeats = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(3usize);
    let batch = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(64usize);
    ensure!(
        (1..=256).contains(&n),
        "bounded replay requires 1..=256 utterances"
    );
    ensure!(
        batch > 0 && repeats > 0,
        "batch and repeats must be positive"
    );
    let t = Instant::now();
    let model = AcousticModel::load(Path::new(&args[1]))?;
    phase("load_model", 0, t);
    let t = Instant::now();
    let mut corpus = viter_io::corpus::scan(
        Path::new(&args[0]),
        &viter_io::corpus::CorpusOptions {
            dictionary: Some(args[2].clone().into()),
            position_dependent: model.position_dependent,
            ..Default::default()
        },
    )?;
    let total = corpus.utts.len();
    ensure!(total >= n, "requested {n} utterances, found {total}");
    // Even spacing samples all chapters without selecting only short utterances.
    corpus.utts = (0..n).map(|i| corpus.utts[i * total / n].clone()).collect();
    viter_io::corpus::remap(&mut corpus, &model.phones)?;
    phase("scan", 0, t);
    let progress = Progress::hidden();
    let t = Instant::now();
    let (left, right) = model.splice.unwrap_or((3, 3));
    let mut mfcc = model.mfcc.clone();
    // Repeated processes must use identical inputs, including no random MFCC dither.
    mfcc.dither = 0.0;
    let store = FeatureStore::build_with(
        &corpus,
        &mfcc,
        &model.deltas.clone().unwrap_or_default(),
        left,
        right,
        &progress,
    )?;
    phase("base_features", 0, t);
    let t = Instant::now();
    let indices: Vec<usize> = (0..n).collect();
    let kind = model
        .lda
        .as_ref()
        .map(FeatureKind::SpliceLda)
        .unwrap_or(FeatureKind::Deltas);
    let feats = store.feats_for_many(&indices, kind);
    phase("derived_features", 0, t);
    let t = Instant::now();
    let bar = progress.bar("graphs", n as u64);
    let graphs = GraphSet::build(
        n,
        |i| {
            let u = &corpus.utts[i];
            match &model.lexicon_probs {
                Some(probs) => u
                    .words
                    .iter()
                    .zip(&u.prons)
                    .map(|(w, p)| viter_train::pronprob::apply(probs, w, p))
                    .collect(),
                None => u.prons.clone(),
            }
        },
        &model.tm,
        &model.ctx,
        &model.graph_opts,
        &bar,
    );
    bar.finish();
    phase("graphs", 0, t);
    let device = if std::env::var_os("VITER_REPLAY_CPU").is_some() {
        Device::cpu()
    } else {
        Device::gpu().context("GPU unavailable; set VITER_REPLAY_CPU=1 for CPU")?
    };
    let am = model.am_si.as_ref().unwrap_or(&model.am);
    let frames: usize = feats.iter().map(|f| f.nrows()).sum();
    println!(
        "metadata utts={n} frames={frames} pdfs={} gaussians={} dims={} batch={batch} threads={} device={:?}",
        am.num_pdfs(),
        am.num_gauss(),
        feats[0].ncols(),
        rayon::current_num_threads(),
        device.adapter_name()
    );
    let opts = align::AlignOptions::default();
    let refs: Vec<_> = feats.iter().collect();
    let sels: Vec<_> = (0..n).map(|i| graphs.pdfs(i)).collect();
    let mut fixed = None;
    for repeat in 0..=repeats {
        // Round zero warms kernels and allocator; later rounds are measured steady state.
        let t = Instant::now();
        let mut scores = Vec::with_capacity(n);
        for (f, p) in refs.chunks(batch).zip(sels.chunks(batch)) {
            scores.extend(device.score_batch_sel(f, am, p));
        }
        phase("score", repeat, t);
        let t = Instant::now();
        let alignments: Vec<_> = scores
            .par_iter()
            .enumerate()
            .map(|(i, scores)| {
                let sel = graphs.pdfs(i);
                let mut cols =
                    vec![usize::MAX; sel.iter().copied().max().unwrap_or(0) as usize + 1];
                for (col, &pdf) in sel.iter().enumerate() {
                    cols[pdf as usize] = col;
                }
                align::align(
                    graphs.graph(i),
                    &model.tm,
                    scores,
                    &|p| cols[p as usize],
                    &opts,
                )
            })
            .collect();
        phase("viterbi", repeat, t);
        let expected = fixed.get_or_insert_with(|| alignments.clone());
        ensure!(
            expected
                .iter()
                .zip(&alignments)
                .all(|(a, b)| a.as_ref().map(|x| (&x.tids, &x.words, &x.prons))
                    == b.as_ref().map(|x| (&x.tids, &x.words, &x.prons))),
            "alignment changed across repetitions"
        );
        let t = Instant::now();
        let bar = progress.bar("align", n as u64);
        let out = viter_train::pipeline::align::align_batch(
            &graphs, &model.tm, am, &device, &feats, &opts, batch, &bar,
        );
        bar.finish();
        phase("align_batch", repeat, t);
        ensure!(
            expected
                .iter()
                .zip(&out.alignments)
                .all(|(a, b)| a.as_ref().map(|x| (&x.tids, &x.words, &x.prons))
                    == b.as_ref().map(|x| (&x.tids, &x.words, &x.prons))),
            "align_batch differs from isolated Viterbi"
        );
        let t = Instant::now();
        let bar = progress.bar("accumulate", n as u64);
        let accum = stats::accumulate(&device, am, &model.tm, expected, &feats, &bar);
        bar.finish();
        phase("accumulate", repeat, t);
        println!(
            "result repeat={repeat} failed={} frames={} loglike={:.12}",
            out.failed,
            accum.total_frames(),
            accum.loglike_per_frame()
        );
        if std::env::var_os("VITER_REPLAY_EXTRA").is_some() {
            extra_phases(
                (
                    repeat,
                    if repeat == repeats {
                        args.get(6).map(String::as_str)
                    } else {
                        None
                    },
                ),
                &model,
                &device,
                &store,
                &feats,
                expected,
                &accum,
            )?;
        }
        if repeat == repeats {
            if let Some(prefix) = args.get(6) {
                dump(prefix, expected, &accum)?;
            }
        }
    }
    Ok(())
}
