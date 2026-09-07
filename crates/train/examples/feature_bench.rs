//! Bounded CPU-only derivation replay. No training, model changes, or GPU.
//! cargo run --release -p viter-train --example feature_bench -- CORPUS MODEL DICT [UTTS=128] [REPEATS=5]
use anyhow::{Result, ensure};
use std::{hint::black_box, path::Path, time::Instant};
use viter_kaldi::{model::AcousticModel, transform::identity_affine, types::Feats};
use viter_train::pipeline::{FeatureKind, FeatureStore, Progress};

fn measure(name: &str, repeats: usize, derive: impl Fn() -> Vec<Feats>) {
    let expected = derive();
    let bytes: usize = expected.iter().map(|f| f.len() * size_of::<f32>()).sum();
    let mut times = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        let features = derive();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        // Accuracy check is deliberately outside the timing, including bit patterns.
        assert!(
            features
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.shape() == b.shape()
                    && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()))
        );
        black_box(&features);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "view={name} median_ms={:.3} min_ms={:.3} retained_bytes={bytes} repeated_features_bitexact=true",
        times[times.len() / 2],
        times[0]
    );
    // Model the proposed zero-copy cache hit: consumers borrow the stored chunk.
    let start = Instant::now();
    for _ in 0..100_000 {
        black_box(expected.as_slice());
    }
    println!(
        "view={name} cached_borrow_ns={:.3}",
        start.elapsed().as_nanos() as f64 / 100_000.0
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 3,
        "usage: feature_bench CORPUS MODEL DICT [UTTS=128] [REPEATS=5]"
    );
    let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(128);
    let repeats: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5);
    ensure!(
        (1..=256).contains(&n) && repeats > 0,
        "requires 1..=256 utterances and positive repetitions"
    );
    let model = AcousticModel::load(Path::new(&args[1]))?;
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
    corpus.utts = (0..n).map(|i| corpus.utts[i * total / n].clone()).collect();
    let mut mfcc = model.mfcc.clone();
    mfcc.dither = 0.0;
    let (left, right) = model.splice.unwrap_or((3, 3));
    let mut store = FeatureStore::build_with(
        &corpus,
        &mfcc,
        &model.deltas.unwrap_or_default(),
        left,
        right,
        &Progress::hidden(),
    )?;
    let indices: Vec<_> = (0..n).collect();
    let frames: usize = indices.iter().map(|&u| store.num_frames(u)).sum();
    println!(
        "metadata utts={n} frames={frames} corpus_utts={total} threads={}",
        rayon::current_num_threads()
    );
    measure("deltas", repeats, || {
        store.feats_for_many(&indices, FeatureKind::Deltas)
    });
    if let Some(lda) = &model.lda {
        measure("lda", repeats, || {
            store.feats_for_many(&indices, FeatureKind::SpliceLda(lda))
        });
        // A nontrivial fixed adaptation exercises the same 40x41 transform as SAT.
        let mut affine = identity_affine(lda.nrows());
        for d in 0..lda.nrows() {
            affine[[d, d]] = 0.95 + (d % 7) as f32 * 0.01;
            affine[[d, (d + 1) % lda.nrows()]] = 0.03;
            affine[[d, lda.nrows()]] = (d as f32 - 20.0) * 0.01;
        }
        for speaker in 0..store.num_speakers() {
            store.set_fmllr(speaker, affine.clone());
        }
        measure("sat", repeats, || {
            store.feats_for_many(&indices, FeatureKind::SpliceLdaFmllr(lda))
        });
    }
    Ok(())
}
