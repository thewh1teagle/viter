//! Deterministic CPU decoder replay; no audio, corpus, training, or GPU required.
//! Run: cargo run --release -p viter-kaldi --example viterbi_bench -- [repetitions]
use ndarray::Array2;
use std::{
    hash::{Hash, Hasher},
    time::Instant,
};
use viter_kaldi::{
    align::{AlignOptions, align, graph_pdfs},
    hmm::{ContextDependency, GraphOptions, HmmTopology, TransitionModel, build_graph},
    types::Pronunciation,
};

fn main() {
    let repetitions: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(5);
    let phones: Vec<u32> = (2..42).collect();
    let topo = HmmTopology::mfa_default(&[1], &phones, 5, 3);
    let sets: Vec<Vec<u32>> = (1..42).map(|p| vec![p]).collect();
    let ctx = ContextDependency::monophone_shared(&sets, &|p| topo.num_pdf_classes(p));
    let tm = TransitionModel::new(&ctx, &topo);
    let mut fixtures = Vec::new();
    for case in 0..12 {
        let words: Vec<_> = (0..12 + case)
            .map(|w| {
                let p: Vec<_> = (0..3 + w % 4)
                    .map(|i| 2 + ((w * 7 + i * 3 + case) % 40) as u32)
                    .collect();
                let mut alt = p.clone();
                alt[1] = 2 + (alt[1] + 4) % 40;
                vec![Pronunciation::plain(p), Pronunciation::plain(alt)]
            })
            .collect();
        let graph = build_graph(&words, &tm, &ctx, &GraphOptions::default());
        let pdfs = graph_pdfs(&graph, &tm);
        let mut cols = vec![usize::MAX; tm.num_pdfs()];
        for (c, &p) in pdfs.iter().enumerate() {
            cols[p as usize] = c;
        }
        let frames = 300 + case * 47;
        // Include flat ties and progressively stronger acoustic pruning.
        let sequence: Vec<_> = words
            .iter()
            .flat_map(|ps| ps[0].phones.iter().copied())
            .collect();
        let scores = Array2::from_shape_fn((frames, pdfs.len()), |(f, c)| {
            if case % 4 == 0 {
                return 0.0;
            }
            let target = sequence[f * sequence.len() / frames];
            let matched = (0..3).any(|class| ctx.compute(&[target], class) == Some(pdfs[c]));
            let noise = ((f * 47 + c * 17 + f * c * 3) % 101) as f32 * 0.03;
            if matched { -noise } else { -50.0 - noise }
        });
        let opts = match case % 6 {
            1 => AlignOptions {
                beam: 0.01,
                retry_beam: 40.0,
                ..Default::default()
            },
            2 => AlignOptions {
                min_active: 0,
                ..Default::default()
            },
            3 => AlignOptions {
                max_active: 100,
                min_active: 20,
                ..Default::default()
            },
            4 => AlignOptions {
                beam: 40.0,
                ..Default::default()
            },
            5 => AlignOptions {
                beam: 1000.0,
                retry_beam: 1000.0,
                ..Default::default()
            },
            _ => AlignOptions::default(),
        };
        fixtures.push((graph, cols, scores, opts));
    }
    let start = Instant::now();
    let mut hash = std::hash::DefaultHasher::new();
    let mut succeeded = 0;
    for _ in 0..repetitions {
        for (graph, cols, scores, opts) in &fixtures {
            let result = align(graph, &tm, scores, &|p| cols[p as usize], opts);
            result.is_some().hash(&mut hash);
            if let Some(a) = result {
                succeeded += 1;
                a.tids.hash(&mut hash);
                a.words.hash(&mut hash);
                a.prons.hash(&mut hash);
                a.loglike.to_bits().hash(&mut hash);
            }
        }
    }
    println!(
        "decodes={} succeeded={succeeded} elapsed_ms={:.3} checksum={:016x}",
        repetitions * fixtures.len(),
        start.elapsed().as_secs_f64() * 1000.0,
        hash.finish()
    );
}
