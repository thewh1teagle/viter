//! Frozen positive-definite fMLLR statistics; no training, audio, or GPU.
//! cargo run --release -p viter-kaldi --example fmllr_bench -- [repetitions]
use std::time::Instant;
use viter_kaldi::transform::{FmllrDiagGmmAccs, FmllrOptions};

fn main() {
    let repetitions: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(5);
    let dim = 40;
    let dim1 = dim + 1;
    let beta = 2000.0;
    let mut k = vec![0.0; dim * dim1];
    let mut g = vec![0.0; dim * dim1 * dim1];
    // Each G is a positive diagonal plus an outer product. K includes a bias
    // and non-diagonal terms to exercise successive dependent row updates.
    for d in 0..dim {
        for i in 0..dim1 {
            k[d * dim1 + i] = if i == d {
                beta * 0.12
            } else {
                beta * (((d * 7 + i * 3) % 17) as f64 - 8.0) * 0.002
            };
            for j in 0..dim1 {
                let vi = ((i * 7 + d * 3) % 11) as f64 * 0.025 - 0.1;
                let vj = ((j * 7 + d * 3) % 11) as f64 * 0.025 - 0.1;
                let diagonal = if i == j {
                    1.0 + ((i + d) % 7) as f64 * 0.1
                } else {
                    0.0
                };
                g[d * dim1 * dim1 + i * dim1 + j] = beta * (diagonal + vi * vj);
            }
        }
    }
    let mut stats = FmllrDiagGmmAccs::new(dim);
    stats.add_batch_sums(beta, &k, &g);
    let opts = FmllrOptions::default();
    let mut hash = 0xcbf29ce484222325u64;
    let start = Instant::now();
    for _ in 0..repetitions {
        let (mat, improvement, count) = stats.update(&opts, None);
        assert!(improvement > 0.0 && improvement.is_finite());
        for byte in mat
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .chain(improvement.to_bits().to_le_bytes())
            .chain(count.to_bits().to_le_bytes())
        {
            hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }
    println!(
        "solves={repetitions} elapsed_ms={:.3} checksum={hash:016x}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}
