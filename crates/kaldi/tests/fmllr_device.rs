//! GPU fMLLR coverage for cached Gaussian scores, their fallback, and tile tails.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use viter_kaldi::device::Device;
use viter_kaldi::gmm::{AmDiagGmm, DiagGmm};
use viter_kaldi::transform::{FmllrDiagGmmAccs, FmllrOptions};
use viter_kaldi::types::{Feats, PdfId};

fn model(dim: usize, rng: &mut StdRng) -> AmDiagGmm {
    let mut am = AmDiagGmm::new();
    // Both sides of the 128-score cache boundary, plus a larger fallback PDF.
    for components in [1, 3, 127, 128, 129, 257] {
        let mut gmm = DiagGmm::new(components, dim);
        for g in 0..components {
            for d in 0..dim {
                // A remote component exercises exactly-zero posteriors.
                let mean = if g == components - 1 && components > 1 {
                    20.0
                } else {
                    rng.random_range(-1.0..1.0)
                };
                let variance = rng.random_range(0.5..2.0);
                gmm.inv_vars[[g, d]] = 1.0 / variance;
                gmm.means_invvars[[g, d]] = mean / variance;
            }
            gmm.weights[g] = 1.0 / components as f32;
        }
        gmm.compute_gconsts();
        am.add_pdf(gmm);
    }
    am
}

fn close(label: &str, actual: f64, expected: f64) {
    let error = (actual - expected).abs() / actual.abs().max(expected.abs()).max(1.0);
    assert!(
        error <= 1e-4,
        "{label}: GPU {actual}, CPU {expected}, error {error}"
    );
}

fn compare(got: &FmllrDiagGmmAccs, want: &FmllrDiagGmmAccs) {
    close("count", got.count(), want.count());
    for i in 0..want.dim() {
        for r in 0..=want.dim() {
            close("K", got.k_at(i, r), want.k_at(i, r));
            for c in 0..=want.dim() {
                close("G", got.g_at(i, r, c), want.g_at(i, r, c));
            }
        }
    }
}

#[test]
fn cached_and_fallback_gaussians_match_cpu_stats_and_transforms() {
    let Some(gpu) = Device::gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xf17_128);
    // The final case retains 4097 nonzero-weight frames after compaction: it
    // exercises the cached pipeline and its final partial 64-frame workgroup.
    for (dim, frames) in [(13, 65), (39, 129), (40, 129), (47, 129), (13, 5122)] {
        let am = model(dim, &mut rng);
        let feats = Feats::from_shape_fn((frames, dim), |_| rng.random_range(-2.0..2.0));
        let pdfs: Vec<PdfId> = (0..frames).map(|t| (t % am.num_pdfs()) as PdfId).collect();
        let weights: Vec<f32> = (0..frames)
            .map(|t| [0.0, 0.25, 1.0, 1.75, 1.0][t % 5])
            .collect();
        let mut want = FmllrDiagGmmAccs::new(dim);
        let mut got = FmllrDiagGmmAccs::new(dim);
        Device::cpu().fmllr_accumulate_batch(&[&feats], &[&pdfs], &[&weights], &am, &mut want);
        gpu.fmllr_accumulate_batch(&[&feats], &[&pdfs], &[&weights], &am, &mut got);
        compare(&got, &want);

        // Enough independent live frames for a full-rank affine solve; force a
        // real update instead of the default low-count identity shortcut.
        let options = FmllrOptions {
            min_count: 1.0,
            ..Default::default()
        };
        let (expected, expected_improvement, _) = want.update(&options, None);
        let (actual, actual_improvement, _) = got.update(&options, None);
        close("improvement", actual_improvement, expected_improvement);
        for (&a, &b) in actual.iter().zip(expected.iter()) {
            close("transform", a as f64, b as f64);
        }
    }
}

#[test]
fn empty_and_zero_weight_batches_preserve_existing_stats() {
    let Some(gpu) = Device::gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut rng = StdRng::seed_from_u64(71);
    let am = model(13, &mut rng);
    let feats = Feats::from_shape_fn((65, 13), |_| rng.random_range(-2.0..2.0));
    let pdfs = vec![0; 65];
    let mut got = FmllrDiagGmmAccs::new(13);
    gpu.fmllr_accumulate_batch(&[&feats], &[&pdfs], &[&[1.0; 65]], &am, &mut got);
    let expected = got.clone();
    gpu.fmllr_accumulate_batch(&[], &[], &[], &am, &mut got);
    gpu.fmllr_accumulate_batch(&[&Feats::zeros((0, 13))], &[&[]], &[&[]], &am, &mut got);
    gpu.fmllr_accumulate_batch(&[&feats], &[&pdfs], &[&[0.0; 65]], &am, &mut got);
    assert_eq!(got.count(), expected.count());
    for i in 0..13 {
        for r in 0..=13 {
            assert_eq!(got.k_at(i, r), expected.k_at(i, r));
            for c in 0..=13 {
                assert_eq!(got.g_at(i, r, c), expected.g_at(i, r, c));
            }
        }
    }
}
