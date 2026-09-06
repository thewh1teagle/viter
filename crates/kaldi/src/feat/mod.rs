//! Kaldi-exact feature extraction and the transforms applied on top of it.
//!
//! The pipeline that `train` and `align` drive through this module, per MFA's
//! stages (see `plans/research/01_kaldi_surface.md`):
//!
//! - mono / triphone: `MfccComputer` (13) -> [`apply_cmvn`] per speaker ->
//!   [`add_deltas`] -> 39 dims
//! - lda / sat: `MfccComputer` (13) -> [`apply_cmvn`] -> [`splice`] (3, 3) ->
//!   [`apply_transform`] with the LDA+MLLT matrix (40) -> optional per-speaker
//!   fMLLR via [`apply_transform`]
//!
//! Everything here is deterministic and matches Kaldi's arithmetic; see the
//! per-item comments for the `file:line` each piece was ported from.

mod mfcc;
mod transform;
mod window;

pub use mfcc::MfccComputer;
pub use window::{MfccOptions, WindowType, num_frames};
pub use transform::{
    CmvnStats, DeltaOptions, add_deltas, apply_cmvn, apply_transform, splice,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end check of the mono/triphone feature pipeline shape and finiteness.
    #[test]
    fn mono_pipeline_yields_39_dims() {
        let opts = MfccOptions::default();
        let computer = MfccComputer::new(opts);
        let wave: Vec<f32> = (0..16_000)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * 0.3
            })
            .collect();

        let mut feats = computer.compute(&wave);
        assert_eq!(feats.shape(), &[100, 13]);

        let mut stats = CmvnStats::new(13);
        stats.accumulate(&feats);
        apply_cmvn(&mut feats, &stats, false);

        let with_deltas = add_deltas(&feats, &DeltaOptions::default());
        assert_eq!(with_deltas.shape(), &[100, 39]);
        assert!(with_deltas.iter().all(|v| v.is_finite()));
    }

    /// End-to-end check of the LDA/SAT feature pipeline.
    #[test]
    fn lda_pipeline_yields_40_dims() {
        let computer = MfccComputer::new(MfccOptions::default());
        let wave: Vec<f32> = (0..16_000)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin() * 0.3
            })
            .collect();

        let mut feats = computer.compute(&wave);
        let mut stats = CmvnStats::new(13);
        stats.accumulate(&feats);
        apply_cmvn(&mut feats, &stats, false);

        let spliced = splice(&feats, 3, 3);
        assert_eq!(spliced.shape(), &[100, 91]);

        // A stand-in LDA+MLLT matrix: [40, 91], no offset column.
        let lda = ndarray::Array2::<f32>::from_shape_fn((40, 91), |(o, i)| {
            if i == o { 1.0 } else { 0.0 }
        });
        let out = apply_transform(&spliced, &lda);
        assert_eq!(out.shape(), &[100, 40]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn num_frames_agrees_with_the_computer() {
        let computer = MfccComputer::new(MfccOptions::default());
        assert_eq!(computer.num_frames(16_000), num_frames(16_000, computer.opts()));
        assert_eq!(computer.num_frames(16_000), 100);
        assert!((computer.frame_shift_s() - 0.01).abs() < 1e-7);
        assert_eq!(computer.dim(), 13);
    }
}
