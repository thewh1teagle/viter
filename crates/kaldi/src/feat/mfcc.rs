//! Kaldi-exact MFCC computation: the per-frame `Compute` loop.
//!
//! Ported from `plans/kaldi/src/feat/feature-mfcc.{h,cc}` (`MfccComputer`) and
//! `plans/kaldi/src/feat/feature-common-inl.h` (`OfflineFeatureTpl::Compute`).
//! Framing, windowing, the mel filterbank and the DCT live in [`super::window`].

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::types::Feats;

use super::window::{
    DitherRng, KALDI_WAVE_SCALE, MelBanks, MfccOptions, dct_matrix, first_sample_of_frame,
    lifter_coeffs, make_window_function, num_frames,
};

/// Kaldi-exact MFCC computer.
pub struct MfccComputer {
    opts: MfccOptions,
    window_function: Vec<f32>,
    mel_banks: MelBanks,
    dct: Vec<Vec<f32>>,
    lifter: Option<Vec<f32>>,
    log_energy_floor: f32,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl MfccComputer {
    pub fn new(opts: MfccOptions) -> Self {
        assert!(
            opts.num_ceps <= opts.num_mel_bins,
            "num-ceps ({}) cannot exceed num-mel-bins ({})",
            opts.num_ceps,
            opts.num_mel_bins
        );
        let window_function = make_window_function(&opts);
        let mel_banks = MelBanks::new(&opts, opts.vtln_warp);
        let dct = dct_matrix(opts.num_ceps as usize, opts.num_mel_bins as usize);
        let lifter = if opts.cepstral_lifter != 0.0 {
            Some(lifter_coeffs(opts.cepstral_lifter, opts.num_ceps as usize))
        } else {
            None
        };
        // feature-mfcc.cc:110: only set when energy_floor > 0.
        let log_energy_floor = if opts.energy_floor > 0.0 {
            opts.energy_floor.ln()
        } else {
            0.0
        };
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(opts.padded_window_size());
        Self {
            opts,
            window_function,
            mel_banks,
            dct,
            lifter,
            log_energy_floor,
            fft,
        }
    }

    pub fn opts(&self) -> &MfccOptions {
        &self.opts
    }

    /// Output dimension: `num_ceps`.
    pub fn dim(&self) -> usize {
        self.opts.num_ceps as usize
    }

    pub fn frame_shift_s(&self) -> f32 {
        self.opts.frame_shift_s()
    }

    /// `NumFrames` for a waveform of `num_samples` samples.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        num_frames(num_samples, &self.opts)
    }

    /// Compute MFCCs for a 16 kHz mono waveform in `-1..1`.
    ///
    /// The samples are multiplied by 32768 internally, because Kaldi computes
    /// features on raw int16 sample values.
    pub fn compute(&self, samples_16k: &[f32]) -> Feats {
        let rows = self.num_frames(samples_16k.len());
        let dim = self.dim();
        if rows == 0 || samples_16k.is_empty() {
            return Feats::zeros((0, dim));
        }

        let wave: Vec<f32> = samples_16k.iter().map(|s| s * KALDI_WAVE_SCALE).collect();

        let frame_length = self.opts.window_size();
        let padded = self.opts.padded_window_size();
        let need_raw_log_energy = self.opts.use_energy && self.opts.raw_energy;

        let mut out = Feats::zeros((rows, dim));
        let mut window = vec![0.0f32; padded];
        let mut spectrum: Vec<Complex<f32>> = self.fft.make_output_vec();
        let mut scratch: Vec<Complex<f32>> = self.fft.make_scratch_vec();
        let mut power = vec![0.0f32; padded / 2 + 1];
        let mut mel_energies = vec![0.0f32; self.mel_banks.num_bins()];
        // Deterministic per-utterance dither stream; only used when dither != 0.
        let mut dither_rng = DitherRng::new(0x5EED_0000_0000_0001);

        for r in 0..rows {
            let raw_log_energy =
                self.extract_window(&wave, r, &mut window, need_raw_log_energy, &mut dither_rng);

            // Zero the padding, then FFT in place (the FFT consumes its input).
            for v in window[frame_length..padded].iter_mut() {
                *v = 0.0;
            }

            let mut energy_post_window = 0.0f32;
            if self.opts.use_energy && !self.opts.raw_energy {
                // feature-mfcc.cc:34: energy of the windowed, preemphasised frame.
                // Kaldi takes the dot product over the whole padded vector, but
                // the padding is zero so it is the same as over the frame.
                let mut e = 0.0f32;
                for &v in window[..frame_length].iter() {
                    e += v * v;
                }
                energy_post_window = e.max(f32::EPSILON).ln();
            }

            self.fft
                .process_with_scratch(&mut window, &mut spectrum, &mut scratch)
                .expect("fft buffer sizes are consistent by construction");

            // ComputePowerSpectrum (feature-functions.cc): |X_k|^2 for
            // k = 0 .. N/2, which is exactly the realfft output length.
            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }

            self.mel_banks.compute(&power, &mut mel_energies);

            // feature-mfcc.cc:53-55: floor by float epsilon, then log.
            for m in mel_energies.iter_mut() {
                *m = m.max(f32::EPSILON).ln();
            }

            let mut row = vec![0.0f32; dim];
            for (k, coeffs) in self.dct.iter().enumerate() {
                let mut acc = 0.0f32;
                for (c, m) in coeffs.iter().zip(mel_energies.iter()) {
                    acc += c * m;
                }
                row[k] = acc;
            }

            if let Some(lifter) = &self.lifter {
                for (v, l) in row.iter_mut().zip(lifter.iter()) {
                    *v *= l;
                }
            }

            if self.opts.use_energy {
                let mut e = if self.opts.raw_energy {
                    raw_log_energy
                } else {
                    energy_post_window
                };
                if self.opts.energy_floor > 0.0 && e < self.log_energy_floor {
                    e = self.log_energy_floor;
                }
                row[0] = e;
            }

            if self.opts.htk_compat {
                // feature-mfcc.cc:68: rotate C0/energy to the end.
                let energy = row[0];
                for i in 0..dim - 1 {
                    row[i] = row[i + 1];
                }
                row[dim - 1] = if self.opts.use_energy {
                    energy
                } else {
                    energy * std::f32::consts::SQRT_2
                };
            }

            for (c, v) in row.iter().enumerate() {
                out[[r, c]] = *v;
            }
        }

        out
    }

    /// `ExtractWindow` + `ProcessWindow` (feature-window.cc:166 and :137).
    /// Returns the raw log-energy when requested, else 0.
    fn extract_window(
        &self,
        wave: &[f32],
        frame: usize,
        window: &mut [f32],
        need_raw_log_energy: bool,
        dither_rng: &mut DitherRng,
    ) -> f32 {
        let frame_length = self.opts.window_size();
        let start_sample = first_sample_of_frame(frame, &self.opts);
        let wave_dim = wave.len() as i64;

        if start_sample >= 0 && start_sample + frame_length as i64 <= wave_dim {
            let s = start_sample as usize;
            window[..frame_length].copy_from_slice(&wave[s..s + frame_length]);
        } else {
            // Reflect around the beginning or end of the waveform, repeatedly
            // if needed (feature-window.cc:203-215).
            for s in 0..frame_length {
                let mut idx = s as i64 + start_sample;
                while idx < 0 || idx >= wave_dim {
                    if idx < 0 {
                        idx = -idx - 1;
                    } else {
                        idx = 2 * wave_dim - 1 - idx;
                    }
                }
                window[s] = wave[idx as usize];
            }
        }

        let frame_view = &mut window[..frame_length];

        // ProcessWindow, in Kaldi's order: dither, DC removal, raw energy,
        // preemphasis, window multiplication.
        if self.opts.dither != 0.0 {
            for v in frame_view.iter_mut() {
                *v += dither_rng.gauss() * self.opts.dither;
            }
        }

        if self.opts.remove_dc_offset {
            let sum: f32 = frame_view.iter().sum();
            let mean = sum / frame_length as f32;
            for v in frame_view.iter_mut() {
                *v -= mean;
            }
        }

        let raw_log_energy = if need_raw_log_energy {
            let mut e = 0.0f32;
            for &v in frame_view.iter() {
                e += v * v;
            }
            e.max(f32::EPSILON).ln()
        } else {
            0.0
        };

        // Preemphasize (feature-window.cc:101), in reverse so that each sample
        // uses the un-preemphasised predecessor.
        let p = self.opts.preemph;
        if p != 0.0 {
            for i in (1..frame_length).rev() {
                frame_view[i] -= p * frame_view[i - 1];
            }
            frame_view[0] -= p * frame_view[0];
        }

        for (v, w) in frame_view.iter_mut().zip(self.window_function.iter()) {
            *v *= w;
        }

        raw_log_energy
    }
}

impl Clone for MfccComputer {
    fn clone(&self) -> Self {
        Self {
            opts: self.opts.clone(),
            window_function: self.window_function.clone(),
            mel_banks: self.mel_banks.clone(),
            dct: self.dct.clone(),
            lifter: self.lifter.clone(),
            log_energy_floor: self.log_energy_floor,
            fft: Arc::clone(&self.fft),
        }
    }
}

impl std::fmt::Debug for MfccComputer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MfccComputer")
            .field("opts", &self.opts)
            .field("num_mel_bins", &self.mel_banks.num_bins())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, freq: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 16_000.0).sin() * 0.5)
            .collect()
    }

    #[test]
    fn compute_shape_and_finiteness() {
        let c = MfccComputer::new(MfccOptions::default());
        let wave = tone(16_000, 440.0);
        let f = c.compute(&wave);
        assert_eq!(f.shape(), &[100, 13]);
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn compute_on_empty_and_tiny_input() {
        let c = MfccComputer::new(MfccOptions::default());
        assert_eq!(c.compute(&[]).shape(), &[0, 13]);
        // Below half a frame shift: no frames at all with snip_edges=false.
        assert_eq!(c.compute(&tone(50, 440.0)).shape(), &[0, 13]);
        // One frame, produced by reflecting at both ends.
        let f = c.compute(&tone(100, 440.0));
        assert_eq!(f.shape(), &[1, 13]);
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn silence_hits_the_epsilon_log_floor_not_negative_infinity() {
        let c = MfccComputer::new(MfccOptions::default());
        let f = c.compute(&vec![0.0f32; 16_000]);
        assert_eq!(f.shape(), &[100, 13]);
        assert!(f.iter().all(|v| v.is_finite()));
        // All mel energies floor to eps, so every log is ln(eps) and only C0
        // (the constant DCT row) is nonzero.
        let ln_eps = f32::EPSILON.ln();
        let expected_c0 = ln_eps * 23.0 * (1.0f32 / 23.0).sqrt();
        assert!((f[[0, 0]] - expected_c0).abs() < 1e-2, "{}", f[[0, 0]]);
        for k in 1..13 {
            assert!(f[[0, k]].abs() < 1e-2, "c{k} = {}", f[[0, k]]);
        }
    }

    #[test]
    fn use_energy_replaces_c0() {
        let mut o = MfccOptions::default();
        o.use_energy = true;
        o.raw_energy = true;
        let c = MfccComputer::new(o.clone());
        let wave = tone(16_000, 440.0);
        let f = c.compute(&wave);

        let mut o2 = o.clone();
        o2.use_energy = false;
        let f2 = MfccComputer::new(o2).compute(&wave);

        // C0 differs, but the higher cepstra are untouched by the energy swap.
        assert!((f[[50, 0]] - f2[[50, 0]]).abs() > 1e-3);
        for k in 1..13 {
            assert!((f[[50, k]] - f2[[50, k]]).abs() < 1e-4);
        }
        // Raw log energy of a 0.5-amplitude tone scaled to int16 is large.
        assert!(f[[50, 0]] > 15.0, "{}", f[[50, 0]]);
    }

    #[test]
    fn energy_floor_clamps_silence_energy() {
        let mut o = MfccOptions::default();
        o.use_energy = true;
        o.raw_energy = true;
        o.energy_floor = 1.0;
        let c = MfccComputer::new(o);
        let f = c.compute(&vec![0.0f32; 16_000]);
        // ln(1.0) == 0.
        assert!(f[[0, 0]].abs() < 1e-6, "{}", f[[0, 0]]);
    }

    #[test]
    fn htk_compat_moves_c0_to_the_end() {
        let base = MfccOptions::default();
        let plain = MfccComputer::new(base.clone()).compute(&tone(16_000, 440.0));
        let mut o = base.clone();
        o.htk_compat = true;
        let htk = MfccComputer::new(o).compute(&tone(16_000, 440.0));
        for k in 0..12 {
            assert!((htk[[10, k]] - plain[[10, k + 1]]).abs() < 1e-4);
        }
        // use_energy is false here, so C0 picks up the sqrt(2) HTK scale.
        let expect = plain[[10, 0]] * std::f32::consts::SQRT_2;
        assert!((htk[[10, 12]] - expect).abs() < 1e-3);
    }

    #[test]
    fn lifter_scales_the_cepstra() {
        let base = MfccOptions::default();
        let lifted = MfccComputer::new(base.clone()).compute(&tone(16_000, 440.0));
        let mut o = base.clone();
        o.cepstral_lifter = 0.0;
        let raw = MfccComputer::new(o).compute(&tone(16_000, 440.0));
        let coeffs = lifter_coeffs(22.0, 13);
        for k in 0..13 {
            let expect = raw[[10, k]] * coeffs[k];
            assert!(
                (lifted[[10, k]] - expect).abs() < 1e-3,
                "k={k}: {} vs {expect}",
                lifted[[10, k]]
            );
        }
    }

    #[test]
    fn a_tone_produces_a_stable_feature_over_time() {
        // A steady periodic signal gives near-identical frames in the middle.
        //
        // Two details make this a fair test rather than a test of float noise.
        // (1) Every partial is a multiple of 100 Hz, which divides both the
        //     16 kHz rate and the 160-sample frame shift, so frames 40 and 60
        //     see bit-identical samples.
        // (2) A mathematically pure sine has a spectral dynamic range near 1e15,
        //     far beyond `f32`, so mel bins away from the tone would hold only
        //     FFT round-off whose log jitters by tens of nats. Kaldi behaves the
        //     same way (its power spectrum is `float` too), so we use a harmonic
        //     stack, as real voiced speech is, to give every bin real energy.
        let harmonics: Vec<f32> = (1..=40).map(|h| h as f32 * 200.0).collect();
        let wave: Vec<f32> = (0..16_000)
            .map(|i| {
                harmonics
                    .iter()
                    .map(|f| (2.0 * std::f32::consts::PI * f * i as f32 / 16_000.0).sin() * 0.01)
                    .sum()
            })
            .collect();

        let c = MfccComputer::new(MfccOptions::default());
        let f = c.compute(&wave);
        for k in 0..13 {
            let a = f[[40, k]];
            let b = f[[60, k]];
            assert!((a - b).abs() < 0.5, "k={k}: {a} vs {b}");
        }
    }

    /// Kaldi reference invariants on a short 1 kHz tone: the `snip_edges=false`
    /// frame count formula from `feature-window.cc:NumFrames`, a finite C0, and
    /// bit-exact repeatability with dither disabled.
    #[test]
    fn short_1khz_tone_matches_kaldi_framing_and_is_deterministic() {
        let num_samples = 400usize;
        let wave: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin())
            .collect();

        let mut o = MfccOptions::default();
        o.dither = 0.0;
        assert!(!o.snip_edges);
        let shift = o.window_shift(); // 160 samples at 16 kHz, 10 ms.
        assert_eq!(shift, 160);

        let c = MfccComputer::new(o);
        let f = c.compute(&wave);

        // Kaldi, snip_edges=false: num_frames = (num_samples + shift/2) / shift.
        let expect_rows = (num_samples + shift / 2) / shift;
        assert_eq!(expect_rows, 3);
        assert_eq!(f.shape(), &[expect_rows, 13]);

        for r in 0..expect_rows {
            for k in 0..13 {
                assert!(
                    f[[r, k]].is_finite(),
                    "non-finite at ({r},{k}): {}",
                    f[[r, k]]
                );
            }
            // C0 of a log-mel DCT is sqrt(1/N) * sum(log mel energies): finite and
            // well above the all-epsilon floor for a full-scale tone.
            // Kaldi scales samples to int16 range, so c0 (= sqrt(1/N) * sum of
            // 23 log mel energies) sits around +100 for a full-scale tone.
            assert!(f[[r, 0]] > 0.0 && f[[r, 0]] < 200.0, "c0 = {}", f[[r, 0]]);
        }

        // dither = 0 makes the pipeline a pure function of the samples.
        assert_eq!(f, c.compute(&wave));
    }

    #[test]
    fn snip_edges_true_drops_the_incomplete_tail() {
        let mut o = MfccOptions::default();
        o.snip_edges = true;
        let c = MfccComputer::new(o);
        let f = c.compute(&tone(16_000, 440.0));
        assert_eq!(f.shape(), &[1 + (16_000 - 400) / 160, 13]);
    }

    #[test]
    fn dither_is_deterministic() {
        let mut o = MfccOptions::default();
        o.dither = 1.0;
        let c = MfccComputer::new(o);
        let wave = tone(16_000, 440.0);
        let a = c.compute(&wave);
        let b = c.compute(&wave);
        assert_eq!(a, b);
    }

    #[test]
    fn preemphasis_changes_the_spectrum() {
        let base = MfccOptions::default();
        let with_pre = MfccComputer::new(base.clone()).compute(&tone(16_000, 440.0));
        let mut o = base;
        o.preemph = 0.0;
        let without = MfccComputer::new(o).compute(&tone(16_000, 440.0));
        let diff: f32 = (0..13)
            .map(|k| (with_pre[[50, k]] - without[[50, k]]).abs())
            .sum();
        assert!(diff > 1e-2, "preemphasis had no effect: {diff}");
    }
}
