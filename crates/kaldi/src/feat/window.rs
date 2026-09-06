//! Kaldi's frame extraction, analysis windows, mel filterbanks and DCT.
//!
//! Ported from `plans/kaldi/src/feat/feature-window.{h,cc}` (framing, dither,
//! DC offset removal, preemphasis, window functions) and
//! `plans/kaldi/src/feat/mel-computations.{h,cc}` (mel scale, `MelBanks` with
//! VTLN warping, lifter coefficients). The DCT matrix is
//! `ComputeDctMatrix` from `plans/kaldi/src/matrix/matrix-functions.cc:592`.
//!
//! Everything follows Kaldi's arithmetic exactly, including where it uses
//! `f32` (`logf`/`expf` in the mel scale, `BaseFloat` accumulation in the window
//! and the mel bins) versus `f64` (the window function and DCT matrix are built
//! in double precision and rounded to `f32`, matching Kaldi's `Vector<BaseFloat>`
//! assignment from `double` expressions).

/// Kaldi scales -1..1 float waveforms to the int16 range before feature
/// extraction, because `WaveData` holds raw int16 sample values.
pub(crate) const KALDI_WAVE_SCALE: f32 = 32768.0;

/// Analysis window shapes, matching Kaldi's `window_type` strings
/// (feature-window.cc:116-133).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WindowType {
    Hamming,
    Hanning,
    /// Kaldi's own window: `pow(0.5 - 0.5*cos(a*i), 0.85)`.
    Povey,
    Rectangular,
    Blackman,
    /// `sin(0.5 * a * i)`.
    Sine,
}

/// Options for Kaldi's `Mfcc`, flattened from `FrameExtractionOptions`,
/// `MelBanksOptions` and `MfccOptions`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MfccOptions {
    pub sample_rate: u32,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub dither: f32,
    pub preemph: f32,
    pub remove_dc_offset: bool,
    pub window: WindowType,
    pub round_to_power_of_two: bool,
    pub snip_edges: bool,
    pub num_mel_bins: u32,
    pub low_freq: f32,
    /// Upper mel cutoff. `<= 0` means "offset from Nyquist"
    /// (mel-computations.cc:46-49).
    pub high_freq: f32,
    pub num_ceps: u32,
    pub use_energy: bool,
    pub energy_floor: f32,
    pub raw_energy: bool,
    pub cepstral_lifter: f32,
    pub htk_compat: bool,
    pub vtln_warp: f32,
    /// Generalized Blackman coefficient (feature-window.h:62).
    pub blackman_coeff: f32,
    /// Lower inflection point of the VTLN warping function.
    pub vtln_low: f32,
    /// Upper inflection point; if negative, offset from Nyquist.
    pub vtln_high: f32,
}

impl Default for MfccOptions {
    /// MFA's defaults, which override several Kaldi defaults.
    ///
    /// Sources: `plans/mfa/montreal_forced_aligner/corpus/features.py`
    /// (`FeatureConfigMixin.__init__`) and `plans/kalpy/kalpy/feat/mfcc.py`
    /// (`MfccComputer.__init__`), which supplies anything MFA leaves unset.
    fn default() -> Self {
        Self {
            // features.py:619 sample_frequency=16000
            sample_rate: 16_000,
            // features.py:615 frame_length=25
            frame_length_ms: 25.0,
            // features.py:614 frame_shift=10
            frame_shift_ms: 10.0,
            // features.py:622 dither=0.0 (MFA disables dither; Kaldi's default is 1.0)
            dither: 0.0,
            // features.py:627 preemphasis_coefficient=0.97
            preemph: 0.97,
            // mfcc.py:88 remove_dc_offset=True (not overridden by MFA)
            remove_dc_offset: true,
            // mfcc.py:89 window_type="povey" (not overridden by MFA; Kaldi default too)
            window: WindowType::Povey,
            // mfcc.py:90 round_to_power_of_two=True
            round_to_power_of_two: true,
            // features.py:616 snip_edges=False (MFA overrides Kaldi's True)
            snip_edges: false,
            // features.py:625 num_mel_bins=23
            num_mel_bins: 23,
            // features.py:617 low_frequency=20
            low_freq: 20.0,
            // features.py:618 high_frequency=7800
            high_freq: 7800.0,
            // features.py:624 num_coefficients=13
            num_ceps: 13,
            // features.py:612 use_energy=False (MFA overrides Kaldi's True: C0 is kept)
            use_energy: false,
            // features.py:623 energy_floor=0.0
            energy_floor: 0.0,
            // features.py:613 raw_energy=False (MFA overrides Kaldi's True)
            raw_energy: false,
            // features.py:626 cepstral_lifter=22
            cepstral_lifter: 22.0,
            // mfcc.py:105 htk_compatibility=False
            htk_compat: false,
            // no VTLN in MFA's alignment pipeline
            vtln_warp: 1.0,
            // mfcc.py:91 blackman_coeff=0.42
            blackman_coeff: 0.42,
            // mfcc.py:96 vtln_low=100
            vtln_low: 100.0,
            // mfcc.py:97 vtln_high=-500
            vtln_high: -500.0,
        }
    }
}

impl MfccOptions {
    /// `FrameExtractionOptions::WindowShift` (feature-window.h:106).
    pub fn window_shift(&self) -> usize {
        (self.sample_rate as f32 * 0.001 * self.frame_shift_ms) as usize
    }

    /// `FrameExtractionOptions::WindowSize` (feature-window.h:109).
    pub fn window_size(&self) -> usize {
        (self.sample_rate as f32 * 0.001 * self.frame_length_ms) as usize
    }

    /// `FrameExtractionOptions::PaddedWindowSize` (feature-window.h:112).
    pub fn padded_window_size(&self) -> usize {
        if self.round_to_power_of_two {
            round_up_to_nearest_power_of_two(self.window_size())
        } else {
            self.window_size()
        }
    }

    /// Frame shift in seconds, as used for CTM / TextGrid boundaries.
    pub fn frame_shift_s(&self) -> f32 {
        self.frame_shift_ms / 1000.0
    }
}

fn round_up_to_nearest_power_of_two(n: usize) -> usize {
    debug_assert!(n > 0);
    n.next_power_of_two()
}

/// `NumFrames` (feature-window.cc:42), with `flush = true`.
pub fn num_frames(num_samples: usize, opts: &MfccOptions) -> usize {
    let frame_shift = opts.window_shift() as i64;
    let frame_length = opts.window_size() as i64;
    let num_samples = num_samples as i64;
    if opts.snip_edges {
        if num_samples < frame_length {
            0
        } else {
            (1 + (num_samples - frame_length) / frame_shift) as usize
        }
    } else {
        ((num_samples + frame_shift / 2) / frame_shift).max(0) as usize
    }
}

/// `FirstSampleOfFrame` (feature-window.cc:30). May be negative when
/// `snip_edges = false`.
pub(super) fn first_sample_of_frame(frame: usize, opts: &MfccOptions) -> i64 {
    let frame_shift = opts.window_shift() as i64;
    if opts.snip_edges {
        frame as i64 * frame_shift
    } else {
        let midpoint = frame_shift * frame as i64 + frame_shift / 2;
        midpoint - opts.window_size() as i64 / 2
    }
}

/// `FeatureWindowFunction` (feature-window.cc:109).
pub(super) fn make_window_function(opts: &MfccOptions) -> Vec<f32> {
    let frame_length = opts.window_size();
    assert!(frame_length > 0, "frame length must be positive");
    let a = 2.0 * std::f64::consts::PI / (frame_length - 1) as f64;
    (0..frame_length)
        .map(|i| {
            let i_fl = i as f64;
            let v = match opts.window {
                WindowType::Hanning => 0.5 - 0.5 * (a * i_fl).cos(),
                WindowType::Sine => (0.5 * a * i_fl).sin(),
                WindowType::Hamming => 0.54 - 0.46 * (a * i_fl).cos(),
                WindowType::Povey => (0.5 - 0.5 * (a * i_fl).cos()).powf(0.85),
                WindowType::Rectangular => 1.0,
                WindowType::Blackman => {
                    let c = opts.blackman_coeff as f64;
                    c - 0.5 * (a * i_fl).cos() + (0.5 - c) * (2.0 * a * i_fl).cos()
                }
            };
            v as f32
        })
        .collect()
}

/// `ComputeLifterCoeffs` (mel-computations.cc:253).
pub(super) fn lifter_coeffs(q: f32, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| 1.0 + 0.5 * q * (std::f32::consts::PI * i as f32 / q).sin())
        .collect()
}

/// `ComputeDctMatrix` (matrix/matrix-functions.cc:592), returning the first
/// `num_ceps` rows of the `num_bins x num_bins` DCT-II matrix.
pub(super) fn dct_matrix(num_ceps: usize, num_bins: usize) -> Vec<Vec<f32>> {
    assert!(num_ceps > 0 && num_bins > 0);
    let n = num_bins as f64;
    let mut m = vec![vec![0.0f32; num_bins]; num_ceps];
    let normalizer0 = (1.0f64 / n).sqrt() as f32;
    for j in 0..num_bins {
        m[0][j] = normalizer0;
    }
    let normalizer = (2.0f64 / n).sqrt() as f32;
    for k in 1..num_ceps {
        for j in 0..num_bins {
            let v = std::f64::consts::PI / n * (j as f64 + 0.5) * k as f64;
            m[k][j] = normalizer * (v.cos() as f32);
        }
    }
    m
}

/// `MelBanks` (mel-computations.cc:33). Each bin is stored as the index of its
/// first nonzero FFT bin plus the triangular weights from there on.
#[derive(Clone, Debug)]
pub(crate) struct MelBanks {
    bins: Vec<(usize, Vec<f32>)>,
}

impl MelBanks {
    /// mel-computations.h:85. Note the `f32` `logf`, which Kaldi relies on.
    #[inline]
    fn mel_scale(freq: f32) -> f32 {
        1127.0f32 * (1.0f32 + freq / 700.0f32).ln()
    }

    /// mel-computations.h:81.
    #[inline]
    fn inverse_mel_scale(mel: f32) -> f32 {
        700.0f32 * ((mel / 1127.0f32).exp() - 1.0f32)
    }

    /// `VtlnWarpFreq` (mel-computations.cc:150).
    fn vtln_warp_freq(
        vtln_low_cutoff: f32,
        vtln_high_cutoff: f32,
        low_freq: f32,
        high_freq: f32,
        vtln_warp_factor: f32,
        freq: f32,
    ) -> f32 {
        if freq < low_freq || freq > high_freq {
            return freq;
        }
        let one = 1.0f32;
        let l = vtln_low_cutoff * one.max(vtln_warp_factor);
        let h = vtln_high_cutoff * one.min(vtln_warp_factor);
        let scale = 1.0f32 / vtln_warp_factor;
        let fl = scale * l;
        let fh = scale * h;
        let scale_left = (fl - low_freq) / (l - low_freq);
        let scale_right = (high_freq - fh) / (high_freq - h);
        if freq < l {
            low_freq + scale_left * (freq - low_freq)
        } else if freq < h {
            scale * freq
        } else {
            high_freq + scale_right * (freq - high_freq)
        }
    }

    /// `VtlnWarpMelFreq` (mel-computations.cc:213).
    fn vtln_warp_mel_freq(
        vtln_low_cutoff: f32,
        vtln_high_cutoff: f32,
        low_freq: f32,
        high_freq: f32,
        vtln_warp_factor: f32,
        mel_freq: f32,
    ) -> f32 {
        Self::mel_scale(Self::vtln_warp_freq(
            vtln_low_cutoff,
            vtln_high_cutoff,
            low_freq,
            high_freq,
            vtln_warp_factor,
            Self::inverse_mel_scale(mel_freq),
        ))
    }

    pub(super) fn new(opts: &MfccOptions, vtln_warp_factor: f32) -> Self {
        let num_bins = opts.num_mel_bins as usize;
        assert!(num_bins >= 3, "must have at least 3 mel bins");
        let sample_freq = opts.sample_rate as f32;
        let window_length_padded = opts.padded_window_size();
        assert!(window_length_padded % 2 == 0, "padded window must be even");
        let num_fft_bins = window_length_padded / 2;
        let nyquist = 0.5 * sample_freq;

        let low_freq = opts.low_freq;
        let high_freq = if opts.high_freq > 0.0 {
            opts.high_freq
        } else {
            nyquist + opts.high_freq
        };
        assert!(
            low_freq >= 0.0
                && low_freq < nyquist
                && high_freq > 0.0
                && high_freq <= nyquist
                && high_freq > low_freq,
            "bad mel frequency range: low {low_freq}, high {high_freq}, nyquist {nyquist}"
        );

        let fft_bin_width = sample_freq / window_length_padded as f32;
        let mel_low_freq = Self::mel_scale(low_freq);
        let mel_high_freq = Self::mel_scale(high_freq);
        let mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1) as f32;

        let vtln_low = opts.vtln_low;
        let vtln_high = if opts.vtln_high < 0.0 {
            opts.vtln_high + nyquist
        } else {
            opts.vtln_high
        };
        if vtln_warp_factor != 1.0 {
            assert!(
                vtln_low >= 0.0
                    && vtln_low > low_freq
                    && vtln_low < high_freq
                    && vtln_high > 0.0
                    && vtln_high < high_freq
                    && vtln_high > vtln_low,
                "bad vtln cutoffs: {vtln_low}, {vtln_high}"
            );
        }

        let mut bins = Vec::with_capacity(num_bins);
        for bin in 0..num_bins {
            let mut left_mel = mel_low_freq + bin as f32 * mel_freq_delta;
            let mut center_mel = mel_low_freq + (bin + 1) as f32 * mel_freq_delta;
            let mut right_mel = mel_low_freq + (bin + 2) as f32 * mel_freq_delta;

            if vtln_warp_factor != 1.0 {
                let warp = |m: f32| {
                    Self::vtln_warp_mel_freq(
                        vtln_low,
                        vtln_high,
                        low_freq,
                        high_freq,
                        vtln_warp_factor,
                        m,
                    )
                };
                left_mel = warp(left_mel);
                center_mel = warp(center_mel);
                right_mel = warp(right_mel);
            }

            let mut weights = vec![0.0f32; num_fft_bins];
            let mut first_index: Option<usize> = None;
            let mut last_index = 0usize;
            for i in 0..num_fft_bins {
                let freq = fft_bin_width * i as f32;
                let mel = Self::mel_scale(freq);
                if mel > left_mel && mel < right_mel {
                    let weight = if mel <= center_mel {
                        (mel - left_mel) / (center_mel - left_mel)
                    } else {
                        (right_mel - mel) / (right_mel - center_mel)
                    };
                    weights[i] = weight;
                    if first_index.is_none() {
                        first_index = Some(i);
                    }
                    last_index = i;
                }
            }
            let first_index =
                first_index.expect("empty mel bin; --num-mel-bins may be too large");
            assert!(last_index >= first_index);
            bins.push((first_index, weights[first_index..=last_index].to_vec()));
        }

        Self { bins }
    }

    pub(super) fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// `MelBanks::Compute` (mel-computations.cc:226). Input is the power
    /// spectrum, output the (linear, not log) mel energies.
    pub(super) fn compute(&self, power_spectrum: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.bins.len());
        for (i, (offset, weights)) in self.bins.iter().enumerate() {
            let seg = &power_spectrum[*offset..*offset + weights.len()];
            let mut energy = 0.0f32;
            for (w, p) in weights.iter().zip(seg.iter()) {
                energy += w * p;
            }
            out[i] = energy;
        }
    }
}

/// Deterministic Gaussian source matching Kaldi's `RandGauss`
/// (base/kaldi-math.h:155): a Box-Muller transform of two uniforms.
///
/// Kaldi seeds from libc `rand`, which we cannot and should not reproduce.
/// MFA sets `dither = 0.0` so this is off by default; when it is on, we use a
/// seeded xoshiro so that repeated runs of the aligner are reproducible.
pub(super) struct DitherRng {
    state: rand_xoshiro::Xoshiro256PlusPlus,
}

impl DitherRng {
    pub(super) fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            state: rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    fn uniform(&mut self) -> f32 {
        use rand::RngExt;
        // Kaldi's RandUniform returns a value in (0, 1), never exactly 0,
        // because Log() of it is taken immediately afterwards.
        let u: f32 = self.state.random::<f32>();
        if u <= 0.0 { f32::MIN_POSITIVE } else { u }
    }

    pub(super) fn gauss(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mfa_defaults_match_the_cited_sources() {
        let o = MfccOptions::default();
        assert_eq!(o.sample_rate, 16_000);
        assert_eq!(o.frame_length_ms, 25.0);
        assert_eq!(o.frame_shift_ms, 10.0);
        assert_eq!(o.dither, 0.0);
        assert_eq!(o.preemph, 0.97);
        assert!(!o.snip_edges);
        assert!(!o.use_energy);
        assert!(!o.raw_energy);
        assert_eq!(o.num_mel_bins, 23);
        assert_eq!(o.num_ceps, 13);
        assert_eq!(o.low_freq, 20.0);
        assert_eq!(o.high_freq, 7800.0);
        assert_eq!(o.cepstral_lifter, 22.0);
        assert_eq!(o.window, WindowType::Povey);
    }

    #[test]
    fn window_and_padding_sizes() {
        let o = MfccOptions::default();
        assert_eq!(o.window_size(), 400); // 25 ms at 16 kHz
        assert_eq!(o.window_shift(), 160); // 10 ms at 16 kHz
        assert_eq!(o.padded_window_size(), 512);
        assert!((o.frame_shift_s() - 0.01).abs() < 1e-7);
    }

    #[test]
    fn num_frames_snip_edges_true_matches_kaldi_formula() {
        let mut o = MfccOptions::default();
        o.snip_edges = true;
        assert_eq!(num_frames(399, &o), 0);
        assert_eq!(num_frames(400, &o), 1);
        assert_eq!(num_frames(559, &o), 1);
        assert_eq!(num_frames(560, &o), 2);
        assert_eq!(num_frames(16_000, &o), 1 + (16_000 - 400) / 160);
    }

    #[test]
    fn num_frames_snip_edges_false_rounds_to_nearest() {
        let o = MfccOptions::default(); // snip_edges = false
        assert_eq!(num_frames(16_000, &o), 100);
        assert_eq!(num_frames(79, &o), 0);
        assert_eq!(num_frames(80, &o), 1);
        assert_eq!(num_frames(240, &o), 2);
    }

    #[test]
    fn first_sample_of_frame_centres_frames_when_not_snipping() {
        let o = MfccOptions::default();
        // midpoint 80 - 200 = -120: the first frame is reflected at the start.
        assert_eq!(first_sample_of_frame(0, &o), -120);
        assert_eq!(first_sample_of_frame(1, &o), 40);
        let mut snip = o.clone();
        snip.snip_edges = true;
        assert_eq!(first_sample_of_frame(0, &snip), 0);
        assert_eq!(first_sample_of_frame(3, &snip), 480);
    }

    #[test]
    fn povey_window_goes_to_zero_at_the_edges() {
        let o = MfccOptions::default();
        let w = make_window_function(&o);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6);
        assert!(w[399].abs() < 1e-6);
        // Peak is at the centre and is 1.
        assert!((w[199].max(w[200]) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn hamming_window_matches_the_definition() {
        let mut o = MfccOptions::default();
        o.window = WindowType::Hamming;
        let w = make_window_function(&o);
        assert!((w[0] - 0.08).abs() < 1e-5);
        assert!((w[399] - 0.08).abs() < 1e-5);
    }

    #[test]
    fn dct_matrix_is_orthonormal_dct_ii() {
        let n = 23;
        let m = dct_matrix(n, n);
        // Rows are unit norm and mutually orthogonal.
        for k in 0..n {
            let norm: f32 = m[k].iter().map(|v| v * v).sum();
            assert!((norm - 1.0).abs() < 1e-4, "row {k} norm {norm}");
        }
        for k in 0..4 {
            for l in (k + 1)..5 {
                let dot: f32 = m[k].iter().zip(m[l].iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-4, "rows {k},{l} dot {dot}");
            }
        }
        // Row 0 is the constant 1/sqrt(N).
        let expect = (1.0f32 / n as f32).sqrt();
        assert!((m[0][0] - expect).abs() < 1e-6);
    }

    #[test]
    fn lifter_coeffs_match_kaldi() {
        let c = lifter_coeffs(22.0, 13);
        assert!((c[0] - 1.0).abs() < 1e-6); // sin(0) = 0
        for (i, v) in c.iter().enumerate() {
            let expect = 1.0 + 0.5 * 22.0 * (std::f32::consts::PI * i as f32 / 22.0).sin();
            assert!((v - expect).abs() < 1e-5);
        }
    }

    #[test]
    fn mel_scale_roundtrip() {
        for f in [20.0f32, 100.0, 1000.0, 4000.0, 7800.0] {
            let back = MelBanks::inverse_mel_scale(MelBanks::mel_scale(f));
            assert!((back - f).abs() < 0.05 * f.max(1.0), "{f} -> {back}");
        }
    }

    #[test]
    fn mel_banks_cover_the_band_and_are_triangular() {
        let o = MfccOptions::default();
        let mb = MelBanks::new(&o, 1.0);
        assert_eq!(mb.num_bins(), 23);
        for (offset, weights) in &mb.bins {
            assert!(!weights.is_empty());
            // Weights rise then fall; the maximum is at most 1.
            let peak = weights.iter().fold(0.0f32, |m, v| m.max(*v));
            assert!(peak > 0.0 && peak <= 1.0 + 1e-6);
            assert!(offset + weights.len() <= o.padded_window_size() / 2);
        }
        // Bins are ordered by increasing centre frequency.
        for i in 1..mb.bins.len() {
            assert!(mb.bins[i].0 >= mb.bins[i - 1].0);
        }
    }

    #[test]
    fn vtln_warp_is_identity_at_factor_one() {
        for f in [50.0f32, 500.0, 3000.0, 7000.0] {
            let w = MelBanks::vtln_warp_freq(100.0, 7500.0, 20.0, 7800.0, 1.0, f);
            assert!((w - f).abs() < 1e-3, "{f} -> {w}");
        }
    }

    #[test]
    fn vtln_warp_fixes_the_endpoints() {
        // F(low_freq) == low_freq and F(high_freq) == high_freq by construction.
        for factor in [0.9f32, 1.1] {
            let lo = MelBanks::vtln_warp_freq(100.0, 7500.0, 20.0, 7800.0, factor, 20.0);
            let hi = MelBanks::vtln_warp_freq(100.0, 7500.0, 20.0, 7800.0, factor, 7800.0);
            assert!((lo - 20.0).abs() < 1e-2, "{factor}: {lo}");
            assert!((hi - 7800.0).abs() < 1e-1, "{factor}: {hi}");
        }
    }

}
