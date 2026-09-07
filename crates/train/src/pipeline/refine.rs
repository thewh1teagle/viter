//! Boundary refinement at 1 ms resolution (`viter align`, on by default).
//!
//! The decoder puts every phone boundary on the 10 ms frame grid, and puts it where
//! the two *boundary* HMM states' likelihoods cross. Those states were trained on
//! exactly the transition frames, so the crossing sits early for boundaries into
//! low-energy segments (closures, stops, silence) and late out of them: on TIMIT,
//! 10-30 ms from the hand label by transition class while boundaries between two
//! voiced segments were unbiased (issue #12).
//!
//! Refinement rescores a window around each boundary with 1 ms-shifted features and
//! the two phones' *middle*-state pdfs (their steady cores), and puts the boundary
//! where the ratio `ll(prev) - ll(next)` crosses the midpoint of its range inside the
//! window — the split that best separates the window into a prev-like and a next-like
//! part. Normalizing to the window's own range makes the criterion self-calibrating
//! per boundary, which is what MFA's `--fine_tune` (kalpy
//! `gmm_interpolate_boundary_fast`) does with boundary-state pdfs in a ±10 ms window.
//! Core pdfs and a wider window were worth another ~4 points at 10 ms on TIMIT.
//!
//! Two corrections come from a 1 ms log-energy contour of the waveform, i.e. from the
//! data rather than from phone identities. A boundary into a transient — a rise of
//! 10 dB or more between the 10 ms before and after some candidate — goes at the
//! rise ([`onset`]): the burst flips the ratio as soon as it enters the 25 ms window,
//! so the crossing was 13 ms early for closure-stop boundaries on TIMIT (30% within
//! 10 ms; 86% at the onset). Elsewhere the crossing level is moved from the midpoint
//! towards the louder side's plateau in proportion to the energy step across the
//! window ([`level`]), which takes out the remaining ~5 ms early bias into quieter
//! segments and ~3 ms late bias into louder ones. Together they were worth another
//! ~10 points at 10 ms on TIMIT.
//!
//! The 1 ms features are the ordinary pipeline (MFCC, speaker CMVN, deltas or
//! splice+LDA, fMLLR) computed on the waveform at each of the ten 1 ms offsets, so
//! every 1 ms frame carries exactly the training-time context; MFA splices 1 ms
//! neighbours instead, a ±3 ms context.
//!
//! Timestamps: with `snip_edges = false` frame `t` is centred at `10 t + 5` ms, so
//! the frame-grid boundary "phone starts at frame `t`" exported as `10 t` is the
//! midpoint between the centres of frames `t - 1` and `t`. The same rule on the 1 ms
//! grid puts the boundary between 1 ms frames `τ - 1` and `τ` at `τ + 4.5` ms, which
//! is why [`CENTRE_MS`] is added to every refined boundary.

use ndarray::Array2;
use rayon::prelude::*;
use viter_kaldi::feat::{self, CmvnStats, DeltaOptions, MfccComputer, MfccOptions};
use viter_kaldi::hmm::TransitionModel;
use viter_kaldi::model::AcousticModel;
use viter_kaldi::transform::Mat;
use viter_kaldi::types::{
    Alignment, Feats, IntervalAlignment, PdfId, PhoneInterval, TransitionId, WordInterval,
};

/// Time unit of a refined alignment's `start_frame` / `end_frame`.
pub const TICK_S: f32 = 0.001;

/// Half a 1 ms frame shift past the frame's centre offset (see the module doc),
/// rounded up to the tick.
const CENTRE_MS: usize = 5;

#[derive(Clone, Debug)]
pub struct RefineOptions {
    /// Largest move of a boundary into either neighbour, in ms. Each side is further
    /// capped at half that neighbour's duration, and never crosses the neighbouring
    /// (already refined) boundary.
    pub max_shift_ms: usize,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self { max_shift_ms: 30 }
    }
}

/// Everything about one utterance's features that refinement must reproduce.
pub struct UttFeatureSetup<'a> {
    pub mfcc: &'a MfccOptions,
    pub cmvn: &'a CmvnStats,
    pub deltas: &'a DeltaOptions,
    /// `(left, right)` splice context and the LDA transform; `None` = deltas.
    pub lda: Option<((usize, usize), &'a Array2<f32>)>,
    pub fmllr: Option<&'a Mat>,
}

/// The ten 1 ms-offset feature streams of one utterance: `streams[o][t]` is the
/// frame at `10 t + o` ms (centred at `10 t + o + 5`).
struct FineFeats {
    streams: Vec<Feats>,
}

impl FineFeats {
    fn compute(samples_16k: &[f32], setup: &UttFeatureSetup<'_>) -> Self {
        let computer = MfccComputer::new(setup.mfcc.clone());
        let per_ms = SAMPLES_PER_MS;
        let streams = (0..10)
            .map(|o| {
                let from = (o * per_ms).min(samples_16k.len());
                let mut f = computer.compute(&samples_16k[from..]);
                feat::apply_cmvn(&mut f, setup.cmvn, false);
                let f = match setup.lda {
                    Some(((l, r), lda)) => feat::apply_transform(&feat::splice(&f, l, r), lda),
                    None => feat::add_deltas(&f, setup.deltas),
                };
                match setup.fmllr {
                    Some(x) => feat::apply_transform(&f, x),
                    None => f,
                }
            })
            .collect();
        Self { streams }
    }

    /// Feature row at `ms`, if the utterance has one.
    fn row(&self, ms: usize) -> Option<&[f32]> {
        let s = &self.streams[ms % 10];
        let t = ms / 10;
        (t < s.nrows()).then(|| s.row(t).to_slice().expect("contiguous"))
    }
}

/// The pdf of a phone's steady core: its middle HMM state if the alignment visited
/// it, else `fallback` (the boundary state) — MFA's topology lets a short phone skip
/// the middle state.
fn core_pdf(tm: &TransitionModel, run: &[TransitionId], fallback: TransitionId) -> PdfId {
    let tid = run
        .iter()
        .copied()
        .find(|&t| tm.transition_id_to_hmm_state(t) == 1)
        .unwrap_or(fallback);
    tm.transition_id_to_pdf(tid)
}

/// Refine one utterance's phone boundaries. `intervals` is on the 10 ms frame grid
/// (from `to_intervals`); the result is on the 1 ms grid with `frame_shift_s = TICK_S`.
pub fn refine_utterance(
    samples_16k: &[f32],
    setup: &UttFeatureSetup<'_>,
    model: &AcousticModel,
    ali: &Alignment,
    intervals: &IntervalAlignment,
    opts: &RefineOptions,
) -> IntervalAlignment {
    let (tm, am) = (&model.tm, &model.am);
    let fine = FineFeats::compute(samples_16k, setup);
    let shift_ms = (intervals.frame_shift_s * 1000.0).round() as usize;
    let frames = ali.tids.len();

    // Boundaries in ms on the frame grid, refined left to right; the ends stay put.
    let grid: Vec<usize> = intervals
        .phones
        .iter()
        .map(|p| p.start_frame as usize * shift_ms)
        .chain(std::iter::once(intervals.num_frames() as usize * shift_ms))
        .collect();
    let mut bounds = grid.clone();

    for k in 1..grid.len() - 1 {
        let frame = intervals.phones[k].start_frame as usize;
        if frame == 0 || frame >= frames {
            continue;
        }
        let prev_start = intervals.phones[k - 1].start_frame as usize;
        let next_end = (intervals.phones[k].end_frame as usize).min(frames);
        let prev_pdf = core_pdf(tm, &ali.tids[prev_start..frame], ali.tids[frame - 1]);
        let next_pdf = core_pdf(tm, &ali.tids[frame..next_end], ali.tids[frame]);

        let b = grid[k];
        // A boundary may move up to half its neighbour's duration into it: `d / 2` to
        // the left, `(d - 1) / 2` to the right. The two never meet, so every phone
        // keeps at least one tick even where two boundaries share a 10 ms phone
        // (`ceil(d / 2) > (d - 1) / 2` for every `d >= 1`).
        let left = opts.max_shift_ms.min((b - grid[k - 1]) / 2);
        let right = opts.max_shift_ms.min((grid[k + 1] - b - 1) / 2);
        if left + right < 2 {
            continue;
        }
        let lo = b - left;
        let ratio: Vec<f32> = (lo..=b + right)
            .map(|ms| match fine.row(ms) {
                Some(x) => am.log_likelihood(prev_pdf, x) - am.log_likelihood(next_pdf, x),
                None => f32::NAN,
            })
            .collect();
        if ratio.iter().any(|r| !r.is_finite()) {
            continue;
        }
        // Candidate boundary `i` sits between 1 ms frames `lo + i - 1` and `lo + i`,
        // i.e. at `lo + i + CENTRE_MS` ms; the energy contour is indexed the same way
        // with ONSET_MS of margin on both sides.
        let energy = energy_contour(samples_16k, lo + CENTRE_MS, ratio.len());
        // Before a pause the window lies in the phone's decay into silence, not
        // between two phones' plateaus, and the energy step there is meaningless:
        // TIMIT's hand label is at or before the window, and a lower level only
        // moves these later.
        let level = if model.silence_phones.contains(&intervals.phones[k].phone) {
            0.5
        } else {
            level(&energy)
        };
        let split = onset(&energy).or_else(|| best_split(&ratio, level));
        if let Some(i) = split {
            bounds[k] = lo + i + CENTRE_MS;
        }
    }

    let phones: Vec<PhoneInterval> = intervals
        .phones
        .iter()
        .enumerate()
        .map(|(k, p)| PhoneInterval {
            phone: p.phone,
            start_frame: bounds[k] as u32,
            end_frame: bounds[k + 1] as u32,
        })
        .collect();
    // Words follow their phones: a word starts where its first phone starts.
    let tick_at = |frame: u32| -> u32 {
        let ms = frame as usize * shift_ms;
        match grid.iter().position(|&g| g == ms) {
            Some(k) => bounds[k] as u32,
            None => *bounds.last().expect("non-empty") as u32,
        }
    };
    let words: Vec<WordInterval> = intervals
        .words
        .iter()
        .map(|w| WordInterval {
            word: w.word,
            pron: w.pron,
            start_frame: tick_at(w.start_frame),
            end_frame: tick_at(w.end_frame),
        })
        .collect();
    IntervalAlignment {
        utt: intervals.utt.clone(),
        frame_shift_s: TICK_S,
        phones,
        words,
    }
}

/// `audio::read_16k` hands every model 16 kHz samples.
const SAMPLES_PER_MS: usize = 16;

/// Log energy (natural log of the mean square; 1 nat = 4.3 dB) of the 5 ms of
/// waveform centred at `ms`; the floor past either end of the waveform.
fn log_energy(samples_16k: &[f32], ms: usize) -> f32 {
    let c = ms * SAMPLES_PER_MS;
    let lo = c.saturating_sub(40).min(samples_16k.len());
    let hi = (c + 40).min(samples_16k.len());
    let n = (hi - lo).max(1) as f32;
    let s: f32 = samples_16k[lo..hi].iter().map(|x| x * x).sum();
    (s / n + 1e-10).ln()
}

/// The 1 ms log-energy contour around `n` candidate boundaries starting at `first`
/// ms: `ONSET_MS` values before, the `n` candidates, `ONSET_MS` after.
fn energy_contour(samples_16k: &[f32], first: usize, n: usize) -> Vec<f32> {
    (0..n + 2 * ONSET_MS)
        .map(|j| log_energy(samples_16k, (first + j).saturating_sub(ONSET_MS)))
        .collect()
}

/// Length of the means compared by [`onset`] and of the contour's margins, ms.
const ONSET_MS: usize = 10;

/// Rise of the mean log energy across a candidate boundary that makes it a
/// transient onset: 2.3 nats = 10 dB.
const ONSET_STEP: f32 = 2.3;

/// A boundary into a transient (a stop burst): the candidate with the largest rise
/// of the mean log energy over the `ONSET_MS` after it against the `ONSET_MS` before
/// it, if that rise is at least `ONSET_STEP`. The burst is a loud transient that
/// flips the pdf ratio as soon as it enters the 25 ms analysis window, so the ratio
/// crossing sits ~13 ms early on TIMIT while the hand label is at the burst onset,
/// which this finds within 10 ms for 86% of closure-stop boundaries. The `ONSET_MS`
/// before the rise must lie inside the window, so an earlier onset just outside it
/// (the burst before a stop-vowel boundary) is not picked up.
fn onset(energy: &[f32]) -> Option<usize> {
    let n = energy.len() - 2 * ONSET_MS;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let rise = |i: usize| {
        let j = i + ONSET_MS;
        mean(&energy[j..j + ONSET_MS]) - mean(&energy[j - ONSET_MS..j])
    };
    let mut best = (ONSET_STEP, None);
    for i in ONSET_MS..n {
        let r = rise(i);
        if r > best.0 || (r == best.0 && best.1.is_none()) {
            best = (r, Some(i));
        }
    }
    best.1
}

/// Crossing level for [`best_split`] from the energy step across the window (mean
/// log energy of its second half minus its first): the midpoint, moved by
/// `LEVEL_PER_NAT` per nat towards the louder side's plateau and clamped to
/// `0.5 ± LEVEL_RANGE`. On TIMIT the midpoint crossing is ~5 ms early into a quieter
/// segment (vowel-closure, vowel-fricative, vowel-nasal) and ~3 ms late into a louder
/// one (nasal-vowel, fricative-vowel), saturating within about a nat either way; a
/// lower level moves the split later, a higher one earlier.
fn level(energy: &[f32]) -> f32 {
    let inner = &energy[ONSET_MS..energy.len() - ONSET_MS];
    let (a, b) = inner.split_at(inner.len() / 2);
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len().max(1) as f32;
    let step = mean(b) - mean(a);
    0.5 + (LEVEL_PER_NAT * step).clamp(-LEVEL_RANGE, LEVEL_RANGE)
}

const LEVEL_PER_NAT: f32 = 0.1;
const LEVEL_RANGE: f32 = 0.15;

/// Index of the first frame of the next phone: the split of the window that
/// maximizes the summed level-centred ratio on the left minus the right, i.e.
/// `sum_{s<i} (r - mid) - sum_{s>=i} (r - mid)` with `mid` at fraction `level` of
/// the way from the window's min to its max. `None` when the ratio is flat.
fn best_split(ratio: &[f32], level: f32) -> Option<usize> {
    let min = ratio.iter().copied().fold(f32::INFINITY, f32::min);
    let max = ratio.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max - min <= 0.0 {
        return None;
    }
    let mid = min + level * (max - min);
    let total: f32 = ratio.iter().map(|r| r - mid).sum();
    let mut best = (f32::NEG_INFINITY, 0usize);
    let mut prefix = 0.0f32;
    for i in 0..=ratio.len() {
        let score = 2.0 * prefix - total;
        if score > best.0 {
            best = (score, i);
        }
        if i < ratio.len() {
            prefix += ratio[i] - mid;
        }
    }
    Some(best.1.min(ratio.len() - 1))
}

/// Refine every aligned utterance in parallel. `setup_for(utt)` provides the
/// utterance's feature pipeline; audio is re-read from `audio_paths`.
pub fn refine_all<'a>(
    audio_paths: &[std::path::PathBuf],
    setup_for: impl Fn(usize) -> UttFeatureSetup<'a> + Sync,
    model: &AcousticModel,
    alignments: &[Option<Alignment>],
    intervals: &[Option<IntervalAlignment>],
    opts: &RefineOptions,
) -> Vec<Option<IntervalAlignment>> {
    intervals
        .par_iter()
        .enumerate()
        .map(|(i, iv)| {
            let iv = iv.as_ref()?;
            let ali = alignments[i].as_ref()?;
            let audio = viter_kaldi::audio::read_16k(&audio_paths[i]).ok()?;
            let setup = setup_for(i);
            Some(refine_utterance(
                &audio.samples,
                &setup,
                model,
                ali,
                iv,
                opts,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_caps_never_let_neighbouring_boundaries_meet() {
        // Boundaries at 0, d and 2d (ms): the first may move right by (d-1)/2 and the
        // second left by d/2; the phone between them must keep at least one tick.
        for d in 1..=60usize {
            let right = 30usize.min((d - 1) / 2);
            let left = 30usize.min(d / 2);
            assert!(right + CENTRE_MS < d - left + CENTRE_MS, "d = {d}");
        }
    }

    #[test]
    fn best_split_finds_a_clean_step() {
        // +1 for 5 frames then -1 for 5 frames: the next phone starts at 5.
        let r = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!(best_split(&r, 0.5), Some(5));
    }

    #[test]
    fn midpoint_calibration_ignores_a_biased_zero() {
        // A ratio that never goes negative still has its step found at the midpoint.
        let r = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0];
        assert_eq!(best_split(&r, 0.5), Some(3));
    }

    #[test]
    fn best_split_is_robust_to_a_blip() {
        let r = [1.0, 1.0, -0.5, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!(best_split(&r, 0.5), Some(6));
    }

    #[test]
    fn flat_ratio_keeps_the_boundary() {
        assert_eq!(best_split(&[2.0, 2.0, 2.0], 0.5), None);
    }

    #[test]
    fn crossing_level_moves_the_split() {
        // A ramp: a lower level counts more of it as prev-like, so the split is later.
        let r: Vec<f32> = (0..20).map(|i| 10.0 - i as f32).collect();
        assert_eq!(best_split(&r, 0.5), Some(10));
        assert_eq!(best_split(&r, 0.35), Some(13));
        assert_eq!(best_split(&r, 0.65), Some(7));
    }

    /// A contour for `n` candidates, quiet (-8) until candidate `at`, then `loud`.
    fn step_contour(n: usize, at: usize, loud: f32) -> Vec<f32> {
        (0..n + 2 * ONSET_MS)
            .map(|j| if j < at + ONSET_MS { -8.0 } else { loud })
            .collect()
    }

    #[test]
    fn onset_finds_a_burst_and_ignores_a_small_step() {
        assert_eq!(onset(&step_contour(40, 27, -2.0)), Some(27));
        assert_eq!(onset(&step_contour(40, 27, -7.0)), None);
    }

    #[test]
    fn onset_needs_its_quiet_side_inside_the_window() {
        // A step at candidate 1 has almost none of its quiet side among the candidates.
        assert_eq!(onset(&step_contour(40, 1, 0.0)), None);
        assert_eq!(onset(&step_contour(40, ONSET_MS, 0.0)), Some(ONSET_MS));
    }

    #[test]
    fn log_energy_is_finite_past_the_waveform() {
        let s = vec![0.5f32; 100];
        assert!(log_energy(&s, 3).is_finite());
        assert!(log_energy(&s, 6).is_finite()); // window straddles the end
        assert!(log_energy(&s, 1000).is_finite()); // entirely past it
        assert!(log_energy(&s, 1000) < log_energy(&s, 3));
    }

    #[test]
    fn level_follows_the_energy_step_and_saturates() {
        assert!((level(&step_contour(40, 20, -8.0)) - 0.5).abs() < 1e-6);
        assert!(level(&step_contour(40, 20, -7.0)) > 0.5);
        assert!((level(&step_contour(40, 20, 0.0)) - 0.65).abs() < 1e-6);
        let quieter: Vec<f32> = step_contour(40, 20, 0.0).into_iter().rev().collect();
        assert!((level(&quieter) - 0.35).abs() < 1e-6);
    }
}
