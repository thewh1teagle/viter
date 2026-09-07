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
use viter_kaldi::gmm::AmDiagGmm;
use viter_kaldi::hmm::TransitionModel;
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
        let per_ms = (setup.mfcc.sample_rate as usize) / 1000;
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
    tm: &TransitionModel,
    am: &AmDiagGmm,
    ali: &Alignment,
    intervals: &IntervalAlignment,
    opts: &RefineOptions,
) -> IntervalAlignment {
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
        let left = opts
            .max_shift_ms
            .min((b - grid[k - 1]) / 2)
            .min(b - bounds[k - 1] - 1);
        let right = opts
            .max_shift_ms
            .min((grid[k + 1] - b) / 2)
            .min(grid[k + 1] - b - 1);
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
        if let Some(i) = best_split(&ratio) {
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

/// Index of the first frame of the next phone: the split of the window that
/// maximizes the summed midpoint-centred ratio on the left minus the right, i.e.
/// `sum_{s<i} (r - mid) - sum_{s>=i} (r - mid)` with `mid` the midpoint of the
/// window's min and max. `None` when the ratio is flat.
fn best_split(ratio: &[f32]) -> Option<usize> {
    let min = ratio.iter().copied().fold(f32::INFINITY, f32::min);
    let max = ratio.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max - min <= 0.0 {
        return None;
    }
    let mid = 0.5 * (min + max);
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
    tm: &TransitionModel,
    am: &AmDiagGmm,
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
                tm,
                am,
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
    fn best_split_finds_a_clean_step() {
        // +1 for 5 frames then -1 for 5 frames: the next phone starts at 5.
        let r = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!(best_split(&r), Some(5));
    }

    #[test]
    fn midpoint_calibration_ignores_a_biased_zero() {
        // A ratio that never goes negative still has its step found at the midpoint.
        let r = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0];
        assert_eq!(best_split(&r), Some(3));
    }

    #[test]
    fn best_split_is_robust_to_a_blip() {
        let r = [1.0, 1.0, -0.5, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!(best_split(&r), Some(6));
    }

    #[test]
    fn flat_ratio_keeps_the_boundary() {
        assert_eq!(best_split(&[2.0, 2.0, 2.0]), None);
    }
}
