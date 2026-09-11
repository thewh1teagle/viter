//! The ETA model: a fixed plan of passes, measured as it runs, extrapolated ahead.
//!
//! The plan ([`super::super::plan::WorkPlan`]) lists every pass the run will
//! perform, in execution order, with its units and the gaussian count of the
//! model it runs against. [`Tracker`] consumes that list with a cursor as counted
//! bars open and increment, and times each pass **by adjacency**: a pass's
//! duration is the time from the previous pass's end (or from `plan()` for the
//! first) to its own end, so every gap — tree build, model update, feature
//! derivation, fMLLR solve — is folded into the pass that follows it and nothing
//! is left out of the clock.
//!
//! [`Tracker::eta`] predicts the remaining time with three rules, in order, for
//! each remaining pass (the reference implementation is `p_final` in the design
//! scratchpad's `sim.py`):
//!
//! 1. this `(stage, step)` already ran in this run → its measured rate;
//! 2. this `step` ran in another stage → that instance's rate carried by the
//!    ratio of the built-in [`PRIOR`] between the two stages, raised (never
//!    lowered) by a least-squares fit `rate = a + b·gauss` when two stage
//!    instances were measured at gaussian levels a factor two apart;
//! 3. never measured → the prior rate scaled by this run's calibration
//!    (elapsed / Σ units × prior over the finished passes).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::super::plan::PlannedPass;

/// Relative seconds per utterance-pass, measured on a GPU LJSpeech run (13,093
/// utterances). Only the ratios matter: rule 3 rescales the whole table by this
/// run's own speed, and rule 2 only ever uses a ratio of two entries.
pub(super) const PRIOR: &[(&str, &str, f64)] = &[
    ("features", "mfcc", 20.841),
    ("mono", "accumulate", 1.000),
    ("mono", "align", 2.864),
    ("mono", "equal align", 0.046),
    ("mono", "graphs", 0.879),
    ("tri", "accumulate", 1.248),
    ("tri", "align", 5.826),
    ("tri", "align (previous model)", 3.577),
    ("tri", "convert alignments", 0.101),
    ("tri", "graphs", 0.465),
    ("tri", "tree stats", 0.817),
    ("lda", "accumulate", 1.377),
    ("lda", "align", 5.603),
    ("lda", "align (previous model)", 6.404),
    ("lda", "convert alignments", 0.117),
    ("lda", "graphs", 0.462),
    ("lda", "lda stats", 5.071),
    ("lda", "mllt", 9.811),
    ("lda", "tree stats", 0.942),
    ("sat", "accumulate", 1.669),
    ("sat", "align", 6.836),
    ("sat", "align (previous model)", 7.225),
    ("sat", "convert alignments", 0.151),
    ("sat", "graphs", 0.617),
    ("sat", "tree stats", 1.075),
    ("sat", "two-feats stats", 19.466),
    ("sat_2", "accumulate", 2.144),
    ("sat_2", "align", 8.754),
    ("sat_2", "align (previous model)", 7.461),
    ("sat_2", "convert alignments", 0.174),
    ("sat_2", "graphs", 0.661),
    ("sat_2", "tree stats", 1.191),
    ("sat_2", "two-feats stats", 46.790),
    ("pronprob", "align", 9.013),
    ("pronprob", "graphs", 0.636),
    ("sat_3", "accumulate", 3.104),
    ("sat_3", "align", 12.223),
    ("sat_3", "align (previous model)", 8.113),
    ("sat_3", "convert alignments", 0.158),
    ("sat_3", "graphs", 0.576),
    ("sat_3", "tree stats", 1.036),
    ("final", "final align", 18.552),
    ("final", "final align (fmllr)", 18.522),
    ("final", "graphs", 0.761),
];

/// The same table measured on a CPU run (1,000 LJSpeech utterances, 20 cores).
/// On the CPU every scoring pass grows almost linearly with the gaussian count,
/// so the late stages weigh far more than on a GPU; a CPU run that used the GPU
/// shape read 50% low for its first quarter.
pub(super) const PRIOR_CPU: &[(&str, &str, f64)] = &[
    ("features", "mfcc", 25.754),
    ("mono", "accumulate", 1.000),
    ("mono", "align", 5.092),
    ("mono", "equal align", 0.145),
    ("mono", "graphs", 0.726),
    ("tri", "accumulate", 10.788),
    ("tri", "align", 13.552),
    ("tri", "align (previous model)", 8.278),
    ("tri", "convert alignments", 0.159),
    ("tri", "graphs", 0.462),
    ("tri", "tree stats", 1.656),
    ("lda", "accumulate", 15.958),
    ("lda", "align", 17.993),
    ("lda", "align (previous model)", 18.209),
    ("lda", "convert alignments", 0.194),
    ("lda", "graphs", 0.484),
    ("lda", "lda stats", 5.036),
    ("lda", "mllt", 11.316),
    ("lda", "tree stats", 1.743),
    ("sat", "accumulate", 19.938),
    ("sat", "align", 22.114),
    ("sat", "align (previous model)", 31.263),
    ("sat", "convert alignments", 0.246),
    ("sat", "graphs", 0.636),
    ("sat", "tree stats", 2.339),
    ("sat", "two-feats stats", 27.694),
    ("sat_2", "accumulate", 40.651),
    ("sat_2", "align", 62.197),
    ("sat_2", "align (previous model)", 26.349),
    ("sat_2", "convert alignments", 0.205),
    ("sat_2", "graphs", 0.546),
    ("sat_2", "tree stats", 1.775),
    ("sat_2", "two-feats stats", 49.558),
    ("pronprob", "align", 85.382),
    ("pronprob", "graphs", 0.466),
    ("sat_3", "accumulate", 46.916),
    ("sat_3", "align", 91.439),
    ("sat_3", "align (previous model)", 98.011),
    ("sat_3", "convert alignments", 0.198),
    ("sat_3", "graphs", 0.546),
    ("sat_3", "tree stats", 1.753),
    ("sat_3", "two-feats stats", 42.903),
    ("final", "final align", 115.318),
    ("final", "final align (fmllr)", 112.260),
    ("final", "graphs", 0.675),
];

/// A cost table keyed by `(stage key, step label)`. Injectable so tests can
/// exercise the three rules against a small synthetic prior.
pub(super) type Prior = &'static [(&'static str, &'static str, f64)];

/// `sat_4` → `sat`; `pronprob_2` → `pronprob`; anything else unchanged.
fn stage_family(stage: &str) -> &str {
    match stage.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => stage,
    }
}

/// Prior lookup: exact `(stage, step)`, then the stage's `_3` sibling (SAT rounds
/// past the third behave like the third), then the unsuffixed family.
pub(super) fn prior_of(prior: Prior, stage: &str, step: &str) -> Option<f64> {
    let family = stage_family(stage);
    let third = format!("{family}_3");
    let get = |st: &str| {
        prior
            .iter()
            .find(|(s, k, _)| *s == st && *k == step)
            .map(|(_, _, v)| *v)
    };
    get(stage).or_else(|| get(&third)).or_else(|| get(family))
}

/// A pass of the plan, plus what the run has measured of it so far.
struct PassRec {
    stage: String,
    step: &'static str,
    units: u64,
    gauss: u64,
    /// Re-derives features first; priced apart from its light twin.
    heavy: bool,
    done: u64,
    /// When a bar first counted a unit on this pass.
    started: Option<Instant>,
    /// When `done` reached `units`. Adjacency duration ends here.
    ended: Option<Instant>,
}

impl PassRec {
    fn finished(&self) -> bool {
        self.done >= self.units
    }
}

/// One measured `(stage, step)`: total units, total adjacency seconds, max gauss.
#[derive(Clone, Default)]
struct Measured {
    units: u64,
    secs: f64,
    gauss: u64,
    /// One `(units, secs)` per finished pass, so the rate can drop an outlier.
    samples: Vec<(u64, f64)>,
}

impl Measured {
    /// Seconds per unit. With two or more passes the single slowest one is left
    /// out: the first pass of a stage instance carries the whole feature
    /// re-derivation behind it (several seconds on a full corpus), and averaging
    /// that into every later pass of the stage read the ETA 60% high for the first
    /// third of each SAT round.
    fn rate(&self) -> f64 {
        if self.samples.len() >= 2 {
            let worst = self
                .samples
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    (a.1.1 / a.1.0.max(1) as f64).total_cmp(&(b.1.1 / b.1.0.max(1) as f64))
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let (u, t) = self
                .samples
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != worst)
                .fold((0u64, 0.0), |(u, t), (_, (pu, ps))| (u + pu, t + ps));
            if u > 0 {
                return t / u as f64;
            }
        }
        self.secs / self.units.max(1) as f64
    }
}

/// Rule 2's table: per `(step, heavy)`, one `(stage, gauss, rate)` per measured
/// stage instance, in plan order.
type ByStep<'a> = HashMap<(&'a str, bool), Vec<(&'a str, u64, f64)>>;

/// The plan cursor, the pass records and the predictor.
pub(super) struct Tracker {
    passes: Vec<PassRec>,
    /// Index of the pass counted bars are currently feeding.
    cursor: usize,
    /// Start of the clock (`plan()`), and the anchor of the first adjacency span.
    start: Option<Instant>,
    /// Bars whose step label disagreed with the cursor's pass.
    mismatches: u64,
    prior: Prior,
}

impl Default for Tracker {
    fn default() -> Self {
        Self {
            passes: Vec::new(),
            cursor: 0,
            start: None,
            mismatches: 0,
            prior: PRIOR,
        }
    }
}

impl Tracker {
    /// Load the plan and start the clock.
    pub(super) fn load(&mut self, passes: &[PlannedPass], now: Instant) {
        self.passes = passes
            .iter()
            .map(|p| PassRec {
                stage: p.stage.clone(),
                step: p.step,
                units: p.units,
                gauss: p.gauss,
                heavy: p.heavy,
                done: 0,
                started: None,
                ended: None,
            })
            .collect();
        self.cursor = 0;
        self.start = Some(now);
        self.mismatches = 0;
    }

    /// Cap every planned pass's gaussian count. The schedule's targets are upper
    /// bounds: Kaldi's `SplitByCount` stops a pdf growing once
    /// `(components + 1) * min_count >= occupancy` (min_count 20), so a corpus of
    /// `frames` frames never holds more than about `frames / 20` gaussians. A
    /// small corpus therefore plateaus far below `max_gaussians`, and predicting
    /// its later stages at the uncapped target overestimated them by ~60% on a
    /// 1000-utterance CPU run.
    pub(super) fn cap_gauss(&mut self, cap: u64) {
        for p in &mut self.passes {
            p.gauss = p.gauss.min(cap);
        }
    }

    pub(super) fn with_prior(&mut self, prior: Prior) {
        self.prior = prior;
    }

    pub(super) fn mismatches(&self) -> u64 {
        self.mismatches
    }

    /// A counted bar opened with `label`. Advance past a finished pass, then
    /// check the label against the cursor's pass and resync on a mismatch.
    ///
    /// Returns the mismatched-against step when the labels disagreed, so the
    /// caller can warn outside the lock.
    pub(super) fn open(&mut self, label: &str, now: Instant) -> Option<(String, String)> {
        let step = strip_iter(label);
        while self.cursor < self.passes.len() && self.passes[self.cursor].finished() {
            self.cursor += 1;
        }
        if self.cursor >= self.passes.len() {
            return None;
        }
        if self.passes[self.cursor].step == step {
            self.passes[self.cursor].started.get_or_insert(now);
            return None;
        }
        self.mismatches += 1;
        let expected = self.passes[self.cursor].step.to_string();
        // Resync: the next unfinished pass carrying this label, if any.
        if let Some(i) = (self.cursor..self.passes.len())
            .find(|&i| self.passes[i].step == step && !self.passes[i].finished())
        {
            self.cursor = i;
        }
        self.passes[self.cursor].started.get_or_insert(now);
        Some((expected, step.to_string()))
    }

    /// Count `n` units on the cursor pass, spilling into the following passes
    /// when a bar overshoots. Stamps each pass's end as it completes.
    pub(super) fn advance(&mut self, mut n: u64, now: Instant) {
        while n > 0 && self.cursor < self.passes.len() {
            let p = &mut self.passes[self.cursor];
            p.started.get_or_insert(now);
            let room = p.units.saturating_sub(p.done);
            let take = room.min(n);
            p.done += take;
            n -= take;
            if p.finished() {
                p.ended.get_or_insert(now);
                self.cursor += 1;
            } else {
                break;
            }
        }
    }

    /// Adjacency duration of pass `i`: from the previous finished pass's end
    /// (or the clock start) to this pass's end.
    fn span(&self, i: usize) -> Option<f64> {
        let end = self.passes[i].ended?;
        let prev = self.passes[..i]
            .iter()
            .rev()
            .find_map(|p| p.ended)
            .or(self.start)?;
        Some(end.saturating_duration_since(prev).as_secs_f64())
    }

    /// Remaining time, or `None` before the first pass has finished.
    pub(super) fn eta(&self, now: Instant) -> Option<Duration> {
        let start = self.start?;
        let elapsed = now.saturating_duration_since(start).as_secs_f64();

        // Rule 1's table, plus the units and prior-weight of everything finished.
        let mut measured: HashMap<(&str, &str, bool), Measured> = HashMap::new();
        let mut prior_work = 0.0;
        let mut units_done = 0u64;
        for i in 0..self.passes.len() {
            let p = &self.passes[i];
            if !p.finished() || p.units == 0 {
                continue;
            }
            let Some(secs) = self.span(i) else { continue };
            let m = measured
                .entry((p.stage.as_str(), p.step, p.heavy))
                .or_default();
            m.units += p.units;
            m.secs += secs;
            m.gauss = m.gauss.max(p.gauss);
            m.samples.push((p.units, secs));
            units_done += p.units;
            prior_work += p.units as f64 * prior_of(self.prior, &p.stage, p.step).unwrap_or(0.0);
        }
        if measured.is_empty() {
            return None;
        }
        let scale = if prior_work > 0.0 {
            elapsed / prior_work
        } else {
            1.0
        };
        // Fallback when the prior knows nothing about this run at all.
        let flat = elapsed / units_done.max(1) as f64;

        // Rule 2's table: per step label, one point (gauss, rate) per stage
        // instance, in plan order, so `last` is the latest measured instance.
        let mut by_step: ByStep<'_> = HashMap::new();
        for p in &self.passes {
            if let Some(m) = measured.get(&(p.stage.as_str(), p.step, p.heavy))
                && !by_step
                    .entry((p.step, p.heavy))
                    .or_default()
                    .iter()
                    .any(|(st, _, _)| *st == p.stage)
            {
                by_step
                    .get_mut(&(p.step, p.heavy))
                    .expect("just inserted")
                    .push((p.stage.as_str(), m.gauss, m.rate()));
            }
        }

        let mut total = 0.0;
        for (i, p) in self.passes.iter().enumerate() {
            let left = p.units.saturating_sub(p.done);
            if left == 0 {
                continue;
            }
            // A pass already 10% through is best described by its own speed.
            if p.done * 10 >= p.units
                && let Some(started) = p.started
            {
                let secs = now.saturating_duration_since(started).as_secs_f64();
                if secs > 0.0 {
                    total += left as f64 * (secs / p.done as f64);
                    continue;
                }
            }
            total += left as f64 * self.rate(i, &measured, &by_step, scale, flat);
        }
        Some(Duration::from_secs_f64(total.max(0.0)))
    }

    /// The three rules, for the pass at `i`.
    fn rate(
        &self,
        i: usize,
        measured: &HashMap<(&str, &str, bool), Measured>,
        by_step: &ByStep<'_>,
        scale: f64,
        flat: f64,
    ) -> f64 {
        let p = &self.passes[i];
        // 1. this exact (stage, step) ran already.
        if let Some(m) = measured.get(&(p.stage.as_str(), p.step, p.heavy))
            && m.units > 0
        {
            return m.rate();
        }
        // 2. this step ran in some other stage: carry the latest instance by the
        //    prior's ratio, then raise it by the gauss fit when we have spread.
        if let Some(points) = by_step.get(&(p.step, p.heavy)).filter(|v| !v.is_empty()) {
            let (st0, _, rate0) = *points.last().expect("non-empty");
            let ratio = match (
                prior_of(self.prior, &p.stage, p.step),
                prior_of(self.prior, st0, p.step),
            ) {
                (Some(a), Some(b)) if b > 0.0 => a / b,
                _ => 1.0,
            };
            let mut r = rate0 * ratio;
            let lo = points.iter().map(|(_, g, _)| *g).min().unwrap_or(0);
            let hi = points.iter().map(|(_, g, _)| *g).max().unwrap_or(0);
            if points.len() >= 2 && hi >= 2 * lo {
                let n = points.len() as f64;
                let mx = points.iter().map(|(_, g, _)| *g as f64).sum::<f64>() / n;
                let my = points.iter().map(|(_, _, r)| *r).sum::<f64>() / n;
                let sxx = points
                    .iter()
                    .map(|(_, g, _)| (*g as f64 - mx).powi(2))
                    .sum::<f64>();
                let sxy = points
                    .iter()
                    .map(|(_, g, r)| (*g as f64 - mx) * (r - my))
                    .sum::<f64>();
                if sxx > 0.0 {
                    let b = (sxy / sxx).max(0.0);
                    r = r.max(my - b * mx + b * p.gauss as f64);
                }
            }
            return r.max(0.0);
        }
        // 3. never measured: the prior scaled to this machine.
        match prior_of(self.prior, &p.stage, p.step) {
            Some(w) => scale * w,
            None => flat,
        }
    }
}

/// Drop a bar label's `iter 12/35 · ` prefix, leaving the plan's step label.
pub(super) fn strip_iter(label: &str) -> &str {
    label
        .strip_prefix("iter ")
        .and_then(|rest| rest.split_once(" · "))
        .map_or(label, |(_, step)| step)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PRIOR: Prior = &[
        ("mono", "align", 1.0),
        ("mono", "accumulate", 2.0),
        ("tri", "align", 4.0),
        ("tri", "accumulate", 8.0),
    ];

    fn pass(stage: &str, step: &'static str, units: u64, gauss: u64) -> PlannedPass {
        PlannedPass {
            stage: stage.to_string(),
            step,
            units,
            gauss,
            heavy: false,
        }
    }

    fn tracker(passes: &[PlannedPass]) -> Tracker {
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        t.load(passes, Instant::now());
        t
    }

    #[test]
    fn strips_the_iteration_prefix() {
        assert_eq!(strip_iter("iter 12/35 · align"), "align");
        assert_eq!(strip_iter("align"), "align");
        assert_eq!(strip_iter("iter 0/3 · equal align"), "equal align");
        assert_eq!(strip_iter("iteration something"), "iteration something");
    }

    #[test]
    fn prior_falls_back_through_sat_3_then_the_family() {
        // Exact hit.
        assert_eq!(prior_of(PRIOR, "sat_2", "align"), Some(8.754));
        // sat_4 has no table entry: the _3 sibling stands in.
        assert_eq!(prior_of(PRIOR, "sat_4", "align"), Some(12.223));
        // pronprob_2 has neither an exact nor a _3 entry: the family answers.
        assert_eq!(prior_of(PRIOR, "pronprob_2", "align"), Some(9.013));
        assert_eq!(prior_of(PRIOR, "mono", "nonsense"), None);
    }

    #[test]
    fn cursor_consumes_the_plan_in_order() {
        let plan = [
            pass("mono", "align", 10, 100),
            pass("mono", "accumulate", 10, 100),
            pass("tri", "align", 10, 200),
        ];
        let mut t = tracker(&plan);
        for step in ["align", "accumulate", "align"] {
            assert!(t.open(step, Instant::now()).is_none(), "{step}");
            t.advance(10, Instant::now());
        }
        assert_eq!(t.mismatches(), 0);
        assert!(t.passes.iter().all(|p| p.finished()));
    }

    #[test]
    fn a_wrong_label_is_counted_and_resynced() {
        let plan = [
            pass("mono", "align", 10, 100),
            pass("mono", "accumulate", 10, 100),
            pass("tri", "align", 10, 200),
        ];
        let mut t = tracker(&plan);
        // The run opens "accumulate" first: a mismatch, resyncing to pass 1.
        let hit = t.open("accumulate", Instant::now());
        assert_eq!(hit, Some(("align".into(), "accumulate".into())));
        assert_eq!(t.cursor, 1);
        assert_eq!(t.mismatches(), 1);
        t.advance(10, Instant::now());
        // The cursor is now past mono's align, so the next "align" lands on tri
        // without a second complaint: one drift, one warning.
        assert!(t.open("align", Instant::now()).is_none());
        assert_eq!(t.mismatches(), 1);
        assert_eq!(t.passes[t.cursor].stage, "tri");
    }

    #[test]
    fn an_unknown_label_keeps_counting_on_the_cursor() {
        let plan = [pass("mono", "align", 10, 100)];
        let mut t = tracker(&plan);
        assert!(t.open("nonsense", Instant::now()).is_some());
        assert_eq!(t.cursor, 0);
        t.advance(10, Instant::now());
        assert!(t.passes[0].finished());
    }

    #[test]
    fn adjacency_folds_a_gap_into_the_following_pass() {
        let plan = [
            pass("mono", "align", 10, 100),
            pass("mono", "accumulate", 10, 100),
        ];
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        let t0 = Instant::now();
        t.load(&plan, t0);
        // Pass 0 runs for 1s.
        t.open("align", t0);
        t.advance(10, t0 + Duration::from_secs(1));
        // A 4s gap (model update), then pass 1 runs for 1s.
        let s1 = t0 + Duration::from_secs(5);
        t.open("accumulate", s1);
        t.advance(10, s1 + Duration::from_secs(1));
        assert!((t.span(0).unwrap() - 1.0).abs() < 1e-6);
        // The gap belongs to the pass that followed it: 5s, not 1s.
        assert!((t.span(1).unwrap() - 5.0).abs() < 1e-6, "{:?}", t.span(1));
    }

    #[test]
    fn eta_is_none_before_any_pass_finishes() {
        let plan = [pass("mono", "align", 10, 100)];
        let mut t = tracker(&plan);
        let now = Instant::now();
        assert!(t.eta(now).is_none());
        t.open("align", now);
        t.advance(5, now);
        // Half-done but nothing finished: still nothing to extrapolate from.
        assert!(t.eta(now + Duration::from_secs(1)).is_none());
    }

    #[test]
    fn rule1_uses_the_same_stage_and_step_measured_rate() {
        // Two `accumulate` passes in mono; the second is predicted from the first.
        let plan = [
            pass("mono", "accumulate", 10, 100),
            pass("mono", "accumulate", 20, 100),
        ];
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        let t0 = Instant::now();
        t.load(&plan, t0);
        t.open("accumulate", t0);
        t.advance(10, t0 + Duration::from_secs(2)); // 0.2 s/unit
        // 20 units left at 0.2 s/unit = 4s, regardless of the prior.
        let eta = t.eta(t0 + Duration::from_secs(2)).unwrap().as_secs_f64();
        assert!((eta - 4.0).abs() < 1e-3, "{eta}");
    }

    #[test]
    fn rule2_carries_the_other_stage_rate_by_the_prior_ratio() {
        // mono/align measured; tri/align predicted. Prior ratio tri:mono = 4:1.
        let plan = [
            pass("mono", "align", 10, 100),
            pass("tri", "align", 10, 100),
        ];
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        let t0 = Instant::now();
        t.load(&plan, t0);
        t.open("align", t0);
        t.advance(10, t0 + Duration::from_secs(1)); // 0.1 s/unit
        // Only one instance measured, so no gauss fit: 10 × 0.1 × 4 = 4s.
        let eta = t.eta(t0 + Duration::from_secs(1)).unwrap().as_secs_f64();
        assert!((eta - 4.0).abs() < 1e-3, "{eta}");
    }

    #[test]
    fn rule2_takes_the_max_with_the_gauss_fit() {
        // Two measured instances 4× apart in gauss, with a steep rate slope; the
        // linear fit extrapolates above the prior-ratio carry and wins.
        let plan = [
            pass("mono", "align", 10, 100),
            pass("tri", "align", 10, 400),
            pass("sat", "align", 10, 800),
        ];
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        let t0 = Instant::now();
        t.load(&plan, t0);
        t.open("align", t0);
        t.advance(10, t0 + Duration::from_secs(1)); // mono: 0.1 s/unit @ 100
        let a = t0 + Duration::from_secs(1);
        t.open("align", a);
        t.advance(10, a + Duration::from_secs(10)); // tri: 1.0 s/unit @ 400
        let now = a + Duration::from_secs(10);
        // Fit over (100, 0.1) and (400, 1.0): b = 0.003, a0 = -0.2, at 800 -> 2.2.
        // The carry (no `sat` entry in the test prior -> ratio 1) gives 1.0.
        let eta = t.eta(now).unwrap().as_secs_f64();
        assert!((eta - 22.0).abs() < 1e-3, "{eta}");
    }

    #[test]
    fn rule3_scales_the_prior_by_this_runs_calibration() {
        // mono/align measured at 0.1 s/unit against a prior weight of 1.0, so the
        // run's scale is 0.1 s per prior unit. mono/accumulate (prior 2.0) is then
        // predicted at 0.2 s/unit.
        let plan = [
            pass("mono", "align", 10, 100),
            pass("mono", "accumulate", 10, 100),
        ];
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        let t0 = Instant::now();
        t.load(&plan, t0);
        t.open("align", t0);
        t.advance(10, t0 + Duration::from_secs(1));
        let eta = t.eta(t0 + Duration::from_secs(1)).unwrap().as_secs_f64();
        assert!((eta - 2.0).abs() < 1e-3, "{eta}");
    }

    #[test]
    fn the_current_pass_is_priced_at_its_own_observed_rate() {
        let plan = [
            pass("mono", "align", 10, 100),
            pass("mono", "accumulate", 100, 100),
        ];
        let mut t = Tracker::default();
        t.with_prior(TEST_PRIOR);
        let t0 = Instant::now();
        t.load(&plan, t0);
        t.open("align", t0);
        t.advance(10, t0 + Duration::from_secs(1));
        let a = t0 + Duration::from_secs(1);
        t.open("accumulate", a);
        t.advance(50, a); // 50% done
        // 5s spent on 50 units = 0.1 s/unit; 50 left -> 5s. (The prior would say
        // 0.2 s/unit = 10s, so this proves the observed rate takes over.)
        let eta = t.eta(a + Duration::from_secs(5)).unwrap().as_secs_f64();
        assert!((eta - 5.0).abs() < 1e-3, "{eta}");
    }

    #[test]
    fn skipping_units_marks_passes_finished_in_order() {
        let plan = [
            pass("mono", "align", 10, 100),
            pass("mono", "accumulate", 10, 100),
        ];
        let mut t = tracker(&plan);
        // A `skip` that spans two passes lands on both.
        t.advance(20, Instant::now());
        assert!(t.passes.iter().all(|p| p.finished()));
        assert_eq!(t.cursor, 2);
    }
}
