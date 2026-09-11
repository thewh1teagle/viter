//! Shared counters, ETA arithmetic and line formatting behind [`super::Progress`].
//!
//! [`Shared`] is the whole run's state: the global `done`/`total` counters, the
//! clock started by `plan()`, the sink and the current stage/sub-step text. It is
//! held behind an `Arc` by every [`super::Bar`], so rayon worker threads increment
//! it directly. Everything rendered — the live line's message, the plain-mode
//! `47% · 12m03s left` suffix, the `--log FILE` suffix with raw counters — is
//! computed here at render time. The percent and the bar show the fraction of
//! *predicted time*, not of utterance-passes: late passes cost several times more
//! than early ones, so the counter's percent would lie. The prediction comes from
//! [`super::eta::Tracker`]; indicatif's own smoothing estimators are never used.

use super::eta::Tracker;
use indicatif::ProgressBar;

/// Columns the template uses before `{msg}`: ` 25% [` + 28 bar chars + `] `.
const LIVE_PREFIX_COLS: usize = 36;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Optional plain-text log every `Progress` appends to (stage lines, every
/// iteration, warnings), ANSI-free. Set once from the CLI (`--log FILE`).
static LOG: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// Route a copy of all progress output to `path` (appended, no colors).
pub fn set_log_file(path: &std::path::Path) -> std::io::Result<()> {
    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    *LOG.lock().unwrap() = Some(f);
    Ok(())
}

pub(super) fn log_line(text: &str) {
    if let Some(f) = LOG.lock().unwrap().as_mut() {
        let _ = writeln!(f, "{}", console::strip_ansi_codes(text));
    }
}

pub(super) const LIVE_TEMPLATE: &str = "{percent:>3}% [{bar:28}] {msg}";
pub(super) const VERB_WIDTH: usize = 16;
/// Minimum gap between two sink calls driven by `inc`.
pub(super) const SINK_THROTTLE: Duration = Duration::from_millis(100);
/// Minimum gap between two live-line message refreshes driven by `inc`.
pub(super) const DRAW_THROTTLE: Duration = Duration::from_millis(50);
/// Minimum gap between two evaluations of the O(passes) ETA model.
pub(super) const ETA_THROTTLE: Duration = Duration::from_millis(50);
/// Resolution the bar is drawn at: the time fraction in parts per thousand.
pub(super) const BAR_TICKS: u64 = 1000;

/// `3/8 Triphone`: the stage's position in the plan plus its verb.
pub(super) fn label(st: &State, name: &str) -> String {
    match st.plan.iter().position(|s| s == name) {
        Some(i) => format!("{}/{} {}", i + 1, st.plan.len(), verb(name)),
        None => verb(name).to_string(),
    }
}

/// Human verb for a stage key, right-aligned like cargo's `Compiling`.
pub(super) fn verb(name: &str) -> &str {
    match name {
        "features" => "Features",
        "mono" => "Monophone",
        "tri" => "Triphone",
        "lda" => "LDA+MLLT",
        "sat" => "SAT/fMLLR",
        "sat_2" => "SAT/fMLLR 2",
        "sat_3" => "SAT/fMLLR 3",
        "sat_4" => "SAT/fMLLR 4",
        "pronprob" => "Pron probs",
        "pronprob_2" => "Pron probs 2",
        "final" | "align" => "Aligning",
        "training" => "Trained",
        other => other,
    }
}

/// Snapshot handed to a [`ProgressSink`]. `eta` is `None` until `done > 0`.
#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub done: u64,
    pub total: u64,
    pub elapsed: Duration,
    pub eta: Option<Duration>,
    /// Fraction of the run's *predicted time* that has elapsed, never decreasing
    /// and exactly 1.0 after `finish()`. This is what the bar draws.
    pub fraction: f64,
    /// Counted bars whose label disagreed with the plan's pass order (0 is healthy).
    pub mismatches: u64,
    /// Human stage verb ("Monophone", "SAT/fMLLR 2", ...), "" before the first stage.
    pub stage: String,
    /// Current sub-step text ("iter 12/35 · align 4,608/13,093"), may be "".
    pub step: String,
}

/// Callback invoked with a [`ProgressEvent`]; may be called from any thread.
pub type ProgressSink = Arc<dyn Fn(&ProgressEvent) + Send + Sync>;
/// Everything a [`Bar`] needs, shared with rayon worker threads.
pub(super) struct Shared {
    /// No terminal output at all (piped stage lines included).
    pub(super) hidden: bool,
    /// No live line (non-tty): plain `eprintln` stage lines instead.
    pub(super) plain: bool,
    /// Global counter of finished utterance-passes.
    pub(super) done: AtomicU64,
    /// Total declared by `plan()`; 0 before it is called.
    pub(super) total: AtomicU64,
    /// `Instant::now()` at `plan()`, as nanos since `epoch`; 0 when unplanned.
    pub(super) started_ns: AtomicU64,
    /// Fixed reference point so instants fit in atomics.
    pub(super) epoch: Instant,
    /// Nanos since `epoch` of the last sink call driven by `inc`.
    pub(super) last_sink_ns: AtomicU64,
    /// Nanos since `epoch` of the last live-line refresh driven by `inc`.
    pub(super) last_draw_ns: AtomicU64,
    pub(super) live: ProgressBar,
    /// The plan of passes, their measured times and the ETA model.
    pub(super) tracker: Mutex<Tracker>,
    /// Last ETA the tracker produced, in nanos, and whether it was `Some`.
    /// Re-evaluated at most every `ETA_THROTTLE`; everything that renders
    /// (live line, log suffix, sink) reads this cache.
    pub(super) eta_ns: AtomicU64,
    pub(super) eta_known: AtomicBool,
    /// Nanos since `epoch` of the last ETA evaluation.
    pub(super) last_eta_ns: AtomicU64,
    /// The monotone time fraction, as parts per million so it fits an atomic.
    pub(super) fraction_ppm: AtomicU64,
    pub(super) sink: Mutex<Option<ProgressSink>>,
    pub(super) state: Mutex<State>,
}

#[derive(Default)]
pub(super) struct State {
    /// Ordered stage keys for the whole run (drives the `3/8` label).
    pub(super) plan: Vec<String>,
    pub(super) stage: String,
    pub(super) stage_started: Option<Instant>,
    /// Last iteration summary of the current stage, shown on the live line and
    /// folded into the stage's finished line.
    pub(super) last: Option<String>,
    /// Current sub-step label ("iter 12/35 · align", "accumulate", ...).
    pub(super) step: String,
    /// Units of the current sub-step, when it is counted.
    pub(super) step_len: u64,
    /// Units of the current sub-step already done (across chunks).
    pub(super) step_done: u64,
    /// Gaussians in the model as of the last iteration summary (0 before mono).
    pub(super) gauss: usize,
}

impl Shared {
    /// Key of the stage currently running ("" before the first stage).
    pub(super) fn stage_key(&self) -> String {
        self.state.lock().unwrap().stage.clone()
    }

    pub(super) fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// Time since `plan()`, or zero when the run was never planned.
    pub(super) fn elapsed(&self) -> Duration {
        match self.started_ns.load(Ordering::Relaxed) {
            0 => Duration::ZERO,
            t0 => Duration::from_nanos(self.now_ns().saturating_sub(t0)),
        }
    }

    /// The cached prediction of the remaining time. `None` before the plan's
    /// first pass has finished (there is nothing measured to extrapolate from).
    pub(super) fn eta(&self) -> Option<Duration> {
        if self.started_ns.load(Ordering::Relaxed) == 0 || !self.eta_known.load(Ordering::Relaxed) {
            return None;
        }
        Some(Duration::from_nanos(self.eta_ns.load(Ordering::Relaxed)))
    }

    /// Re-run the O(passes) model and refresh the ETA cache and the bar
    /// fraction, at most every `ETA_THROTTLE`. `force` bypasses the throttle
    /// (stage boundaries, `finish`).
    pub(super) fn refresh_eta(&self, force: bool) {
        if self.started_ns.load(Ordering::Relaxed) == 0 {
            return;
        }
        let now_ns = self.now_ns();
        if !force {
            let last = self.last_eta_ns.load(Ordering::Relaxed);
            if now_ns.saturating_sub(last) < ETA_THROTTLE.as_nanos() as u64
                || self
                    .last_eta_ns
                    .compare_exchange(last, now_ns, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
            {
                return;
            }
        } else {
            self.last_eta_ns.store(now_ns, Ordering::Relaxed);
        }
        let eta = self.tracker.lock().unwrap().eta(Instant::now());
        match eta {
            Some(e) => {
                self.eta_ns.store(e.as_nanos() as u64, Ordering::Relaxed);
                self.eta_known.store(true, Ordering::Relaxed);
            }
            None => self.eta_known.store(false, Ordering::Relaxed),
        }
        self.bump_fraction(eta);
    }

    /// `elapsed / (elapsed + eta)`, clamped so it can only ever grow.
    fn bump_fraction(&self, eta: Option<Duration>) {
        let Some(eta) = eta else { return };
        let el = self.elapsed().as_secs_f64();
        let denom = el + eta.as_secs_f64();
        if denom <= 0.0 {
            return;
        }
        let ppm = ((el / denom).clamp(0.0, 1.0) * 1e6) as u64;
        self.fraction_ppm.fetch_max(ppm, Ordering::Relaxed);
    }

    /// The monotone fraction of the run's predicted time that has elapsed.
    pub(super) fn fraction(&self) -> f64 {
        self.fraction_ppm.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Pin the bar at 100%: the run is over whatever the model predicted.
    pub(super) fn complete(&self) {
        self.fraction_ppm.store(1_000_000, Ordering::Relaxed);
        self.eta_ns.store(0, Ordering::Relaxed);
        self.eta_known.store(true, Ordering::Relaxed);
    }

    pub(super) fn event(&self) -> ProgressEvent {
        let done = self.done.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        let st = self.state.lock().unwrap();
        ProgressEvent {
            done,
            total,
            elapsed: self.elapsed(),
            eta: self.eta(),
            fraction: self.fraction(),
            mismatches: self.tracker.lock().unwrap().mismatches(),
            stage: if st.stage.is_empty() {
                String::new()
            } else {
                verb(&st.stage).to_string()
            },
            step: step_text(&st),
        }
    }

    /// Call the sink unconditionally (stage boundaries, finish).
    pub(super) fn emit(&self) {
        let sink = self.sink.lock().unwrap().clone();
        if let Some(sink) = sink {
            self.last_sink_ns.store(self.now_ns(), Ordering::Relaxed);
            sink(&self.event());
        }
    }

    /// Call the sink at most every `SINK_THROTTLE` (driven by `inc`).
    pub(super) fn emit_throttled(&self) {
        let has = self.sink.lock().unwrap().is_some();
        if !has {
            return;
        }
        let now = self.now_ns();
        let last = self.last_sink_ns.load(Ordering::Relaxed);
        if now.saturating_sub(last) < SINK_THROTTLE.as_nanos() as u64 {
            return;
        }
        if self
            .last_sink_ns
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        let sink = self.sink.lock().unwrap().clone();
        if let Some(sink) = sink {
            sink(&self.event());
        }
    }

    /// Repaint the live line's message from the current state.
    pub(super) fn redraw(&self) {
        if self.hidden || self.plain {
            return;
        }
        let msg = self.live_message();
        self.live.set_message(msg);
    }

    /// Everything after the bar: counts, ETA, stage, sub-step, last summary.
    pub(super) fn live_message(&self) -> String {
        let done = self.done.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        let mut parts = Vec::new();
        if let Some(eta) = self.eta() {
            parts.push(format!("{} left", fmt_duration(eta)));
        }
        let st = self.state.lock().unwrap();
        if !st.stage.is_empty() {
            parts.push(verb(&st.stage).to_string());
        }
        let step = step_text(&st);
        if !step.is_empty() {
            parts.push(step);
        }
        // The raw counters stay visible as text; the percent is the time fraction.
        parts.push(format!(
            "{}/{}",
            group(done as usize),
            group(total as usize)
        ));
        if let Some(last) = &st.last {
            // The iteration count is already on the line (`iter 12/35`).
            let last = last
                .split_once(" iters · ")
                .map_or(last.as_str(), |(_, r)| r);
            parts.push(last.to_string());
        }
        let msg = parts.join(" · ");
        // One line, never wrapped: ` 25% [<bar>] ` takes LIVE_PREFIX_COLS.
        let cols = console::Term::stderr().size().1 as usize;
        let room = cols.saturating_sub(LIVE_PREFIX_COLS).max(20);
        console::truncate_str(&msg, room, "…").into_owned()
    }

    /// Advance the global counter and refresh the line/sink cheaply.
    pub(super) fn add(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.done.fetch_add(n, Ordering::Relaxed);
        self.tracker.lock().unwrap().advance(n, Instant::now());
        self.refresh_eta(false);
        if !self.hidden && !self.plain {
            self.live
                .set_position((self.fraction() * BAR_TICKS as f64) as u64);
            let now = self.now_ns();
            let last = self.last_draw_ns.load(Ordering::Relaxed);
            if now.saturating_sub(last) >= DRAW_THROTTLE.as_nanos() as u64
                && self
                    .last_draw_ns
                    .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                self.redraw();
            }
        }
        self.emit_throttled();
    }

    /// ` · 1234/2600 · 47% · 12m03s elapsed · 4m01s left`, appended to every
    /// stage, stage_done and iteration line written to `--log FILE`. Raw
    /// integers so the log is easy to parse.
    pub(super) fn log_suffix(&self) -> String {
        let done = self.done.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        let pct = self.fraction() * 100.0;
        let eta = match self.eta() {
            Some(e) => fmt_duration(e),
            None => "?".to_string(),
        };
        format!(
            " · {done}/{total} · {pct:.0}% · {} elapsed · {eta} left",
            fmt_duration(self.elapsed())
        )
    }

    /// `47% · 12m03s left` (the time fraction), appended to plain-mode stage lines.
    pub(super) fn overall(&self) -> String {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return String::new();
        }
        let pct = self.fraction() * 100.0;
        match self.eta() {
            Some(eta) => format!("{pct:.0}% · {} left", fmt_duration(eta)),
            None => format!("{pct:.0}%"),
        }
    }
}

/// `iter 12/35 · align 4,608/13,093` from the current sub-step.
pub(super) fn step_text(st: &State) -> String {
    if st.step.is_empty() {
        return String::new();
    }
    if st.step_len > 0 {
        format!(
            "{} {}/{}",
            st.step,
            group(st.step_done.min(st.step_len) as usize),
            group(st.step_len as usize)
        )
    } else {
        st.step.clone()
    }
}

/// Data for one iteration's summary.
pub struct IterationSummary<'a> {
    pub stage: &'a str,
    pub iteration: usize,
    pub num_iterations: usize,
    /// Average acoustic log-likelihood per frame from the accumulated stats.
    pub loglike_per_frame: f64,
    pub gaussians: usize,
    /// Utterances that produced no alignment this iteration.
    pub failed: usize,
    pub elapsed: Duration,
}

impl IterationSummary<'_> {
    /// Compact form for the live line and the stage's finished line.
    pub(super) fn render_short(&self) -> String {
        let mut s = format!(
            "{} iters · {} gauss · loglike {:.2}",
            self.iteration,
            group(self.gaussians),
            self.loglike_per_frame
        );
        if self.failed > 0 {
            s.push_str(&format!(" · {} failed", self.failed));
        }
        s
    }

    /// Full form for piped output.
    pub(super) fn render_long(&self) -> String {
        format!(
            "{} {:>3}/{}  loglike/frame {:>9.4}  gauss {:>6}  failed {:>4}  {}",
            self.stage,
            self.iteration,
            self.num_iterations,
            self.loglike_per_frame,
            self.gaussians,
            self.failed,
            fmt_duration(self.elapsed)
        )
    }
}

/// Thousands separators: 15023 -> "15,023".
pub(super) fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub(super) fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{}m{:02}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!(
            "{}h{:02}m",
            (secs / 3600.0) as u64,
            ((secs % 3600.0) / 60.0) as u64
        )
    }
}
