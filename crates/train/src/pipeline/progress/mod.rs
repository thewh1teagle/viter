//! One progress bar for a whole training run, tqdm-style.
//!
//! The unit is an *utterance-pass*: one utterance processed by one pass (mfcc,
//! graph build, alignment, accumulation, tree stats, ...). [`Progress::plan`]
//! declares the exact number of units the run will increment, so 100% means the
//! run is over; [`Progress::finish`] warns when the counters disagree.
//!
//! The ETA is *not* tqdm's formula on those counters: a pass over the same
//! utterances costs three to seven times more in the late stages (100k gaussians,
//! fMLLR) than in mono, so the raw counter reads 50% low mid-run. Instead the plan
//! lists every pass with its units and model size, each pass is timed as it runs,
//! and the rest are predicted from the measured ones (see [`eta`]). The bar and
//! the percent therefore show the fraction of *predicted time*, `elapsed /
//! (elapsed + eta)`, clamped so it never goes backwards; the raw counters stay
//! visible as text. indicatif's own `{eta}`/`{per_sec}` estimators are never used
//! (they smooth); its [`ProgressBar`] is only the renderer of one live line.
//!
//! Stages are text on that line plus one persistent line each when they finish.
//! Piped or non-tty output gets plain stage lines (with `· 47% · 12m03s left`
//! appended) and every fifth iteration summary; `hidden()` prints nothing.
//! `--log FILE` ([`set_log_file`]) receives every stage line, iteration summary
//! and warning in all modes.

mod eta;
mod render;

use super::plan::WorkPlan;
use console::style;
use eta::Tracker;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use render::{
    BAR_TICKS, LIVE_TEMPLATE, Shared, State, VERB_WIDTH, fmt_duration, group, label, log_line, verb,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use render::{IterationSummary, ProgressEvent, ProgressSink, set_log_file};

/// Owner of the terminal output for a training run. Stages borrow it. All
/// drawing goes to stderr so piping stdout stays clean.
pub struct Progress {
    multi: MultiProgress,
    /// The single live line for the whole run.
    live: ProgressBar,
    shared: Arc<Shared>,
}

impl Progress {
    /// Visible progress on a tty, plain lines otherwise.
    pub fn new() -> Self {
        Self::build(false, !console::user_attended_stderr())
    }

    /// Fully silent: used by `align_corpus` when a caller wants no output, and by tests.
    pub fn hidden() -> Self {
        Self::build(true, true)
    }

    fn build(hidden: bool, plain: bool) -> Self {
        let multi = MultiProgress::with_draw_target(if hidden || plain {
            ProgressDrawTarget::hidden()
        } else {
            ProgressDrawTarget::stderr()
        });
        let live = multi.add(ProgressBar::new(0));
        live.set_style(bar_style());
        Self {
            multi,
            live: live.clone(),
            shared: Arc::new(Shared {
                hidden,
                plain,
                done: AtomicU64::new(0),
                total: AtomicU64::new(0),
                started_ns: AtomicU64::new(0),
                epoch: Instant::now(),
                last_sink_ns: AtomicU64::new(0),
                last_draw_ns: AtomicU64::new(0),
                live,
                tracker: Mutex::new(Tracker::default()),
                eta_ns: AtomicU64::new(0),
                eta_known: AtomicBool::new(false),
                last_eta_ns: AtomicU64::new(0),
                fraction_ppm: AtomicU64::new(0),
                sink: Mutex::new(None),
                state: Mutex::new(State::default()),
            }),
        }
    }

    /// Attach a callback. Called at most every 100 ms from `inc`, plus always on
    /// `stage`, `stage_done` and `finish`. May be called from any thread.
    pub fn with_sink(self, sink: ProgressSink) -> Self {
        *self.shared.sink.lock().unwrap() = Some(sink);
        self
    }

    /// Declare the run: every pass it will perform, in order. Starts the clock.
    ///
    /// The plan fixes both the unit total (100% of the counter is the end of the
    /// run) and the cost model the ETA extrapolates with.
    pub fn plan(&self, plan: &WorkPlan) {
        let stages = plan.keys();
        let total_units = plan.total();
        {
            let mut st = self.shared.state.lock().unwrap();
            st.plan = stages.iter().map(|s| s.to_string()).collect();
        }
        self.shared.total.store(total_units, Ordering::Relaxed);
        self.shared.done.store(0, Ordering::Relaxed);
        self.shared.fraction_ppm.store(0, Ordering::Relaxed);
        self.shared.eta_known.store(false, Ordering::Relaxed);
        let now = Instant::now();
        self.shared.tracker.lock().unwrap().load(&plan.passes, now);
        // Never 0: 0 is the "unplanned" marker.
        self.shared
            .started_ns
            .store(self.shared.now_ns().max(1), Ordering::Relaxed);
        if !self.shared.hidden && !self.shared.plain {
            self.live.set_length(BAR_TICKS);
            self.live.set_position(0);
            self.shared.redraw();
        }
        let chain = stages
            .iter()
            .map(|s| verb(s))
            .collect::<Vec<_>>()
            .join(" → ");
        self.println(format!(
            "{:>w$} {} stages · {} · {} utt-passes",
            style("Plan").green().bold(),
            stages.len(),
            style(chain).dim(),
            style(group(total_units as usize)).dim(),
            w = VERB_WIDTH
        ));
    }

    /// Begin a stage: the live line now names it. Records the stage start time.
    pub fn stage(&self, key: &str, detail: &str) {
        let lab = {
            let mut st = self.shared.state.lock().unwrap();
            st.stage = key.to_string();
            st.stage_started = Some(Instant::now());
            st.last = None;
            st.step.clear();
            st.step_len = 0;
            st.step_done = 0;
            label(&st, key)
        };
        log_line(&format!(
            "{lab:>w$} {detail}{}",
            self.shared.log_suffix(),
            w = VERB_WIDTH
        ));
        if !self.shared.hidden && self.shared.plain {
            eprintln!(
                "{lab:>w$} {detail} · {}",
                self.shared.overall(),
                w = VERB_WIDTH
            );
        }
        self.shared.refresh_eta(true);
        self.shared.redraw();
        self.shared.emit();
        tracing::debug!(stage = key, "{detail}");
    }

    /// Set the sub-step text without a counter.
    pub fn step(&self, what: &str) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.step = what.to_string();
            st.step_len = 0;
            st.step_done = 0;
        }
        self.shared.redraw();
    }

    /// A counted loop of `len` utterances. Every [`Bar::inc`] adds to the global counter.
    pub fn bar(&self, what: &str, len: u64) -> Bar {
        self.make_bar(what.to_string(), len, 0)
    }

    /// Same, labelled `iter {iter}/{total} · {what}`.
    pub fn iter_bar(&self, iter: usize, total: usize, what: &str, len: u64) -> Bar {
        self.make_bar(format!("iter {iter}/{total} · {what}"), len, 0)
    }

    /// An uncounted sub-step (0 units); `inc` on it only updates the label count.
    pub fn spinner(&self, what: &str) -> Bar {
        self.make_bar(what.to_string(), 0, 0)
    }

    fn make_bar(&self, step: String, len: u64, offset: u64) -> Bar {
        // A counted bar claims the plan's next pass. A chunked phase opens one bar
        // per chunk on the same pass; the cursor only moves once its units are in.
        if len > 0 && offset == 0 {
            let hit = self
                .shared
                .tracker
                .lock()
                .unwrap()
                .open(&step, Instant::now());
            if let Some((expected, got)) = hit {
                tracing::warn!(expected, got, "progress plan mismatch");
                log_line(&format!(
                    "progress plan mismatch: expected pass {expected:?}, got {got:?}"
                ));
            }
        }
        {
            let mut st = self.shared.state.lock().unwrap();
            st.step = step;
            st.step_len = len;
            st.step_done = offset;
        }
        self.shared.redraw();
        Bar {
            shared: Arc::clone(&self.shared),
            counted: len > 0,
            started: Instant::now(),
            local: AtomicU64::new(0),
        }
    }

    /// Planned work that turned out unnecessary: advance the counter by `units`.
    /// Consumes the cursor's pass(es) exactly as [`Bar::inc`] would, so the plan
    /// stays in step and the skipped passes count as instantaneous.
    pub fn skip(&self, units: u64) {
        self.shared.add(units);
    }

    /// Counted bars whose step label disagreed with the plan's next pass. A
    /// healthy run reports 0; anything else means the stage code and `plan.rs`
    /// have drifted apart.
    /// Bound the planned gaussian counts by what the data can support (see
    /// `Tracker::cap_gauss`); called once the feature frame count is known.
    /// Use the CPU-measured cost shape for kinds of work not yet measured in this
    /// run (the default shape comes from a GPU run).
    pub fn cpu_prior(&self) {
        self.shared
            .tracker
            .lock()
            .unwrap()
            .with_prior(eta::PRIOR_CPU);
    }

    pub fn cap_gauss(&self, cap: u64) {
        self.shared.tracker.lock().unwrap().cap_gauss(cap);
    }

    pub fn mismatches(&self) -> u64 {
        self.shared.tracker.lock().unwrap().mismatches()
    }

    /// Per-iteration summary: updates the live line; prints a line only when piped
    /// (every fifth iteration and the last) so logs stay readable.
    pub fn iteration_summary(&self, s: &IterationSummary<'_>) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.last = Some(s.render_short());
            st.gauss = s.gaussians;
        }
        log_line(&format!(
            "{:>w$} {}{}",
            "",
            s.render_long(),
            self.shared.log_suffix(),
            w = VERB_WIDTH
        ));
        if !self.shared.hidden
            && self.shared.plain
            && (s.iteration.is_multiple_of(5) || s.iteration == s.num_iterations)
        {
            eprintln!("{:>w$} {}", "", s.render_long(), w = VERB_WIDTH);
        }
        self.shared.redraw();
        tracing::debug!(
            stage = s.stage,
            iteration = s.iteration,
            loglike_per_frame = s.loglike_per_frame,
            gaussians = s.gaussians,
            failed = s.failed,
            elapsed_s = s.elapsed.as_secs_f64(),
            "iteration complete"
        );
    }

    /// Finish a stage: persistent line with the caller's detail, the last
    /// iteration summary and the time since [`Progress::stage`].
    pub fn stage_done(&self, key: &str, detail: &str) {
        let (last, elapsed, lab) = {
            let mut st = self.shared.state.lock().unwrap();
            let e = st.stage_started.take().map(|t| t.elapsed());
            st.step.clear();
            st.step_len = 0;
            st.step_done = 0;
            let last = st.last.take();
            (last, e, label(&st, key))
        };
        let mut parts: Vec<String> = Vec::new();
        if !detail.is_empty() {
            parts.push(detail.to_string());
        }
        if let Some(l) = last {
            parts.push(l);
        }
        let elapsed = elapsed.map(fmt_duration).unwrap_or_default();
        let line = format!(
            "{:>w$} {}  {}",
            style(&lab).green().bold(),
            parts.join(" · "),
            style(&elapsed).dim(),
            w = VERB_WIDTH
        );
        self.shared.refresh_eta(true);
        log_line(&format!("{line}{}", self.shared.log_suffix()));
        if !self.shared.hidden {
            if self.shared.plain {
                eprintln!("{line} · {}", self.shared.overall());
            } else {
                let _ = self.multi.println(line);
            }
        }
        self.shared.redraw();
        self.shared.emit();
    }

    /// One-line message printed above the live line (or plainly when piped).
    pub fn line(&self, text: impl AsRef<str>) {
        self.println(text.as_ref().to_string());
    }

    fn println(&self, text: String) {
        log_line(&text);
        if self.shared.hidden {
            return;
        }
        if self.shared.plain {
            eprintln!("{text}");
        } else {
            let _ = self.multi.println(text);
        }
    }

    pub fn warn(&self, text: impl AsRef<str>) {
        self.println(format!(
            "{:>w$} {}",
            style("warning").yellow().bold(),
            text.as_ref(),
            w = VERB_WIDTH
        ));
        tracing::warn!("{}", text.as_ref());
    }

    /// End of run: clears the live line, warns if `done != total`, final sink call.
    pub fn finish(&self) {
        self.shared.complete();
        self.live.finish_and_clear();
        let done = self.done();
        let total = self.total();
        if total > 0 && done != total {
            tracing::warn!(done, total, "progress plan mismatch");
            log_line(&format!(
                "progress plan mismatch: done {done} != total {total}"
            ));
        }
        self.shared.emit();
    }

    pub fn done(&self) -> u64 {
        self.shared.done.load(Ordering::Relaxed)
    }

    pub fn total(&self) -> u64 {
        self.shared.total.load(Ordering::Relaxed)
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(LIVE_TEMPLATE)
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=> ")
}

/// A counted (or uncounted) sub-step spanning one loop.
pub struct Bar {
    shared: Arc<Shared>,
    /// Counted bars advance the global counter; spinners only relabel.
    counted: bool,
    started: Instant,
    /// Units this bar itself counted (one chunk of a phase), for the pass log.
    local: AtomicU64,
}

impl Drop for Bar {
    /// Every counted bar leaves one `pass` line in `--log FILE`: what ran, how
    /// many units, how long, and the model size it ran against. That is the raw
    /// data behind the ETA model; nothing is estimated from it at run time.
    fn drop(&mut self) {
        if !self.counted {
            return;
        }
        let units = self.local.load(Ordering::Relaxed);
        let (step, gauss) = {
            let st = self.shared.state.lock().unwrap();
            (st.step.clone(), st.gauss)
        };
        log_line(&format!(
            "pass\t{}\t{step}\tunits={units}\tsecs={:.3}\tgauss={gauss}\tat={:.3}",
            self.shared.stage_key(),
            self.started.elapsed().as_secs_f64(),
            self.shared.elapsed().as_secs_f64()
        ));
    }
}

impl Bar {
    /// Record `n` more utterances of this sub-step. On a counted bar this also
    /// advances the run's global counter.
    pub fn inc(&self, n: u64) {
        if n == 0 {
            return;
        }
        {
            let mut st = self.shared.state.lock().unwrap();
            st.step_done = st.step_done.saturating_add(n);
        }
        if self.counted {
            self.local.fetch_add(n, Ordering::Relaxed);
            self.shared.add(n);
        }
    }

    pub fn finish(self) {}

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// One logical pass spanning several chunks that each open their own [`Bar`].
///
/// Chunked passes interleave two steps (align, accumulate) over the same chunk
/// loop, and all of `Progress`'s bars share one live line, so a step cannot hold
/// a bar open across chunks. A phase instead makes a fresh bar per chunk whose
/// label count starts where earlier chunks stopped, keeping the displayed total
/// at the whole subset (`align 4,608/13,093`).
pub struct Phase<'p> {
    progress: &'p Progress,
    step: String,
    len: u64,
    done: u64,
}

impl<'p> Phase<'p> {
    pub fn new(progress: &'p Progress, what: &str, len: u64) -> Self {
        Self {
            progress,
            step: what.to_string(),
            len,
            done: 0,
        }
    }

    pub fn iter(progress: &'p Progress, iter: usize, total: usize, what: &str, len: u64) -> Self {
        Self {
            progress,
            step: format!("iter {iter}/{total} · {what}"),
            len,
            done: 0,
        }
    }

    /// A bar for the next chunk, labelled from where the previous chunk stopped.
    pub fn bar(&self) -> Bar {
        self.progress
            .make_bar(self.step.clone(), self.len, self.done)
    }

    pub fn done(&mut self, n: u64) {
        self.done += n;
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
