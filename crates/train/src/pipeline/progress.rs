//! Cargo-style progress reporting shared by every training stage.
//!
//! One live line per stage that rewrites in place (`   Triphone iter 12/35 · align
//! ━━━━╸━━━ 3,400/5,000  loglike -98.1  eta 1m02s`), and one persistent line per
//! finished stage with its key numbers and elapsed time. Warnings print above the
//! live line so they are never lost. Piped or non-tty output gets plain stage
//! lines plus every fifth iteration; `RUST_LOG=info` restores the per-iteration
//! log through tracing.

use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Optional plain-text log every `Progress` appends to (stage lines, every
/// iteration, warnings), ANSI-free. Set once from the CLI (`--log FILE`).
static LOG: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// Route a copy of all progress output to `path` (appended, no colors).
pub fn set_log_file(path: &std::path::Path) -> std::io::Result<()> {
    let f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    *LOG.lock().unwrap() = Some(f);
    Ok(())
}

fn log_line(text: &str) {
    if let Some(f) = LOG.lock().unwrap().as_mut() {
        let _ = writeln!(f, "{}", console::strip_ansi_codes(text));
    }
}

const LIVE_TEMPLATE: &str =
    "{prefix:>12.green.bold} {msg} {bar:24.cyan/dim} {pos}/{len} {elapsed_precise:.dim}";
const SPIN_TEMPLATE: &str = "{prefix:>12.green.bold} {msg} {spinner:.cyan}";
const VERB_WIDTH: usize = 12;

/// Human verb for a stage key, right-aligned like cargo's `Compiling`.
fn verb(name: &str) -> &str {
    match name {
        "features" => "Features",
        "mono" => "Monophone",
        "tri" => "Triphone",
        "lda" => "LDA+MLLT",
        "sat" => "SAT/fMLLR",
        "final" | "align" => "Aligning",
        "training" => "Trained",
        other => other,
    }
}

/// Owner of the terminal output for a training run. Stages borrow it. All
/// drawing goes to stderr so piping stdout stays clean.
pub struct Progress {
    multi: MultiProgress,
    /// The single live line for the current stage.
    live: ProgressBar,
    quiet: bool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Ordered stage names for the whole run.
    plan: Vec<String>,
    stage: String,
    stage_started: Option<Instant>,
    /// Last iteration summary of the current stage, shown on the live line and
    /// folded into the stage's finished line.
    last: Option<String>,
    /// Current sub-step label ("align", "accumulate", ...).
    step: String,
}

impl Progress {
    /// Visible progress on a tty, plain lines otherwise.
    pub fn new() -> Self {
        let quiet = !console::user_attended_stderr();
        let multi = MultiProgress::with_draw_target(if quiet {
            ProgressDrawTarget::hidden()
        } else {
            ProgressDrawTarget::stderr()
        });
        let live = multi.add(ProgressBar::hidden());
        Self { multi, live, quiet, state: Mutex::new(State::default()) }
    }

    /// Fully silent: used by `align_corpus` when a caller wants no output, and by tests.
    pub fn hidden() -> Self {
        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());
        let live = multi.add(ProgressBar::hidden());
        Self { multi, live, quiet: true, state: Mutex::new(State::default()) }
    }

    /// Declare the ordered list of stages for this run and print the chain once.
    pub fn plan(&self, stages: &[&str]) {
        self.state.lock().unwrap().plan = stages.iter().map(|s| s.to_string()).collect();
        let chain = stages.iter().map(|s| verb(s)).collect::<Vec<_>>().join(" → ");
        self.println(format!(
            "{:>w$} {} stages · {}",
            style("Plan").green().bold(),
            stages.len(),
            style(chain).dim(),
            w = VERB_WIDTH
        ));
    }

    /// Begin a stage: the live line now belongs to it.
    pub fn stage(&self, name: &str, detail: &str) {
        {
            let mut st = self.state.lock().unwrap();
            st.stage = name.to_string();
            st.stage_started = Some(Instant::now());
            st.last = None;
            st.step.clear();
        }
        let counter = self.counter(name);
        log_line(&format!("{:>w$} {counter}{detail}", verb(name), w = VERB_WIDTH));
        if self.quiet {
            eprintln!("{:>w$} {counter}{detail}", verb(name), w = VERB_WIDTH);
        } else {
            self.live.set_style(spin_style());
            self.live.set_prefix(verb(name).to_string());
            self.live.set_message(format!("{counter}{}", style(detail).dim()));
            self.live.enable_steady_tick(Duration::from_millis(100));
        }
        tracing::debug!(stage = name, "{detail}");
    }

    fn counter(&self, name: &str) -> String {
        let st = self.state.lock().unwrap();
        st.plan
            .iter()
            .position(|s| s == name)
            .map(|i| format!("[{}/{}] ", i + 1, st.plan.len()))
            .unwrap_or_default()
    }

    /// A determinate bar with `len` units of work on the live line.
    pub fn bar(&self, msg: impl Into<String>, len: u64) -> Bar {
        let msg = msg.into();
        self.state.lock().unwrap().step = msg.clone();
        if !self.quiet {
            self.live.set_style(bar_style());
            self.live.set_length(len);
            self.live.set_position(0);
            self.live.set_message(self.live_message(&msg));
        }
        Bar { pb: self.live.clone(), quiet: self.quiet }
    }

    /// A bar labelled for one step of one iteration: "iter 12/40 · align".
    pub fn iter_bar(&self, _stage: &str, iter: usize, total: usize, step: &str, len: u64) -> Bar {
        self.bar(format!("iter {iter}/{total} · {step}"), len)
    }

    /// A spinner for work whose size is not known up front.
    pub fn spinner(&self, msg: impl Into<String>) -> Bar {
        let msg = msg.into();
        self.state.lock().unwrap().step = msg.clone();
        if !self.quiet {
            self.live.set_style(spin_style());
            self.live.set_message(self.live_message(&msg));
        }
        Bar { pb: self.live.clone(), quiet: self.quiet }
    }

    /// Live-line text: the sub-step plus the last iteration summary, dimmed.
    fn live_message(&self, step: &str) -> String {
        let st = self.state.lock().unwrap();
        match &st.last {
            Some(last) => format!("{step}  {}", style(last).dim()),
            None => step.to_string(),
        }
    }

    /// One-line message printed above the live line (or plainly when piped).
    pub fn line(&self, text: impl AsRef<str>) {
        self.println(text.as_ref().to_string());
    }

    fn println(&self, text: String) {
        log_line(&text);
        if self.quiet {
            eprintln!("{text}");
        } else {
            let _ = self.multi.println(text);
        }
    }

    /// Per-iteration summary: updates the live line; prints a line only when piped
    /// (every fifth iteration and the last) so logs stay readable.
    pub fn iteration_summary(&self, s: &IterationSummary<'_>) {
        let short = s.render_short();
        self.state.lock().unwrap().last = Some(short.clone());
        log_line(&format!("{:>w$} {}", "", s.render_long(), w = VERB_WIDTH));
        if self.quiet {
            if s.iteration % 5 == 0 || s.iteration == s.num_iterations {
                eprintln!("{:>w$} {}", "", s.render_long(), w = VERB_WIDTH);
            }
        } else {
            let step = self.state.lock().unwrap().step.clone();
            self.live.set_message(format!("{step}  {}", style(&short).dim()));
        }
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
    /// iteration summary and the elapsed time.
    pub fn stage_done(&self, name: &str, detail: &str) {
        let (last, elapsed) = {
            let mut st = self.state.lock().unwrap();
            let e = st.stage_started.take().map(|t| t.elapsed());
            (st.last.take(), e)
        };
        let mut parts: Vec<String> = Vec::new();
        if !detail.is_empty() {
            parts.push(detail.to_string());
        }
        if let Some(l) = last {
            parts.push(l);
        }
        let elapsed = elapsed.map(fmt_duration).unwrap_or_default();
        if !self.quiet {
            self.live.set_style(spin_style());
            self.live.set_message(String::new());
            self.live.disable_steady_tick();
        }
        self.println(format!(
            "{:>w$} {}  {}",
            style(verb(name)).green().bold(),
            parts.join(" · "),
            style(elapsed).dim(),
            w = VERB_WIDTH
        ));
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

    /// Clear the live line when a run finishes.
    pub fn finish(&self) {
        self.live.finish_and_clear();
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
        .progress_chars("━╸ ")
}

fn spin_style() -> ProgressStyle {
    ProgressStyle::with_template(SPIN_TEMPLATE).unwrap_or_else(|_| ProgressStyle::default_spinner())
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
    fn render_short(&self) -> String {
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
    fn render_long(&self) -> String {
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

/// Handle on the live line for one step. Dropping it leaves the line in place;
/// the next step or `stage_done` replaces it.
pub struct Bar {
    pb: ProgressBar,
    quiet: bool,
}

impl Bar {
    pub fn inc(&self, n: u64) {
        if !self.quiet {
            self.pb.inc(n);
        }
    }
    pub fn set_position(&self, n: u64) {
        if !self.quiet {
            self.pb.set_position(n);
        }
    }
    pub fn set_message(&self, msg: impl Into<String>) {
        if !self.quiet {
            self.pb.set_message(msg.into());
        }
    }
    pub fn set_length(&self, len: u64) {
        if !self.quiet {
            self.pb.set_length(len);
        }
    }
    pub fn elapsed(&self) -> Duration {
        self.pb.elapsed()
    }
    pub fn finish(self) {}
}

/// Thousands separators: 15023 -> "15,023".
fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{}m{:02}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{}h{:02}m", (secs / 3600.0) as u64, ((secs % 3600.0) / 60.0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_progress_does_not_panic() {
        let p = Progress::hidden();
        p.plan(&["mono", "tri"]);
        p.stage("mono", "40 iterations");
        let bar = p.bar("test", 10);
        bar.inc(5);
        bar.finish();
        p.iteration_summary(&IterationSummary {
            stage: "mono",
            iteration: 1,
            num_iterations: 40,
            loglike_per_frame: -55.5,
            gaussians: 135,
            failed: 0,
            elapsed: Duration::from_secs(3),
        });
        p.stage_done("mono", "1000 gaussians");
        p.finish();
    }

    #[test]
    fn duration_and_grouping() {
        assert_eq!(fmt_duration(Duration::from_secs_f64(3.14)), "3.1s");
        assert_eq!(fmt_duration(Duration::from_secs(90)), "1m30s");
        assert_eq!(fmt_duration(Duration::from_secs(3725)), "1h02m");
        assert_eq!(group(15023), "15,023");
        assert_eq!(group(999), "999");
        assert_eq!(group(1000000), "1,000,000");
    }

    #[test]
    fn verbs_are_stable() {
        assert_eq!(verb("mono"), "Monophone");
        assert_eq!(verb("sat"), "SAT/fMLLR");
        assert_eq!(verb("custom"), "custom");
    }
}
