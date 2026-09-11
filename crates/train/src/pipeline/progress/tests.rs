//! Tests for the run-wide progress bar: the plan cursor, the counters, the
//! time fraction and the sink contract.

use super::super::plan::{PlannedPass, PlannedStage};
use super::render::step_text;
use super::*;
use std::sync::Mutex as StdMutex;
fn pass(stage: &str, step: &'static str, units: u64) -> PlannedPass {
    PlannedPass {
        stage: stage.to_string(),
        step,
        units,
        gauss: 1000,
        heavy: false,
    }
}

/// A three-stage, 100-unit plan: mfcc 40, align 35, accumulate 25.
fn work_plan() -> WorkPlan {
    WorkPlan {
        stages: vec![
            PlannedStage {
                key: "features".into(),
                units: 40,
            },
            PlannedStage {
                key: "mono".into(),
                units: 35,
            },
            PlannedStage {
                key: "tri".into(),
                units: 25,
            },
        ],
        passes: vec![
            pass("features", "mfcc", 40),
            pass("mono", "align", 35),
            pass("tri", "accumulate", 25),
        ],
    }
}

fn planned() -> Progress {
    let p = Progress::hidden();
    p.plan(&work_plan());
    p
}

#[test]
fn hidden_progress_does_not_panic() {
    let p = planned();
    p.stage("mono", "40 iterations");
    let bar = p.bar("mfcc", 10);
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
fn bars_summing_to_the_plan_reach_the_total() {
    let p = planned();
    assert_eq!(p.total(), 100);
    assert_eq!(p.done(), 0);
    for (what, len) in [("mfcc", 40u64), ("align", 35), ("accumulate", 25)] {
        let b = p.bar(what, len);
        for _ in 0..len {
            b.inc(1);
        }
        b.finish();
    }
    p.finish();
    assert_eq!(p.done(), p.total());
}

#[test]
fn skip_counts_toward_the_total() {
    let p = planned();
    let b = p.bar("mfcc", 60);
    b.inc(60);
    b.finish();
    p.skip(40);
    assert_eq!(p.done(), 100);
}

#[test]
fn spinner_inc_does_not_count() {
    let p = planned();
    let s = p.spinner("build tree");
    s.inc(7);
    s.finish();
    assert_eq!(p.done(), 0);
}

#[test]
fn chunked_inc_matches_a_single_bar() {
    let p = planned();
    let mut phase = Phase::new(&p, "align", 100);
    for _ in 0..4 {
        let b = phase.bar();
        b.inc(25);
        b.finish();
        phase.done(25);
    }
    assert_eq!(p.done(), 100);
}

#[test]
fn phase_bar_offsets_the_label_count() {
    let p = planned();
    let mut phase = Phase::iter(&p, 3, 8, "align", 100);
    let b = phase.bar();
    b.inc(10);
    b.finish();
    phase.done(10);
    let _b2 = phase.bar();
    let st = p.shared.state.lock().unwrap();
    assert_eq!(st.step, "iter 3/8 · align");
    assert_eq!(st.step_done, 10);
    assert_eq!(st.step_len, 100);
    assert_eq!(step_text(&st), "iter 3/8 · align 10/100");
}

#[test]
fn sink_fires_on_stage_boundaries_and_is_throttled_on_inc() {
    let events: Arc<StdMutex<Vec<ProgressEvent>>> = Arc::new(StdMutex::new(Vec::new()));
    let rec = Arc::clone(&events);
    let p = Progress::hidden().with_sink(Arc::new(move |e: &ProgressEvent| {
        rec.lock().unwrap().push(e.clone());
    }));
    p.plan(&WorkPlan {
        stages: vec![PlannedStage {
            key: "mono".into(),
            units: 100,
        }],
        passes: vec![pass("mono", "align", 100)],
    });
    p.stage("mono", "40 iterations");
    let n_after_stage = events.lock().unwrap().len();
    assert_eq!(n_after_stage, 1, "stage() always emits");
    assert_eq!(events.lock().unwrap()[0].stage, "Monophone");

    // 100 fast increments must not produce 100 sink calls (100 ms throttle).
    let b = p.bar("align", 100);
    for _ in 0..100 {
        b.inc(1);
    }
    b.finish();
    let n_after_inc = events.lock().unwrap().len();
    assert!(
        n_after_inc - n_after_stage <= 3,
        "inc sink calls should be throttled, got {}",
        n_after_inc - n_after_stage
    );

    p.stage_done("mono", "done");
    assert_eq!(events.lock().unwrap().len(), n_after_inc + 1);
    p.finish();
    let evs = events.lock().unwrap();
    assert_eq!(evs.len(), n_after_inc + 2);
    let last = evs.last().unwrap();
    assert_eq!(last.done, 100);
    assert_eq!(last.total, 100);
    // done is monotone non-decreasing across every event.
    assert!(evs.windows(2).all(|w| w[0].done <= w[1].done));
}

#[test]
fn eta_is_none_until_a_pass_finishes_then_drives_the_fraction() {
    let p = planned();
    // Nothing measured yet: no prediction, and the bar sits at 0.
    assert!(p.shared.eta().is_none());
    assert_eq!(p.shared.fraction(), 0.0);
    // Finishing the first pass gives the model something to extrapolate from.
    let b = p.bar("mfcc", 40);
    b.inc(40);
    b.finish();
    p.shared.refresh_eta(true);
    assert!(p.shared.eta().is_some());
    let mid = p.shared.fraction();
    assert!(mid > 0.0 && mid < 1.0, "{mid}");
    // The fraction only ever grows, and finish() pins it at exactly 1.0.
    let b = p.bar("align", 35);
    b.inc(35);
    b.finish();
    assert!(p.shared.fraction() >= mid);
    p.finish();
    assert_eq!(p.shared.fraction(), 1.0);
    assert_eq!(p.shared.eta(), Some(Duration::ZERO));
}

#[test]
fn a_bar_that_disagrees_with_the_plan_is_counted_as_a_mismatch() {
    let p = planned();
    assert_eq!(p.mismatches(), 0);
    // The plan's first pass is "mfcc"; opening "accumulate" is a drift.
    let b = p.bar("accumulate", 25);
    b.inc(25);
    b.finish();
    assert_eq!(p.mismatches(), 1);
    // ... and it resynced onto the tri accumulate pass, so the count is right.
    assert_eq!(p.done(), 25);
}

#[test]
fn the_event_carries_the_fraction_and_the_mismatch_count() {
    let events: Arc<StdMutex<Vec<ProgressEvent>>> = Arc::new(StdMutex::new(Vec::new()));
    let rec = Arc::clone(&events);
    let p = Progress::hidden().with_sink(Arc::new(move |e: &ProgressEvent| {
        rec.lock().unwrap().push(e.clone());
    }));
    p.plan(&work_plan());
    for (what, len) in [("mfcc", 40u64), ("align", 35), ("accumulate", 25)] {
        let b = p.bar(what, len);
        b.inc(len);
        b.finish();
        p.stage_done(what, "");
    }
    p.finish();
    let evs = events.lock().unwrap();
    // Monotone fraction, ending at exactly 1.0, with the plan never drifting.
    assert!(evs.windows(2).all(|w| w[0].fraction <= w[1].fraction));
    let last = evs.last().unwrap();
    assert_eq!(last.fraction, 1.0);
    assert_eq!(last.mismatches, 0);
    assert_eq!(last.done, last.total);
}

#[test]
fn stage_done_elapsed_is_measured_since_stage() {
    let p = planned();
    p.stage("mono", "40 iterations");
    let started = {
        let st = p.shared.state.lock().unwrap();
        st.stage_started.unwrap()
    };
    assert!(started.elapsed() < Duration::from_secs(1));
    p.stage_done("mono", "done");
    // stage_done consumes the start instant, so a second call finds none.
    assert!(p.shared.state.lock().unwrap().stage_started.is_none());
}

#[test]
fn labels_use_the_plan_index() {
    let p = planned();
    let st = p.shared.state.lock().unwrap();
    assert_eq!(label(&st, "tri"), "3/3 Triphone");
    assert_eq!(label(&st, "sat"), "SAT/fMLLR");
}

#[test]
fn log_suffix_carries_raw_counters() {
    let p = planned();
    assert!(
        p.shared.log_suffix().starts_with(" · 0/100 · 0% · "),
        "{}",
        p.shared.log_suffix()
    );
    assert!(p.shared.log_suffix().ends_with("? left"));
    let b = p.bar("mfcc", 40);
    b.inc(1500);
    b.finish();
    let s = p.shared.log_suffix();
    // Raw integers, no thousands separators.
    assert!(s.contains(" · 1500/100 · "), "{s}");
    assert!(s.contains(" elapsed · "), "{s}");
}

#[test]
fn duration_and_grouping() {
    assert_eq!(fmt_duration(Duration::from_secs_f64(3.16)), "3.2s");
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
    assert_eq!(verb("sat_3"), "SAT/fMLLR 3");
    assert_eq!(verb("pronprob"), "Pron probs");
    assert_eq!(verb("custom"), "custom");
}
