//! The work plan: every utterance-pass the whole training run will perform.
//!
//! One *unit* is one utterance processed by one pass (mfcc, graph build, alignment,
//! accumulation, tree statistics, ...). Every counted pass in the pipeline is a loop
//! that calls `Bar::inc` exactly once per utterance, including utterances that fail,
//! so the number enumerated here is the number the run increments. Passes that are
//! not per-utterance loops (tree questions, tree build, model init, the per-speaker
//! fMLLR solve) count zero.
//!
//! The plan is a flat list of [`PlannedPass`] in execution order; [`PlannedStage`] is
//! the per-stage aggregate of that list. Each pass also carries the number of
//! gaussians the model has while it runs, which the progress predictor uses to
//! extrapolate a pass's cost across stages.
//!
//! This module deliberately does not depend on `progress.rs`: it is pure arithmetic
//! over the config, so it can be unit tested without any terminal machinery.

use crate::config::{GaussianSchedule, StageSpec, TrainConfig};

/// One planned stage: key (as passed to `Progress::stage`) and its units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStage {
    pub key: String,
    pub units: u64,
}

/// One logical pass over a subset of utterances.
///
/// A chunked pass (one that opens several `Bar`s, one per chunk) is a single entry:
/// `units` is the whole pass. `step` is the bar label without the `iter i/N · `
/// prefix, so `Progress` can match a bar against the pass the cursor is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedPass {
    /// Stage key, as `Progress::stage` receives it.
    pub stage: String,
    /// Bar label without the `iter i/N · ` prefix.
    pub step: &'static str,
    pub units: u64,
    /// Gaussians of the model this pass runs against.
    pub gauss: u64,
    /// The pass re-derives the stage's feature view (empty or invalidated cache):
    /// the first pass of a stage's loop, the accumulate after an MLLT/fMLLR update,
    /// and the setup passes that derive the whole subset. Such a pass costs several
    /// seconds more than its twin, so the ETA model prices the two kinds apart.
    pub heavy: bool,
}

/// The passes a run will perform, in order, and their per-stage aggregate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkPlan {
    pub stages: Vec<PlannedStage>,
    pub passes: Vec<PlannedPass>,
}

impl WorkPlan {
    /// Total utterance-passes of the run.
    pub fn total(&self) -> u64 {
        self.stages.iter().map(|s| s.units).sum()
    }

    /// Stage keys in order ("features" first, "final" last when it runs).
    pub fn keys(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.key.as_str()).collect()
    }
}

/// Number of iterations in `1..=num_iterations` that appear in `schedule`.
///
/// Stage schedules (`MonoConfig::realignment_iterations` and friends) are lists that
/// may include 0 or values past the last iteration; the loop in
/// `pipeline::iterate::run_iterations` only runs `1..=num_iterations` and asks
/// `contains` for each, so only that intersection produces work.
fn hits(schedule: &[usize], iteration: usize) -> bool {
    schedule.contains(&iteration)
}

/// Accumulates the passes of one run, tracking the stage each belongs to.
struct Builder {
    stages: Vec<PlannedStage>,
    passes: Vec<PlannedPass>,
}

impl Builder {
    fn new() -> Self {
        Self {
            stages: Vec::new(),
            passes: Vec::new(),
        }
    }

    fn stage(&mut self, key: impl Into<String>) {
        self.stages.push(PlannedStage {
            key: key.into(),
            units: 0,
        });
    }

    /// Add a pass to the stage most recently opened.
    fn pass(&mut self, step: &'static str, units: u64, gauss: u64) {
        self.push(step, units, gauss, false);
    }

    /// A pass that re-derives features first (see `PlannedPass::heavy`).
    fn pass_heavy(&mut self, step: &'static str, units: u64, gauss: u64) {
        self.push(step, units, gauss, true);
    }

    fn push(&mut self, step: &'static str, units: u64, gauss: u64, heavy: bool) {
        let stage = self.stages.last_mut().expect("a stage is open");
        stage.units += units;
        let stage = stage.key.clone();
        self.passes.push(PlannedPass {
            stage,
            step,
            units,
            gauss,
            heavy,
        });
    }

    fn finish(self) -> WorkPlan {
        WorkPlan {
            stages: self.stages,
            passes: self.passes,
        }
    }
}

/// The five passes every tree-building stage (tri, lda, sat) runs before its
/// iteration loop, minus stage-specific extras. `prev_gauss` is the gaussian count
/// of the model these passes run against — the *previous* stage's final model, since
/// this stage's own model does not exist until the tree is built.
fn tree_stage_setup(b: &mut Builder, s: u64, prev_gauss: u64, lda: bool) {
    // `align_subset_with_previous`: derive the subset's features, build graphs with
    // the previous model, align.
    b.pass_heavy("graphs", s, prev_gauss);
    b.pass("align (previous model)", s, prev_gauss);
    if lda {
        b.pass("lda stats", s, prev_gauss);
    }
    // Tree statistics derive the whole subset first (not chunked, see `tri.rs`).
    b.pass_heavy("tree stats", s, prev_gauss);
    b.pass("convert alignments", s, prev_gauss);
    // Graphs for the new tree; the new model exists but has not been updated yet.
    b.pass("graphs", s, prev_gauss);
}

/// The iteration loop shared by every stage: realign when scheduled, optionally a
/// hook pass, then accumulate. `gauss` yields the gaussian target in force at each
/// iteration. `hook_bar` names the hook with its own per-utterance bar (MLLT) and
/// the iterations it runs on; `clear_iters` are the iterations whose hook
/// invalidates the derived-feature cache (MLLT or fMLLR), so that iteration's
/// accumulate re-derives everything and is heavy. The loop starts with an empty
/// cache, so its first pass is heavy too.
fn iteration_loop(
    b: &mut Builder,
    s: u64,
    num_iterations: usize,
    realign: &[usize],
    hook_bar: Option<(&'static str, &[usize])>,
    clear_iters: &[usize],
    sched: &mut GaussianSchedule,
) {
    let mut first = true;
    for i in 1..=num_iterations {
        let g = sched.current() as u64;
        if hits(realign, i) {
            if first {
                b.pass_heavy("align", s, g);
            } else {
                b.pass("align", s, g);
            }
            first = false;
        }
        if let Some((name, iters)) = hook_bar
            && hits(iters, i)
        {
            if first {
                b.pass_heavy(name, s, g);
            } else {
                b.pass(name, s, g);
            }
            first = false;
        }
        if first || hits(clear_iters, i) {
            b.pass_heavy("accumulate", s, g);
        } else {
            b.pass("accumulate", s, g);
        }
        first = false;
        sched.step(i);
    }
}

/// Enumerate every utterance-pass the run will perform, in execution order.
///
/// `final_alignment` adds the "final" stage (two full-corpus passes, plus one more
/// when the schedule trained a SAT model, which aligns twice).
///
/// Gaussian counts follow this rule: passes that run before a stage's first
/// iteration update use the previous stage's final gaussian count (0 for mono and
/// features, whose models do not exist yet); the passes of iteration `i` use the
/// `GaussianSchedule` target in force for iteration `i` (i.e. after iteration
/// `i-1`'s update). Mono's `initial_gaussians` is the configured floor (135 by
/// default) — the true count `gmm_init_mono` produces is one per pdf and is not
/// knowable at plan time. Exactness here is not critical: `gauss` only drives the
/// predictor's across-stage linear extrapolation, while order and units must match
/// the run exactly.
pub fn work_plan(cfg: &TrainConfig, num_utts: usize, final_alignment: bool) -> WorkPlan {
    let n = num_utts as u64;
    let schedule = cfg.effective_schedule(num_utts);
    let mut b = Builder::new();

    b.stage("features");
    b.pass("mfcc", n, 0);

    // Gaussian count of the model the previous stage left behind.
    let mut prev_gauss = 0u64;

    for spec in &schedule {
        // `StageCtx::subset_for`: 0 means "the whole corpus".
        let s = match cfg.subset_size(spec.subset(), num_utts) {
            0 => n,
            want => want as u64,
        };
        b.stage(spec.key());
        match spec {
            StageSpec::Mono { .. } => {
                let c = &cfg.mono;
                let mut sched = GaussianSchedule::new(
                    c.initial_gaussians,
                    c.max_gaussians,
                    c.final_gaussian_iteration(),
                );
                b.pass("graphs", s, 0);
                b.pass("equal align", s, 0);
                // Iteration 0's accumulate over the flat-start alignment (derives).
                b.pass_heavy("accumulate", s, sched.current() as u64);
                sched.step(0);
                iteration_loop(
                    &mut b,
                    s,
                    c.num_iterations,
                    &c.realignment_iterations(),
                    None,
                    &[],
                    &mut sched,
                );
                prev_gauss = sched.current() as u64;
            }
            StageSpec::Tri {
                num_leaves,
                max_gaussians,
                ..
            } => {
                // `pipeline::mod` overrides the tree size per round from the spec.
                let c = &cfg.tri;
                let mut sched = GaussianSchedule::new(
                    *num_leaves,
                    *max_gaussians,
                    c.final_gaussian_iteration(),
                );
                tree_stage_setup(&mut b, s, prev_gauss, false);
                iteration_loop(
                    &mut b,
                    s,
                    c.num_iterations,
                    &c.realignment_iterations(),
                    None,
                    &[],
                    &mut sched,
                );
                prev_gauss = sched.current() as u64;
            }
            StageSpec::Lda {
                num_leaves,
                max_gaussians,
                ..
            } => {
                let c = &cfg.lda;
                let mut sched = GaussianSchedule::new(
                    *num_leaves,
                    *max_gaussians,
                    c.final_gaussian_iteration(),
                );
                tree_stage_setup(&mut b, s, prev_gauss, true);
                iteration_loop(
                    &mut b,
                    s,
                    c.num_iterations,
                    &c.realignment_iterations(),
                    Some(("mllt", &c.mllt_iterations)),
                    &c.mllt_iterations,
                    &mut sched,
                );
                prev_gauss = sched.current() as u64;
            }
            StageSpec::Sat {
                num_iterations,
                quick,
                num_leaves,
                max_gaussians,
                ..
            } => {
                // `pipeline::mod` overrides these per round from the `StageSpec`.
                let c = crate::config::SatConfig {
                    num_iterations: *num_iterations,
                    quick: *quick,
                    num_leaves: *num_leaves,
                    max_gaussians: *max_gaussians,
                    ..cfg.sat.clone()
                };
                let mut sched = GaussianSchedule::new(
                    c.initial_gaussians(),
                    c.max_gaussians,
                    c.final_gaussian_iteration(),
                );
                tree_stage_setup(&mut b, s, prev_gauss, false);
                // The fMLLR hook solves per speaker, not per utterance: 0 units, so
                // it is not a pass at all.
                iteration_loop(
                    &mut b,
                    s,
                    c.num_iterations,
                    &c.realignment_iterations(),
                    None,
                    &c.fmllr_iterations(),
                    &mut sched,
                );
                prev_gauss = sched.current() as u64;
                // The two-feats statistics pass that builds the speaker-independent
                // companion model. Skipped when no fMLLR transforms exist, in which
                // case the stage code calls `Progress::skip(s)`.
                b.pass("two-feats stats", s, prev_gauss);
            }
            StageSpec::PronProbs { .. } => {
                // The fMLLR second pass is disabled in the code.
                b.pass("graphs", s, prev_gauss);
                b.pass("align", s, prev_gauss);
            }
        }
    }

    if final_alignment {
        let sat = schedule.iter().any(|s| matches!(s, StageSpec::Sat { .. }));
        b.stage("final");
        b.pass("graphs", n, prev_gauss);
        b.pass("final align", n, prev_gauss);
        if sat {
            // With a speaker-adapted model, a second alignment pass under the fMLLR
            // transforms; the graphs from the first pass are reused.
            b.pass("final align (fmllr)", n, prev_gauss);
        }
    }

    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Stages;

    fn units(plan: &WorkPlan, key: &str) -> u64 {
        plan.stages
            .iter()
            .find(|s| s.key == key)
            .unwrap_or_else(|| panic!("no stage {key} in {:?}", plan.keys()))
            .units
    }

    fn steps(plan: &WorkPlan, key: &str) -> Vec<&'static str> {
        plan.passes
            .iter()
            .filter(|p| p.stage == key)
            .map(|p| p.step)
            .collect()
    }

    // Default schedules, from `config.rs` tests:
    //   mono: I=40, realign [0,1..10,12,14,16,18,20,23,26,29,32,35,38]
    //         -> intersect 1..=40 drops the 0: 21 realignments.
    //   tri:  I=35, realign [10,20,30] -> 3.
    //   lda:  I=35, realign [10,20,30] -> 3, mllt [2,4,6,12] -> 4.
    //   sat:  I=35, realign [10,20,30] -> 3 (non-quick);
    //         quick: I=20, realign [10,15] -> 2.

    #[test]
    fn small_corpus_uses_the_whole_corpus_everywhere() {
        let cfg = TrainConfig::default();
        let n = 20u64;
        let plan = work_plan(&cfg, n as usize, true);

        // Every subset (10000..150000) exceeds n=20, so subset_size returns 0 = all.
        // Optional rounds whose subset > n end the schedule: sat_4 has subset 0 so it
        // is not optional-dropped by size, but pronprob_2 (subset 150000, optional)
        // breaks the loop, so sat_4 never runs either.
        assert_eq!(
            plan.keys(),
            vec![
                "features", "mono", "tri", "lda", "sat", "sat_2", "pronprob", "sat_3", "final"
            ]
        );

        assert_eq!(units(&plan, "features"), n);
        // mono: (3 setup + 21 realign + 40 accumulate) * 20 = 64 * 20 = 1280
        assert_eq!(units(&plan, "mono"), 64 * n);
        // tri: (5 + 3 + 35) * 20 = 43 * 20 = 860
        assert_eq!(units(&plan, "tri"), 43 * n);
        // lda: (6 + 3 + 4 + 35) * 20 = 48 * 20 = 960
        assert_eq!(units(&plan, "lda"), 48 * n);
        // sat round (non-quick, I=35): (6 + 3 + 35) * 20 = 44 * 20 = 880
        assert_eq!(units(&plan, "sat"), 44 * n);
        assert_eq!(units(&plan, "sat_2"), 44 * n);
        assert_eq!(units(&plan, "sat_3"), 44 * n);
        // pronprob: 2 * 20 = 40
        assert_eq!(units(&plan, "pronprob"), 2 * n);
        // final with SAT: 3 * 20 = 60
        assert_eq!(units(&plan, "final"), 3 * n);

        // total = (1 + 64 + 43 + 48 + 44*3 + 2 + 3) * 20 = 293 * 20 = 5860
        assert_eq!(plan.total(), 293 * n);
    }

    #[test]
    fn ljspeech_sized_corpus_applies_the_subsets() {
        let cfg = TrainConfig::default();
        let n = 13093usize;
        let plan = work_plan(&cfg, n, true);

        // `optional_rounds_end_the_schedule_on_a_small_corpus`: through sat_3.
        assert_eq!(
            plan.keys(),
            vec![
                "features", "mono", "tri", "lda", "sat", "sat_2", "pronprob", "sat_3", "final"
            ]
        );

        assert_eq!(units(&plan, "features"), 13093);
        // mono subset 10000 < n -> S = 10000; 64 passes = 640_000
        assert_eq!(units(&plan, "mono"), 64 * 10_000);
        // tri subset 20000 >= n -> S = n (subset_size drops it); 43 * 13093
        assert_eq!(units(&plan, "tri"), 43 * 13_093);
        // lda subset 20000 >= n -> S = n; 48 * 13093
        assert_eq!(units(&plan, "lda"), 48 * 13_093);
        // sat subset 20000 >= n -> S = n; 44 * 13093
        assert_eq!(units(&plan, "sat"), 44 * 13_093);
        // sat_2 subset 50000 >= n -> S = n
        assert_eq!(units(&plan, "sat_2"), 44 * 13_093);
        // pronprob subset 50000 >= n -> S = n; 2 * 13093
        assert_eq!(units(&plan, "pronprob"), 2 * 13_093);
        // sat_3 subset 150000 >= n -> S = n
        assert_eq!(units(&plan, "sat_3"), 44 * 13_093);
        assert_eq!(units(&plan, "final"), 3 * 13_093);

        // total = 64*10000 + (1 + 43 + 48 + 44 + 44 + 2 + 44 + 4) * 13093
        //       = 640_000 + 229 * 13_093 = 640_000 + 2_998_297 = 3_638_297
        assert_eq!(plan.total(), 640_000 + 229 * 13_093);
        assert_eq!(plan.total(), 3_638_297);
    }

    #[test]
    fn large_corpus_runs_every_round_with_real_subsets() {
        let cfg = TrainConfig::default();
        let n = 200_000usize;
        let plan = work_plan(&cfg, n, true);

        // Every subset is smaller than the corpus, so nothing is dropped.
        assert_eq!(
            plan.keys(),
            vec![
                "features",
                "mono",
                "tri",
                "lda",
                "sat",
                "sat_2",
                "pronprob",
                "sat_3",
                "pronprob_2",
                "sat_4",
                "final"
            ]
        );

        assert_eq!(units(&plan, "features"), 200_000);
        assert_eq!(units(&plan, "mono"), 64 * 10_000); //   640_000
        assert_eq!(units(&plan, "tri"), 43 * 20_000); //    860_000
        assert_eq!(units(&plan, "lda"), 48 * 20_000); //    960_000
        assert_eq!(units(&plan, "sat"), 44 * 20_000); //    880_000
        assert_eq!(units(&plan, "sat_2"), 44 * 50_000); // 2_200_000
        assert_eq!(units(&plan, "pronprob"), 2 * 50_000); //  100_000
        assert_eq!(units(&plan, "sat_3"), 44 * 150_000); // 6_600_000
        assert_eq!(units(&plan, "pronprob_2"), 2 * 150_000); // 300_000
        // sat_4: subset 0 -> whole corpus, quick, I=20, realign [10,15] = 2:
        // (6 + 2 + 20) * 200_000 = 28 * 200_000 = 5_600_000
        assert_eq!(units(&plan, "sat_4"), 28 * 200_000);
        assert_eq!(units(&plan, "final"), 3 * 200_000); //  600_000

        assert_eq!(
            plan.total(),
            200_000
                + 640_000
                + 860_000
                + 960_000
                + 880_000
                + 2_200_000
                + 100_000
                + 6_600_000
                + 300_000
                + 5_600_000
                + 600_000
        );
        assert_eq!(plan.total(), 18_940_000);
    }

    #[test]
    fn mono_and_tri_only_without_lda_or_sat() {
        let cfg = TrainConfig {
            stages: Stages {
                lda: false,
                sat: false,
                ..Stages::default()
            },
            ..TrainConfig::default()
        };
        let n = 200_000usize;
        let plan = work_plan(&cfg, n, true);

        // Disabled kinds are dropped without ending the run, so both pron-prob rounds
        // survive (pronprob_2 is optional with subset 150000 <= n).
        assert_eq!(
            plan.keys(),
            vec!["features", "mono", "tri", "pronprob", "pronprob_2", "final"]
        );
        // No SAT stage: the final pass is a single alignment, 2n not 3n.
        assert_eq!(units(&plan, "final"), 2 * 200_000);
        assert_eq!(steps(&plan, "final"), vec!["graphs", "final align"]);
        assert_eq!(
            plan.total(),
            200_000                 // features
                + 64 * 10_000       // mono
                + 43 * 20_000       // tri
                + 2 * 50_000        // pronprob
                + 2 * 150_000       // pronprob_2
                + 2 * 200_000 // final
        );
        // 200_000 + 640_000 + 860_000 + 100_000 + 300_000 + 400_000
        assert_eq!(plan.total(), 2_500_000);
    }

    #[test]
    fn final_alignment_flag_adds_exactly_the_final_stage() {
        let cfg = TrainConfig::default();
        let with = work_plan(&cfg, 13_093, true);
        let without = work_plan(&cfg, 13_093, false);
        assert!(!without.keys().contains(&"final"));
        assert_eq!(with.total() - without.total(), 3 * 13_093);
        assert_eq!(&with.stages[..with.stages.len() - 1], &without.stages[..]);
        assert_eq!(
            &with.passes[..with.passes.len() - 3],
            &without.passes[..],
            "the final stage only appends"
        );
    }

    #[test]
    fn subsetting_off_uses_the_whole_corpus_for_every_stage() {
        let cfg = TrainConfig {
            subset: false,
            ..TrainConfig::default()
        };
        let n = 200_000u64;
        let plan = work_plan(&cfg, n as usize, false);
        assert_eq!(units(&plan, "mono"), 64 * n);
        assert_eq!(units(&plan, "sat_2"), 44 * n);
        assert_eq!(units(&plan, "pronprob"), 2 * n);
    }

    #[test]
    fn keys_are_the_schedule_between_features_and_final() {
        let cfg = TrainConfig::default();
        for n in [20usize, 13_093, 200_000] {
            let plan = work_plan(&cfg, n, true);
            let keys = plan.keys();
            assert_eq!(keys[0], "features");
            assert_eq!(*keys.last().unwrap(), "final");
            let schedule: Vec<String> = cfg.effective_schedule(n).iter().map(|s| s.key()).collect();
            assert_eq!(&keys[1..keys.len() - 1], &schedule[..]);
            assert!(plan.stages.iter().all(|s| s.units > 0));
        }
    }

    #[test]
    fn passes_aggregate_to_the_stages() {
        let cfg = TrainConfig::default();
        for n in [20usize, 13_093, 200_000] {
            let plan = work_plan(&cfg, n, true);
            // Passes are grouped by stage, in the same order as `stages`.
            let mut order: Vec<&str> = Vec::new();
            for p in &plan.passes {
                if order.last() != Some(&p.stage.as_str()) {
                    order.push(&p.stage);
                }
            }
            assert_eq!(order, plan.keys());
            // And they sum to the same units.
            for st in &plan.stages {
                let sum: u64 = plan
                    .passes
                    .iter()
                    .filter(|p| p.stage == st.key)
                    .map(|p| p.units)
                    .sum();
                assert_eq!(sum, st.units, "stage {}", st.key);
            }
            let total: u64 = plan.passes.iter().map(|p| p.units).sum();
            assert_eq!(total, plan.total());
            assert!(plan.passes.iter().all(|p| p.units > 0));
        }
    }

    #[test]
    fn the_pass_sequence_starts_with_features_and_mono() {
        let cfg = TrainConfig::default();
        let plan = work_plan(&cfg, 20, true);
        let head: Vec<(&str, &str, u64)> = plan.passes[..12]
            .iter()
            .map(|p| (p.stage.as_str(), p.step, p.units))
            .collect();
        assert_eq!(
            head,
            vec![
                ("features", "mfcc", 20),
                ("mono", "graphs", 20),
                ("mono", "equal align", 20),
                ("mono", "accumulate", 20), // iteration 0
                ("mono", "align", 20),      // iteration 1 realigns
                ("mono", "accumulate", 20),
                ("mono", "align", 20), // iteration 2
                ("mono", "accumulate", 20),
                ("mono", "align", 20), // iteration 3
                ("mono", "accumulate", 20),
                ("mono", "align", 20), // iteration 4
                ("mono", "accumulate", 20),
            ]
        );
    }

    #[test]
    fn tree_stages_open_with_the_previous_model_and_sat_ends_with_two_feats() {
        let cfg = TrainConfig::default();
        let plan = work_plan(&cfg, 20, true);

        assert_eq!(
            &steps(&plan, "tri")[..5],
            &[
                "graphs",
                "align (previous model)",
                "tree stats",
                "convert alignments",
                "graphs"
            ]
        );
        assert_eq!(
            &steps(&plan, "lda")[..6],
            &[
                "graphs",
                "align (previous model)",
                "lda stats",
                "tree stats",
                "convert alignments",
                "graphs"
            ]
        );
        for key in ["sat", "sat_2", "sat_3"] {
            let s = steps(&plan, key);
            assert_eq!(*s.last().unwrap(), "two-feats stats", "{key}");
            assert_eq!(s[s.len() - 2], "accumulate", "{key}");
        }
        assert_eq!(steps(&plan, "pronprob"), vec!["graphs", "align"]);
        assert_eq!(
            steps(&plan, "final"),
            vec!["graphs", "final align", "final align (fmllr)"]
        );
    }

    #[test]
    fn lda_mllt_passes_land_on_the_hook_iterations() {
        let cfg = TrainConfig::default();
        let plan = work_plan(&cfg, 20, true);
        let mllt = plan
            .passes
            .iter()
            .filter(|p| p.stage == "lda" && p.step == "mllt")
            .count();
        // mllt_iterations [2, 4, 6, 12], all within 1..=35.
        assert_eq!(mllt, 4);
    }

    #[test]
    fn gaussian_counts_grow_within_a_stage_and_carry_across() {
        let cfg = TrainConfig::default();
        let plan = work_plan(&cfg, 20, true);

        // Features and mono's flat start run against no model.
        assert_eq!(plan.passes[0].gauss, 0);
        assert_eq!(plan.passes[1].gauss, 0); // mono graphs
        assert_eq!(plan.passes[2].gauss, 0); // equal align

        // Mono iteration 0 targets the configured initial count, then grows.
        assert_eq!(plan.passes[3].gauss, cfg.mono.initial_gaussians as u64);
        let mono: Vec<u64> = plan
            .passes
            .iter()
            .filter(|p| p.stage == "mono")
            .map(|p| p.gauss)
            .collect();
        assert!(mono.windows(2).all(|w| w[0] <= w[1]), "monotone: {mono:?}");
        // Integer truncation of the increment overshoots the configured max a little
        // (135 + 31 * ((1000 - 135) / 30) = 1003); that is what the run does too.
        assert_eq!(*mono.last().unwrap(), 1003);

        // Tri's setup passes run against mono's final model.
        let tri: Vec<u64> = plan
            .passes
            .iter()
            .filter(|p| p.stage == "tri")
            .map(|p| p.gauss)
            .collect();
        assert_eq!(tri[0], 1003);
        assert_eq!(tri[4], 1003); // the last of the five setup passes
        // The spec, not the config, sets this round's tree size (2000 -> 10000).
        assert_eq!(tri[5], 2000); // iteration 1, against the new tree
        assert_eq!(*tri.last().unwrap(), 10_000);

        // The last SAT round's max carries into the final alignment.
        let sat3_max = plan
            .passes
            .iter()
            .filter(|p| p.stage == "sat_3")
            .map(|p| p.gauss)
            .max()
            .unwrap();
        assert!(sat3_max >= 40_000, "sat_3 grew to {sat3_max}");
        assert!(plan.passes.iter().all(|p| p.gauss <= sat3_max));
        for p in plan.passes.iter().filter(|p| p.stage == "final") {
            assert_eq!(p.gauss, sat3_max);
        }
    }
}
