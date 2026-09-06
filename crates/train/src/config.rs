//! Training hyperparameters, copied from MFA.
//!
//! Every default here cites the python file and line it came from. Sources:
//! `plans/mfa/montreal_forced_aligner/acoustic_modeling/{base,monophone,triphone,lda,sat}.py`,
//! `plans/mfa/montreal_forced_aligner/alignment/mixins.py`,
//! `plans/mfa/montreal_forced_aligner/corpus/features.py`,
//! `plans/mfa/montreal_forced_aligner/dictionary/mixins.py`.

use viter_kaldi::align::AlignOptions;
use viter_kaldi::feat::{DeltaOptions, MfccOptions};
use viter_kaldi::hmm::GraphOptions;
use viter_kaldi::transform::{FmllrOptions, FmllrUpdateType};
use serde::{Deserialize, Serialize};

/// Which stages to run. Monophone is always run (everything else bootstraps from it).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Stages {
    pub tri: bool,
    pub lda: bool,
    pub sat: bool,
}

impl Default for Stages {
    /// MFA's default pipeline is mono -> tri -> lda -> sat
    /// (`acoustic_modeling/trainer.py:194-213`).
    fn default() -> Self {
        Self { tri: true, lda: true, sat: true }
    }
}

/// Monophone stage. `acoustic_modeling/monophone.py:166-186`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonoConfig {
    /// `base.py:91` num_iterations = 40.
    pub num_iterations: usize,
    /// `monophone.py:169` initial_gaussians = 135. Note MFA overwrites this with the
    /// gaussian count that `gmm_init_mono` actually produced (`monophone.py:351-352`),
    /// which is one gaussian per pdf; the configured value is only a floor for the
    /// first mixup target used in the iteration-0 update (`monophone.py:294`).
    pub initial_gaussians: usize,
    /// `monophone.py:171` max_gaussians = 1000.
    pub max_gaussians: usize,
    /// `monophone.py:172` power = 0.25.
    pub power: f32,
    /// `monophone.py:173` boost_silence = 1.25.
    pub boost_silence: f32,
    /// `monophone.py:170` initial_beam = 6, used only on iteration 1
    /// (`monophone.py:236-238`).
    pub initial_beam: f32,
    /// `monophone.py:293` min_gaussian_occupancy = 3.0, applied only to the
    /// iteration-0 update after the equal-align flat start (kalpy default is 10.0).
    pub min_gaussian_occupancy: f64,
    /// `monophone.py:168` subset = 2000 utterances.
    pub subset: usize,
}

impl Default for MonoConfig {
    fn default() -> Self {
        Self {
            num_iterations: 40,
            initial_gaussians: 135,
            max_gaussians: 1000,
            power: 0.25,
            boost_silence: 1.25,
            initial_beam: 6.0,
            min_gaussian_occupancy: 3.0,
            subset: 2000,
        }
    }
}

impl MonoConfig {
    /// `monophone.py:208-220 compute_calculated_properties`. Realign on iteration 0,
    /// then every iteration up to num_iterations/4, then with a gap > 1 up to
    /// num_iterations/2, then with a gap > 2 for the rest.
    ///
    /// For num_iterations = 40 this yields
    /// `[0,1,..,10,12,14,16,18,20,23,26,29,32,35,38]`.
    pub fn realignment_iterations(&self) -> Vec<usize> {
        let n = self.num_iterations;
        let mut out = vec![0usize];
        for i in 1..n {
            let last = *out.last().unwrap();
            if i <= n / 4 {
                out.push(i);
            } else if i <= n * 2 / 4 {
                if i - last > 1 {
                    out.push(i);
                }
            } else if i - last > 2 {
                out.push(i);
            }
        }
        out
    }

    /// `monophone.py:210` final_gaussian_iteration = num_iterations - 10.
    pub fn final_gaussian_iteration(&self) -> usize {
        self.num_iterations.saturating_sub(10)
    }
}

/// Triphone stage. `acoustic_modeling/triphone.py:215-236`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriConfig {
    /// `triphone.py:218` num_iterations = 35.
    pub num_iterations: usize,
    /// `triphone.py:219` num_leaves = 1000. Also the initial gaussian count
    /// (`triphone.py:226`, `:325`).
    pub num_leaves: usize,
    /// `triphone.py:220` max_gaussians = 10000.
    pub max_gaussians: usize,
    /// `triphone.py:221` cluster_threshold = -1 (Kaldi reads this as "use thresh").
    pub cluster_threshold: f64,
    /// `triphone.py:222` boost_silence = 1.25.
    pub boost_silence: f32,
    /// `triphone.py:223` power = 0.25.
    pub power: f32,
    /// `triphone.py:217` subset = 5000 utterances.
    pub subset: usize,
    /// Tree split threshold. Kaldi `build-tree.cc` / kalpy `tree.cpp:1169` default
    /// thresh = 300.0 (see `plans/research/01_kaldi_surface.md` section 3).
    pub tree_thresh: f64,
    /// Tree stats variance floor, kalpy `AccumulateTreeStatsOptions` default 0.01.
    pub tree_var_floor: f64,
}

impl Default for TriConfig {
    fn default() -> Self {
        Self {
            num_iterations: 35,
            num_leaves: 1000,
            max_gaussians: 10000,
            cluster_threshold: -1.0,
            boost_silence: 1.25,
            power: 0.25,
            subset: 5000,
            tree_thresh: 300.0,
            tree_var_floor: 0.01,
        }
    }
}

impl TriConfig {
    /// `triphone.py:319-326`: realign every 10 iterations, skipping 0.
    /// For 35 iterations: `[10, 20, 30]`.
    pub fn realignment_iterations(&self) -> Vec<usize> {
        (0..self.num_iterations).step_by(10).filter(|&i| i != 0).collect()
    }

    /// `triphone.py:326` final_gaussian_iteration = num_iterations - 10.
    pub fn final_gaussian_iteration(&self) -> usize {
        self.num_iterations.saturating_sub(10)
    }
}

/// LDA+MLLT stage. `acoustic_modeling/lda.py:226-252`. Inherits the triphone
/// realignment schedule (`LdaTrainer(TriphoneTrainer)`, `lda.py:191`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LdaConfig {
    /// Inherited from `triphone.py:218` num_iterations = 35.
    pub num_iterations: usize,
    /// `lda.py:229` num_leaves = 2500.
    pub num_leaves: usize,
    /// `lda.py:230` max_gaussians = 15000.
    pub max_gaussians: usize,
    /// `lda.py:231` lda_dimension = 40.
    pub lda_dimension: usize,
    /// `lda.py:233` splice_left_context = 3.
    pub splice_left: usize,
    /// `lda.py:234` splice_right_context = 3.
    pub splice_right: usize,
    /// `lda.py:235` random_prune = 4.0.
    pub random_prune: f64,
    /// `lda.py:236` boost_silence = 1.0.
    pub boost_silence: f32,
    /// `lda.py:237` power = 0.25.
    pub power: f32,
    /// `lda.py:228` subset = 10000 utterances.
    pub subset: usize,
    /// `lda.py:320` mllt_iterations = [2, 4, 6, 12].
    pub mllt_iterations: Vec<usize>,
    /// Tree split threshold, as triphone.
    pub tree_thresh: f64,
    /// `triphone.py:221` cluster_threshold = -1, inherited.
    pub cluster_threshold: f64,
    /// Tree stats variance floor.
    pub tree_var_floor: f64,
}

impl Default for LdaConfig {
    fn default() -> Self {
        Self {
            num_iterations: 35,
            num_leaves: 2500,
            max_gaussians: 15000,
            lda_dimension: 40,
            splice_left: 3,
            splice_right: 3,
            random_prune: 4.0,
            boost_silence: 1.0,
            power: 0.25,
            subset: 10000,
            mllt_iterations: vec![2, 4, 6, 12],
            tree_thresh: 300.0,
            cluster_threshold: -1.0,
            tree_var_floor: 0.01,
        }
    }
}

impl LdaConfig {
    /// Inherited from `triphone.py:319-324`: every 10 iterations, skipping 0.
    pub fn realignment_iterations(&self) -> Vec<usize> {
        (0..self.num_iterations).step_by(10).filter(|&i| i != 0).collect()
    }
    /// Inherited `triphone.py:326`.
    pub fn final_gaussian_iteration(&self) -> usize {
        self.num_iterations.saturating_sub(10)
    }
    /// Dimension of spliced features feeding the LDA estimate.
    pub fn spliced_dim(&self, base_dim: usize) -> usize {
        base_dim * (self.splice_left + self.splice_right + 1)
    }
}

/// SAT (fMLLR) stage. `acoustic_modeling/sat.py:165-226`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SatConfig {
    /// Inherited from `triphone.py:218` num_iterations = 35.
    pub num_iterations: usize,
    /// `sat.py:168` num_leaves = 2500.
    pub num_leaves: usize,
    /// `sat.py:169` max_gaussians = 15000.
    pub max_gaussians: usize,
    /// `sat.py:170` power = 0.2.
    pub power: f32,
    /// `sat.py:171` boost_silence = 1.0.
    pub boost_silence: f32,
    /// `sat.py:167` subset = 10000 utterances.
    pub subset: usize,
    /// `sat.py:219` fmllr_iterations = [2, 4, 6, 12] (non-quick path).
    pub fmllr_iterations: Vec<usize>,
    /// `sat.py:172` quick = false. When true, `sat.py:221-226` overrides the
    /// realignment schedule to [10, 15], fmllr to [2, 6, 12], final gaussian
    /// iteration to num_iterations - 5 and initial gaussians to max/2.
    pub quick: bool,
    /// fMLLR estimation options: `features.py:635` silence_weight = 0.0,
    /// `features.py:634` fmllr_update_type = "full"; min_count 500.0 and
    /// num_iters 40 are the kalpy `FmllrOptions` defaults
    /// (`plans/kalpy/kalpy/feat/fmllr.py:62-71`,
    /// `plans/research/01_kaldi_surface.md` section 5).
    pub fmllr: FmllrOptions,
    /// `features.py:635` silence_weight = 0.0: silence posteriors are zeroed when
    /// accumulating fMLLR stats.
    pub silence_weight: f32,
    /// Tree split threshold, as triphone.
    pub tree_thresh: f64,
    /// Inherited `triphone.py:221` cluster_threshold = -1.
    pub cluster_threshold: f64,
    /// Tree stats variance floor.
    pub tree_var_floor: f64,
}

impl Default for SatConfig {
    fn default() -> Self {
        Self {
            num_iterations: 35,
            num_leaves: 2500,
            max_gaussians: 15000,
            power: 0.2,
            boost_silence: 1.0,
            subset: 10000,
            fmllr_iterations: vec![2, 4, 6, 12],
            quick: false,
            fmllr: FmllrOptions {
                update_type: FmllrUpdateType::Full,
                min_count: 500.0,
                num_iters: 40,
            },
            silence_weight: 0.0,
            tree_thresh: 300.0,
            cluster_threshold: -1.0,
            tree_var_floor: 0.01,
        }
    }
}

impl SatConfig {
    /// `sat.py:214-226`. Non-quick inherits triphone's every-10 schedule;
    /// quick uses [10, 15].
    pub fn realignment_iterations(&self) -> Vec<usize> {
        if self.quick {
            vec![10, 15]
        } else {
            (0..self.num_iterations).step_by(10).filter(|&i| i != 0).collect()
        }
    }

    /// `sat.py:218-222`.
    pub fn fmllr_iterations(&self) -> Vec<usize> {
        if self.quick { vec![2, 6, 12] } else { self.fmllr_iterations.clone() }
    }

    /// `sat.py:223` (quick) vs inherited `triphone.py:326`.
    pub fn final_gaussian_iteration(&self) -> usize {
        if self.quick {
            self.num_iterations.saturating_sub(5)
        } else {
            self.num_iterations.saturating_sub(10)
        }
    }

    /// `sat.py:224-226` (quick) vs `triphone.py:325` initial_gaussians = num_leaves.
    pub fn initial_gaussians(&self) -> usize {
        if self.quick {
            (self.max_gaussians / 2).max(self.num_leaves)
        } else {
            self.num_leaves
        }
    }
}

/// Everything needed to run a training job.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Seed for all randomness (gaussian split perturbation, equal-align paths,
    /// MLLT random pruning, subset selection). Deterministic given this value.
    pub seed: u64,
    pub mfcc: MfccOptions,
    pub deltas: DeltaOptions,
    /// `dictionary/mixins.py` num_silence_states = 5.
    pub num_sil_states: usize,
    /// `dictionary/mixins.py` num_non_silence_states = 3.
    pub num_nonsil_states: usize,
    /// `dictionary/mixins.py:88,97` position_dependent_phones = True.
    pub position_dependent: bool,
    pub graph: GraphOptions,
    pub align: AlignOptions,
    pub mono: MonoConfig,
    pub tri: TriConfig,
    pub lda: LdaConfig,
    pub sat: SatConfig,
    pub stages: Stages,
    /// Train each stage on MFA's per-stage utterance subset (`base.py:190-194`),
    /// then do a final full-corpus alignment. False = every stage sees everything.
    pub subset: bool,
    /// Utterances scored per `Device::score_batch` call.
    pub batch_utts: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            mfcc: MfccOptions::default(),
            deltas: DeltaOptions::default(),
            num_sil_states: 5,
            num_nonsil_states: 3,
            position_dependent: true,
            graph: GraphOptions::default(),
            // `alignment/mixins.py:69-76`: acoustic_scale 0.1, beam 10, retry_beam 40.
            align: AlignOptions::default(),
            mono: MonoConfig::default(),
            tri: TriConfig::default(),
            lda: LdaConfig::default(),
            sat: SatConfig::default(),
            stages: Stages::default(),
            subset: true,
            batch_utts: 64,
        }
    }
}

impl TrainConfig {
    /// Per-stage subset size, or 0 for "use everything". `base.py:190-194`,
    /// `base.py:210-216`: a subset at least as large as the corpus is dropped.
    pub fn subset_for(&self, stage: Stage, num_utts: usize) -> usize {
        if !self.subset {
            return 0;
        }
        let want = match stage {
            Stage::Mono => self.mono.subset,
            Stage::Tri => self.tri.subset,
            Stage::Lda => self.lda.subset,
            Stage::Sat => self.sat.subset,
        };
        if want == 0 || want >= num_utts { 0 } else { want }
    }
}

/// Identifies a training stage, for subset lookup and progress labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Mono,
    Tri,
    Lda,
    Sat,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Mono => "mono",
            Stage::Tri => "tri",
            Stage::Lda => "lda",
            Stage::Sat => "sat",
        }
    }
}

/// MFA's gaussian growth: `current_gaussians` starts at `initial_gaussians` and
/// grows by `gaussian_increment` after every iteration up to and including
/// `final_gaussian_iteration` (`base.py:277-279`, `base.py:367-381`,
/// `base.py:461-464`).
#[derive(Clone, Copy, Debug)]
pub struct GaussianSchedule {
    current: usize,
    increment: usize,
    final_iteration: usize,
}

impl GaussianSchedule {
    pub fn new(initial_gaussians: usize, max_gaussians: usize, final_iteration: usize) -> Self {
        // `base.py:464`: int((max - initial) / final_gaussian_iteration).
        let increment = if final_iteration == 0 {
            0
        } else {
            max_gaussians.saturating_sub(initial_gaussians) / final_iteration
        };
        Self { current: initial_gaussians, increment, final_iteration }
    }

    /// Mixup target for the update at this iteration.
    pub fn current(&self) -> usize {
        self.current
    }

    /// Apply the post-update increment for `iteration` (`base.py:379-380`).
    pub fn step(&mut self, iteration: usize) {
        if iteration <= self.final_iteration {
            self.current += self.increment;
        }
    }

    pub fn increment(&self) -> usize {
        self.increment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_realignment_schedule_matches_mfa() {
        let cfg = MonoConfig::default();
        // Reproduces monophone.py:211-220 for num_iterations = 40.
        let expected: Vec<usize> = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38,
        ];
        assert_eq!(cfg.realignment_iterations(), expected);
        assert_eq!(cfg.final_gaussian_iteration(), 30);
    }

    #[test]
    fn tri_realignment_schedule_matches_mfa() {
        let cfg = TriConfig::default();
        assert_eq!(cfg.realignment_iterations(), vec![10, 20, 30]);
        assert_eq!(cfg.final_gaussian_iteration(), 25);
    }

    #[test]
    fn sat_schedules() {
        let cfg = SatConfig::default();
        assert_eq!(cfg.realignment_iterations(), vec![10, 20, 30]);
        assert_eq!(cfg.fmllr_iterations(), vec![2, 4, 6, 12]);
        assert_eq!(cfg.initial_gaussians(), 2500);

        let quick = SatConfig { quick: true, ..SatConfig::default() };
        assert_eq!(quick.realignment_iterations(), vec![10, 15]);
        assert_eq!(quick.fmllr_iterations(), vec![2, 6, 12]);
        assert_eq!(quick.final_gaussian_iteration(), 30);
        assert_eq!(quick.initial_gaussians(), 7500);
    }

    #[test]
    fn gaussian_schedule_reaches_max_at_final_iteration() {
        // mono: initial 135, max 1000, final iteration 30.
        let mut s = GaussianSchedule::new(135, 1000, 30);
        assert_eq!(s.increment(), (1000 - 135) / 30);
        assert_eq!(s.current(), 135);
        for i in 1..=30 {
            s.step(i);
        }
        // 135 + 30*28 = 975, just under max, exactly as MFA's integer division gives.
        assert_eq!(s.current(), 135 + 30 * ((1000 - 135) / 30));
        assert!(s.current() <= 1000);
        // Past the final iteration the count freezes.
        s.step(31);
        assert_eq!(s.current(), 135 + 30 * ((1000 - 135) / 30));
    }

    #[test]
    fn subset_larger_than_corpus_is_dropped() {
        let cfg = TrainConfig::default();
        assert_eq!(cfg.subset_for(Stage::Mono, 500), 0);
        assert_eq!(cfg.subset_for(Stage::Mono, 50_000), 2000);
        let no = TrainConfig { subset: false, ..TrainConfig::default() };
        assert_eq!(no.subset_for(Stage::Tri, 50_000), 0);
    }
}
