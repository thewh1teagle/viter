//! The progress total is the real number of utterance-passes a run performs.
//!
//! `plan::work_plan` enumerates every per-utterance pass before training starts; the
//! stage code increments the same counter once per utterance per pass. This trains the
//! small `tests/data/lj20` fixture end to end and checks that the two meet exactly:
//! the counter never goes backwards and the last event has `done == total ==
//! work_plan(..).total()`. A mismatch means a pass was added, removed or skipped in one
//! place and not the other.
//!
//! The same run also checks the CONTRACT2 time model: the bar's `fraction` never goes
//! backwards, ends at exactly 1.0, and the plan's pass cursor saw no mismatched bar
//! labels (`ProgressEvent::mismatches`).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use viter_io::corpus::{CorpusOptions, SpeakerSource};
use viter_kaldi::device::Device;
use viter_train::config::TrainConfig;
use viter_train::pipeline::{ProgressEvent, ProgressSink, TrainOptions, train_with, work_plan};

fn repo_root() -> PathBuf {
    // crates/train/ -> crates/ -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate lives at <repo>/crates/train")
        .to_path_buf()
}

/// The default recipe is far too slow for CPU CI (35 SAT iterations over two rounds),
/// so the iteration counts are cut down. Every stage kind the plan knows about still
/// runs — mono, tri, lda (with an MLLT iteration), one SAT round (with an fMLLR
/// iteration) and a pron-prob round — plus the final two-pass alignment, so every
/// branch of `work_plan` is exercised.
fn fast_config() -> TrainConfig {
    use viter_train::config::StageSpec;
    let mut cfg = TrainConfig {
        subset: false,
        ..TrainConfig::default()
    };
    cfg.mono.num_iterations = 3;
    cfg.tri.num_iterations = 3;
    cfg.lda.num_iterations = 3;
    cfg.sat.num_iterations = 3;
    cfg.schedule = vec![
        StageSpec::Mono { subset: 0 },
        StageSpec::Tri {
            subset: 0,
            num_leaves: 100,
            max_gaussians: 500,
        },
        StageSpec::Lda {
            subset: 0,
            num_leaves: 100,
            max_gaussians: 500,
        },
        StageSpec::Sat {
            round: 1,
            subset: 0,
            num_leaves: 100,
            max_gaussians: 500,
            num_iterations: 3,
            quick: false,
            optional: false,
        },
        StageSpec::PronProbs {
            round: 1,
            subset: 0,
            optional: false,
        },
    ];
    cfg
}

#[test]
fn progress_reaches_exactly_the_planned_total() {
    let root = repo_root();
    let corpus_dir = root.join("tests/data/lj20");
    let dict = root.join("tests/data/cmudict.dict");
    if !corpus_dir.is_dir() || !dict.is_file() {
        eprintln!("fixture {} missing; skipping", corpus_dir.display());
        return;
    }

    let opts = CorpusOptions {
        dictionary: Some(dict),
        position_dependent: true,
        speaker_from: SpeakerSource::ParentDir,
        ..CorpusOptions::default()
    };
    let corpus = viter_io::corpus::scan(&corpus_dir, &opts).expect("scanning the lj20 fixture");
    assert!(!corpus.utts.is_empty(), "fixture corpus is empty");

    let cfg = fast_config();
    let expected = work_plan(&cfg, corpus.utts.len(), true);
    assert!(expected.total() > 0, "the plan must count some work");

    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = events.clone();
    let sink: ProgressSink = Arc::new(move |e: &ProgressEvent| {
        seen.lock().expect("sink mutex").push(e.clone());
    });

    let started = std::time::Instant::now();
    let trained = train_with(
        &corpus,
        &cfg,
        &Device::cpu(),
        None,
        &TrainOptions {
            final_alignment: true,
            progress: Some(sink),
            quiet: true,
        },
    )
    .expect("training the lj20 fixture");
    eprintln!(
        "lj20: {} utterance-passes in {:.1}s",
        expected.total(),
        started.elapsed().as_secs_f64()
    );
    assert!(
        !trained.alignments.is_empty(),
        "final alignment produced nothing"
    );

    let events = events.lock().expect("sink mutex");
    assert!(!events.is_empty(), "the sink was never called");

    let mut prev = 0u64;
    let mut prev_fraction = 0.0f64;
    for e in events.iter() {
        assert!(
            e.done >= prev,
            "progress went backwards: {} after {prev}",
            e.done
        );
        assert_eq!(e.total, expected.total(), "the total changed mid-run");
        assert!(
            e.done <= e.total,
            "progress overshot: {} > {}",
            e.done,
            e.total
        );
        assert!(
            e.fraction >= prev_fraction - 1e-9,
            "the time fraction went backwards: {} after {prev_fraction}",
            e.fraction
        );
        assert!(
            (0.0..=1.0).contains(&e.fraction),
            "the time fraction left [0, 1]: {}",
            e.fraction
        );
        assert_eq!(
            e.mismatches, 0,
            "the stage code opened a bar the plan did not expect              (stage {:?}, step {:?}); plan.rs and the stage labels disagree",
            e.stage, e.step
        );
        prev = e.done;
        prev_fraction = e.fraction;
    }

    let last = events.last().expect("checked non-empty");
    assert_eq!(
        last.done, last.total,
        "the run ended at {}/{} utterance-passes; the plan and the stage loops disagree",
        last.done, last.total
    );
    assert_eq!(
        last.fraction, 1.0,
        "the bar must end at exactly 1.0, not {}",
        last.fraction
    );
    assert_eq!(last.mismatches, 0, "the run reported plan mismatches");
}
