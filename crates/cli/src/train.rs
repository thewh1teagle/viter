//! `viter train` — scan a corpus, run the GMM training pipeline, save the model.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Args;
use owo_colors::OwoColorize;
use viter_io::textgrid;
use viter_train::config::TrainConfig;

use super::{
    CorpusArgs, ensure_parent, field, fmt_duration, fmt_rtf, header, output_path, success, warn,
};

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Corpus directory: audio files with matching .txt/.lab transcripts, scanned recursively
    #[arg(value_name = "CORPUS_DIR")]
    pub corpus: PathBuf,

    /// Where to write the trained model
    #[arg(short, long, value_name = "MODEL.viter", default_value = "model.viter")]
    pub out: PathBuf,

    /// Also write TextGrids of the final training alignments, mirroring the corpus structure
    #[arg(long, value_name = "DIR")]
    pub out_textgrids: Option<PathBuf>,

    /// Directory for per-stage intermediate models and the training log
    #[arg(long, value_name = "DIR")]
    pub work_dir: Option<PathBuf>,

    #[command(flatten)]
    pub corpus_args: CorpusArgs,

    /// Skip the LDA+MLLT stage
    #[arg(long)]
    pub no_lda: bool,

    /// Skip the speaker-adaptive (fMLLR) stage
    #[arg(long)]
    pub no_sat: bool,

    /// Skip the triphone stage (implies --no-lda --no-sat)
    #[arg(long)]
    pub no_tri: bool,

    /// Number of SAT/fMLLR rounds (MFA's default schedule has 4)
    #[arg(long, value_name = "N")]
    pub sat_rounds: Option<usize>,

    /// Skip the pronunciation-probability estimation rounds
    #[arg(long)]
    pub no_pron_probs: bool,

    /// Train on the full corpus at every stage instead of MFA's per-stage subsets
    #[arg(long)]
    pub no_subset: bool,

    /// Random seed; training is deterministic given the same seed and corpus
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub seed: u64,

    /// Force CPU scoring even when a GPU is available
    #[arg(long)]
    pub cpu: bool,

    /// Append a plain-text copy of the progress output (every iteration) to this file
    #[arg(long, value_name = "FILE")]
    pub log: Option<std::path::PathBuf>,
}

pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    if let Some(log) = &args.log {
        viter_train::pipeline::progress::set_log_file(log)
            .map_err(|e| anyhow::anyhow!("cannot open log file {}: {e}", log.display()))?;
    }
    let started = Instant::now();

    // --- corpus -----------------------------------------------------------
    header("Corpus");
    let opts = args.corpus_args.to_options();
    let corpus = viter_io::corpus::scan(&args.corpus, &opts)
        .with_context(|| format!("failed to scan corpus at {}", args.corpus.display()))?;
    anyhow::ensure!(
        !corpus.utts.is_empty(),
        "no utterances found in {} (expected audio files with matching .txt or .lab transcripts)",
        args.corpus.display()
    );
    print_corpus_summary(&corpus);

    // --- config -----------------------------------------------------------
    // CONTRACT-DEVIATION: plans/CONTRACTS.md describes `TrainConfig::stages` only as "which of
    // tri/lda/sat to run", without naming its fields. This assumes `Stages { tri, lda, sat }`
    // with `bool` fields, which is the minimal shape satisfying that comment.
    let mut cfg = TrainConfig::default();
    cfg.seed = args.seed;
    cfg.subset = !args.no_subset;
    // The corpus was built with (or without) word-position tags; the model must record
    // the same choice so `align` tags transcripts identically.
    cfg.position_dependent = !args.corpus_args.no_position_dependent;
    cfg.graph.position_dependent = cfg.position_dependent;
    if args.no_tri {
        cfg.stages.tri = false;
    }
    if args.no_tri || args.no_lda {
        cfg.stages.lda = false;
    }
    if args.no_tri || args.no_sat {
        // SAT needs a triphone model; without LDA it trains on delta features,
        // which is what MFA 3.x exports by default (`uses_splices: false`).
        cfg.stages.sat = false;
    }
    if args.no_pron_probs {
        cfg.stages.pron_probs = false;
    }
    if let Some(rounds) = args.sat_rounds {
        // Keep the first `rounds` SAT entries of the schedule; drop the rest, and any
        // pron-prob round that would then have no SAT stage left to follow it.
        let mut seen = 0usize;
        cfg.schedule.retain(|s| match s {
            viter_train::config::StageSpec::Sat { .. } => {
                seen += 1;
                seen <= rounds
            }
            viter_train::config::StageSpec::PronProbs { .. } => seen < rounds,
            _ => true,
        });
    }

    let device = super::device(args.cpu);

    header("Training");
    field("stages", stage_list(&cfg, corpus.utts.len()));
    field(
        "device",
        match device.adapter_name() {
            Some(n) => format!("{:?} · {n}", device.kind()),
            None => format!("{:?}", device.kind()),
        },
    );
    field("seed", cfg.seed);

    // Measuring total audio up front doubles as a decode check of every file.
    let audio_seconds = super::corpus_audio_seconds(&corpus.utts);

    // --- train ------------------------------------------------------------
    let trained = viter_train::pipeline::train_with(
        &corpus,
        &cfg,
        &device,
        args.work_dir.as_deref(),
        &viter_train::pipeline::TrainOptions {
            final_alignment: args.out_textgrids.is_some(),
            ..Default::default()
        },
    )
    .context("training failed")?;

    // --- save -------------------------------------------------------------
    ensure_parent(&args.out)?;
    trained
        .model
        .save(&args.out)
        .with_context(|| format!("failed to write model to {}", args.out.display()))?;

    // --- optional TextGrids ----------------------------------------------
    let mut written = 0usize;
    if let Some(dir) = &args.out_textgrids {
        written = write_training_textgrids(&trained, &corpus, dir)?;
    }

    // --- summary ----------------------------------------------------------
    let elapsed = started.elapsed();
    header("Summary");
    field("utterances", corpus.utts.len());
    if args.out_textgrids.is_some() {
        let aligned = trained.alignments.len();
        let failed = corpus.utts.len().saturating_sub(aligned);
        field("aligned", aligned);
        if failed > 0 {
            field("failed", failed.to_string().yellow().to_string());
        } else {
            field("failed", 0);
        }
    }
    field(
        "audio",
        fmt_duration(std::time::Duration::from_secs_f64(audio_seconds)),
    );
    field("elapsed", fmt_duration(elapsed));
    field("RTF", fmt_rtf(elapsed, audio_seconds));
    if let Some(rss) = super::peak_rss() {
        field("peak RSS", rss);
    }
    if let Some(dir) = &args.out_textgrids {
        field("TextGrids", format!("{written} -> {}", dir.display()));
    } else {
        field(
            "TextGrids",
            "run `viter align` to align the training corpus",
        );
    }
    success(&format!("model written to {}", args.out.display().bold()));
    Ok(())
}

/// Which stages the config will run, as a `mono -> tri -> lda -> sat ...` chain.
fn stage_list(cfg: &TrainConfig, num_utts: usize) -> String {
    cfg.effective_schedule(num_utts)
        .iter()
        .map(|s| s.key())
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// Print utterance/speaker/phone counts and the worst OOV offenders.
fn print_corpus_summary(corpus: &viter_io::corpus::Corpus) {
    field("utterances", corpus.utts.len());
    field("speakers", corpus.speakers.len());
    // The symbol table includes <eps> at id 0, which is not a real phone.
    field("phones", corpus.phones.len().saturating_sub(1));
    field("silence phones", corpus.silence_phones.len());
    super::report_pron_stats(&corpus.utts);

    let oov_types = corpus.oov_words.len();
    if oov_types == 0 {
        field("OOV words", "none".green().to_string());
        return;
    }
    let oov_tokens: usize = corpus.oov_words.values().sum();
    field(
        "OOV words",
        format!("{oov_types} types / {oov_tokens} tokens")
            .yellow()
            .to_string(),
    );

    let mut top: Vec<(&String, &usize)> = corpus.oov_words.iter().collect();
    // Most frequent first; ties broken alphabetically so the output is deterministic.
    top.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (word, count) in top.into_iter().take(10) {
        println!("      {:<24} {}", word.yellow(), count.to_string().dimmed());
    }
    if oov_types > 10 {
        println!(
            "      {}",
            format!("... and {} more", oov_types - 10).dimmed()
        );
    }
}

/// Write the final training alignments as TextGrids under `dir`, mirroring the corpus layout.
fn write_training_textgrids(
    trained: &viter_train::pipeline::Trained,
    corpus: &viter_io::corpus::Corpus,
    dir: &std::path::Path,
) -> anyhow::Result<usize> {
    use std::collections::HashMap;

    header("TextGrids");
    let by_id: HashMap<&str, &viter_kaldi::types::Utterance> =
        corpus.utts.iter().map(|u| (u.id.as_str(), u)).collect();

    let mut written = 0usize;
    for ali in &trained.alignments {
        let Some(utt) = by_id.get(ali.utt.as_str()) else {
            warn(&format!("alignment for unknown utterance {}", ali.utt));
            continue;
        };
        // Frame-level tids become phone/word intervals against the trained transition model.
        let intervals = viter_kaldi::hmm::to_intervals(
            &trained.model.tm,
            ali,
            &utt.prons,
            trained.model.mfcc.frame_shift_ms / 1000.0,
        );
        let duration = match viter_kaldi::audio::read(&utt.audio) {
            Ok(a) if a.sample_rate > 0 => a.samples.len() as f64 / a.sample_rate as f64,
            _ => intervals.duration_s() as f64,
        };
        let tg = textgrid::from_alignment(&intervals, &trained.model.phones, &utt.words, duration);
        let path = output_path(dir, &utt.id, "TextGrid");
        ensure_parent(&path)?;
        tg.write(&path)
            .with_context(|| format!("failed to write {}", path.display()))?;
        written += 1;
    }
    field("written", written);
    Ok(written)
}
