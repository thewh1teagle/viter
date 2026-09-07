//! `viter align` — force-align a corpus (or one file) with a trained model.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Args;
use owo_colors::OwoColorize;
use viter_io::{corpus, ctm, textgrid};
use viter_kaldi::align::AlignOptions;
use viter_kaldi::model::AcousticModel;

use super::{
    CorpusArgs, ensure_parent, field, fmt_duration, fmt_rtf, header, output_path, success, warn,
};

#[derive(Args, Debug)]
pub struct AlignArgs {
    /// Corpus directory, or a single audio file (then --text or a sibling .txt/.lab is used)
    #[arg(value_name = "CORPUS_DIR|AUDIO")]
    pub input: PathBuf,

    /// Trained model
    #[arg(value_name = "MODEL.viter")]
    pub model: PathBuf,

    /// Output directory for TextGrids, mirroring the input structure
    #[arg(short, long, value_name = "DIR", default_value = "aligned")]
    pub out: PathBuf,

    /// Transcript for single-file mode (overrides a sibling .txt/.lab)
    #[arg(long, value_name = "TEXT")]
    pub text: Option<String>,

    #[command(flatten)]
    pub corpus_args: CorpusArgs,

    /// Viterbi beam
    #[arg(long, value_name = "F")]
    pub beam: Option<f32>,

    /// Wider beam used to retry an utterance that failed to align
    #[arg(long, value_name = "F")]
    pub retry_beam: Option<f32>,

    /// Acoustic scale applied to frame log-likelihoods
    #[arg(long, value_name = "F")]
    pub acoustic_scale: Option<f32>,

    /// Boost silence pdf weights during alignment (MFA align default 1.0)
    #[arg(long, value_name = "F")]
    pub boost_silence: Option<f32>,

    /// Global probability of optional silence between words (MFA default 0.5)
    #[arg(long, value_name = "P")]
    pub silence_prob: Option<f32>,

    /// Probability of silence at the start of an utterance (MFA default 0.5)
    #[arg(long, value_name = "P")]
    pub initial_silence_prob: Option<f32>,

    /// Multiplier on ending with silence (MFA default 1.0)
    #[arg(long, value_name = "F")]
    pub final_silence_correction: Option<f32>,

    /// Multiplier on ending without silence (MFA default 1.0)
    #[arg(long, value_name = "F")]
    pub final_non_silence_correction: Option<f32>,

    /// Also write a CTM file next to the TextGrids
    #[arg(long)]
    pub ctm: bool,

    /// Force CPU scoring even when a GPU is available
    #[arg(long)]
    pub cpu: bool,

    /// Append a plain-text copy of the progress output (every iteration) to this file
    #[arg(long, value_name = "FILE")]
    pub log: Option<std::path::PathBuf>,
}

pub fn run(args: AlignArgs) -> anyhow::Result<()> {
    if let Some(log) = &args.log {
        viter_train::pipeline::progress::set_log_file(log)
            .map_err(|e| anyhow::anyhow!("cannot open log file {}: {e}", log.display()))?;
    }
    let started = Instant::now();

    // --- model ------------------------------------------------------------
    header("Model");
    let model = AcousticModel::load(&args.model)
        .with_context(|| format!("failed to load model {}", args.model.display()))?;
    field("path", args.model.display());
    field("phones", model.phones.len().saturating_sub(1));
    field("pdfs", model.tm.num_pdfs());
    field("feature dim", model.feature_dim());
    field(
        "speaker adapted",
        if model.am_si.is_some() {
            "yes (fMLLR)"
        } else {
            "no"
        },
    );

    // --- corpus -----------------------------------------------------------
    header("Corpus");
    let mut opts = args.corpus_args.to_options();
    // The model's phone inventory decides how transcripts must be tagged; a mismatched flag
    // would silently produce phones the model has never seen.
    opts.position_dependent = model.position_dependent;
    if args.corpus_args.no_position_dependent {
        // Explicit override, for models whose recorded flag is wrong.
        opts.position_dependent = false;
    }

    let mut corpus = if args.input.is_dir() {
        corpus::scan(&args.input, &opts)
            .with_context(|| format!("failed to scan corpus at {}", args.input.display()))?
    } else {
        anyhow::ensure!(
            args.input.is_file(),
            "{} is neither a directory nor a file",
            args.input.display()
        );
        let text = resolve_transcript(&args)?;
        corpus::single(&args.input, &text, &opts, Some(&model.phones))
            .with_context(|| format!("failed to prepare {}", args.input.display()))?
    };
    anyhow::ensure!(
        !corpus.utts.is_empty(),
        "no utterances found in {}",
        args.input.display()
    );

    // Re-resolve the corpus phone ids against the model's symbol table.
    corpus::remap(&mut corpus, &model.phones)
        .context("corpus phones do not match the model's phone set")?;

    field("utterances", corpus.utts.len());
    field("speakers", corpus.speakers.len());
    super::report_pron_stats(&corpus.utts);
    if !corpus.oov_words.is_empty() {
        let types = corpus.oov_words.len();
        let tokens: usize = corpus.oov_words.values().sum();
        field(
            "OOV words",
            format!("{types} types / {tokens} tokens")
                .yellow()
                .to_string(),
        );
    }

    // --- align ------------------------------------------------------------
    header("Aligning");
    let device = super::device(args.cpu);
    field(
        "device",
        match device.adapter_name() {
            Some(n) => format!("{:?} · {n}", device.kind()),
            None => format!("{:?}", device.kind()),
        },
    );

    let align_opts = align_options(&args);
    if let Some(o) = &align_opts {
        field("beam", format!("{} (retry {})", o.beam, o.retry_beam));
        field("acoustic scale", o.acoustic_scale);
    }

    let audio_seconds = super::corpus_audio_seconds(&corpus.utts);

    let overrides = viter_train::pipeline::AlignOverrides {
        silence_prob: args.silence_prob,
        initial_silence_prob: args.initial_silence_prob,
        final_silence_correction: args.final_silence_correction,
        final_non_silence_correction: args.final_non_silence_correction,
        boost_silence: args.boost_silence,
    };
    let results = viter_train::pipeline::align_corpus_with(
        &corpus,
        &model,
        &device,
        align_opts.as_ref(),
        &overrides,
    )
    .context("alignment failed")?;
    anyhow::ensure!(
        results.len() == corpus.utts.len(),
        "aligner returned {} results for {} utterances",
        results.len(),
        corpus.utts.len()
    );

    // --- write ------------------------------------------------------------
    header("Output");
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("cannot create {}", args.out.display()))?;

    let mut written = 0usize;
    let mut failed = Vec::new();
    // Collected for the optional CTM, which wants every alignment in one call.
    let mut ctm_rows: Vec<(viter_kaldi::types::IntervalAlignment, &[String])> = Vec::new();

    for (utt, result) in corpus.utts.iter().zip(results.into_iter()) {
        let Some(intervals) = result else {
            failed.push(utt.id.clone());
            continue;
        };
        let duration = match viter_kaldi::audio::read(&utt.audio) {
            Ok(a) if a.sample_rate > 0 => a.samples.len() as f64 / a.sample_rate as f64,
            _ => intervals.duration_s() as f64,
        };
        let tg = textgrid::from_alignment(&intervals, &model.phones, &utt.words, duration);
        let path = output_path(&args.out, &utt.id, "TextGrid");
        ensure_parent(&path)?;
        tg.write(&path)
            .with_context(|| format!("failed to write {}", path.display()))?;
        written += 1;
        if args.ctm {
            ctm_rows.push((intervals, utt.words.as_slice()));
        }
    }

    if args.ctm {
        let path = args.out.join("alignment.ctm");
        ctm::write_ctm(&path, &ctm_rows, &model.phones)
            .with_context(|| format!("failed to write {}", path.display()))?;
        field("CTM", path.display());
    }

    // --- summary ----------------------------------------------------------
    let elapsed = started.elapsed();
    header("Summary");
    field("utterances", corpus.utts.len());
    field("aligned", written);
    if failed.is_empty() {
        field("failed", 0);
    } else {
        field("failed", failed.len().to_string().yellow().to_string());
        for id in failed.iter().take(10) {
            println!("      {}", id.yellow());
        }
        if failed.len() > 10 {
            println!(
                "      {}",
                format!("... and {} more", failed.len() - 10).dimmed()
            );
        }
        warn("failed utterances usually mean a transcript/audio mismatch or too narrow a beam");
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
    success(&format!(
        "{} TextGrid(s) in {}",
        written,
        args.out.display().bold()
    ));
    Ok(())
}

/// Build [`AlignOptions`] only if the user overrode something; otherwise let the pipeline use
/// the model's own defaults.
fn align_options(args: &AlignArgs) -> Option<AlignOptions> {
    if args.beam.is_none() && args.retry_beam.is_none() && args.acoustic_scale.is_none() {
        return None;
    }
    // CONTRACT-DEVIATION: plans/CONTRACTS.md does not state that `AlignOptions` derives `Default`.
    // This uses it to fill the fields the user did not override; the documented per-field
    // defaults (beam 10, retry_beam 40, acoustic_scale 0.1) are what it must produce.
    let mut o = AlignOptions::default();
    if let Some(b) = args.beam {
        o.beam = b;
    }
    if let Some(b) = args.retry_beam {
        o.retry_beam = b;
    }
    if let Some(s) = args.acoustic_scale {
        o.acoustic_scale = s;
    }
    Some(o)
}

/// Transcript for single-file mode: `--text`, else a sibling `.txt` or `.lab`.
fn resolve_transcript(args: &AlignArgs) -> anyhow::Result<String> {
    if let Some(t) = &args.text {
        return Ok(t.clone());
    }
    for ext in ["txt", "lab"] {
        let mut p = args.input.clone();
        p.set_extension(ext);
        if p.is_file() {
            return std::fs::read_to_string(&p)
                .with_context(|| format!("failed to read {}", p.display()));
        }
    }
    anyhow::bail!(
        "no transcript for {}: pass --text \"...\" or add a sibling .txt/.lab file",
        args.input.display()
    )
}
