//! Command-line surface: `train`, `align`, `serve`, plus the output helpers they share.

pub mod align;
pub mod import;
pub mod serve;
pub mod train;

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use owo_colors::OwoColorize;
use viter_io::corpus::{CorpusOptions, SpeakerSource};
use viter_kaldi::device::Device;

/// Rusty forced aligner. Train, align, serve. One binary.
#[derive(Parser, Debug)]
#[command(
    name = "viter",
    version,
    about = "Rusty forced aligner. Train, align, serve.",
    long_about = "viter trains Kaldi-style GMM acoustic models, force-aligns speech to \
                  transcripts, and serves a browser viewer for the resulting TextGrids.",
    propagate_version = true
)]
pub struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Train an acoustic model from a corpus of audio + transcripts
    Train(train::TrainArgs),
    /// Align a corpus with a trained model, write TextGrids
    Align(align::AlignArgs),
    /// Convert a Montreal Forced Aligner acoustic model into a .viter model
    Import(import::ImportArgs),
    /// Serve a browser viewer for TextGrids + audio
    Serve(serve::ServeArgs),
}

impl Cli {
    pub fn run(self) -> anyhow::Result<()> {
        match self.cmd {
            Cmd::Train(a) => train::run(a),
            Cmd::Align(a) => align::run(a),
            Cmd::Import(a) => import::run(a),
            Cmd::Serve(a) => serve::run(a),
        }
    }
}

/// Install the tracing subscriber. `RUST_LOG` overrides the default `info` level.
pub fn init_logging() -> anyhow::Result<()> {
    use tracing_subscriber::EnvFilter;
    // Default: warnings only, plus the one-line device announcement. The progress
    // bars and per-iteration summaries are printed by the train pipeline itself,
    // so INFO tracing would only duplicate them. `RUST_LOG=info` or `debug` for more.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    // VITER_TIMING=1 stamps every log line with seconds since start, which
    // is the cheapest way to see where a run spends its time (`RUST_LOG=debug`).
    let timing = std::env::var_os("VITER_TIMING").is_some();
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    if timing {
        builder
            .with_timer(tracing_subscriber::fmt::time::uptime())
            .init();
    } else {
        builder.without_time().init();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// shared arguments
// ---------------------------------------------------------------------------

/// Corpus-shaping flags shared by `train` and `align`.
#[derive(Args, Debug, Clone)]
pub struct CorpusArgs {
    /// Pronunciation dictionary (MFA/CMUdict format). Without one, each transcript token is
    /// treated as a single phone.
    #[arg(long, value_name = "FILE")]
    pub dict: Option<PathBuf>,

    /// Phone used for words missing from the dictionary
    #[arg(long, default_value = "spn", value_name = "PHONE")]
    pub oov_phone: String,

    /// Optional silence phone inserted between words
    #[arg(long, default_value = "sil", value_name = "PHONE")]
    pub silence_phone: String,

    /// Disable word-position-dependent phones (AH_B / AH_I / AH_E / AH_S)
    #[arg(long)]
    pub no_position_dependent: bool,

    /// How to derive a speaker id from each file's path
    #[arg(long, value_enum, default_value_t = SpeakerMode::ParentDir)]
    pub speakers: SpeakerMode,

    /// With `--speakers prefix`, how many leading characters of the file stem name the speaker
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub speaker_prefix_len: usize,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerMode {
    /// Each file's parent directory is its speaker
    ParentDir,
    /// The first `--speaker-prefix-len` characters of the file stem
    Prefix,
    /// The whole corpus is one speaker
    Single,
}

impl CorpusArgs {
    pub fn to_options(&self) -> CorpusOptions {
        CorpusOptions {
            dictionary: self.dict.clone(),
            oov_phone: self.oov_phone.clone(),
            silence_phone: self.silence_phone.clone(),
            position_dependent: !self.no_position_dependent,
            speaker_from: match self.speakers {
                SpeakerMode::ParentDir => SpeakerSource::ParentDir,
                SpeakerMode::Prefix => SpeakerSource::Prefix(self.speaker_prefix_len),
                SpeakerMode::Single => SpeakerSource::Single,
            },
        }
    }
}

/// Pick the compute device, honouring `--cpu`.
pub fn device(force_cpu: bool) -> Device {
    if force_cpu {
        Device::cpu()
    } else {
        Device::auto()
    }
}

// ---------------------------------------------------------------------------
// output helpers
// ---------------------------------------------------------------------------

/// A bold section header, e.g. `== Corpus ==`.
pub fn header(text: &str) {
    println!();
    println!("{} {}", "==".dimmed(), text.bold());
}

/// A `label: value` line, aligned to a common column.
pub fn field(label: &str, value: impl std::fmt::Display) {
    println!("  {:<18} {}", format!("{label}:").dimmed(), value);
}

/// A green success line.
pub fn success(text: &str) {
    println!("{} {}", "✓".green().bold(), text);
}

/// A yellow warning line.
pub fn warn(text: &str) {
    println!("{} {}", "!".yellow().bold(), text);
}

/// Format a duration as `1h 02m 03s` / `2m 03s` / `3.4s`.
pub fn fmt_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{}m {:02}s", secs as u64 / 60, secs as u64 % 60)
    } else {
        let s = secs as u64;
        format!("{}h {:02}m {:02}s", s / 3600, (s % 3600) / 60, s % 60)
    }
}

/// Real-time factor: processing time divided by audio duration. `—` when audio length is unknown.
pub fn fmt_rtf(elapsed: std::time::Duration, audio_seconds: f64) -> String {
    if audio_seconds > 0.0 {
        format!("{:.3}x", elapsed.as_secs_f64() / audio_seconds)
    } else {
        "—".to_string()
    }
}

/// Total duration of a corpus's audio, by decoding each file's header.
///
/// Used only for the RTF line, so a file that cannot be probed is skipped rather than fatal.
/// Report what the lexicon actually contributed: how many word tokens have more than one
/// pronunciation to choose between, and whether the dictionary carried MFA's probability columns
/// (which switch the graph's silence costs from the global defaults to per-pronunciation ones).
pub fn report_pron_stats(utts: &[viter_kaldi::types::Utterance]) {
    let mut multi = 0usize;
    let mut total = 0usize;
    let mut has_probs = false;
    for u in utts {
        for alts in &u.prons {
            total += 1;
            if alts.len() > 1 {
                multi += 1;
            }
            for p in alts {
                if p.prob.is_some()
                    || p.silence_after_prob.is_some()
                    || p.silence_before_correction.is_some()
                    || p.non_silence_before_correction.is_some()
                {
                    has_probs = true;
                }
            }
        }
    }
    let pct = if total == 0 {
        0.0
    } else {
        100.0 * multi as f64 / total as f64
    };
    field("multi-pron words", format!("{multi} / {total} ({pct:.1}%)"));
    field(
        "pron probabilities",
        if has_probs {
            "yes (dictionary has silence columns)"
        } else {
            "no"
        },
    );
}

pub fn corpus_audio_seconds(utts: &[viter_kaldi::types::Utterance]) -> f64 {
    utts.iter()
        .filter_map(|u| viter_kaldi::audio::read(&u.audio).ok())
        .map(|a| {
            if a.sample_rate == 0 {
                0.0
            } else {
                a.samples.len() as f64 / a.sample_rate as f64
            }
        })
        .sum()
}

/// Where an utterance's output file goes: `out_dir` joined with the utterance id (which is the
/// corpus-relative path without extension), so the input tree structure is mirrored.
pub fn output_path(out_dir: &Path, utt_id: &str, extension: &str) -> PathBuf {
    let mut p = out_dir.join(utt_id);
    p.set_extension(extension);
    p
}

/// Create the parent directory of `path` if it does not exist.
pub fn ensure_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Peak resident set size of this process, from /proc on Linux, as a human string.
/// Returns None where unavailable.
pub fn peak_rss() -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: u64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(if kb >= 1024 * 1024 {
        format!("{:.1} GB", kb as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} MB", kb / 1024)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn durations_are_human_readable() {
        assert_eq!(fmt_duration(Duration::from_secs_f64(3.44)), "3.4s");
        assert_eq!(fmt_duration(Duration::from_secs(123)), "2m 03s");
        assert_eq!(fmt_duration(Duration::from_secs(3723)), "1h 02m 03s");
    }

    #[test]
    fn rtf_is_elapsed_over_audio() {
        assert_eq!(fmt_rtf(Duration::from_secs(60), 600.0), "0.100x");
        assert_eq!(fmt_rtf(Duration::from_secs(60), 0.0), "—");
    }

    #[test]
    fn output_paths_mirror_the_corpus_tree() {
        let out = Path::new("/out");
        assert_eq!(
            output_path(out, "spk1/utt_003", "TextGrid"),
            Path::new("/out/spk1/utt_003.TextGrid")
        );
        assert_eq!(
            output_path(out, "utt", "TextGrid"),
            Path::new("/out/utt.TextGrid")
        );
    }

    #[test]
    fn cli_parses() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
