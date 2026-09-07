//! `viter import` — convert a Montreal Forced Aligner acoustic model into a
//! `.viter` model file.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Args;
use viter_kaldi::kaldi_io;

use super::{field, fmt_duration, header, success, warn};

#[derive(Args, Debug)]
pub struct ImportArgs {
    /// An MFA acoustic model: either the `.zip` or an unpacked model directory
    #[arg(value_name = "MFA_MODEL.zip|DIR")]
    pub input: PathBuf,

    /// Where to write the converted model
    #[arg(short, long, value_name = "MODEL.viter", default_value = "model.viter")]
    pub out: PathBuf,
}

pub fn run(args: ImportArgs) -> anyhow::Result<()> {
    let started = Instant::now();

    header("Import");
    field("source", args.input.display());

    let (model, report) = kaldi_io::import_mfa(&args.input)
        .with_context(|| format!("cannot import {}", args.input.display()))?;

    field("phones", report.num_phones);
    field("pdfs", report.num_pdfs);
    field("gaussians", report.num_gauss);
    field(
        "transition ids",
        format!(
            "{} ({} states)",
            report.num_transition_ids, report.num_transition_states
        ),
    );
    field("feature dim", report.feature_dim);
    field(
        "pipeline",
        match (&model.splice, &model.deltas) {
            (Some((l, r)), _) => format!("mfcc + cmvn + splice({l},{r}) + lda"),
            (None, Some(d)) => format!("mfcc + cmvn + deltas(order {})", d.order),
            (None, None) => "mfcc + cmvn".to_string(),
        },
    );
    field(
        "speaker adaptation",
        if report.has_alignment_model {
            "yes (fMLLR, with an alignment model)"
        } else {
            "no"
        },
    );
    if let Some((rows, cols)) = report.lda_shape {
        field(
            "lda.mat",
            format!(
                "{rows}x{cols}{}",
                if model.lda.is_some() {
                    ""
                } else {
                    " (present but unused)"
                }
            ),
        );
    }

    // The rebuilt transition model must have exactly as many states as Kaldi
    // stored, or the tree and topology disagree with final.mdl.
    if report.num_transition_states != report.kaldi_num_transition_states {
        warn(&format!(
            "rebuilt {} transition states but final.mdl holds {}",
            report.num_transition_states, report.kaldi_num_transition_states
        ));
    }

    super::ensure_parent(&args.out)?;
    model
        .save(&args.out)
        .with_context(|| format!("cannot write {}", args.out.display()))?;

    let size = std::fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    println!();
    success(&format!(
        "wrote {} ({:.1} MB) in {}",
        args.out.display(),
        size as f64 / (1024.0 * 1024.0),
        fmt_duration(started.elapsed())
    ));
    Ok(())
}
