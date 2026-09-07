//! `train` and `import_mfa`.

use std::path::PathBuf;

use anyhow::Context;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use viter_io::corpus::{CorpusOptions, SpeakerSource};
use viter_kaldi::device::Device;
use viter_train::config::{StageSpec, TrainConfig};

use crate::convert::{err, py_err, train_config};
use crate::model::Model;

/// Train an acoustic model from a corpus directory of audio files with matching
/// `.txt`/`.lab` transcripts.
///
/// The keyword arguments mirror `viter train`'s flags one for one. Returns the trained
/// model, and also writes it to `out` when that is given.
#[pyfunction]
#[pyo3(signature = (corpus_dir, out = None, *, dict = None, config = None, cpu = false,
                    seed = None, no_tri = false, no_lda = false, no_sat = false,
                    no_pron_probs = false, sat_rounds = None, no_subset = false,
                    position_dependent = true, work_dir = None))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub fn train(
    py: Python<'_>,
    corpus_dir: PathBuf,
    out: Option<PathBuf>,
    dict: Option<PathBuf>,
    config: Option<&Bound<'_, PyAny>>,
    cpu: bool,
    seed: Option<u64>,
    no_tri: bool,
    no_lda: bool,
    no_sat: bool,
    no_pron_probs: bool,
    sat_rounds: Option<usize>,
    no_subset: bool,
    position_dependent: bool,
    work_dir: Option<PathBuf>,
) -> PyResult<Model> {
    let mut cfg = match config {
        Some(obj) => train_config(obj)?,
        None => TrainConfig::default(),
    };
    if let Some(s) = seed {
        cfg.seed = s;
    }
    cfg.subset = !no_subset;
    // The corpus is built with (or without) word-position tags; the model records the same
    // choice so `align` tags transcripts identically.
    cfg.position_dependent = position_dependent;
    cfg.graph.position_dependent = position_dependent;
    if no_tri {
        cfg.stages.tri = false;
    }
    if no_tri || no_lda {
        cfg.stages.lda = false;
    }
    if no_tri || no_sat {
        cfg.stages.sat = false;
    }
    if no_pron_probs {
        cfg.stages.pron_probs = false;
    }
    if let Some(rounds) = sat_rounds {
        // Keep the first `rounds` SAT entries; drop the rest, and any pron-prob round that
        // would then have no SAT stage left to follow it.
        let mut seen = 0usize;
        cfg.schedule.retain(|s| match s {
            StageSpec::Sat { .. } => {
                seen += 1;
                seen <= rounds
            }
            StageSpec::PronProbs { .. } => seen < rounds,
            _ => true,
        });
    }

    let opts = CorpusOptions {
        dictionary: dict,
        position_dependent,
        speaker_from: SpeakerSource::ParentDir,
        ..CorpusOptions::default()
    };

    // Training is the one call long enough that Ctrl-C must not have to wait for it: signals
    // are checked on either side, so an interrupt raised during the run is delivered as soon
    // as the GIL comes back.
    py.check_signals()?;
    let device = if cpu { Device::cpu() } else { Device::auto() };

    let trained = py_err(py.detach(|| {
        let corpus = viter_io::corpus::scan(&corpus_dir, &opts)
            .with_context(|| format!("failed to scan corpus at {}", corpus_dir.display()))?;
        anyhow::ensure!(
            !corpus.utts.is_empty(),
            "no utterances found in {} (expected audio files with matching .txt or .lab \
             transcripts)",
            corpus_dir.display()
        );
        viter_train::pipeline::train(&corpus, &cfg, &device, work_dir.as_deref())
            .context("training failed")
    }))?;
    py.check_signals()?;

    if let Some(path) = &out {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(err)?;
        }
        trained.model.save(path).map_err(err)?;
    }
    Ok(Model::from_parts(trained.model, device, opts.dictionary))
}

/// Convert a Montreal Forced Aligner acoustic model (a `.zip` or an unpacked directory)
/// into a viter model, optionally writing it to `out`.
#[pyfunction]
#[pyo3(signature = (path, out = None))]
pub fn import_mfa(py: Python<'_>, path: PathBuf, out: Option<PathBuf>) -> PyResult<Model> {
    let model = py_err(py.detach(|| {
        let (model, _report) = viter_kaldi::kaldi_io::import_mfa(&path)
            .with_context(|| format!("cannot import {}", path.display()))?;
        anyhow::Ok(model)
    }))?;

    if let Some(target) = &out {
        if let Some(parent) = target.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(err)?;
        }
        model.save(target).map_err(err)?;
    }
    Ok(Model::from_parts(model, Device::auto(), None))
}
