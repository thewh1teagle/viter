//! `train` and `import_mfa`.

use std::path::PathBuf;
use std::sync::Arc;

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
///
/// `progress` is a callable taking one dict with the keys `done`, `total`, `fraction`,
/// `elapsed` (seconds), `eta` (seconds or `None`), `stage`, `step` and `mismatches`
/// (a plan-vs-run disagreement counter, normally 0). It is called at most
/// ~10 times a second and on every stage change, with `done == total` and `fraction == 1.0`
/// exactly once at the end; the unit of `done`/`total` is one utterance-pass, while
/// `fraction` is the elapsed share of the *predicted* total time. Exceptions it raises are
/// reported as unraisable and do not stop training. `quiet=True` suppresses the terminal bar
/// while still calling `progress`.
///
/// ```python
/// from tqdm import tqdm
/// bar = tqdm(total=1000, unit="permille")
/// def on_progress(info):
///     bar.n = int(1000 * info["fraction"])
///     bar.set_description(info["stage"])
///     bar.refresh()
/// viter.train("corpus", "model.viter", progress=on_progress, quiet=True)
/// ```
#[pyfunction]
#[pyo3(signature = (corpus_dir, out = None, *, dict = None, config = None, cpu = false,
                    seed = None, no_tri = false, no_lda = false, no_sat = false,
                    no_pron_probs = false, sat_rounds = None, no_subset = false,
                    position_dependent = true, work_dir = None, progress = None,
                    quiet = false))]
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
    progress: Option<&Bound<'_, PyAny>>,
    quiet: bool,
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

    // The callback is invoked from the training threads, which have no GIL of their own.
    // A raising callback must not abort a run that may already be many minutes in, so its
    // error is reported as unraisable and swallowed.
    let sink: Option<viter_train::pipeline::ProgressSink> = progress.map(|cb| {
        let cb: Py<PyAny> = cb.clone().unbind();
        Arc::new(move |ev: &viter_train::pipeline::ProgressEvent| {
            Python::attach(|py| {
                let call = || -> PyResult<()> {
                    let info = pyo3::types::PyDict::new(py);
                    info.set_item("done", ev.done)?;
                    info.set_item("total", ev.total)?;
                    info.set_item("fraction", ev.fraction)?;
                    info.set_item("elapsed", ev.elapsed.as_secs_f64())?;
                    info.set_item("eta", ev.eta.map(|d| d.as_secs_f64()))?;
                    info.set_item("stage", ev.stage.as_str())?;
                    info.set_item("step", ev.step.as_str())?;
                    info.set_item("mismatches", ev.mismatches)?;
                    cb.call1(py, (info,))?;
                    Ok(())
                };
                if let Err(e) = call() {
                    e.write_unraisable(py, None);
                }
            });
        }) as viter_train::pipeline::ProgressSink
    });
    let train_opts = viter_train::pipeline::TrainOptions {
        final_alignment: false,
        progress: sink,
        quiet,
    };

    let trained = py_err(py.detach(|| {
        let corpus = viter_io::corpus::scan(&corpus_dir, &opts)
            .with_context(|| format!("failed to scan corpus at {}", corpus_dir.display()))?;
        anyhow::ensure!(
            !corpus.utts.is_empty(),
            "no utterances found in {} (expected audio files with matching .txt or .lab \
             transcripts)",
            corpus_dir.display()
        );
        viter_train::pipeline::train_with(&corpus, &cfg, &device, work_dir.as_deref(), &train_opts)
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
