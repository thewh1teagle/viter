//! Turning Python arguments into Rust values, and Rust errors into `ViterError`.

use std::path::PathBuf;

use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyString};
use viter_kaldi::audio::Audio;

use crate::ViterError;

/// Every Rust error the bindings surface becomes a `ViterError`, with anyhow's full
/// `{:#}` chain as the message so the cause is not lost.
pub(crate) fn err(e: impl std::fmt::Display) -> PyErr {
    ViterError::new_err(format!("{e:#}"))
}

/// Same, for an `anyhow::Error` behind a `Result`.
pub(crate) fn py_err<T>(r: anyhow::Result<T>) -> PyResult<T> {
    r.map_err(err)
}

/// An audio argument: either a path to read, or samples already in memory.
pub(crate) enum AudioArg {
    Path(PathBuf),
    Samples(Audio),
}

impl AudioArg {
    /// The file name to use as the utterance id. In-memory audio has none, so it gets a
    /// synthetic stem — `corpus::single` only ever uses the path for the id and speaker.
    pub(crate) fn id_path(&self, index: usize) -> PathBuf {
        match self {
            AudioArg::Path(p) => p.clone(),
            AudioArg::Samples(_) => PathBuf::from(format!("utt{index:06}")),
        }
    }
}

/// Coerce one `audio` argument.
///
/// `str`/`os.PathLike` is a file to read; anything array-like is treated as a mono waveform
/// in `[-1, 1]`, and then `sample_rate` is required. A 2-D array, or a `sample_rate` passed
/// alongside a path, is a `TypeError` rather than a silently wrong alignment.
pub(crate) fn audio_arg(obj: &Bound<'_, PyAny>, sample_rate: Option<u32>) -> PyResult<AudioArg> {
    if let Some(path) = as_path(obj)? {
        if sample_rate.is_some() {
            return Err(PyTypeError::new_err(
                "sample_rate is only valid together with in-memory samples, not with an audio path",
            ));
        }
        return Ok(AudioArg::Path(path));
    }

    let Some(rate) = sample_rate else {
        return Err(PyTypeError::new_err(
            "sample_rate is required when audio is passed as samples",
        ));
    };
    if rate == 0 {
        return Err(PyTypeError::new_err("sample_rate must be positive"));
    }

    let samples = samples_of(obj)?;
    Ok(AudioArg::Samples(Audio {
        samples,
        sample_rate: rate,
    }))
}

/// `str` or `os.PathLike` -> a path; anything else -> `None`.
///
/// `bytes` is deliberately not a path here: it is ambiguous with a raw buffer, and the
/// contract only promises `str | PathLike`.
fn as_path(obj: &Bound<'_, PyAny>) -> PyResult<Option<PathBuf>> {
    if obj.is_instance_of::<PyString>() {
        return Ok(Some(PathBuf::from(obj.extract::<String>()?)));
    }
    if obj.hasattr("__fspath__")? {
        let s = obj.py().import("os")?.call_method1("fspath", (obj,))?;
        return Ok(Some(PathBuf::from(s.extract::<String>()?)));
    }
    Ok(None)
}

/// A 1-D waveform as `f32`, from a float32/float64 numpy array (zero-copy read) or from any
/// sequence of numbers.
fn samples_of(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, f32>>() {
        check_1d(a.shape())?;
        return Ok(a.as_slice()?.to_vec());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        check_1d(a.shape())?;
        return Ok(a.as_slice()?.iter().map(|&x| x as f32).collect());
    }
    // Not a float array: a list/tuple of numbers, or an int array, still works.
    obj.extract::<Vec<f32>>().map_err(|_| {
        PyTypeError::new_err(
            "audio must be a path, a 1-D numpy float32/float64 array, or a sequence of floats",
        )
    })
}

fn check_1d(shape: &[usize]) -> PyResult<()> {
    if shape.len() == 1 {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "audio must be 1-D (mono); got shape {shape:?}"
        )))
    }
}

/// A `config` argument for `train`: a path to a TOML file, or a dict of the same shape.
///
/// Both go through `TrainConfig`'s `Deserialize`, so the accepted keys are exactly the ones
/// `--config` would take.
pub(crate) fn train_config(obj: &Bound<'_, PyAny>) -> PyResult<viter_train::config::TrainConfig> {
    if let Some(path) = as_path(obj)? {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| err(format!("cannot read config {}: {e}", path.display())))?;
        return toml::from_str(&text)
            .map_err(|e| err(format!("invalid config {}: {e}", path.display())));
    }
    if obj.cast::<PyDict>().is_ok() {
        // json is the pivot: it is the one serde format both Python and serde_json agree on
        // without a bespoke visitor, and TrainConfig is plain data.
        let json: String = obj
            .py()
            .import("json")?
            .call_method1("dumps", (obj,))?
            .extract()?;
        return serde_json::from_str(&json).map_err(|e| err(format!("invalid config: {e}")));
    }
    Err(PyTypeError::new_err(
        "config must be a dict or a path to a TOML file",
    ))
}
