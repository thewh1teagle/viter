//! The result types: `Phone`, `Word`, `Alignment`, `AlignSummary`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use viter_io::textgrid::{self, TextGrid};
use viter_kaldi::types::{IntervalAlignment, SymbolTable};

use crate::convert::err;

/// One phone interval, in seconds. The label is untagged (`AH_B` -> `AH`), matching
/// what the TextGrid writer emits.
#[pyclass(module = "viter._viter", frozen, get_all)]
#[derive(Clone)]
pub struct Phone {
    pub label: String,
    pub start: f64,
    pub end: f64,
}

#[pymethods]
impl Phone {
    fn __repr__(&self) -> String {
        format!(
            "Phone(label={:?}, start={:.3}, end={:.3})",
            self.label, self.start, self.end
        )
    }
}

/// One word interval, with the phones that fall inside it.
#[pyclass(module = "viter._viter", frozen, get_all)]
#[derive(Clone)]
pub struct Word {
    pub label: String,
    pub start: f64,
    pub end: f64,
    pub phones: Vec<Phone>,
}

#[pymethods]
impl Word {
    fn __repr__(&self) -> String {
        format!(
            "Word(label={:?}, start={:.3}, end={:.3}, phones={})",
            self.label,
            self.start,
            self.end,
            self.phones.len()
        )
    }
}

/// A finished alignment of one utterance.
#[pyclass(module = "viter._viter", frozen)]
pub struct Alignment {
    #[pyo3(get)]
    pub words: Vec<Word>,
    #[pyo3(get)]
    pub phones: Vec<Phone>,
    #[pyo3(get)]
    pub duration: f64,
    /// Kept so `textgrid()` can re-render exactly what `viter align` writes.
    grid: TextGrid,
}

#[pymethods]
impl Alignment {
    /// Write the alignment as a Praat TextGrid (long format).
    fn to_textgrid(&self, path: PathBuf) -> PyResult<()> {
        self.grid.write(&path).map_err(err)
    }

    /// The TextGrid text, exactly as `to_textgrid` would write it.
    fn textgrid(&self) -> String {
        self.grid.to_string_long()
    }

    /// Plot the alignment over `audio`; see `viter.plot.plot`.
    #[pyo3(signature = (audio, sample_rate = None, **kwargs))]
    fn plot<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        audio: &Bound<'py, PyAny>,
        sample_rate: Option<&Bound<'py, PyAny>>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let func = py.import("viter.plot")?.getattr("plot")?;
        let args = (slf, audio, sample_rate);
        func.call(args, kwargs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Alignment(words={}, phones={}, duration={:.3})",
            self.words.len(),
            self.phones.len(),
            self.duration
        )
    }
}

impl Alignment {
    /// Build from an interval alignment, going through the same TextGrid conversion the CLI
    /// uses so Python times and TextGrid times can never disagree.
    pub fn build(
        intervals: &IntervalAlignment,
        phones: &SymbolTable,
        words: &[String],
        duration: f64,
    ) -> Self {
        let grid = textgrid::from_alignment(intervals, phones, words, duration);
        // `from_alignment` puts words first, then phones, and fills the gaps between
        // intervals with empty-labelled ones. Empty intervals are silence padding, not
        // aligned units, so they are dropped here.
        let tier = |name: &str| {
            grid.tiers
                .iter()
                .find(|t| t.name == name)
                .map(|t| {
                    t.intervals
                        .iter()
                        .filter(|iv| !iv.text.is_empty())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };

        let phone_list: Vec<Phone> = tier("phones")
            .into_iter()
            .map(|iv| Phone {
                label: iv.text.clone(),
                start: iv.xmin,
                end: iv.xmax,
            })
            .collect();

        let word_list: Vec<Word> = tier("words")
            .into_iter()
            .map(|iv| Word {
                label: iv.text.clone(),
                start: iv.xmin,
                end: iv.xmax,
                // A phone belongs to the word whose span contains its midpoint, which is
                // robust to the 1 ms boundary refinement nudging edges either way.
                phones: phone_list
                    .iter()
                    .filter(|p| {
                        let mid = 0.5 * (p.start + p.end);
                        mid >= iv.xmin && mid < iv.xmax
                    })
                    .cloned()
                    .collect(),
            })
            .collect();

        Self {
            words: word_list,
            phones: phone_list,
            duration: grid.xmax,
            grid,
        }
    }
}

/// What `Model.align_corpus` did.
#[pyclass(module = "viter._viter", frozen, get_all)]
pub struct AlignSummary {
    /// Utterances found in the corpus.
    pub utterances: usize,
    /// How many produced a TextGrid.
    pub aligned: usize,
    /// Ids of the utterances that failed to align.
    pub failed: Vec<String>,
    /// Out-of-vocabulary word -> number of occurrences.
    pub oov_words: BTreeMap<String, usize>,
}

#[pymethods]
impl AlignSummary {
    fn __repr__(&self) -> String {
        format!(
            "AlignSummary(utterances={}, aligned={}, failed={}, oov_words={})",
            self.utterances,
            self.aligned,
            self.failed.len(),
            self.oov_words.len()
        )
    }
}
