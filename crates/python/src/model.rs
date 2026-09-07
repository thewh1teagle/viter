//! The `Model` class: load a `.viter` model and align with it.

use std::path::{Path, PathBuf};

use anyhow::Context;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use viter_io::corpus::{Corpus, CorpusOptions, SpeakerSource};
use viter_io::{corpus, ctm, textgrid};
use viter_kaldi::align::AlignOptions;
use viter_kaldi::audio::Audio;
use viter_kaldi::device::Device;
use viter_kaldi::model::AcousticModel;
use viter_kaldi::types::IntervalAlignment;
use viter_train::pipeline::{AlignOverrides, refine};

use crate::alignment::{AlignSummary, Alignment};
use crate::convert::{AudioArg, audio_arg, err, py_err};

/// Speaker used when the caller does not name one. fMLLR is estimated per speaker, so
/// putting everything under one name is the "no speaker information" case.
const DEFAULT_SPEAKER: &str = "default";

/// A trained acoustic model, ready to align.
#[pyclass(module = "viter._viter", frozen)]
pub struct Model {
    pub(crate) inner: AcousticModel,
    /// The compute device, probed once at construction rather than per call.
    device: Device,
    /// Dictionary passed at construction; every alignment resolves words through it.
    dict: Option<PathBuf>,
}

#[pymethods]
impl Model {
    /// Load a `.viter` model.
    ///
    /// `dict` is the pronunciation dictionary used to turn transcripts into phones; without
    /// one, each transcript token is taken to be a phone already. `cpu` forces CPU scoring
    /// even when a GPU is available.
    #[new]
    #[pyo3(signature = (path, *, dict = None, cpu = false))]
    fn new(py: Python<'_>, path: PathBuf, dict: Option<PathBuf>, cpu: bool) -> PyResult<Self> {
        let (inner, device) = py_err(py.detach(|| {
            let inner = AcousticModel::load(&path)
                .map_err(|e| anyhow::anyhow!("failed to load model {}: {e}", path.display()))?;
            let device = if cpu { Device::cpu() } else { Device::auto() };
            anyhow::Ok((inner, device))
        }))?;
        if let Some(d) = &dict
            && !d.is_file()
        {
            return Err(err(format!("dictionary {} does not exist", d.display())));
        }
        Ok(Self {
            inner,
            device,
            dict,
        })
    }

    /// The model's phone inventory, without the `<eps>` symbol at id 0.
    #[getter]
    fn phones(&self) -> Vec<String> {
        (1..self.inner.phones.len() as u32)
            .map(|id| self.inner.phones.sym(id).to_string())
            .collect()
    }

    #[getter]
    fn feature_dim(&self) -> usize {
        self.inner.feature_dim()
    }

    /// Whether the model carries an fMLLR alignment model (a SAT model).
    #[getter]
    fn speaker_adapted(&self) -> bool {
        self.inner.am_si.is_some()
    }

    /// Write the model to `path`.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.save(&path)).map_err(err)
    }

    /// Align one utterance.
    ///
    /// `audio` is a path to read, or a 1-D float waveform in `[-1, 1]` (then `sample_rate`
    /// is required). `text` is the transcript, or a path to a `.txt`/`.lab` file. Raises
    /// `ViterError` if the utterance cannot be aligned.
    #[pyo3(signature = (audio, text, *, sample_rate = None, beam = None, retry_beam = None,
                        refine = true, speaker = None))]
    #[allow(clippy::too_many_arguments)]
    fn align(
        &self,
        py: Python<'_>,
        audio: &Bound<'_, PyAny>,
        text: &str,
        sample_rate: Option<u32>,
        beam: Option<f32>,
        retry_beam: Option<f32>,
        refine: bool,
        speaker: Option<String>,
    ) -> PyResult<Alignment> {
        let arg = audio_arg(audio, sample_rate)?;
        let speaker = speaker.unwrap_or_else(|| DEFAULT_SPEAKER.to_string());
        let mut out = self.align_items(
            py,
            vec![(arg, text.to_string())],
            &[speaker],
            beam,
            retry_beam,
            refine,
        )?;
        out.pop()
            .flatten()
            .ok_or_else(|| err("utterance failed to align (try a wider beam)"))
    }

    /// Align a batch of utterances, returning `None` for each one that failed.
    ///
    /// `items` holds `(audio, text)` or `(audio, text, sample_rate)` tuples. `speakers`
    /// names the speaker of each item; items sharing a speaker share one fMLLR transform.
    #[pyo3(signature = (items, *, speakers = None, beam = None, retry_beam = None, refine = true))]
    fn align_many(
        &self,
        py: Python<'_>,
        items: Vec<Bound<'_, PyAny>>,
        speakers: Option<Vec<String>>,
        beam: Option<f32>,
        retry_beam: Option<f32>,
        refine: bool,
    ) -> PyResult<Vec<Option<Alignment>>> {
        if let Some(s) = &speakers
            && s.len() != items.len()
        {
            return Err(PyValueError::new_err(format!(
                "speakers has {} entries but items has {}",
                s.len(),
                items.len()
            )));
        }
        let mut parsed = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let n = item.len().map_err(|_| {
                PyValueError::new_err(format!(
                    "items[{i}] must be a (audio, text) or (audio, text, sample_rate) tuple"
                ))
            })?;
            if n != 2 && n != 3 {
                return Err(PyValueError::new_err(format!(
                    "items[{i}] has {n} elements; expected (audio, text) or \
                     (audio, text, sample_rate)"
                )));
            }
            let audio = item.get_item(0)?;
            let text: String = item.get_item(1)?.extract()?;
            let rate = if n == 3 {
                item.get_item(2)?.extract::<Option<u32>>()?
            } else {
                None
            };
            parsed.push((audio_arg(&audio, rate)?, text));
        }
        let speakers = speakers.unwrap_or_else(|| vec![DEFAULT_SPEAKER.to_string(); parsed.len()]);
        self.align_items(py, parsed, &speakers, beam, retry_beam, refine)
    }

    /// Align a whole corpus directory and write TextGrids under `out_dir`, mirroring the
    /// corpus layout — the same output `viter align` produces.
    #[pyo3(signature = (corpus_dir, out_dir, *, ctm = false, beam = None, retry_beam = None,
                        refine = true))]
    #[allow(clippy::too_many_arguments)]
    fn align_corpus(
        &self,
        py: Python<'_>,
        corpus_dir: PathBuf,
        out_dir: PathBuf,
        ctm: bool,
        beam: Option<f32>,
        retry_beam: Option<f32>,
        refine: bool,
    ) -> PyResult<AlignSummary> {
        let opts = self.corpus_options(SpeakerSource::ParentDir);
        let align_opts = align_options(beam, retry_beam);
        let over = overrides(refine);

        py_err(py.detach(|| {
            let mut corpus = corpus::scan(&corpus_dir, &opts)
                .with_context(|| format!("failed to scan corpus at {}", corpus_dir.display()))?;
            anyhow::ensure!(
                !corpus.utts.is_empty(),
                "no utterances found in {}",
                corpus_dir.display()
            );
            corpus::remap(&mut corpus, &self.inner.phones)
                .context("corpus phones do not match the model's phone set")?;

            let results = viter_train::pipeline::align_corpus_with(
                &corpus,
                &self.inner,
                &self.device,
                align_opts.as_ref(),
                &over,
            )
            .context("alignment failed")?;

            std::fs::create_dir_all(&out_dir)
                .with_context(|| format!("cannot create {}", out_dir.display()))?;

            let mut aligned = 0usize;
            let mut failed = Vec::new();
            let mut ctm_rows: Vec<(IntervalAlignment, &[String])> = Vec::new();

            for (utt, result) in corpus.utts.iter().zip(results) {
                let Some(intervals) = result else {
                    failed.push(utt.id.clone());
                    continue;
                };
                let duration = audio_duration(&utt.audio).unwrap_or(intervals.duration_s() as f64);
                let tg =
                    textgrid::from_alignment(&intervals, &self.inner.phones, &utt.words, duration);
                let path = output_path(&out_dir, &utt.id, "TextGrid");
                ensure_parent(&path)?;
                tg.write(&path)
                    .with_context(|| format!("failed to write {}", path.display()))?;
                aligned += 1;
                if ctm {
                    ctm_rows.push((intervals, utt.words.as_slice()));
                }
            }

            if ctm {
                let path = out_dir.join("alignment.ctm");
                ctm::write_ctm(&path, &ctm_rows, &self.inner.phones)
                    .with_context(|| format!("failed to write {}", path.display()))?;
            }

            Ok(AlignSummary {
                utterances: corpus.utts.len(),
                aligned,
                failed,
                oov_words: corpus.oov_words.clone(),
            })
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(phones={}, feature_dim={}, speaker_adapted={})",
            self.inner.phones.len().saturating_sub(1),
            self.inner.feature_dim(),
            self.inner.am_si.is_some()
        )
    }
}

impl Model {
    /// Wrap a model that is already in memory (from `train` or `import_mfa`).
    pub(crate) fn from_parts(inner: AcousticModel, device: Device, dict: Option<PathBuf>) -> Self {
        Self {
            inner,
            device,
            dict,
        }
    }

    /// Corpus options that agree with the model: the phone tagging is the model's, so a
    /// transcript can never be resolved to phones the model has never seen.
    fn corpus_options(&self, speaker_from: SpeakerSource) -> CorpusOptions {
        CorpusOptions {
            dictionary: self.dict.clone(),
            position_dependent: self.inner.position_dependent,
            speaker_from,
            ..CorpusOptions::default()
        }
    }

    /// Shared body of `align` and `align_many`: build an in-memory corpus, align it, and
    /// convert the intervals to `Alignment`s.
    fn align_items(
        &self,
        py: Python<'_>,
        items: Vec<(AudioArg, String)>,
        speakers: &[String],
        beam: Option<f32>,
        retry_beam: Option<f32>,
        refine: bool,
    ) -> PyResult<Vec<Option<Alignment>>> {
        let align_opts = align_options(beam, retry_beam);
        let over = overrides(refine);
        let opts = self.corpus_options(SpeakerSource::Single);

        // Split the arguments into what the corpus needs (an id path per item) and the
        // waveforms, which only the in-memory aligner sees.
        let mut entries = Vec::with_capacity(items.len());
        let mut waveforms: Vec<Option<Audio>> = Vec::with_capacity(items.len());
        for (i, (arg, text)) in items.into_iter().enumerate() {
            entries.push((arg.id_path(i), speakers[i].clone(), text));
            waveforms.push(match arg {
                AudioArg::Samples(a) => Some(a),
                AudioArg::Path(_) => None,
            });
        }

        py_err(py.detach(|| {
            let refs: Vec<(&Path, &str, &str)> = entries
                .iter()
                .map(|(p, s, t)| (p.as_path(), s.as_str(), t.as_str()))
                .collect();
            let corpus = corpus::from_items(&refs, &opts, Some(&self.inner.phones))
                .context("failed to prepare the utterances")?;
            anyhow::ensure!(
                corpus.utts.len() == refs.len(),
                "{} of {} transcripts had no usable words",
                refs.len() - corpus.utts.len(),
                refs.len()
            );

            // Any in-memory sample array forces the in-memory path; the rest are read from
            // disk there too, so both cases share one code path and one set of features.
            let results = if waveforms.iter().any(|w| w.is_some()) {
                let audio = load_all(&corpus, &waveforms)?;
                viter_train::pipeline::align_corpus_with_audio(
                    &corpus,
                    &audio,
                    &self.inner,
                    &self.device,
                    align_opts.as_ref(),
                    &over,
                )
            } else {
                viter_train::pipeline::align_corpus_with(
                    &corpus,
                    &self.inner,
                    &self.device,
                    align_opts.as_ref(),
                    &over,
                )
            }
            .context("alignment failed")?;

            Ok(corpus
                .utts
                .iter()
                .zip(results)
                .enumerate()
                .map(|(i, (utt, result))| {
                    result.map(|intervals| {
                        let duration = match &waveforms[i] {
                            Some(a) if a.sample_rate > 0 => a.duration_s() as f64,
                            _ => {
                                audio_duration(&utt.audio).unwrap_or(intervals.duration_s() as f64)
                            }
                        };
                        Alignment::build(&intervals, &self.inner.phones, &utt.words, duration)
                    })
                })
                .collect())
        }))
    }
}

/// Waveform per utterance: the samples the caller passed, or the file read from disk.
fn load_all(corpus: &Corpus, waveforms: &[Option<Audio>]) -> anyhow::Result<Vec<Audio>> {
    corpus
        .utts
        .iter()
        .zip(waveforms)
        .map(|(utt, w)| match w {
            Some(a) => Ok(a.clone()),
            None => viter_kaldi::audio::read(&utt.audio)
                .with_context(|| format!("reading audio for utterance {}", utt.id)),
        })
        .collect()
}

/// Duration of an audio file in seconds, or `None` if it cannot be decoded.
fn audio_duration(path: &Path) -> Option<f64> {
    match viter_kaldi::audio::read(path) {
        Ok(a) if a.sample_rate > 0 => Some(a.samples.len() as f64 / a.sample_rate as f64),
        _ => None,
    }
}

/// Only build `AlignOptions` when something was overridden; otherwise the pipeline uses the
/// model's own defaults.
fn align_options(beam: Option<f32>, retry_beam: Option<f32>) -> Option<AlignOptions> {
    if beam.is_none() && retry_beam.is_none() {
        return None;
    }
    let mut o = AlignOptions::default();
    if let Some(b) = beam {
        o.beam = b;
    }
    if let Some(b) = retry_beam {
        o.retry_beam = b;
    }
    Some(o)
}

fn overrides(refine: bool) -> AlignOverrides {
    AlignOverrides {
        refine: refine.then(refine::RefineOptions::default),
        ..AlignOverrides::default()
    }
}

/// `out_dir/<utt id>.<ext>`, mirroring the corpus tree (the utterance id is its relative path).
fn output_path(out_dir: &Path, utt_id: &str, extension: &str) -> PathBuf {
    let mut p = out_dir.join(utt_id);
    p.set_extension(extension);
    p
}

fn ensure_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
