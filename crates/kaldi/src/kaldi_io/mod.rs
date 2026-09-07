//! Import a Montreal Forced Aligner (Kaldi) acoustic model into viter's
//! [`AcousticModel`].
//!
//! An MFA model is a zip (or an unpacked directory) holding Kaldi binary files:
//!
//! ```text
//! final.mdl         TransitionModel + AmDiagGmm  (speaker-adapted)
//! final.alimdl      the same, speaker-independent (first fMLLR pass)
//! tree              ContextDependency
//! lda.mat           an LDA+MLLT matrix (only used when meta says uses_splices)
//! phones.txt        symbol table, "<symbol>\t<id>" per line
//! meta.json         feature and graph settings
//! ```
//!
//! The one non-mechanical step is the transition model. Kaldi's transition ids
//! are defined by its tuple ordering, which viter's [`TransitionModel::new`]
//! rebuilds from the imported tree and topology. Rather than trust the two
//! orderings to agree, the importer builds viter's model and then copies
//! Kaldi's probabilities across by matching `(phone, hmm_state, forward_pdf,
//! self_loop_pdf)` tuples, so the resulting ids are viter's own and the
//! probabilities are MFA's.

pub mod objects;
pub mod read;

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::hmm::transition::{TransitionModel, Tuple};
use crate::model::AcousticModel;
use crate::types::{PhoneId, SymbolTable};

pub use read::{KaldiReadError, KaldiReader};

/// Errors from importing an MFA model.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("i/o error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not a readable zip archive: {source}")]
    Zip {
        path: String,
        #[source]
        source: zip::result::ZipError,
    },

    #[error("{path} does not look like an MFA model: no {missing} inside it")]
    Missing { path: String, missing: String },

    #[error("failed to parse {file}: {source}")]
    Parse {
        file: String,
        #[source]
        source: KaldiReadError,
    },

    #[error("failed to parse {file}: {msg}")]
    BadContent { file: String, msg: String },

    #[error("the imported tree and topology do not agree with final.mdl: {0}")]
    Inconsistent(String),
}

type Result<T> = std::result::Result<T, ImportError>;

/// What an import produced, beyond the model itself: the counts worth printing
/// and checking against Kaldi's.
#[derive(Clone, Debug)]
pub struct ImportReport {
    pub num_pdfs: usize,
    pub num_gauss: usize,
    pub num_transition_ids: usize,
    pub num_transition_states: usize,
    /// Tuple count Kaldi itself stored, for comparison with the rebuilt model.
    pub kaldi_num_transition_states: usize,
    pub feature_dim: usize,
    pub num_phones: usize,
    pub has_alignment_model: bool,
    pub lda_shape: Option<(usize, usize)>,
}

/// Import an MFA model from a zip file or an unpacked directory.
pub fn import_mfa(path: &Path) -> Result<(AcousticModel, ImportReport)> {
    let files = ModelFiles::open(path)?;
    build(files)
}

// ---------------------------------------------------------------------------
// file access: a directory and a zip look the same to the importer
// ---------------------------------------------------------------------------

enum ModelFiles {
    Dir(PathBuf),
    Zip {
        path: PathBuf,
        archive: zip::ZipArchive<std::fs::File>,
        /// Prefix every member shares, e.g. `ljspeech_ipa/`.
        prefix: String,
    },
}

impl ModelFiles {
    fn open(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path).map_err(|source| ImportError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if meta.is_dir() {
            return Ok(ModelFiles::Dir(path.to_path_buf()));
        }

        let file = std::fs::File::open(path).map_err(|source| ImportError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let archive = zip::ZipArchive::new(file).map_err(|source| ImportError::Zip {
            path: path.display().to_string(),
            source,
        })?;

        // MFA zips wrap everything in a directory named after the model, but
        // accept a flat archive too.
        let prefix = archive
            .file_names()
            .find(|n| n.ends_with("meta.json"))
            .map(|n| n.trim_end_matches("meta.json").to_string())
            .unwrap_or_default();

        Ok(ModelFiles::Zip {
            path: path.to_path_buf(),
            archive,
            prefix,
        })
    }

    fn display(&self) -> String {
        match self {
            ModelFiles::Dir(p) => p.display().to_string(),
            ModelFiles::Zip { path, .. } => path.display().to_string(),
        }
    }

    /// Read a member, or `None` when it is absent.
    fn get(&mut self, name: &str) -> Result<Option<Vec<u8>>> {
        match self {
            ModelFiles::Dir(dir) => {
                let p = dir.join(name);
                match std::fs::read(&p) {
                    Ok(b) => Ok(Some(b)),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                    Err(source) => Err(ImportError::Io {
                        path: p.display().to_string(),
                        source,
                    }),
                }
            }
            ModelFiles::Zip {
                path,
                archive,
                prefix,
            } => {
                let full = format!("{prefix}{name}");
                let mut entry = match archive.by_name(&full) {
                    Ok(e) => e,
                    Err(zip::result::ZipError::FileNotFound) => return Ok(None),
                    Err(source) => {
                        return Err(ImportError::Zip {
                            path: path.display().to_string(),
                            source,
                        });
                    }
                };
                let mut buf = Vec::with_capacity(entry.size() as usize);
                entry
                    .read_to_end(&mut buf)
                    .map_err(|source| ImportError::Io { path: full, source })?;
                Ok(Some(buf))
            }
        }
    }

    /// Read a member that must be there.
    fn require(&mut self, name: &str) -> Result<Vec<u8>> {
        let display = self.display();
        self.get(name)?.ok_or_else(|| ImportError::Missing {
            path: display,
            missing: name.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// assembly
// ---------------------------------------------------------------------------

fn build(mut files: ModelFiles) -> Result<(AcousticModel, ImportReport)> {
    let meta_bytes = files.require("meta.json")?;
    let meta: serde_json::Value =
        serde_json::from_slice(&meta_bytes).map_err(|e| ImportError::BadContent {
            file: "meta.json".into(),
            msg: e.to_string(),
        })?;

    // --- phones -----------------------------------------------------------
    let phones_txt = files.require("phones.txt")?;
    let phones = parse_symbol_table(&phones_txt)?;

    let silence_phone_name = meta
        .get("optional_silence_phone")
        .and_then(|v| v.as_str())
        .unwrap_or("sil");
    let oov_phone_name = meta
        .get("oov_phone")
        .and_then(|v| v.as_str())
        .unwrap_or("spn");
    let mut silence_phones: Vec<PhoneId> = [silence_phone_name, oov_phone_name]
        .iter()
        .filter_map(|n| phones.id(n))
        .collect();
    silence_phones.sort_unstable();
    silence_phones.dedup();
    let silence_phone = phones.id(silence_phone_name).unwrap_or(1);

    // --- tree -------------------------------------------------------------
    let tree_bytes = files.require("tree")?;
    let ctx = {
        let mut r = KaldiReader::new(&tree_bytes).map_err(|source| ImportError::Parse {
            file: "tree".into(),
            source,
        })?;
        objects::read_context_dependency(&mut r).map_err(|source| ImportError::Parse {
            file: "tree".into(),
            source,
        })?
    };

    // --- final.mdl --------------------------------------------------------
    let mdl_bytes = files.require("final.mdl")?;
    let (kaldi_tm, am) = {
        let mut r = KaldiReader::new(&mdl_bytes).map_err(|source| ImportError::Parse {
            file: "final.mdl".into(),
            source,
        })?;
        let tm = objects::read_transition_model(&mut r).map_err(|source| ImportError::Parse {
            file: "final.mdl".into(),
            source,
        })?;
        let am = objects::read_am_diag_gmm(&mut r).map_err(|source| ImportError::Parse {
            file: "final.mdl".into(),
            source,
        })?;
        (tm, am)
    };

    // --- final.alimdl (optional) -----------------------------------------
    let am_si = match files.get("final.alimdl")? {
        None => None,
        Some(bytes) => {
            let mut r = KaldiReader::new(&bytes).map_err(|source| ImportError::Parse {
                file: "final.alimdl".into(),
                source,
            })?;
            // The alignment model carries its own transition model, which we
            // skip: only the GMMs differ, and the transition ids must be the
            // ones from final.mdl.
            objects::read_transition_model(&mut r).map_err(|source| ImportError::Parse {
                file: "final.alimdl".into(),
                source,
            })?;
            Some(
                objects::read_am_diag_gmm(&mut r).map_err(|source| ImportError::Parse {
                    file: "final.alimdl".into(),
                    source,
                })?,
            )
        }
    };

    // --- transition model -------------------------------------------------
    let mut tm = TransitionModel::new(&ctx, &kaldi_tm.topo);
    if tm.num_transition_states() != kaldi_tm.tuples.len() {
        return Err(ImportError::Inconsistent(format!(
            "rebuilt {} transition states from tree+topology, but final.mdl has {}",
            tm.num_transition_states(),
            kaldi_tm.tuples.len()
        )));
    }
    if tm.num_pdfs() != am.num_pdfs() {
        return Err(ImportError::Inconsistent(format!(
            "tree yields {} pdfs but final.mdl has {} GMMs",
            tm.num_pdfs(),
            am.num_pdfs()
        )));
    }
    let foreign = kaldi_log_probs_by_tuple(&kaldi_tm)?;
    tm.set_log_probs_by_tuple(&foreign)
        .map_err(ImportError::Inconsistent)?;

    // --- features ---------------------------------------------------------
    let feats = meta.get("features");
    let mfcc = mfcc_from_meta(feats);
    let uses_splices = feats
        .and_then(|f| f.get("uses_splices"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let uses_deltas = feats
        .and_then(|f| f.get("uses_deltas"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // lda.mat ships even when unused; parse it whenever present so a malformed
    // one is caught here, but only attach it when the pipeline splices.
    let lda_parsed = match files.get("lda.mat")? {
        None => None,
        Some(bytes) => {
            Some(
                objects::read_matrix_file(&bytes).map_err(|source| ImportError::Parse {
                    file: "lda.mat".into(),
                    source,
                })?,
            )
        }
    };
    let lda_shape = lda_parsed.as_ref().map(|m| (m.nrows(), m.ncols()));

    // meta.json's `uses_splices` is not reliable: MFA flips the flag on its
    // training worker (`acoustic_modeling/lda.py:376`) but exports the config
    // object's default, so an LDA+MLLT model can ship with `uses_splices:
    // false`. The GMM dimension is authoritative — an LDA model's Gaussians
    // live in the transform's output space, so when `lda.mat` maps
    // `num_ceps * (left+1+right)` inputs onto exactly `am.dim()` outputs, the
    // pipeline splices whatever the flag claims.
    let splice_ctx = (
        feats
            .and_then(|f| f.get("splice_left_context"))
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize,
        feats
            .and_then(|f| f.get("splice_right_context"))
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize,
    );
    let lda_fits_a_spliced_pipeline = lda_parsed.as_ref().is_some_and(|m| {
        let spliced = mfcc.num_ceps as usize * (splice_ctx.0 + 1 + splice_ctx.1);
        m.nrows() == am.dim() && (m.ncols() == spliced || m.ncols() == spliced + 1)
    });
    let uses_splices = uses_splices || lda_fits_a_spliced_pipeline;

    let (splice, lda, deltas) = if uses_splices {
        (Some(splice_ctx), lda_parsed, None)
    } else {
        (
            None,
            None,
            uses_deltas.then(crate::feat::DeltaOptions::default),
        )
    };

    // --- graph ------------------------------------------------------------
    let f32_meta = |key: &str, default: f32| {
        meta.get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(default)
    };
    let position_dependent = meta
        .get("dictionaries")
        .and_then(|d| d.get("position_dependent_phones"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let graph_opts = crate::hmm::GraphOptions {
        transition_scale: 1.0,
        self_loop_scale: 0.1,
        silence_phone,
        silence_prob: f32_meta("silence_probability", 0.5),
        initial_silence_prob: f32_meta("initial_silence_probability", 0.5),
        final_silence_correction: f32_meta("final_silence_correction", 1.0),
        final_non_silence_correction: f32_meta("final_non_silence_correction", 1.0),
        position_dependent,
    };

    // --- provenance -------------------------------------------------------
    let mut model_meta = BTreeMap::new();
    model_meta.insert("imported_from".into(), "montreal-forced-aligner".into());
    for key in [
        "version",
        "architecture",
        "train_date",
        "language",
        "phone_set_type",
    ] {
        if let Some(v) = meta.get(key).and_then(|v| v.as_str()) {
            model_meta.insert(format!("mfa_{key}"), v.to_string());
        }
    }
    if let Some(t) = meta.get("training") {
        for key in ["num_speakers", "num_utterances", "audio_duration"] {
            if let Some(v) = t.get(key) {
                model_meta.insert(format!("mfa_{key}"), v.to_string());
            }
        }
    }

    let uses_sat = feats
        .and_then(|f| f.get("uses_speaker_adaptation"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let am_si = if uses_sat { am_si } else { None };
    let fmllr = am_si
        .as_ref()
        .map(|_| crate::transform::FmllrOptions::default());

    let report = ImportReport {
        num_pdfs: am.num_pdfs(),
        num_gauss: am.num_gauss(),
        num_transition_ids: tm.num_transition_ids(),
        num_transition_states: tm.num_transition_states(),
        kaldi_num_transition_states: kaldi_tm.tuples.len(),
        feature_dim: am.dim(),
        num_phones: phones.len(),
        has_alignment_model: am_si.is_some(),
        lda_shape,
    };

    let model = AcousticModel {
        version: crate::model::FORMAT_VERSION,
        phones,
        silence_phones,
        position_dependent,
        mfcc,
        deltas,
        splice,
        lda,
        topo: kaldi_tm.topo,
        ctx,
        tm,
        am,
        am_si,
        fmllr,
        graph_opts,
        lexicon_probs: None,
        meta: model_meta,
    };

    Ok((model, report))
}

/// Group Kaldi's flat `log_probs` vector into the per-tuple slices
/// [`TransitionModel::set_log_probs_by_tuple`] wants.
///
/// Kaldi's transition ids run 1.. in tuple order, each tuple owning as many ids
/// as its topology state has transitions.
fn kaldi_log_probs_by_tuple(
    km: &objects::KaldiTransitionModel,
) -> Result<BTreeMap<Tuple, Vec<f32>>> {
    let mut out = BTreeMap::new();
    let mut tid = 1usize;
    for tuple in &km.tuples {
        let states = km.topo.try_for_phone(tuple.phone).ok_or_else(|| {
            ImportError::Inconsistent(format!("phone {} has no topology entry", tuple.phone))
        })?;
        let state = states.get(tuple.hmm_state).ok_or_else(|| {
            ImportError::Inconsistent(format!(
                "phone {} has no hmm-state {}",
                tuple.phone, tuple.hmm_state
            ))
        })?;
        let n = state.transitions.len();
        if tid + n > km.log_probs.len() {
            return Err(ImportError::Inconsistent(format!(
                "final.mdl's <LogProbs> holds {} entries, too few for its {} tuples",
                km.log_probs.len(),
                km.tuples.len()
            )));
        }
        out.insert(*tuple, km.log_probs[tid..tid + n].to_vec());
        tid += n;
    }
    Ok(out)
}

/// Parse a Kaldi/OpenFst symbol table: `<symbol>\t<id>` per line, ids ascending
/// from 0 with `<eps>` first, which is exactly how [`SymbolTable`] is indexed.
fn parse_symbol_table(bytes: &[u8]) -> Result<SymbolTable> {
    let text = std::str::from_utf8(bytes).map_err(|e| ImportError::BadContent {
        file: "phones.txt".into(),
        msg: e.to_string(),
    })?;
    let mut table = SymbolTable::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let (sym, id) =
            line.rsplit_once(char::is_whitespace)
                .ok_or_else(|| ImportError::BadContent {
                    file: "phones.txt".into(),
                    msg: format!(
                        "line {}: expected \"<symbol> <id>\", got {line:?}",
                        lineno + 1
                    ),
                })?;
        let id: u32 = id.trim().parse().map_err(|_| ImportError::BadContent {
            file: "phones.txt".into(),
            msg: format!("line {}: {id:?} is not an integer id", lineno + 1),
        })?;
        let assigned = table.add(sym.trim());
        if assigned != id {
            return Err(ImportError::BadContent {
                file: "phones.txt".into(),
                msg: format!(
                    "line {}: symbol {sym:?} has id {id}, but the table is at {assigned}; \
                     phones.txt must list ids densely from 0",
                    lineno + 1
                ),
            });
        }
    }
    Ok(table)
}

/// Build [`MfccOptions`] from meta.json's `features` block, leaving viter's
/// MFA-derived defaults in place for anything the file does not name.
fn mfcc_from_meta(feats: Option<&serde_json::Value>) -> crate::feat::MfccOptions {
    let mut o = crate::feat::MfccOptions::default();
    let Some(f) = feats else { return o };

    let num = |k: &str| f.get(k).and_then(|v| v.as_f64());
    let boolean = |k: &str| f.get(k).and_then(|v| v.as_bool());

    if let Some(v) = num("sample_frequency") {
        o.sample_rate = v as u32;
    }
    if let Some(v) = num("frame_length") {
        o.frame_length_ms = v as f32;
    }
    if let Some(v) = num("frame_shift") {
        o.frame_shift_ms = v as f32;
    }
    if let Some(v) = num("dither") {
        o.dither = v as f32;
    }
    if let Some(v) = num("preemphasis_coefficient") {
        o.preemph = v as f32;
    }
    if let Some(v) = boolean("snip_edges") {
        o.snip_edges = v;
    }
    if let Some(v) = num("num_mel_bins") {
        o.num_mel_bins = v as u32;
    }
    if let Some(v) = num("low_frequency") {
        o.low_freq = v as f32;
    }
    if let Some(v) = num("high_frequency") {
        o.high_freq = v as f32;
    }
    if let Some(v) = num("num_coefficients") {
        o.num_ceps = v as u32;
    }
    if let Some(v) = boolean("use_energy") {
        o.use_energy = v;
    }
    if let Some(v) = num("energy_floor") {
        o.energy_floor = v as f32;
    }
    if let Some(v) = num("cepstral_lifter") {
        o.cepstral_lifter = v as f32;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_symbol_table_in_file_order() {
        let t = parse_symbol_table("<eps>\t0\nsil\t1\nspn\t2\nae\t3\n".as_bytes()).unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(t.id("sil"), Some(1));
        assert_eq!(t.sym(3), "ae");
    }

    #[test]
    fn rejects_a_symbol_table_with_a_gap() {
        assert!(parse_symbol_table(b"<eps>\t0\nsil\t5\n").is_err());
    }

    #[test]
    fn meta_features_override_the_defaults() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"num_coefficients": 20, "use_energy": true}"#).unwrap();
        let o = mfcc_from_meta(Some(&v));
        assert_eq!(o.num_ceps, 20);
        assert!(o.use_energy);
        // Untouched fields keep viter's MFA-derived defaults.
        assert_eq!(o.num_mel_bins, 23);
    }
}
