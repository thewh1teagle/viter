//! The `.viter` acoustic model file: one postcard-encoded blob holding
//! everything alignment needs — the phone table, the feature pipeline
//! configuration, the HMM topology and tree, the transition model and the
//! GMMs.
//!
//! Layout on disk:
//!
//! ```text
//! offset 0  : b"VITR"                (4 bytes, magic)
//! offset 4  : u32 little-endian      (format version)
//! offset 8  : postcard(AcousticModel)
//! ```

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::types::{PhoneId, SymbolTable};

/// Magic bytes at the head of every `.viter` file.
pub const MAGIC: &[u8; 4] = b"VITR";

/// Current on-disk format version. Bump on any breaking layout change.
pub const FORMAT_VERSION: u32 = 2;

/// Errors from reading or writing a `.viter` file.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("i/o error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },

    #[error("{path} is not a viter model file (bad magic {found:02x?}, expected {MAGIC:02x?})")]
    BadMagic { path: String, found: [u8; 4] },

    #[error("{path} is truncated: {len} bytes, need at least {need}")]
    Truncated { path: String, len: usize, need: usize },

    #[error(
        "{path} has format version {found}, but this build reads version {expected}"
    )]
    VersionMismatch {
        path: String,
        found: u32,
        expected: u32,
    },

    #[error("failed to decode {path}: {source}")]
    Decode {
        path: String,
        #[source]
        source: postcard::Error,
    },

    #[error("failed to encode model: {source}")]
    Encode {
        #[source]
        source: postcard::Error,
    },
}

/// A trained acoustic model, self-contained: everything needed to align audio.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticModel {
    /// Format version; equals [`FORMAT_VERSION`] for models this build writes.
    pub version: u32,
    /// Phone symbols, position-tagged when `position_dependent` is set.
    pub phones: SymbolTable,
    /// Phones treated as silence (silence, spoken noise, ...).
    pub silence_phones: Vec<PhoneId>,
    /// Whether phone symbols carry `_B`/`_I`/`_E`/`_S` word-position suffixes.
    pub position_dependent: bool,

    /// MFCC extraction settings the model was trained with.
    pub mfcc: crate::feat::MfccOptions,
    /// Delta/delta-delta settings — `Some` for mono and tri models.
    pub deltas: Option<crate::feat::DeltaOptions>,
    /// Left/right splice context — `Some` for lda and sat models.
    pub splice: Option<(usize, usize)>,
    /// Composed LDA+MLLT transform applied to spliced features.
    pub lda: Option<ndarray::Array2<f32>>,

    pub topo: crate::hmm::HmmTopology,
    pub ctx: crate::hmm::ContextDependency,
    pub tm: crate::hmm::TransitionModel,

    /// The speaker-adapted model (Kaldi's `final.mdl`).
    pub am: crate::gmm::AmDiagGmm,
    /// The speaker-independent alignment model (Kaldi's `final.alimdl`), used
    /// for the first fMLLR pass. `Some` only for SAT models.
    pub am_si: Option<crate::gmm::AmDiagGmm>,
    /// fMLLR estimation settings — `Some` when `am_si` is present.
    pub fmllr: Option<crate::transform::FmllrOptions>,

    /// Graph construction settings (self-loop scale, transition scale, ...).
    pub graph_opts: crate::hmm::GraphOptions,

    /// Free-form provenance: `trained_on`, `date`, `viter_version`,
    /// `num_utts`, and anything else a training run wants to record.
    pub meta: std::collections::BTreeMap<String, String>,
}

impl AcousticModel {
    /// Serialize to `path`, creating parent directories as needed.
    pub fn save(&self, path: &Path) -> Result<(), ModelError> {
        let body = postcard::to_allocvec(self).map_err(|source| ModelError::Encode { source })?;

        let mut bytes = Vec::with_capacity(8 + body.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&body);

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|source| ModelError::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
        }
        fs::write(path, &bytes).map_err(|source| ModelError::Io {
            path: path.display().to_string(),
            source,
        })
    }

    /// Read a model back, verifying magic bytes and format version.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let bytes = fs::read(path).map_err(|source| ModelError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_bytes(&bytes, &path.display().to_string())
    }

    /// Parse an in-memory `.viter` image. `path` is used only for error text.
    pub fn from_bytes(bytes: &[u8], path: &str) -> Result<Self, ModelError> {
        if bytes.len() < 8 {
            return Err(ModelError::Truncated {
                path: path.to_string(),
                len: bytes.len(),
                need: 8,
            });
        }
        if &bytes[0..4] != MAGIC {
            let mut found = [0u8; 4];
            found.copy_from_slice(&bytes[0..4]);
            return Err(ModelError::BadMagic {
                path: path.to_string(),
                found,
            });
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != FORMAT_VERSION {
            return Err(ModelError::VersionMismatch {
                path: path.to_string(),
                found: version,
                expected: FORMAT_VERSION,
            });
        }
        postcard::from_bytes(&bytes[8..]).map_err(|source| ModelError::Decode {
            path: path.to_string(),
            source,
        })
    }

    /// Serialize to an in-memory `.viter` image (magic + version + postcard).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ModelError> {
        let body = postcard::to_allocvec(self).map_err(|source| ModelError::Encode { source })?;
        let mut bytes = Vec::with_capacity(8 + body.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }

    /// Dimension of the features this model scores, i.e. the dimension of
    /// `am`'s Gaussians after the whole feature pipeline has run.
    pub fn feature_dim(&self) -> usize {
        self.am.dim()
    }

    /// Whether this model expects speaker adaptation (a first alignment pass
    /// with `am_si`, then fMLLR, then a second pass with `am`).
    pub fn is_speaker_adapted(&self) -> bool {
        self.am_si.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_and_version_are_the_documented_prefix() {
        assert_eq!(MAGIC, b"VITR");
        assert_eq!(FORMAT_VERSION, 2);
    }

    #[test]
    fn short_input_is_truncated_error() {
        let err = AcousticModel::from_bytes(b"CRA", "x.viter").unwrap_err();
        assert!(matches!(err, ModelError::Truncated { len: 3, need: 8, .. }));
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let mut bytes = b"NOPE".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(0);
        let err = AcousticModel::from_bytes(&bytes, "x.viter").unwrap_err();
        match err {
            ModelError::BadMagic { found, .. } => assert_eq!(&found, b"NOPE"),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn wrong_version_is_rejected_before_decoding() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&99u32.to_le_bytes());
        // Deliberately garbage body: version must be checked first.
        bytes.extend_from_slice(&[0xff; 16]);
        let err = AcousticModel::from_bytes(&bytes, "x.viter").unwrap_err();
        match err {
            ModelError::VersionMismatch { found, expected, .. } => {
                assert_eq!(found, 99);
                assert_eq!(expected, FORMAT_VERSION);
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn corrupt_body_with_valid_header_is_a_decode_error() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&[0xff; 4]);
        let err = AcousticModel::from_bytes(&bytes, "x.viter").unwrap_err();
        assert!(matches!(err, ModelError::Decode { .. }));
    }

    #[test]
    fn error_messages_name_the_file() {
        let err = AcousticModel::from_bytes(b"XXXXyyyy", "/tmp/model.viter").unwrap_err();
        assert!(err.to_string().contains("/tmp/model.viter"));
    }
}
