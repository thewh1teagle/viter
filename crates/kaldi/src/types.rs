//! Shared small types used across every module and crate. See plans/CONTRACTS.md.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Feature matrix, `[frames, dim]`, row-major.
pub type Feats = ndarray::Array2<f32>;

/// Phone symbol id. 0 is `<eps>`; silence phones come first after that.
pub type PhoneId = u32;
/// Index of a pdf (tied HMM state distribution) in the acoustic model.
pub type PdfId = u32;
/// Kaldi-style transition id, 1-based. 0 is reserved for epsilon / non-emitting.
pub type TransitionId = u32;
/// Index into `Utterance::words`.
pub type WordId = u32;

/// Bidirectional phone string <-> id table. Id 0 is always `<eps>`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymbolTable {
    syms: Vec<String>,
    ids: HashMap<String, PhoneId>,
}

impl SymbolTable {
    pub fn new() -> Self {
        let mut t = Self::default();
        t.add("<eps>");
        t
    }
    /// Add a symbol, returning its id (existing id if already present).
    pub fn add(&mut self, s: &str) -> PhoneId {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.syms.len() as PhoneId;
        self.syms.push(s.to_string());
        self.ids.insert(s.to_string(), id);
        id
    }
    pub fn id(&self, s: &str) -> Option<PhoneId> {
        self.ids.get(s).copied()
    }
    pub fn sym(&self, id: PhoneId) -> &str {
        &self.syms[id as usize]
    }
    pub fn len(&self) -> usize {
        self.syms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.syms.is_empty()
    }
    /// All ids except `<eps>`.
    pub fn phone_ids(&self) -> impl Iterator<Item = PhoneId> + '_ {
        (1..self.syms.len() as PhoneId).into_iter()
    }
    pub fn symbols(&self) -> &[String] {
        &self.syms
    }
}

/// One pronunciation of a word, with MFA's per-pronunciation probabilities
/// (dictionary columns `prob silence_after silence_before_corr non_silence_before_corr`).
/// `None` means "not given": the graph then uses its global defaults, exactly
/// like MFA's lexicon (`plans/kalpy/kalpy/fstext/lexicon.py:498-560`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pronunciation {
    /// Phones, position-tagged if the corpus was built that way.
    pub phones: Vec<PhoneId>,
    /// Pronunciation probability; graph cost is `|ln p|` with p floored at 0.01.
    pub prob: Option<f32>,
    /// P(silence follows this word); cost `-ln p` on the silence branch after the word,
    /// `-ln(1-p)` on the no-silence branch. Default: global `silence_prob`.
    pub silence_after_prob: Option<f32>,
    /// Multiplies the probability of a silence *before* this word: cost `-ln c` added to
    /// the incoming silence branch. Default 1.0 (cost 0).
    pub silence_before_correction: Option<f32>,
    /// Same for the no-silence branch before the word. Default 1.0.
    pub non_silence_before_correction: Option<f32>,
}

impl Pronunciation {
    pub fn plain(phones: Vec<PhoneId>) -> Self {
        Self {
            phones,
            prob: None,
            silence_after_prob: None,
            silence_before_correction: None,
            non_silence_before_correction: None,
        }
    }
}

/// One audio file with its transcript, resolved to phone ids.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Utterance {
    /// Unique id, typically the relative path without extension.
    pub id: String,
    pub speaker: String,
    pub audio: PathBuf,
    /// Words as written in the transcript, in order.
    pub words: Vec<String>,
    /// Per word, its candidate pronunciations (at least one). The aligner picks one
    /// per utterance, like MFA's lexicon fan-out.
    pub prons: Vec<Vec<Pronunciation>>,
    /// Raw transcript text.
    pub text: String,
}

impl Utterance {
    /// First pronunciation of every word: the phone sequence a caller that does
    /// not care about alternatives can use.
    pub fn first_phones(&self) -> Vec<Vec<PhoneId>> {
        self.prons.iter().map(|p| p[0].phones.clone()).collect()
    }
}

/// Frame-level Viterbi result for one utterance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alignment {
    pub utt: String,
    /// One transition id per frame.
    pub tids: Vec<TransitionId>,
    /// Word ids emitted along the best path, in order (one per word in the graph).
    pub words: Vec<WordId>,
    /// Chosen pronunciation index per word, parallel to `words`.
    pub prons: Vec<u32>,
    /// Total log-likelihood of the path (acoustic + graph, unscaled by acoustic_scale).
    pub loglike: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhoneInterval {
    pub phone: PhoneId,
    pub start_frame: u32,
    /// Exclusive.
    pub end_frame: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WordInterval {
    pub word: WordId,
    /// Which pronunciation of the word was aligned.
    pub pron: u32,
    pub start_frame: u32,
    /// Exclusive.
    pub end_frame: u32,
}

/// Alignment converted to phone and word intervals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntervalAlignment {
    pub utt: String,
    pub frame_shift_s: f32,
    pub phones: Vec<PhoneInterval>,
    pub words: Vec<WordInterval>,
}

impl IntervalAlignment {
    pub fn num_frames(&self) -> u32 {
        self.phones.last().map(|p| p.end_frame).unwrap_or(0)
    }
    pub fn duration_s(&self) -> f32 {
        self.num_frames() as f32 * self.frame_shift_s
    }
}

/// Word-position suffixes for position-dependent phones (MFA default).
pub const POS_BEGIN: &str = "_B";
pub const POS_END: &str = "_E";
pub const POS_INTERNAL: &str = "_I";
pub const POS_SINGLETON: &str = "_S";

/// Strip a word-position suffix if present.
pub fn untag_phone(s: &str) -> &str {
    for suf in [POS_BEGIN, POS_END, POS_INTERNAL, POS_SINGLETON] {
        if let Some(base) = s.strip_suffix(suf) {
            return base;
        }
    }
    s
}
