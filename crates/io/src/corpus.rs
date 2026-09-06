//! Corpus scanning: pair audio files with transcripts, resolve words to phone ids.
//!
//! Mirrors MFA's corpus loading semantics (`plans/mfa/montreal_forced_aligner/corpus`)
//! and its dictionary parsing (`plans/mfa/montreal_forced_aligner/dictionary`).

use anyhow::{Context, Result, anyhow, bail};
use viter_kaldi::types::{
    POS_BEGIN, POS_END, POS_INTERNAL, POS_SINGLETON, PhoneId, Pronunciation, SymbolTable, Utterance,
    untag_phone,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Audio extensions recognised while scanning a corpus.
pub const AUDIO_EXTS: &[&str] = &["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "aif"];
/// Transcript extensions, in preference order.
pub const TRANSCRIPT_EXTS: &[&str] = &["lab", "txt", "textgrid"];

/// MFA's default punctuation set (`DEFAULT_PUNCTUATION`, `plans/mfa/montreal_forced_aligner/data.py:70`).
const DEFAULT_PUNCTUATION: &str =
    "、。।，？！!@<>→\"”()“„–,.:;—¿?¡：）|؟\\&%#*،~【】，…‥「」『』〝〟″⟨⟩♪・‚‘‹›«»～′$+=‘۔―";
/// MFA's `DEFAULT_CLITIC_MARKERS` (`data.py:78`) — kept inside words.
const CLITIC_MARKERS: &str = "'’‘";

// ---------------------------------------------------------------------------
// Dictionary
// ---------------------------------------------------------------------------

pub use crate::dict::{DictEntry, Dictionary};

use crate::files::{find_transcript, has_ext, read_transcript, speaker_name, utt_id};
pub(crate) use crate::files::read_text_file;

/// Normalize a dictionary headword: strip `(2)` variant suffix, then normalize like a
/// transcript word so lookups line up.
pub(crate) fn normalize_dict_key(raw: &str) -> String {
    let base = strip_variant_suffix(raw);
    normalize_word(base)
}

/// `WORD(2)` -> `WORD`.
pub(crate) fn strip_variant_suffix(w: &str) -> &str {
    if let Some(open) = w.rfind('(')
        && w.ends_with(')')
        && open + 1 < w.len() - 1
        && w[open + 1..w.len() - 1].chars().all(|c| c.is_ascii_digit())
    {
        return &w[..open];
    }
    w
}

/// MFA-style word normalization for dictionary lookup: lowercase, strip surrounding
/// punctuation, keep apostrophes (clitic markers) inside the word.
pub fn normalize_word(w: &str) -> String {
    let lowered = w.to_lowercase();
    let trimmed = lowered.trim_matches(|c: char| DEFAULT_PUNCTUATION.contains(c));
    // A word made purely of clitic markers is not a word.
    if trimmed.chars().all(|c| CLITIC_MARKERS.contains(c)) && !trimmed.is_empty() {
        return String::new();
    }
    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Where the speaker name comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeakerSource {
    /// Name of the directory containing the audio file (MFA's default layout).
    ParentDir,
    /// First `n` characters of the file stem.
    Prefix(usize),
    /// One speaker for the whole corpus.
    Single,
}

#[derive(Clone, Debug)]
pub struct CorpusOptions {
    pub dictionary: Option<PathBuf>,
    /// Phone used for out-of-vocabulary words. MFA: `spn`.
    pub oov_phone: String,
    /// Optional silence phone. MFA: `sil`.
    pub silence_phone: String,
    /// Append `_B _I _E _S` word-position tags to non-silence phones.
    pub position_dependent: bool,
    pub speaker_from: SpeakerSource,
}

impl Default for CorpusOptions {
    fn default() -> Self {
        Self {
            dictionary: None,
            oov_phone: "spn".to_string(),
            silence_phone: "sil".to_string(),
            position_dependent: true,
            speaker_from: SpeakerSource::ParentDir,
        }
    }
}

/// A scanned corpus with a phone symbol table shared by all utterances.
#[derive(Clone, Debug)]
pub struct Corpus {
    pub utts: Vec<Utterance>,
    pub speakers: Vec<String>,
    /// `<eps>`=0, then `sil`=1, `spn`=2, then the lexicon phones.
    pub phones: SymbolTable,
    pub silence_phones: Vec<PhoneId>,
    /// OOV word -> number of occurrences.
    pub oov_words: BTreeMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/// Scan `dir` recursively for audio files paired with a transcript of the same stem.
///
/// With a dictionary: transcript tokens are words looked up in the lexicon, keeping every
/// pronunciation listed for them; unknown words map to a single `oov_phone` pronunciation
/// and are counted in `Corpus::oov_words`.
/// Without a dictionary: tokens are phones (phoneme-string input). A `|` token, or a run of
/// two or more spaces, ends a word; with no separator anywhere every token is its own
/// one-phone word.
pub fn scan(dir: &Path, opts: &CorpusOptions) -> Result<Corpus> {
    if !dir.is_dir() {
        bail!("corpus path {} is not a directory", dir.display());
    }
    let dict = match &opts.dictionary {
        Some(p) => Some(Dictionary::load(p)?),
        None => None,
    };

    let mut builder = CorpusBuilder::new(opts, dict.as_ref());

    // Collect (audio, transcript) pairs, sorted for determinism.
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(dir).follow_links(true).sort_by_file_name() {
        let entry = entry.with_context(|| format!("walking {}", dir.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        if has_ext(entry.path(), AUDIO_EXTS) {
            files.push(entry.into_path());
        }
    }
    files.sort();

    for audio in files {
        let Some(text) = find_transcript(&audio)? else {
            tracing::warn!("no transcript found for {}", audio.display());
            continue;
        };
        let id = utt_id(dir, &audio);
        let speaker = speaker_name(dir, &audio, &opts.speaker_from);
        builder.push(id, speaker, audio, &text)?;
    }

    if builder.utts.is_empty() {
        bail!(
            "no audio/transcript pairs found under {} (looked for {:?} with {:?})",
            dir.display(),
            AUDIO_EXTS,
            TRANSCRIPT_EXTS
        );
    }
    Ok(builder.finish())
}

/// Build a one-utterance corpus from a single audio file plus inline transcript text.
///
/// If `phones` is given, that symbol table is used as-is and every phone the transcript
/// needs must already be in it (align against a trained model).
pub fn single(
    audio: &Path,
    transcript: &str,
    opts: &CorpusOptions,
    phones: Option<&SymbolTable>,
) -> Result<Corpus> {
    let dict = match &opts.dictionary {
        Some(p) => Some(Dictionary::load(p)?),
        None => None,
    };
    let mut builder = CorpusBuilder::new(opts, dict.as_ref());

    // A transcript that names an existing file is read from it.
    let text = {
        let as_path = Path::new(transcript);
        if !transcript.contains('\n') && as_path.is_file() {
            read_transcript(as_path)?
        } else {
            transcript.to_string()
        }
    };

    let parent = audio.parent().unwrap_or(Path::new("."));
    let id = audio
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "utt".to_string());
    let speaker = speaker_name(parent, audio, &opts.speaker_from);
    builder.push(id, speaker, audio.to_path_buf(), &text)?;

    let mut corpus = builder.finish();
    if let Some(model_phones) = phones {
        remap(&mut corpus, model_phones)?;
    }
    Ok(corpus)
}

/// Re-express the corpus's phone ids in a model's symbol table.
///
/// Unknown phones are an error, listing every missing symbol.
pub fn remap(corpus: &mut Corpus, model_phones: &SymbolTable) -> Result<()> {
    let mut missing: BTreeSet<String> = BTreeSet::new();
    let mut mapping: Vec<PhoneId> = Vec::with_capacity(corpus.phones.len());
    for id in 0..corpus.phones.len() as PhoneId {
        let sym = corpus.phones.sym(id);
        match model_phones.id(sym) {
            Some(new) => mapping.push(new),
            None => {
                missing.insert(sym.to_string());
                mapping.push(0);
            }
        }
    }
    if !missing.is_empty() {
        return Err(anyhow!(
            "phones missing from the model's symbol table: {}",
            missing.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    for utt in &mut corpus.utts {
        for word in &mut utt.prons {
            for pron in word.iter_mut() {
                for p in pron.phones.iter_mut() {
                    *p = mapping[*p as usize];
                }
            }
        }
    }
    corpus.silence_phones = corpus
        .silence_phones
        .iter()
        .map(|p| mapping[*p as usize])
        .collect();
    corpus.phones = model_phones.clone();
    Ok(())
}

/// Untag `AH_B` -> `AH` for output.
pub fn untag(phone: &str) -> &str {
    untag_phone(phone)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

struct CorpusBuilder<'a> {
    opts: &'a CorpusOptions,
    dict: Option<&'a Dictionary>,
    phones: SymbolTable,
    silence_phones: Vec<PhoneId>,
    speakers: Vec<String>,
    utts: Vec<Utterance>,
    oov_words: BTreeMap<String, usize>,
}

impl<'a> CorpusBuilder<'a> {
    fn new(opts: &'a CorpusOptions, dict: Option<&'a Dictionary>) -> Self {
        // `<eps>`=0, `sil`=1, `spn`=2 always come first, in that order.
        let mut phones = SymbolTable::new();
        let sil = phones.add(&opts.silence_phone);
        let spn = phones.add(&opts.oov_phone);
        // Pre-register lexicon phones so ids are corpus-content independent.
        if let Some(d) = dict {
            for p in d.phones() {
                add_phone_variants(&mut phones, &p, opts);
            }
        }
        Self {
            opts,
            dict,
            phones,
            silence_phones: vec![sil, spn],
            speakers: Vec::new(),
            utts: Vec::new(),
            oov_words: BTreeMap::new(),
        }
    }

    fn speaker_id(&mut self, name: &str) -> String {
        if !self.speakers.iter().any(|s| s == name) {
            self.speakers.push(name.to_string());
        }
        name.to_string()
    }

    fn push(&mut self, id: String, speaker: String, audio: PathBuf, text: &str) -> Result<()> {
        let groups = if self.dict.is_some() {
            // Dictionary mode: one group per whitespace token, each a written word.
            text.split_whitespace().map(|t| vec![t]).collect()
        } else {
            split_phone_words(text)
        };
        if groups.is_empty() {
            tracing::warn!("empty transcript for {}", audio.display());
            return Ok(());
        }
        let mut words: Vec<String> = Vec::with_capacity(groups.len());
        let mut prons: Vec<Vec<Pronunciation>> = Vec::with_capacity(groups.len());

        for toks in &groups {
            let entries: Vec<DictEntry> = match self.dict {
                Some(d) => match d.lookup(toks[0]) {
                    Some(es) => es.to_vec(),
                    None => {
                        if normalize_word(toks[0]).is_empty() {
                            // Pure punctuation: not a word at all.
                            continue;
                        }
                        *self.oov_words.entry(normalize_word(toks[0])).or_insert(0) += 1;
                        vec![DictEntry::plain(&[self.opts.oov_phone.clone()])]
                    }
                },
                // Phoneme-string mode: the tokens of the group are the word's phones.
                None => vec![DictEntry::plain(
                    &toks.iter().map(|t| untag_phone(t)).collect::<Vec<_>>(),
                )],
            };
            words.push(toks.join(" "));
            prons.push(entries.iter().map(|e| self.resolve_pron(e)).collect());
        }

        if words.is_empty() {
            tracing::warn!("transcript for {} had no usable words", audio.display());
            return Ok(());
        }

        let speaker = self.speaker_id(&speaker);
        self.utts.push(Utterance {
            id,
            speaker,
            audio,
            words,
            prons,
            text: text.trim().to_string(),
        });
        Ok(())
    }

    /// Apply position tags (if enabled) and intern the phones of one dictionary entry.
    /// Tagging is per pronunciation, so alternatives of different length tag correctly.
    fn resolve_pron(&mut self, entry: &DictEntry) -> Pronunciation {
        let n = entry.phones.len();
        let mut out = Vec::with_capacity(n);
        for (i, p) in entry.phones.iter().enumerate() {
            let base = untag_phone(p);
            let sym = if !self.opts.position_dependent || self.is_special(base) {
                base.to_string()
            } else if n == 1 {
                format!("{base}{POS_SINGLETON}")
            } else if i == 0 {
                format!("{base}{POS_BEGIN}")
            } else if i + 1 == n {
                format!("{base}{POS_END}")
            } else {
                format!("{base}{POS_INTERNAL}")
            };
            out.push(self.phones.add(&sym));
        }
        Pronunciation {
            phones: out,
            prob: entry.prob,
            silence_after_prob: entry.silence_after_prob,
            silence_before_correction: entry.silence_before_correction,
            non_silence_before_correction: entry.non_silence_before_correction,
        }
    }

    /// Silence and oov phones are never position-tagged.
    fn is_special(&self, base: &str) -> bool {
        base == self.opts.silence_phone || base == self.opts.oov_phone
    }

    fn finish(self) -> Corpus {
        Corpus {
            utts: self.utts,
            speakers: self.speakers,
            phones: self.phones,
            silence_phones: self.silence_phones,
            oov_words: self.oov_words,
        }
    }
}

/// Split a phoneme-string transcript into words.
///
/// A `|` token, or a run of two or more spaces (or a tab), ends a word; single spaces
/// separate the phones inside a word. With neither separator present anywhere, every token is
/// its own one-phone word (the original behaviour).
fn split_phone_words(text: &str) -> Vec<Vec<&str>> {
    // Chunks are the pieces left by the word separators; a `|` splits a chunk further.
    let mut chunks: Vec<&str> = Vec::new();
    let mut saw_separator = false;
    for line in text.lines() {
        for wide in line.split("  ").flat_map(|c| c.split('\t')) {
            if wide.trim().is_empty() {
                continue;
            }
            if !chunks.is_empty() {
                saw_separator = true;
            }
            chunks.push(wide);
        }
    }
    let mut words: Vec<Vec<&str>> = Vec::new();
    for chunk in chunks {
        let mut cur: Vec<&str> = Vec::new();
        for tok in chunk.split_whitespace() {
            if tok == "|" {
                saw_separator = true;
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            } else {
                cur.push(tok);
            }
        }
        if !cur.is_empty() {
            words.push(cur);
        }
    }
    if !saw_separator {
        // No separator anywhere: each token is its own one-phone word.
        return words.into_iter().flatten().map(|t| vec![t]).collect();
    }
    words
}

/// Register every position-tagged variant of a lexicon phone up front so that the
/// symbol table is a function of the dictionary, not of which words happened to appear.
fn add_phone_variants(phones: &mut SymbolTable, p: &str, opts: &CorpusOptions) {
    let base = untag_phone(p);
    if base == opts.silence_phone || base == opts.oov_phone {
        phones.add(base);
        return;
    }
    if opts.position_dependent {
        for suf in [POS_BEGIN, POS_END, POS_INTERNAL, POS_SINGLETON] {
            phones.add(&format!("{base}{suf}"));
        }
    } else {
        phones.add(base);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "viter-corpus-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Phone symbols of one word's chosen pronunciation.
    fn syms(c: &Corpus, u: &Utterance, w: usize, pron: usize) -> Vec<&'static str> {
        u.prons[w][pron]
            .phones
            .iter()
            .map(|p| Box::leak(c.phones.sym(*p).to_string().into_boxed_str()) as &'static str)
            .collect()
    }

    #[test]
    fn word_normalization() {
        assert_eq!(normalize_word("Hello,"), "hello");
        assert_eq!(normalize_word("\"quoted\"."), "quoted");
        assert_eq!(normalize_word("don't"), "don't");
        assert_eq!(normalize_word("..."), "");
        assert_eq!(normalize_word("O'NEILL"), "o'neill");
    }

    #[test]
    fn scan_with_dictionary_tags_positions() {
        let root = tmpdir("scan");
        let spk = root.join("spk1");
        std::fs::create_dir_all(&spk).unwrap();
        std::fs::write(spk.join("a.wav"), b"").unwrap();
        std::fs::write(spk.join("a.lab"), "hello a xyzzy").unwrap();
        let dictp = root.join("d.txt");
        std::fs::write(&dictp, "HELLO HH AH L OW\nHELLO(2) HH AH L\nA AH\n").unwrap();

        let opts = CorpusOptions {
            dictionary: Some(dictp),
            ..Default::default()
        };
        let c = scan(&root, &opts).unwrap();
        assert_eq!(c.utts.len(), 1);
        assert_eq!(c.speakers, vec!["spk1".to_string()]);
        assert_eq!(c.phones.sym(1), "sil");
        assert_eq!(c.phones.sym(2), "spn");
        assert_eq!(c.silence_phones, vec![1, 2]);

        let u = &c.utts[0];
        assert_eq!(u.words, vec!["hello", "a", "xyzzy"]);
        // Both pronunciations survive, each tagged for its own length.
        assert_eq!(u.prons[0].len(), 2);
        assert_eq!(syms(&c, u, 0, 0), vec!["HH_B", "AH_I", "L_I", "OW_E"]);
        assert_eq!(syms(&c, u, 0, 1), vec!["HH_B", "AH_I", "L_E"]);
        // Single-phone word -> _S
        assert_eq!(syms(&c, u, 1, 0), vec!["AH_S"]);
        // OOV -> one spn pronunciation, untagged, no probabilities.
        assert_eq!(u.prons[2].len(), 1);
        assert_eq!(syms(&c, u, 2, 0), vec!["spn"]);
        assert_eq!(u.prons[2][0].prob, None);
        assert_eq!(c.oov_words.get("xyzzy"), Some(&1));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scan_phoneme_string_mode() {
        let root = tmpdir("phon");
        std::fs::write(root.join("b.wav"), b"").unwrap();
        std::fs::write(root.join("b.txt"), "AA B sil").unwrap();
        let opts = CorpusOptions {
            speaker_from: SpeakerSource::Single,
            ..Default::default()
        };
        let c = scan(&root, &opts).unwrap();
        let u = &c.utts[0];
        // No separator: three one-phone words.
        assert_eq!(u.prons.len(), 3);
        assert!(u.prons.iter().all(|p| p.len() == 1));
        assert_eq!(syms(&c, u, 0, 0), vec!["AA_S"]);
        assert_eq!(syms(&c, u, 2, 0), vec!["sil"]);
        assert_eq!(u.speaker, "speaker");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn phone_words_split_on_pipe() {
        assert_eq!(
            split_phone_words("HH AH | L OW | B"),
            vec![vec!["HH", "AH"], vec!["L", "OW"], vec!["B"]]
        );
    }

    #[test]
    fn phone_words_split_on_double_space() {
        assert_eq!(
            split_phone_words("HH AH  L OW   B"),
            vec![vec!["HH", "AH"], vec!["L", "OW"], vec!["B"]]
        );
        // No separator at all: every token is its own word.
        assert_eq!(split_phone_words("HH AH L"), vec![vec!["HH"], vec!["AH"], vec!["L"]]);
    }

    #[test]
    fn scan_phoneme_string_with_separators() {
        let root = tmpdir("phonsep");
        std::fs::write(root.join("p.wav"), b"").unwrap();
        std::fs::write(root.join("p.txt"), "HH AH | L OW  B").unwrap();
        let opts = CorpusOptions {
            speaker_from: SpeakerSource::Single,
            ..Default::default()
        };
        let c = scan(&root, &opts).unwrap();
        let u = &c.utts[0];
        assert_eq!(u.words, vec!["HH AH", "L OW", "B"]);
        assert_eq!(syms(&c, u, 0, 0), vec!["HH_B", "AH_E"]);
        assert_eq!(syms(&c, u, 1, 0), vec!["L_B", "OW_E"]);
        assert_eq!(syms(&c, u, 2, 0), vec!["B_S"]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn remap_reports_missing_phones() {
        let root = tmpdir("remap");
        std::fs::write(root.join("c.wav"), b"").unwrap();
        std::fs::write(root.join("c.txt"), "AA").unwrap();
        let opts = CorpusOptions {
            position_dependent: false,
            ..Default::default()
        };
        let mut c = scan(&root, &opts).unwrap();

        let mut model = SymbolTable::new();
        model.add("sil");
        model.add("spn");
        model.add("ZZ");
        assert!(remap(&mut c, &model).is_err());

        model.add("AA");
        remap(&mut c, &model).unwrap();
        assert_eq!(c.phones.sym(c.utts[0].prons[0][0].phones[0]), "AA");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn untag_roundtrip() {
        assert_eq!(untag("AH_B"), "AH");
        assert_eq!(untag("sil"), "sil");
    }

}
