//! MFA/CMUdict pronunciation dictionary parsing.
//!
//! Mirrors `plans/kalpy/kalpy/fstext/lexicon.py:40-95`: every pronunciation listed for a word
//! is kept, in file order, with MFA's optional probability columns.

use anyhow::{Context, Result, bail};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::corpus::{normalize_dict_key, normalize_word, read_text_file};

/// One dictionary entry: the phone *strings* of a pronunciation plus MFA's optional
/// probability columns. Phones become ids (and get position tags) in `CorpusBuilder`.
#[derive(Clone, Debug, PartialEq)]
pub struct DictEntry {
    pub phones: Vec<String>,
    pub prob: Option<f32>,
    pub silence_after_prob: Option<f32>,
    pub silence_before_correction: Option<f32>,
    pub non_silence_before_correction: Option<f32>,
}

impl DictEntry {
    /// A pronunciation with no probability columns.
    pub fn plain<P: AsRef<str>>(phones: &[P]) -> Self {
        Self {
            phones: phones.iter().map(|p| p.as_ref().to_string()).collect(),
            prob: None,
            silence_after_prob: None,
            silence_before_correction: None,
            non_silence_before_correction: None,
        }
    }
}

/// Pronunciation dictionary: word -> every pronunciation listed for it, in file order.
#[derive(Clone, Debug, Default)]
pub struct Dictionary {
    map: HashMap<String, Vec<DictEntry>>,
}

impl Dictionary {
    /// Load an MFA/CMUdict-format dictionary.
    ///
    /// Each line is `word [prob [sil_after [sil_before_corr non_sil_before_corr]]] p1 p2 ...`.
    /// The numeric columns are optional and only recognised in the counts MFA recognises
    /// (1, 2 or 4), matching `plans/kalpy/kalpy/fstext/lexicon.py:60-95`: a field counts as a
    /// probability only if it looks like `\d+.\d+`, so a phone named `0` stays a phone.
    /// `word(2)` variant suffixes are stripped, so every variant lands on the same word.
    /// Lookup is case-insensitive; all pronunciations are kept, duplicates dropped.
    pub fn load(path: &Path) -> Result<Self> {
        let text = read_text_file(path)
            .with_context(|| format!("reading dictionary {}", path.display()))?;
        let mut map: HashMap<String, Vec<DictEntry>> = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            let Some((raw_word, rest)) = fields.split_first() else {
                continue;
            };
            let word = normalize_dict_key(raw_word);
            if word.is_empty() {
                continue;
            }
            let Some(entry) = parse_entry(rest) else {
                continue;
            };
            push_entry(&mut map, word, entry);
        }
        if map.is_empty() {
            bail!("dictionary {} contained no entries", path.display());
        }
        Ok(Self { map })
    }

    /// Build a dictionary directly from entries (testing / programmatic use). Repeating a
    /// word adds another pronunciation, as in a file.
    pub fn from_entries<I, W, P>(entries: I) -> Self
    where
        I: IntoIterator<Item = (W, Vec<P>)>,
        W: AsRef<str>,
        P: AsRef<str>,
    {
        let mut map: HashMap<String, Vec<DictEntry>> = HashMap::new();
        for (w, ps) in entries {
            let key = normalize_dict_key(w.as_ref());
            if key.is_empty() || ps.is_empty() {
                continue;
            }
            push_entry(&mut map, key, DictEntry::plain(&ps));
        }
        Self { map }
    }

    /// Look up a word; the word is normalized (lowercased, punctuation stripped) first.
    /// Returns every pronunciation, in dictionary order.
    pub fn lookup(&self, w: &str) -> Option<&[DictEntry]> {
        let key = normalize_word(w);
        if key.is_empty() {
            return None;
        }
        self.map.get(&key).map(|v| v.as_slice())
    }

    /// Every distinct phone symbol used by the dictionary.
    pub fn phones(&self) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        for entries in self.map.values() {
            for e in entries {
                for p in &e.phones {
                    set.insert(p.clone());
                }
            }
        }
        set
    }

    /// Number of distinct words.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Append a pronunciation to a word, skipping an identical phone sequence (keep first).
fn push_entry(map: &mut HashMap<String, Vec<DictEntry>>, word: String, entry: DictEntry) {
    let v = map.entry(word).or_default();
    if v.iter().any(|e| e.phones == entry.phones) {
        return;
    }
    v.push(entry);
}

/// Split the post-word fields into MFA's optional probability columns and the phones.
///
/// MFA consumes 1, 2 or 4 leading numeric columns (never 3): prob; prob + silence_after;
/// prob + silence_after + silence_before_correction + non_silence_before_correction.
fn parse_entry(fields: &[&str]) -> Option<DictEntry> {
    let mut n = 0;
    while n < 4 && n < fields.len() && is_prob_field(fields[n]) {
        n += 1;
    }
    // Three numeric columns is not a shape MFA emits: treat the third as a phone.
    if n == 3 {
        n = 2;
    }
    // Every numeric column must leave at least one phone behind.
    while n > 0 && n >= fields.len() {
        n -= 1;
    }
    let nums: Vec<f32> = fields[..n].iter().filter_map(|f| f.parse().ok()).collect();
    let phones: Vec<String> = fields[n..].iter().map(|s| s.to_string()).collect();
    if phones.is_empty() {
        return None;
    }
    Some(DictEntry {
        phones,
        prob: nums.first().copied(),
        silence_after_prob: nums.get(1).copied(),
        silence_before_correction: nums.get(2).copied(),
        non_silence_before_correction: nums.get(3).copied(),
    })
}

/// MFA's `prob_pattern`, `r"\b\d+\.\d+\b"`: digits, a dot, digits. A bare integer or a
/// phone symbol is not a probability column.
fn is_prob_field(f: &str) -> bool {
    match f.split_once('.') {
        Some((a, b)) => {
            !a.is_empty()
                && !b.is_empty()
                && a.bytes().all(|c| c.is_ascii_digit())
                && b.bytes().all(|c| c.is_ascii_digit())
        }
        None => false,
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::strip_variant_suffix;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "viter-dict-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn variant_suffix_and_prob_fields() {
        assert_eq!(strip_variant_suffix("WORD(2)"), "WORD");
        assert_eq!(strip_variant_suffix("WORD"), "WORD");
        assert_eq!(strip_variant_suffix("()"), "()");
        assert!(is_prob_field("0.5"));
        assert!(is_prob_field("1.64"));
        // MFA's pattern needs a decimal point: a bare integer is a phone, not a column.
        assert!(!is_prob_field("1"));
        assert!(!is_prob_field("AH0"));
        assert!(!is_prob_field("0."));
    }

    #[test]
    fn dictionary_keeps_all_pronunciations() {
        let d = tmpdir("dict");
        let p = d.join("d.txt");
        std::fs::write(
            &p,
            "HELLO\tHH AH0 L OW1\nHELLO(2) HH EH0 L OW1\nHELLO(3) HH AH0 L OW1\nWORLD 1.0 0.5 W ER1 L D\n",
        )
        .unwrap();
        let dict = Dictionary::load(&p).unwrap();
        // Both variants land on the same word, in file order; the duplicate is dropped.
        let hello = dict.lookup("hello").unwrap();
        assert_eq!(hello.len(), 2);
        assert_eq!(hello[0].phones, ["HH", "AH0", "L", "OW1"]);
        assert_eq!(hello[1].phones, ["HH", "EH0", "L", "OW1"]);
        assert_eq!(hello[0].prob, None);

        let world = dict.lookup("World!").unwrap();
        assert_eq!(world.len(), 1);
        assert_eq!(world[0].phones, ["W", "ER1", "L", "D"]);
        assert_eq!(world[0].prob, Some(1.0));
        assert_eq!(world[0].silence_after_prob, Some(0.5));
        assert_eq!(world[0].silence_before_correction, None);

        assert!(dict.lookup("nope").is_none());
        assert!(dict.phones().contains("ER1"));
        assert!(dict.phones().contains("EH0"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn dictionary_four_column_probabilities() {
        let d = tmpdir("dict4");
        let p = d.join("d.txt");
        // The real shape of data/dict/ljspeech_ipa.dict.
        std::fs::write(&p, "a\u{26a}\t0.99\t0.08\t1.64\t0.79\ta\u{26a}\n<unk>\t0.99\t0.22\t1.0\t1.0\tspn\n")
            .unwrap();
        let dict = Dictionary::load(&p).unwrap();
        let e = &dict.lookup("a\u{26a}").unwrap()[0];
        assert_eq!(e.phones, ["a\u{26a}"]);
        assert_eq!(e.prob, Some(0.99));
        assert_eq!(e.silence_after_prob, Some(0.08));
        assert_eq!(e.silence_before_correction, Some(1.64));
        assert_eq!(e.non_silence_before_correction, Some(0.79));
        assert_eq!(dict.lookup("<unk>").unwrap()[0].phones, ["spn"]);
        std::fs::remove_dir_all(&d).ok();
    }

}

