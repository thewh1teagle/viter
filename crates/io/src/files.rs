//! Locating and reading the files a corpus is made of: audio/transcript pairing,
//! text decoding, utterance ids and speaker names.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use crate::corpus::{SpeakerSource, TRANSCRIPT_EXTS};

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

pub(crate) fn has_ext(p: &Path, exts: &[&str]) -> bool {
    match p.extension().and_then(|e| e.to_str()) {
        Some(e) => {
            let e = e.to_lowercase();
            exts.iter().any(|x| *x == e)
        }
        None => false,
    }
}

/// Locate a transcript next to `audio`: `x.lab`, `x.txt` or `x.TextGrid`.
pub(crate) fn find_transcript(audio: &Path) -> Result<Option<String>> {
    for ext in TRANSCRIPT_EXTS {
        for cand in ext_candidates(audio, ext) {
            if cand.is_file() {
                return Ok(Some(read_transcript(&cand)?));
            }
        }
    }
    Ok(None)
}

/// `x.wav` + "txt" -> [`x.txt`, `x.TXT`, `x.TextGrid`-style casings].
fn ext_candidates(audio: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v = vec![
        audio.with_extension(ext),
        audio.with_extension(ext.to_uppercase()),
    ];
    if ext == "textgrid" {
        v.push(audio.with_extension("TextGrid"));
        v.push(audio.with_extension("Textgrid"));
    }
    v
}

/// Read a transcript file; a `.TextGrid` contributes the joined non-empty texts of its
/// first interval tier (MFA-style corpora keep the transcript in a TextGrid).
pub(crate) fn read_transcript(path: &Path) -> Result<String> {
    if has_ext(path, &["textgrid"]) {
        let tg = crate::textgrid::TextGrid::read(path)?;
        let tier = tg
            .tiers
            .first()
            .ok_or_else(|| anyhow!("TextGrid {} has no tiers", path.display()))?;
        let text = tier
            .intervals
            .iter()
            .map(|i| i.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        return Ok(text);
    }
    read_text_file(path)
}

/// Read a text file, decoding UTF-8/UTF-16 by BOM sniffing (falls back to UTF-8 lossy).
pub(crate) fn read_text_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(crate::textgrid::decode_bytes(&bytes))
}

/// Utterance id: path relative to the corpus root, without extension, with `/`
/// separators, so outputs mirror the corpus tree (MFA does the same).
pub(crate) fn utt_id(root: &Path, audio: &Path) -> String {
    let rel = audio.strip_prefix(root).unwrap_or(audio);
    let stem = rel.with_extension("");
    let s = stem.to_string_lossy().replace('\\', "/");
    if s.is_empty() { "utt".to_string() } else { s }
}

pub(crate) fn speaker_name(root: &Path, audio: &Path, source: &SpeakerSource) -> String {
    let stem = audio
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match source {
        SpeakerSource::Single => "speaker".to_string(),
        SpeakerSource::Prefix(n) => {
            let n = (*n).min(stem.chars().count());
            let p: String = stem.chars().take(n).collect();
            if p.is_empty() { stem } else { p }
        }
        SpeakerSource::ParentDir => {
            let parent = audio.parent();
            match parent {
                // A file sitting directly in the corpus root has no speaker directory.
                Some(p) if p != root => p
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "speaker".to_string()),
                _ => "speaker".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_prefix() {
        let name = speaker_name(
            Path::new("/c"),
            Path::new("/c/LJ001-0001.wav"),
            &SpeakerSource::Prefix(5),
        );
        assert_eq!(name, "LJ001");
    }
}
