//! CTM (Conversation Time Marked) output.
//!
//! Kaldi convention (`plans/kaldi/src` `nbest-to-ctm`): one line per token,
//! `<utterance-id> <channel> <start-seconds> <duration-seconds> <label>`, with the
//! channel always `1` and times printed with two decimals. MFA has no CTM writer of its
//! own, so the Kaldi format is used; phone labels are untagged (`AH_B` -> `AH`) as MFA
//! does for all exported labels.

use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::path::Path;
use viter_kaldi::types::{IntervalAlignment, SymbolTable, untag_phone};

/// Number of decimals Kaldi's CTM writer uses for times.
const TIME_DECIMALS: usize = 2;

/// Write a combined CTM containing the phone rows of every alignment.
///
/// Each entry pairs an alignment with that utterance's word strings; a companion
/// `.words.ctm` file is written next to `path` holding the word-level rows, matching the
/// usual Kaldi split into phone and word CTMs while keeping one call site.
pub fn write_ctm(
    path: &Path,
    alis: &[(IntervalAlignment, &[String])],
    phones: &SymbolTable,
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let mut phone_out = String::new();
    let mut word_out = String::new();
    for (ali, words) in alis {
        phone_out.push_str(&phone_ctm(ali, phones));
        word_out.push_str(&word_ctm(ali, words));
    }

    std::fs::write(path, &phone_out).with_context(|| format!("writing {}", path.display()))?;

    let words_path = words_path_for(path);
    std::fs::write(&words_path, &word_out)
        .with_context(|| format!("writing {}", words_path.display()))?;
    Ok(())
}

/// `out/aligned.ctm` -> `out/aligned.words.ctm`.
fn words_path_for(path: &Path) -> std::path::PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "alignment".to_string());
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_else(|| "ctm".to_string());
    let parent = path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{stem}.words.{ext}"))
}

/// Phone-level CTM rows for one utterance, labels untagged.
pub fn phone_ctm(ali: &IntervalAlignment, phones: &SymbolTable) -> String {
    let shift = ali.frame_shift_s as f64;
    let mut s = String::new();
    for p in &ali.phones {
        let start = p.start_frame as f64 * shift;
        let dur = (p.end_frame - p.start_frame) as f64 * shift;
        write_row(
            &mut s,
            &ali.utt,
            start,
            dur,
            untag_phone(phones.sym(p.phone)),
        );
    }
    s
}

/// Word-level CTM rows for one utterance. Words with no matching string are skipped.
pub fn word_ctm(ali: &IntervalAlignment, words: &[String]) -> String {
    let shift = ali.frame_shift_s as f64;
    let mut s = String::new();
    for w in &ali.words {
        let Some(label) = words.get(w.word as usize) else {
            continue;
        };
        if label.is_empty() {
            continue;
        }
        let start = w.start_frame as f64 * shift;
        let dur = (w.end_frame - w.start_frame) as f64 * shift;
        write_row(&mut s, &ali.utt, start, dur, label);
    }
    s
}

fn write_row(out: &mut String, utt: &str, start: f64, dur: f64, label: &str) {
    // Writing into a String cannot fail.
    let _ = writeln!(
        out,
        "{utt} 1 {start:.prec$} {dur:.prec$} {label}",
        prec = TIME_DECIMALS
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use viter_kaldi::types::{PhoneInterval, WordInterval};

    fn fixture() -> (IntervalAlignment, SymbolTable, Vec<String>) {
        let mut st = SymbolTable::new();
        st.add("sil");
        st.add("spn");
        let hh = st.add("HH_B");
        let ow = st.add("OW_E");
        let ali = IntervalAlignment {
            utt: "utt1".to_string(),
            frame_shift_s: 0.01,
            phones: vec![
                PhoneInterval {
                    phone: 1,
                    start_frame: 0,
                    end_frame: 10,
                },
                PhoneInterval {
                    phone: hh,
                    start_frame: 10,
                    end_frame: 25,
                },
                PhoneInterval {
                    phone: ow,
                    start_frame: 25,
                    end_frame: 40,
                },
            ],
            words: vec![WordInterval {
                word: 0,
                pron: 0,
                start_frame: 10,
                end_frame: 40,
            }],
        };
        (ali, st, vec!["hello".to_string()])
    }

    #[test]
    fn phone_rows_are_untagged_kaldi_format() {
        let (ali, st, _) = fixture();
        let s = phone_ctm(&ali, &st);
        assert_eq!(
            s,
            "utt1 1 0.00 0.10 sil\nutt1 1 0.10 0.15 HH\nutt1 1 0.25 0.15 OW\n"
        );
    }

    #[test]
    fn word_rows() {
        let (ali, _, words) = fixture();
        assert_eq!(word_ctm(&ali, &words), "utt1 1 0.10 0.30 hello\n");
    }

    #[test]
    fn writes_both_files() {
        let dir = std::env::temp_dir().join(format!("viter-ctm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aligned.ctm");
        let (ali, st, words) = fixture();
        write_ctm(&path, &[(ali, words.as_slice())], &st).unwrap();

        let phones = std::fs::read_to_string(&path).unwrap();
        assert!(phones.contains("HH"));
        assert!(!phones.contains("HH_B"));
        let w = std::fs::read_to_string(dir.join("aligned.words.ctm")).unwrap();
        assert!(w.contains("hello"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
