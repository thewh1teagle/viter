//! Aligning a corpus whose audio is already in memory.
//!
//! [`align_corpus_with_audio`] is [`super::align_corpus_with`] with the waveforms
//! handed in directly instead of read from `Utterance::audio`, for callers (the
//! Python bindings, a server) that hold samples rather than files.

use anyhow::{Result, bail};
use viter_io::corpus::Corpus;
use viter_kaldi::align::AlignOptions;
use viter_kaldi::audio::{Audio, to_16k};
use viter_kaldi::device::Device;
use viter_kaldi::model::AcousticModel;
use viter_kaldi::types::IntervalAlignment;

use super::full_pass::{AlignOverrides, align_corpus_reading};

/// As [`super::align_corpus_with`], but `audio[i]` is the waveform of
/// `corpus.utts[i]` (any sample rate; converted with [`to_16k`]).
/// `corpus.utts[i].audio` is never opened.
///
/// Errors if `audio.len() != corpus.utts.len()`.
pub fn align_corpus_with_audio(
    corpus: &Corpus,
    audio: &[Audio],
    model: &AcousticModel,
    device: &Device,
    opts: Option<&AlignOptions>,
    over: &AlignOverrides,
) -> Result<Vec<Option<IntervalAlignment>>> {
    check_lengths(audio.len(), corpus.utts.len())?;
    align_corpus_reading(corpus, model, device, opts, over, &|i, _| {
        Ok(to_16k(&audio[i]))
    })
}

/// `audio.len()` must line up one-to-one with `corpus.utts`.
fn check_lengths(audio: usize, utts: usize) -> Result<()> {
    if audio != utts {
        bail!("audio/corpus length mismatch: {audio} waveforms for {utts} utterances");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::features::FeatureStore;
    use crate::pipeline::progress::Progress;
    use viter_kaldi::feat::{DeltaOptions, MfccOptions};
    use viter_kaldi::types::Utterance;

    fn tone(rate: u32, secs: f32, freq: f32) -> Audio {
        let n = (rate as f32 * secs) as usize;
        Audio {
            samples: (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.5)
                .collect(),
            sample_rate: rate,
        }
    }

    fn utt(id: &str, path: &std::path::Path) -> Utterance {
        Utterance {
            id: id.into(),
            speaker: "spk".into(),
            audio: path.to_path_buf(),
            words: Vec::new(),
            prons: Vec::new(),
            text: String::new(),
        }
    }

    fn corpus_of(utts: Vec<Utterance>) -> Corpus {
        Corpus {
            utts,
            speakers: vec!["spk".into()],
            phones: viter_kaldi::types::SymbolTable::default(),
            silence_phones: Vec::new(),
            oov_words: Default::default(),
        }
    }

    /// The reader plumbing: features built from in-memory samples are bit-identical
    /// to features built by reading the same waveform back from a wav file.
    #[test]
    fn in_memory_features_match_the_file_path() {
        let dir = std::env::temp_dir().join("viter_inmem_features_test");
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let wavs: Vec<_> = (0..2)
            .map(|i| {
                let p = dir.join(format!("u{i}.wav"));
                viter_kaldi::audio::write_wav(&p, &tone(22_050, 0.4, 220.0 * (i + 1) as f32))
                    .expect("write wav");
                p
            })
            .collect();

        let corpus = corpus_of(vec![utt("u0", &wavs[0]), utt("u1", &wavs[1])]);

        let mfcc = MfccOptions::default();
        let deltas = DeltaOptions::default();
        let progress = Progress::hidden();

        let from_path =
            FeatureStore::build_with(&corpus, &mfcc, &deltas, 3, 3, &progress).expect("path build");

        // Same waveforms, but supplied as samples read straight from the files.
        let samples: Vec<Audio> = wavs
            .iter()
            .map(|p| viter_kaldi::audio::read(p).expect("read"))
            .collect();
        let from_mem =
            FeatureStore::build_with_audio(&corpus, &mfcc, &deltas, 3, 3, &progress, &|i, _| {
                Ok(to_16k(&samples[i]))
            })
            .expect("mem build");

        assert_eq!(from_path.len(), from_mem.len());
        for u in 0..from_path.len() {
            assert_eq!(from_path.base(u), from_mem.base(u), "utterance {u}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn length_mismatch_is_an_error() {
        assert!(check_lengths(2, 2).is_ok());
        let err = check_lengths(0, 1).expect_err("length mismatch");
        assert!(err.to_string().contains("length mismatch"), "{err}");
    }
}
