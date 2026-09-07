//! Splitting a full-corpus pass into speaker-aligned chunks.
//!
//! `FeatureStore` keeps 13-dim MFCCs, which is cheap; what is expensive is the
//! derived view a pass consumes (39-dim deltas, 40-dim splice+LDA). Materializing
//! that for the whole corpus at once costs ~0.2 GB per hour of audio, so full
//! passes derive one chunk at a time and drop it before the next.
//!
//! Chunks are speaker-grouped but frame-bounded: utterances are walked speaker by
//! speaker so a chunk holds whole speakers wherever it can, but a chunk closes as
//! soon as it reaches the target even in the middle of a speaker. A single-speaker
//! corpus (the common TTS case) therefore still yields many chunks. Per-speaker work
//! (fMLLR) is accumulate-then-solve across chunks (`sat::FmllrEstimator`), so a
//! speaker spanning several chunks is the same computation.

use super::features::FeatureStore;
use crate::config::TrainConfig;

/// Chunk size for a pass: `TrainConfig::chunk_frames`, overridable with
/// `VITER_CHUNK_FRAMES` (used by the equivalence check — chunking must not change
/// the output, so forcing tiny chunks must produce identical models and
/// TextGrids). An unparseable or zero value falls back to the configured default.
/// Both the full alignment passes and the per-iteration training passes read it.
pub fn frames_for(cfg: &TrainConfig) -> usize {
    match std::env::var("VITER_CHUNK_FRAMES") {
        Ok(v) => v.trim().parse::<usize>().ok().filter(|&n| n > 0),
        Err(_) => None,
    }
    .unwrap_or(cfg.chunk_frames)
}

/// Split `utts` (corpus indices) into consecutive groups holding about
/// `target_frames` base frames each.
///
/// Utterances are grouped by speaker first — speakers in `FeatureStore::speaker_of`
/// order, utterances inside a speaker in the order `utts` gives — and the resulting
/// sequence is then cut every `target_frames` frames. A chunk therefore packs small
/// speakers together and a speaker larger than the target spans several chunks. An
/// utterance is never split. Deterministic.
pub fn by_frames(feats: &FeatureStore, utts: &[usize], target_frames: usize) -> Vec<Vec<usize>> {
    if utts.is_empty() {
        return Vec::new();
    }
    // Group by speaker, preserving the order `utts` gives inside each speaker and
    // ordering the groups by speaker index (`speaker_of`).
    let mut by_spk: Vec<Vec<usize>> = vec![Vec::new(); feats.num_speakers()];
    for &u in utts {
        by_spk[feats.speaker_of(u)].push(u);
    }

    let target = target_frames.max(1);
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_frames = 0usize;
    for u in by_spk.into_iter().flatten() {
        let frames = feats.num_frames(u);
        // Adding this utterance would overshoot: close the current chunk first. A
        // single utterance longer than the target then becomes a chunk of its own.
        if !cur.is_empty() && cur_frames + frames > target {
            out.push(std::mem::take(&mut cur));
            cur_frames = 0;
        }
        cur.push(u);
        cur_frames += frames;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    tracing::debug!(
        utterances = utts.len(),
        chunks = out.len(),
        target_frames = target,
        "split pass into chunks"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::progress::Progress;
    use viter_io::corpus::Corpus;
    use viter_kaldi::audio::Audio;
    use viter_kaldi::feat::MfccOptions;

    /// A store over synthetic audio: `spk_utts[s]` = durations (seconds) of speaker s.
    fn store(spk_utts: &[&[f32]]) -> FeatureStore {
        use viter_kaldi::types::Utterance;
        let mut utts = Vec::new();
        let mut speakers = Vec::new();
        let mut durs = Vec::new();
        for (s, group) in spk_utts.iter().enumerate() {
            speakers.push(format!("spk{s}"));
            for (i, &d) in group.iter().enumerate() {
                utts.push(Utterance {
                    id: format!("spk{s}-{i}"),
                    speaker: format!("spk{s}"),
                    audio: std::path::PathBuf::from("/dev/null"),
                    text: String::new(),
                    words: Vec::new(),
                    prons: Vec::new(),
                });
                durs.push(d);
            }
        }
        let corpus = Corpus {
            utts,
            speakers,
            phones: Default::default(),
            silence_phones: Vec::new(),
            oov_words: Default::default(),
        };
        let progress = Progress::hidden();
        FeatureStore::build_with_audio(
            &corpus,
            &MfccOptions::default(),
            &Default::default(),
            3,
            3,
            &progress,
            &|i, _u| {
                let n = (durs[i] * 16000.0) as usize;
                Ok(Audio {
                    samples: (0..n).map(|k| ((k % 97) as f32 - 48.0) / 1000.0).collect(),
                    sample_rate: 16000,
                })
            },
        )
        .unwrap()
    }

    #[test]
    fn speaker_order_is_preserved() {
        // Utterances come out grouped by speaker index, in the order given inside
        // each speaker, whatever the chunk boundaries are.
        let f = store(&[&[1.0, 1.0], &[1.0], &[2.0, 1.0]]);
        let all: Vec<usize> = (0..f.len()).collect();
        for target in [1usize, 10, 100, 500, 5000] {
            let flat: Vec<usize> = by_frames(&f, &all, target).concat();
            assert_eq!(flat, all, "target={target}");
            let spks: Vec<usize> = flat.iter().map(|&u| f.speaker_of(u)).collect();
            let mut sorted = spks.clone();
            sorted.sort_unstable();
            assert_eq!(spks, sorted, "speakers out of order at target={target}");
        }
    }

    #[test]
    fn covers_every_utterance_in_order() {
        let f = store(&[&[1.0, 1.0], &[1.0], &[2.0, 1.0]]);
        let all: Vec<usize> = (0..f.len()).collect();
        for target in [1usize, 100, 5000] {
            let flat: Vec<usize> = by_frames(&f, &all, target).concat();
            assert_eq!(flat, all, "target={target}");
        }
    }

    #[test]
    fn huge_target_is_one_chunk() {
        let f = store(&[&[1.0, 1.0], &[1.0]]);
        let all: Vec<usize> = (0..f.len()).collect();
        let chunks = by_frames(&f, &all, usize::MAX);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], all);
    }

    #[test]
    fn tiny_target_is_one_chunk_per_utterance() {
        let f = store(&[&[1.0, 1.0], &[1.0], &[2.0]]);
        let all: Vec<usize> = (0..f.len()).collect();
        let chunks = by_frames(&f, &all, 1);
        assert_eq!(chunks.len(), 4);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c, &vec![i]);
        }
    }

    #[test]
    fn one_big_speaker_yields_many_chunks() {
        // The single-speaker corpus (LJSpeech) case: a speaker far larger than the
        // target must still be cut into chunks, or peak memory never drops.
        let f = store(&[&[1.0; 12]]);
        let all: Vec<usize> = (0..f.len()).collect();
        let per_utt = f.num_frames(0);
        let chunks = by_frames(&f, &all, per_utt * 3);
        assert_eq!(chunks.len(), 4);
        for c in &chunks {
            assert_eq!(c.len(), 3);
        }
        assert_eq!(chunks.concat(), all);
    }

    #[test]
    fn sizes_stay_near_the_target() {
        let f = store(&[&[1.0], &[1.0], &[1.0], &[1.0]]);
        let all: Vec<usize> = (0..f.len()).collect();
        let per_utt = f.num_frames(0);
        // Two utterances' worth per chunk; each is its own speaker.
        let chunks = by_frames(&f, &all, per_utt * 2);
        assert_eq!(chunks.len(), 2);
        for c in &chunks {
            assert_eq!(c.len(), 2);
            let frames: usize = c.iter().map(|&u| f.num_frames(u)).sum();
            assert!(frames <= per_utt * 2, "chunk overshot the target");
        }
    }

    #[test]
    fn small_speakers_still_pack_together() {
        let f = store(&[&[1.0], &[1.0], &[8.0]]);
        let all: Vec<usize> = (0..f.len()).collect();
        let target = f.num_frames(0) * 3;
        let chunks = by_frames(&f, &all, target);
        // The two short speakers share the first chunk; the long one spills over.
        assert_eq!(chunks[0], vec![0, 1]);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), all);
    }

    #[test]
    fn frames_for_prefers_the_env_override() {
        let cfg = TrainConfig::default();
        // Safety: single-threaded test scope; the override is read only here.
        unsafe { std::env::set_var("VITER_CHUNK_FRAMES", "1234") };
        assert_eq!(frames_for(&cfg), 1234);
        unsafe { std::env::set_var("VITER_CHUNK_FRAMES", "0") };
        assert_eq!(frames_for(&cfg), cfg.chunk_frames);
        unsafe { std::env::remove_var("VITER_CHUNK_FRAMES") };
        assert_eq!(frames_for(&cfg), cfg.chunk_frames);
    }

    #[test]
    fn empty_input_is_no_chunks() {
        let f = store(&[&[1.0]]);
        assert!(by_frames(&f, &[], 100).is_empty());
    }

    #[test]
    fn subset_only_covers_the_given_utterances() {
        let f = store(&[&[1.0, 1.0], &[1.0], &[1.0]]);
        let subset = vec![1usize, 3];
        let flat: Vec<usize> = by_frames(&f, &subset, 1).concat();
        assert_eq!(flat, subset);
    }
}
