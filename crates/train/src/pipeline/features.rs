//! In-memory feature store.
//!
//! Base MFCC + per-speaker CMVN for every utterance is computed once, in parallel,
//! and kept in RAM. Everything a stage actually consumes (deltas, or splice + LDA
//! (+ fMLLR)) is derived from that base on demand, because those derivations are
//! cheap relative to MFCC and the derived matrices are large.
//!
//! MFA equivalent: `corpus/features.py` `FeatureArchive` — CMVN is computed per
//! speaker (`corpus/features.py`, `CmvnComputer` over a speaker's utterances) with
//! variance normalization off (kalpy applies `ApplyCmvn(..., norm_vars=False)`),
//! deltas for mono/tri, splice+LDA for lda/sat, and fMLLR appended per speaker for
//! the adapted passes.

use anyhow::{Context, Result};
use viter_kaldi::feat::{
    self, CmvnStats, DeltaOptions, MfccComputer, MfccOptions,
};
use viter_kaldi::transform::Mat;
use viter_kaldi::types::Feats;
use viter_io::corpus::Corpus;
use rayon::prelude::*;
use std::collections::HashMap;

use super::progress::Progress;

/// Which feature view a stage wants.
///
/// Borrowed transforms rather than owned so callers can hold one LDA matrix for a
/// whole stage without cloning it per utterance.
///
// CONTRACT-DEVIATION: plans/CONTRACTS.md writes `FeatureKind` without a lifetime
// (`SpliceLda(&Mat)`). A reference variant requires one, so the enum is
// `FeatureKind<'a>`. The variants and their meaning are unchanged.
#[derive(Clone, Copy, Debug)]
pub enum FeatureKind<'a> {
    /// mono / tri: cmvn'd MFCC + deltas (13 -> 39).
    Deltas,
    /// lda: cmvn'd MFCC, spliced +-3, then the LDA(+MLLT) transform (-> 40).
    SpliceLda(&'a Mat),
    /// sat: as `SpliceLda`, then the speaker's fMLLR transform.
    SpliceLdaFmllr(&'a Mat),
}

/// Base features for a corpus, plus the per-speaker transforms derived during training.
pub struct FeatureStore {
    /// Per utterance (indexed as in `Corpus::utts`): MFCC with speaker CMVN applied.
    base: Vec<Feats>,
    /// Speaker index per utterance.
    utt_speaker: Vec<usize>,
    /// Per-speaker CMVN statistics, kept so a caller can inspect or re-apply them.
    cmvn: Vec<CmvnStats>,
    /// Per-speaker fMLLR transform, `[dim, dim+1]`, filled in by the SAT stage.
    fmllr: Vec<Option<Mat>>,
    deltas: DeltaOptions,
    splice_left: usize,
    splice_right: usize,
    frame_shift_s: f32,
    num_speakers: usize,
}

impl FeatureStore {
    /// Compute MFCCs for the whole corpus in parallel and normalize per speaker.
    // CONTRACT-DEVIATION: plans/CONTRACTS.md gives `build(corpus, mfcc)`. The progress bar
    // is a required feature of this crate, so the `Progress` sink is threaded through
    // as a third argument rather than constructed (and thus duplicated) internally.
    pub fn build(corpus: &Corpus, mfcc: &MfccOptions, progress: &Progress) -> Result<Self> {
        Self::build_with(corpus, mfcc, &DeltaOptions::default(), 3, 3, progress)
    }

    /// As `build`, but with explicit delta and splice settings (from `TrainConfig`).
    pub fn build_with(
        corpus: &Corpus,
        mfcc: &MfccOptions,
        deltas: &DeltaOptions,
        splice_left: usize,
        splice_right: usize,
        progress: &Progress,
    ) -> Result<Self> {
        let computer = MfccComputer::new(mfcc.clone());
        let frame_shift_s = computer.frame_shift_s();

        // Speaker name -> dense index. Corpus::speakers is the canonical order.
        let mut speaker_index: HashMap<&str, usize> = HashMap::new();
        for (i, s) in corpus.speakers.iter().enumerate() {
            speaker_index.insert(s.as_str(), i);
        }
        let num_speakers = corpus.speakers.len().max(1);

        let utt_speaker: Vec<usize> = corpus
            .utts
            .iter()
            .map(|u| speaker_index.get(u.speaker.as_str()).copied().unwrap_or(0))
            .collect();

        let bar = progress.bar("mfcc", corpus.utts.len() as u64);
        let mut base: Vec<Feats> = corpus
            .utts
            .par_iter()
            .map(|u| -> Result<Feats> {
                let audio = viter_kaldi::audio::read_16k(&u.audio)
                    .with_context(|| format!("reading audio for utterance {}", u.id))?;
                let f = computer.compute(&audio.samples);
                bar.inc(1);
                Ok(f)
            })
            .collect::<Result<Vec<_>>>()?;
        bar.finish();
        let total_frames: usize = base.iter().map(|f| f.nrows()).sum();
        let dim0 = base.first().map(|f| f.ncols()).unwrap_or(0);
        // Base + deltas (3x) + splice (7x) is the worst case held at once.
        let est_bytes = total_frames * dim0 * 4 * (1 + 3 + 7);
        tracing::info!(
            utterances = base.len(),
            frames = total_frames,
            hours = total_frames as f64 / 360_000.0,
            est_feature_ram_gb = est_bytes as f64 / 1e9,
            "features extracted"
        );

        // Per-speaker CMVN stats, then apply in place. MFA computes CMVN over all of a
        // speaker's utterances and applies with norm_vars = False.
        let dim = base.first().map(|f| f.ncols()).unwrap_or(mfcc.num_ceps as usize);
        let mut cmvn = vec![CmvnStats::new(dim); num_speakers];
        for (i, f) in base.iter().enumerate() {
            cmvn[utt_speaker[i]].accumulate(f);
        }

        let cmvn_ref = &cmvn;
        let spk = &utt_speaker;
        base.par_iter_mut().enumerate().for_each(|(i, f)| {
            feat::apply_cmvn(f, &cmvn_ref[spk[i]], false);
        });

        Ok(Self {
            base,
            utt_speaker,
            cmvn,
            fmllr: vec![None; num_speakers],
            deltas: deltas.clone(),
            splice_left,
            splice_right,
            frame_shift_s,
            num_speakers,
        })
    }

    pub fn len(&self) -> usize {
        self.base.len()
    }
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }
    pub fn num_speakers(&self) -> usize {
        self.num_speakers
    }
    pub fn frame_shift_s(&self) -> f32 {
        self.frame_shift_s
    }
    pub fn speaker_of(&self, utt: usize) -> usize {
        self.utt_speaker[utt]
    }
    pub fn cmvn_stats(&self, speaker: usize) -> &CmvnStats {
        &self.cmvn[speaker]
    }
    /// Frames in an utterance, without materializing any derived features.
    pub fn num_frames(&self, utt: usize) -> usize {
        self.base[utt].nrows()
    }
    /// Raw cmvn'd MFCC for an utterance.
    pub fn base(&self, utt: usize) -> &Feats {
        &self.base[utt]
    }
    /// Dimension of the base (cmvn'd MFCC) features.
    pub fn base_dim(&self) -> usize {
        self.base.first().map(|f| f.ncols()).unwrap_or(0)
    }

    /// Spliced base features — the input to LDA estimation and to the LDA transform.
    pub fn spliced(&self, utt: usize) -> Feats {
        feat::splice(&self.base[utt], self.splice_left, self.splice_right)
    }

    /// Dimension of `spliced`.
    pub fn spliced_dim(&self) -> usize {
        self.base_dim() * (self.splice_left + self.splice_right + 1)
    }

    pub fn splice_context(&self) -> (usize, usize) {
        (self.splice_left, self.splice_right)
    }

    pub fn delta_options(&self) -> &DeltaOptions {
        &self.deltas
    }

    /// Derive the features one stage wants for one utterance.
    ///
    /// `SpliceLdaFmllr` falls back to the unadapted features when the speaker has no
    /// transform yet, which is what MFA does before the first fMLLR estimation.
    pub fn feats_for(&self, utt: usize, kind: FeatureKind<'_>) -> Feats {
        match kind {
            FeatureKind::Deltas => feat::add_deltas(&self.base[utt], &self.deltas),
            FeatureKind::SpliceLda(lda) => feat::apply_transform(&self.spliced(utt), lda),
            FeatureKind::SpliceLdaFmllr(lda) => {
                let lda_feats = feat::apply_transform(&self.spliced(utt), lda);
                match &self.fmllr[self.utt_speaker[utt]] {
                    Some(x) => feat::apply_transform(&lda_feats, x),
                    None => lda_feats,
                }
            }
        }
    }

    /// Derive features for a set of utterances in parallel. Stages cache the result
    /// for the length of one stage so per-iteration loops do not recompute splices.
    pub fn feats_for_many(&self, utts: &[usize], kind: FeatureKind<'_>) -> Vec<Feats> {
        utts.par_iter().map(|&u| self.feats_for(u, kind)).collect()
    }

    /// Apply a speaker's fMLLR to already-LDA-transformed features. Used when a stage
    /// holds a cached LDA view and only the adaptation changed.
    pub fn adapt(&self, utt: usize, lda_feats: &Feats) -> Feats {
        match &self.fmllr[self.utt_speaker[utt]] {
            Some(x) => feat::apply_transform(lda_feats, x),
            None => lda_feats.clone(),
        }
    }

    pub fn fmllr_for(&self, speaker: usize) -> Option<&Mat> {
        self.fmllr[speaker].as_ref()
    }

    pub fn fmllr_for_utt(&self, utt: usize) -> Option<&Mat> {
        self.fmllr[self.utt_speaker[utt]].as_ref()
    }

    pub fn set_fmllr(&mut self, speaker: usize, mat: Mat) {
        self.fmllr[speaker] = Some(mat);
    }

    pub fn clear_fmllr(&mut self) {
        for x in &mut self.fmllr {
            *x = None;
        }
    }

    pub fn has_any_fmllr(&self) -> bool {
        self.fmllr.iter().any(|x| x.is_some())
    }

    /// All utterance indices grouped by speaker; used by fMLLR estimation, which
    /// accumulates one set of stats per speaker.
    pub fn utts_by_speaker(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.num_speakers];
        for (utt, &spk) in self.utt_speaker.iter().enumerate() {
            out[spk].push(utt);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_kind_is_copy_and_cheap() {
        // Compile-time check that a stage can hold one matrix and pass the kind around.
        fn takes(_k: FeatureKind<'_>) {}
        let m: Mat = ndarray::Array2::zeros((40, 92));
        let k = FeatureKind::SpliceLda(&m);
        takes(k);
        takes(k);
    }
}
