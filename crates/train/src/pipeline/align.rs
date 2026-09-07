//! Shared alignment machinery: graph construction, batched scoring, Viterbi.
//!
//! Every stage realigns the same way (MFA `alignment/base.py align_utterances` ->
//! kalpy `gmm/align.py` -> `gmm_align_compiled`, gmm.cpp:2048):
//! optionally boost silence in the acoustic model, score frames against the pdfs the
//! utterance's graph can reach, run beam Viterbi, retry with the wider retry beam,
//! then undo the boost. Graphs are built once per stage (they only change when the
//! tree changes) and reused across every iteration of that stage.

use rayon::prelude::*;
use viter_kaldi::align::{self, AlignOptions};
use viter_kaldi::device::Device;
use viter_kaldi::gmm::AmDiagGmm;
use viter_kaldi::hmm::{self, ContextDependency, Graph, GraphOptions, TransitionModel};
use viter_kaldi::types::{Alignment, Feats, PdfId};

use super::progress::Bar;

/// Decoding graphs for a set of utterances, built once per stage.
pub struct GraphSet {
    /// One graph per entry of the stage's utterance list, in that order.
    graphs: Vec<Graph>,
    /// Unique pdfs each graph touches, precomputed for `Device::score`.
    pdfs: Vec<Vec<PdfId>>,
}

impl GraphSet {
    /// Build one graph per utterance. `words_of(i)` gives the candidate pronunciations per
    /// word for the i'th utterance of the stage's list.
    pub fn build(
        n: usize,
        words_of: impl Fn(usize) -> Vec<Vec<viter_kaldi::types::Pronunciation>> + Sync,
        tm: &TransitionModel,
        ctx: &ContextDependency,
        opts: &GraphOptions,
        bar: &Bar,
    ) -> Self {
        let built: Vec<(Graph, Vec<PdfId>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let words = words_of(i);
                let g = hmm::build_graph(&words, tm, ctx, opts);
                let p = align::graph_pdfs(&g, tm);
                bar.inc(1);
                (g, p)
            })
            .collect();
        let mut graphs = Vec::with_capacity(n);
        let mut pdfs = Vec::with_capacity(n);
        for (g, p) in built {
            graphs.push(g);
            pdfs.push(p);
        }
        Self { graphs, pdfs }
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }
    pub fn graph(&self, i: usize) -> &Graph {
        &self.graphs[i]
    }

    /// Re-cost every graph from the current transition model (see
    /// [`Graph::apply_transition_probs`]); training calls this before each realignment.
    pub fn apply_transition_probs(&mut self, tm: &TransitionModel, opts: &GraphOptions) {
        self.graphs.par_iter_mut().for_each(|g| {
            g.apply_transition_probs(tm, opts.transition_scale, opts.self_loop_scale)
        });
    }
    pub fn pdfs(&self, i: usize) -> &[PdfId] {
        &self.pdfs[i]
    }
}

/// Result of realigning a stage's utterance list.
pub struct AlignOutcome {
    /// One entry per utterance in the stage list; `None` when no path was found even
    /// at the retry beam.
    pub alignments: Vec<Option<Alignment>>,
    pub failed: usize,
}

impl AlignOutcome {
    pub fn ok_count(&self) -> usize {
        self.alignments.len() - self.failed
    }
}

/// Align a batch of utterances against prebuilt graphs.
///
/// `feats(i)` supplies the stage's feature view for utterance i; it is called once per
/// utterance and the matrix is dropped as soon as its alignment is produced, so a
/// caller that already has features cached can hand them back by reference.
///
/// Scoring is batched: utterances are processed in chunks of `batch`, and each chunk
/// asks the device for one score matrix over the union of that chunk's graph pdfs.
/// The per-utterance Viterbi then runs in parallel over the chunk.
pub fn align_batch(
    graphs: &GraphSet,
    tm: &TransitionModel,
    am: &AmDiagGmm,
    device: &Device,
    feats: &[Feats],
    opts: &AlignOptions,
    batch: usize,
    bar: &Bar,
) -> AlignOutcome {
    assert_eq!(
        graphs.len(),
        feats.len(),
        "graph count must match feature count"
    );
    let batch = batch.max(1);
    let n = graphs.len();

    let chunks: Vec<(usize, usize)> = (0..n)
        .step_by(batch)
        .map(|start| (start, (start + batch).min(n)))
        .collect();

    let mut alignments: Vec<Option<Alignment>> = Vec::with_capacity(n);
    let mut t_score = std::time::Duration::ZERO;
    let mut t_vit = std::time::Duration::ZERO;
    for (start, end) in chunks {
        let _t1 = std::time::Instant::now();
        // Every utterance is scored only against the pdfs its own graph can reach.
        let refs: Vec<&Feats> = (start..end).map(|i| &feats[i]).collect();
        let sels: Vec<&[PdfId]> = (start..end).map(|i| graphs.pdfs(i)).collect();
        let scores = device.score_batch_sel(&refs, am, &sels);
        t_score += _t1.elapsed();
        let _t2 = std::time::Instant::now();

        let out: Vec<Option<Alignment>> = (start..end)
            .into_par_iter()
            .zip(scores.par_iter())
            .map(|(i, sc)| {
                // Dense pdf -> column table for this utterance; a lookup per arc in
                // the innermost Viterbi loop, so no hashing.
                let sel = graphs.pdfs(i);
                let max_pdf = sel.iter().copied().max().unwrap_or(0) as usize;
                let mut col_of = vec![usize::MAX; max_pdf + 1];
                for (c, &p) in sel.iter().enumerate() {
                    col_of[p as usize] = c;
                }
                let pdf_col = |p: PdfId| col_of[p as usize];
                let a = align::align(graphs.graph(i), tm, sc, &pdf_col, opts);
                bar.inc(1);
                a
            })
            .collect();
        t_vit += _t2.elapsed();
        alignments.extend(out);
    }
    tracing::debug!(
        score_ms = t_score.as_millis(),
        viterbi_ms = t_vit.as_millis(),
        "align_batch phases"
    );

    let failed = alignments.iter().filter(|a| a.is_none()).count();
    AlignOutcome { alignments, failed }
}

/// Align with silence boosting applied for the duration of the pass, then undone.
///
/// MFA boosts silence only for alignment, never for accumulation
/// (`acoustic_modeling/base.py` align_options -> kalpy `gmm.cpp:203 boost_silence`,
/// undone by scaling with 1/boost).
pub fn align_boosted(
    graphs: &GraphSet,
    tm: &TransitionModel,
    am: &AmDiagGmm,
    silence_pdfs: &[PdfId],
    boost: f32,
    device: &Device,
    feats: &[Feats],
    opts: &AlignOptions,
    batch: usize,
    bar: &Bar,
) -> AlignOutcome {
    if boost == 1.0 || silence_pdfs.is_empty() {
        return align_batch(graphs, tm, am, device, feats, opts, batch, bar);
    }
    // Boosting mutates the model, so work on a copy: the caller's `am` is the model
    // being trained and accumulation must see it unboosted.
    let mut boosted = am.clone();
    boosted.boost_silence(silence_pdfs, boost);
    align_batch(graphs, tm, &boosted, device, feats, opts, batch, bar)
}

/// Equal-align flat start (MFA `mono_align_equal`, kalpy `gmm.cpp:2027` ->
/// `fst::EqualAlign`). Deterministic given the seed: each utterance derives its own
/// stream from the base seed and its index.
pub fn equal_align_all(
    graphs: &GraphSet,
    tm: &TransitionModel,
    num_frames: &[usize],
    seed: u64,
    bar: &Bar,
) -> AlignOutcome {
    use rand::SeedableRng;
    let alignments: Vec<Option<Alignment>> = (0..graphs.len())
        .into_par_iter()
        .map(|i| {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
                seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let a = align::equal_align(graphs.graph(i), tm, num_frames[i], &mut rng);
            bar.inc(1);
            a
        })
        .collect();
    let failed = alignments.iter().filter(|a| a.is_none()).count();
    AlignOutcome { alignments, failed }
}

/// Alignment options for one iteration. MFA widens nothing except monophone's first
/// iteration, which uses `initial_beam` (`monophone.py:232-238`).
pub fn iteration_align_options(base: &AlignOptions, initial_beam: Option<f32>) -> AlignOptions {
    match initial_beam {
        Some(beam) => AlignOptions {
            beam,
            ..base.clone()
        },
        None => base.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_counts() {
        let o = AlignOutcome {
            alignments: vec![None, None, None],
            failed: 2,
        };
        assert_eq!(o.ok_count(), 1);
    }

    #[test]
    fn initial_beam_overrides_only_beam() {
        let base = AlignOptions {
            beam: 10.0,
            retry_beam: 40.0,
            ..Default::default()
        };
        let first = iteration_align_options(&base, Some(6.0));
        assert_eq!(first.beam, 6.0);
        assert_eq!(first.retry_beam, 40.0);
        let later = iteration_align_options(&base, None);
        assert_eq!(later.beam, 10.0);
    }
}
