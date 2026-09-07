//! Port of `plans/kaldi/src/decoder/faster-decoder.{h,cc}` as a beam Viterbi over
//! [`crate::hmm::Graph`].
//!
//! Token passing with one token per graph state per frame: `ProcessEmitting` advances tokens
//! along transition-id arcs, adding the (scaled) acoustic cost; `ProcessNonemitting` closes the
//! graph over epsilon arcs at the same frame. Cutoffs follow Kaldi's `GetCutoff`, including the
//! `max_active` / `min_active` adaptive beam. On failure to reach a final state, alignment is
//! retried with `retry_beam`, exactly as `gmm_align_compiled` does.

use crate::hmm::{Graph, NO_WORD, TransitionModel};
use crate::types::{Alignment, PdfId, TransitionId, WordId};
use ndarray::{Array2, ArrayView1};

/// Kaldi `FasterDecoderOptions` plus the alignment-level retry beam.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct AlignOptions {
    pub beam: f32,
    pub retry_beam: f32,
    pub acoustic_scale: f32,
    pub max_active: usize,
    pub min_active: usize,
    /// Kaldi `beam_delta`, used when the adaptive beam kicks in.
    pub beam_delta: f32,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            beam: 10.0,
            retry_beam: 40.0,
            acoustic_scale: 0.1,
            max_active: usize::MAX,
            // Kaldi's FasterDecoder default; this decoder is used mostly for alignment, so the
            // default is small (faster-decoder.h).
            min_active: 20,
            beam_delta: 0.5,
        }
    }
}

/// The static arc data and score column are resolved once, including across retries.
/// Emitting and epsilon arcs retain their original relative order within each state.
struct PreparedGraph<'a> {
    graph: &'a Graph,
    arcs: Vec<(crate::hmm::Arc, usize)>,
    /// `(start, end of emitting arcs, end of epsilon arcs)` per graph state.
    ranges: Vec<(usize, usize, usize)>,
}

impl<'a> PreparedGraph<'a> {
    fn new(graph: &'a Graph, tm: &TransitionModel, pdf_col: &dyn Fn(PdfId) -> usize) -> Self {
        let mut arcs = Vec::new();
        let mut ranges = Vec::with_capacity(graph.num_states());
        for state in &graph.states {
            let start = arcs.len();
            for &arc in &state.arcs {
                if arc.tid != 0 {
                    arcs.push((arc, pdf_col(tm.transition_id_to_pdf(arc.tid))));
                }
            }
            let emitting_end = arcs.len();
            for &arc in &state.arcs {
                if arc.tid == 0 {
                    arcs.push((arc, 0));
                }
            }
            ranges.push((start, emitting_end, arcs.len()));
        }
        Self {
            graph,
            arcs,
            ranges,
        }
    }
}

/// Only candidates that win recombination enter the arena. Arc labels/costs live in PreparedGraph;
/// accumulated costs live in Active so the hot loop does not read the traceback arena.
#[derive(Clone, Copy, Debug)]
struct Token {
    prev: usize,
    arc: usize,
    /// Scaling is performed in f32, exactly as in DecodableAmDiagGmmScaled.
    ac_cost: f32,
}

struct Arena {
    toks: Vec<Token>,
}

impl Arena {
    fn new() -> Self {
        Self { toks: Vec::new() }
    }
    fn push(&mut self, t: Token) -> usize {
        self.toks.push(t);
        self.toks.len() - 1
    }
}

/// The active token set for one frame: at most one token per graph state.
struct Active {
    slot: Vec<usize>,
    cost: Vec<f64>,
    states: Vec<u32>,
}

impl Active {
    fn new(num_states: usize) -> Self {
        Self {
            slot: vec![usize::MAX; num_states],
            cost: vec![f64::INFINITY; num_states],
            states: Vec::new(),
        }
    }
    fn clear(&mut self) {
        for &s in &self.states {
            self.slot[s as usize] = usize::MAX;
        }
        self.states.clear();
    }
    fn get(&self, state: u32) -> Option<usize> {
        let i = self.slot[state as usize];
        (i != usize::MAX).then_some(i)
    }
    /// Do not allocate traceback storage for candidates that lose recombination.
    /// Strict comparison preserves the first path on ties, including epsilon paths.
    fn insert(&mut self, arena: &mut Arena, state: u32, token: Token, cost: f64) -> bool {
        let s = state as usize;
        if self.slot[s] == usize::MAX {
            self.states.push(state);
        } else if cost.partial_cmp(&self.cost[s]) != Some(std::cmp::Ordering::Less) {
            return false;
        }
        self.slot[s] = arena.push(token);
        self.cost[s] = cost;
        true
    }
}

struct Decoder<'a> {
    graph: &'a PreparedGraph<'a>,
    opts: &'a AlignOptions,
    beam: f32,
    arena: Arena,
    cur: Active,
    next: Active,
    queue: Vec<u32>,
    tmp: Vec<f64>,
}

impl<'a> Decoder<'a> {
    fn new(graph: &'a PreparedGraph<'a>, opts: &'a AlignOptions, beam: f32) -> Self {
        let n = graph.graph.num_states();
        Self {
            graph,
            opts,
            beam,
            arena: Arena::new(),
            cur: Active::new(n),
            next: Active::new(n),
            queue: Vec::new(),
            tmp: Vec::new(),
        }
    }

    /// Kaldi `GetCutoff`. Returns `(cutoff, adaptive_beam, best_state)`.
    fn get_cutoff(&mut self) -> (f64, f32, Option<u32>) {
        let mut best_cost = f64::INFINITY;
        let mut best_state = None;
        if self.opts.max_active == usize::MAX && self.opts.min_active == 0 {
            for &s in &self.cur.states {
                let c = self.cur.cost[s as usize];
                if c < best_cost {
                    best_cost = c;
                    best_state = Some(s);
                }
            }
            return (best_cost + self.beam as f64, self.beam, best_state);
        }

        self.tmp.clear();
        for &s in &self.cur.states {
            let c = self.cur.cost[s as usize];
            self.tmp.push(c);
            if c < best_cost {
                best_cost = c;
                best_state = Some(s);
            }
        }
        let beam_cutoff = best_cost + self.beam as f64;
        let mut max_active_cutoff = f64::INFINITY;
        let mut min_active_cutoff = f64::INFINITY;

        if self.tmp.len() > self.opts.max_active {
            self.tmp
                .select_nth_unstable_by(self.opts.max_active, |a, b| a.total_cmp(b));
            max_active_cutoff = self.tmp[self.opts.max_active];
        }
        if max_active_cutoff < beam_cutoff {
            let adaptive = (max_active_cutoff - best_cost) as f32 + self.opts.beam_delta;
            return (max_active_cutoff, adaptive, best_state);
        }
        // Usually the ordinary beam already retains min_active tokens. Establish
        // that with an early-exit count instead of partitioning the whole frontier
        // every frame; nth-element is only needed when the beam must be widened.
        if (self.tmp.len() <= self.opts.max_active || self.opts.min_active < self.opts.max_active)
            && self
                .tmp
                .iter()
                .filter(|&&c| c <= beam_cutoff)
                .nth(self.opts.min_active)
                .is_some()
        {
            return (beam_cutoff, self.beam, best_state);
        }
        if self.tmp.len() > self.opts.min_active {
            if self.opts.min_active == 0 {
                min_active_cutoff = best_cost;
            } else {
                // Kaldi restricts the nth_element range to the first max_active entries when
                // the array was already partitioned above.
                let end = self
                    .tmp
                    .len()
                    .min(if self.tmp.len() > self.opts.max_active {
                        self.opts.max_active
                    } else {
                        self.tmp.len()
                    });
                if self.opts.min_active < end {
                    self.tmp[..end]
                        .select_nth_unstable_by(self.opts.min_active, |a, b| a.total_cmp(b));
                    min_active_cutoff = self.tmp[self.opts.min_active];
                }
            }
        }
        if min_active_cutoff > beam_cutoff {
            let adaptive = (min_active_cutoff - best_cost) as f32 + self.opts.beam_delta;
            (min_active_cutoff, adaptive, best_state)
        } else {
            (beam_cutoff, self.beam, best_state)
        }
    }

    /// Kaldi `ProcessNonemitting`: close the active set over epsilon arcs.
    fn process_nonemitting(&mut self, cutoff: f64) {
        self.queue.clear();
        self.queue.extend_from_slice(&self.cur.states);
        while let Some(state) = self.queue.pop() {
            let Some(tok_idx) = self.cur.get(state) else {
                continue;
            };
            let tok_cost = self.cur.cost[state as usize];
            if tok_cost > cutoff {
                continue;
            }
            let (_, start, end) = self.graph.ranges[state as usize];
            for arc_idx in start..end {
                let arc = &self.graph.arcs[arc_idx].0;
                let cost = tok_cost + arc.cost as f64;
                if cost > cutoff {
                    continue;
                }
                let token = Token {
                    prev: tok_idx,
                    arc: arc_idx,
                    ac_cost: 0.0,
                };
                if self.cur.insert(&mut self.arena, arc.next, token, cost) {
                    self.queue.push(arc.next);
                }
            }
        }
    }

    /// Kaldi `ProcessEmitting`. Returns the cutoff to use for the following nonemitting pass.
    fn process_emitting(&mut self, row: ArrayView1<'_, f32>) -> f64 {
        let (weight_cutoff, adaptive_beam, best_state) = self.get_cutoff();
        let mut next_cutoff = f64::INFINITY;

        // Process the best token first for a tight bound on the next cutoff.
        if let Some(state) = best_state {
            let tok_cost = self.cur.cost[state as usize];
            let (start, end, _) = self.graph.ranges[state as usize];
            for &(arc, col) in &self.graph.arcs[start..end] {
                let ac_cost = -(self.opts.acoustic_scale * row[col]) as f64;
                let w = arc.cost as f64 + tok_cost + ac_cost;
                if w + (adaptive_beam as f64) < next_cutoff {
                    next_cutoff = w + adaptive_beam as f64;
                }
            }
        }

        self.next.clear();
        // `cur.states` is not mutated during this loop, so iterating over a snapshot is safe and
        // avoids borrowing conflicts with the arena.
        let states = std::mem::take(&mut self.cur.states);
        for &state in &states {
            let tok_idx = self.cur.slot[state as usize];
            if tok_idx == usize::MAX {
                continue;
            }
            let tok_cost = self.cur.cost[state as usize];
            if tok_cost >= weight_cutoff {
                continue; // pruned
            }
            let (start, end, _) = self.graph.ranges[state as usize];
            for arc_idx in start..end {
                let (arc, col) = &self.graph.arcs[arc_idx];
                let ac_cost = -(self.opts.acoustic_scale * row[*col]) as f64;
                let w = arc.cost as f64 + tok_cost + ac_cost;
                if w >= next_cutoff {
                    continue;
                }
                let token = Token {
                    prev: tok_idx,
                    arc: arc_idx,
                    ac_cost: ac_cost as f32,
                };
                self.next.insert(&mut self.arena, arc.next, token, w);
                if w + (adaptive_beam as f64) < next_cutoff {
                    next_cutoff = w + adaptive_beam as f64;
                }
            }
        }
        self.cur.states = states;
        self.cur.clear();
        std::mem::swap(&mut self.cur, &mut self.next);
        next_cutoff
    }

    fn reached_final(&self) -> bool {
        self.cur
            .states
            .iter()
            .any(|&s| self.graph.graph.is_final(s) && self.cur.cost[s as usize].is_finite())
    }

    /// Kaldi `GetBestPath`, restricted to final states when one was reached.
    fn best_token(&self) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        let reached_final = self.reached_final();
        if reached_final {
            for &s in &self.cur.states {
                let Some(fc) = self.graph.graph.final_cost(s) else {
                    continue;
                };
                let idx = self.cur.slot[s as usize];
                let cost = self.cur.cost[s as usize] + fc as f64;
                if cost.is_finite() && best.is_none_or(|(_, b)| cost < b) {
                    best = Some((idx, fc as f64));
                }
            }
        } else {
            for &s in &self.cur.states {
                let idx = self.cur.slot[s as usize];
                let cost = self.cur.cost[s as usize];
                if best.is_none_or(|(_, best_cost)| cost < best_cost) {
                    best = Some((idx, cost));
                }
            }
        }
        if reached_final {
            best
        } else {
            best.map(|(idx, _)| (idx, 0.0))
        }
    }
}

/// Beam-Viterbi align one utterance against its decoding graph.
///
/// `scores` is `[frames, cols]`; `pdf_col` maps a pdf-id to its column. Values are
/// log-likelihoods and are multiplied by `opts.acoustic_scale`, as
/// `DecodableAmDiagGmmScaled` does.
///
/// Returns `None` if no path reaches a final state even after retrying with `retry_beam`.
pub fn align(
    graph: &Graph,
    tm: &TransitionModel,
    scores: &Array2<f32>,
    pdf_col: &dyn Fn(PdfId) -> usize,
    opts: &AlignOptions,
) -> Option<Alignment> {
    let prepared = PreparedGraph::new(graph, tm, pdf_col);
    if let Some(a) = decode(&prepared, scores, opts, opts.beam) {
        return Some(a);
    }
    if opts.retry_beam > opts.beam {
        tracing::debug!(
            beam = opts.beam,
            retry_beam = opts.retry_beam,
            "alignment did not reach a final state; retrying with the retry beam"
        );
        return decode(&prepared, scores, opts, opts.retry_beam);
    }
    None
}

fn decode(
    graph: &PreparedGraph<'_>,
    scores: &Array2<f32>,
    opts: &AlignOptions,
    beam: f32,
) -> Option<Alignment> {
    let num_frames = scores.nrows();
    let mut d = Decoder::new(graph, opts, beam);

    // Kaldi seeds the start state with a dummy token, then closes over epsilon arcs.
    let start = d.arena.push(Token {
        prev: usize::MAX,
        arc: usize::MAX,
        ac_cost: 0.0,
    });
    d.cur.slot[0] = start;
    d.cur.cost[0] = 0.0;
    d.cur.states.push(0);
    d.process_nonemitting(f64::INFINITY);

    for frame in 0..num_frames {
        let cutoff = d.process_emitting(scores.row(frame));
        d.process_nonemitting(cutoff);
        if d.cur.states.is_empty() {
            tracing::warn!(frame, "all tokens pruned away");
            return None;
        }
    }

    if !d.reached_final() {
        return None;
    }
    let (best, final_cost) = d.best_token()?;

    // Backtrace.
    let mut tids: Vec<TransitionId> = Vec::with_capacity(num_frames);
    let mut words: Vec<WordId> = Vec::new();
    let mut prons: Vec<u32> = Vec::new();
    let mut total_ac = 0.0f64;
    let mut total_graph = final_cost;
    let mut cur = best;
    while cur != usize::MAX {
        let t = d.arena.toks[cur];
        if t.prev == usize::MAX {
            break; // the dummy start token carries no arc
        }
        let arc = &graph.arcs[t.arc].0;
        if arc.tid != 0 {
            tids.push(arc.tid);
        }
        if arc.word != NO_WORD {
            words.push(arc.word);
            prons.push(arc.pron);
        }
        total_ac += t.ac_cost as f64;
        total_graph += arc.cost as f64;
        cur = t.prev;
    }
    tids.reverse();
    words.reverse();
    prons.reverse();

    if tids.len() != num_frames {
        tracing::warn!(
            got = tids.len(),
            want = num_frames,
            "best path has the wrong number of emitting arcs"
        );
        return None;
    }

    // kalpy: likelihood = -(graph_cost + acoustic_cost) / acoustic_scale, giving an unscaled
    // log-likelihood (gmm.cpp:2048 gmm_align_compiled).
    let loglike = if opts.acoustic_scale != 0.0 {
        -(total_graph + total_ac) / opts.acoustic_scale as f64
    } else {
        -(total_graph + total_ac)
    };

    Some(Alignment {
        utt: String::new(),
        tids,
        words,
        prons,
        loglike: loglike as f32,
    })
}

/// Unique pdfs used by a graph, sorted. Used to request a scores matrix with only those columns.
pub fn graph_pdfs(graph: &Graph, tm: &TransitionModel) -> Vec<PdfId> {
    crate::hmm::graph_pdfs(graph, tm)
}

#[cfg(test)]
#[path = "viterbi_tests.rs"]
mod tests;
