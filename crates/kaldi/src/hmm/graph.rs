//! Per-utterance decoding graph, built directly from a phone sequence.
//!
//! This replaces Kaldi's `TrainingGraphCompiler` pipeline (compose L with G, compose with the
//! on-demand context FST C, compose with H, determinize, minimize, `AddSelfLoops`). With the
//! phone sequence given, every one of those operations has a closed form; see
//! `plans/research/02_openfst_path.md`.
//!
//! What is reproduced here, arc for arc:
//!
//! - The optional-silence structure of MFA's lexicon FST
//!   (`plans/kalpy/kalpy/fstext/lexicon.py:349-476`): optional silence at the start with
//!   `initial_silence_prob`, optional silence after every word with `silence_prob`, and final
//!   corrections on the silence and non-silence hub states.
//! - `GetHmmAsFsa` (`plans/kaldi/src/hmm/hmm-utils.cc`): per phone, an arc per non-self-loop
//!   topology transition, labelled with its transition id and weighted by
//!   `-transition_scale * GetTransitionLogProbIgnoringSelfLoops`.
//! - `AddSelfLoopsReorder` with `reorder = true`, Kaldi's default in `TrainingGraphCompiler` and
//!   therefore what MFA uses: the self-loop is attached to the state the forward transition
//!   *arrives at*, and every arc and final-prob leaving that state is scaled by
//!   `-self_loop_scale * GetNonSelfLoopLogProb`.
//!
//! Triphone context comes from `ContextDependency::compute` over the window
//! `[left, center, right]`, with 0 for positions outside the utterance. At an optional-silence
//! junction the neighbour depends on which branch is taken, so both variants of the affected
//! phone HMM are emitted, exactly as composition with C would give.

use super::context::ContextDependency;
use super::selfloops::add_self_loops;
use super::topology::HmmTopology;
use super::transition::TransitionModel;
use crate::types::{PdfId, PhoneId, Pronunciation, TransitionId, WordId};

/// Sentinel in [`Arc::word`] meaning "this arc carries no word label".
pub const NO_WORD: WordId = WordId::MAX;

/// Sentinel in [`Arc::pron`] meaning "this arc selects no pronunciation".
pub const NO_PRON: u32 = u32::MAX;

/// An arc of the decoding graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arc {
    /// Transition id, or 0 for a non-emitting (epsilon) arc.
    pub tid: TransitionId,
    /// Word label, or [`NO_WORD`].
    pub word: WordId,
    /// Which pronunciation of `word` this arc enters, or [`NO_PRON`]. Set on exactly the
    /// arcs that carry a word label, so the Viterbi backtrace recovers the chosen pron.
    pub pron: u32,
    pub next: u32,
    /// `-log prob`, with `transition_scale` / `self_loop_scale` already applied.
    pub cost: f32,
    /// The part of `cost` that does not come from the transition model (lexicon and
    /// silence probabilities, fixed topology probabilities of non-emitting states).
    /// [`Graph::apply_transition_probs`] rebuilds `cost` from it.
    pub lm: f32,
}

#[derive(Clone, Debug, Default)]
pub struct GraphState {
    pub arcs: Vec<Arc>,
}

/// A decoding graph. State 0 is the start state.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub states: Vec<GraphState>,
    /// `(state, final cost)`.
    pub finals: Vec<(u32, f32)>,
    /// The transition-model-free part of each final cost, parallel to `finals`.
    finals_lm: Vec<f32>,
    /// Transition-state of the arcs entering each state (`AddSelfLoopsReorder`'s
    /// `state_in`), recorded when the self-loops are added.
    state_in: Vec<Option<u32>>,
}

impl Graph {
    pub(super) fn add_state(&mut self) -> u32 {
        self.states.push(GraphState::default());
        (self.states.len() - 1) as u32
    }

    pub(super) fn add_arc(&mut self, from: u32, arc: Arc) {
        self.states[from as usize].arcs.push(arc);
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn final_cost(&self, state: u32) -> Option<f32> {
        self.finals
            .iter()
            .find(|(s, _)| *s == state)
            .map(|(_, c)| *c)
    }

    pub fn is_final(&self, state: u32) -> bool {
        self.finals.iter().any(|(s, _)| *s == state)
    }

    /// Record the entering transition-states and the final costs' model-free part;
    /// called once, after the self-loop arcs are in place and before any costing.
    pub(super) fn set_state_in(&mut self, state_in: Vec<Option<u32>>) {
        self.state_in = state_in;
        self.finals_lm = self.finals.iter().map(|&(_, c)| c).collect();
    }

    /// Re-cost every arc and final from the current transition model, the way Kaldi's
    /// training compiles graphs with zero scales and `gmm-align-compiled` adds the
    /// model's probabilities on every pass (`AddTransitionProbs`, hmm-utils.cc:1065):
    /// forward arcs cost `-transition_scale * log p(tid)` ignoring self-loops, and
    /// everything leaving (or ending at) a state entered by transition-state `T`
    /// carries `-self_loop_scale * GetNonSelfLoopLogProb(T)`, with the self-loop
    /// itself at `-self_loop_scale * log p(loop)` (`AddSelfLoopsReorder`).
    ///
    /// Training must call this before each realignment: the probabilities are
    /// re-estimated every iteration, and a graph costed from the initial topology keeps
    /// the state-skips at `1/3` all stage long, where the trained model puts them near
    /// the floor. That let one- and two-frame phones stay cheap and biased stops and
    /// closures by ~10 ms against hand labels (issue #12).
    pub fn apply_transition_probs(
        &mut self,
        tm: &TransitionModel,
        transition_scale: f32,
        self_loop_scale: f32,
    ) {
        debug_assert_eq!(self.state_in.len(), self.states.len());
        let state_in = &self.state_in;
        for (s, st) in self.states.iter_mut().enumerate() {
            let non_self_loop = state_in[s].map_or(0.0, |ts| {
                -self_loop_scale * tm.get_non_self_loop_log_prob(ts)
            });
            for a in &mut st.arcs {
                let is_loop = a.tid != 0 && a.next as usize == s && tm.is_self_loop(a.tid);
                a.cost = if is_loop {
                    -self_loop_scale * tm.get_transition_log_prob(a.tid)
                } else if a.tid != 0 {
                    a.lm - transition_scale * tm.get_transition_log_prob_ignoring_self_loops(a.tid)
                        + non_self_loop
                } else {
                    a.lm + non_self_loop
                };
            }
        }
        for (i, (s, c)) in self.finals.iter_mut().enumerate() {
            *c = self.finals_lm[i]
                + state_in[*s as usize].map_or(0.0, |ts| {
                    -self_loop_scale * tm.get_non_self_loop_log_prob(ts)
                });
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GraphOptions {
    pub transition_scale: f32,
    pub self_loop_scale: f32,
    /// The optional-silence phone, inserted at the start, between words, and at the end.
    pub silence_phone: PhoneId,
    /// MFA `silence_probability`: probability of silence following a word.
    pub silence_prob: f32,
    /// MFA `initial_silence_probability`.
    pub initial_silence_prob: f32,
    /// MFA `final_silence_correction` / `final_non_silence_correction`; 1.0 means no correction.
    pub final_silence_correction: f32,
    pub final_non_silence_correction: f32,
    /// If true, the caller has already applied word-position tags; the graph never retags.
    pub position_dependent: bool,
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            transition_scale: 1.0,
            self_loop_scale: 0.1,
            silence_phone: 1,
            silence_prob: 0.5,
            initial_silence_prob: 0.5,
            final_silence_correction: 1.0,
            final_non_silence_correction: 1.0,
            position_dependent: true,
        }
    }
}

/// `-log p`, with `p <= 0` mapping to infinity and `p == 1` to exactly zero.
fn neg_log(p: f32) -> f32 {
    if p <= 0.0 { f32::INFINITY } else { -p.ln() }
}

/// Builder state: a chain of "positions", where every position is a set of alternative
/// (left context, phone) pairs sharing an entry state.
struct Builder<'a> {
    graph: Graph,
    tm: &'a TransitionModel,
    ctx: &'a ContextDependency,
    topo: &'a HmmTopology,
    opts: &'a GraphOptions,
}

impl<'a> Builder<'a> {
    /// Emit one phone HMM between `from` and `to`, with the given context window and word label.
    ///
    /// This is `GetHmmAsFsa` followed by `AddSelfLoopsReorder`, fused. The topology's states map
    /// to freshly allocated graph states; state 0 of the topology is `from`, and its final state
    /// is `to`.
    ///
    /// With `reorder = true` the self-loop of transition-state `T` lives on the state that `T`'s
    /// forward arc *enters*, not the state it leaves. Kaldi achieves this by duplicating states
    /// until each has a single distinct incoming input symbol, then attaching one self-loop per
    /// state; because our topologies are left-to-right chains built one phone at a time, each
    /// interior state already has exactly one incoming transition-state, so the duplication is a
    /// no-op and we can attach the self-loop directly.
    ///
    /// This function emits *only* the forward arcs (that is `GetHmmAsFsa` with
    /// `transition_scale` applied). The self-loops and the `GetNonSelfLoopLogProb` rescaling of
    /// outgoing arcs are done afterwards, once the whole graph exists, by [`add_self_loops`] --
    /// they have to be, because an HMM's exit state is shared with whatever follows it, and in
    /// the reorder convention that exit state owns the last emitting state's self-loop and its
    /// outgoing arcs (which belong to the *next* phone) carry the rescaling.
    fn emit_phone(&mut self, window: &[PhoneId], from: u32, to: u32, word: WordId, pron: u32) {
        let phone = window[self.ctx.p];
        let entry = self.topo.topology_for_phone(phone).to_vec();
        let num_states = entry.len();

        // pdf per pdf-class for this context window.
        let num_classes = self.topo.num_pdf_classes(phone);
        let mut pdfs = Vec::with_capacity(num_classes);
        for class in 0..num_classes as i32 {
            let pdf = self.ctx.compute(window, class).unwrap_or_else(|| {
                panic!(
                    "context-dependency object produced no answer for pdf-class {class}, \
                     window {window:?}"
                )
            });
            pdfs.push(pdf);
        }

        // Graph state per topology state. Topology state 0 is `from`; the final one is `to`.
        let mut gstate = vec![0u32; num_states];
        gstate[0] = from;
        gstate[num_states - 1] = to;
        for s in 1..num_states - 1 {
            gstate[s] = self.graph.add_state();
        }

        // Forward (non-self-loop) arcs, as GetHmmAsFsa emits them.
        struct Pending {
            from_state: usize,
            dest_state: usize,
            tid: TransitionId,
            log_prob: f32,
        }
        let mut pending = Vec::new();
        for (hmm_state, st) in entry.iter().enumerate() {
            for (trans_idx, &(dest, prob)) in st.transitions.iter().enumerate() {
                if dest == hmm_state {
                    continue; // self-loops are added below, not here
                }
                let (tid, log_prob) = if st.is_emitting() {
                    let pdf = pdfs[st.pdf_class as usize];
                    let tstate = self
                        .tm
                        .tuple_to_transition_state(phone, hmm_state, pdf, pdf);
                    let tid = self.tm.pair_to_transition_id(tstate, trans_idx as u32);
                    (
                        tid,
                        self.tm.get_transition_log_prob_ignoring_self_loops(tid),
                    )
                } else {
                    // Non-emitting state: no transition-state, unestimated probability.
                    (0, prob.ln())
                };
                pending.push(Pending {
                    from_state: hmm_state,
                    dest_state: dest,
                    tid,
                    log_prob,
                });
            }
        }

        for p in &pending {
            // Emitting arcs are costed from the model by `apply_transition_probs`; only a
            // non-emitting state's fixed topology probability is model-free.
            let lm = if p.tid == 0 {
                -self.opts.transition_scale * p.log_prob
            } else {
                0.0
            };
            // The word label goes on the first emitting arc of the word (equivalently, the first
            // arc out of the HMM's start state, which is emitting for every topology we build).
            let (word_label, pron_label) = if p.from_state == 0 {
                (word, pron)
            } else {
                (NO_WORD, NO_PRON)
            };
            self.graph.add_arc(
                gstate[p.from_state],
                Arc {
                    tid: p.tid,
                    word: word_label,
                    pron: pron_label,
                    next: gstate[p.dest_state],
                    cost: lm,
                    lm,
                },
            );
        }
    }
}

/// Which side of an optional-silence junction a path is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Branch {
    /// Came through silence.
    Silence,
    /// Came directly, no silence.
    Direct,
}

/// Pron cost, MFA `lexicon.py:519-523` / `:638-643`: `|ln p|` with `p` floored at 0.01.
fn pron_cost(prob: Option<f32>) -> f32 {
    match prob {
        None => 0.0,
        Some(p) => {
            let p = if p < 0.01 { 0.01 } else { p };
            p.ln().abs()
        }
    }
}

/// `-ln c` for an optional correction factor; `None` (or 0, as MFA's falsy test treats it) -> 0.
fn corr_cost(c: Option<f32>) -> f32 {
    match c {
        Some(c) if c != 0.0 => neg_log(c),
        _ => 0.0,
    }
}

/// The two costs MFA puts on the optional-silence branch *after* a pronunciation
/// (`lexicon.py:546-560`): the pron's own `silence_after_probability` if given, else the global
/// `silence_probability`.
fn after_costs(pron: &Pronunciation, opts: &GraphOptions) -> (f32, f32) {
    match pron.silence_after_prob {
        Some(p) if p != 0.0 => (neg_log(p), neg_log(1.0 - p)),
        _ => {
            if opts.silence_prob > 0.0 {
                (neg_log(opts.silence_prob), neg_log(1.0 - opts.silence_prob))
            } else {
                (f32::INFINITY, 0.0)
            }
        }
    }
}

/// A place a path can wait between words: one graph state, plus which branch of the
/// optional-silence junction it came through (that decides the left context of what follows)
/// and the phone that context is.
#[derive(Clone, Copy, Debug)]
struct Hub {
    branch: Branch,
    state: u32,
    left: PhoneId,
}

/// Build the per-utterance decoding graph.
///
/// `words[i]` holds word `i`'s candidate pronunciations, already position-tagged if
/// `opts.position_dependent`. Every pronunciation is a parallel branch; the word index and the
/// pronunciation index are tagged together on the first emitting arc of that branch, so the
/// Viterbi backtrace recovers both.
///
/// The silence structure follows MFA's `_create_word_fst` / `create_fsts`
/// (`plans/kalpy/kalpy/fstext/lexicon.py:349-600`): MFA's three-state hub means the junction
/// between word `i` and word `i+1` carries `after_i(sil) + before_{i+1}(sil corr)` on the silence
/// path and `after_i(nosil) + before_{i+1}(nonsil corr)` on the direct path.
pub fn build_graph(
    words: &[Vec<Pronunciation>],
    tm: &TransitionModel,
    ctx: &ContextDependency,
    opts: &GraphOptions,
) -> Graph {
    let topo = tm.topology();
    let mut b = Builder {
        graph: Graph::default(),
        tm,
        ctx,
        topo,
        opts,
    };
    b.graph.add_state(); // start state 0

    // Drop words with no usable pronunciation; the remaining ones keep their original index so
    // the word labels still point into the caller's `words`.
    let live: Vec<(WordId, &Vec<Pronunciation>)> = words
        .iter()
        .enumerate()
        .filter(|(_, ps)| ps.iter().any(|p| !p.phones.is_empty()))
        .map(|(i, ps)| (i as WordId, ps))
        .collect();

    if live.is_empty() {
        return build_silence_only(&mut b);
    }

    // First phones of the next word's pronunciations: the right-context alternatives a word's
    // last phone sees when no silence follows.
    let first_phones_of = |k: usize| -> Vec<PhoneId> {
        match live.get(k) {
            None => vec![0],
            Some((_, ps)) => {
                let mut v: Vec<PhoneId> = ps
                    .iter()
                    .filter(|p| !p.phones.is_empty())
                    .map(|p| p.phones[0])
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            }
        }
    };

    // Junction 0: the initial optional silence (`create_fsts`, lexicon.py:369-424).
    let mut hubs: Vec<Hub> = Vec::new();
    {
        let start = 0u32;
        let sil_ok = opts.initial_silence_prob > 0.0;
        if sil_ok {
            let direct = b.graph.add_state();
            b.graph.add_arc(
                start,
                Arc {
                    tid: 0,
                    word: NO_WORD,
                    pron: NO_PRON,
                    next: direct,
                    cost: neg_log(1.0 - opts.initial_silence_prob),
                    lm: neg_log(1.0 - opts.initial_silence_prob),
                },
            );
            hubs.push(Hub {
                branch: Branch::Direct,
                state: direct,
                left: 0,
            });

            let sil_in = b.graph.add_state();
            b.graph.add_arc(
                start,
                Arc {
                    tid: 0,
                    word: NO_WORD,
                    pron: NO_PRON,
                    next: sil_in,
                    cost: neg_log(opts.initial_silence_prob),
                    lm: neg_log(opts.initial_silence_prob),
                },
            );
            // The silence phone's right context is whatever pronunciation follows; silence is
            // context-independent in every MFA tree, so one variant with the first alternative's
            // phone is enough and matches what composition with C collapses to.
            let right = *first_phones_of(0).first().unwrap_or(&0);
            let sil_out = b.graph.add_state();
            let window = make_window(ctx, 0, opts.silence_phone, right);
            b.emit_phone(&window, sil_in, sil_out, NO_WORD, NO_PRON);
            hubs.push(Hub {
                branch: Branch::Silence,
                state: sil_out,
                left: opts.silence_phone,
            });
        } else {
            hubs.push(Hub {
                branch: Branch::Direct,
                state: start,
                left: 0,
            });
        }
    }

    for (wi, (word, prons)) in live.iter().enumerate() {
        let next_firsts = first_phones_of(wi + 1);
        let last_word = wi + 1 == live.len();
        let mut next_hubs: Vec<Hub> = Vec::new();

        for (pi, pron) in prons.iter().enumerate() {
            if pron.phones.is_empty() {
                continue;
            }
            let entry_cost = pron_cost(pron.prob);
            let sil_before = corr_cost(pron.silence_before_correction);
            let nonsil_before = corr_cost(pron.non_silence_before_correction);
            let (sil_after, nonsil_after) = after_costs(pron, opts);

            // Entry state of this pronunciation, one per incoming branch: the branch decides
            // both the left context of the first phone and the before-correction paid.
            let mut entries: Vec<(u32, PhoneId)> = Vec::new();
            for hub in &hubs {
                let before = match hub.branch {
                    Branch::Silence => sil_before,
                    Branch::Direct => nonsil_before,
                };
                let cost = entry_cost + before;
                if !cost.is_finite() {
                    continue;
                }
                let e = b.graph.add_state();
                b.graph.add_arc(
                    hub.state,
                    Arc {
                        tid: 0,
                        word: NO_WORD,
                        pron: NO_PRON,
                        next: e,
                        cost,
                        lm: cost,
                    },
                );
                entries.push((e, hub.left));
            }
            if entries.is_empty() {
                continue;
            }

            // Right-context alternatives after this pronunciation's last phone: silence, or each
            // possible first phone of the next word (0 at the end of the utterance).
            let mut rights: Vec<(Branch, PhoneId)> = Vec::new();
            if sil_after.is_finite() {
                rights.push((Branch::Silence, opts.silence_phone));
            }
            if nonsil_after.is_finite() {
                for &f in &next_firsts {
                    rights.push((Branch::Direct, f));
                }
            }
            if rights.is_empty() {
                continue;
            }

            // Walk the pronunciation's phones. `cur` holds (state, left phone) pairs; only the
            // last phone fans out over the right-context alternatives.
            let n = pron.phones.len();
            let mut cur: Vec<(u32, PhoneId)> = entries;
            for (k, &phone) in pron.phones.iter().enumerate() {
                let is_first = k == 0;
                let (word_label, pron_label) = if is_first {
                    (*word, pi as u32)
                } else {
                    (NO_WORD, NO_PRON)
                };
                if k + 1 < n {
                    let right = pron.phones[k + 1];
                    let exit = b.graph.add_state();
                    for &(from, left) in &cur {
                        let window = make_window(ctx, left, phone, right);
                        b.emit_phone(&window, from, exit, word_label, pron_label);
                    }
                    cur = vec![(exit, phone)];
                } else {
                    // Last phone: one exit per right-context alternative.
                    let exits: Vec<u32> = rights.iter().map(|_| b.graph.add_state()).collect();
                    for &(from, left) in &cur {
                        for (ri, &(_, right)) in rights.iter().enumerate() {
                            let window = make_window(ctx, left, phone, right);
                            b.emit_phone(&window, from, exits[ri], word_label, pron_label);
                        }
                    }
                    // Pay the after-costs and, on the silence branch, emit the silence HMM.
                    for (ri, &(branch, _)) in rights.iter().enumerate() {
                        match branch {
                            Branch::Direct => {
                                let s = b.graph.add_state();
                                b.graph.add_arc(
                                    exits[ri],
                                    Arc {
                                        tid: 0,
                                        word: NO_WORD,
                                        pron: NO_PRON,
                                        next: s,
                                        cost: nonsil_after,
                                        lm: nonsil_after,
                                    },
                                );
                                next_hubs.push(Hub {
                                    branch: Branch::Direct,
                                    state: s,
                                    left: phone,
                                });
                            }
                            Branch::Silence => {
                                let sil_in = b.graph.add_state();
                                b.graph.add_arc(
                                    exits[ri],
                                    Arc {
                                        tid: 0,
                                        word: NO_WORD,
                                        pron: NO_PRON,
                                        next: sil_in,
                                        cost: sil_after,
                                        lm: sil_after,
                                    },
                                );
                                let sil_out = b.graph.add_state();
                                let right = if last_word {
                                    0
                                } else {
                                    *next_firsts.first().unwrap_or(&0)
                                };
                                let window = make_window(ctx, phone, opts.silence_phone, right);
                                b.emit_phone(&window, sil_in, sil_out, NO_WORD, NO_PRON);
                                next_hubs.push(Hub {
                                    branch: Branch::Silence,
                                    state: sil_out,
                                    left: opts.silence_phone,
                                });
                            }
                        }
                    }
                }
            }
        }

        if next_hubs.is_empty() {
            // Nothing survived (every branch had infinite cost); fall back to silence only.
            b.graph = Graph::default();
            b.graph.add_state();
            return build_silence_only(&mut b);
        }
        hubs = next_hubs;
    }

    // End of utterance: MFA's final weights on the silence / non-silence hub states.
    let sil_final = neg_log(opts.final_silence_correction);
    let nonsil_final = neg_log(opts.final_non_silence_correction);
    for hub in &hubs {
        let cost = match hub.branch {
            Branch::Silence => sil_final,
            Branch::Direct => nonsil_final,
        };
        if cost.is_finite() {
            b.graph.finals.push((hub.state, cost));
        }
    }

    let mut graph = std::mem::take(&mut b.graph);
    add_self_loops(&mut graph, tm);
    graph.apply_transition_probs(tm, opts.transition_scale, opts.self_loop_scale);
    graph
}

/// A graph for an utterance with no words: optional silence only.
fn build_silence_only(b: &mut Builder<'_>) -> Graph {
    let start = 0u32;
    let end = b.graph.add_state();
    let sil = b.opts.silence_phone;
    let window = make_window(b.ctx, 0, sil, 0);
    b.emit_phone(&window, start, end, NO_WORD, NO_PRON);
    b.graph
        .finals
        .push((end, neg_log(b.opts.final_silence_correction)));
    let mut graph = std::mem::take(&mut b.graph);
    add_self_loops(&mut graph, b.tm);
    graph.apply_transition_probs(b.tm, b.opts.transition_scale, b.opts.self_loop_scale);
    graph
}

/// Build the context window the tree expects. For `n == 1` only the phone itself matters; for
/// `n == 3, p == 1` it is `[left, center, right]`. Wider or off-centre windows fill the extra
/// positions with 0 ("no phone here"), which is what Kaldi's context FST supplies at the edges.
fn make_window(
    ctx: &ContextDependency,
    left: PhoneId,
    center: PhoneId,
    right: PhoneId,
) -> Vec<PhoneId> {
    let mut w = vec![0 as PhoneId; ctx.n];
    w[ctx.p] = center;
    if ctx.p >= 1 {
        w[ctx.p - 1] = left;
    }
    if ctx.p + 1 < ctx.n {
        w[ctx.p + 1] = right;
    }
    w
}

/// Every pdf reachable in the graph, sorted and unique. Used to request a scores matrix holding
/// only the columns the graph can actually use.
pub fn graph_pdfs(graph: &Graph, tm: &TransitionModel) -> Vec<PdfId> {
    let mut pdfs: Vec<PdfId> = graph
        .states
        .iter()
        .flat_map(|s| s.arcs.iter())
        .filter(|a| a.tid != 0)
        .map(|a| tm.transition_id_to_pdf(a.tid))
        .collect();
    pdfs.sort_unstable();
    pdfs.dedup();
    pdfs
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
