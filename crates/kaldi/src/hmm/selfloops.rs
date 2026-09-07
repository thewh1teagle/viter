//! `AddSelfLoopsReorder` and the state-splitting it needs, split out of [`super::graph`].
//!
//! Kaldi runs these over the finished graph, not per phone, because an HMM's exit state is
//! shared with whatever follows it; see the module docs of [`super::graph`].

use super::graph::{Arc, Graph, NO_PRON, NO_WORD};
use super::transition::TransitionModel;
use crate::types::TransitionId;

/// Kaldi's `MakePrecedingInputSymbolsSameClass` (`fstext/fstext-utils-inl.h`), with the class
/// function being `TidToTstateMapper`: epsilon maps to class `None`, a transition-id to its
/// transition-state.
///
/// Duplicates any state entered by arcs of more than one class, one copy per class, so that
/// afterwards every state has a single well-defined entering transition-state. This is what makes
/// the reorder self-loop attachment well-defined: a shared HMM exit state that two different
/// transition-states lead into (which happens here whenever the same phone is emitted with two
/// different right contexts, giving different pdfs and hence different transition-states) must be
/// split, or the two paths would share one self-loop.
///
/// Returns the entering transition-state of every state of the (possibly enlarged) graph.
pub(super) fn make_preceding_input_symbols_same_class(
    graph: &mut Graph,
    tm: &TransitionModel,
) -> Vec<Option<u32>> {
    // Classes entering each original state.
    let orig_n = graph.num_states();
    let class_of = |tid: TransitionId| -> Option<u32> {
        if tid == 0 {
            None
        } else {
            Some(tm.transition_id_to_transition_state(tid))
        }
    };

    let mut classes: Vec<Vec<Option<u32>>> = vec![Vec::new(); orig_n];
    for st in &graph.states {
        for a in &st.arcs {
            let c = class_of(a.tid);
            let v = &mut classes[a.next as usize];
            if !v.contains(&c) {
                v.push(c);
            }
        }
    }

    // For states with more than one entering class, allocate a copy per extra class. The
    // original state keeps class[0]; copies take the rest and duplicate its outgoing arcs and
    // final-prob.
    let mut copy_for: Vec<Vec<(Option<u32>, u32)>> = vec![Vec::new(); orig_n];
    for s in 0..orig_n {
        if classes[s].len() <= 1 {
            continue;
        }
        let cs = classes[s].clone();
        copy_for[s].push((cs[0], s as u32));
        for &c in &cs[1..] {
            let new_s = graph.add_state();
            let arcs = graph.states[s].arcs.clone();
            graph.states[new_s as usize].arcs = arcs;
            if let Some(fc) = graph.final_cost(s as u32) {
                graph.finals.push((new_s, fc));
            }
            copy_for[s].push((c, new_s));
        }
    }

    // Repoint every arc at the copy matching its own class.
    for s in 0..graph.num_states() {
        let arcs = std::mem::take(&mut graph.states[s].arcs);
        let arcs = arcs
            .into_iter()
            .map(|mut a| {
                let dest = a.next as usize;
                if dest < orig_n && !copy_for[dest].is_empty() {
                    let c = class_of(a.tid);
                    if let Some(&(_, target)) = copy_for[dest].iter().find(|(cc, _)| *cc == c) {
                        a.next = target;
                    }
                }
                a
            })
            .collect();
        graph.states[s].arcs = arcs;
    }

    // Now every state has at most one entering class; read it back off the final graph.
    let mut state_in: Vec<Option<u32>> = vec![None; graph.num_states()];
    for st in &graph.states {
        for a in &st.arcs {
            let c = class_of(a.tid);
            if c.is_some() {
                state_in[a.next as usize] = c;
            }
        }
    }
    state_in
}

/// The structural half of Kaldi's `AddSelfLoopsReorder` (`hmm-utils.cc:472`), run over
/// the finished graph: add the self-loop arcs and record each state's entering
/// transition-state. The costs — the loop's own and the `GetNonSelfLoopLogProb`
/// rescaling of everything leaving that state — are applied by
/// [`Graph::apply_transition_probs`], which training re-runs whenever the model's
/// probabilities change.
///
/// For every state `s`, `state_in[s]` is the transition-state of the arcs entering it (Kaldi
/// guarantees this is unique by duplicating states in `MakePrecedingInputSymbolsSameClass`; our
/// construction already gives at most one distinct entering transition-state per state, which is
/// asserted in debug builds). In the reorder convention that state carries the entering
/// transition's self-loop, and the arcs leaving it -- which belong to the *next* phone, or are
/// the epsilon arcs of the optional-silence structure -- are the ones that get rescaled, so
/// this must run on the whole graph rather than per phone.
pub(super) fn add_self_loops(graph: &mut Graph, tm: &TransitionModel) {
    let state_in = make_preceding_input_symbols_same_class(graph, tm);
    for (s, ts) in state_in.iter().enumerate() {
        let Some(ts) = ts else { continue };
        if let Some(loop_tid) = tm.self_loop_of(*ts) {
            graph.states[s].arcs.push(Arc {
                tid: loop_tid,
                word: NO_WORD,
                pron: NO_PRON,
                next: s as u32,
                cost: 0.0,
                lm: 0.0,
            });
        }
    }
    graph.set_state_in(state_in);
}
