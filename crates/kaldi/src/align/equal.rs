//! Port of `fst::EqualAlign` (`plans/kaldi/src/fstext/fstext-utils-inl.h:857`), used for the
//! monophone flat start (`gmm_align_equal`).
//!
//! A path is drawn through the graph, ignoring self-loops; then self-loops are inserted
//! along it, spread as evenly as possible, until the path has exactly `num_frames` emitting arcs.
//!
//! Kaldi draws the path uniformly at random over each state's arcs, which was written for
//! topologies whose only choices are the optional silences. MFA's phone topology also has
//! state-skip arcs (state 0 to state 2 or straight to the exit), so under Kaldi's rule two
//! thirds of the phones start the flat start with a state or the whole phone skipped, and
//! the trained model depends on the draw: on TIMIT the monophone model's boundaries within
//! 10 ms of the hand labels varied from 52% to 58% across seeds, with a tenth of the
//! utterances failing the first alignment on bad draws. MFA itself sees one fixed draw
//! (`srand(1234)` before every utterance). Here the emitting choice is the smallest forward
//! step, so every phone visits all of its states and the frames are split between them;
//! only the optional silences are still drawn at random, as in Kaldi. That gives 57-58%
//! whatever the seed, with no first-iteration failures.

use crate::hmm::{Graph, NO_WORD, TransitionModel};
use crate::types::{Alignment, TransitionId, WordId};
use rand::{Rng, RngExt};

/// Kaldi's `num_retries` default in `EqualAlign`.
const NUM_RETRIES: usize = 10;

/// Index of a self-loop arc with a non-epsilon input label out of `state`, if any.
/// Kaldi `FindSelfLoopWithILabel`.
fn find_self_loop(graph: &Graph, state: u32) -> Option<usize> {
    graph.states[state as usize]
        .arcs
        .iter()
        .position(|a| a.next == state && a.tid != 0)
}

/// The arc out of `state` that advances its HMM by the fewest states, if it has emitting
/// arcs: the flat start walks every state of every phone (see the module doc). The silence
/// topology has backward arcs, so only forward steps count.
fn smallest_forward_step(graph: &Graph, tm: &TransitionModel, state: u32) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (destination hmm state, arc index)
    for (i, a) in graph.states[state as usize].arcs.iter().enumerate() {
        if a.tid == 0 || a.next == state {
            continue;
        }
        let ts = tm.transition_id_to_transition_state(a.tid);
        let src = tm.transition_state_to_hmm_state(ts);
        let idx = tm.transition_id_to_transition_index(a.tid);
        let phone = tm.transition_state_to_phone(ts);
        let dest = tm.topology().topology_for_phone(phone)[src].transitions[idx].0;
        if dest > src && best.is_none_or(|b| dest < b.0) {
            best = Some((dest, i));
        }
    }
    best.map(|b| b.1)
}

/// Kaldi `EqualAlign`: build an alignment of exactly `num_frames` frames by choosing a
/// path through the graph (emitting arcs: the smallest forward step; optional silences:
/// at random) and padding it with self-loops.
///
/// Returns `None` if even the shortest randomly drawn path is longer than `num_frames`, or if the
/// path has no self-loops to lengthen it with.
pub fn equal_align(
    graph: &Graph,
    tm: &TransitionModel,
    num_frames: usize,
    rng: &mut impl Rng,
) -> Option<Alignment> {
    if graph.states.is_empty() {
        tracing::warn!("EqualAlign: empty graph");
        return None;
    }

    // First select a path through the graph. `path` holds the states visited; `arc_offsets[i]`
    // is the arc index taken out of `path[i]`.
    let mut path: Vec<u32> = Vec::new();
    let mut arc_offsets: Vec<usize> = Vec::new();
    let mut num_ilabels = 0usize;
    let mut attempted: Vec<usize> = Vec::new();

    let mut ended_final = false;
    for _ in 0..NUM_RETRIES {
        num_ilabels = 0;
        arc_offsets.clear();
        path.clear();
        path.push(0);

        // Guard against pathological graphs: a random walk that keeps failing to terminate.
        let mut steps = 0usize;
        let max_steps = 100 * (num_frames + graph.num_states() + 1);
        loop {
            let s = *path.last().unwrap();
            let num_arcs = graph.states[s as usize].arcs.len();
            let mut num_arcs_tot = num_arcs;
            if graph.is_final(s) {
                num_arcs_tot += 1;
            }
            if num_arcs_tot == 0 {
                // Dead end with no final-prob: this path cannot be completed.
                break;
            }
            steps += 1;
            if steps > max_steps {
                break;
            }
            let offset = smallest_forward_step(graph, tm, s)
                .unwrap_or_else(|| rng.random_range(0..num_arcs_tot));
            if offset < num_arcs {
                let arc = graph.states[s as usize].arcs[offset];
                if arc.next == s {
                    continue; // don't take this self-loop arc
                }
                arc_offsets.push(offset);
                path.push(arc.next);
                if arc.tid != 0 {
                    num_ilabels += 1;
                }
            } else {
                break; // chose the final-prob
            }
        }
        attempted.push(num_ilabels);

        // Kaldi retries while the drawn path is too long. A path that ran into a dead end rather
        // than a final state is unusable, so it is retried too. Under the MFA non-silence
        // topology the first state of a phone has a skip arc straight to the phone exit, so a
        // drawn path can also be too *short* to pad: if it visits no state carrying a self-loop
        // it cannot be stretched to `num_frames`, and that is worth another draw as well.
        ended_final = path.last().is_some_and(|&s| graph.is_final(s));
        let paddable =
            num_ilabels == num_frames || path.iter().any(|&s| find_self_loop(graph, s).is_some());
        if ended_final && num_ilabels <= num_frames && paddable {
            break;
        }
    }

    if !ended_final {
        tracing::warn!("EqualAlign: could not draw a path reaching a final state");
        return None;
    }
    if num_ilabels > num_frames {
        tracing::warn!(
            ?attempted,
            num_frames,
            "EqualAlign: utterance has too few frames to align"
        );
        return None;
    }

    let self_loop_offsets: Vec<Option<usize>> =
        path.iter().map(|&s| find_self_loop(graph, s)).collect();
    let num_self_loops = self_loop_offsets.iter().filter(|o| o.is_some()).count();

    if num_self_loops == 0 && num_ilabels < num_frames {
        tracing::warn!("EqualAlign: no self-loops on the chosen path; cannot match length");
        return None;
    }

    let num_extra = num_frames - num_ilabels;
    let min_num_loops = if num_extra != 0 && num_self_loops != 0 {
        num_extra / num_self_loops
    } else {
        0
    };
    let num_with_one_more = num_extra - min_num_loops * num_self_loops;

    let mut tids: Vec<TransitionId> = Vec::with_capacity(num_frames);
    let mut words: Vec<WordId> = Vec::new();
    let mut prons: Vec<u32> = Vec::new();
    let mut counter = 0usize;
    let mut total_cost = 0.0f64;

    for (i, &state) in path.iter().enumerate() {
        // First, add any self-loops that are needed here.
        if let Some(off) = self_loop_offsets[i] {
            let num_loops = min_num_loops + usize::from(counter < num_with_one_more);
            counter += 1;
            let arc = graph.states[state as usize].arcs[off];
            for _ in 0..num_loops {
                tids.push(arc.tid);
                if arc.word != NO_WORD {
                    words.push(arc.word);
                    prons.push(arc.pron);
                }
                total_cost += arc.cost as f64;
            }
        }
        if i + 1 < path.len() {
            let arc = graph.states[state as usize].arcs[arc_offsets[i]];
            debug_assert_eq!(arc.next, path[i + 1]);
            if arc.tid != 0 {
                tids.push(arc.tid);
            }
            if arc.word != NO_WORD {
                words.push(arc.word);
                prons.push(arc.pron);
            }
            total_cost += arc.cost as f64;
        } else {
            total_cost += graph.final_cost(state).unwrap_or(0.0) as f64;
        }
    }

    if tids.len() != num_frames {
        tracing::warn!(
            got = tids.len(),
            want = num_frames,
            "EqualAlign: produced the wrong number of frames"
        );
        return None;
    }
    debug_assert!(
        tids.iter()
            .all(|&t| (t as usize) <= tm.num_transition_ids())
    );

    Some(Alignment {
        utt: String::new(),
        tids,
        words,
        prons,
        // Graph cost only; a flat-start alignment has no acoustic score.
        loglike: -total_cost as f32,
    })
}

#[cfg(test)]
mod tests {

    /// `build_graph` input for words with a single plain pronunciation each.
    use crate::types::{PhoneId, Pronunciation};

    fn plain(words: &[&[PhoneId]]) -> Vec<Vec<Pronunciation>> {
        words
            .iter()
            .map(|p| vec![Pronunciation::plain(p.to_vec())])
            .collect()
    }
    use super::*;
    use crate::hmm::{ContextDependency, GraphOptions, HmmTopology, build_graph};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn setup() -> (TransitionModel, ContextDependency, GraphOptions) {
        let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
        let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3]];
        let t = topo.clone();
        let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
        let tm = TransitionModel::new(&ctx, &topo);
        let opts = GraphOptions {
            silence_phone: 1,
            silence_prob: 0.0,
            initial_silence_prob: 0.0,
            ..Default::default()
        };
        (tm, ctx, opts)
    }

    #[test]
    fn produces_exactly_num_frames() {
        let (tm, ctx, gopts) = setup();
        let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for frames in [6usize, 10, 25, 100] {
            let ali = equal_align(&g, &tm, frames, &mut rng)
                .unwrap_or_else(|| panic!("equal_align failed for {frames} frames"));
            assert_eq!(ali.tids.len(), frames);
        }
    }

    #[test]
    fn output_splits_into_the_right_phones() {
        let (tm, ctx, gopts) = setup();
        let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let ali = equal_align(&g, &tm, 20, &mut rng).unwrap();
        let runs = crate::hmm::split_to_phones(&tm, &ali.tids);
        assert_eq!(runs.len(), 2);
        assert_eq!(tm.transition_id_to_phone(runs[0][0]), 2);
        assert_eq!(tm.transition_id_to_phone(runs[1][0]), 3);
    }

    #[test]
    fn lengths_are_roughly_equal() {
        let (tm, ctx, gopts) = setup();
        let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let ali = equal_align(&g, &tm, 60, &mut rng).unwrap();
        let runs = crate::hmm::split_to_phones(&tm, &ali.tids);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs.iter().map(|r| r.len()).sum::<usize>(), 60);
        // The padding is spread evenly over the self-loop-bearing states on the drawn path, but
        // under the MFA topology the skip arc out of state 0 means the two phones need not carry
        // the same number of such states, so only a loose balance is guaranteed: no phone may be
        // starved of frames.
        for r in &runs {
            assert!(r.len() >= 2, "phone run length {} is too short", r.len());
        }
    }

    #[test]
    fn too_few_frames_returns_none() {
        let (tm, ctx, gopts) = setup();
        let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        // Under the MFA topology each Bakis phone can be traversed in a single emitting arc via
        // the state-0 skip, so two phones need two frames; one frame is impossible.
        assert!(equal_align(&g, &tm, 1, &mut rng).is_none());
    }

    #[test]
    fn deterministic_for_a_fixed_seed() {
        let (tm, ctx, gopts) = setup();
        let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
        let a = equal_align(&g, &tm, 15, &mut Xoshiro256PlusPlus::seed_from_u64(42)).unwrap();
        let b = equal_align(&g, &tm, 15, &mut Xoshiro256PlusPlus::seed_from_u64(42)).unwrap();
        assert_eq!(a.tids, b.tids);
    }

    #[test]
    fn word_labels_present() {
        let (tm, ctx, gopts) = setup();
        let g = build_graph(&plain(&[&[2], &[3]]), &tm, &ctx, &gopts);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let ali = equal_align(&g, &tm, 18, &mut rng).unwrap();
        assert_eq!(ali.words, vec![0, 1]);
    }
}
