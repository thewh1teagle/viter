//! Tests for [`super`]: per-utterance decoding graph construction.

use super::*;
use crate::hmm::context::ContextDependency;
use crate::hmm::topology::HmmTopology;

/// `build_graph` input for words with a single plain pronunciation each.
use crate::types::{PhoneId, Pronunciation};

fn plain(words: &[&[PhoneId]]) -> Vec<Vec<Pronunciation>> {
    words
        .iter()
        .map(|p| vec![Pronunciation::plain(p.to_vec())])
        .collect()
}

fn mono_setup() -> (TransitionModel, ContextDependency, GraphOptions) {
    let topo = HmmTopology::mfa_default(&[1], &[2, 3, 4], 5, 3);
    let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3], vec![4]];
    let t = topo.clone();
    let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
    let tm = TransitionModel::new(&ctx, &topo);
    let opts = GraphOptions {
        silence_phone: 1,
        ..Default::default()
    };
    (tm, ctx, opts)
}

/// Every state must be able to reach a final state, or the decoder can dead-end.
fn assert_coaccessible(g: &Graph) {
    let n = g.num_states();
    let mut back: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (s, st) in g.states.iter().enumerate() {
        for a in &st.arcs {
            back[a.next as usize].push(s as u32);
        }
    }
    let mut seen = vec![false; n];
    let mut stack: Vec<u32> = g.finals.iter().map(|(s, _)| *s).collect();
    for s in &stack {
        seen[*s as usize] = true;
    }
    while let Some(s) = stack.pop() {
        for &p in &back[s as usize] {
            if !seen[p as usize] {
                seen[p as usize] = true;
                stack.push(p);
            }
        }
    }
    assert!(
        seen.iter().all(|b| *b),
        "graph has states that cannot reach a final state"
    );
}

#[test]
fn single_word_graph_is_wellformed() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &opts);
    assert!(!g.finals.is_empty());
    assert_coaccessible(&g);
    // Every non-epsilon arc carries a valid transition id.
    for st in &g.states {
        for a in &st.arcs {
            assert!(a.tid as usize <= tm.num_transition_ids());
            assert!(a.cost.is_finite(), "arc cost must be finite");
        }
    }
}

#[test]
fn word_label_on_first_arc_only() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2, 3], &[4]]), &tm, &ctx, &opts);
    let mut labels: Vec<WordId> = g
        .states
        .iter()
        .flat_map(|s| s.arcs.iter())
        .filter(|a| a.word != NO_WORD)
        .map(|a| a.word)
        .collect();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels, vec![0, 1]);
    // Every word-labelled arc must be emitting (the first arc of the word's first phone).
    for st in &g.states {
        for a in &st.arcs {
            if a.word != NO_WORD {
                assert_ne!(a.tid, 0, "word label must sit on an emitting arc");
                assert_eq!(tm.transition_id_to_hmm_state(a.tid), 0);
            }
        }
    }
}

#[test]
fn self_loops_are_present_and_scaled() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2]]), &tm, &ctx, &opts);
    let mut found = 0;
    for (s, st) in g.states.iter().enumerate() {
        for a in &st.arcs {
            if a.next as usize == s && a.tid != 0 {
                assert!(tm.is_self_loop(a.tid));
                let expect = -opts.self_loop_scale * tm.get_transition_log_prob(a.tid);
                assert!((a.cost - expect).abs() < 1e-5, "self-loop cost mis-scaled");
                found += 1;
            }
        }
    }
    assert!(found > 0, "graph should contain self-loops");
}

#[test]
fn reorder_places_self_loop_after_forward_transition() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2]]), &tm, &ctx, &opts);
    // Find the arc out of the start region carrying phone 2's hmm-state 0 forward
    // transition, then check the state it lands in owns that same transition-state's
    // self-loop. That is precisely the reorder=true convention.
    for (s, st) in g.states.iter().enumerate() {
        for a in &st.arcs {
            if a.tid == 0 || a.next as usize == s {
                continue;
            }
            if tm.transition_id_to_phone(a.tid) != 2 {
                continue;
            }
            let tstate = tm.transition_id_to_transition_state(a.tid);
            let Some(loop_tid) = tm.self_loop_of(tstate) else {
                continue;
            };
            let dest = &g.states[a.next as usize];
            assert!(
                dest.arcs
                    .iter()
                    .any(|x| x.tid == loop_tid && x.next == a.next),
                "destination of a forward arc must carry that transition-state's self-loop"
            );
        }
    }
}

#[test]
fn optional_silence_branches_exist() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2], &[3]]), &tm, &ctx, &opts);
    // Silence must be reachable: some arc uses a silence transition id.
    let uses_silence = g
        .states
        .iter()
        .flat_map(|s| s.arcs.iter())
        .any(|a| a.tid != 0 && tm.transition_id_to_phone(a.tid) == 1);
    assert!(uses_silence, "optional silence should appear in the graph");
    assert_coaccessible(&g);
}

#[test]
fn silence_probability_costs_match_neg_log() {
    let (tm, ctx, mut opts) = mono_setup();
    opts.silence_prob = 0.25;
    opts.initial_silence_prob = 0.25;
    let g = build_graph(&plain(&[&[2]]), &tm, &ctx, &opts);
    let eps_costs: Vec<f32> = g
        .states
        .iter()
        .flat_map(|s| s.arcs.iter())
        .filter(|a| a.tid == 0)
        .map(|a| a.cost)
        .collect();
    let want_sil = -0.25f32.ln();
    let want_non = -0.75f32.ln();
    assert!(eps_costs.iter().any(|c| (c - want_sil).abs() < 1e-5));
    assert!(eps_costs.iter().any(|c| (c - want_non).abs() < 1e-5));
}

#[test]
fn final_corrections_applied() {
    let (tm, ctx, mut opts) = mono_setup();
    opts.final_silence_correction = 0.5;
    opts.final_non_silence_correction = 2.0;
    let g = build_graph(&plain(&[&[2]]), &tm, &ctx, &opts);
    // AddSelfLoopsReorder also scales the final-prob of a state entered by a transition-state
    // by -self_loop_scale * GetNonSelfLoopLogProb(T) (hmm-utils.cc:533), so the correction is
    // the final cost minus that offset.
    let offsets: Vec<f32> = g
        .finals
        .iter()
        .map(|&(s, c)| {
            let ts = g
                .states
                .iter()
                .flat_map(|st| st.arcs.iter())
                .find(|a| a.next == s && a.tid != 0)
                .map(|a| tm.transition_id_to_transition_state(a.tid));
            let adj = ts.map_or(0.0, |t| -opts.self_loop_scale * tm.get_non_self_loop_log_prob(t));
            c - adj
        })
        .collect();
    assert!(offsets.iter().any(|c| (c + 0.5f32.ln()).abs() < 1e-5));
    assert!(offsets.iter().any(|c| (c + 2.0f32.ln()).abs() < 1e-5));
}

#[test]
fn empty_utterance_is_silence_only() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[]), &tm, &ctx, &opts);
    assert!(!g.finals.is_empty());
    assert_coaccessible(&g);
    for st in &g.states {
        for a in &st.arcs {
            if a.tid != 0 {
                assert_eq!(tm.transition_id_to_phone(a.tid), 1);
            }
        }
    }
}

#[test]
fn graph_pdfs_are_sorted_unique_and_in_range() {
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &opts);
    let pdfs = graph_pdfs(&g, &tm);
    assert!(!pdfs.is_empty());
    assert!(pdfs.windows(2).all(|w| w[0] < w[1]));
    assert!(pdfs.iter().all(|p| (*p as usize) < tm.num_pdfs()));
}

#[test]
fn forward_arc_cost_matches_add_transition_probs() {
    // For a state entered by transition-state T, the arc cost must be
    //   -transition_scale * logProbIgnoringSelfLoops(arc)
    //   - self_loop_scale * nonSelfLoopLogProb(T),
    // which is exactly GetScaledTransitionLogProb negated (hmm-utils.cc:1066).
    let (tm, ctx, opts) = mono_setup();
    let g = build_graph(&plain(&[&[2]]), &tm, &ctx, &opts);
    for (s, st) in g.states.iter().enumerate() {
        // Determine the transition-state entering this state, if unique.
        let mut entering: Option<u32> = None;
        let mut ok = true;
        for (t, tst) in g.states.iter().enumerate() {
            for a in &tst.arcs {
                if a.next as usize == s && a.tid != 0 && t != s {
                    let ts = tm.transition_id_to_transition_state(a.tid);
                    match entering {
                        None => entering = Some(ts),
                        Some(prev) if prev != ts => ok = false,
                        _ => {}
                    }
                }
            }
        }
        let Some(ts) = entering else { continue };
        if !ok {
            continue;
        }
        for a in &st.arcs {
            if a.tid == 0 || a.next as usize == s {
                continue;
            }
            let want = -opts.transition_scale
                * tm.get_transition_log_prob_ignoring_self_loops(a.tid)
                - opts.self_loop_scale * tm.get_non_self_loop_log_prob(ts);
            assert!(
                (a.cost - want).abs() < 1e-5,
                "arc cost {} != expected {want}",
                a.cost
            );
        }
    }
}


/// The silence branch after a word costs `-ln(silence_after_prob)` and the direct branch
/// `-ln(1 - silence_after_prob)`, taken from the pronunciation rather than the global default
/// (`lexicon.py:546-560`). The two epsilon arcs leaving the word's last-phone exits must differ
/// by exactly that log ratio.
#[test]
fn silence_after_prob_sets_the_branch_costs() {
    let (tm, ctx, mut opts) = mono_setup();
    // The self-loop rescaling adds a per-state offset to every outgoing arc; turning it off
    // leaves the lexicon costs on the epsilon arcs exactly as MFA writes them.
    opts.self_loop_scale = 0.0;
    let p = 0.9f32;
    let words = vec![vec![Pronunciation {
        phones: vec![2],
        prob: None,
        silence_after_prob: Some(p),
        silence_before_correction: None,
        non_silence_before_correction: None,
    }]];
    let g = build_graph(&words, &tm, &ctx, &opts);

    // The after-costs live on the epsilon arcs leaving the word's exit states. Collect every
    // epsilon arc cost in the graph that is not part of the initial-silence structure (state 0).
    let mut eps: Vec<f32> = Vec::new();
    for (s, st) in g.states.iter().enumerate() {
        if s == 0 {
            continue;
        }
        for a in &st.arcs {
            if a.tid == 0 && a.word == NO_WORD {
                eps.push(a.cost);
            }
        }
    }
    let want_sil = -p.ln();
    let want_nonsil = -(1.0 - p).ln();
    assert!(
        eps.iter().any(|c| (c - want_sil).abs() < 1e-4),
        "no arc carrying -ln(0.9) = {want_sil}; saw {eps:?}"
    );
    assert!(
        eps.iter().any(|c| (c - want_nonsil).abs() < 1e-4),
        "no arc carrying -ln(0.1) = {want_nonsil}; saw {eps:?}"
    );
    // The silence branch is the cheaper of the two, by exactly ln((1-p)/p) reversed.
    assert!(want_sil < want_nonsil);
    assert!(((want_nonsil - want_sil) - (p / (1.0 - p)).ln()).abs() < 1e-5);
    assert_coaccessible(&g);
}

/// Pronunciation probability becomes `|ln p|` on the arc entering the pronunciation, floored at
/// 0.01 (`lexicon.py:519-523`).
#[test]
fn pron_probability_floors_at_one_percent() {
    let (tm, ctx, mut opts) = mono_setup();
    opts.self_loop_scale = 0.0;
    for (prob, want) in [(0.5f32, 0.5f32.ln().abs()), (0.001, 0.01f32.ln().abs())] {
        let words = vec![vec![Pronunciation {
            phones: vec![2],
            prob: Some(prob),
            silence_after_prob: None,
            silence_before_correction: None,
            non_silence_before_correction: None,
        }]];
        let g = build_graph(&words, &tm, &ctx, &opts);
        let found = g.states.iter().any(|st| {
            st.arcs
                .iter()
                .any(|a| a.tid == 0 && (a.cost - want).abs() < 1e-4)
        });
        assert!(found, "no entry arc costing {want} for prob {prob}");
    }
}

/// Every pronunciation alternative is reachable and tags its own index.
#[test]
fn every_pronunciation_is_tagged_with_its_index() {
    let (tm, ctx, opts) = mono_setup();
    let words = vec![vec![
        Pronunciation::plain(vec![2, 3]),
        Pronunciation::plain(vec![4]),
    ]];
    let g = build_graph(&words, &tm, &ctx, &opts);
    let mut seen: Vec<(WordId, u32)> = g
        .states
        .iter()
        .flat_map(|st| st.arcs.iter())
        .filter(|a| a.word != NO_WORD)
        .map(|a| (a.word, a.pron))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen, vec![(0, 0), (0, 1)]);
    assert_coaccessible(&g);
}
