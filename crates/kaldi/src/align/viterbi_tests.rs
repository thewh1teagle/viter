//! Tests for [`super::viterbi`]: beam Viterbi alignment.

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

fn setup() -> (TransitionModel, ContextDependency, GraphOptions) {
    let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
    let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3]];
    let t = topo.clone();
    let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
    let tm = TransitionModel::new(&ctx, &topo);
    let opts = GraphOptions {
        silence_phone: 1,
        // Turn off optional silence so the test graph is a clean linear chain.
        silence_prob: 0.0,
        initial_silence_prob: 0.0,
        ..Default::default()
    };
    (tm, ctx, opts)
}

#[test]
fn aligns_a_single_phone() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let frames = 6;
    // Flat scores: any path of the right length works.
    let scores = Array2::zeros((frames, pdfs.len()));
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let opts = AlignOptions::default();
    let ali = align(&g, &tm, &scores, &col, &opts).expect("alignment should succeed");
    assert_eq!(ali.tids.len(), frames);
    // Every frame belongs to phone 2.
    assert!(ali.tids.iter().all(|&t| tm.transition_id_to_phone(t) == 2));
    assert_eq!(ali.words, vec![0]);
    assert!(ali.loglike.is_finite());
}

#[test]
fn alignment_splits_into_the_expected_phones() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let frames = 12;
    let scores = Array2::zeros((frames, pdfs.len()));
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let ali = align(&g, &tm, &scores, &col, &AlignOptions::default()).unwrap();
    let runs = crate::hmm::split_to_phones(&tm, &ali.tids);
    assert_eq!(runs.len(), 2);
    assert_eq!(tm.transition_id_to_phone(runs[0][0]), 2);
    assert_eq!(tm.transition_id_to_phone(runs[1][0]), 3);
    assert_eq!(runs.iter().map(|r| r.len()).sum::<usize>(), frames);
}

#[test]
fn acoustic_scores_steer_the_boundary() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let frames = 12;
    // Favour phone 2's pdfs for the first 9 frames, phone 3's afterwards.
    let p2: Vec<PdfId> = (0..3).map(|c| ctx.compute(&[2], c).unwrap()).collect();
    let p3: Vec<PdfId> = (0..3).map(|c| ctx.compute(&[3], c).unwrap()).collect();
    let mut scores = Array2::from_elem((frames, pdfs.len()), -30.0f32);
    for f in 0..frames {
        let set = if f < 9 { &p2 } else { &p3 };
        for p in set {
            scores[[f, col(*p)]] = 0.0;
        }
    }
    let ali = align(&g, &tm, &scores, &col, &AlignOptions::default()).unwrap();
    let runs = crate::hmm::split_to_phones(&tm, &ali.tids);
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].len(), 9, "boundary should follow the acoustics");
    assert_eq!(runs[1].len(), 3);
}

#[test]
fn too_few_frames_fails() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    // Each Bakis phone can be crossed in one frame via the state-0 skip arc, so two phones need
    // two frames; a single frame cannot be aligned.
    let scores: Array2<f32> = Array2::zeros((1, pdfs.len()));
    assert!(align(&g, &tm, &scores, &col, &AlignOptions::default()).is_none());
}

#[test]
fn narrow_beam_recovered_by_retry() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let frames = 8;
    let scores = Array2::zeros((frames, pdfs.len()));
    let opts = AlignOptions {
        beam: 0.001,
        retry_beam: 100.0,
        ..Default::default()
    };
    // With a tiny beam the path may still be found (the graph is small), but the retry path
    // must at minimum produce a valid alignment.
    let ali = align(&g, &tm, &scores, &col, &opts).expect("retry beam should recover");
    assert_eq!(ali.tids.len(), frames);
}

#[test]
fn max_active_limits_the_search_without_breaking_it() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2, 3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let scores: Array2<f32> = Array2::zeros((10, pdfs.len()));
    let opts = AlignOptions {
        // Aggressive, but not below the number of states a single frame of the reordered
        // graph legitimately needs live at once (state splitting for the reorder
        // convention widens the frontier).
        max_active: 8,
        min_active: 1,
        ..Default::default()
    };
    let ali = align(&g, &tm, &scores, &col, &opts).expect("should still align");
    assert_eq!(ali.tids.len(), 10);
}

/// Full round trip on a two-word graph with optional silence live: build with a monophone
/// ContextDependency, run both `equal_align` and `align` on synthetic scores, and check that
/// `split_to_phones` recovers the expected phone sequence from each.
#[test]
fn two_word_round_trip_through_both_aligners() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
    let sets: Vec<PhoneId> = vec![1, 2, 3];
    let sets: Vec<Vec<PhoneId>> = sets.into_iter().map(|p| vec![p]).collect();
    let t = topo.clone();
    let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
    let tm = TransitionModel::new(&ctx, &topo);
    // Optional silence on, but zero-probability at the ends so the maximising path is the
    // plain two-phone backbone and the expected phone sequence is unambiguous.
    let gopts = GraphOptions {
        silence_phone: 1,
        silence_prob: 0.0,
        initial_silence_prob: 0.0,
        ..Default::default()
    };
    // Two words of one phone each: "2" then "3".
    let g = build_graph(&plain(&[&[2], &[3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let frames = 18;

    // Synthetic scores favouring phone 2 in the first half and phone 3 in the second.
    let p2: Vec<PdfId> = (0..3).map(|c| ctx.compute(&[2], c).unwrap()).collect();
    let p3: Vec<PdfId> = (0..3).map(|c| ctx.compute(&[3], c).unwrap()).collect();
    let mut scores = Array2::from_elem((frames, pdfs.len()), -20.0f32);
    for f in 0..frames {
        for p in if f < frames / 2 { &p2 } else { &p3 } {
            scores[[f, col(*p)]] = 0.0;
        }
    }

    let viterbi = align(&g, &tm, &scores, &col, &AlignOptions::default())
        .expect("viterbi alignment should succeed");
    assert_eq!(viterbi.tids.len(), frames);
    assert_eq!(viterbi.words, vec![0, 1]);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2024);
    let equal =
        crate::align::equal_align(&g, &tm, frames, &mut rng).expect("equal_align should succeed");
    assert_eq!(equal.tids.len(), frames);

    for (name, ali) in [("viterbi", &viterbi), ("equal", &equal)] {
        let (runs, was_ok) = crate::hmm::split_to_phones_checked(&tm, &ali.tids);
        assert!(was_ok, "{name}: split_to_phones reported an inconsistency");
        let phones: Vec<PhoneId> = runs
            .iter()
            .map(|r| tm.transition_id_to_phone(r[0]))
            .collect();
        assert_eq!(phones, vec![2, 3], "{name}: unexpected phone sequence");
        assert_eq!(
            runs.iter().map(|r| r.len()).sum::<usize>(),
            frames,
            "{name}: phone runs must cover every frame"
        );
    }

    // The Viterbi boundary should follow the synthetic scores.
    let runs = crate::hmm::split_to_phones(&tm, &viterbi.tids);
    assert_eq!(runs[0].len(), frames / 2);
}

#[test]
fn word_labels_recovered_in_order() {
    let (tm, ctx, gopts) = setup();
    let g = build_graph(&plain(&[&[2], &[3]]), &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();
    let scores: Array2<f32> = Array2::zeros((10, pdfs.len()));
    let ali = align(&g, &tm, &scores, &col, &AlignOptions::default()).unwrap();
    assert_eq!(ali.words, vec![0, 1]);
}

/// Pronunciation fan-out: one word with two alternatives, synthetic scores that only fit the
/// second one. The backtrace must report pronunciation index 1.
#[test]
fn pronunciation_alternative_is_recovered() {
    let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
    let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3]];
    let t = topo.clone();
    let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
    let tm = TransitionModel::new(&ctx, &topo);
    let gopts = GraphOptions {
        silence_phone: 1,
        silence_prob: 0.0,
        initial_silence_prob: 0.0,
        ..Default::default()
    };
    // One word, pronounced either /2/ or /3/.
    let words = vec![vec![
        Pronunciation::plain(vec![2]),
        Pronunciation::plain(vec![3]),
    ]];
    let g = build_graph(&words, &tm, &ctx, &gopts);
    let pdfs = graph_pdfs(&g, &tm);
    let col = |p: PdfId| pdfs.iter().position(|x| *x == p).unwrap();

    // Score only phone 3's pdfs: the second pronunciation is the only affordable path.
    let p3: Vec<PdfId> = (0..3).map(|c| ctx.compute(&[3], c).unwrap()).collect();
    let frames = 12;
    let mut scores = Array2::from_elem((frames, pdfs.len()), -50.0f32);
    for f in 0..frames {
        for p in &p3 {
            scores[[f, col(*p)]] = 0.0;
        }
    }
    let ali = align(&g, &tm, &scores, &col, &AlignOptions::default()).unwrap();
    assert_eq!(ali.words, vec![0]);
    assert_eq!(ali.prons, vec![1]);
    let runs = crate::hmm::split_to_phones(&tm, &ali.tids);
    let phones: Vec<PhoneId> = runs
        .iter()
        .map(|r| tm.transition_id_to_phone(r[0]))
        .collect();
    assert_eq!(phones, vec![3]);
}

/// Equal-cost parallel arcs must retain the first pronunciation, and epsilon
/// recombination must retain the first path that reached a state (LIFO closure).
#[test]
fn recombination_preserves_ties_and_epsilon_word_labels() {
    use crate::hmm::{Arc, GraphState, NO_PRON};
    let (tm, _, _) = setup();
    let arc = |tid, next, word, pron| Arc {
        tid,
        next,
        word,
        pron,
        cost: 0.0,
        lm: 0.0,
    };
    let mut graph = Graph::default();
    graph.states = vec![
        GraphState {
            arcs: vec![arc(1, 1, 7, 0), arc(1, 1, 7, 1)],
        },
        GraphState {
            arcs: vec![arc(0, 2, NO_WORD, NO_PRON), arc(0, 3, NO_WORD, NO_PRON)],
        },
        GraphState {
            arcs: vec![arc(0, 4, 8, 0)],
        },
        GraphState {
            arcs: vec![arc(0, 4, 8, 1)],
        },
        GraphState::default(),
    ];
    graph.finals.push((4, 0.0));
    let result = align(
        &graph,
        &tm,
        &Array2::zeros((1, 1)),
        &|_| 0,
        &AlignOptions::default(),
    )
    .unwrap();
    assert_eq!(result.tids, vec![1]);
    assert_eq!(result.words, vec![7, 8]);
    assert_eq!(result.prons, vec![0, 1]);
    assert_eq!(result.loglike.to_bits(), (-0.0f32).to_bits());
}

/// A better epsilon path can replace a token after descendants already refer
/// to it. Backpointers must remain immutable, even while state costs change.
#[test]
fn epsilon_replacement_does_not_rewrite_existing_descendants() {
    use crate::hmm::{Arc, GraphState};
    let (tm, _, _) = setup();
    let arc = |next, cost, word| Arc {
        tid: 0,
        next,
        word,
        pron: 0,
        cost,
        lm: cost,
    };
    let mut graph = Graph::default();
    graph.states = vec![
        GraphState {
            arcs: vec![arc(1, 0.0, 1), arc(2, 5.0, 2)],
        },
        GraphState {
            arcs: vec![arc(2, 0.0, 3)],
        },
        GraphState {
            arcs: vec![arc(3, 0.0, 4)],
        },
        GraphState::default(),
    ];
    graph.finals.push((3, 0.0));
    let opts = AlignOptions::default();
    let prepared = PreparedGraph::new(&graph, &tm, &|_| 0);
    let mut decoder = Decoder::new(&prepared, &opts, opts.beam);
    let start = decoder.arena.push(Token {
        prev: usize::MAX,
        arc: usize::MAX,
        ac_cost: 0.0,
    });
    decoder.cur.slot[0] = start;
    decoder.cur.cost[0] = 0.0;
    decoder.cur.states.push(0);
    decoder.process_nonemitting(f64::INFINITY);
    let old_descendant = decoder
        .arena
        .toks
        .iter()
        .find(|token| token.arc != usize::MAX && prepared.arcs[token.arc].0.next == 3)
        .unwrap();
    let old_parent = decoder.arena.toks[old_descendant.prev];
    assert_eq!(prepared.arcs[old_parent.arc].0.word, 2);
    let current = decoder.arena.toks[decoder.cur.slot[3]];
    let current_parent = decoder.arena.toks[current.prev];
    assert_eq!(prepared.arcs[current_parent.arc].0.word, 3);
    assert_eq!(decoder.cur.cost[3], 0.0);
}

/// Compare the adaptive cutoff to a full-sort oracle over many tied frontiers.
/// This covers both min-active widening and max-active narrowing independently
/// of any graph/score fixture, including impossible min-active requirements.
#[test]
fn adaptive_cutoff_matches_sorted_frontiers() {
    let (tm, _, _) = setup();
    let mut graph = Graph::default();
    graph.states.resize(100, Default::default());
    let prepared = PreparedGraph::new(&graph, &tm, &|_| 0);
    for n in [0, 1, 8, 20, 21, 40, 100] {
        for min in [0, 1, 20, 99, 100] {
            for max in [0, 1, 20, 100, usize::MAX] {
                for beam in [0.0, 0.5, 10.0, 40.0] {
                    let opts = AlignOptions {
                        min_active: min,
                        max_active: max,
                        beam,
                        ..Default::default()
                    };
                    let mut d = Decoder::new(&prepared, &opts, beam);
                    for s in 0..n {
                        d.cur.states.push(s as u32);
                        d.cur.cost[s] = ((s * 17 + n) % 31) as f64;
                    }
                    let mut sorted = d.cur.cost[..n].to_vec();
                    sorted.sort_by(f64::total_cmp);
                    let best = sorted.first().copied().unwrap_or(f64::INFINITY);
                    let ordinary = best + beam as f64;
                    let max_cutoff = sorted.get(max).copied().unwrap_or(f64::INFINITY);
                    let min_cutoff = if n > min {
                        if min == 0 {
                            best
                        } else if min < n.min(max) {
                            sorted[min]
                        } else {
                            f64::INFINITY
                        }
                    } else {
                        f64::INFINITY
                    };
                    let expected = if max == usize::MAX && min == 0 {
                        (ordinary, beam)
                    } else if max_cutoff < ordinary {
                        (max_cutoff, (max_cutoff - best) as f32 + opts.beam_delta)
                    } else if min_cutoff > ordinary {
                        (min_cutoff, (min_cutoff - best) as f32 + opts.beam_delta)
                    } else {
                        (ordinary, beam)
                    };
                    let got = d.get_cutoff();
                    assert_eq!(
                        got.0.to_bits(),
                        expected.0.to_bits(),
                        "n={n} min={min} max={max} beam={beam}"
                    );
                    assert_eq!(
                        got.1.to_bits(),
                        expected.1.to_bits(),
                        "n={n} min={min} max={max} beam={beam}"
                    );
                }
            }
        }
    }
}
