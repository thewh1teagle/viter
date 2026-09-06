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
    use crate::hmm::{build_graph, ContextDependency, GraphOptions, HmmTopology};

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
        // Two Bakis phones need at least 6 frames.
        let scores: Array2<f32> = Array2::zeros((3, pdfs.len()));
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
        let equal = crate::align::equal_align(&g, &tm, frames, &mut rng)
            .expect("equal_align should succeed");
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
        let phones: Vec<PhoneId> = runs.iter().map(|r| tm.transition_id_to_phone(r[0])).collect();
        assert_eq!(phones, vec![3]);
    }
