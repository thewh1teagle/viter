use ndarray::Array2;
use viter_kaldi::align::{AlignOptions, align};
use viter_kaldi::hmm::{
    ContextDependency, GraphOptions, HmmTopology, TransitionModel, build_graph, split_to_phones,
};
use viter_kaldi::types::{PhoneId, Pronunciation};

fn main() {
    let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
    let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3]];
    let t = topo.clone();
    let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
    let tm = TransitionModel::new(&ctx, &topo);

    for ts in 1..=tm.num_transition_states() as u32 {
        let tup = tm.tuples()[ts as usize - 1];
        let n = tm.num_transition_indices(ts);
        let ids: Vec<_> = (0..n)
            .map(|i| {
                let tid = tm.pair_to_transition_id(ts, i as u32);
                (
                    tid,
                    tm.is_self_loop(tid),
                    tm.is_final(tid),
                    tm.transition_id_to_pdf(tid),
                )
            })
            .collect();
        println!(
            "ts{ts} {tup:?} selfloop={:?} nonself_logp={} ids(tid,self,final,pdf)={ids:?}",
            tm.self_loop_of(ts),
            tm.get_non_self_loop_log_prob(ts)
        );
    }

    let opts = GraphOptions {
        silence_phone: 1,
        silence_prob: 0.0,
        initial_silence_prob: 0.0,
        ..Default::default()
    };
    let words = vec![
        vec![Pronunciation::plain(vec![2])],
        vec![Pronunciation::plain(vec![3])],
    ];
    let g = build_graph(&words, &tm, &ctx, &opts);
    println!("--- graph, {} states", g.num_states());
    for (i, st) in g.states.iter().enumerate() {
        for a in &st.arcs {
            println!("  {i} -> {} tid={} cost={:.4}", a.next, a.tid, a.cost);
        }
    }
    println!("finals {:?}", g.finals);

    // 6 frames: phone A (2) best for frame 0, phone B (3) best for frames 1..5.
    let pdfs: Vec<_> = viter_kaldi::hmm::graph_pdfs(&g, &tm);
    println!("pdfs {pdfs:?}");
    let ncol = tm.num_pdfs();
    let mut scores = Array2::<f32>::from_elem((6, ncol), -50.0);
    let a_pdfs: Vec<usize> = (0..3)
        .map(|c| ctx.compute(&[2], c).unwrap() as usize)
        .collect();
    let b_pdfs: Vec<usize> = (0..3)
        .map(|c| ctx.compute(&[3], c).unwrap() as usize)
        .collect();
    for &p in &a_pdfs {
        scores[[0, p]] = 0.0;
    }
    for f in 1..6 {
        for &p in &b_pdfs {
            scores[[f, p]] = 0.0;
        }
    }

    let aopts = AlignOptions {
        acoustic_scale: 1.0,
        ..Default::default()
    };
    let ali = align(&g, &tm, &scores, &|p| p as usize, &aopts).expect("align failed");
    let phones: Vec<PhoneId> = ali
        .tids
        .iter()
        .map(|&t| tm.transition_id_to_phone(t))
        .collect();
    let hmm_states: Vec<usize> = ali
        .tids
        .iter()
        .map(|&t| tm.transition_id_to_hmm_state(t))
        .collect();
    println!(
        "tids {:?}\nphones {phones:?}\nstates {hmm_states:?}",
        ali.tids
    );
    let runs = split_to_phones(&tm, &ali.tids);
    println!(
        "split: {:?}",
        runs.iter()
            .map(|r| (tm.transition_id_to_phone(r[0]), r.len()))
            .collect::<Vec<_>>()
    );
    assert_eq!(phones, vec![2, 3, 3, 3, 3, 3], "expected [A,B,B,B,B,B]");
    println!("OK");
}
