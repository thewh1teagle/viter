//! Phonetic decision-tree building: Kaldi's `tree/` directory in Rust.
//!
//! Modules map onto Kaldi sources one-for-one:
//!
//! | module | Kaldi source |
//! |---|---|
//! | [`event_map`] | `tree/event-map.{h,cc}` |
//! | [`clusterable`] | `tree/clusterable-classes.{h,cc}` + the `Clusterable` interface |
//! | [`heap`] | libstdc++'s `std::priority_queue` ordering (parity helper) |
//! | [`cluster`] | `tree/cluster-utils.{h,cc}` (bottom-up, refine, k-means) |
//! | [`tree_cluster`] | `tree/cluster-utils.cc:1032` (`TreeCluster`) |
//! | [`stats`] | `tree/build-tree-utils.{h,cc}` statistics helpers |
//! | [`cluster_map`] | `tree/build-tree-utils.{h,cc}` leaf clustering + `GetStubMap` |
//! | [`build`] | `tree/build-tree-utils.cc` splitting, `tree/build-tree.cc` (`BuildTree`) |
//! | [`questions`] | `tree/build-tree-questions.h`, `AutomaticallyObtainQuestions`, roots |
//! | [`accu`] | `hmm/tree-accu.{h,cc}` (`AccumulateTreeStats`) |
//!
//! The MFA/kalpy pipeline is:
//!
//! ```text
//! accumulate_tree_stats  (per utterance, merged with merge_tree_stats)
//!   -> stats_map_to_vec
//!   -> automatically_obtain_questions(stats, phone_sets, &[1], 1)
//!   -> make_questions(questions, 3)
//!   -> build_tree(qopts, roots..., stats, 300.0, num_leaves, -1.0, 1, true)
//! ```

pub mod accu;
pub mod build;
pub mod cluster;
pub mod cluster_map;
pub mod clusterable;
pub mod event_map;
pub mod heap;
pub mod questions;
pub mod stats;
pub mod tree_cluster;

pub use accu::{
    AccumulateTreeStatsOptions, accumulate_tree_stats, merge_tree_stats, stats_map_to_vec,
};
pub use build::{
    SplitResult, build_tree, find_best_split_for_key, per_leaf_counts, split_decision_tree,
    summed_normalizer,
};
pub use cluster::{
    ClusterKMeansOptions, RefineClustersOptions, cluster_bottom_up,
    cluster_bottom_up_compartmentalized, cluster_kmeans, refine_clusters,
};
pub use cluster_map::{
    cluster_event_map, cluster_event_map_get_mapping, cluster_event_map_restricted_by_keys,
    cluster_event_map_restricted_by_map, cluster_event_map_to_n_clusters_restricted_by_map,
    get_stub_map, map_event_map_leaves, renumber_event_map, share_event_map_leaves,
};
pub use stats::{
    AllKeysType, BuildTreeStats, convert_stats, filter_stats_by_key, find_all_keys,
    objf_given_map, possible_values, split_stats_by_key, split_stats_by_map, sum_normalizer,
    sum_objf, sum_stats, sum_stats_vec,
};
pub use tree_cluster::{TreeClusterOptions, tree_cluster};
pub use clusterable::{
    GaussClusterable, add_to_clusters, add_to_clusters_optimized, ensure_not_null,
    sum_clusterable, sum_clusterable_normalizer, sum_clusterable_objf,
};
pub use event_map::{
    EventAnswer, EventKey, EventMap, EventType, EventValue, K_PDF_CLASS, check_event,
    event_to_string, get_tree_structure, lookup,
};
pub use questions::{
    Questions, QuestionsForKey, automatically_obtain_questions, kmeans_cluster_phones,
    make_questions, make_questions_with, mfa_roots, mfa_roots_with, read_roots, write_roots,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PhoneId;

    /// End-to-end: stats for two phones in two left contexts should produce a
    /// tree that splits on left context and maps every stat to a leaf.
    #[test]
    fn end_to_end_questions_then_tree() {
        // Phones: 1 = silence (ci), 10, 11 = real phones.
        let mut stats: BuildTreeStats = Vec::new();
        let mut push = |left: i32, centre: i32, right: i32, pc: i32, x: f32| {
            let mut e: EventType = vec![
                (K_PDF_CLASS, pc),
                (0, left),
                (1, centre),
                (2, right),
            ];
            e.sort_by_key(|p| p.0);
            let mut c = GaussClusterable::new(2, 0.01);
            c.add_stats(&[x, x], 60.0);
            c.add_stats(&[x + 1.0, x - 1.0], 60.0);
            stats.push((e, c));
        };
        for pc in 0..3 {
            push(10, 11, 10, pc, pc as f32);
            push(11, 11, 10, pc, 10.0 + pc as f32);
            push(10, 10, 11, pc, 20.0 + pc as f32);
            push(11, 10, 11, pc, 30.0 + pc as f32);
        }

        let phone_sets: Vec<Vec<PhoneId>> = vec![vec![10], vec![11]];
        let auto_q = automatically_obtain_questions(&stats, &phone_sets, &[1], 1);
        assert!(!auto_q.is_empty());
        let qopts = make_questions(&auto_q, 3);

        let (phone_sets, share_roots, do_split) = mfa_roots(&[vec![10], vec![11]], &[]);
        // phone2num_pdf_classes indexed by phone id.
        let mut phone2num = vec![0usize; 12];
        phone2num[10] = 3;
        phone2num[11] = 3;

        let (tree, num_leaves) = build_tree(
            &qopts,
            &phone_sets,
            &phone2num,
            &share_roots,
            &do_split,
            &stats,
            1.0,
            0,
            -1.0,
            1,
            false,
        );
        assert!(num_leaves >= 2);
        let mut seen = std::collections::HashSet::new();
        for (e, _) in &stats {
            let leaf = tree.map(e).expect("every stat maps to a leaf");
            assert!((leaf as usize) < num_leaves);
            seen.insert(leaf);
        }
        assert_eq!(seen.len(), num_leaves, "every leaf should be reached");
    }

    #[test]
    fn round_num_leaves_gives_multiple_of_eight() {
        // Many distinct contexts so the tree can grow past 8 leaves.
        let mut stats: BuildTreeStats = Vec::new();
        for left in 10..20i32 {
            for pc in 0..3 {
                let mut e: EventType =
                    vec![(K_PDF_CLASS, pc), (0, left), (1, 10), (2, 10)];
                e.sort_by_key(|p| p.0);
                let mut c = GaussClusterable::new(1, 0.01);
                let x = (left as f32) * 5.0 + pc as f32;
                c.add_stats(&[x], 200.0);
                c.add_stats(&[x + 2.0], 200.0);
                stats.push((e, c));
            }
        }
        let questions: Vec<Vec<PhoneId>> =
            (10..20u32).map(|p| vec![p]).chain([(10..20u32).collect()]).collect();
        let qopts = make_questions(&questions, 3);
        let mut phone2num = vec![0usize; 21];
        for p in 10..20 {
            phone2num[p] = 3;
        }
        let (tree, num_leaves) = build_tree(
            &qopts,
            &[vec![10]],
            &phone2num,
            &[true],
            &[true],
            &stats,
            1.0,
            0,
            0.0,
            1,
            true,
        );
        assert_eq!(num_leaves % 8, 0, "round_num_leaves should give a multiple of 8");
        assert!(num_leaves > 0);
        for (e, _) in &stats {
            assert!(tree.map(e).is_some());
        }
    }

    /// End-to-end: synthetic stats for 4 phones x 3 pdf-classes forming two
    /// clearly separable phone groups ({10,11} vs {20,21}), driven through
    /// `automatically_obtain_questions` -> `make_questions` -> `build_tree`
    /// with `max_leaves = 8`, the way MFA's `_setup_tree` drives kalpy.
    #[test]
    fn end_to_end_two_separable_phone_groups() {
        const PHONES: [i32; 4] = [10, 11, 20, 21];
        const MAX_LEAVES: usize = 8;
        const NUM_PDF_CLASSES: usize = 3;

        // Group A (10, 11) sits near 0; group B (20, 21) near 100. Within a
        // group the two phones differ only slightly, and each pdf-class is
        // offset so the classes are separable too.
        let mut stats: BuildTreeStats = Vec::new();
        for &centre in &PHONES {
            let group_base = if centre < 20 { 0.0f32 } else { 100.0 };
            let phone_off = if centre % 10 == 0 { 0.0f32 } else { 1.0 };
            for pc in 0..NUM_PDF_CLASSES as i32 {
                for &left in &PHONES {
                    let mut e: EventType = vec![
                        (K_PDF_CLASS, pc),
                        (0, left),
                        (1, centre),
                        (2, 10),
                    ];
                    e.sort_by_key(|p| p.0);
                    let x = group_base + phone_off + 10.0 * pc as f32;
                    let mut c = GaussClusterable::new(2, 0.01);
                    c.add_stats(&[x, -x], 100.0);
                    c.add_stats(&[x + 1.0, -x - 1.0], 100.0);
                    stats.push((e, c));
                }
            }
        }

        // Each phone is its own root/phone-set, as MFA writes them.
        let phone_sets: Vec<Vec<PhoneId>> =
            PHONES.iter().map(|&p| vec![p as PhoneId]).collect();

        // Automatic questions must recover the two groups.
        let auto_q = automatically_obtain_questions(&stats, &phone_sets, &[1], 1);
        assert!(auto_q.contains(&vec![10, 11]), "group A missing: {auto_q:?}");
        assert!(auto_q.contains(&vec![20, 21]), "group B missing: {auto_q:?}");

        let qopts = make_questions(&auto_q, 3);
        let (roots, share_roots, do_split) =
            mfa_roots(&[vec![10, 11], vec![20, 21]], &[]);

        let mut phone2num = vec![0usize; 22];
        for &p in &PHONES {
            phone2num[p as usize] = NUM_PDF_CLASSES;
        }

        let (tree, num_leaves) = build_tree(
            &qopts,
            &roots,
            &phone2num,
            &share_roots,
            &do_split,
            &stats,
            1.0,
            MAX_LEAVES,
            -1.0,
            1,
            false,
        );

        assert!(num_leaves <= MAX_LEAVES, "num_leaves {num_leaves} > {MAX_LEAVES}");
        assert!(
            num_leaves >= NUM_PDF_CLASSES,
            "num_leaves {num_leaves} < {NUM_PDF_CLASSES} pdf-classes"
        );

        // Every (phone, pdf_class) event maps to some pdf, for every context.
        for &centre in &PHONES {
            for pc in 0..NUM_PDF_CLASSES as i32 {
                for &left in &PHONES {
                    let mut e: EventType = vec![
                        (K_PDF_CLASS, pc),
                        (0, left),
                        (1, centre),
                        (2, 10),
                    ];
                    e.sort_by_key(|p| p.0);
                    let pdf = tree
                        .map(&e)
                        .unwrap_or_else(|| panic!("no pdf for {e:?}"));
                    assert!((pdf as usize) < num_leaves);
                }
            }
        }
    }
}
