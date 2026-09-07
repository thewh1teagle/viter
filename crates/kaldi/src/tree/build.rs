//! Port of the tree-growing half of `plans/kaldi/src/tree/build-tree-utils.{h,cc}`
//! and of `plans/kaldi/src/tree/build-tree.cc` (`BuildTree`). The statistics
//! helpers it builds on live in [`super::stats`].
//!
//! Split order and tie-breaking follow Kaldi exactly: `SplitDecisionTree` keeps
//! a max-heap keyed on the best available improvement of each initial leaf's
//! sub-tree, and `DecisionTreeSplitter::DoSplit` descends into whichever child
//! has the larger `BestSplit()` (yes-child on ties).

use super::cluster::refine_clusters;
use super::cluster_map::{
    cluster_event_map_restricted_by_map, cluster_event_map_to_n_clusters_restricted_by_map,
    get_stub_map, renumber_event_map,
};
use super::clusterable::{
    GaussClusterable, add_to_clusters_optimized, ensure_not_null, sum_clusterable,
    sum_clusterable_normalizer, sum_clusterable_objf,
};
use super::event_map::{EventKey, EventMap, EventValue, lookup};
use super::heap::{Heap, OrdF64};
use super::questions::Questions;
use super::stats::{
    BuildTreeStats, filter_stats_by_key, possible_values, split_stats_by_key, split_stats_by_map,
    sum_normalizer, sum_stats_vec,
};

// ---------------------------------------------------------------------------
// Splitting (build-tree-utils.cc:325 onwards)
// ---------------------------------------------------------------------------

/// Kaldi `ComputeInitialSplit`: try each of the key's questions on the
/// per-value summed stats, returning the best objf change (>= 0) and its
/// yes-set.
fn compute_initial_split(
    summed_stats: &[Option<GaussClusterable>],
    q_opts: &Questions,
    key: EventKey,
) -> (f64, Vec<EventValue>) {
    let key_opts = q_opts.get(key);
    let total = match sum_clusterable(summed_stats) {
        None => return (0.0, Vec::new()),
        Some(t) => t,
    };
    let unsplit_objf = total.objf();

    let mut best_idx: i32 = -1;
    let mut best_objf_change = 0.0f64;

    for (i, yes_set) in key_opts.questions.iter().enumerate() {
        let mut assignments = vec![0usize; summed_stats.len()];
        for &v in yes_set {
            assert!(v >= 0);
            if (v as usize) < assignments.len() {
                assignments[v as usize] = 1;
            }
        }
        let mut clusters: Vec<Option<GaussClusterable>> = vec![None, None];
        add_to_clusters_optimized(summed_stats, &assignments, &total, &mut clusters);
        let this_objf = sum_clusterable_objf(&clusters);
        let this_objf_change = this_objf - unsplit_objf;
        if this_objf_change > best_objf_change {
            best_objf_change = this_objf_change;
            best_idx = i as i32;
        }
    }
    if best_idx != -1 {
        (
            best_objf_change,
            key_opts.questions[best_idx as usize].clone(),
        )
    } else {
        (best_objf_change, Vec::new())
    }
}

/// Kaldi `FindBestSplitForKey`. Returns `(improvement, yes_set)`.
pub fn find_best_split_for_key(
    stats: &BuildTreeStats,
    q_opts: &Questions,
    key: EventKey,
) -> (f64, Vec<EventValue>) {
    if stats.len() <= 1 {
        return (0.0, Vec::new());
    }
    if !possible_values(key, stats).0 {
        return (0.0, Vec::new());
    }
    let split_stats = split_stats_by_key(stats, key);
    let mut summed_stats = sum_stats_vec(&split_stats);

    let (mut improvement, mut yes_set) = compute_initial_split(&summed_stats, q_opts, key);

    let mut assignments = vec![0usize; summed_stats.len()];
    for &v in &yes_set {
        assert!(v >= 0);
        if (v as usize) < assignments.len() {
            assignments[v as usize] = 1;
        }
    }
    let mut clusters: Vec<Option<GaussClusterable>> = vec![None, None];
    super::clusterable::add_to_clusters(&summed_stats, &assignments, &mut clusters);

    ensure_not_null(&mut summed_stats);
    ensure_not_null(&mut clusters);

    let refine_opts = q_opts.get(key).refine_opts;
    if refine_opts.num_iters > 0 {
        let points: Vec<GaussClusterable> = summed_stats
            .into_iter()
            .map(|c| c.expect("ensure_not_null"))
            .collect();
        let mut clusters: Vec<GaussClusterable> = clusters
            .into_iter()
            .map(|c| c.expect("ensure_not_null"))
            .collect();
        let refine_impr = refine_clusters(&points, &mut clusters, &mut assignments, refine_opts);
        improvement += refine_impr;
        yes_set = assignments
            .iter()
            .enumerate()
            .filter(|(_, a)| **a == 1)
            .map(|(i, _)| i as EventValue)
            .collect();
    }
    (improvement, yes_set)
}

/// Kaldi `DecisionTreeSplitter`: a node of the tree being grown, holding its
/// stats and the best question it could ask.
struct DecisionTreeSplitter {
    yes: Option<Box<DecisionTreeSplitter>>,
    no: Option<Box<DecisionTreeSplitter>>,
    leaf: i32,
    stats: BuildTreeStats,
    best_split_impr: f64,
    key: EventKey,
    yes_set: Vec<EventValue>,
}

impl DecisionTreeSplitter {
    fn new(leaf: i32, stats: BuildTreeStats, q_opts: &Questions) -> Self {
        let mut s = Self {
            yes: None,
            no: None,
            leaf,
            stats,
            best_split_impr: 0.0,
            key: 0,
            yes_set: Vec::new(),
        };
        s.find_best_split(q_opts);
        s
    }

    fn find_best_split(&mut self, q_opts: &Questions) {
        self.best_split_impr = 0.0;
        for key in q_opts.keys() {
            let (impr, temp_yes_set) = find_best_split_for_key(&self.stats, q_opts, key);
            if impr > self.best_split_impr {
                self.best_split_impr = impr;
                self.yes_set = temp_yes_set;
                self.key = key;
            }
        }
    }

    fn best_split(&self) -> f64 {
        self.best_split_impr
    }

    fn get_map(&self) -> EventMap {
        match (&self.yes, &self.no) {
            (Some(y), Some(n)) => EventMap::Split {
                key: self.key,
                yes_set: self.yes_set.clone(),
                yes: Box::new(y.get_map()),
                no: Box::new(n.get_map()),
            },
            _ => EventMap::Constant(self.leaf as crate::types::PdfId),
        }
    }

    fn do_split(&mut self, next_leaf: &mut i32, q_opts: &Questions) {
        if self.yes.is_none() {
            self.do_split_internal(next_leaf, q_opts);
        } else {
            {
                let yes_best = self.yes.as_ref().unwrap().best_split();
                let no_best = self.no.as_ref().unwrap().best_split();
                if yes_best >= no_best {
                    self.yes.as_mut().unwrap().do_split(next_leaf, q_opts);
                } else {
                    self.no.as_mut().unwrap().do_split(next_leaf, q_opts);
                }
            }
            self.best_split_impr = self
                .yes
                .as_ref()
                .unwrap()
                .best_split()
                .max(self.no.as_ref().unwrap().best_split());
        }
    }

    fn do_split_internal(&mut self, next_leaf: &mut i32, q_opts: &Questions) {
        assert!(self.best_split_impr > 0.0);
        let yes_leaf = self.leaf;
        let no_leaf = *next_leaf;
        *next_leaf += 1;
        self.leaf = -1;
        let mut yes_stats: BuildTreeStats = Vec::new();
        let mut no_stats: BuildTreeStats = Vec::new();
        for item in std::mem::take(&mut self.stats) {
            let val = lookup(&item.0, self.key).expect("DoSplitInternal: key has no value");
            if self.yes_set.binary_search(&val).is_ok() {
                yes_stats.push(item);
            } else {
                no_stats.push(item);
            }
        }
        self.yes = Some(Box::new(DecisionTreeSplitter::new(
            yes_leaf, yes_stats, q_opts,
        )));
        self.no = Some(Box::new(DecisionTreeSplitter::new(
            no_leaf, no_stats, q_opts,
        )));
        self.best_split_impr = self
            .yes
            .as_ref()
            .unwrap()
            .best_split()
            .max(self.no.as_ref().unwrap().best_split());
    }
}

/// Result of `split_decision_tree`.
pub struct SplitResult {
    pub map: EventMap,
    pub num_leaves: i32,
    /// Total likelihood improvement from all splits.
    pub obj_impr: f64,
    /// The smallest improvement of any split actually taken.
    pub smallest_split_change: f64,
}

/// Kaldi `SplitDecisionTree`. `max_leaves <= 0` means no maximum.
pub fn split_decision_tree(
    input_map: &EventMap,
    stats: &BuildTreeStats,
    q_opts: &Questions,
    thresh: f64,
    max_leaves: i32,
    num_leaves: &mut i32,
) -> SplitResult {
    assert!(*num_leaves > 0);
    let mut like_impr = 0.0f64;
    let mut smallest_split_change = 1.0e+20f64;

    let split_stats = split_stats_by_map(stats, input_map);
    assert!(!split_stats.is_empty());
    let mut builders: Vec<DecisionTreeSplitter> = split_stats
        .into_iter()
        .enumerate()
        .map(|(i, s)| DecisionTreeSplitter::new(i as i32, s, q_opts))
        .collect();

    // Max-heap on (improvement, index), exactly Kaldi's
    // priority_queue<pair<BaseFloat, size_t>>.
    let mut queue: Heap<(OrdF64, usize)> = Heap::new();
    for (i, b) in builders.iter().enumerate() {
        queue.push((OrdF64(b.best_split()), i));
    }
    while let Some(&(OrdF64(top), _)) = queue.peek() {
        if !(top > thresh && (max_leaves <= 0 || *num_leaves < max_leaves)) {
            break;
        }
        let (OrdF64(impr), i) = queue.pop().unwrap();
        smallest_split_change = smallest_split_change.min(impr);
        like_impr += impr;
        builders[i].do_split(num_leaves, q_opts);
        queue.push((OrdF64(builders[i].best_split()), i));
    }

    let sub_trees: Vec<Option<EventMap>> = builders.iter().map(|b| Some(b.get_map())).collect();
    let map = input_map.copy_with(&sub_trees);

    SplitResult {
        map,
        num_leaves: *num_leaves,
        obj_impr: like_impr,
        smallest_split_change,
    }
}

// ---------------------------------------------------------------------------
// BuildTree (build-tree.cc:136)
// ---------------------------------------------------------------------------

/// Kaldi `BuildTree`. Returns `(tree, num_leaves)`.
///
/// * `thresh` — likelihood-improvement threshold for splitting (MFA: 300.0).
/// * `max_leaves` — 0 means no maximum.
/// * `cluster_thresh` — if negative, Kaldi replaces it with the smallest split
///   change actually taken (MFA passes -1, so this is the usual path). If it is
///   exactly 0.0 the post-clustering pass is skipped entirely.
/// * `round_num_leaves` — round the leaf count down to a multiple of 8.
#[allow(clippy::too_many_arguments)]
pub fn build_tree(
    qopts: &Questions,
    phone_sets: &[Vec<crate::types::PhoneId>],
    phone2num_pdf_classes: &[usize],
    share_roots: &[bool],
    do_split: &[bool],
    stats: &BuildTreeStats,
    thresh: f64,
    max_leaves: usize,
    cluster_thresh: f64,
    p: usize,
    round_num_leaves: bool,
) -> (EventMap, usize) {
    assert!(thresh > 0.0 || max_leaves > 0);
    assert!(!stats.is_empty());
    assert!(
        !phone_sets.is_empty()
            && phone_sets.len() == share_roots.len()
            && do_split.len() == phone_sets.len()
    );

    let mut num_leaves = 0i32;
    let tree_stub = get_stub_map(
        p,
        phone_sets,
        phone2num_pdf_classes,
        share_roots,
        &mut num_leaves,
    );

    let mut nonsplit_phones: Vec<EventValue> = Vec::new();
    for (i, set) in phone_sets.iter().enumerate() {
        if !do_split[i] {
            nonsplit_phones.extend(set.iter().map(|&ph| ph as EventValue));
        }
    }
    nonsplit_phones.sort_unstable();
    assert!(nonsplit_phones.windows(2).all(|w| w[0] < w[1]));

    let filtered_stats = filter_stats_by_key(stats, p as EventKey, &nonsplit_phones, false);

    // Kaldi's SplitDecisionTree asserts its SplitStatsByMap output is non-empty,
    // which only holds when at least one root is splittable and has stats. If
    // every root is `not-split` (or the filtered stats are otherwise empty)
    // there is nothing to split, so the stub map is the answer.
    let (tree_split, smallest_split_change) = if filtered_stats.is_empty() {
        (tree_stub.clone(), 0.0)
    } else {
        let split = split_decision_tree(
            &tree_stub,
            &filtered_stats,
            qopts,
            thresh,
            max_leaves as i32,
            &mut num_leaves,
        );
        (split.map, split.smallest_split_change)
    };

    let mut cluster_thresh = cluster_thresh;
    if cluster_thresh < 0.0 {
        cluster_thresh = smallest_split_change;
    }

    if cluster_thresh != 0.0 {
        let (tree_clustered, num_removed) =
            cluster_event_map_restricted_by_map(&tree_split, stats, cluster_thresh, &tree_stub);
        let (tree_renumbered, num_leaves_out) = if round_num_leaves {
            let num_leaves_required = ((num_leaves - num_removed) / 8) * 8;
            let (tree_rounded, _num_removed_in_rounding) =
                cluster_event_map_to_n_clusters_restricted_by_map(
                    &tree_clustered,
                    stats,
                    num_leaves_required,
                    &tree_stub,
                );
            renumber_event_map(&tree_rounded)
        } else {
            renumber_event_map(&tree_clustered)
        };
        (tree_renumbered, num_leaves_out.max(0) as usize)
    } else if round_num_leaves {
        let num_leaves_required = (num_leaves / 8) * 8;
        let (tree_rounded, _num_removed_in_rounding) =
            cluster_event_map_to_n_clusters_restricted_by_map(
                &tree_split,
                stats,
                num_leaves_required,
                &tree_stub,
            );
        let (tree_renumbered, num_leaves_out) = renumber_event_map(&tree_rounded);
        (tree_renumbered, num_leaves_out.max(0) as usize)
    } else {
        (tree_split, num_leaves.max(0) as usize)
    }
}

/// Kaldi `SumNormalizer` over the summed per-leaf stats: convenience used when
/// warning about low-count pdfs after tree building.
pub fn per_leaf_counts(stats: &BuildTreeStats, tree: &EventMap) -> Vec<f64> {
    split_stats_by_map(stats, tree)
        .iter()
        .map(sum_normalizer)
        .collect()
}

/// Kaldi `SumClusterableNormalizer` re-exported for callers that hold optional
/// per-leaf sums.
pub fn summed_normalizer(v: &[Option<GaussClusterable>]) -> f64 {
    sum_clusterable_normalizer(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::event_map::{EventType, K_PDF_CLASS};
    use crate::tree::questions::{Questions, QuestionsForKey};

    fn stat(phones: [i32; 3], pdf_class: i32, x: f32, count: f64) -> (EventType, GaussClusterable) {
        let mut e: EventType = vec![
            (K_PDF_CLASS, pdf_class),
            (0, phones[0]),
            (1, phones[1]),
            (2, phones[2]),
        ];
        e.sort_by_key(|p| p.0);
        let mut c = GaussClusterable::new(1, 0.01);
        c.add_stats(&[x], count);
        c.add_stats(&[x + 1.0], count);
        (e, c)
    }

    fn toy_stats() -> BuildTreeStats {
        let mut stats = Vec::new();
        for (li, &left) in [1i32, 2].iter().enumerate() {
            for pc in 0..3 {
                stats.push(stat([left, 3, 4], pc, 10.0 * li as f32 + pc as f32, 100.0));
            }
        }
        stats
    }

    fn toy_questions() -> Questions {
        let mut q = Questions::new();
        for key in 0..3 {
            q.set(
                key,
                QuestionsForKey::new(vec![vec![1], vec![1, 2], vec![3], vec![4]], 0),
            );
        }
        q.set(
            K_PDF_CLASS,
            QuestionsForKey::new(vec![vec![0], vec![0, 1]], 0),
        );
        q
    }

    #[test]
    fn build_tree_splits_on_left_context() {
        let stats = toy_stats();
        let q = toy_questions();
        let (tree, num_leaves) = build_tree(
            &q,
            &[vec![3]],
            &[0, 3, 3, 3, 3],
            &[true],
            &[true],
            &stats,
            1.0,
            0,
            0.0,
            1,
            false,
        );
        assert!(num_leaves > 1, "should have split");
        // Distinct left contexts land on different leaves for pdf-class 1.
        let e1: EventType = vec![(K_PDF_CLASS, 1), (0, 1), (1, 3), (2, 4)];
        let e2: EventType = vec![(K_PDF_CLASS, 1), (0, 2), (1, 3), (2, 4)];
        assert_ne!(tree.map(&e1), tree.map(&e2));
        // Every stat maps somewhere.
        for (e, _) in &stats {
            assert!(tree.map(e).is_some());
        }
    }

    #[test]
    fn build_tree_no_split_gives_stub_leaves() {
        let stats = toy_stats();
        let q = toy_questions();
        let (_, num_leaves) = build_tree(
            &q,
            &[vec![3]],
            &[0, 3, 3, 3, 3],
            &[true],
            &[false], // do not split this root
            &stats,
            1.0,
            0,
            0.0,
            1,
            false,
        );
        assert_eq!(num_leaves, 1);
    }
}
