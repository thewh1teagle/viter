//! Port of `plans/kaldi/src/tree/build-tree-questions.h` (`Questions`,
//! `QuestionsForKey`) and of `AutomaticallyObtainQuestions` /
//! `ObtainSetsOfPhones` / `KMeansClusterPhones` / `ReadRootsFile` from
//! `plans/kaldi/src/tree/build-tree.cc:559-900`.
//!
//! Also holds the two MFA/kalpy-shaped helpers: `make_questions` assembles the
//! `Questions` object the way `plans/kalpy/extensions/tree/tree.cpp:1169` does,
//! and `mfa_roots` reproduces the roots file MFA writes in
//! `plans/mfa/montreal_forced_aligner/dictionary/mixins.py:804`.

use super::cluster::{ClusterKMeansOptions, cluster_kmeans};
use super::stats::{BuildTreeStats, filter_stats_by_key, split_stats_by_key, sum_stats_vec};
use super::tree_cluster::{TreeClusterOptions, tree_cluster};
use super::clusterable::{GaussClusterable, ensure_not_null};
use super::event_map::{EventKey, EventValue, K_PDF_CLASS};
use crate::types::PhoneId;

pub use super::cluster::RefineClustersOptions;

/// Kaldi `QuestionsForKey`: the questions to try for one event key, plus the
/// refinement options used to specialize a question at each tree node.
#[derive(Clone, Debug)]
pub struct QuestionsForKey {
    /// Each entry is a sorted "yes set" of values.
    pub questions: Vec<Vec<EventValue>>,
    pub refine_opts: RefineClustersOptions,
}

impl QuestionsForKey {
    /// Kaldi's constructor: `refine_opts(num_iters, 2)`.
    pub fn new(questions: Vec<Vec<EventValue>>, num_iters: i32) -> Self {
        Self {
            questions,
            refine_opts: RefineClustersOptions::new(num_iters, 2),
        }
    }

    /// Kaldi `Check()`: every question must be sorted.
    pub fn check(&self) -> bool {
        self.questions
            .iter()
            .all(|q| q.windows(2).all(|w| w[0] <= w[1]))
    }
}

/// Kaldi `Questions`: per-key question sets. `BTreeMap` gives the same
/// ascending key order as Kaldi's `std::map`, which matters because
/// `DecisionTreeSplitter::FindBestSplit` iterates keys in that order and keeps
/// the first strictly-better split.
#[derive(Clone, Debug, Default)]
pub struct Questions {
    pub per_key: std::collections::BTreeMap<EventKey, QuestionsForKey>,
}

impl Questions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Kaldi `SetQuestionsOf`.
    pub fn set(&mut self, key: EventKey, q: QuestionsForKey) {
        assert!(q.check(), "questions for key {key} are not sorted");
        self.per_key.insert(key, q);
    }

    /// Kaldi `HasQuestionsForKey`.
    pub fn has_key(&self, k: EventKey) -> bool {
        self.per_key.contains_key(&k)
    }

    /// Kaldi `GetQuestionsOf`. Panics if the key has no questions.
    pub fn get(&self, k: EventKey) -> &QuestionsForKey {
        self.per_key
            .get(&k)
            .unwrap_or_else(|| panic!("Questions: no options for key {k}"))
    }

    /// Kaldi `GetKeysWithQuestions` (ascending).
    pub fn keys(&self) -> Vec<EventKey> {
        self.per_key.keys().copied().collect()
    }
}

/// Build the `Questions` object exactly as kalpy's `build_tree`
/// (`plans/kalpy/extensions/tree/tree.cpp:1169`) does: the same phone-set
/// questions for every context position `0..n-1`, plus pdf-class questions
/// `[[0], [0,1], ...]` for `kPdfClass`.
///
/// `max_num_pdf_classes` is the largest number of pdf-classes of any phone in
/// the topology (3 for MFA's non-silence topology, 5 for silence), giving
/// `max_num_pdf_classes - 1` prefix questions.
///
/// `num_iters_refine` is kalpy's default 0, i.e. questions are used as given.
pub fn make_questions_with(
    phone_questions: &[Vec<PhoneId>],
    n: usize,
    max_num_pdf_classes: usize,
    num_iters_refine: i32,
) -> Questions {
    // kalpy sorts each question and then sorts-and-uniqs the list of questions.
    let mut questions: Vec<Vec<EventValue>> = phone_questions
        .iter()
        .map(|q| {
            let mut q: Vec<EventValue> = q.iter().map(|&p| p as EventValue).collect();
            q.sort_unstable();
            assert!(
                q.windows(2).all(|w| w[0] < w[1]),
                "Questions contain duplicate phones"
            );
            q
        })
        .collect();
    questions.sort();
    questions.dedup();

    let mut qo = Questions::new();
    for key in 0..n {
        qo.set(
            key as EventKey,
            QuestionsForKey::new(questions.clone(), num_iters_refine),
        );
    }
    let mut pdfclass_questions: Vec<Vec<EventValue>> = Vec::new();
    for i in 0..max_num_pdf_classes.saturating_sub(1) {
        pdfclass_questions.push((0..=i as EventValue).collect());
    }
    qo.set(
        K_PDF_CLASS,
        QuestionsForKey::new(pdfclass_questions, num_iters_refine),
    );
    qo
}

/// `make_questions_with` with MFA/kalpy's defaults: `max_num_pdf_classes = 3`
/// (from the 3-state non-silence topology; MFA's 5-state silence phones are
/// context-independent roots and never asked pdf-class questions beyond that)
/// and `num_iters_refine = 0`.
///
/// CONTRACT-DEVIATION: the contract fixes the pdf-class questions at
/// `[[0],[0,1]]`; kalpy derives them from the topology's max pdf-class count.
/// Use `make_questions_with` to pass a different maximum (e.g. 5 for MFA's
/// silence topology, giving `[[0],[0,1],[0,1,2],[0,1,2,3]]`).
pub fn make_questions(phone_questions: &[Vec<PhoneId>], n: usize) -> Questions {
    make_questions_with(phone_questions, n, 3, 0)
}

// ---------------------------------------------------------------------------
// AutomaticallyObtainQuestions (build-tree.cc:615)
// ---------------------------------------------------------------------------

/// Kaldi `ObtainSetsOfPhones`: turn a `TreeCluster` result into phone sets.
/// For every node of the cluster tree, the union of the phone sets under it is
/// one question; the list is reversed so top-level questions come first, then
/// the original phone sets are appended and duplicates removed.
fn obtain_sets_of_phones(
    phone_sets: &[Vec<PhoneId>],
    assignments: &[usize],
    clust_assignments: &[usize],
    num_leaves: usize,
) -> Vec<Vec<PhoneId>> {
    assert!(num_leaves < clust_assignments.len());
    assert_eq!(assignments.len(), phone_sets.len());
    let mut raw_sets: Vec<Vec<PhoneId>> = vec![Vec::new(); clust_assignments.len()];
    for (i, &clust) in assignments.iter().enumerate() {
        assert!(clust < num_leaves);
        raw_sets[clust].extend(phone_sets[i].iter().copied());
    }
    // Propagate each node's phones up to its parent (root excluded).
    for j in 0..clust_assignments.len() {
        let parent = clust_assignments[j];
        raw_sets[j].sort_unstable();
        assert!(raw_sets[j].windows(2).all(|w| w[0] < w[1]));
        if parent < clust_assignments.len() - 1 {
            let to_add = raw_sets[j].clone();
            raw_sets[parent].extend(to_add);
        }
    }
    raw_sets.reverse();
    for set in phone_sets {
        raw_sets.push(set.clone());
    }
    // RemoveDuplicates: keep the first occurrence of each distinct set.
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(raw_sets.len());
    for s in raw_sets {
        if s.is_empty() {
            continue;
        }
        if seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

/// Sum stats per phone set for the retained pdf-classes; shared by
/// `automatically_obtain_questions` and `kmeans_cluster_phones`.
fn summed_stats_per_set(
    stats: &BuildTreeStats,
    phone_sets: &[Vec<PhoneId>],
    all_pdf_classes: &[i32],
    p: usize,
) -> Vec<GaussClusterable> {
    let mut phones: Vec<PhoneId> = Vec::new();
    for set in phone_sets {
        assert!(!set.is_empty(), "Empty phone set");
        assert!(
            set.windows(2).all(|w| w[0] < w[1]),
            "Phone set contains duplicate phones"
        );
        phones.extend(set.iter().copied());
    }
    phones.sort_unstable();
    assert!(
        phones.windows(2).all(|w| w[0] < w[1]),
        "Phones present in more than one phone set"
    );
    assert!(!phones.is_empty(), "No phones provided");

    let mut all_pdf_classes: Vec<EventValue> = all_pdf_classes.to_vec();
    all_pdf_classes.sort_unstable();
    all_pdf_classes.dedup();
    assert!(!all_pdf_classes.is_empty());

    let retained = filter_stats_by_key(stats, K_PDF_CLASS, &all_pdf_classes, true);
    let split = split_stats_by_key(&retained, p as EventKey);
    let mut summed = sum_stats_vec(&split);

    let max_phone = *phones.last().unwrap() as usize;
    if summed.len() < max_phone + 1 {
        summed.resize(max_phone + 1, None);
    }
    ensure_not_null(&mut summed);

    phone_sets
        .iter()
        .map(|set| {
            let mut acc = summed[set[0] as usize]
                .clone()
                .expect("ensure_not_null filled every slot");
            for &ph in &set[1..] {
                acc.add(summed[ph as usize].as_ref().expect("filled"));
            }
            acc
        })
        .collect()
}

/// Kaldi `AutomaticallyObtainQuestions` (`build-tree.cc:615`), the function
/// kalpy exposes as `automatically_obtain_questions`
/// (`plans/kalpy/extensions/tree/tree.cpp:684`).
///
/// Clusters the phone sets by their stats for the given pdf-classes (MFA: `[1]`,
/// i.e. only the central state) into a binary tree, and returns one phone set
/// per node of that tree.
pub fn automatically_obtain_questions(
    stats: &BuildTreeStats,
    phone_sets_in: &[Vec<PhoneId>],
    all_pdf_classes: &[i32],
    p: usize,
) -> Vec<Vec<PhoneId>> {
    let phone_sets: Vec<Vec<PhoneId>> = phone_sets_in
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.sort_unstable();
            s
        })
        .collect();

    let per_set = summed_stats_per_set(stats, &phone_sets, all_pdf_classes, p);

    let mut topts = TreeClusterOptions::default();
    // Kaldi: slow-but-accurate, since there are typically few phones.
    topts.kmeans_cfg.num_tries = 10;

    let (_impr, _clusters, assignments, clust_assignments, num_leaves) =
        tree_cluster(&per_set, per_set.len(), &topts);

    obtain_sets_of_phones(&phone_sets, &assignments, &clust_assignments, num_leaves)
}

/// Kaldi `KMeansClusterPhones`: partition the phone sets into `num_classes`
/// classes with k-means on the same per-set stats.
pub fn kmeans_cluster_phones(
    stats: &BuildTreeStats,
    phone_sets_in: &[Vec<PhoneId>],
    all_pdf_classes: &[i32],
    p: usize,
    num_classes: usize,
) -> Vec<Vec<PhoneId>> {
    let phone_sets: Vec<Vec<PhoneId>> = phone_sets_in
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.sort_unstable();
            s
        })
        .collect();
    let per_set = summed_stats_per_set(stats, &phone_sets, all_pdf_classes, p);

    let opts = ClusterKMeansOptions::default();
    let (_impr, _clusters, assignments) = cluster_kmeans(&per_set, num_classes, &opts);

    let mut sets_out: Vec<Vec<PhoneId>> = vec![Vec::new(); num_classes];
    assert_eq!(assignments.len(), phone_sets.len());
    for (i, &class_idx) in assignments.iter().enumerate() {
        sets_out[class_idx].extend(phone_sets[i].iter().copied());
    }
    for s in sets_out.iter_mut() {
        s.sort_unstable();
    }
    sets_out
}

// ---------------------------------------------------------------------------
// Roots
// ---------------------------------------------------------------------------

/// Kaldi `ReadRootsFile`. Lines are `shared|not-shared split|not-split p1 p2 ...`.
/// Returns `(phone_sets, is_shared_root, is_split_root)`.
pub fn read_roots(text: &str) -> (Vec<Vec<PhoneId>>, Vec<bool>, Vec<bool>) {
    let mut phone_sets = Vec::new();
    let mut is_shared = Vec::new();
    let mut is_split = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let shared = it.next().unwrap_or("");
        let split = it.next().unwrap_or("");
        assert!(
            shared == "shared" || shared == "not-shared",
            "Bad line {} in roots: {line}",
            line_number + 1
        );
        assert!(
            split == "split" || split == "not-split",
            "Bad line {} in roots: {line}",
            line_number + 1
        );
        is_shared.push(shared == "shared");
        is_split.push(split == "split");
        let mut set: Vec<PhoneId> = it
            .map(|t| {
                t.parse::<PhoneId>()
                    .unwrap_or_else(|_| panic!("Bad phone id '{t}' in roots line {line}"))
            })
            .collect();
        set.sort_unstable();
        assert!(
            !set.is_empty() && set[0] > 0 && set.windows(2).all(|w| w[0] < w[1]),
            "Bad line {} in roots (empty, non-positive or duplicate phone-ids): {line}",
            line_number + 1
        );
        phone_sets.push(set);
    }
    assert!(!phone_sets.is_empty(), "Empty roots file");
    (phone_sets, is_shared, is_split)
}

/// Render roots back into Kaldi's `roots.int` text form.
pub fn write_roots(
    phone_sets: &[Vec<PhoneId>],
    is_shared: &[bool],
    is_split: &[bool],
) -> String {
    let mut out = String::new();
    for i in 0..phone_sets.len() {
        out.push_str(if is_shared[i] { "shared " } else { "not-shared " });
        out.push_str(if is_split[i] { "split" } else { "not-split" });
        for ph in &phone_sets[i] {
            out.push(' ');
            out.push_str(&ph.to_string());
        }
        out.push('\n');
    }
    out
}

/// MFA's tree roots, as written by `_write_phone_sets`
/// (`plans/mfa/montreal_forced_aligner/dictionary/mixins.py:804`) with the
/// default `shared_silence_phones = False`:
///
/// * one root per silence phone group (its position variants share a root),
///   `shared split`;
/// * one root per non-silence phone group (again, position variants of a phone
///   share a root), `shared split`.
///
/// `silence_phone_groups` and `nonsilence_phone_groups` each hold the position
/// variants of one base phone. Returns `(phone_sets, share_roots, do_split)`.
///
/// CONTRACT-DEVIATION: plans/CONTRACTS.md summarises silence roots as "shared, not
/// split (ci)". MFA only writes `not-shared not-split` for silence when
/// `shared_silence_phones` is true, which is not the default; this function
/// follows MFA's actual default output. Pass `shared_silence = true` to get the
/// single `not-shared not-split` silence root.
pub fn mfa_roots_with(
    nonsilence_phone_groups: &[Vec<PhoneId>],
    silence_phone_groups: &[Vec<PhoneId>],
    shared_silence: bool,
) -> (Vec<Vec<PhoneId>>, Vec<bool>, Vec<bool>) {
    let mut phone_sets: Vec<Vec<PhoneId>> = Vec::new();
    let mut share_roots: Vec<bool> = Vec::new();
    let mut do_split: Vec<bool> = Vec::new();

    if shared_silence {
        let mut all: Vec<PhoneId> = silence_phone_groups.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        if !all.is_empty() {
            phone_sets.push(all);
            share_roots.push(false);
            do_split.push(false);
        }
    } else {
        for group in silence_phone_groups {
            let mut g = group.clone();
            g.sort_unstable();
            g.dedup();
            if g.is_empty() {
                continue;
            }
            phone_sets.push(g);
            share_roots.push(true);
            do_split.push(true);
        }
    }

    for group in nonsilence_phone_groups {
        let mut g = group.clone();
        g.sort_unstable();
        g.dedup();
        if g.is_empty() {
            continue;
        }
        phone_sets.push(g);
        share_roots.push(true);
        do_split.push(true);
    }

    (phone_sets, share_roots, do_split)
}

/// `mfa_roots_with` for the common shape where each silence phone is its own
/// group (MFA's default, `shared_silence_phones = False`).
pub fn mfa_roots(
    nonsilence_phone_groups: &[Vec<PhoneId>],
    silence_phones: &[PhoneId],
) -> (Vec<Vec<PhoneId>>, Vec<bool>, Vec<bool>) {
    let silence_groups: Vec<Vec<PhoneId>> = silence_phones.iter().map(|&p| vec![p]).collect();
    mfa_roots_with(nonsilence_phone_groups, &silence_groups, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::event_map::EventType;

    #[test]
    fn make_questions_matches_kalpy_shape() {
        let q = make_questions(&[vec![3, 4], vec![3]], 3);
        assert_eq!(q.keys(), vec![-1, 0, 1, 2]);
        // Questions are sorted and uniqued across the list.
        assert_eq!(q.get(0).questions, vec![vec![3], vec![3, 4]]);
        assert_eq!(q.get(K_PDF_CLASS).questions, vec![vec![0], vec![0, 1]]);
        assert_eq!(q.get(0).refine_opts.num_iters, 0);
        assert_eq!(q.get(0).refine_opts.top_n, 2);
    }

    #[test]
    fn make_questions_with_five_pdf_classes() {
        let q = make_questions_with(&[vec![3]], 3, 5, 0);
        assert_eq!(
            q.get(K_PDF_CLASS).questions,
            vec![vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]]
        );
    }

    #[test]
    fn roots_roundtrip() {
        let text = "not-shared not-split 1 2 3\nshared split 5 4\n";
        let (sets, shared, split) = read_roots(text);
        assert_eq!(sets, vec![vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(shared, vec![false, true]);
        assert_eq!(split, vec![false, true]);
        let out = write_roots(&sets, &shared, &split);
        assert_eq!(out, "not-shared not-split 1 2 3\nshared split 4 5\n");
    }

    #[test]
    fn mfa_roots_default_shape() {
        let (sets, shared, split) = mfa_roots(&[vec![10, 11, 12, 13], vec![20, 21]], &[1, 2]);
        assert_eq!(sets, vec![vec![1], vec![2], vec![10, 11, 12, 13], vec![20, 21]]);
        assert_eq!(shared, vec![true, true, true, true]);
        assert_eq!(split, vec![true, true, true, true]);
    }

    #[test]
    fn mfa_roots_shared_silence() {
        let (sets, shared, split) =
            mfa_roots_with(&[vec![10, 11]], &[vec![1], vec![2, 3]], true);
        assert_eq!(sets, vec![vec![1, 2, 3], vec![10, 11]]);
        assert_eq!(shared, vec![false, true]);
        assert_eq!(split, vec![false, true]);
    }

    fn phone_stat(phone: PhoneId, pdf_class: i32, x: f32) -> (EventType, GaussClusterable) {
        let e: EventType = vec![(K_PDF_CLASS, pdf_class), (1, phone as EventValue)];
        let mut c = GaussClusterable::new(1, 0.01);
        c.add_stats(&[x], 50.0);
        c.add_stats(&[x + 0.5], 50.0);
        (e, c)
    }

    #[test]
    fn automatic_questions_group_similar_phones() {
        // Phones 1,2 near 0; phones 3,4 near 20.
        let stats: BuildTreeStats = vec![
            phone_stat(1, 1, 0.0),
            phone_stat(2, 1, 0.3),
            phone_stat(3, 1, 20.0),
            phone_stat(4, 1, 20.3),
            // pdf-class 0 stats must be ignored by the [1] filter.
            phone_stat(1, 0, 100.0),
        ];
        let phone_sets = vec![vec![1], vec![2], vec![3], vec![4]];
        let questions = automatically_obtain_questions(&stats, &phone_sets, &[1], 1);
        assert!(!questions.is_empty());
        // Kaldi's ObtainSetsOfPhones never emits the top-level all-phones set:
        // the propagation loop in build-tree.cc:582 skips nodes whose parent is
        // the root, so the root cluster stays empty and is dropped. The
        // interesting output is the intermediate cluster sets.
        assert!(questions.contains(&vec![1, 2]));
        assert!(questions.contains(&vec![3, 4]));
        // The singleton input sets are always present.
        for s in &phone_sets {
            assert!(questions.contains(s), "missing singleton {s:?}");
        }
        // No duplicates.
        let mut sorted = questions.clone();
        sorted.sort();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len());
    }

    #[test]
    fn kmeans_cluster_phones_partitions() {
        let stats: BuildTreeStats = vec![
            phone_stat(1, 1, 0.0),
            phone_stat(2, 1, 0.3),
            phone_stat(3, 1, 20.0),
            phone_stat(4, 1, 20.3),
        ];
        let sets =
            kmeans_cluster_phones(&stats, &[vec![1], vec![2], vec![3], vec![4]], &[1], 1, 2);
        assert_eq!(sets.len(), 2);
        let total: usize = sets.iter().map(|s| s.len()).sum();
        assert_eq!(total, 4);
    }
}
