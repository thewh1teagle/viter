//! Port of the leaf-clustering and stub-map parts of
//! `plans/kaldi/src/tree/build-tree-utils.{h,cc}`:
//! `ClusterEventMapGetMapping`, `ClusterEventMap`,
//! `ClusterEventMapRestrictedBy{Map,Keys}`,
//! `ClusterEventMapToNClustersRestrictedByMap`, `RenumberEventMap`,
//! `MapEventMapLeaves`, `ShareEventMapLeaves` and `GetStubMap`.

use super::cluster::{cluster_bottom_up, cluster_bottom_up_compartmentalized};
use super::clusterable::GaussClusterable;
use super::event_map::{EventKey, EventMap, EventType, EventValue, K_PDF_CLASS};
use super::stats::{BuildTreeStats, split_stats_by_key, split_stats_by_map, sum_stats_vec};

// ---------------------------------------------------------------------------
// Clustering the tree (build-tree-utils.cc:611 onwards)
// ---------------------------------------------------------------------------

/// Kaldi `ClusterEventMapGetMapping`: bottom-up cluster the leaves of `e_in`
/// that the given stats reach, appending remappings to `mapping`. Returns the
/// number of leaves removed.
pub fn cluster_event_map_get_mapping(
    e_in: &EventMap,
    stats: &BuildTreeStats,
    thresh: f64,
    mapping: &mut Vec<Option<EventMap>>,
) -> i32 {
    assert!(!stats.is_empty());
    let split_stats = split_stats_by_map(stats, e_in);
    let summed_stats = sum_stats_vec(&split_stats);

    let mut indexes: Vec<usize> = Vec::new();
    let mut summed_stats_contiguous: Vec<GaussClusterable> = Vec::new();
    let mut max_index = 0usize;
    for (i, s) in summed_stats.iter().enumerate() {
        if let Some(s) = s {
            indexes.push(i);
            summed_stats_contiguous.push(s.clone());
            if i > max_index {
                max_index = i;
            }
        }
    }
    if summed_stats_contiguous.is_empty() {
        return 0;
    }

    let (_change, _clusters, assignments) = cluster_bottom_up(&summed_stats_contiguous, thresh, 0);
    assert_eq!(assignments.len(), summed_stats_contiguous.len());
    let num_clust = assignments.iter().max().unwrap() + 1;
    let num_combined = summed_stats_contiguous.len() as i32 - num_clust as i32;
    assert!(num_combined >= 0);

    if max_index >= mapping.len() {
        mapping.resize(max_index + 1, None);
    }
    for i in 0..summed_stats_contiguous.len() {
        let index = indexes[i];
        // Map to an index that already exists in this part of the tree, so we
        // do not collide with leaf ids used elsewhere.
        let new_index = indexes[assignments[i]];
        assert!(
            mapping[index].is_none(),
            "overlapping index sets in cluster"
        );
        mapping[index] = Some(EventMap::Constant(new_index as crate::types::PdfId));
    }
    num_combined
}

/// Kaldi `ClusterEventMap`.
pub fn cluster_event_map(e_in: &EventMap, stats: &BuildTreeStats, thresh: f64) -> (EventMap, i32) {
    let mut mapping = Vec::new();
    let num_removed = cluster_event_map_get_mapping(e_in, stats, thresh, &mut mapping);
    (e_in.copy_with(&mapping), num_removed)
}

/// Kaldi `ClusterEventMapRestrictedByMap`: cluster leaves separately within
/// each bucket of `e_restrict` (in `BuildTree`, the stub map / tree roots), so
/// leaves from different roots are never merged.
pub fn cluster_event_map_restricted_by_map(
    e_in: &EventMap,
    stats: &BuildTreeStats,
    thresh: f64,
    e_restrict: &EventMap,
) -> (EventMap, i32) {
    let mut mapping: Vec<Option<EventMap>> = Vec::new();
    let mut num_removed = 0;
    let split_stats = split_stats_by_map(stats, e_restrict);
    for s in &split_stats {
        if !s.is_empty() {
            num_removed += cluster_event_map_get_mapping(e_in, s, thresh, &mut mapping);
        }
    }
    (e_in.copy_with(&mapping), num_removed)
}

/// Kaldi `ClusterEventMapRestrictedByKeys`.
pub fn cluster_event_map_restricted_by_keys(
    e_in: &EventMap,
    stats: &BuildTreeStats,
    thresh: f64,
    keys: &[EventKey],
) -> (EventMap, i32) {
    fn helper(
        e_in: &EventMap,
        stats: &BuildTreeStats,
        thresh: f64,
        keys: &[EventKey],
        mapping: &mut Vec<Option<EventMap>>,
    ) -> i32 {
        if keys.is_empty() {
            return cluster_event_map_get_mapping(e_in, stats, thresh, mapping);
        }
        let last = keys[keys.len() - 1];
        let rest = &keys[..keys.len() - 1];
        let split_stats = split_stats_by_key(stats, last);
        let mut ans = 0;
        for s in &split_stats {
            if !s.is_empty() {
                ans += helper(e_in, s, thresh, rest, mapping);
            }
        }
        ans
    }
    let mut mapping = Vec::new();
    let n = helper(e_in, stats, thresh, keys, &mut mapping);
    (e_in.copy_with(&mapping), n)
}

/// Kaldi `ClusterEventMapToNClustersRestrictedByMap`: merge leaves within each
/// bucket of `e_restrict` until exactly `num_clusters_required` remain.
/// Used by `BuildTree` for `round_num_leaves`.
pub fn cluster_event_map_to_n_clusters_restricted_by_map(
    e_in: &EventMap,
    stats: &BuildTreeStats,
    num_clusters_required: i32,
    e_restrict: &EventMap,
) -> (EventMap, i32) {
    let split_stats = split_stats_by_map(stats, e_restrict);
    if (num_clusters_required as usize) < split_stats.len() {
        return (e_in.clone(), 0);
    }

    let mut indexes: Vec<Vec<usize>> = vec![Vec::new(); split_stats.len()];
    let mut summed_contiguous: Vec<Vec<GaussClusterable>> = vec![Vec::new(); split_stats.len()];
    let mut max_index = 0usize;
    let mut num_non_empty_clusters_required = num_clusters_required;
    let mut num_non_empty_clusters = 0i32;

    for (i, s) in split_stats.iter().enumerate() {
        if !s.is_empty() {
            let split_i = split_stats_by_map(s, e_in);
            let summed_i = sum_stats_vec(&split_i);
            for (j, c) in summed_i.iter().enumerate() {
                if let Some(c) = c {
                    num_non_empty_clusters += 1;
                    indexes[i].push(j);
                    summed_contiguous[i].push(c.clone());
                    if j > max_index {
                        max_index = j;
                    }
                }
            }
        } else {
            num_non_empty_clusters_required -= 1;
        }
    }

    if num_non_empty_clusters_required > num_non_empty_clusters {
        return (e_in.clone(), 0);
    }

    let (_change, assignments) = cluster_bottom_up_compartmentalized(
        &summed_contiguous,
        f64::INFINITY,
        num_non_empty_clusters_required.max(0) as usize,
    );
    assert_eq!(assignments.len(), split_stats.len());

    let mut num_combined = 0i32;
    for i in 0..split_stats.len() {
        if assignments[i].is_empty() {
            continue;
        }
        let num_clust_i = assignments[i].iter().max().unwrap() + 1;
        num_combined += summed_contiguous[i].len() as i32 - num_clust_i as i32;
    }

    let mut leaf_mapping: Vec<Option<EventMap>> = vec![None; max_index + 1];
    for i in 0..split_stats.len() {
        for j in 0..summed_contiguous[i].len() {
            let index = indexes[i][j];
            let new_index = indexes[i][assignments[i][j]];
            assert!(leaf_mapping[index].is_none());
            leaf_mapping[index] = Some(EventMap::Constant(new_index as crate::types::PdfId));
        }
    }
    (e_in.copy_with(&leaf_mapping), num_combined)
}

/// Kaldi `RenumberEventMap`: renumber leaves to 0..num_leaves-1 in increasing
/// order of their previous ids. Returns `(map, num_leaves)`.
pub fn renumber_event_map(e_in: &EventMap) -> (EventMap, i32) {
    let mut initial_leaves = Vec::new();
    e_in.multi_map(&[], &mut initial_leaves);
    if initial_leaves.is_empty() {
        return (e_in.clone(), 0);
    }
    initial_leaves.sort_unstable();
    initial_leaves.dedup();
    let max_leaf_plus_one = initial_leaves[initial_leaves.len() - 1] + 1;
    let mut mapping: Vec<Option<EventMap>> = vec![None; max_leaf_plus_one.max(0) as usize];
    let mut cur_leaf = 0i32;
    for &l in &initial_leaves {
        assert!(l >= 0 && l < max_leaf_plus_one);
        mapping[l as usize] = Some(EventMap::Constant(cur_leaf as crate::types::PdfId));
        cur_leaf += 1;
    }
    (e_in.copy_with(&mapping), cur_leaf)
}

/// Kaldi `MapEventMapLeaves`.
pub fn map_event_map_leaves(e_in: &EventMap, mapping_in: &[i32]) -> EventMap {
    let mapping: Vec<Option<EventMap>> = mapping_in
        .iter()
        .map(|&m| Some(EventMap::Constant(m as crate::types::PdfId)))
        .collect();
    e_in.copy_with(&mapping)
}

/// Kaldi `ShareEventMapLeaves`: force all leaves reachable for the values in
/// each bucket of `values` to share a single leaf, then renumber.
pub fn share_event_map_leaves(
    e_in: &EventMap,
    key: EventKey,
    values: &[Vec<EventValue>],
) -> (EventMap, i32) {
    let mut pdfs: Vec<Vec<i32>> = vec![Vec::new(); values.len()];
    for (i, bucket) in values.iter().enumerate() {
        let mut evec: EventType = Vec::new();
        for &v in bucket {
            evec.push((key, v));
            e_in.multi_map(&evec, &mut pdfs[i]);
            evec.pop();
        }
        pdfs[i].sort_unstable();
        pdfs[i].dedup();
    }
    let mut remapping: Vec<Option<EventMap>> = Vec::new();
    for bucket in &pdfs {
        if bucket.is_empty() {
            continue;
        }
        let map_to_this = bucket[0];
        for &leaf in &bucket[1..] {
            assert!(leaf >= 0);
            let leaf = leaf as usize;
            if remapping.len() <= leaf {
                remapping.resize(leaf + 1, None);
            }
            assert!(remapping[leaf].is_none());
            remapping[leaf] = Some(EventMap::Constant(map_to_this as crate::types::PdfId));
        }
    }
    let shared = e_in.copy_with(&remapping);
    renumber_event_map(&shared)
}

/// Kaldi `GetStubMap`: build the initial (pre-split) tree from the roots.
/// One leaf per shared root, or one leaf per pdf-class for a non-shared root.
pub fn get_stub_map(
    p: usize,
    phone_sets: &[Vec<crate::types::PhoneId>],
    phone2num_pdf_classes: &[usize],
    share_roots: &[bool],
    num_leaves_out: &mut i32,
) -> EventMap {
    assert!(!phone_sets.is_empty() && share_roots.len() == phone_sets.len());
    {
        let mut all = std::collections::HashSet::new();
        for set in phone_sets {
            assert!(!set.is_empty());
            assert!(
                set.windows(2).all(|w| w[0] < w[1]),
                "phone set not sorted/uniq"
            );
            for &ph in set {
                assert!(all.insert(ph), "phone {ph} in more than one root");
            }
        }
    }
    get_stub_map_inner(
        p,
        phone_sets,
        phone2num_pdf_classes,
        share_roots,
        num_leaves_out,
    )
}

fn get_stub_map_inner(
    p: usize,
    phone_sets: &[Vec<crate::types::PhoneId>],
    phone2num_pdf_classes: &[usize],
    share_roots: &[bool],
    num_leaves_out: &mut i32,
) -> EventMap {
    let max_set_size = phone_sets.iter().map(|s| s.len()).max().unwrap_or(0);
    let highest_numbered_phone = phone_sets
        .iter()
        .map(|s| *s.iter().max().unwrap())
        .max()
        .unwrap_or(0) as i64;

    if phone_sets.len() == 1 {
        if share_roots[0] {
            let leaf = *num_leaves_out;
            *num_leaves_out += 1;
            return EventMap::Constant(leaf as crate::types::PdfId);
        }
        // Not sharing roots: split on pdf-class.
        let mut max_len = 0usize;
        for (i, &phone) in phone_sets[0].iter().enumerate() {
            let len = phone2num_pdf_classes[phone as usize];
            assert!(len > 0);
            if i == 0 {
                max_len = len;
            } else {
                max_len = max_len.max(len);
            }
        }
        let mut m = Vec::with_capacity(max_len);
        for pc in 0..max_len {
            m.push((pc as EventValue, *num_leaves_out));
            *num_leaves_out += 1;
        }
        return super::event_map::table_from_answers(K_PDF_CLASS, &m);
    }

    if max_set_size == 1 && phone_sets.len() as i64 <= 2 * highest_numbered_phone {
        // Table split on the central phone -- more efficient.
        let mut m = Vec::with_capacity(phone_sets.len());
        for i in 0..phone_sets.len() {
            let sub = get_stub_map_inner(
                p,
                std::slice::from_ref(&phone_sets[i]),
                phone2num_pdf_classes,
                &share_roots[i..i + 1],
                num_leaves_out,
            );
            m.push((phone_sets[i][0] as EventValue, sub));
        }
        return super::event_map::table_from_maps(p as EventKey, m);
    }

    // Otherwise split the list of roots in half and recurse.
    let half_sz = phone_sets.len() / 2;
    let map1 = get_stub_map_inner(
        p,
        &phone_sets[..half_sz],
        phone2num_pdf_classes,
        &share_roots[..half_sz],
        num_leaves_out,
    );
    let map2 = get_stub_map_inner(
        p,
        &phone_sets[half_sz..],
        phone2num_pdf_classes,
        &share_roots[half_sz..],
        num_leaves_out,
    );
    let mut all_in_first_set: Vec<EventValue> = phone_sets[..half_sz]
        .iter()
        .flat_map(|s| s.iter().map(|&ph| ph as EventValue))
        .collect();
    all_in_first_set.sort_unstable();
    EventMap::Split {
        key: p as EventKey,
        yes_set: all_in_first_set,
        yes: Box::new(map1),
        no: Box::new(map2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_map_shared_root_is_one_leaf() {
        let mut n = 0;
        let m = get_stub_map(1, &[vec![3]], &[0, 3, 3, 3, 3], &[true], &mut n);
        assert_eq!(n, 1);
        assert_eq!(m, EventMap::Constant(0));
    }

    #[test]
    fn stub_map_nonshared_root_splits_on_pdf_class() {
        let mut n = 0;
        let m = get_stub_map(1, &[vec![3]], &[0, 3, 3, 3, 3], &[false], &mut n);
        assert_eq!(n, 3);
        assert_eq!(m.map(&[(K_PDF_CLASS, 0)]), Some(0));
        assert_eq!(m.map(&[(K_PDF_CLASS, 2)]), Some(2));
    }

    #[test]
    fn stub_map_multiple_roots_are_distinct() {
        let mut n = 0;
        let m = get_stub_map(
            1,
            &[vec![1], vec![2], vec![3]],
            &[0, 3, 3, 3],
            &[true, true, true],
            &mut n,
        );
        assert_eq!(n, 3);
        let a = m.map(&[(K_PDF_CLASS, 0), (1, 1)]).unwrap();
        let b = m.map(&[(K_PDF_CLASS, 0), (1, 2)]).unwrap();
        let c = m.map(&[(K_PDF_CLASS, 0), (1, 3)]).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn stub_map_many_roots_recurses() {
        // 6 singleton roots with small ids: the recursion path (not the table
        // path) is taken because phone_sets.len() > 2 * highest phone is false
        // here; either way every root must get its own leaf.
        let sets: Vec<Vec<crate::types::PhoneId>> = (1..=6u32).map(|p| vec![p]).collect();
        let mut n = 0;
        let m = get_stub_map(1, &sets, &[3; 7], &[true; 6], &mut n);
        assert_eq!(n, 6);
        let leaves: std::collections::HashSet<_> = (1..=6u32)
            .map(|p| m.map(&[(K_PDF_CLASS, 0), (1, p as EventValue)]).unwrap())
            .collect();
        assert_eq!(leaves.len(), 6);
    }

    #[test]
    fn renumber_makes_leaves_contiguous() {
        let m = EventMap::Split {
            key: 0,
            yes_set: vec![1],
            yes: Box::new(EventMap::Constant(5)),
            no: Box::new(EventMap::Constant(9)),
        };
        let (r, n) = renumber_event_map(&m);
        assert_eq!(n, 2);
        assert_eq!(r.map(&[(0, 1)]), Some(0));
        assert_eq!(r.map(&[(0, 0)]), Some(1));
    }

    #[test]
    fn map_event_map_leaves_applies_mapping() {
        let m = EventMap::Split {
            key: 0,
            yes_set: vec![1],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        let mapped = map_event_map_leaves(&m, &[7, 8]);
        assert_eq!(mapped.map(&[(0, 1)]), Some(7));
        assert_eq!(mapped.map(&[(0, 0)]), Some(8));
    }

    #[test]
    fn share_event_map_leaves_merges() {
        let m = EventMap::Split {
            key: 0,
            yes_set: vec![1],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        let (shared, n) = share_event_map_leaves(&m, 0, &[vec![0, 1]]);
        assert_eq!(n, 1);
        assert_eq!(shared.map(&[(0, 1)]), Some(0));
        assert_eq!(shared.map(&[(0, 0)]), Some(0));
    }

    fn leaf_stat(leaf_phone: EventValue, x: f32) -> (EventType, GaussClusterable) {
        let e: EventType = vec![(K_PDF_CLASS, 0), (0, leaf_phone)];
        let mut c = GaussClusterable::new(1, 0.01);
        c.add_stats(&[x], 100.0);
        c.add_stats(&[x + 1.0], 100.0);
        (e, c)
    }

    #[test]
    fn cluster_event_map_merges_similar_leaves() {
        // Four leaves keyed on key 0; two pairs are near-identical.
        let m = super::super::event_map::table_from_answers(0, &[(1, 0), (2, 1), (3, 2), (4, 3)]);
        let stats: BuildTreeStats = vec![
            leaf_stat(1, 0.0),
            leaf_stat(2, 0.02),
            leaf_stat(3, 50.0),
            leaf_stat(4, 50.02),
        ];
        let (clustered, num_removed) = cluster_event_map(&m, &stats, 1.0e10);
        assert!(num_removed >= 1);
        let (renumbered, n) = renumber_event_map(&clustered);
        assert!(n < 4);
        assert_eq!(
            renumbered.map(&[(K_PDF_CLASS, 0), (0, 1)]),
            renumbered.map(&[(K_PDF_CLASS, 0), (0, 2)])
        );
    }

    #[test]
    fn restricted_by_map_never_merges_across_roots() {
        let m = super::super::event_map::table_from_answers(0, &[(1, 0), (2, 1), (3, 2), (4, 3)]);
        // Restrict map: phones 1,2 -> root 0; phones 3,4 -> root 1.
        let restrict = EventMap::Split {
            key: 0,
            yes_set: vec![1, 2],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        // All four leaves identical, so an unrestricted cluster would merge all.
        let stats: BuildTreeStats = vec![
            leaf_stat(1, 0.0),
            leaf_stat(2, 0.0),
            leaf_stat(3, 0.0),
            leaf_stat(4, 0.0),
        ];
        let (clustered, _) = cluster_event_map_restricted_by_map(&m, &stats, 1.0e10, &restrict);
        let (renumbered, n) = renumber_event_map(&clustered);
        assert_eq!(n, 2, "one leaf per root, no cross-root merging");
        assert_ne!(
            renumbered.map(&[(K_PDF_CLASS, 0), (0, 1)]),
            renumbered.map(&[(K_PDF_CLASS, 0), (0, 3)])
        );
    }

    #[test]
    fn to_n_clusters_hits_the_requested_count() {
        let m = super::super::event_map::table_from_answers(0, &[(1, 0), (2, 1), (3, 2), (4, 3)]);
        let restrict = EventMap::Split {
            key: 0,
            yes_set: vec![1, 2],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        let stats: BuildTreeStats = vec![
            leaf_stat(1, 0.0),
            leaf_stat(2, 0.1),
            leaf_stat(3, 50.0),
            leaf_stat(4, 50.1),
        ];
        let (rounded, num_removed) =
            cluster_event_map_to_n_clusters_restricted_by_map(&m, &stats, 3, &restrict);
        assert_eq!(num_removed, 1);
        let (_, n) = renumber_event_map(&rounded);
        assert_eq!(n, 3);
    }
}
