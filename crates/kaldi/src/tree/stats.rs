//! Port of the statistics half of `plans/kaldi/src/tree/build-tree-utils.{h,cc}`:
//! the `BuildTreeStatsType` helpers that split, filter and sum tree statistics.
//! The tree-growing half lives in [`super::build`].

use super::clusterable::{GaussClusterable, sum_clusterable_objf};
use super::event_map::{EventKey, EventMap, EventType, EventValue, lookup};

/// Kaldi `BuildTreeStatsType`.
pub type BuildTreeStats = Vec<(EventType, GaussClusterable)>;

// ---------------------------------------------------------------------------
// Basic stats manipulation
// ---------------------------------------------------------------------------

/// Kaldi `PossibleValues`: the sorted set of values `key` takes in `stats`.
/// Returns `(all_present, values)` — `all_present` is false if any event
/// lacks the key.
pub fn possible_values(key: EventKey, stats: &BuildTreeStats) -> (bool, Vec<EventValue>) {
    let mut all_present = true;
    let mut values = std::collections::BTreeSet::new();
    for (evec, _) in stats {
        match lookup(evec, key) {
            Some(v) => {
                values.insert(v);
            }
            None => all_present = false,
        }
    }
    (all_present, values.into_iter().collect())
}

/// Kaldi `AllKeysType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllKeysType {
    InsistIdentical,
    Intersection,
    Union,
}

/// Kaldi `FindAllKeys`.
pub fn find_all_keys(stats: &BuildTreeStats, keys_type: AllKeysType) -> Vec<EventKey> {
    let mut iter = stats.iter();
    let mut keys: Vec<EventKey> = match iter.next() {
        None => return Vec::new(),
        Some((e, _)) => e.iter().map(|(k, _)| *k).collect(),
    };
    for (e, _) in iter {
        let keys2: Vec<EventKey> = e.iter().map(|(k, _)| *k).collect();
        match keys_type {
            AllKeysType::InsistIdentical => {
                assert_eq!(keys, keys2, "FindAllKeys: keys in events are not identical");
            }
            AllKeysType::Intersection => {
                keys = keys.iter().copied().filter(|k| keys2.contains(k)).collect();
            }
            AllKeysType::Union => {
                for k in keys2 {
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
                keys.sort_unstable();
            }
        }
    }
    keys
}

/// Kaldi `SplitStatsByMap`: bucket stats by the answer the map gives them.
pub fn split_stats_by_map(stats: &BuildTreeStats, e: &EventMap) -> Vec<BuildTreeStats> {
    let mut size = 0usize;
    for (evec, _) in stats {
        let ans = e.map(evec).unwrap_or_else(|| {
            panic!(
                "SplitStatsByMap: could not map event vector {}; check that \
                 context-width/central-position match the stats, and that \
                 context-independent phones do not share roots with others",
                super::event_map::event_to_string(evec)
            )
        });
        size = size.max(ans as usize + 1);
    }
    let mut out = vec![Vec::new(); size];
    for item in stats {
        let ans = e.map(&item.0).expect("checked above");
        out[ans as usize].push(item.clone());
    }
    out
}

/// Kaldi `SplitStatsByKey`: bucket stats by the value of `key` (which must be
/// present in every event).
pub fn split_stats_by_key(stats: &BuildTreeStats, key: EventKey) -> Vec<BuildTreeStats> {
    let mut size = 0usize;
    for (evec, _) in stats {
        let val = lookup(evec, key).unwrap_or_else(|| {
            panic!(
                "SplitStatsByKey: key {key} not present in event {}",
                super::event_map::event_to_string(evec)
            )
        });
        size = size.max(val as usize + 1);
    }
    let mut out = vec![Vec::new(); size];
    for item in stats {
        let val = lookup(&item.0, key).expect("checked above");
        out[val as usize].push(item.clone());
    }
    out
}

/// Kaldi `FilterStatsByKey`. `values` must be sorted and unique.
pub fn filter_stats_by_key(
    stats: &BuildTreeStats,
    key: EventKey,
    values: &[EventValue],
    include_if_present: bool,
) -> BuildTreeStats {
    debug_assert!(values.windows(2).all(|w| w[0] < w[1]));
    stats
        .iter()
        .filter(|(evec, _)| {
            let val = lookup(evec, key).unwrap_or_else(|| {
                panic!(
                    "FilterStatsByKey: key {key} not present in event {}",
                    super::event_map::event_to_string(evec)
                )
            });
            values.binary_search(&val).is_ok() == include_if_present
        })
        .cloned()
        .collect()
}

/// Kaldi `SumStats`: sum every clusterable in `stats`; `None` if empty.
pub fn sum_stats(stats: &BuildTreeStats) -> Option<GaussClusterable> {
    let mut ans: Option<GaussClusterable> = None;
    for (_, c) in stats {
        match &mut ans {
            None => ans = Some(c.clone()),
            Some(a) => a.add(c),
        }
    }
    ans
}

/// Kaldi `SumNormalizer`.
pub fn sum_normalizer(stats: &BuildTreeStats) -> f64 {
    stats.iter().map(|(_, c)| c.normalizer()).sum()
}

/// Kaldi `SumObjf`.
pub fn sum_objf(stats: &BuildTreeStats) -> f64 {
    stats.iter().map(|(_, c)| c.objf()).sum()
}

/// Kaldi `SumStatsVec`.
pub fn sum_stats_vec(stats: &[BuildTreeStats]) -> Vec<Option<GaussClusterable>> {
    stats.iter().map(sum_stats).collect()
}

/// Kaldi `ObjfGivenMap`: total objective function of the stats under the map.
pub fn objf_given_map(stats: &BuildTreeStats, e: &EventMap) -> f64 {
    let split = split_stats_by_map(stats, e);
    let summed = sum_stats_vec(&split);
    sum_clusterable_objf(&summed)
}

/// Kaldi `ConvertStats`: shift/drop context keys to move stats from an
/// (oldN, oldP) context window to a (newN, newP) one. Returns false if the new
/// window cannot be derived from the old one.
pub fn convert_stats(
    old_n: i32,
    old_p: i32,
    new_n: i32,
    new_p: i32,
    stats: &mut BuildTreeStats,
) -> bool {
    assert!(old_n > 0 && new_n > 0 && old_p >= 0 && new_p >= 0 && new_p < new_n && old_p < old_n);
    if new_n > old_n {
        return false;
    }
    let shift = new_p - old_p; // <= 0 in the supported case
    for (evec, _) in stats.iter_mut() {
        let mut new_evec: EventType = Vec::with_capacity(evec.len());
        for &(key, value) in evec.iter() {
            if key >= 0 && key < old_n {
                let key = key + shift;
                if key >= 0 && key < new_n {
                    new_evec.push((key, value));
                }
            } else {
                new_evec.push((key, value));
            }
        }
        *evec = new_evec;
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::event_map::K_PDF_CLASS;

    fn stat(left: i32, pdf_class: i32) -> (EventType, GaussClusterable) {
        let mut e: EventType = vec![(K_PDF_CLASS, pdf_class), (0, left), (1, 3), (2, 4)];
        e.sort_by_key(|p| p.0);
        let mut c = GaussClusterable::new(1, 0.01);
        c.add_stats(&[left as f32 + pdf_class as f32], 100.0);
        (e, c)
    }

    fn toy() -> BuildTreeStats {
        let mut v = Vec::new();
        for left in [1, 2] {
            for pc in 0..3 {
                v.push(stat(left, pc));
            }
        }
        v
    }

    #[test]
    fn possible_values_and_split_by_key() {
        let stats = toy();
        let (all, vals) = possible_values(0, &stats);
        assert!(all);
        assert_eq!(vals, vec![1, 2]);
        let split = split_stats_by_key(&stats, K_PDF_CLASS);
        assert_eq!(split.len(), 3);
        assert_eq!(split[0].len(), 2);
    }

    #[test]
    fn possible_values_reports_missing_key() {
        let mut stats = toy();
        stats.push((vec![(K_PDF_CLASS, 0)], GaussClusterable::new(1, 0.01)));
        let (all, _) = possible_values(0, &stats);
        assert!(!all);
    }

    #[test]
    fn filter_stats_by_key_excludes_and_includes() {
        let stats = toy();
        let kept = filter_stats_by_key(&stats, 0, &[1], false);
        assert_eq!(kept.len(), 3);
        assert!(kept.iter().all(|(e, _)| lookup(e, 0) == Some(2)));
        let kept = filter_stats_by_key(&stats, K_PDF_CLASS, &[1], true);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn sums_are_additive() {
        let stats = toy();
        let total = sum_stats(&stats).unwrap();
        assert!((total.count - 600.0).abs() < 1e-9);
        assert!((sum_normalizer(&stats) - 600.0).abs() < 1e-9);
        assert_eq!(sum_stats(&Vec::new()).is_none(), true);
    }

    #[test]
    fn find_all_keys_modes() {
        let stats = toy();
        assert_eq!(
            find_all_keys(&stats, AllKeysType::InsistIdentical),
            vec![-1, 0, 1, 2]
        );
        let mut mixed = toy();
        mixed.push((
            vec![(K_PDF_CLASS, 0), (1, 3)],
            GaussClusterable::new(1, 0.01),
        ));
        assert_eq!(
            find_all_keys(&mixed, AllKeysType::Intersection),
            vec![-1, 1]
        );
        assert_eq!(find_all_keys(&mixed, AllKeysType::Union), vec![-1, 0, 1, 2]);
    }

    #[test]
    fn split_stats_by_map_buckets_by_answer() {
        let stats = toy();
        let m = EventMap::Split {
            key: 0,
            yes_set: vec![1],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        let split = split_stats_by_map(&stats, &m);
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].len(), 3);
        assert_eq!(split[1].len(), 3);
        // objf_given_map equals the sum of the two bucket objfs.
        let expect = sum_stats(&split[0]).unwrap().objf() + sum_stats(&split[1]).unwrap().objf();
        assert!((objf_given_map(&stats, &m) - expect).abs() < 1e-9);
    }

    #[test]
    fn convert_stats_shifts_keys() {
        let mut stats = toy();
        assert!(convert_stats(3, 1, 1, 0, &mut stats));
        for (e, _) in &stats {
            assert_eq!(e.len(), 2);
            assert!(lookup(e, 0).is_some());
        }
        // Cannot grow the context window.
        let mut stats2 = toy();
        assert!(!convert_stats(3, 1, 5, 2, &mut stats2));
    }
}
