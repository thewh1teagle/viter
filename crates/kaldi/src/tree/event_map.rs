//! Port of `plans/kaldi/src/tree/event-map.{h,cc}`.
//!
//! An `EventMap` maps an `EventType` (a sorted list of (key, value) pairs) to an
//! answer (a pdf-id / leaf index). Kaldi has three variants: `ConstantEventMap`,
//! `TableEventMap` and `SplitEventMap`; here they are one enum.

use serde::{Deserialize, Serialize};

/// Kaldi `EventKeyType`. -1 is `kPdfClass`; 0..N-1 are context positions.
pub type EventKey = i32;
/// Kaldi `EventValueType` (a phone id, or a pdf-class).
pub type EventValue = i32;
/// Kaldi `EventAnswerType`; -1 means "no answer".
pub type EventAnswer = i32;
/// Sorted-by-key, unique-key list of (key, value) pairs.
pub type EventType = Vec<(EventKey, EventValue)>;

/// Kaldi `kPdfClass` (`tree/context-dep.h`).
pub const K_PDF_CLASS: EventKey = -1;

/// Look up `key` in a sorted event vector. Kaldi `EventMap::Lookup`.
pub fn lookup(event: &[(EventKey, EventValue)], key: EventKey) -> Option<EventValue> {
    // Kaldi does a linear scan for small events and a binary search otherwise;
    // both return the same result on the sorted, unique-key vectors we use.
    match event.binary_search_by(|probe| probe.0.cmp(&key)) {
        Ok(i) => Some(event[i].1),
        Err(_) => None,
    }
}

/// Kaldi `EventMap::Check`: events must be sorted with unique keys.
pub fn check_event(event: &[(EventKey, EventValue)]) -> bool {
    event.windows(2).all(|w| w[0].0 < w[1].0)
}

/// Kaldi `EventTypeToString`, used in error messages.
pub fn event_to_string(event: &[(EventKey, EventValue)]) -> String {
    event
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Kaldi's three `EventMap` subclasses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventMap {
    /// `ConstantEventMap`: always answers with this leaf.
    Constant(crate::types::PdfId),
    /// `TableEventMap`: index by the value of `key`; `None` entries are Kaldi's NULL.
    Table {
        key: EventKey,
        table: Vec<Option<Box<EventMap>>>,
    },
    /// `SplitEventMap`: a decision-tree node. `yes_set` is sorted.
    Split {
        key: EventKey,
        yes_set: Vec<EventValue>,
        yes: Box<EventMap>,
        no: Box<EventMap>,
    },
}

impl EventMap {
    /// Kaldi `Map`. Returns `None` where Kaldi returns `false`.
    pub fn map(&self, event: &[(EventKey, EventValue)]) -> Option<crate::types::PdfId> {
        match self {
            EventMap::Constant(a) => Some(*a),
            EventMap::Table { key, table } => {
                let v = lookup(event, *key)?;
                if v >= 0 && (v as usize) < table.len() {
                    match &table[v as usize] {
                        Some(m) => m.map(event),
                        None => None,
                    }
                } else {
                    None
                }
            }
            EventMap::Split {
                key,
                yes_set,
                yes,
                no,
            } => {
                let v = lookup(event, *key)?;
                if yes_set.binary_search(&v).is_ok() {
                    yes.map(event)
                } else {
                    no.map(event)
                }
            }
        }
    }

    /// Kaldi `MultiMap`: appends every answer reachable given a partially
    /// specified event. Not deduplicated (Kaldi's caller does `SortAndUniq`).
    ///
    /// CONTRACT-DEVIATION: collects `EventAnswer` (i32) rather than `PdfId`,
    /// because callers such as `RenumberEventMap` sort and compare answers and
    /// must be able to see Kaldi's negative sentinels.
    pub fn multi_map(&self, event: &[(EventKey, EventValue)], out: &mut Vec<EventAnswer>) {
        match self {
            EventMap::Constant(a) => out.push(*a as EventAnswer),
            EventMap::Table { key, table } => match lookup(event, *key) {
                Some(v) => {
                    if v >= 0 && (v as usize) < table.len() {
                        if let Some(m) = &table[v as usize] {
                            m.multi_map(event, out);
                        }
                    }
                }
                None => {
                    for e in table.iter().flatten() {
                        e.multi_map(event, out);
                    }
                }
            },
            EventMap::Split {
                key,
                yes_set,
                yes,
                no,
            } => match lookup(event, *key) {
                Some(v) => {
                    if yes_set.binary_search(&v).is_ok() {
                        yes.multi_map(event, out);
                    } else {
                        no.multi_map(event, out);
                    }
                }
                None => {
                    yes.multi_map(event, out);
                    no.multi_map(event, out);
                }
            },
        }
    }

    /// Kaldi `MaxResult()`: the largest answer in the map, or `i32::MIN` if empty.
    pub fn max_result(&self) -> EventAnswer {
        let mut tmp = Vec::new();
        self.multi_map(&[], &mut tmp);
        tmp.into_iter().max().unwrap_or(i32::MIN)
    }

    /// Kaldi `GetChildren`.
    pub fn children(&self) -> Vec<&EventMap> {
        match self {
            EventMap::Constant(_) => Vec::new(),
            EventMap::Table { table, .. } => table.iter().flatten().map(|b| b.as_ref()).collect(),
            EventMap::Split { yes, no, .. } => vec![yes.as_ref(), no.as_ref()],
        }
    }

    /// Kaldi `Copy(new_leaves)`: deep copy, substituting sub-maps at leaves.
    /// A leaf `l` with `new_leaves[l] = Some(m)` is replaced by a copy of `m`.
    pub fn copy_with(&self, new_leaves: &[Option<EventMap>]) -> EventMap {
        match self {
            EventMap::Constant(a) => {
                let i = *a as i64;
                if i >= 0 && (i as usize) < new_leaves.len() {
                    if let Some(m) = &new_leaves[i as usize] {
                        return m.clone();
                    }
                }
                EventMap::Constant(*a)
            }
            EventMap::Table { key, table } => EventMap::Table {
                key: *key,
                table: table
                    .iter()
                    .map(|e| e.as_ref().map(|m| Box::new(m.copy_with(new_leaves))))
                    .collect(),
            },
            EventMap::Split {
                key,
                yes_set,
                yes,
                no,
            } => EventMap::Split {
                key: *key,
                yes_set: yes_set.clone(),
                yes: Box::new(yes.copy_with(new_leaves)),
                no: Box::new(no.copy_with(new_leaves)),
            },
        }
    }

    /// Kaldi `Prune()`: drop branches that answer only -1.
    ///
    /// CONTRACT-DEVIATION: `Constant` holds a `PdfId` (u32) per plans/CONTRACTS.md,
    /// so Kaldi's sentinel leaf -1 is represented as `u32::MAX` here.
    pub fn prune(&self) -> Option<EventMap> {
        match self {
            EventMap::Constant(a) => {
                if *a as i32 == -1 {
                    None
                } else {
                    Some(EventMap::Constant(*a))
                }
            }
            EventMap::Table { key, table } => {
                let mut out: Vec<Option<Box<EventMap>>> = Vec::new();
                for (value, entry) in table.iter().enumerate() {
                    if let Some(m) = entry {
                        if let Some(p) = m.prune() {
                            if out.len() <= value {
                                out.resize_with(value + 1, || None);
                            }
                            out[value] = Some(Box::new(p));
                        }
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(EventMap::Table {
                        key: *key,
                        table: out,
                    })
                }
            }
            EventMap::Split {
                key,
                yes_set,
                yes,
                no,
            } => match (yes.prune(), no.prune()) {
                (None, None) => None,
                (None, Some(n)) => Some(n),
                (Some(y), None) => Some(y),
                (Some(y), Some(n)) => Some(EventMap::Split {
                    key: *key,
                    yes_set: yes_set.clone(),
                    yes: Box::new(y),
                    no: Box::new(n),
                }),
            },
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, EventMap::Constant(_))
    }
}

/// Build a `TableEventMap` from a value->answer map, like Kaldi's
/// `TableEventMap(key, std::map<EventValueType, EventAnswerType>)`.
pub fn table_from_answers(key: EventKey, m: &[(EventValue, EventAnswer)]) -> EventMap {
    let max = m.iter().map(|(v, _)| *v).max().unwrap_or(-1);
    let mut table: Vec<Option<Box<EventMap>>> = vec![None; (max + 1).max(0) as usize];
    for (v, a) in m {
        table[*v as usize] = Some(Box::new(EventMap::Constant(*a as crate::types::PdfId)));
    }
    EventMap::Table { key, table }
}

/// Build a `TableEventMap` from a value->submap list.
pub fn table_from_maps(key: EventKey, m: Vec<(EventValue, EventMap)>) -> EventMap {
    let max = m.iter().map(|(v, _)| *v).max().unwrap_or(-1);
    let mut table: Vec<Option<Box<EventMap>>> = vec![None; (max + 1).max(0) as usize];
    for (v, sub) in m {
        table[v as usize] = Some(Box::new(sub));
    }
    EventMap::Table { key, table }
}

// ---------------------------------------------------------------------------
// GetTreeStructure (event-map.cc:426)
// ---------------------------------------------------------------------------

/// Kaldi `GetTreeStructure`. Returns `(num_leaves, parents)`, where `parents` has
/// one entry per node (leaves numbered first, root last, `parents[i] > i` except
/// at the root). Returns `None` if the map is not a proper tree with uniquely,
/// contiguously numbered leaves.
pub fn get_tree_structure(map: &EventMap) -> Option<(usize, Vec<i32>)> {
    if map.is_leaf() {
        let leaf = map.map(&[])?;
        if leaf != 0 {
            return None;
        }
        return Some((1, vec![0]));
    }

    // Node identity is by pointer in Kaldi; here by address of the borrowed node.
    let mut nonleaf_nodes: Vec<*const EventMap> = Vec::new();
    let mut nonleaf_parents: std::collections::HashMap<*const EventMap, *const EventMap> =
        std::collections::HashMap::new();
    let mut leaf_parents: Vec<Option<*const EventMap>> = Vec::new();

    let top: *const EventMap = map;
    let mut queue: Vec<&EventMap> = vec![map];
    nonleaf_nodes.push(top);
    nonleaf_parents.insert(top, top);

    while let Some(parent) = queue.pop() {
        let children = parent.children();
        if children.is_empty() {
            return None;
        }
        let pp: *const EventMap = parent;
        for child in children {
            if child.is_leaf() {
                let leaf = child.map(&[])? as i32;
                if leaf < 0 {
                    return None;
                }
                let leaf = leaf as usize;
                if leaf_parents.len() <= leaf {
                    leaf_parents.resize(leaf + 1, None);
                }
                if leaf_parents[leaf].is_some() {
                    return None; // repeated leaf
                }
                leaf_parents[leaf] = Some(pp);
            } else {
                let cp: *const EventMap = child;
                nonleaf_nodes.push(cp);
                nonleaf_parents.insert(cp, pp);
                queue.push(child);
            }
        }
    }

    if leaf_parents.iter().any(|p| p.is_none()) {
        return None; // non-consecutively numbered leaves
    }
    if leaf_parents.is_empty() {
        return None;
    }

    let num_leaves = leaf_parents.len();
    let num_nodes = num_leaves + nonleaf_nodes.len();
    let mut nonleaf_indices: std::collections::HashMap<*const EventMap, i32> =
        std::collections::HashMap::new();
    for (i, n) in nonleaf_nodes.iter().enumerate() {
        nonleaf_indices.insert(*n, (num_nodes - i - 1) as i32);
    }

    let mut parents = vec![0i32; num_nodes];
    for (i, p) in leaf_parents.iter().enumerate() {
        parents[i] = *nonleaf_indices.get(&p.unwrap())?;
    }
    for n in nonleaf_nodes.iter() {
        let index = *nonleaf_indices.get(n)?;
        let parent_index = *nonleaf_indices.get(nonleaf_parents.get(n)?)?;
        parents[index as usize] = parent_index;
    }
    for (i, p) in parents.iter().enumerate() {
        let ok = *p > i as i32 || (i + 1 == num_nodes && *p == i as i32);
        if !ok {
            return None;
        }
    }
    Some((num_leaves, parents))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(pairs: &[(EventKey, EventValue)]) -> EventType {
        pairs.to_vec()
    }

    #[test]
    fn constant_maps_everything() {
        let m = EventMap::Constant(7);
        assert_eq!(m.map(&ev(&[(0, 3)])), Some(7));
        assert_eq!(m.max_result(), 7);
    }

    #[test]
    fn table_and_split() {
        let t = table_from_answers(1, &[(2, 0), (5, 1)]);
        assert_eq!(t.map(&ev(&[(1, 2)])), Some(0));
        assert_eq!(t.map(&ev(&[(1, 5)])), Some(1));
        assert_eq!(t.map(&ev(&[(1, 4)])), None);
        assert_eq!(t.map(&ev(&[(0, 4)])), None);

        let s = EventMap::Split {
            key: 0,
            yes_set: vec![1, 3],
            yes: Box::new(EventMap::Constant(10)),
            no: Box::new(EventMap::Constant(11)),
        };
        assert_eq!(s.map(&ev(&[(0, 3)])), Some(10));
        assert_eq!(s.map(&ev(&[(0, 2)])), Some(11));
        let mut all = Vec::new();
        s.multi_map(&[], &mut all);
        all.sort_unstable();
        assert_eq!(all, vec![10, 11]);
    }

    #[test]
    fn copy_with_replaces_leaves() {
        let s = EventMap::Split {
            key: 0,
            yes_set: vec![1],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        let new_leaves = vec![None, Some(EventMap::Constant(42))];
        let c = s.copy_with(&new_leaves);
        assert_eq!(c.map(&ev(&[(0, 1)])), Some(0));
        assert_eq!(c.map(&ev(&[(0, 0)])), Some(42));
    }

    #[test]
    fn tree_structure_of_binary_split() {
        let s = EventMap::Split {
            key: 0,
            yes_set: vec![1],
            yes: Box::new(EventMap::Constant(0)),
            no: Box::new(EventMap::Constant(1)),
        };
        let (n, parents) = get_tree_structure(&s).unwrap();
        assert_eq!(n, 2);
        assert_eq!(parents, vec![2, 2, 2]);
    }

    #[test]
    fn lookup_and_check() {
        assert_eq!(lookup(&[(-1, 1), (0, 5)], -1), Some(1));
        assert_eq!(lookup(&[(-1, 1), (0, 5)], 2), None);
        assert!(check_event(&[(-1, 1), (0, 5)]));
        assert!(!check_event(&[(0, 5), (-1, 1)]));
    }
}
