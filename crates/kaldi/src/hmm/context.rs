//! Port of `plans/kaldi/src/tree/context-dep.{h,cc}`: `ContextDependency::Compute`,
//! `GetPdfInfo`, and `MonophoneContextDependencyShared` (via `GetStubMap`,
//! `plans/kaldi/src/tree/build-tree-utils.cc`).

use crate::tree::{EventMap, EventType, EventValue};
use crate::types::{PdfId, PhoneId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Kaldi's `kPdfClass` event key.
pub const PDF_CLASS_KEY: i32 = -1;

/// Maps a phone context window plus a pdf-class to a pdf-id.
///
/// Monophone systems use `n = 1, p = 0`; triphone systems `n = 3, p = 1`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextDependency {
    /// Context width (Kaldi `N_`).
    pub n: usize,
    /// Central position within the window (Kaldi `P_`).
    pub p: usize,
    pub map: EventMap,
}

impl ContextDependency {
    pub fn new(n: usize, p: usize, map: EventMap) -> Self {
        assert!(p < n, "central position must lie inside the context window");
        Self { n, p, map }
    }

    /// Kaldi `MonophoneContextDependencyShared` (context-dep.cc): `N = 1`, `P = 0`, roots not
    /// shared, so `GetStubMap` builds a table over the phone at position 0 whose leaves are
    /// tables over pdf-class.
    pub fn monophone_shared(
        phone_sets: &[Vec<PhoneId>],
        num_pdf_classes: &dyn Fn(PhoneId) -> usize,
    ) -> Self {
        let share_roots = vec![false; phone_sets.len()];
        let mut num_leaves = 0;
        let map = get_stub_map(
            0,
            phone_sets,
            num_pdf_classes,
            &share_roots,
            &mut num_leaves,
        );
        Self::new(1, 0, map)
    }

    /// Kaldi `ContextDependency::Compute`. `window` must have length `n`; entries are phone ids
    /// with 0 meaning "outside the utterance".
    pub fn compute(&self, window: &[PhoneId], pdf_class: i32) -> Option<PdfId> {
        assert_eq!(window.len(), self.n, "context window has the wrong width");
        let mut event: EventType = Vec::with_capacity(self.n + 1);
        // kPdfClass is -1, which sorts before every context position, so pushing it first keeps
        // the event vector sorted by key, as EventMap::map requires.
        event.push((PDF_CLASS_KEY, pdf_class));
        for (i, &phone) in window.iter().enumerate() {
            event.push((i as i32, phone as EventValue));
        }
        self.map.map(&event)
    }

    pub fn num_pdfs(&self) -> usize {
        (self.map.max_result() + 1).max(0) as usize
    }

    pub fn context_width(&self) -> usize {
        self.n
    }

    pub fn central_position(&self) -> usize {
        self.p
    }

    /// Kaldi `ContextDependency::GetPdfInfo` (the `num_pdf_classes` overload, context-dep.cc).
    ///
    /// For each pdf-id, the sorted, unique list of `(phone, pdf_class)` pairs that can produce it.
    /// The result is indexed by pdf-id and has length `num_pdfs()`.
    pub fn get_pdf_info(
        &self,
        phones: &[PhoneId],
        num_pdf_classes: &dyn Fn(PhoneId) -> usize,
    ) -> Vec<Vec<(PhoneId, i32)>> {
        let mut out: Vec<Vec<(PhoneId, i32)>> = vec![Vec::new(); self.num_pdfs()];
        for &phone in phones {
            let len = num_pdf_classes(phone) as i32;
            for pos in 0..len {
                // Kaldi builds a 2-element event with only the central position and the
                // pdf-class set, then sorts it; MultiMap explores every branch whose key is
                // absent from the event.
                let mut event: EventType =
                    vec![(self.p as i32, phone as EventValue), (PDF_CLASS_KEY, pos)];
                event.sort_unstable();
                let mut pdfs = Vec::new();
                self.map.multi_map(&event, &mut pdfs);
                pdfs.sort_unstable();
                pdfs.dedup();
                if pdfs.is_empty() {
                    tracing::warn!(
                        phone,
                        pdf_class = pos,
                        "GetPdfInfo: no pdfs for this (phone, pdf-class)"
                    );
                }
                for pdf in pdfs {
                    if let Some(slot) = out.get_mut(pdf as usize) {
                        slot.push((phone, pos));
                    }
                }
            }
        }
        for v in &mut out {
            v.sort_unstable();
            v.dedup();
        }
        out
    }
}

/// Kaldi `GetStubMap` (build-tree-utils.cc). Builds the initial, unsplit tree: one leaf per
/// phone-set when roots are shared, or one leaf per (phone-set, pdf-class) when they are not.
///
/// `num_leaves` is threaded through exactly as Kaldi's `num_leaves_out`, so leaf numbering — and
/// therefore pdf numbering — matches.
pub fn get_stub_map(
    p: usize,
    phone_sets: &[Vec<PhoneId>],
    num_pdf_classes: &dyn Fn(PhoneId) -> usize,
    share_roots: &[bool],
    num_leaves: &mut PdfId,
) -> EventMap {
    assert!(!phone_sets.is_empty());
    assert_eq!(phone_sets.len(), share_roots.len());

    let max_set_size = phone_sets.iter().map(|s| s.len()).max().unwrap_or(0);
    let highest_phone = phone_sets
        .iter()
        .flat_map(|s| s.iter().copied())
        .max()
        .unwrap_or(0);

    if phone_sets.len() == 1 {
        if share_roots[0] {
            let leaf = *num_leaves;
            *num_leaves += 1;
            return EventMap::Constant(leaf);
        }
        // Not sharing roots: split on pdf-class, one leaf per class.
        let mut max_len = 0usize;
        for (i, &phone) in phone_sets[0].iter().enumerate() {
            let len = num_pdf_classes(phone);
            assert!(len > 0, "phone {phone} has no pdf classes");
            if i == 0 {
                max_len = len;
            } else if len != max_len {
                tracing::warn!(
                    len,
                    max_len,
                    "mismatching HMM lengths within a phone set (unusual but not fatal)"
                );
                max_len = max_len.max(len);
            }
        }
        let mut table: Vec<Option<Box<EventMap>>> = Vec::with_capacity(max_len);
        for _ in 0..max_len {
            let leaf = *num_leaves;
            *num_leaves += 1;
            table.push(Some(Box::new(EventMap::Constant(leaf))));
        }
        return EventMap::Table {
            key: PDF_CLASS_KEY,
            table,
        };
    }

    if max_set_size == 1 && phone_sets.len() <= 2 * highest_phone as usize {
        // Every set is a single phone and the table would not be too sparse: index on phone.
        let mut by_phone: BTreeMap<PhoneId, EventMap> = BTreeMap::new();
        for (i, set) in phone_sets.iter().enumerate() {
            let sub = get_stub_map(
                p,
                std::slice::from_ref(set),
                num_pdf_classes,
                &share_roots[i..i + 1],
                num_leaves,
            );
            let phone = set[0];
            assert!(
                by_phone.insert(phone, sub).is_none(),
                "phone {phone} appears in more than one phone set"
            );
        }
        let max_phone = by_phone.keys().copied().max().unwrap_or(0) as usize;
        let mut table: Vec<Option<Box<EventMap>>> = (0..=max_phone).map(|_| None).collect();
        for (phone, sub) in by_phone {
            table[phone as usize] = Some(Box::new(sub));
        }
        return EventMap::Table {
            key: p as i32,
            table,
        };
    }

    // Otherwise split the list of phone sets in half and recurse. Note Kaldi builds the first
    // half's map before the second's, which fixes leaf numbering.
    let half = phone_sets.len() / 2;
    let map1 = get_stub_map(
        p,
        &phone_sets[..half],
        num_pdf_classes,
        &share_roots[..half],
        num_leaves,
    );
    let map2 = get_stub_map(
        p,
        &phone_sets[half..],
        num_pdf_classes,
        &share_roots[half..],
        num_leaves,
    );
    let mut yes_set: Vec<EventValue> = phone_sets[..half]
        .iter()
        .flat_map(|s| s.iter().map(|&x| x as EventValue))
        .collect();
    yes_set.sort_unstable();
    debug_assert!(
        yes_set.windows(2).all(|w| w[0] < w[1]),
        "phone sets must be disjoint"
    );
    EventMap::Split {
        key: p as i32,
        yes_set,
        yes: Box::new(map1),
        no: Box::new(map2),
    }
}

/// All phones mentioned anywhere in a list of phone sets, sorted and unique.
pub fn all_phones(phone_sets: &[Vec<PhoneId>]) -> Vec<PhoneId> {
    let set: BTreeSet<PhoneId> = phone_sets.iter().flat_map(|s| s.iter().copied()).collect();
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two silence phones sharing nothing, three normal phones: the monophone layout MFA uses.
    fn mono() -> ContextDependency {
        let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3], vec![4], vec![5]];
        let npdf = |p: PhoneId| if p <= 2 { 5 } else { 3 };
        ContextDependency::monophone_shared(&sets, &npdf)
    }

    #[test]
    fn monophone_pdf_count_is_sum_of_pdf_classes() {
        let ctx = mono();
        // 2 silence phones * 5 + 3 phones * 3 = 19.
        assert_eq!(ctx.num_pdfs(), 19);
        assert_eq!(ctx.n, 1);
        assert_eq!(ctx.p, 0);
    }

    #[test]
    fn monophone_pdfs_are_contiguous_per_phone() {
        let ctx = mono();
        // Leaves are numbered phone by phone, class by class, in phone-set order.
        assert_eq!(ctx.compute(&[1], 0), Some(0));
        assert_eq!(ctx.compute(&[1], 4), Some(4));
        assert_eq!(ctx.compute(&[2], 0), Some(5));
        assert_eq!(ctx.compute(&[3], 0), Some(10));
        assert_eq!(ctx.compute(&[3], 2), Some(12));
        assert_eq!(ctx.compute(&[5], 2), Some(18));
    }

    #[test]
    fn compute_out_of_range_class_is_none() {
        let ctx = mono();
        // Phone 3 has only 3 pdf classes.
        assert_eq!(ctx.compute(&[3], 4), None);
    }

    #[test]
    fn shared_root_gives_one_leaf_per_set() {
        let sets: Vec<Vec<PhoneId>> = vec![vec![1, 2]];
        let mut n = 0;
        let map = get_stub_map(0, &sets, &|_| 3, &[true], &mut n);
        assert_eq!(n, 1);
        assert!(matches!(map, EventMap::Constant(0)));
    }

    #[test]
    fn get_pdf_info_inverts_compute() {
        let ctx = mono();
        let phones: Vec<PhoneId> = vec![1, 2, 3, 4, 5];
        let npdf = |p: PhoneId| if p <= 2 { 5 } else { 3 };
        let info = ctx.get_pdf_info(&phones, &npdf);
        assert_eq!(info.len(), 19);
        // Monophone: every pdf belongs to exactly one (phone, class).
        for (pdf, pairs) in info.iter().enumerate() {
            assert_eq!(pairs.len(), 1, "pdf {pdf} should map to one pair");
            let (phone, class) = pairs[0];
            assert_eq!(ctx.compute(&[phone], class), Some(pdf as PdfId));
        }
    }

    #[test]
    fn all_phones_sorted_unique() {
        assert_eq!(all_phones(&[vec![3, 1], vec![2, 1]]), vec![1, 2, 3]);
    }
}
