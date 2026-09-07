//! Port of `plans/kaldi/src/hmm/tree-accu.{h,cc}` (`AccumulateTreeStats`).
//!
//! From an alignment (transition ids) and its feature matrix, accumulate one
//! `GaussClusterable` per (context window, pdf-class) event.

use super::clusterable::GaussClusterable;
use super::event_map::{EventKey, EventType, EventValue, K_PDF_CLASS};
use super::stats::BuildTreeStats;
use crate::hmm::TransitionModel;
use crate::types::{Feats, PhoneId, TransitionId};
use std::collections::HashMap;

/// Kaldi `AccumulateTreeStatsOptions` / `AccumulateTreeStatsInfo`.
#[derive(Clone, Debug)]
pub struct AccumulateTreeStatsOptions {
    /// Variance floor for the clusterable stats (MFA/Kaldi default 0.01).
    pub var_floor: f64,
    /// Context window size, `N` (3 for triphone).
    pub context_width: usize,
    /// Central position in the window, `P` (1 for triphone).
    pub central_position: usize,
    /// Context-independent phones (silence, spn); must be sorted.
    pub ci_phones: Vec<PhoneId>,
}

impl Default for AccumulateTreeStatsOptions {
    fn default() -> Self {
        Self {
            var_floor: 0.01,
            context_width: 3,
            central_position: 1,
            ci_phones: Vec::new(),
        }
    }
}

impl AccumulateTreeStatsOptions {
    fn check(&self) {
        assert!(
            self.context_width > self.central_position,
            "Invalid options: central-position={} context-width={}",
            self.central_position,
            self.context_width
        );
        assert!(
            self.ci_phones.windows(2).all(|w| w[0] < w[1]),
            "ci_phones must be sorted and unique"
        );
    }
}

/// Kaldi `SplitToPhones` restricted to what tree-accu needs: split the
/// transition-id sequence into one run per phone instance. A new run starts
/// whenever the current tid is not a continuation of the previous phone —
/// detected as in `hmm-utils.cc:723` by the hmm-state going backwards or the
/// phone changing, with self-loops continuing the current state.
fn split_to_phones(tm: &TransitionModel, tids: &[TransitionId]) -> Option<Vec<Vec<TransitionId>>> {
    if tids.is_empty() {
        return Some(Vec::new());
    }
    let mut out: Vec<Vec<TransitionId>> = Vec::new();
    let mut cur: Vec<TransitionId> = Vec::new();
    let mut cur_phone: Option<PhoneId> = None;
    let mut cur_state: usize = 0;
    for &t in tids {
        if t == 0 {
            return None;
        }
        let phone = tm.transition_id_to_phone(t);
        let state = tm.transition_id_to_hmm_state(t);
        let self_loop = tm.is_self_loop(t);
        let starts_new = match cur_phone {
            None => true,
            Some(cp) => {
                cp != phone
                    || (!self_loop && state <= cur_state)
                    || (self_loop && state != cur_state)
            }
        };
        if starts_new {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur_phone = Some(phone);
        }
        cur_state = state;
        cur.push(t);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Some(out)
}

/// The pdf-class of a transition-id: the `pdf_class` of the HMM state of its
/// phone in the topology (Kaldi `TransitionModel::TransitionIdToPdfClass`).
fn transition_id_to_pdf_class(tm: &TransitionModel, t: TransitionId) -> i32 {
    let phone = tm.transition_id_to_phone(t);
    let state = tm.transition_id_to_hmm_state(t);
    tm.topology().topology_for_phone(phone)[state].pdf_class
}

/// Kaldi `AccumulateTreeStats`: add this utterance's stats into `stats`.
///
/// For every context window position whose central slot is a real phone, an
/// event is built from the phones in the window (key `j` = window position
/// `0..N-1`) plus `kPdfClass`. If the central phone is context-independent, the
/// non-central keys are *omitted* rather than zeroed, exactly as Kaldi does, so
/// that no question can ever be asked about them.
pub fn accumulate_tree_stats(
    opts: &AccumulateTreeStatsOptions,
    tm: &TransitionModel,
    tids: &[TransitionId],
    feats: &Feats,
    stats: &mut HashMap<EventType, GaussClusterable>,
) {
    opts.check();
    // Kaldi's real SplitToPhones (hmm-utils.cc:672-700), not a state-order
    // heuristic: MFA's 5-state silence has legal backward arcs (4 -> 2, 3 -> 1)
    // that a "state went backwards" rule would split into spurious phone instances.
    let (split_alignment, was_ok) = crate::hmm::split_to_phones_checked(tm, tids);
    if !was_ok {
        return; // bad alignment: Kaldi warns and skips (tree-accu.cc:42-45)
    }
    assert_eq!(
        feats.nrows(),
        tids.len(),
        "AccumulateTreeStats: feature/alignment length mismatch"
    );
    let dim = feats.ncols();
    let n = opts.context_width as i64;
    let p = opts.central_position as i64;
    let num_phones = split_alignment.len() as i64;

    let mut cur_pos = 0usize;
    let mut i = -n;
    while i < num_phones {
        let central = i + p;
        if central >= 0 && central < num_phones {
            let central_phone = tm.transition_id_to_phone(split_alignment[central as usize][0]);
            let is_ctx_dep = opts.ci_phones.binary_search(&central_phone).is_err();
            let mut evec: EventType = Vec::with_capacity(opts.context_width + 1);
            for j in 0..n {
                let phone: PhoneId = if i + j >= 0 && i + j < num_phones {
                    tm.transition_id_to_phone(split_alignment[(i + j) as usize][0])
                } else {
                    0 // out of window; ContextDependency uses 0 for this
                };
                if is_ctx_dep || j == p {
                    evec.push((j as EventKey, phone as EventValue));
                }
            }
            for &t in &split_alignment[central as usize] {
                let mut evec_more = evec.clone();
                evec_more.push((K_PDF_CLASS, transition_id_to_pdf_class(tm, t)));
                evec_more.sort_by_key(|pair| pair.0);
                let entry = stats
                    .entry(evec_more)
                    .or_insert_with(|| GaussClusterable::new(dim, opts.var_floor));
                let row = feats.row(cur_pos);
                let row: Vec<f32> = row.to_vec();
                entry.add_stats(&row, 1.0);
                cur_pos += 1;
            }
        }
        i += 1;
    }
    assert_eq!(
        cur_pos,
        tids.len(),
        "AccumulateTreeStats: consumed {cur_pos} frames of {}",
        tids.len()
    );
}

/// Merge one utterance/job's stats map into another (Kaldi's `sum-tree-stats`).
pub fn merge_tree_stats(
    into: &mut HashMap<EventType, GaussClusterable>,
    from: HashMap<EventType, GaussClusterable>,
) {
    for (k, v) in from {
        match into.get_mut(&k) {
            Some(existing) => existing.add(&v),
            None => {
                into.insert(k, v);
            }
        }
    }
}

/// Convert the accumulated map into the vector form `build_tree` takes,
/// ordered by event (Kaldi's `CopyMapToVector` over a `std::map`, i.e. sorted).
pub fn stats_map_to_vec(stats: HashMap<EventType, GaussClusterable>) -> BuildTreeStats {
    let mut v: BuildTreeStats = stats.into_iter().collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_adds_counts() {
        let mut a: HashMap<EventType, GaussClusterable> = HashMap::new();
        let e: EventType = vec![(K_PDF_CLASS, 1), (1, 5)];
        let mut c = GaussClusterable::new(2, 0.01);
        c.add_stats(&[1.0, 2.0], 1.0);
        a.insert(e.clone(), c.clone());
        let mut b: HashMap<EventType, GaussClusterable> = HashMap::new();
        b.insert(e.clone(), c);
        merge_tree_stats(&mut a, b);
        assert!((a[&e].count - 2.0).abs() < 1e-12);
        assert!((a[&e].stats[[0, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn stats_map_to_vec_is_sorted() {
        let mut m: HashMap<EventType, GaussClusterable> = HashMap::new();
        m.insert(
            vec![(K_PDF_CLASS, 1), (1, 9)],
            GaussClusterable::new(1, 0.01),
        );
        m.insert(
            vec![(K_PDF_CLASS, 0), (1, 9)],
            GaussClusterable::new(1, 0.01),
        );
        m.insert(
            vec![(K_PDF_CLASS, 0), (1, 3)],
            GaussClusterable::new(1, 0.01),
        );
        let v = stats_map_to_vec(m);
        let keys: Vec<EventType> = v.into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec![
                vec![(K_PDF_CLASS, 0), (1, 3)],
                vec![(K_PDF_CLASS, 0), (1, 9)],
                vec![(K_PDF_CLASS, 1), (1, 9)],
            ]
        );
    }
}
