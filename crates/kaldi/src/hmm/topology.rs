//! Port of `plans/kaldi/src/hmm/hmm-topology.{h,cc}`.
//!
//! Values for the MFA default topology come from
//! `plans/mfa/montreal_forced_aligner/dictionary/mixins.py:669` (`_write_topo`).

use crate::types::PhoneId;
use serde::{Deserialize, Serialize};

/// Kaldi's `kNoPdf`: a non-emitting HMM state.
pub const NO_PDF: i32 = -1;

/// One state of a phone HMM.
///
/// Kaldi distinguishes `forward_pdf_class` from `self_loop_pdf_class`; those differ only for
/// chain models, which viter does not build. `pdf_class` here is both.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HmmState {
    /// -1 (`NO_PDF`) means non-emitting.
    pub pdf_class: i32,
    /// `(dest state, probability)`, in the order they appear in the topology. This order defines
    /// the transition-index, and hence the transition-id, so it must not be permuted.
    pub transitions: Vec<(usize, f32)>,
}

impl HmmState {
    pub fn new(pdf_class: i32, transitions: Vec<(usize, f32)>) -> Self {
        Self {
            pdf_class,
            transitions,
        }
    }
    pub fn is_emitting(&self) -> bool {
        self.pdf_class != NO_PDF
    }
    /// Transition index of the self-loop out of `self_index`, if any.
    pub fn self_loop_index(&self, self_index: usize) -> Option<usize> {
        self.transitions
            .iter()
            .position(|&(dest, _)| dest == self_index)
    }
}

/// One `<TopologyEntry>`: a group of phones sharing an HMM prototype.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TopologyEntry {
    /// Phones covered by this entry, sorted and unique.
    pub phones: Vec<PhoneId>,
    /// The prototype states. The last one is the final, non-emitting state.
    pub states: Vec<HmmState>,
}

/// The set of HMM prototypes for all phones.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HmmTopology {
    pub entries: Vec<TopologyEntry>,
}

impl HmmTopology {
    /// MFA's default topology (`_write_topo`, mixins.py:669-720).
    ///
    /// Silence (`num_sil_states`, MFA default 5): state 0 fans out to states `0..N-2` with uniform
    /// probability `1/(N-1)`; states `1..N-2` fan out to `1..N-1` with the same uniform
    /// probability; state `N-1` self-loops with 0.75 and exits to the final state `N` with 0.25.
    ///
    /// Non-silence (`num_nonsil_states`, MFA default 3): a Bakis left-to-right chain, self-loop
    /// 0.5 / forward 0.5, with the last emitting state going forward to the final state.
    pub fn mfa_default(
        silence_phones: &[PhoneId],
        other_phones: &[PhoneId],
        num_sil_states: usize,
        num_nonsil_states: usize,
    ) -> Self {
        let mut entries = Vec::new();
        if !silence_phones.is_empty() {
            entries.push(TopologyEntry {
                phones: sorted_uniq(silence_phones),
                states: silence_states(num_sil_states),
            });
        }
        if !other_phones.is_empty() {
            entries.push(TopologyEntry {
                phones: sorted_uniq(other_phones),
                states: bakis_states(num_nonsil_states),
            });
        }
        Self { entries }
    }

    /// Panics if `p` is not covered by any entry, matching Kaldi's `TopologyForPhone`, which
    /// raises there too. Callers that may see uncovered phones should use [`Self::try_for_phone`].
    pub fn topology_for_phone(&self, p: PhoneId) -> &[HmmState] {
        self.try_for_phone(p)
            .unwrap_or_else(|| panic!("TopologyForPhone(): phone {p} not covered"))
    }

    pub fn try_for_phone(&self, p: PhoneId) -> Option<&[HmmState]> {
        self.entries
            .iter()
            .find(|e| e.phones.binary_search(&p).is_ok())
            .map(|e| e.states.as_slice())
    }

    /// `max pdf_class + 1` over the entry's states (hmm-topology.cc:339).
    pub fn num_pdf_classes(&self, p: PhoneId) -> usize {
        let entry = self.topology_for_phone(p);
        let max = entry.iter().map(|s| s.pdf_class).fold(0, i32::max);
        (max + 1) as usize
    }

    /// All covered phones, sorted and unique. Kaldi keeps `phones_` sorted, and iteration order
    /// over it feeds `ComputeTuples`, so the sort matters for transition-id parity.
    pub fn phones(&self) -> Vec<PhoneId> {
        let mut v: Vec<PhoneId> = self
            .entries
            .iter()
            .flat_map(|e| e.phones.iter().copied())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Minimum number of frames a phone can occupy (hmm-topology.cc:350). Shortest-path over the
    /// prototype counting emitting states.
    pub fn min_length(&self, p: PhoneId) -> usize {
        let entry = self.topology_for_phone(p);
        let n = entry.len();
        let mut min_length = vec![usize::MAX; n];
        min_length[0] = if entry[0].is_emitting() { 1 } else { 0 };
        let mut changed = true;
        while changed {
            changed = false;
            for s in 0..n {
                if min_length[s] == usize::MAX {
                    continue;
                }
                for &(next, _) in &entry[s].transitions {
                    // Cost of entering `next` is 1 if `next` is emitting; a self-loop never
                    // shortens anything so the `next != s` case is the only one that can improve.
                    let cand = min_length[s] + usize::from(entry[next].is_emitting());
                    if next != s && cand < min_length[next] {
                        min_length[next] = cand;
                        changed = true;
                    }
                }
            }
        }
        min_length[n - 1]
    }

    /// Per-phone pdf-class count, indexed by phone id; uncovered phones get 0.
    pub fn phone_to_num_pdf_classes(&self) -> Vec<usize> {
        let phones = self.phones();
        let max = phones.last().copied().unwrap_or(0) as usize;
        let mut v = vec![0usize; max + 1];
        for p in phones {
            v[p as usize] = self.num_pdf_classes(p);
        }
        v
    }

    /// Kaldi `HmmTopology::Check` in the parts that matter to us: every entry has at least two
    /// states, the last state is non-emitting with no transitions out, pdf-classes are contiguous
    /// from zero, and every emitting state's transitions sum to one.
    pub fn check(&self) -> Result<(), String> {
        for entry in &self.entries {
            let n = entry.states.len();
            if n < 2 {
                return Err("topology entry needs at least one emitting + one final state".into());
            }
            let last = &entry.states[n - 1];
            if last.is_emitting() || !last.transitions.is_empty() {
                return Err("last state of a topology entry must be non-emitting and final".into());
            }
            let mut seen = vec![false; n];
            for (i, st) in entry.states.iter().enumerate().take(n - 1) {
                if !st.is_emitting() {
                    continue;
                }
                let c = st.pdf_class as usize;
                if c >= n {
                    return Err(format!("pdf-class {c} out of range"));
                }
                seen[c] = true;
                if st.transitions.is_empty() {
                    return Err(format!("emitting state {i} has no transitions"));
                }
                let sum: f32 = st.transitions.iter().map(|t| t.1).sum();
                if (sum - 1.0).abs() > 1e-3 {
                    return Err(format!("state {i} transitions sum to {sum}, not 1"));
                }
                for &(dest, prob) in &st.transitions {
                    if dest >= n {
                        return Err(format!("transition to out-of-range state {dest}"));
                    }
                    if prob <= 0.0 {
                        return Err("zero or negative transition probability".into());
                    }
                }
            }
            let num_classes = seen.iter().take_while(|b| **b).count();
            if seen[num_classes..].iter().any(|b| *b) {
                return Err("pdf-classes are not contiguous from zero".into());
            }
        }
        Ok(())
    }
}

fn sorted_uniq(v: &[PhoneId]) -> Vec<PhoneId> {
    let mut v = v.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// MFA silence prototype, `mixins.py:679-702`.
fn silence_states(n: usize) -> Vec<HmmState> {
    assert!(n >= 2, "silence topology needs >= 2 states");
    let transp = 1.0f32 / (n - 1) as f32;
    let mut states = Vec::with_capacity(n + 1);
    for i in 0..n {
        let transitions: Vec<(usize, f32)> = if i == 0 {
            // `for x in range(num_silence_states - 1)` -> 0 .. n-2
            (0..n - 1).map(|x| (x, transp)).collect()
        } else if i < n - 1 {
            // `for x in range(1, num_silence_states)` -> 1 .. n-1
            (1..n).map(|x| (x, transp)).collect()
        } else {
            vec![(i, 0.75), (n, 0.25)]
        };
        states.push(HmmState::new(i as i32, transitions));
    }
    // `<State> {num_silence_states} </State>` -- the final, non-emitting state.
    states.push(HmmState::new(NO_PDF, Vec::new()));
    states
}

/// MFA non-silence prototype (`dictionary/mixins.py:729-749`): Bakis left-to-right,
/// self 0.5 / forward 0.5 on every state except the last emitting one, which has
/// no self-loop and exits with probability 1.0 (it lasts exactly one frame).
fn bakis_states(n: usize) -> Vec<HmmState> {
    assert!(n >= 1, "non-silence topology needs >= 1 emitting state");
    let mut states = Vec::with_capacity(n + 1);
    for i in 0..n {
        if i + 1 == n {
            states.push(HmmState::new(i as i32, vec![(i + 1, 1.0)]));
        } else {
            states.push(HmmState::new(i as i32, vec![(i, 0.5), (i + 1, 0.5)]));
        }
    }
    states.push(HmmState::new(NO_PDF, Vec::new()));
    states
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topo() -> HmmTopology {
        HmmTopology::mfa_default(&[1, 2], &[3, 4, 5], 5, 3)
    }

    #[test]
    fn mfa_default_shapes() {
        let t = topo();
        assert_eq!(t.entries.len(), 2);
        // 5 emitting + 1 final.
        assert_eq!(t.topology_for_phone(1).len(), 6);
        // 3 emitting + 1 final.
        assert_eq!(t.topology_for_phone(3).len(), 4);
        assert_eq!(t.num_pdf_classes(1), 5);
        assert_eq!(t.num_pdf_classes(3), 3);
        assert_eq!(t.phones(), vec![1, 2, 3, 4, 5]);
        t.check().expect("mfa default topology must be valid");
    }

    #[test]
    fn silence_transitions_match_mfa() {
        let t = topo();
        let sil = t.topology_for_phone(1);
        let quarter = 0.25f32;
        // State 0 -> 0..3, uniform 1/4.
        assert_eq!(
            sil[0].transitions,
            vec![(0, quarter), (1, quarter), (2, quarter), (3, quarter)]
        );
        // State 1 -> 1..4, uniform 1/4.
        assert_eq!(
            sil[1].transitions,
            vec![(1, quarter), (2, quarter), (3, quarter), (4, quarter)]
        );
        // Last emitting state: self 0.75, exit 0.25 to the final state 5.
        assert_eq!(sil[4].transitions, vec![(4, 0.75), (5, 0.25)]);
        assert!(!sil[5].is_emitting());
        assert!(sil[5].transitions.is_empty());
    }

    #[test]
    fn bakis_transitions() {
        let t = topo();
        let p = t.topology_for_phone(4);
        assert_eq!(p[0].transitions, vec![(0, 0.5), (1, 0.5)]);
        assert_eq!(p[1].transitions, vec![(1, 0.5), (2, 0.5)]);
        // Last emitting state: no self-loop, exits with probability 1 (mixins.py:743-746).
        assert_eq!(p[2].transitions, vec![(3, 1.0)]);
        assert_eq!(p[0].self_loop_index(0), Some(0));
        assert_eq!(p[2].self_loop_index(2), None);
    }

    #[test]
    fn min_lengths() {
        let t = topo();
        // Bakis: must pass through all 3 emitting states.
        assert_eq!(t.min_length(3), 3);
        // Silence: 0 -> 3 -> 4 -> final, so 3 emitting frames minimum.
        assert_eq!(t.min_length(1), 3);
    }

    #[test]
    fn phone_to_num_pdf_classes_indexed_by_phone() {
        let t = topo();
        let v = t.phone_to_num_pdf_classes();
        assert_eq!(v[1], 5);
        assert_eq!(v[3], 3);
        assert_eq!(v[0], 0);
    }

    #[test]
    fn uncovered_phone_is_none() {
        let t = topo();
        assert!(t.try_for_phone(99).is_none());
    }
}
