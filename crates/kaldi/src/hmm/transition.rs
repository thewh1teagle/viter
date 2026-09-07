//! Port of `plans/kaldi/src/hmm/transition-model.{h,cc}`.
//!
//! The tuple ordering established by [`TransitionModel::new`] defines every transition id, so it
//! must match Kaldi exactly: tuples are gathered per pdf (via `GetPdfInfo`) and then sorted
//! lexicographically by `(phone, hmm_state, forward_pdf, self_loop_pdf)`.

use super::context::ContextDependency;
use super::topology::HmmTopology;
use crate::types::{PdfId, PhoneId, TransitionId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A `(phone, hmm_state, forward_pdf, self_loop_pdf)` tuple: Kaldi's transition-state, minus one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Tuple {
    pub phone: PhoneId,
    pub hmm_state: usize,
    pub forward_pdf: PdfId,
    pub self_loop_pdf: PdfId,
}

/// Transition statistics, indexed by transition id (element 0 unused, like Kaldi's `Vector`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TransitionAccs(pub Vec<f64>);

impl TransitionAccs {
    pub fn new(num_transition_ids: usize) -> Self {
        Self(vec![0.0; num_transition_ids + 1])
    }
    pub fn add(&mut self, other: &Self) {
        assert_eq!(self.0.len(), other.0.len(), "mismatched accumulator sizes");
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a += *b;
        }
    }
    pub fn total(&self) -> f64 {
        self.0.iter().sum()
    }
}

/// Kaldi `MleTransitionUpdateConfig`.
#[derive(Clone, Copy, Debug)]
pub struct MleTransitionUpdateConfig {
    pub floor: f32,
    pub mincount: f32,
}

impl Default for MleTransitionUpdateConfig {
    fn default() -> Self {
        Self {
            floor: 0.01,
            mincount: 5.0,
        }
    }
}

/// Kaldi `TransitionModel`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionModel {
    topo: HmmTopology,
    /// Sorted; indexed by `transition_state - 1`.
    tuples: Vec<Tuple>,
    /// `state2id[tstate]` is the first transition id of that transition-state; length
    /// `tuples.len() + 2`.
    state2id: Vec<TransitionId>,
    /// `id2state[tid]` is the transition-state owning `tid`; length `num_transition_ids + 1`.
    id2state: Vec<u32>,
    /// `id2pdf[tid]`, precomputed like Kaldi's `id2pdf_id_`.
    id2pdf: Vec<PdfId>,
    /// Natural log of each transition probability, indexed by transition id.
    log_probs: Vec<f32>,
    /// `log(1 - self_loop_prob)` per transition-state; index 0 unused.
    non_self_loop_log_probs: Vec<f32>,
    num_pdfs: usize,
}

impl TransitionModel {
    /// Kaldi `TransitionModel::TransitionModel` (transition-model.cc:245): `ComputeTuples`,
    /// `ComputeDerived`, `InitializeProbs`, `Check`.
    pub fn new(ctx: &ContextDependency, topo: &HmmTopology) -> Self {
        let mut tm = Self {
            topo: topo.clone(),
            tuples: Vec::new(),
            state2id: Vec::new(),
            id2state: Vec::new(),
            id2pdf: Vec::new(),
            log_probs: Vec::new(),
            non_self_loop_log_probs: Vec::new(),
            num_pdfs: 0,
        };
        tm.compute_tuples(ctx);
        tm.compute_derived();
        tm.initialize_probs();
        tm.check();
        tm
    }

    /// Kaldi `ComputeTuplesIsHmm`. viter only builds HMM (non-chain) models, where the forward
    /// and self-loop pdf-classes of every state coincide, so this is the only variant needed.
    fn compute_tuples(&mut self, ctx: &ContextDependency) {
        let phones = self.topo.phones();
        assert!(!phones.is_empty(), "topology covers no phones");
        let topo = self.topo.clone();
        let num_pdf_classes = |p: PhoneId| topo.num_pdf_classes(p);
        let pdf_info = ctx.get_pdf_info(&phones, &num_pdf_classes);

        // (phone, pdf_class) -> hmm-states of that phone emitting that class, in state order.
        let mut to_hmm_state_list: BTreeMap<(PhoneId, i32), Vec<usize>> = BTreeMap::new();
        for &phone in &phones {
            for (j, state) in self.topo.topology_for_phone(phone).iter().enumerate() {
                if state.is_emitting() {
                    to_hmm_state_list
                        .entry((phone, state.pdf_class))
                        .or_default()
                        .push(j);
                }
            }
        }

        for (pdf, pairs) in pdf_info.iter().enumerate() {
            let pdf = pdf as PdfId;
            for &(phone, pdf_class) in pairs {
                let state_vec = to_hmm_state_list
                    .get(&(phone, pdf_class))
                    .unwrap_or_else(|| {
                        panic!("no hmm-state for phone {phone} pdf-class {pdf_class}")
                    });
                for &hmm_state in state_vec {
                    self.tuples.push(Tuple {
                        phone,
                        hmm_state,
                        forward_pdf: pdf,
                        self_loop_pdf: pdf,
                    });
                }
            }
        }
        // This sort defines the transition ids.
        self.tuples.sort_unstable();
    }

    /// Kaldi `ComputeDerived`.
    fn compute_derived(&mut self) {
        self.state2id = vec![0; self.tuples.len() + 2];
        let mut cur_transition_id: TransitionId = 1;
        self.num_pdfs = 0;
        for tstate in 1..=self.tuples.len() + 1 {
            self.state2id[tstate] = cur_transition_id;
            if tstate <= self.tuples.len() {
                let t = self.tuples[tstate - 1];
                self.num_pdfs = self.num_pdfs.max(1 + t.forward_pdf as usize);
                self.num_pdfs = self.num_pdfs.max(1 + t.self_loop_pdf as usize);
                let state = &self.topo.topology_for_phone(t.phone)[t.hmm_state];
                cur_transition_id += state.transitions.len() as TransitionId;
            }
        }

        self.id2state = vec![0; cur_transition_id as usize];
        self.id2pdf = vec![0; cur_transition_id as usize];
        for tstate in 1..=self.tuples.len() {
            for tid in self.state2id[tstate]..self.state2id[tstate + 1] {
                self.id2state[tid as usize] = tstate as u32;
            }
        }
        // Needs id2state populated first, since is_self_loop consults it.
        for tstate in 1..=self.tuples.len() {
            for tid in self.state2id[tstate]..self.state2id[tstate + 1] {
                self.id2pdf[tid as usize] = if self.is_self_loop(tid) {
                    self.tuples[tstate - 1].self_loop_pdf
                } else {
                    self.tuples[tstate - 1].forward_pdf
                };
            }
        }
    }

    /// Kaldi `InitializeProbs`: take the probabilities straight from the topology.
    fn initialize_probs(&mut self) {
        self.log_probs = vec![0.0; self.num_transition_ids() + 1];
        for tid in 1..=self.num_transition_ids() as TransitionId {
            let tstate = self.id2state[tid as usize] as usize;
            let tindex = (tid - self.state2id[tstate]) as usize;
            let t = self.tuples[tstate - 1];
            let entry = self.topo.topology_for_phone(t.phone);
            let prob = entry[t.hmm_state].transitions[tindex].1;
            assert!(
                prob > 0.0,
                "zero probability in topology; remove the entry instead"
            );
            if prob > 1.0 {
                tracing::warn!(prob, "transition probability greater than one");
            }
            self.log_probs[tid as usize] = prob.ln();
        }
        self.compute_derived_of_probs();
    }

    /// Kaldi `ComputeDerivedOfProbs`.
    fn compute_derived_of_probs(&mut self) {
        self.non_self_loop_log_probs = vec![0.0; self.num_transition_states() + 1];
        for tstate in 1..=self.num_transition_states() as u32 {
            match self.self_loop_of(tstate) {
                None => self.non_self_loop_log_probs[tstate as usize] = 0.0, // log(1.0)
                Some(tid) => {
                    let self_loop_prob = self.get_transition_log_prob(tid).exp();
                    let mut non_self_loop_prob = 1.0 - self_loop_prob;
                    if non_self_loop_prob <= 0.0 {
                        tracing::warn!(non_self_loop_prob, "non-self-loop probability <= 0");
                        non_self_loop_prob = 1.0e-10;
                    }
                    self.non_self_loop_log_probs[tstate as usize] = non_self_loop_prob.ln();
                }
            }
        }
    }

    /// Replace the transition probabilities with a set trained elsewhere,
    /// matching by tuple rather than by transition id. Used when importing a
    /// Kaldi model, whose tuple order need not be the one `new` rebuilds, so
    /// the log probs must be permuted into place. `foreign` maps each tuple to
    /// its per-index log probabilities in topology transition order; a missing
    /// or wrong-length entry is an error, so an import fails loudly.
    pub fn set_log_probs_by_tuple(
        &mut self,
        foreign: &BTreeMap<Tuple, Vec<f32>>,
    ) -> Result<(), String> {
        let mut new_log_probs = vec![0.0f32; self.num_transition_ids() + 1];
        for tstate in 1..=self.num_transition_states() {
            let tuple = self.tuples[tstate - 1];
            let probs = foreign
                .get(&tuple)
                .ok_or_else(|| format!("no imported probabilities for tuple {tuple:?}"))?;
            let n = self.num_transition_indices(tstate as u32);
            if probs.len() != n {
                let got = probs.len();
                return Err(format!(
                    "tuple {tuple:?}: {n} transitions here, {got} imported"
                ));
            }
            let first = self.state2id[tstate] as usize;
            new_log_probs[first..first + n].copy_from_slice(probs);
        }
        self.log_probs = new_log_probs;
        self.compute_derived_of_probs();
        Ok(())
    }

    /// Kaldi `Check`, in debug builds.
    fn check(&self) {
        assert!(self.num_transition_ids() != 0 && self.num_transition_states() != 0);
        debug_assert_eq!(
            (1..=self.num_transition_states() as u32)
                .map(|ts| self.num_transition_indices(ts))
                .sum::<usize>(),
            self.num_transition_ids()
        );
        debug_assert!(
            self.log_probs[1..]
                .iter()
                .all(|p| *p <= 0.0 && p.is_finite()),
            "log probs must be finite and non-positive"
        );
    }

    pub fn topology(&self) -> &HmmTopology {
        &self.topo
    }

    pub fn tuples(&self) -> &[Tuple] {
        &self.tuples
    }

    pub fn num_transition_ids(&self) -> usize {
        self.id2state.len().saturating_sub(1)
    }

    pub fn num_transition_states(&self) -> usize {
        self.tuples.len()
    }

    pub fn num_pdfs(&self) -> usize {
        self.num_pdfs
    }

    pub fn num_transition_indices(&self, trans_state: u32) -> usize {
        (self.state2id[trans_state as usize + 1] - self.state2id[trans_state as usize]) as usize
    }

    pub fn transition_id_to_transition_state(&self, t: TransitionId) -> u32 {
        debug_assert!(t != 0 && (t as usize) < self.id2state.len());
        self.id2state[t as usize]
    }

    pub fn transition_id_to_transition_index(&self, t: TransitionId) -> usize {
        let tstate = self.transition_id_to_transition_state(t);
        (t - self.state2id[tstate as usize]) as usize
    }

    pub fn transition_id_to_pdf(&self, t: TransitionId) -> PdfId {
        self.id2pdf[t as usize]
    }

    pub fn transition_id_to_phone(&self, t: TransitionId) -> PhoneId {
        self.tuples[self.transition_id_to_transition_state(t) as usize - 1].phone
    }

    pub fn transition_id_to_hmm_state(&self, t: TransitionId) -> usize {
        self.tuples[self.transition_id_to_transition_state(t) as usize - 1].hmm_state
    }

    pub fn transition_state_to_phone(&self, trans_state: u32) -> PhoneId {
        self.tuples[trans_state as usize - 1].phone
    }

    pub fn transition_state_to_hmm_state(&self, trans_state: u32) -> usize {
        self.tuples[trans_state as usize - 1].hmm_state
    }

    pub fn transition_state_to_forward_pdf(&self, trans_state: u32) -> PdfId {
        self.tuples[trans_state as usize - 1].forward_pdf
    }

    pub fn transition_state_to_self_loop_pdf(&self, trans_state: u32) -> PdfId {
        self.tuples[trans_state as usize - 1].self_loop_pdf
    }

    pub fn transition_state_to_forward_pdf_class(&self, trans_state: u32) -> i32 {
        let t = self.tuples[trans_state as usize - 1];
        self.topo.topology_for_phone(t.phone)[t.hmm_state].pdf_class
    }

    /// Same as the forward pdf-class for HMM (non-chain) models.
    pub fn transition_state_to_self_loop_pdf_class(&self, trans_state: u32) -> i32 {
        self.transition_state_to_forward_pdf_class(trans_state)
    }

    /// True if this transition id is the self-loop of its transition-state.
    pub fn is_self_loop(&self, t: TransitionId) -> bool {
        let tstate = self.transition_id_to_transition_state(t) as usize;
        let tindex = (t - self.state2id[tstate]) as usize;
        let tuple = self.tuples[tstate - 1];
        let entry = self.topo.topology_for_phone(tuple.phone);
        match entry[tuple.hmm_state].transitions.get(tindex) {
            Some(&(dest, _)) => dest == tuple.hmm_state,
            None => false,
        }
    }

    /// True if this transition id leads to the final (non-emitting) state of the phone's topology.
    pub fn is_final(&self, t: TransitionId) -> bool {
        let tstate = self.transition_id_to_transition_state(t) as usize;
        let tindex = (t - self.state2id[tstate]) as usize;
        let tuple = self.tuples[tstate - 1];
        let entry = self.topo.topology_for_phone(tuple.phone);
        entry[tuple.hmm_state].transitions[tindex].0 + 1 == entry.len()
    }

    /// Kaldi `SelfLoopOf`: the self-loop transition id of a transition-state, if it has one.
    pub fn self_loop_of(&self, trans_state: u32) -> Option<TransitionId> {
        let tuple = self.tuples[trans_state as usize - 1];
        let entry = self.topo.topology_for_phone(tuple.phone);
        entry[tuple.hmm_state]
            .self_loop_index(tuple.hmm_state)
            .map(|idx| self.pair_to_transition_id(trans_state, idx as u32))
    }

    pub fn get_transition_log_prob(&self, t: TransitionId) -> f32 {
        self.log_probs[t as usize]
    }

    pub fn get_transition_prob(&self, t: TransitionId) -> f32 {
        self.log_probs[t as usize].exp()
    }

    /// Kaldi `GetNonSelfLoopLogProb`: `log(1 - p_selfloop)` for the transition-state.
    pub fn get_non_self_loop_log_prob(&self, trans_state: u32) -> f32 {
        self.non_self_loop_log_probs[trans_state as usize]
    }

    /// Kaldi `GetTransitionLogProbIgnoringSelfLoops`: the transition's log prob renormalised so
    /// that the non-self-loop transitions out of the state sum to one.
    pub fn get_transition_log_prob_ignoring_self_loops(&self, t: TransitionId) -> f32 {
        debug_assert!(!self.is_self_loop(t));
        self.log_probs[t as usize]
            - self.get_non_self_loop_log_prob(self.transition_id_to_transition_state(t))
    }

    pub fn pair_to_transition_id(&self, trans_state: u32, trans_index: u32) -> TransitionId {
        debug_assert!(
            (trans_index as usize) < self.num_transition_indices(trans_state),
            "transition index out of range"
        );
        self.state2id[trans_state as usize] + trans_index
    }

    /// Kaldi `TupleToTransitionState`; binary search over the sorted tuple list.
    pub fn tuple_to_transition_state(
        &self,
        phone: PhoneId,
        hmm_state: usize,
        fwd_pdf: PdfId,
        self_pdf: PdfId,
    ) -> u32 {
        self.try_tuple_to_transition_state(phone, hmm_state, fwd_pdf, self_pdf)
            .unwrap_or_else(|| {
                panic!(
                    "TupleToTransitionState: no tuple ({phone}, {hmm_state}, {fwd_pdf}, \
                     {self_pdf}) -- incompatible tree and model?"
                )
            })
    }

    pub fn try_tuple_to_transition_state(
        &self,
        phone: PhoneId,
        hmm_state: usize,
        fwd_pdf: PdfId,
        self_pdf: PdfId,
    ) -> Option<u32> {
        let key = Tuple {
            phone,
            hmm_state,
            forward_pdf: fwd_pdf,
            self_loop_pdf: self_pdf,
        };
        self.tuples.binary_search(&key).ok().map(|i| i as u32 + 1)
    }

    /// Kaldi `TransitionModel::Accumulate`, which for a hard (Viterbi) alignment just adds the
    /// weight to the transition id's count.
    pub fn accumulate(&self, stats: &mut TransitionAccs, t: TransitionId, weight: f64) {
        debug_assert!(t != 0 && (t as usize) < stats.0.len());
        stats.0[t as usize] += weight;
    }

    /// Kaldi `MleUpdate` (transition-model.cc:475), the non-shared branch.
    ///
    /// Returns `(objf improvement, total count)`, both unnormalised, as Kaldi's out-params.
    pub fn mle_update(
        &mut self,
        stats: &TransitionAccs,
        opts: &MleTransitionUpdateConfig,
    ) -> (f64, f64) {
        assert_eq!(
            stats.0.len(),
            self.num_transition_ids() + 1,
            "stats must be indexed by transition id"
        );
        let mut count_sum = 0.0f64;
        let mut objf_impr_sum = 0.0f64;
        let mut num_skipped = 0usize;
        let mut num_floored = 0usize;
        let floor = opts.floor as f64;

        for tstate in 1..=self.num_transition_states() as u32 {
            let n = self.num_transition_indices(tstate);
            debug_assert!(n >= 1);
            if n <= 1 {
                continue; // no point updating if there is only one transition
            }
            let counts: Vec<f64> = (0..n)
                .map(|i| stats.0[self.pair_to_transition_id(tstate, i as u32) as usize])
                .collect();
            let tstate_tot: f64 = counts.iter().sum();
            count_sum += tstate_tot;
            if tstate_tot < opts.mincount as f64 {
                num_skipped += 1;
                continue;
            }
            let old_probs: Vec<f64> = (0..n)
                .map(|i| {
                    self.get_transition_prob(self.pair_to_transition_id(tstate, i as u32)) as f64
                })
                .collect();
            let mut new_probs: Vec<f64> = counts.iter().map(|c| c / tstate_tot).collect();
            // Kaldi floors and renormalises three times.
            for _ in 0..3 {
                let sum: f64 = new_probs.iter().sum();
                for p in &mut new_probs {
                    *p /= sum;
                }
                for p in &mut new_probs {
                    *p = p.max(floor);
                }
            }
            for i in 0..n {
                if new_probs[i] == floor {
                    num_floored += 1;
                }
                objf_impr_sum += counts[i] * (new_probs[i].ln() - old_probs[i].ln());
            }
            for i in 0..n {
                let tid = self.pair_to_transition_id(tstate, i as u32);
                let lp = new_probs[i].ln() as f32;
                assert!(lp.is_finite(), "log prob is inf or NaN: bad stats?");
                self.log_probs[tid as usize] = lp;
            }
        }
        tracing::info!(
            objf_per_frame = if count_sum > 0.0 {
                objf_impr_sum / count_sum
            } else {
                0.0
            },
            frames = count_sum,
            num_floored,
            num_skipped,
            num_transition_states = self.num_transition_states(),
            "TransitionModel::MleUpdate"
        );
        self.compute_derived_of_probs();
        (objf_impr_sum, count_sum)
    }

    /// Every pdf reachable from one of the given phones. Used for silence boosting.
    pub fn silence_pdfs(&self, silence_phones: &[PhoneId]) -> Vec<PdfId> {
        let mut pdfs: Vec<PdfId> = self
            .tuples
            .iter()
            .filter(|t| silence_phones.contains(&t.phone))
            .flat_map(|t| [t.forward_pdf, t.self_loop_pdf])
            .collect();
        pdfs.sort_unstable();
        pdfs.dedup();
        pdfs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::context::ContextDependency;

    fn setup() -> (ContextDependency, HmmTopology, TransitionModel) {
        let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
        let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3]];
        let t = topo.clone();
        let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
        let tm = TransitionModel::new(&ctx, &topo);
        (ctx, topo, tm)
    }

    #[test]
    fn tuples_are_sorted_and_cover_every_state() {
        let (_, topo, tm) = setup();
        assert!(tm.tuples().windows(2).all(|w| w[0] < w[1]));
        // Monophone: one tuple per (phone, emitting hmm-state).
        let expected: usize = topo
            .phones()
            .iter()
            .map(|&p| topo.topology_for_phone(p).len() - 1)
            .sum();
        assert_eq!(tm.num_transition_states(), expected);
        // 5 (silence) + 3 + 3 = 11 emitting states.
        assert_eq!(tm.num_transition_states(), 11);
    }

    #[test]
    fn transition_id_round_trips() {
        let (_, _, tm) = setup();
        for tid in 1..=tm.num_transition_ids() as TransitionId {
            let ts = tm.transition_id_to_transition_state(tid);
            let idx = tm.transition_id_to_transition_index(tid);
            assert_eq!(tm.pair_to_transition_id(ts, idx as u32), tid);
            let phone = tm.transition_state_to_phone(ts);
            let hs = tm.transition_state_to_hmm_state(ts);
            let f = tm.transition_state_to_forward_pdf(ts);
            let s = tm.transition_state_to_self_loop_pdf(ts);
            assert_eq!(tm.tuple_to_transition_state(phone, hs, f, s), ts);
        }
    }

    #[test]
    fn ids_are_contiguous_and_one_based() {
        let (_, _, tm) = setup();
        // Bakis phones: state 0 has 3 skip arcs, state 1 has 2, the last emitting state has
        // only its forward transition: 2 phones * (3 + 2 + 1) = 12. Silence: 4/4/4/4/2 = 18.
        let expected: usize = tm
            .tuples()
            .iter()
            .map(|t| {
                tm.topology().topology_for_phone(t.phone)[t.hmm_state]
                    .transitions
                    .len()
            })
            .sum();
        assert_eq!(expected, 12 + 18);
        assert_eq!(tm.num_transition_ids(), expected);
        assert_eq!(tm.state2id[1], 1);
    }

    #[test]
    fn self_loop_and_final_flags() {
        let (_, _, tm) = setup();
        // Bakis phone 2, state 0: no self-loop, three skip arcs to 1, 2 and the final state 3.
        let ts0 = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 2 && t.hmm_state == 0)
            .unwrap() as u32
            + 1;
        assert_eq!(tm.self_loop_of(ts0), None);
        assert_eq!(tm.num_transition_indices(ts0), 3);
        assert!(!tm.is_final(tm.pair_to_transition_id(ts0, 0)));
        assert!(!tm.is_final(tm.pair_to_transition_id(ts0, 1)));
        // The third arc goes straight to the final state 3.
        assert!(tm.is_final(tm.pair_to_transition_id(ts0, 2)));

        // State 1 is the Bakis state with a self-loop.
        let ts = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 2 && t.hmm_state == 1)
            .unwrap() as u32
            + 1;
        let loop_tid = tm.self_loop_of(ts).unwrap();
        assert!(tm.is_self_loop(loop_tid));
        assert!(!tm.is_final(loop_tid));
        let fwd = tm.pair_to_transition_id(ts, 1);
        assert!(!tm.is_self_loop(fwd));
        // 1 -> 2 is not the final state (which is index 3).
        assert!(!tm.is_final(fwd));

        let ts_last = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 2 && t.hmm_state == 2)
            .unwrap() as u32
            + 1;
        // The last emitting state has no self-loop: its only transition (index 0) is final.
        assert_eq!(tm.self_loop_of(ts_last), None);
        assert_eq!(tm.num_transition_indices(ts_last), 1);
        assert!(tm.is_final(tm.pair_to_transition_id(ts_last, 0)));
    }

    #[test]
    fn log_probs_come_from_topology() {
        let (_, _, tm) = setup();
        let ts = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 2 && t.hmm_state == 1)
            .unwrap() as u32
            + 1;
        let loop_tid = tm.self_loop_of(ts).unwrap();
        assert!((tm.get_transition_prob(loop_tid) - 0.5).abs() < 1e-6);
        // log(1 - 0.5).
        assert!((tm.get_non_self_loop_log_prob(ts) - 0.5f32.ln()).abs() < 1e-6);
        // The forward transition renormalised against non-self-loop mass is probability 1.
        let fwd = tm.pair_to_transition_id(ts, 1);
        assert!(tm.get_transition_log_prob_ignoring_self_loops(fwd).abs() < 1e-6);
    }

    #[test]
    fn transition_id_to_pdf_matches_ctx() {
        let (ctx, _, tm) = setup();
        for tid in 1..=tm.num_transition_ids() as TransitionId {
            let phone = tm.transition_id_to_phone(tid);
            let ts = tm.transition_id_to_transition_state(tid);
            let class = tm.transition_state_to_forward_pdf_class(ts);
            assert_eq!(
                tm.transition_id_to_pdf(tid),
                ctx.compute(&[phone], class).unwrap()
            );
        }
    }

    #[test]
    fn mle_update_moves_probs_toward_counts() {
        let (_, _, mut tm) = setup();
        let ts = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 2 && t.hmm_state == 1)
            .unwrap() as u32
            + 1;
        let loop_tid = tm.self_loop_of(ts).unwrap();
        let fwd = tm.pair_to_transition_id(ts, 1);
        let mut stats = TransitionAccs::new(tm.num_transition_ids());
        tm.accumulate(&mut stats, loop_tid, 90.0);
        tm.accumulate(&mut stats, fwd, 10.0);
        let (objf, count) = tm.mle_update(&stats, &MleTransitionUpdateConfig::default());
        assert_eq!(count, 100.0);
        assert!(
            objf > 0.0,
            "objf should improve when moving to the ML estimate"
        );
        assert!((tm.get_transition_prob(loop_tid) - 0.9).abs() < 1e-5);
        assert!((tm.get_transition_prob(fwd) - 0.1).abs() < 1e-5);
        // Derived quantity must be refreshed.
        assert!((tm.get_non_self_loop_log_prob(ts) - 0.1f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn mle_update_skips_low_count_states_and_floors() {
        let (_, _, mut tm) = setup();
        let ts = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 2 && t.hmm_state == 1)
            .unwrap() as u32
            + 1;
        let loop_tid = tm.self_loop_of(ts).unwrap();
        let fwd = tm.pair_to_transition_id(ts, 1);
        let before = tm.get_transition_prob(loop_tid);
        let mut stats = TransitionAccs::new(tm.num_transition_ids());
        // Below mincount = 5.0 -> skipped.
        tm.accumulate(&mut stats, loop_tid, 2.0);
        tm.mle_update(&stats, &MleTransitionUpdateConfig::default());
        assert_eq!(tm.get_transition_prob(loop_tid), before);

        // All mass on the self-loop -> the forward prob is floored at 0.01.
        let mut stats = TransitionAccs::new(tm.num_transition_ids());
        tm.accumulate(&mut stats, loop_tid, 100.0);
        tm.mle_update(&stats, &MleTransitionUpdateConfig::default());
        assert!((tm.get_transition_prob(fwd) - 0.01).abs() < 1e-5);
    }

    #[test]
    fn silence_pdfs_are_only_silence() {
        let (ctx, _, tm) = setup();
        let sil = tm.silence_pdfs(&[1]);
        assert_eq!(sil.len(), 5);
        for class in 0..5 {
            assert!(sil.contains(&ctx.compute(&[1], class).unwrap()));
        }
    }
}
