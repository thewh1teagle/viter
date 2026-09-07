//! Port of the parts of `plans/kaldi/src/hmm/hmm-utils.cc` that operate on alignments:
//! `SplitToPhones` (:723), `ConvertAlignment` (:1013), and the CTM-style interval extraction
//! kalpy does in `plans/kalpy/kalpy/gmm/data.py:191`.

use super::context::ContextDependency;
use super::transition::TransitionModel;
use crate::types::{
    Alignment, IntervalAlignment, PhoneId, PhoneInterval, Pronunciation, TransitionId, WordInterval,
};

/// Kaldi `IsReordered` (hmm-utils.cc). Determines from the alignment itself whether self-loops
/// come after (`true`) or before (`false`) the forward transition.
pub fn is_reordered(tm: &TransitionModel, alignment: &[TransitionId]) -> bool {
    for w in alignment.windows(2) {
        let ts1 = tm.transition_id_to_transition_state(w[0]);
        let ts2 = tm.transition_id_to_transition_state(w[1]);
        if ts1 != ts2 {
            let loop1 = tm.is_self_loop(w[0]);
            let loop2 = tm.is_self_loop(w[1]);
            debug_assert!(
                !(loop1 && loop2),
                "invalid alignment: two adjacent self-loops"
            );
            if loop1 {
                return true; // reordered: self-loop last
            }
            if loop2 {
                return false; // not reordered: self-loop first
            }
        }
    }
    // Only one transition-state in the whole sequence.
    match (alignment.first(), alignment.last()) {
        (Some(&f), Some(&l)) => {
            if tm.is_self_loop(f) {
                false
            } else {
                tm.is_self_loop(l)
            }
        }
        _ => false,
    }
}

/// Kaldi `SplitToPhones` (hmm-utils.cc:723): split a transition-id sequence into per-phone runs.
///
/// The reorder convention is inferred from the alignment, as Kaldi does.
pub fn split_to_phones(tm: &TransitionModel, tids: &[TransitionId]) -> Vec<Vec<TransitionId>> {
    split_to_phones_checked(tm, tids).0
}

/// As [`split_to_phones`], but also returns Kaldi's `was_ok` consistency flag: false when the
/// alignment does not start at a phone start or end at a phone end.
pub fn split_to_phones_checked(
    tm: &TransitionModel,
    tids: &[TransitionId],
) -> (Vec<Vec<TransitionId>>, bool) {
    if tids.is_empty() {
        return (Vec::new(), true);
    }
    let reordered = is_reordered(tm, tids);
    let mut was_ok = true;
    let mut end_points: Vec<usize> = Vec::new();

    let mut i = 0usize;
    while i < tids.len() {
        let tid = tids[i];
        if tm.is_final(tid) {
            if !reordered {
                end_points.push(i + 1);
            } else {
                // Reordered: the trailing self-loops of this state belong to the same phone.
                while i + 1 < tids.len() && tm.is_self_loop(tids[i + 1]) {
                    debug_assert_eq!(
                        tm.transition_id_to_transition_state(tids[i]),
                        tm.transition_id_to_transition_state(tids[i + 1])
                    );
                    i += 1;
                }
                end_points.push(i + 1);
            }
        } else if i + 1 == tids.len() {
            // Should have been caught by the is_final check above.
            was_ok = false;
            end_points.push(i + 1);
        } else {
            let this_state = tm.transition_id_to_transition_state(tids[i]);
            let next_state = tm.transition_id_to_transition_state(tids[i + 1]);
            if this_state != next_state {
                let this_phone = tm.transition_state_to_phone(this_state);
                let next_phone = tm.transition_state_to_phone(next_state);
                if this_phone != next_phone {
                    // The phone changed without a final transition: an error.
                    was_ok = false;
                    end_points.push(i + 1);
                }
            }
        }
        i += 1;
    }

    let mut out = Vec::with_capacity(end_points.len());
    let mut cur = 0usize;
    for &end in &end_points {
        let tstate = tm.transition_id_to_transition_state(tids[cur]);
        let phone = tm.transition_state_to_phone(tstate);
        let first_class = tm.topology().topology_for_phone(phone)[0].pdf_class;
        if first_class != super::topology::NO_PDF && tm.transition_state_to_hmm_state(tstate) != 0 {
            was_ok = false;
        }
        out.push(tids[cur..end].to_vec());
        cur = end;
    }
    (out, was_ok)
}

/// Kaldi `ChangeReorderingOfAlignment`: swap the position of self-loops relative to the forward
/// transition, within each run of one transition-state.
fn change_reordering(tm: &TransitionModel, alignment: &mut [TransitionId]) {
    let n = alignment.len();
    let mut i = 0usize;
    while i < n {
        let ts = tm.transition_id_to_transition_state(alignment[i]);
        let mut j = i;
        while j < n && tm.transition_id_to_transition_state(alignment[j]) == ts {
            j += 1;
        }
        // Within [i, j) there is exactly one non-self-loop; move it to the other end.
        let run = &mut alignment[i..j];
        if let Some(pos) = run.iter().position(|&t| !tm.is_self_loop(t)) {
            let forward = run[pos];
            if pos == 0 {
                // Was leading (not reordered) -> move to the back.
                run.copy_within(1.., 0);
                let last = run.len() - 1;
                run[last] = forward;
            } else if pos == run.len() - 1 {
                // Was trailing (reordered) -> move to the front.
                run.copy_within(..pos, 1);
                run[0] = forward;
            }
        }
        i = j;
    }
}

/// Kaldi `ConvertAlignment` (hmm-utils.cc:1013), restricted to the same-topology, no-subsampling,
/// no-phone-map case, which is what MFA's `convert_alignments` uses.
///
/// `phone_window_context` is accepted for interface compatibility with the contract; the actual
/// window width and central position always come from `new_ctx`, which is what Kaldi uses.
/// Returns `None` if the old alignment could not be split into phones consistently, or if some
/// phone window has no answer in the new tree.
pub fn convert_alignment(
    old: &TransitionModel,
    new: &TransitionModel,
    new_ctx: &ContextDependency,
    phone_window_context: usize,
    tids: &[TransitionId],
) -> Option<Vec<TransitionId>> {
    let _ = phone_window_context;
    if tids.is_empty() {
        return Some(Vec::new());
    }
    let old_is_reordered = is_reordered(old, tids);
    let (old_split, ok) = split_to_phones_checked(old, tids);
    if !ok {
        return None;
    }
    let phone_seq_len = old_split.len();
    let phones: Vec<PhoneId> = old_split
        .iter()
        .map(|run| old.transition_id_to_phone(run[0]))
        .collect();

    let n = new_ctx.n as i32;
    let p = new_ctx.p as i32;
    // Graphs built by this crate are reordered (Kaldi's default), so the output is too.
    let new_is_reordered = true;

    let mut out: Vec<TransitionId> = Vec::with_capacity(tids.len());
    // Kaldi sweeps win_start from -N to phone_seq_len + N and keeps the windows whose central
    // position lands inside the sequence; that is just central_pos = 0 .. phone_seq_len - 1.
    for central_pos in 0..phone_seq_len {
        let win_start = central_pos as i32 - p;
        let mut window = vec![0 as PhoneId; n as usize];
        for offset in 0..n {
            let idx = win_start + offset;
            if idx >= 0 && (idx as usize) < phone_seq_len {
                window[offset as usize] = phones[idx as usize];
            }
        }
        let converted = convert_alignment_for_phone(
            old,
            new,
            new_ctx,
            &old_split[central_pos],
            &window,
            old_is_reordered,
            new_is_reordered,
        )?;
        out.extend_from_slice(&converted);
    }
    debug_assert_eq!(out.len(), tids.len());
    Some(out)
}

/// Kaldi `ConvertAlignmentForPhone`, same-topology and same-length branch.
fn convert_alignment_for_phone(
    old: &TransitionModel,
    new: &TransitionModel,
    new_ctx: &ContextDependency,
    old_phone_alignment: &[TransitionId],
    new_phone_window: &[PhoneId],
    old_is_reordered: bool,
    new_is_reordered: bool,
) -> Option<Vec<TransitionId>> {
    let new_central_phone = new_phone_window[new_ctx.p];
    let old_central_phone = old.transition_id_to_phone(old_phone_alignment[0]);
    let old_entry = old.topology().try_for_phone(old_central_phone)?;
    let new_entry = new.topology().try_for_phone(new_central_phone)?;
    if old_entry != new_entry {
        // Topology mismatch. Kaldi would generate a random path here; the contract restricts us
        // to the same-topology case, so report failure instead of fabricating an alignment.
        tracing::warn!(
            old_central_phone,
            new_central_phone,
            "ConvertAlignment: topology mismatch, unsupported"
        );
        return None;
    }

    let num_classes = new.topology().num_pdf_classes(new_central_phone);
    let mut pdf_ids = Vec::with_capacity(num_classes);
    for class in 0..num_classes as i32 {
        pdf_ids.push(new_ctx.compute(new_phone_window, class)?);
    }

    let mut out = Vec::with_capacity(old_phone_alignment.len());
    for &old_tid in old_phone_alignment {
        let old_tstate = old.transition_id_to_transition_state(old_tid);
        let fwd_class = old.transition_state_to_forward_pdf_class(old_tstate);
        let loop_class = old.transition_state_to_self_loop_pdf_class(old_tstate);
        let hmm_state = old.transition_id_to_hmm_state(old_tid);
        let trans_idx = old.transition_id_to_transition_index(old_tid);
        let new_fwd = *pdf_ids.get(fwd_class as usize)?;
        let new_loop = *pdf_ids.get(loop_class as usize)?;
        let new_tstate =
            new.try_tuple_to_transition_state(new_central_phone, hmm_state, new_fwd, new_loop)?;
        out.push(new.pair_to_transition_id(new_tstate, trans_idx as u32));
    }

    if new_is_reordered != old_is_reordered {
        change_reordering(new, &mut out);
    }
    Some(out)
}

/// Convert a frame-level alignment into phone and word intervals.
///
/// Phone runs come from [`split_to_phones`]; word boundaries come from the word ids the decoder
/// recorded along the best path, re-associated with phone runs the way kalpy does in
/// `data.py:246-262`: word `k`'s interval starts at the phone run where its first phone begins
/// and ends where its last phone ends, with inserted silence left outside every word.
pub fn to_intervals(
    tm: &TransitionModel,
    ali: &Alignment,
    words_prons: &[Vec<Pronunciation>],
    frame_shift_s: f32,
) -> IntervalAlignment {
    let runs = split_to_phones(tm, &ali.tids);

    let mut phones: Vec<PhoneInterval> = Vec::with_capacity(runs.len());
    let mut frame = 0u32;
    for run in &runs {
        let phone = tm.transition_id_to_phone(run[0]);
        let start = frame;
        frame += run.len() as u32;
        phones.push(PhoneInterval {
            phone,
            start_frame: start,
            end_frame: frame,
        });
    }

    // Walk the phone runs alongside the decoded word sequence. Each word consumes as many
    // consecutive non-silence-insertion runs as it has phones; anything left over is silence.
    let mut words: Vec<WordInterval> = Vec::new();
    let mut pi = 0usize;
    for (wk, &w) in ali.words.iter().enumerate() {
        let pron_idx = ali.prons.get(wk).copied().unwrap_or(0);
        let pron_idx = if pron_idx == super::NO_PRON {
            0
        } else {
            pron_idx
        };
        let Some(alts) = words_prons.get(w as usize) else {
            continue;
        };
        let Some(expected) = alts
            .get(pron_idx as usize)
            .or_else(|| alts.first())
            .map(|p| &p.phones)
        else {
            continue;
        };
        // Skip runs that do not start this word (inserted optional silence).
        while pi < phones.len() && !expected.is_empty() && phones[pi].phone != expected[0] {
            pi += 1;
        }
        if pi >= phones.len() {
            break;
        }
        let start = phones[pi].start_frame;
        let take = expected.len().min(phones.len() - pi);
        let end = phones[pi + take - 1].end_frame;
        pi += take;
        words.push(WordInterval {
            word: w,
            pron: pron_idx,
            start_frame: start,
            end_frame: end,
        });
    }

    IntervalAlignment {
        utt: ali.utt.clone(),
        frame_shift_s,
        phones,
        words,
    }
}

#[cfg(test)]
mod tests {

    /// `build_graph` input for words with a single plain pronunciation each.
    fn plain(words: &[&[PhoneId]]) -> Vec<Vec<Pronunciation>> {
        words
            .iter()
            .map(|p| vec![Pronunciation::plain(p.to_vec())])
            .collect()
    }
    use super::*;
    use crate::hmm::context::ContextDependency;
    use crate::hmm::topology::HmmTopology;

    fn setup() -> (TransitionModel, ContextDependency) {
        let topo = HmmTopology::mfa_default(&[1], &[2, 3], 5, 3);
        let sets: Vec<Vec<PhoneId>> = vec![vec![1], vec![2], vec![3]];
        let t = topo.clone();
        let ctx = ContextDependency::monophone_shared(&sets, &move |p| t.num_pdf_classes(p));
        let tm = TransitionModel::new(&ctx, &topo);
        (tm, ctx)
    }

    /// A reordered alignment for one Bakis phone: each state's forward transition followed by
    /// `extra[i]` self-loops.
    fn bakis_alignment(tm: &TransitionModel, phone: PhoneId, extra: &[usize]) -> Vec<TransitionId> {
        let mut out = Vec::new();
        for (hmm_state, &n) in extra.iter().enumerate() {
            let ts = tm
                .tuples()
                .iter()
                .position(|t| t.phone == phone && t.hmm_state == hmm_state)
                .unwrap() as u32
                + 1;
            let fwd = tm.topology().topology_for_phone(phone)[hmm_state]
                .transitions
                .iter()
                .position(|&(d, _)| d != hmm_state)
                .unwrap();
            out.push(tm.pair_to_transition_id(ts, fwd as u32));
            let loop_tid = tm.self_loop_of(ts).unwrap();
            for _ in 0..n {
                out.push(loop_tid);
            }
        }
        out
    }

    #[test]
    fn detects_reordering() {
        let (tm, _) = setup();
        let ali = bakis_alignment(&tm, 2, &[1, 0, 2]);
        assert!(is_reordered(&tm, &ali));
    }

    #[test]
    fn split_to_phones_one_phone() {
        let (tm, _) = setup();
        let ali = bakis_alignment(&tm, 2, &[1, 0, 2]);
        let (runs, ok) = split_to_phones_checked(&tm, &ali);
        assert!(ok);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len(), ali.len());
    }

    #[test]
    fn split_to_phones_two_phones() {
        let (tm, _) = setup();
        let mut ali = bakis_alignment(&tm, 2, &[0, 1, 0]);
        let second = bakis_alignment(&tm, 3, &[2, 0, 0]);
        let first_len = ali.len();
        ali.extend_from_slice(&second);
        let (runs, ok) = split_to_phones_checked(&tm, &ali);
        assert!(ok);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].len(), first_len);
        assert_eq!(runs[1].len(), second.len());
        assert_eq!(tm.transition_id_to_phone(runs[0][0]), 2);
        assert_eq!(tm.transition_id_to_phone(runs[1][0]), 3);
    }

    #[test]
    fn split_to_phones_empty() {
        let (tm, _) = setup();
        assert!(split_to_phones(&tm, &[]).is_empty());
    }

    #[test]
    fn convert_alignment_is_identity_for_the_same_model() {
        let (tm, ctx) = setup();
        let mut ali = bakis_alignment(&tm, 2, &[1, 0, 1]);
        ali.extend(bakis_alignment(&tm, 3, &[0, 2, 0]));
        let out = convert_alignment(&tm, &tm, &ctx, 1, &ali).expect("conversion must succeed");
        assert_eq!(out, ali);
    }

    #[test]
    fn convert_alignment_empty() {
        let (tm, ctx) = setup();
        assert_eq!(convert_alignment(&tm, &tm, &ctx, 1, &[]), Some(Vec::new()));
    }

    #[test]
    fn change_reordering_round_trips() {
        let (tm, _) = setup();
        let ali = bakis_alignment(&tm, 2, &[2, 1, 0]);
        let mut flipped = ali.clone();
        change_reordering(&tm, &mut flipped);
        assert_ne!(flipped, ali);
        assert!(!is_reordered(&tm, &flipped));
        let mut back = flipped.clone();
        change_reordering(&tm, &mut back);
        assert_eq!(back, ali);
    }

    #[test]
    fn to_intervals_phone_boundaries() {
        let (tm, _) = setup();
        let mut ali = bakis_alignment(&tm, 2, &[1, 0, 0]); // 4 frames
        let n1 = ali.len();
        ali.extend(bakis_alignment(&tm, 3, &[0, 0, 1])); // 4 frames
        let total = ali.len();
        let a = Alignment {
            utt: "u".into(),
            tids: ali,
            words: vec![0, 1],
            prons: vec![0, 0],
            loglike: 0.0,
        };
        let iv = to_intervals(&tm, &a, &plain(&[&[2], &[3]]), 0.01);
        assert_eq!(iv.phones.len(), 2);
        assert_eq!(iv.phones[0].phone, 2);
        assert_eq!(iv.phones[0].start_frame, 0);
        assert_eq!(iv.phones[0].end_frame, n1 as u32);
        assert_eq!(iv.phones[1].end_frame, total as u32);
        assert_eq!(iv.num_frames(), total as u32);
        assert_eq!(iv.words.len(), 2);
        assert_eq!(iv.words[0].start_frame, 0);
        assert_eq!(iv.words[1].end_frame, total as u32);
    }

    #[test]
    fn to_intervals_skips_inserted_silence() {
        let (tm, _) = setup();
        // silence, then word phone 2.
        let sil_ts = tm
            .tuples()
            .iter()
            .position(|t| t.phone == 1 && t.hmm_state == 4)
            .unwrap() as u32
            + 1;
        // Silence state 4 exits to the final state at index 1 of its transitions.
        let sil_final = tm.pair_to_transition_id(sil_ts, 1);
        let mut ali = vec![sil_final];
        ali.extend(bakis_alignment(&tm, 2, &[0, 0, 0]));
        let a = Alignment {
            utt: "u".into(),
            tids: ali,
            words: vec![0],
            prons: vec![0],
            loglike: 0.0,
        };
        let iv = to_intervals(&tm, &a, &plain(&[&[2]]), 0.01);
        assert_eq!(iv.phones[0].phone, 1);
        assert_eq!(iv.words.len(), 1);
        // The word starts after the silence run.
        assert_eq!(iv.words[0].start_frame, 1);
    }
}
