//! Readers for the Kaldi objects a `.mdl` / `tree` / `.mat` file holds, each a
//! direct transcription of the corresponding C++ `Read` method.

use ndarray::Array2;

use super::read::{KaldiReadError, KaldiReader};
use crate::gmm::{AmDiagGmm, DiagGmm};
use crate::hmm::context::ContextDependency;
use crate::hmm::topology::{HmmState, HmmTopology, TopologyEntry};
use crate::hmm::transition::Tuple;
use crate::tree::event_map::EventMap;
use crate::types::{PdfId, PhoneId};

type Result<T> = std::result::Result<T, KaldiReadError>;

/// A `TransitionModel` exactly as Kaldi stored it: the topology, the tuple list
/// in Kaldi's own order, and the per-transition-id log probabilities.
pub struct KaldiTransitionModel {
    pub topo: HmmTopology,
    pub tuples: Vec<Tuple>,
    /// Indexed by transition id; element 0 is Kaldi's unused slot.
    pub log_probs: Vec<f32>,
}

/// `HmmTopology::Read`, binary branch (`plans/kaldi/src/hmm/hmm-topology.cc`).
///
/// Kaldi stores `phones_` and `phone2idx_` as integer vectors, then the entries.
/// A leading `-1` in place of the entry count marks a non-HMM (chain) topology
/// where forward and self-loop pdf-classes differ per state.
pub fn read_topology(r: &mut KaldiReader) -> Result<HmmTopology> {
    r.expect("<Topology>")?;
    let phones = r.int_vec()?;
    let phone2idx = r.int_vec()?;

    let mut sz = r.i32()?;
    let is_hmm = sz != -1;
    if !is_hmm {
        sz = r.i32()?;
    }
    if sz < 0 {
        return Err(r.malformed(format!("negative topology entry count {sz}")));
    }

    let mut entries: Vec<Vec<HmmState>> = Vec::with_capacity(sz as usize);
    for _ in 0..sz {
        let num_states = r.i32()?;
        let mut states = Vec::with_capacity(num_states.max(0) as usize);
        for _ in 0..num_states {
            let forward_pdf_class = r.i32()?;
            let self_loop_pdf_class = if is_hmm { forward_pdf_class } else { r.i32()? };
            if self_loop_pdf_class != forward_pdf_class {
                return Err(r.malformed(
                    "chain-style topology (forward != self-loop pdf-class) is not supported",
                ));
            }
            let num_trans = r.i32()?;
            let mut transitions = Vec::with_capacity(num_trans.max(0) as usize);
            for _ in 0..num_trans {
                let dst = r.i32()?;
                let prob = r.float()?;
                transitions.push((dst as usize, prob));
            }
            states.push(HmmState::new(forward_pdf_class, transitions));
        }
        entries.push(states);
    }
    r.expect("</Topology>")?;

    // viter's `HmmTopology` keeps the phone list on each entry rather than in a
    // separate `phone2idx_` table, so invert Kaldi's mapping.
    let mut per_entry: Vec<Vec<PhoneId>> = vec![Vec::new(); entries.len()];
    for &phone in &phones {
        let idx = phone2idx
            .get(phone as usize)
            .copied()
            .filter(|&i| i >= 0)
            .ok_or_else(|| r.malformed(format!("phone {phone} has no topology entry")))?;
        let idx = idx as usize;
        if idx >= per_entry.len() {
            return Err(r.malformed(format!("phone {phone} maps to entry {idx}, out of range")));
        }
        per_entry[idx].push(phone as PhoneId);
    }

    Ok(HmmTopology {
        entries: entries
            .into_iter()
            .zip(per_entry)
            .map(|(states, phones)| TopologyEntry { phones, states })
            .collect(),
    })
}

/// `TransitionModel::Read` (`plans/kaldi/src/hmm/transition-model.cc`).
///
/// The tuple block is headed by either `<Triples>` (self-loop pdf implied equal
/// to the forward pdf) or `<Tuples>` (both stored).
pub fn read_transition_model(r: &mut KaldiReader) -> Result<KaldiTransitionModel> {
    r.expect("<TransitionModel>")?;
    let topo = read_topology(r)?;

    let head = r.token()?;
    let has_self_loop_pdf = match head.as_str() {
        "<Tuples>" => true,
        "<Triples>" => false,
        _ => return Err(r.malformed(format!("expected <Tuples> or <Triples>, got {head}"))),
    };
    let size = r.i32()?;
    if size < 0 {
        return Err(r.malformed(format!("negative tuple count {size}")));
    }
    let mut tuples = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let phone = r.i32()? as PhoneId;
        let hmm_state = r.i32()?;
        let forward_pdf = r.i32()? as PdfId;
        let self_loop_pdf = if has_self_loop_pdf {
            r.i32()? as PdfId
        } else {
            forward_pdf
        };
        tuples.push(Tuple {
            phone,
            hmm_state: hmm_state.max(0) as usize,
            forward_pdf,
            self_loop_pdf,
        });
    }
    let close = r.token()?;
    if close != "</Tuples>" && close != "</Triples>" {
        return Err(r.malformed(format!("expected </Tuples> or </Triples>, got {close}")));
    }

    r.expect("<LogProbs>")?;
    let log_probs = r.vector()?;
    r.expect("</LogProbs>")?;
    r.expect("</TransitionModel>")?;

    Ok(KaldiTransitionModel {
        topo,
        tuples,
        log_probs,
    })
}

/// `DiagGmm::Read` (`plans/kaldi/src/gmm/diag-gmm.cc`). The `<GCONSTS>` block is
/// optional; Kaldi recomputes them anyway, and so do we.
pub fn read_diag_gmm(r: &mut KaldiReader) -> Result<DiagGmm> {
    let open = r.token()?;
    if open != "<DiagGMM>" && open != "<DiagGMMBegin>" {
        return Err(r.malformed(format!("expected <DiagGMM>, got {open}")));
    }
    let mut tok = r.token()?;
    if tok == "<GCONSTS>" {
        let _ = r.vector()?;
        tok = r.token()?;
    }
    if tok != "<WEIGHTS>" {
        return Err(r.malformed(format!("expected <WEIGHTS>, got {tok}")));
    }
    let weights = r.vector()?;

    r.expect("<MEANS_INVVARS>")?;
    let (mr, mc, mdata) = r.matrix()?;
    r.expect("<INV_VARS>")?;
    let (vr, vc, vdata) = r.matrix()?;

    if (mr, mc) != (vr, vc) || mr != weights.len() {
        return Err(r.malformed(format!(
            "GMM shape mismatch: {} weights, means {mr}x{mc}, inv_vars {vr}x{vc}",
            weights.len()
        )));
    }

    let close = r.token()?;
    if close != "</DiagGMM>" && close != "<DiagGMMEnd>" {
        return Err(r.malformed(format!("expected </DiagGMM>, got {close}")));
    }

    let means_invvars = Array2::from_shape_vec((mr, mc), mdata)
        .map_err(|e| r.malformed(format!("means_invvars: {e}")))?;
    let inv_vars = Array2::from_shape_vec((vr, vc), vdata)
        .map_err(|e| r.malformed(format!("inv_vars: {e}")))?;

    let mut gmm = DiagGmm {
        weights,
        means_invvars,
        inv_vars,
        gconsts: vec![0.0; mr],
    };
    // Kaldi ends `DiagGmm::Read` with ComputeGconsts() rather than trusting the
    // stored ones, so the imported model matches what Kaldi would score with.
    gmm.compute_gconsts();
    Ok(gmm)
}

/// `AmDiagGmm::Read` (`plans/kaldi/src/gmm/am-diag-gmm.cc`).
pub fn read_am_diag_gmm(r: &mut KaldiReader) -> Result<AmDiagGmm> {
    r.expect("<DIMENSION>")?;
    let dim = r.i32()?;
    r.expect("<NUMPDFS>")?;
    let num_pdfs = r.i32()?;
    if num_pdfs <= 0 {
        return Err(r.malformed(format!("bad pdf count {num_pdfs}")));
    }
    let mut am = AmDiagGmm::new();
    for i in 0..num_pdfs {
        let gmm = read_diag_gmm(r)?;
        if gmm.dim() != dim as usize {
            return Err(r.malformed(format!(
                "pdf {i} has dim {} but the model header says {dim}",
                gmm.dim()
            )));
        }
        am.add_pdf(gmm);
    }
    Ok(am)
}

/// `EventMap::Read` (`plans/kaldi/src/tree/event-map.cc`). `NULL` children are
/// legal inside a table, so this returns an `Option`.
pub fn read_event_map(r: &mut KaldiReader) -> Result<Option<EventMap>> {
    match r.peek() {
        Some(b'N') => {
            r.expect("NULL")?;
            Ok(None)
        }
        Some(b'C') => {
            r.expect("CE")?;
            Ok(Some(EventMap::Constant(r.i32()? as PdfId)))
        }
        Some(b'T') => {
            r.expect("TE")?;
            let key = r.i32()?;
            // `TableEventMap::Read` reads the size as a `uint32`, which Kaldi
            // marks with a negative size byte.
            let size = r.u32()?;
            r.expect("(")?;
            let mut table = Vec::with_capacity(size as usize);
            for _ in 0..size {
                table.push(read_event_map(r)?.map(Box::new));
            }
            r.expect(")")?;
            Ok(Some(EventMap::Table { key, table }))
        }
        Some(b'S') => {
            r.expect("SE")?;
            let key = r.i32()?;
            // The yes-set is a `ConstIntegerSet`, whose Read is just
            // ReadIntegerVector (`util/const-integer-set-inl.h:82`).
            let yes_set = r.int_vec()?;
            r.expect("{")?;
            let yes = read_event_map(r)?
                .ok_or_else(|| r.malformed("split node with a NULL yes-child"))?;
            let no =
                read_event_map(r)?.ok_or_else(|| r.malformed("split node with a NULL no-child"))?;
            r.expect("}")?;
            Ok(Some(EventMap::Split {
                key,
                yes_set,
                yes: Box::new(yes),
                no: Box::new(no),
            }))
        }
        other => Err(r.malformed(format!(
            "expected an event-map node (N/C/T/S), found {:?}",
            other.map(|c| c as char)
        ))),
    }
}

/// `ContextDependency::Read` (`plans/kaldi/src/tree/context-dep.cc`).
pub fn read_context_dependency(r: &mut KaldiReader) -> Result<ContextDependency> {
    r.expect("ContextDependency")?;
    let n = r.i32()?;
    let p = r.i32()?;
    let mut tok = r.token()?;
    if tok == "ToLength" {
        // Obsolete field Kaldi still tolerates; read and discard it.
        let _ = read_event_map(r)?;
        tok = r.token()?;
    }
    if tok != "ToPdf" {
        return Err(r.malformed(format!("expected ToPdf, got {tok}")));
    }
    let map = read_event_map(r)?.ok_or_else(|| r.malformed("tree has a NULL root"))?;
    r.expect("EndContextDependency")?;
    Ok(ContextDependency::new(
        n.max(0) as usize,
        p.max(0) as usize,
        map,
    ))
}

/// A standalone Kaldi matrix file (`lda.mat` and friends).
pub fn read_matrix_file(bytes: &[u8]) -> Result<Array2<f32>> {
    let mut r = KaldiReader::new(bytes)?;
    let (rows, cols, data) = r.matrix()?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| KaldiReadError::Malformed {
        offset: 0,
        msg: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(out: &mut Vec<u8>, t: &str) {
        out.extend_from_slice(t.as_bytes());
        out.push(b' ');
    }
    fn i32b(out: &mut Vec<u8>, v: i32) {
        out.push(4);
        out.extend_from_slice(&v.to_le_bytes());
    }
    /// A `uint32` as `WriteBasicType` emits it: a *negative* size byte.
    fn u32b(out: &mut Vec<u8>, v: u32) {
        out.push((-4i8) as u8);
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn ivec(out: &mut Vec<u8>, v: &[i32]) {
        out.push(4);
        out.extend_from_slice(&(v.len() as i32).to_le_bytes());
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    #[test]
    fn reads_a_constant_event_map() {
        let mut b = vec![0u8, b'B'];
        tok(&mut b, "CE");
        i32b(&mut b, 12);
        let mut r = KaldiReader::new(&b).unwrap();
        assert!(matches!(
            read_event_map(&mut r).unwrap(),
            Some(EventMap::Constant(12))
        ));
    }

    #[test]
    fn reads_a_split_over_two_constants_with_a_const_integer_set_yes_side() {
        let mut b = vec![0u8, b'B'];
        tok(&mut b, "SE");
        i32b(&mut b, -1); // key = kPdfClass
        ivec(&mut b, &[0, 1]);
        tok(&mut b, "{");
        tok(&mut b, "CE");
        i32b(&mut b, 3);
        tok(&mut b, "CE");
        i32b(&mut b, 4);
        tok(&mut b, "}");

        let mut r = KaldiReader::new(&b).unwrap();
        match read_event_map(&mut r).unwrap().unwrap() {
            EventMap::Split {
                key,
                yes_set,
                yes,
                no,
            } => {
                assert_eq!(key, -1);
                assert_eq!(yes_set, vec![0, 1]);
                assert!(matches!(*yes, EventMap::Constant(3)));
                assert!(matches!(*no, EventMap::Constant(4)));
            }
            other => panic!("expected a split, got {other:?}"),
        }
    }

    #[test]
    fn a_table_keeps_null_children_as_none() {
        let mut b = vec![0u8, b'B'];
        tok(&mut b, "TE");
        i32b(&mut b, 0);
        u32b(&mut b, 2); // TableEventMap stores its size as uint32
        tok(&mut b, "(");
        tok(&mut b, "NULL");
        tok(&mut b, "CE");
        i32b(&mut b, 7);
        tok(&mut b, ")");

        let mut r = KaldiReader::new(&b).unwrap();
        match read_event_map(&mut r).unwrap().unwrap() {
            EventMap::Table { key, table } => {
                assert_eq!(key, 0);
                assert!(table[0].is_none());
                assert!(matches!(table[1].as_deref(), Some(EventMap::Constant(7))));
            }
            other => panic!("expected a table, got {other:?}"),
        }
    }

    #[test]
    fn reads_a_two_phone_topology_and_reattaches_the_phone_lists() {
        let mut b = vec![0u8, b'B'];
        tok(&mut b, "<Topology>");
        ivec(&mut b, &[1, 2]); // phones
        ivec(&mut b, &[-1, 0, 0]); // phone2idx: both phones use entry 0
        i32b(&mut b, 1); // one entry
        i32b(&mut b, 2); // two states
        // state 0: pdf class 0, self loop + forward
        i32b(&mut b, 0);
        i32b(&mut b, 2);
        i32b(&mut b, 0);
        b.push(4);
        b.extend_from_slice(&0.5f32.to_le_bytes());
        i32b(&mut b, 1);
        b.push(4);
        b.extend_from_slice(&0.5f32.to_le_bytes());
        // state 1: final, non-emitting, no transitions
        i32b(&mut b, -1);
        i32b(&mut b, 0);
        tok(&mut b, "</Topology>");

        let mut r = KaldiReader::new(&b).unwrap();
        let topo = read_topology(&mut r).unwrap();
        assert_eq!(topo.entries.len(), 1);
        assert_eq!(topo.entries[0].phones, vec![1, 2]);
        assert_eq!(topo.entries[0].states.len(), 2);
        assert_eq!(
            topo.entries[0].states[0].transitions,
            vec![(0, 0.5), (1, 0.5)]
        );
        assert!(!topo.entries[0].states[1].is_emitting());
    }
}
