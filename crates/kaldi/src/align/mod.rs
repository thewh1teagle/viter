//! Viterbi alignment over an [`crate::hmm::Graph`].
//!
//! [`viterbi`] ports Kaldi's `FasterDecoder`; [`equal`] ports `fst::EqualAlign`, used for the
//! monophone flat start. See plans/CONTRACTS.md.

pub mod equal;
pub mod viterbi;

pub use equal::equal_align;
pub use viterbi::{AlignOptions, align, graph_pdfs};

use crate::types::{Alignment, TransitionId};

/// Per-frame posteriors from a Viterbi alignment: weight 1.0 on the aligned transition id.
///
/// This is what `AccumAmDiagGmm::AccumulateForGmm` consumes when accumulating from a hard
/// alignment, and it is what MFA's training loop uses at every stage.
pub fn alignment_to_posteriors(ali: &Alignment) -> Vec<(TransitionId, f32)> {
    ali.tids.iter().map(|&t| (t, 1.0)).collect()
}

#[derive(Debug, thiserror::Error)]
pub enum AlignError {
    #[error("no path through the graph, even with the retry beam")]
    NoPath,
    #[error("utterance has too few frames ({frames}) to align")]
    TooShort { frames: usize },
    #[error("scores matrix has {cols} columns but the graph needs a column for pdf {pdf}")]
    MissingPdf {
        cols: usize,
        pdf: crate::types::PdfId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posteriors_are_unit_weight_per_frame() {
        let ali = Alignment {
            utt: "u".into(),
            tids: vec![3, 3, 7],
            words: vec![],
            prons: vec![],
            loglike: 0.0,
        };
        let post = alignment_to_posteriors(&ali);
        assert_eq!(post, vec![(3, 1.0), (3, 1.0), (7, 1.0)]);
    }
}
