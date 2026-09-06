//! HMM topology, transition model, context dependency, and per-utterance decoding graphs.
//!
//! Ports of `plans/kaldi/src/hmm/hmm-topology.{h,cc}`, `transition-model.{h,cc}`,
//! `hmm-utils.cc`, and `plans/kaldi/src/tree/context-dep.{h,cc}`. See plans/CONTRACTS.md.

pub mod context;
pub mod graph;
mod selfloops;
pub mod topology;
pub mod transition;
pub mod utils;

pub use context::{ContextDependency, PDF_CLASS_KEY};
pub use graph::{build_graph, graph_pdfs, Arc, Graph, GraphOptions, GraphState, NO_PRON, NO_WORD};
pub use topology::{HmmState, HmmTopology, TopologyEntry, NO_PDF};
pub use transition::{
    MleTransitionUpdateConfig, TransitionAccs, TransitionModel, Tuple,
};
pub use utils::{
    convert_alignment, is_reordered, split_to_phones, split_to_phones_checked, to_intervals,
};

/// Errors from HMM construction.
#[derive(Debug, thiserror::Error)]
pub enum HmmError {
    #[error("invalid topology: {0}")]
    Topology(String),
    #[error("phone {0} is not covered by the topology")]
    UncoveredPhone(crate::types::PhoneId),
    #[error("context-dependency object gave no pdf for window {window:?} pdf-class {pdf_class}")]
    NoPdf {
        window: Vec<crate::types::PhoneId>,
        pdf_class: i32,
    },
}
