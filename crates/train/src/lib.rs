//! Training pipeline: monophone -> triphone -> LDA+MLLT -> SAT (fMLLR).

pub mod config;
pub mod mono;
pub mod tri;
pub mod lda;
pub mod sat;
pub mod pipeline;
