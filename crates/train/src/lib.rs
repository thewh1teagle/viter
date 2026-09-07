//! Training pipeline: monophone -> triphone -> LDA+MLLT -> SAT (fMLLR).

pub mod config;
pub mod lda;
pub mod mono;
pub mod pipeline;
pub mod pronprob;
pub mod sat;
pub mod tri;
