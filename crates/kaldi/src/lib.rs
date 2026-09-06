//! Kaldi math in Rust. Features, GMM, HMM, Viterbi alignment, tree clustering,
//! feature-space transforms, and a CPU/GPU device layer for batched scoring.

pub mod types;
pub mod audio;
pub mod feat;
pub mod device;
pub mod gmm;
pub mod hmm;
pub mod align;
pub mod tree;
pub mod transform;
pub mod model;
