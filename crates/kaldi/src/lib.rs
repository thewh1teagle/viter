//! Kaldi math in Rust. Features, GMM, HMM, Viterbi alignment, tree clustering,
//! feature-space transforms, and a CPU/GPU device layer for batched scoring.

pub mod align;
pub mod audio;
pub mod device;
pub mod feat;
pub mod gmm;
pub mod hmm;
pub mod kaldi_io;
pub mod model;
pub mod transform;
pub mod tree;
pub mod types;
