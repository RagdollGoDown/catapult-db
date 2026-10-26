//! Specialized data structures for graph search operations.
//!
//! This module provides efficient data structures tailored for approximate nearest
//! neighbor search on proximity graphs.
//!
//! # Submodules
//!
//! - [`candidates`]: Priority queue structures for maintaining top-k search candidates
//! - [`catapults`]: FIFO evicting sets for storing cached starting points from previous searches
//! - [`editable`]: Editable neighbor sets for graphs built in memory
//! - [`fixed`]: Immutable neighbor sets for graph nodes
//! - [`visited`]: Bitmap-based visited node tracking for efficient duplicate detection
//!
//! [`NeighborSet`] is the read interface beam search uses for either kind.

pub mod candidates;
pub mod catapults;
pub mod editable;
pub mod fixed;
mod neighbor_set;
pub mod visited;

pub use neighbor_set::NeighborSet;
