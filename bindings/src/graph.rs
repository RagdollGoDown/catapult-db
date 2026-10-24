use catapult::{
    numerics::AlignedBlock,
    search::{AdjacencyGraph as InternalGraph, SearchStrategy},
    sets::catapults::LruSet,
    statistics::Stats,
};
use pyo3::{PyResult, pyclass, pymethods};
use std::path::PathBuf;

use crate::vecpy::VecPy;

/// Python wrapper for the Catapult AdjacencyGraph.
///
/// This class provides approximate nearest neighbor search on a flat proximity graph
/// using beam search with optional LSH-cached starting points (catapults).
///
/// # Thread Safety
/// This class is marked as `unsendable`, meaning it should not be accessed from
/// multiple Python threads. The implementation uses internal mutations that are
/// not thread-safe from Python's perspective.
#[pyclass(unsendable)]
pub struct AdjacencyGraph {
    inner: InternalGraph<LruSet>,
}

#[pymethods]
impl AdjacencyGraph {
    /// Load a graph from disk files.
    ///
    /// Args:
    ///     graph_path: Path to the graph metadata file
    ///     payload_path: Path to the graph payload/data file
    ///     catapults_enabled: Whether to enable LSH-based catapult acceleration
    ///     num_hash: Number of hash functions for LSH (default: 5)
    ///     bucket_capacity: Maximum number of catapults stored per LSH bucket (default: 30)
    ///     seed: Random seed for LSH hash functions (default: 42)
    ///
    /// Returns:
    ///     A new AdjacencyGraph instance ready for search
    #[staticmethod]
    #[pyo3(signature = (graph_path, payload_path, catapults_enabled, num_hash=5, bucket_capacity=30, seed=42))]
    pub fn load(
        graph_path: String,
        payload_path: String,
        catapults_enabled: bool,
        num_hash: usize,
        bucket_capacity: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let strategy = if catapults_enabled {
            SearchStrategy::Catapult
        } else {
            SearchStrategy::Vanilla
        };

        let graph = InternalGraph::load_flat_from_path(
            PathBuf::from(graph_path),
            PathBuf::from(payload_path),
            num_hash,
            bucket_capacity,
            seed,
            strategy,
        );

        Ok(Self { inner: graph })
    }

    /// Perform beam search to find approximate nearest neighbors.
    ///
    /// Args:
    ///     query: Query vector as a list of f32 values
    ///     num_neighbors: Number of nearest neighbors to return (k)
    ///     beam_width: Width of the search beam (larger = more accurate but slower)
    ///
    /// Returns:
    ///     A list of node IDs representing the k approximate nearest neighbors,
    ///     ordered by distance (closest first)
    pub fn search(&mut self, query: VecPy, num_neighbors: usize, beam_width: usize) -> Vec<usize> {
        // Convert the query to aligned blocks
        let query_aligned = align_query(&query.inner);

        // Perform the search
        let mut stats = Stats::new();
        let results = self
            .inner
            .beam_search(&query_aligned, num_neighbors, beam_width, &mut stats);

        // Convert CandidateEntry to usize node IDs
        results
            .into_iter()
            .map(|candidate| candidate.index.internal)
            .collect()
    }

    /// Perform batch search for multiple queries.
    ///
    /// Args:
    ///     queries: List of query vectors
    ///     num_neighbors: Number of nearest neighbors to return for each query
    ///     beam_width: Width of the search beam
    ///
    /// Returns:
    ///     A list of result lists, one for each query
    pub fn batch_search(
        &mut self,
        queries: Vec<VecPy>,
        num_neighbors: usize,
        beam_width: usize,
    ) -> Vec<Vec<usize>> {
        queries
            .into_iter()
            .map(|q| self.search(q, num_neighbors, beam_width))
            .collect()
    }

    /// Get the number of nodes in the graph.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Helper function to convert a query vector to aligned blocks.
fn align_query(query: &[f32]) -> Vec<AlignedBlock> {
    AlignedBlock::allocate_padded(query.to_vec())
}
