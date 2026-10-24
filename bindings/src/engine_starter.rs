use catapult::{
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{
        NodeId,
        hash_start::{EngineStarter as InternalStarter, EngineStarterParams},
    },
    sets::catapults::LruSet,
};
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

use crate::vecpy::VecPy;

/// Starting points selected for a query.
///
/// Attributes:
///     signature: LSH signature (bucket index) of the query
///     catapults: Node IDs cached in the query's bucket (may be empty)
///     starting_node: The default starting node, always present
#[pyclass(frozen, get_all)]
pub struct StartingPoints {
    signature: usize,
    catapults: Vec<usize>,
    starting_node: usize,
}

#[pymethods]
impl StartingPoints {
    fn __repr__(&self) -> String {
        format!(
            "StartingPoints(signature={}, catapults={:?}, starting_node={})",
            self.signature, self.catapults, self.starting_node
        )
    }
}

/// Python wrapper for the Catapult EngineStarter.
///
/// Maps query vectors to LSH buckets of cached starting points (catapults)
/// from previous searches. Each bucket evicts its least recently used entry
/// when full.
#[pyclass]
pub struct EngineStarter {
    inner: InternalStarter<LruSet>,
    num_hash: usize,
    dim: usize,
}

#[pymethods]
impl EngineStarter {
    /// Create a new engine starter.
    ///
    /// Args:
    ///     num_hash: Number of LSH hash bits (creates 2^num_hash buckets)
    ///     bucket_capacity: Maximum number of catapults stored per bucket
    ///     dim: Dimension of the query vectors
    ///     starting_node: Default starting node, always returned
    ///     seed: Random seed for LSH hyperplane generation (default: 42)
    ///     enabled_catapults: Whether to look up catapults (default: True)
    ///
    /// Raises:
    ///     ValueError: If num_hash or bucket_capacity is out of range
    #[new]
    #[pyo3(signature = (num_hash, bucket_capacity, dim, starting_node, seed=42, enabled_catapults=true))]
    pub fn new(
        num_hash: usize,
        bucket_capacity: usize,
        dim: usize,
        starting_node: usize,
        seed: u64,
        enabled_catapults: bool,
    ) -> PyResult<Self> {
        if num_hash >= usize::BITS as usize {
            return Err(PyValueError::new_err(format!(
                "num_hash must be less than {}",
                usize::BITS
            )));
        }
        if bucket_capacity == 0 {
            return Err(PyValueError::new_err(
                "bucket_capacity must be greater than 0",
            ));
        }

        // The hasher works on whole SIMD blocks, queries are zero-padded to match
        let plane_dim = dim.div_ceil(SIMD_LANECOUNT) * SIMD_LANECOUNT;
        let params = EngineStarterParams::new(
            num_hash,
            bucket_capacity,
            plane_dim,
            NodeId {
                internal: starting_node,
            },
            seed,
            enabled_catapults,
        );

        Ok(Self {
            inner: InternalStarter::new(params),
            num_hash,
            dim,
        })
    }

    /// Select starting points for a query.
    ///
    /// Args:
    ///     query: Query vector as a list of f32 values
    ///
    /// Returns:
    ///     The StartingPoints for the query's LSH bucket
    ///
    /// Raises:
    ///     ValueError: If the query does not have the configured dimension
    pub fn select_starting_points(&self, query: VecPy) -> PyResult<StartingPoints> {
        if query.inner.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "query has dimension {}, expected {}",
                query.inner.len(),
                self.dim
            )));
        }

        let query_aligned = AlignedBlock::allocate_padded(query.inner);
        let points = self.inner.select_starting_points(&query_aligned);

        Ok(StartingPoints {
            signature: points.signature,
            catapults: points.catapults.into_iter().map(|c| c.internal).collect(),
            starting_node: points.starting_node.internal,
        })
    }

    /// Cache a node as a catapult in the given LSH bucket.
    ///
    /// Args:
    ///     signature: LSH signature (bucket index), as returned by select_starting_points
    ///     node: Node ID to cache
    ///
    /// Raises:
    ///     ValueError: If signature is not a valid bucket index
    pub fn new_catapult(&self, signature: usize, node: usize) -> PyResult<()> {
        let num_buckets = 1usize << self.num_hash;
        if signature >= num_buckets {
            return Err(PyValueError::new_err(format!(
                "signature {} out of range, there are {} buckets",
                signature, num_buckets
            )));
        }

        self.inner
            .new_catapult(signature, NodeId { internal: node });
        Ok(())
    }

    /// Remove all cached catapults from every bucket.
    pub fn clear_all_catapults(&self) {
        self.inner.clear_all_catapults();
    }

    /// The default starting node.
    #[getter]
    pub fn starting_node(&self) -> usize {
        self.inner.starting_node().internal
    }
}
