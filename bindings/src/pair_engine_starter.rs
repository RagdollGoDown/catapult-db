use catapult::{
    search::{
        NodeId,
        hash_start::{EngineStarter as InternalStarter, EngineStarterParams},
    },
    sets::catapults::LruSet,
};
use pyo3::{PyResult, pyclass, pymethods};

use crate::{
    engine_starter::{aligned_query, check_signature, validate_starter_params},
    vecpy::VecPy,
};

/// Bucket contents selected for a query.
///
/// Attributes:
///     signature: LSH signature (bucket index) of the query
///     catapults: (key, value) pairs cached in the query's bucket (may be empty)
#[pyclass(frozen, get_all)]
pub struct PairStartingPoints {
    signature: usize,
    catapults: Vec<(usize, usize)>,
}

#[pymethods]
impl PairStartingPoints {
    fn __repr__(&self) -> String {
        format!(
            "PairStartingPoints(signature={}, catapults={:?})",
            self.signature, self.catapults
        )
    }
}

/// Catapult EngineStarter whose buckets store (key, value) integer pairs.
///
/// Uses the same LSH hashing and per-bucket LRU eviction as EngineStarter, but
/// each entry is a pair instead of a single node ID, so callers can keep all
/// their per-entry state in the buckets. Inserting an identical pair refreshes
/// it instead of storing it twice.
#[pyclass]
pub struct PairEngineStarter {
    inner: InternalStarter<LruSet<(usize, usize)>>,
    num_hash: usize,
    dim: usize,
}

#[pymethods]
impl PairEngineStarter {
    /// Create a new pair engine starter.
    ///
    /// Args:
    ///     num_hash: Number of LSH hash bits (creates 2^num_hash buckets)
    ///     bucket_capacity: Maximum number of pairs stored per bucket
    ///     dim: Dimension of the query vectors
    ///     seed: Random seed for LSH hyperplane generation (default: 42)
    ///
    /// Raises:
    ///     ValueError: If num_hash or bucket_capacity is out of range
    #[new]
    #[pyo3(signature = (num_hash, bucket_capacity, dim, seed=42))]
    pub fn new(num_hash: usize, bucket_capacity: usize, dim: usize, seed: u64) -> PyResult<Self> {
        let plane_dim = validate_starter_params(num_hash, bucket_capacity, dim)?;
        // Pairs carry no graph, so the default starting node is never used
        let params = EngineStarterParams::new(
            num_hash,
            bucket_capacity,
            plane_dim,
            NodeId { internal: 0 },
            seed,
            true,
        );

        Ok(Self {
            inner: InternalStarter::new(params),
            num_hash,
            dim,
        })
    }

    /// Select the bucket contents for a query.
    ///
    /// Args:
    ///     query: Query vector as a list of f32 values
    ///
    /// Returns:
    ///     The PairStartingPoints for the query's LSH bucket
    ///
    /// Raises:
    ///     ValueError: If the query does not have the configured dimension
    pub fn select_starting_points(&self, query: VecPy) -> PyResult<PairStartingPoints> {
        let query_aligned = aligned_query(query, self.dim)?;
        let points = self.inner.select_starting_points(&query_aligned);

        Ok(PairStartingPoints {
            signature: points.signature,
            catapults: points.catapults,
        })
    }

    /// Cache a (key, value) pair in the given LSH bucket.
    ///
    /// Args:
    ///     signature: LSH signature (bucket index), as returned by select_starting_points
    ///     key: First element of the pair
    ///     value: Second element of the pair
    ///
    /// Raises:
    ///     ValueError: If signature is not a valid bucket index
    pub fn new_catapult(&self, signature: usize, key: usize, value: usize) -> PyResult<()> {
        check_signature(signature, self.num_hash)?;
        self.inner.new_catapult(signature, (key, value));
        Ok(())
    }

    /// Pairs cached in one LSH bucket, oldest first.
    ///
    /// Raises:
    ///     ValueError: If signature is not a valid bucket index
    pub fn catapults_in_bucket(&self, signature: usize) -> PyResult<Vec<(usize, usize)>> {
        check_signature(signature, self.num_hash)?;
        Ok(self.inner.catapults_in_bucket(signature))
    }

    /// All cached pairs, bucket by bucket.
    pub fn entries(&self) -> Vec<(usize, usize)> {
        (0..self.inner.num_buckets())
            .flat_map(|signature| self.inner.catapults_in_bucket(signature))
            .collect()
    }

    /// Remove all cached pairs from every bucket.
    pub fn clear_all_catapults(&self) {
        self.inner.clear_all_catapults();
    }

    /// Number of LSH buckets (2^num_hash).
    #[getter]
    pub fn num_buckets(&self) -> usize {
        self.inner.num_buckets()
    }
}
