use std::collections::{HashMap, HashSet};

use catapult::{
    numerics::AlignedBlock,
    search::{
        AdjacencyGraph, NodeId, SearchStrategy, VamanaParams,
        hash_start::{EngineStarter as InternalStarter, EngineStarterParams},
    },
    sets::{catapults::LruSet, editable::FlatEditableSet},
    statistics::Stats,
};
use pyo3::{
    Bound, FromPyObject, PyAny, PyResult, buffer::PyBuffer, exceptions::PyValueError, pyclass,
    pymethods, types::PyAnyMethods,
};

use crate::engine_starter::validate_starter_params;

type EditableGraph = AdjacencyGraph<LruSet, FlatEditableSet>;

/// A float vector read from Python.
///
/// Accepts any object exposing a one-dimensional float32 buffer (e.g. a float32 NumPy
/// array, copied once) or any sequence of numbers (e.g. a list or a float64 array).
pub struct FloatVec {
    pub inner: Vec<f32>,
}

impl<'a> FromPyObject<'a> for FloatVec {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(buffer) = PyBuffer::<f32>::get(ob) {
            if buffer.dimensions() != 1 {
                return Err(PyValueError::new_err(format!(
                    "expected a 1-dimensional vector, got {} dimensions",
                    buffer.dimensions()
                )));
            }
            return Ok(Self {
                inner: buffer.to_vec(ob.py())?,
            });
        }
        Ok(Self {
            inner: ob.extract()?,
        })
    }
}

/// Work counters of a VamanaIndex, summed over several operations.
///
/// Attributes:
///     beam_calls: Number of beam searches run
///     nodes_visited: Number of nodes expanded by those searches
///     dists_computed: Number of distance computations (vector comparisons)
///     searches_with_catapults: Searches whose results descend from a catapult
#[pyclass(frozen, get_all)]
pub struct IndexStats {
    beam_calls: usize,
    nodes_visited: usize,
    dists_computed: usize,
    searches_with_catapults: usize,
}

impl From<&Stats> for IndexStats {
    fn from(stats: &Stats) -> Self {
        Self {
            beam_calls: stats.get_beam_calls(),
            nodes_visited: stats.get_nodes_visited(),
            dists_computed: stats.get_computed_dists(),
            searches_with_catapults: stats.get_searches_with_catapults(),
        }
    }
}

#[pymethods]
impl IndexStats {
    fn __repr__(&self) -> String {
        format!(
            "IndexStats(beam_calls={}, nodes_visited={}, dists_computed={}, searches_with_catapults={})",
            self.beam_calls, self.nodes_visited, self.dists_computed, self.searches_with_catapults
        )
    }
}

/// Result of one VamanaIndex search.
///
/// Attributes:
///     keys: Keys of the nearest vectors found, closest first
///     distances: Their squared L2 distances to the query
///     nodes_visited: Number of nodes this search expanded
///     dists_computed: Number of distance computations (vector comparisons) it made
///     used_catapults: Whether a result descends from a catapult starting point
#[pyclass(frozen, get_all)]
pub struct SearchResult {
    keys: Vec<String>,
    distances: Vec<f32>,
    nodes_visited: usize,
    dists_computed: usize,
    used_catapults: bool,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(keys={:?}, distances={:?}, nodes_visited={}, dists_computed={}, used_catapults={})",
            self.keys, self.distances, self.nodes_visited, self.dists_computed, self.used_catapults
        )
    }

    fn __len__(&self) -> usize {
        self.keys.len()
    }
}

/// An in-memory Vamana graph (DiskANN) of vectors addressed by string keys.
///
/// Vectors are added with `upsert` or `build` and removed with `remove`, which deletes
/// lazily as in FreshDiskANN: removed vectors are no longer returned, but stay in the
/// graph until `consolidate` rewires the edges around them. Node ids are never reused,
/// so removed vectors keep their memory.
///
/// Distances are squared L2. For unit vectors, cosine similarity is `1 - d / 2`.
///
/// Searches and graph updates count their work in two separate IndexStats,
/// `search_stats` and `update_stats`.
///
/// Like AdjacencyGraph, instances must not be shared between Python threads.
#[pyclass(unsendable)]
pub struct VamanaIndex {
    graph: EditableGraph,
    params: VamanaParams,
    starter_params: EngineStarterParams,
    dim: usize,
    node_by_key: HashMap<String, NodeId>,
    /// The key of every node, `None` once removed.
    key_by_node: Vec<Option<String>>,
    pending_deletes: usize,
    search_stats: Stats,
    update_stats: Stats,
}

impl VamanaIndex {
    fn empty_graph(starter_params: EngineStarterParams) -> EditableGraph {
        let strategy = if starter_params.enabled_catapults {
            SearchStrategy::Catapult
        } else {
            SearchStrategy::Vanilla
        };
        AdjacencyGraph::new_flat(vec![], InternalStarter::new(starter_params), strategy)
    }

    /// Pads `vector` into SIMD blocks after checking its dimension.
    fn aligned(&self, vector: FloatVec, what: &str) -> PyResult<Box<[AlignedBlock]>> {
        if vector.inner.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "{what} has dimension {}, expected {}",
                vector.inner.len(),
                self.dim
            )));
        }
        Ok(AlignedBlock::allocate_padded(vector.inner).into_boxed_slice())
    }

    fn record_nodes(&mut self, keys: Vec<String>, ids: &[NodeId]) {
        for (key, &id) in keys.into_iter().zip(ids) {
            debug_assert_eq!(id.internal, self.key_by_node.len());
            self.key_by_node.push(Some(key.clone()));
            self.node_by_key.insert(key, id);
        }
    }
}

#[pymethods]
impl VamanaIndex {
    /// Create an empty index.
    ///
    /// Args:
    ///     dim: Dimension of the vectors
    ///     max_degree: Target out-degree R of the graph (default: 32)
    ///     build_beam_width: Beam width L of the searches run to insert a vector (default: 64)
    ///     alpha: Pruning factor, at least 1; larger keeps more long edges (default: 1.2)
    ///     catapults_enabled: Whether searches use and cache LSH catapults (default: False)
    ///     num_hash: Number of LSH hash bits for catapults (default: 8)
    ///     bucket_capacity: Maximum number of catapults per LSH bucket (default: 40)
    ///     seed: Random seed for LSH hyperplane generation (default: 42)
    ///
    /// Raises:
    ///     ValueError: If a parameter is out of range
    #[new]
    #[pyo3(signature = (
        dim,
        max_degree=32,
        build_beam_width=64,
        alpha=1.2,
        catapults_enabled=false,
        num_hash=8,
        bucket_capacity=40,
        seed=42,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
        catapults_enabled: bool,
        num_hash: usize,
        bucket_capacity: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be greater than 0"));
        }
        if max_degree == 0 || build_beam_width == 0 {
            return Err(PyValueError::new_err(
                "max_degree and build_beam_width must be greater than 0",
            ));
        }
        if alpha.is_nan() || alpha < 1.0 {
            return Err(PyValueError::new_err("alpha must be at least 1"));
        }
        let plane_dim = validate_starter_params(num_hash, bucket_capacity, dim)?;
        let starter_params = EngineStarterParams::new(
            num_hash,
            bucket_capacity,
            plane_dim,
            NodeId { internal: 0 },
            seed,
            catapults_enabled,
        );

        Ok(Self {
            graph: Self::empty_graph(starter_params),
            params: VamanaParams::new(max_degree, build_beam_width, alpha),
            starter_params,
            dim,
            node_by_key: HashMap::new(),
            key_by_node: Vec::new(),
            pending_deletes: 0,
            search_stats: Stats::new(),
            update_stats: Stats::new(),
        })
    }

    /// Insert a vector, replacing the vector stored under `key` if there is one.
    ///
    /// A replaced vector is removed as by `remove` and the new one inserted as a new node.
    ///
    /// Raises:
    ///     ValueError: If the vector does not have the index's dimension
    pub fn upsert(&mut self, key: String, vector: FloatVec) -> PyResult<()> {
        let payload = self.aligned(vector, "vector")?;
        self.remove(&key);
        let id = self
            .graph
            .insert(payload, &self.params, &mut self.update_stats);
        self.record_nodes(vec![key], &[id]);
        Ok(())
    }

    /// Add many new vectors at once, as DiskANN's in-memory build does.
    ///
    /// Faster than one `upsert` per vector. If the index holds no vectors, the search
    /// start becomes the medoid of the new ones. Vectors are linked in the given order,
    /// so shuffle them first if they come sorted or clustered.
    ///
    /// Args:
    ///     keys: Keys of the new vectors, all distinct and not in the index
    ///     vectors: One vector per key (e.g. a 2-D float32 NumPy array)
    ///
    /// Raises:
    ///     ValueError: If a key is repeated or already present, the lengths differ, or
    ///         a vector does not have the index's dimension
    pub fn build(&mut self, keys: Vec<String>, vectors: Vec<FloatVec>) -> PyResult<()> {
        if keys.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "got {} keys but {} vectors",
                keys.len(),
                vectors.len()
            )));
        }
        let mut seen = HashSet::new();
        for key in &keys {
            if self.node_by_key.contains_key(key) || !seen.insert(key) {
                return Err(PyValueError::new_err(format!(
                    "key {key:?} is repeated or already in the index"
                )));
            }
        }
        let payloads = vectors
            .into_iter()
            .map(|vector| self.aligned(vector, "vector"))
            .collect::<PyResult<Vec<_>>>()?;

        let ids = self
            .graph
            .build(payloads, &self.params, &mut self.update_stats);
        self.record_nodes(keys, &ids);
        Ok(())
    }

    /// Remove the vector stored under `key`.
    ///
    /// The vector is no longer returned by searches; its edges stay in the graph until
    /// `consolidate` runs.
    ///
    /// Returns:
    ///     False if `key` was not in the index
    pub fn remove(&mut self, key: &str) -> bool {
        let Some(id) = self.node_by_key.remove(key) else {
            return false;
        };
        self.key_by_node[id.internal] = None;
        self.graph.delete(id, &self.params, &mut self.update_stats);
        self.pending_deletes += 1;
        true
    }

    /// Rewire the graph's edges around removed vectors and disconnect them.
    ///
    /// Searches stay correct without it, but get slower as removed vectors pile up.
    pub fn consolidate(&mut self) {
        if self.pending_deletes == 0 {
            return;
        }
        self.graph
            .consolidate_deletes(&self.params, &mut self.update_stats);
        self.pending_deletes = 0;
    }

    /// Find the vectors nearest to `query`.
    ///
    /// Args:
    ///     query: Query vector (e.g. a 1-D float32 NumPy array)
    ///     k: Number of nearest vectors to return (default: 1)
    ///     beam_width: Width of the search beam, at least k (default: 32)
    ///
    /// Returns:
    ///     A SearchResult with at most k keys; fewer if the index holds fewer vectors
    ///
    /// Raises:
    ///     ValueError: If the query does not have the index's dimension, k is 0, or
    ///         beam_width is smaller than k
    #[pyo3(signature = (query, k=1, beam_width=32))]
    pub fn search(
        &mut self,
        query: FloatVec,
        k: usize,
        beam_width: usize,
    ) -> PyResult<SearchResult> {
        if k == 0 || beam_width < k {
            return Err(PyValueError::new_err(
                "k must be positive and beam_width at least k",
            ));
        }
        let query = self.aligned(query, "query")?;

        let mut stats = Stats::new();
        let found = self.graph.beam_search(&query, k, beam_width, &mut stats);
        self.search_stats = self.search_stats.merge(&stats);

        Ok(SearchResult {
            keys: found
                .iter()
                .map(|entry| {
                    self.key_by_node[entry.index.internal]
                        .clone()
                        .expect("search returned a removed node")
                })
                .collect(),
            distances: found.iter().map(|entry| entry.distance.0).collect(),
            nodes_visited: stats.get_nodes_visited(),
            dists_computed: stats.get_computed_dists(),
            used_catapults: found.iter().any(|entry| entry.has_catapult_ancestor),
        })
    }

    /// Remove every vector and catapult, keeping the parameters. Stats are kept.
    pub fn clear(&mut self) {
        self.graph = Self::empty_graph(self.starter_params);
        self.node_by_key.clear();
        self.key_by_node.clear();
        self.pending_deletes = 0;
    }

    /// Remove all cached catapults from every LSH bucket.
    pub fn clear_catapults(&self) {
        self.graph.clear_all_catapults();
    }

    /// Work done by all searches since creation or the last `reset_stats`.
    #[getter]
    pub fn search_stats(&self) -> IndexStats {
        IndexStats::from(&self.search_stats)
    }

    /// Work done by upserts, builds, removes and consolidations since creation or the
    /// last `reset_stats`.
    #[getter]
    pub fn update_stats(&self) -> IndexStats {
        IndexStats::from(&self.update_stats)
    }

    /// Zero both `search_stats` and `update_stats`.
    pub fn reset_stats(&mut self) {
        self.search_stats = Stats::new();
        self.update_stats = Stats::new();
    }

    /// Keys of the vectors in the index, in no particular order.
    pub fn keys(&self) -> Vec<String> {
        self.node_by_key.keys().cloned().collect()
    }

    /// Dimension of the vectors.
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of graph nodes, removed ones included.
    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.graph.len()
    }

    /// Number of vectors removed since the last `consolidate`.
    #[getter]
    pub fn pending_deletes(&self) -> usize {
        self.pending_deletes
    }

    /// Number of vectors in the index (removed ones excluded).
    pub fn __len__(&self) -> usize {
        self.node_by_key.len()
    }

    pub fn __contains__(&self, key: &str) -> bool {
        self.node_by_key.contains_key(key)
    }

    fn __repr__(&self) -> String {
        format!(
            "VamanaIndex(dim={}, len={}, num_nodes={}, max_degree={}, alpha={})",
            self.dim,
            self.node_by_key.len(),
            self.graph.len(),
            self.params.max_degree,
            self.params.alpha
        )
    }
}
