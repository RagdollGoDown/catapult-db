use std::sync::RwLock;

use crate::search::NodeId;
use crate::sets::catapults::CatapultEvictionPolicy;
use crate::{numerics::AlignedBlock, search::hash_start::hyperplane_hasher::SimilarityHasher};

/// Manages LSH-based catapult storage and starting point selection for graph searches.
///
/// Uses locality-sensitive hashing to map query vectors to buckets of cached starting
/// points (catapults) from previous successful searches. Each bucket is a thread-safe
/// evicting structure that stores node indices discovered by similar queries.
pub struct EngineStarter<T: CatapultEvictionPolicy> {
    hasher: SimilarityHasher,
    starting_node: NodeId,
    catapults: Box<[RwLock<T>]>,
    enabled_catapults: bool,
}

/// The result of starting point selection, containing the LSH signature and catapults.
pub struct StartingPoints<K = NodeId> {
    /// The LSH signature (bucket index) for the query
    pub signature: usize,

    /// Catapult entries retrieved from the LSH bucket (may be empty)
    pub catapults: Vec<K>,

    /// The base starting node that is always included
    pub starting_node: NodeId,
}

/// Configuration parameters for creating an `EngineStarter`.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct EngineStarterParams {
    /// Number of LSH hash bits (determines 2^num_hash buckets)
    pub num_hash: usize,

    /// Capacity of each bucket in the LSH table
    pub bucket_capacity: usize,

    /// Dimension of input vectors in f32 elements
    pub plane_dim: usize,

    /// Default starting node index to always include
    pub starting_node: NodeId,

    /// Random seed for deterministic LSH hyperplane generation
    pub seed: u64,

    /// Whether to enable catapult lookups (if false, only returns starting_node)
    pub enabled_catapults: bool,
}

impl EngineStarterParams {
    /// Creates a new parameter configuration for an `EngineStarter`.
    ///
    /// # Arguments
    /// * `num_hash` - Number of LSH hash bits (creates 2^num_hash buckets)
    /// * `plane_dim` - Vector dimension in f32 elements
    /// * `starting_node` - Default starting node index
    /// * `seed` - Random seed for LSH hyperplane generation
    /// * `enabled_catapults` - Whether to enable catapult lookups
    ///
    /// # Returns
    /// A new `EngineStarterParams` instance
    pub fn new(
        num_hash: usize,
        bucket_capacity: usize,
        plane_dim: usize,
        starting_node: NodeId,
        seed: u64,
        enabled_catapults: bool,
    ) -> Self {
        Self {
            num_hash,
            bucket_capacity,
            plane_dim,
            starting_node,
            seed,
            enabled_catapults,
        }
    }
}

impl<T> EngineStarter<T>
where
    T: CatapultEvictionPolicy,
{
    /// Creates a new `EngineStarter` with the specified parameters.
    ///
    /// Initializes the LSH hasher and creates 2^num_hash empty catapult buckets,
    /// each protected by an RwLock for thread-safe concurrent access.
    ///
    /// # Arguments
    /// * `params` - Configuration parameters
    ///
    /// # Returns
    /// A new `EngineStarter` instance ready for starting point selection
    pub fn new(params: EngineStarterParams) -> Self {
        let num_hash = params.num_hash;
        let plane_dim = params.plane_dim;
        let starting_node = params.starting_node;
        let seed = params.seed;
        let enabled_catapults = params.enabled_catapults;

        let hasher = SimilarityHasher::new_seeded(num_hash, plane_dim, seed);

        let amount_of_catapult_sets = 1 << num_hash;
        let mut catapult_vecs = Vec::with_capacity(amount_of_catapult_sets);
        for _ in 0..amount_of_catapult_sets {
            catapult_vecs.push(RwLock::new(T::new(params.bucket_capacity)));
        }

        Self {
            hasher,
            starting_node,
            catapults: catapult_vecs.into_boxed_slice(),
            enabled_catapults,
        }
    }

    /// Selects starting points for a query by hashing it to a catapult bucket.
    ///
    /// Computes the LSH signature for the query and retrieves cached catapults from
    /// the corresponding bucket. The base starting node is always included. If catapults
    /// are disabled, an empty catapults vector is returned.
    ///
    /// # Arguments
    /// * `query` - The query vector as aligned blocks
    ///
    /// # Returns
    /// A `StartingPoints` struct containing the signature, catapults, and starting node
    pub fn select_starting_points(&self, query: &[AlignedBlock]) -> StartingPoints<T::Item> {
        let signature = self.hasher.hash_int(query);
        let catapults = if self.enabled_catapults {
            self.catapults[signature].read().unwrap().to_vec()
        } else {
            vec![]
        };
        StartingPoints {
            signature,
            catapults,
            starting_node: self.starting_node,
        }
    }

    /// Records a new catapult for a specific LSH signature bucket.
    ///
    /// This is typically called after a successful search to cache the best result
    /// as a starting point for future queries with the same signature.
    ///
    /// # Arguments
    /// * `signature` - The LSH signature (bucket index) to insert into
    /// * `new_cata` - The entry to cache as a catapult
    pub fn new_catapult(&self, signature: usize, new_cata: T::Item) {
        self.catapults[signature].write().unwrap().insert(new_cata);
    }

    /// Returns the catapults currently cached in one LSH signature bucket.
    ///
    /// Unlike `select_starting_points`, this ignores `enabled_catapults`.
    ///
    /// # Arguments
    /// * `signature` - The LSH signature (bucket index) to read
    pub fn catapults_in_bucket(&self, signature: usize) -> Vec<T::Item> {
        self.catapults[signature].read().unwrap().to_vec()
    }

    /// Number of LSH signature buckets (2^num_hash).
    pub fn num_buckets(&self) -> usize {
        self.catapults.len()
    }

    /// Clears all cached catapults from all buckets.
    ///
    /// This is useful for benchmarking to measure performance without cached starting
    /// points, or to reset state between different workloads.
    pub fn clear_all_catapults(&self) {
        for catapult_set in self.catapults.iter() {
            catapult_set.write().unwrap().clear();
        }
    }

    pub fn starting_node(&self) -> NodeId {
        self.starting_node
    }

    /// Replaces the default starting node returned with every selection.
    pub fn set_starting_node(&mut self, starting_node: NodeId) {
        self.starting_node = starting_node;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{numerics::SIMD_LANECOUNT, sets::catapults::LruSet};

    type TestEngineStarter = EngineStarter<LruSet>;

    const DEFAULT_NUM_HASH: usize = 8;
    const DEFAULT_BUCKET_CAP: usize = 40;
    const DEFAULT_STARTING_NODE: usize = 1000;
    const DEFAULT_SEED: u64 = 42;

    /// Helper to create test parameters with common defaults
    fn default_params() -> EngineStarterParams {
        EngineStarterParams::new(
            DEFAULT_NUM_HASH,
            DEFAULT_BUCKET_CAP,
            SIMD_LANECOUNT,
            NodeId {
                internal: DEFAULT_STARTING_NODE,
            },
            DEFAULT_SEED,
            true,
        )
    }

    /// Helper to create test parameters with a custom seed
    fn params_with_seed(seed: u64) -> EngineStarterParams {
        EngineStarterParams::new(
            DEFAULT_NUM_HASH,
            DEFAULT_BUCKET_CAP,
            SIMD_LANECOUNT,
            NodeId {
                internal: DEFAULT_STARTING_NODE,
            },
            seed,
            true,
        )
    }

    /// Helper to create test parameters with custom dimensions
    fn params_with_dims(plane_dim: usize) -> EngineStarterParams {
        EngineStarterParams::new(
            DEFAULT_NUM_HASH,
            DEFAULT_BUCKET_CAP,
            plane_dim,
            NodeId {
                internal: DEFAULT_STARTING_NODE,
            },
            DEFAULT_SEED,
            true,
        )
    }

    /// Helper to create a simple single-block query
    fn create_test_query(value: f32) -> Vec<AlignedBlock> {
        vec![AlignedBlock::new([value; SIMD_LANECOUNT])]
    }

    /// Helper to create a starter and get signature for a query
    fn get_signature_for_query(starter: &TestEngineStarter, query: &[AlignedBlock]) -> usize {
        starter.select_starting_points(query).signature
    }

    /// Helper to check if a node is in the starting points (either catapults or starting_node)
    fn contains_node(result: &StartingPoints, node: NodeId) -> bool {
        result.catapults.contains(&node) || result.starting_node == node
    }

    /// Helper to assert starting node is present
    fn assert_contains_starting_node(result: &StartingPoints) {
        assert_eq!(
            result.starting_node,
            NodeId {
                internal: DEFAULT_STARTING_NODE
            }
        );
    }

    #[test]
    fn test_new_with_seed() {
        let starter = TestEngineStarter::new(default_params());
        assert_eq!(
            starter.starting_node,
            NodeId {
                internal: DEFAULT_STARTING_NODE
            }
        );
        assert_eq!(starter.catapults.len(), 1 << DEFAULT_NUM_HASH);
    }

    #[test]
    fn test_new_creates_correct_catapult_count() {
        let starter = TestEngineStarter::new(default_params());
        assert_eq!(
            starter.starting_node,
            NodeId {
                internal: DEFAULT_STARTING_NODE
            }
        );
        assert_eq!(starter.catapults.len(), 1 << DEFAULT_NUM_HASH);
    }

    #[test]
    fn test_select_starting_points_includes_starting_node() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(1.0);
        let result = starter.select_starting_points(&query);
        assert_contains_starting_node(&result);
    }

    #[test]
    fn test_insert_and_retrieve_catapult() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(1.0);
        let signature = get_signature_for_query(&starter, &query);

        starter.new_catapult(signature, NodeId { internal: 42 });

        let result = starter.select_starting_points(&query);
        assert!(contains_node(&result, NodeId { internal: 42 }));
        assert_contains_starting_node(&result);

        starter.clear_all_catapults();
        let result = starter.select_starting_points(&query);
        assert!(!contains_node(&result, NodeId { internal: 42 }));
        assert_contains_starting_node(&result);
    }

    #[test]
    fn test_catapult_persists_across_queries() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(1.0);
        let signature = get_signature_for_query(&starter, &query);
        starter.new_catapult(signature, NodeId { internal: 99 });

        for _ in 0..3 {
            let result = starter.select_starting_points(&query);
            assert!(contains_node(&result, NodeId { internal: 99 }));
        }
    }

    #[test]
    fn test_same_query_returns_consistent_signature() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(1.0);

        let result1 = starter.select_starting_points(&query);
        let result2 = starter.select_starting_points(&query);

        assert_eq!(result1.signature, result2.signature);
        assert_eq!(result1.catapults, result2.catapults);
        assert_eq!(result1.starting_node, result2.starting_node);
        assert_eq!(result1.starting_node, starter.starting_node());
    }

    #[test]
    fn test_different_queries_different_signatures() {
        let starter = TestEngineStarter::new(default_params());
        let query1 = create_test_query(1.0);
        let query2 = create_test_query(-1.0);

        let result1 = starter.select_starting_points(&query1);
        let result2 = starter.select_starting_points(&query2);

        assert_ne!(result1.signature, result2.signature);
    }
    #[test]
    fn test_determinism_with_seed() {
        let starter1 = TestEngineStarter::new(params_with_seed(42));
        let starter2 = TestEngineStarter::new(params_with_seed(42));
        let query = create_test_query(1.5);

        let sig1 = get_signature_for_query(&starter1, &query);
        let sig2 = get_signature_for_query(&starter2, &query);

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_different_seeds_different_signatures() {
        let starter1 = TestEngineStarter::new(params_with_seed(42));
        let starter2 = TestEngineStarter::new(params_with_seed(99));
        let query = create_test_query(1.0);

        let sig1 = get_signature_for_query(&starter1, &query);
        let sig2 = get_signature_for_query(&starter2, &query);

        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_multiple_catapults_same_signature() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(1.0);
        let signature = get_signature_for_query(&starter, &query);

        starter.new_catapult(signature, NodeId { internal: 100 });
        starter.new_catapult(signature, NodeId { internal: 200 });
        starter.new_catapult(signature, NodeId { internal: 300 });

        let result = starter.select_starting_points(&query);

        assert!(contains_node(&result, NodeId { internal: 100 }));
        assert!(contains_node(&result, NodeId { internal: 200 }));
        assert!(contains_node(&result, NodeId { internal: 300 }));
        assert_contains_starting_node(&result);
    }

    #[test]
    fn test_different_signatures_independent_catapults() {
        let starter = TestEngineStarter::new(default_params());
        let query1 = create_test_query(1.0);
        let query2 = create_test_query(-1.0);

        let sig1 = get_signature_for_query(&starter, &query1);
        let sig2 = get_signature_for_query(&starter, &query2);

        starter.new_catapult(sig1, NodeId { internal: 111 });
        starter.new_catapult(sig2, NodeId { internal: 222 });

        let result1 = starter.select_starting_points(&query1);
        let result2 = starter.select_starting_points(&query2);

        assert!(contains_node(&result1, NodeId { internal: 111 }));
        assert!(!contains_node(&result1, NodeId { internal: 222 }));
        assert!(contains_node(&result2, NodeId { internal: 222 }));
        assert!(!contains_node(&result2, NodeId { internal: 111 }));
    }

    #[test]
    fn test_signature_within_bounds() {
        let starter = TestEngineStarter::new(default_params());

        for i in 0..10 {
            let query = create_test_query(i as f32);
            let signature = get_signature_for_query(&starter, &query);
            assert!(signature < (1 << DEFAULT_NUM_HASH));
        }
    }

    #[test]
    fn test_multidimensional_query() {
        let starter = TestEngineStarter::new(params_with_dims(SIMD_LANECOUNT * 4));
        let query = vec![
            AlignedBlock::new([1.0; SIMD_LANECOUNT]),
            AlignedBlock::new([2.0; SIMD_LANECOUNT]),
            AlignedBlock::new([3.0; SIMD_LANECOUNT]),
            AlignedBlock::new([4.0; SIMD_LANECOUNT]),
        ];

        let result = starter.select_starting_points(&query);

        assert!(result.signature < (1 << DEFAULT_NUM_HASH));
        assert_contains_starting_node(&result);
    }

    #[test]
    fn test_consistency_across_multiple_calls() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(3.0);

        let result1 = starter.select_starting_points(&query);
        let result2 = starter.select_starting_points(&query);
        let result3 = starter.select_starting_points(&query);

        assert_eq!(result1.signature, result2.signature);
        assert_eq!(result2.signature, result3.signature);
        assert_eq!(result1.catapults, result2.catapults);
        assert_eq!(result1.starting_node, result2.starting_node);
        assert_eq!(result2.catapults, result3.catapults);
        assert_eq!(result2.starting_node, result3.starting_node);
    }

    #[test]
    fn test_fifo_eviction_behavior() {
        let starter = TestEngineStarter::new(default_params());
        let query = create_test_query(1.0);
        let signature = get_signature_for_query(&starter, &query);

        // Insert more than FifoSet capacity (30 items)
        for i in 0..45 {
            starter.new_catapult(signature, NodeId { internal: i });
        }

        let result = starter.select_starting_points(&query);

        // Should have at most 30 catapults
        assert!(result.catapults.len() <= 40);
        assert_contains_starting_node(&result);

        // Oldest entries should be evicted (0-4 should be gone)
        for i in 0..5 {
            assert!(!contains_node(&result, NodeId { internal: i }));
        }

        // Newest entries should be present (30-34)
        for i in 40..45 {
            assert!(contains_node(&result, NodeId { internal: i }));
        }
    }

    #[test]
    fn test_empty_catapult_only_returns_starting_node() {
        let custom_starting = 999;
        let params = EngineStarterParams::new(
            DEFAULT_NUM_HASH,
            DEFAULT_BUCKET_CAP,
            SIMD_LANECOUNT,
            NodeId {
                internal: custom_starting,
            },
            DEFAULT_SEED,
            true,
        );
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(5.5);

        let result = starter.select_starting_points(&query);

        assert_eq!(result.catapults.len(), 0);
        assert_eq!(
            result.starting_node,
            NodeId {
                internal: custom_starting
            }
        );
    }
}
