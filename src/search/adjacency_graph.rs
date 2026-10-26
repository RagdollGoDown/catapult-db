use hashbrown::HashSet;

use crate::{
    numerics::{AlignedBlock, VectorLike},
    search::{
        NodeId, SearchStrategy,
        hash_start::{EngineStarter, StartingPoints},
        node::Node,
    },
    sets::{
        NeighborSet,
        candidates::{CandidateEntry, SmallestKCandidates},
        catapults::CatapultEvictionPolicy,
        editable::FlatEditableSet,
        fixed::FlatFixedSet,
    },
    statistics::Stats,
};

/// An in-memory proximity graph for approximate nearest neighbor (ANN) search.
///
/// This structure stores a graph where each node contains a vector embedding (payload)
/// and its neighbor connections. The graph supports beam search with optional catapult
/// acceleration, using LSH to map queries to cached starting points from previous searches.
///
/// # Type Parameters
/// * `EvictPolicy` - The eviction strategy for catapult storage (e.g., `FifoSet<30>`)
/// * `N` - The neighbor set type: immutable [`FlatFixedSet`] for graphs loaded from
///   disk (the default), or [`FlatEditableSet`] for graphs built and modified in memory
///
/// # Invariants
/// - `adjacency[i]` represents node `i` in the graph
/// - Each `Node.neighbors` entry is a valid index into `adjacency`
/// - All node payloads have the same dimensionality
///
/// # Core Algorithm
/// The main search method [`beam_search`](Self::beam_search) implements best-first beam search:
/// 1. Maps query to LSH signature and retrieves cached catapults
/// 2. Maintains a candidate set of at most `beam_width` closest nodes
/// 3. Repeatedly expands the best unvisited candidate, adding its neighbors
/// 4. Stops when all candidates in the beam have been visited
/// 5. Caches the best result as a catapult for future similar queries
pub struct AdjacencyGraph<EvictPolicy, N = FlatFixedSet>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
    N: NeighborSet,
{
    adjacency: Vec<Node<N>>,
    starter: EngineStarter<EvictPolicy>,
    strategy: SearchStrategy,
}

impl<EvictPolicy, N> AdjacencyGraph<EvictPolicy, N>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
    N: NeighborSet,
{
    /// Creates a new flat (single-layer) adjacency graph for ANN search.
    ///
    /// # Arguments
    /// * `adj` - Vector of nodes representing the graph, where `adj[i]` is node `i`
    /// * `engine` - LSH-based starting point selector managing catapult buckets
    /// * `catapults` - Configuration for whether catapults are enabled
    ///
    /// # Returns
    /// A new `AdjacencyGraph` instance ready for beam search
    pub fn new_flat(
        adj: Vec<Node<N>>,
        engine: EngineStarter<EvictPolicy>,
        strategy: SearchStrategy,
    ) -> Self {
        Self {
            adjacency: adj,
            starter: engine,
            strategy,
        }
    }
}

impl<EvictPolicy, N> AdjacencyGraph<EvictPolicy, N>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
    N: NeighborSet,
{
    /// Computes distances from the query to a set of node indices.
    ///
    /// Creates candidate entries for each provided index by computing the squared L2
    /// distance from the query to that node's payload.
    ///
    /// # Arguments
    /// * `indices` - Node indices to compute distances for
    /// * `query` - Query vector as aligned blocks
    /// * `catapult_marker` - Whether to mark these candidates as catapult-derived
    /// * `stats` - Statistics tracker to update with distance computations
    ///
    /// # Returns
    /// A vector of candidate entries with computed distances
    fn distances_from_indices(
        &self,
        indices: &[NodeId],
        query: &[AlignedBlock],
        catapult_marker: bool,
        stats: &mut Stats,
    ) -> Vec<CandidateEntry> {
        stats.bump_computed_dists(indices.len());

        indices
            .iter()
            .map(|&index| {
                let starting_point = &self.adjacency[index.internal];
                let starting_score = starting_point.payload.l2_squared(query);

                CandidateEntry {
                    distance: starting_score.into(),
                    index,
                    has_catapult_ancestor: catapult_marker,
                }
            })
            .collect()
    }

    /// Performs a best-first beam search starting from the given candidates.
    ///
    /// This is the core search algorithm that maintains a beam of at most `beam_width`
    /// candidates and iteratively expands the closest unvisited candidate until exhaustion.
    /// The search explores the graph by following neighbor edges, maintaining the k-best
    /// nodes encountered.
    ///
    /// # Algorithm Steps
    /// 1. Initialize candidate beam with starting points
    /// 2. Find the best unvisited candidate in the beam
    /// 3. Expand it by computing distances to all its neighbors
    /// 4. Add neighbors to the beam (with automatic eviction if over capacity)
    /// 5. Mark the expanded node as visited
    /// 6. Repeat from step 2 until no unvisited candidates remain
    /// 7. Return the top-k candidates by distance
    ///
    /// # Arguments
    /// * `query` - Target query vector as aligned blocks
    /// * `starting_candidates` - Initial candidates to seed the search
    /// * `k` - Number of nearest neighbors to return
    /// * `beam_width` - Maximum number of candidates to maintain (must be ≥ k)
    /// * `layer` - layer context for neighbor retrieval (unit type for flat graphs)
    /// * `stats` - Statistics tracker for performance monitoring
    ///
    /// # Returns
    /// A vector of the k nearest candidate entries, sorted by distance
    ///
    /// # Panics
    /// * Panics if `beam_width < k`
    /// * Panics if starting_candidates is empty
    /// * Panics if neighbor indices are out of bounds (graph invariant violation)
    #[tracing::instrument(level = "trace", skip_all, fields(k, beam_width))]
    fn beam_search_raw(
        &self,
        query: &[AlignedBlock],
        starting_candidates: &[CandidateEntry],
        k: usize,
        beam_width: usize,
        stats: &mut Stats,
    ) -> Vec<CandidateEntry> {
        assert!(beam_width >= k);
        stats.bump_beam_calls();

        let mut candidates: SmallestKCandidates = SmallestKCandidates::new(beam_width);
        let mut visited = HashSet::new();

        candidates.insert_batch(starting_candidates);

        // among the suggested entry points, one of them is 'the best'. Let's identify it.
        let initial_best_node = candidates.iter().min().copied().expect(
            "Corrupted starting point entries. Provide a non-empty starting_indices lists.",
        );

        let mut best_candidate: Option<CandidateEntry> = Some(initial_best_node);

        // while we have some node on which to expand (at first, the best LSH entry point),
        // we keep expanding it (i.e. looking at its neighbors for better guesses)
        while let Some(best_candidate_node) = best_candidate {
            let best_candidate_neighs =
                &self.adjacency[best_candidate_node.index.internal].neighbors;
            // identify the neighbors of our current best guess.
            // All of these guys become candidates for expansion. if we have too many candidates
            // (beam width parameter), the `candidates` data structure takes care of removing the
            // worst ones (and the duplicates).
            let neighbors = best_candidate_neighs.as_slice();

            // Record each (src → dst) edge as considered before computing distances.
            if stats.has_adv_tracking() {
                let src = best_candidate_node.index.internal;
                for &dst in neighbors.iter() {
                    stats.record_considered_edge(src, dst.internal);
                }
            }

            let neighbor_distances = self.distances_from_indices(
                neighbors,
                query,
                best_candidate_node.has_catapult_ancestor,
                stats,
            );

            candidates.insert_batch(&neighbor_distances);

            // mark our current node as visited (not to be expanded again)
            visited.insert(best_candidate_node.index);
            stats.bump_nodes_visited();

            // and find some other guy to expand, if possible. If not, we call it a day and return our best guesses.
            best_candidate = candidates
                .iter()
                .filter(|&elem| !visited.contains(&elem.index))
                .min()
                .copied()
        }

        // Post-search: record used edges — (src, dst) where both src and dst were visited
        // in this search. Done once per search to avoid cross-query contamination.
        if stats.has_adv_tracking() {
            for &src in &visited {
                for &dst in self.adjacency[src.internal].neighbors.as_slice() {
                    if visited.contains(&dst) {
                        stats.record_used_edge(src.internal, dst.internal);
                    }
                }
            }
        }

        // we have beam_width neighbors, we only need k so we need to rerank
        let mut candidate_vec = candidates.into_iter().collect::<Vec<_>>();
        candidate_vec.sort(); // note: implicitly relying on CandidateEntry ordering here

        // and return the best k, job done :)
        candidate_vec.into_iter().take(k).collect()
    }
}

impl<EvictPolicy, N> AdjacencyGraph<EvictPolicy, N>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
    N: NeighborSet,
{
    /// Performs approximate k-nearest neighbor search using LSH-accelerated beam search.
    ///
    /// This is the main entry point for querying the graph. It uses LSH to map the query
    /// to a catapult bucket, retrieves cached starting points from similar previous queries,
    /// runs beam search, and caches the best result for future queries.
    ///
    /// # Arguments
    /// * `query` - Query vector as aligned blocks
    /// * `k` - Number of nearest neighbors to return
    /// * `beam_width` - Maximum beam size during search (must be ≥ k)
    /// * `stats` - Statistics tracker for performance monitoring
    ///
    /// # Returns
    /// A vector of the k nearest candidate entries, sorted by ascending distance
    ///
    /// # Behavior
    /// 1. Hashes query to LSH signature
    /// 2. Retrieves catapults from the corresponding bucket + base starting node
    /// 3. Runs beam search from these starting points
    /// 4. Caches the best result as a catapult (if catapults enabled)
    /// 5. Updates statistics with catapult usage
    ///
    /// An empty graph returns no results.
    #[tracing::instrument(level = "trace", skip_all, fields(k, beam_width))]
    pub fn beam_search(
        &self,
        query: &[AlignedBlock],
        k: usize,
        beam_width: usize,
        stats: &mut Stats,
    ) -> Vec<CandidateEntry> {
        if self.adjacency.is_empty() {
            return Vec::new();
        }

        let hash_search = if let SearchStrategy::LshApg(lsh_apg) = &self.strategy {
            let mut lshapg_candidates = Vec::new();
            for candidate_set in lsh_apg
                .iter()
                .map(|zorder| zorder.query_k_closest(query, 4 * k))
            {
                lshapg_candidates.extend(candidate_set.iter());
            }
            lshapg_candidates.sort();
            lshapg_candidates.dedup();
            StartingPoints {
                signature: 0,
                catapults: lshapg_candidates,
                starting_node: self.starter.starting_node(),
            }
        } else {
            self.starter.select_starting_points(query)
        };

        // Convert catapults to candidate entries (marked as having catapult ancestry)
        let mut distances = self.distances_from_indices(&hash_search.catapults, query, true, stats);
        distances.sort();
        distances.shrink_to(k);

        // Add the starting node (not a catapult, so marked as false)
        let starting_node_entry =
            self.distances_from_indices(&[hash_search.starting_node], query, false, stats);
        distances.extend(starting_node_entry);

        let search_results = self.beam_search_raw(query, &distances, k, beam_width, stats);
        let best_result = search_results[0].index;

        if matches!(self.strategy, SearchStrategy::Catapult) {
            self.starter
                .new_catapult(hash_search.signature, best_result);
            if search_results.iter().any(|e| e.has_catapult_ancestor) {
                stats.bump_searches_with_catapults();
            }
        }
        search_results
    }

    /// Clears all cached catapults from all LSH buckets.
    ///
    /// This is useful for benchmarking to measure performance without the benefit
    /// of cached starting points, or to reset state between different workloads.
    pub fn clear_all_catapults(&self) {
        self.starter.clear_all_catapults();
    }

    /// Returns the number of nodes in the graph.
    ///
    /// # Returns
    /// The total number of nodes stored in the adjacency list
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// The out-neighbors of node `id`.
    pub fn neighbors(&self, id: NodeId) -> &[NodeId] {
        self.adjacency[id.internal].neighbors.as_slice()
    }

    /// The vector stored at node `id`.
    pub fn payload(&self, id: NodeId) -> &[AlignedBlock] {
        &self.adjacency[id.internal].payload
    }

    /// The node every search starts from (in addition to any catapults).
    pub fn starting_node(&self) -> NodeId {
        self.starter.starting_node()
    }

    /// Changes the node every search starts from, e.g. to a new medoid.
    pub fn set_starting_node(&mut self, id: NodeId) {
        assert!(
            id.internal < self.adjacency.len(),
            "starting node out of range"
        );
        self.starter.set_starting_node(id);
    }

    /// Returns the total number of directed edges in the graph (sum of all neighbor list lengths). Does not include catapult edges.
    pub fn total_edge_count(&self) -> usize {
        self.adjacency.iter().map(|n| n.neighbors.len()).sum()
    }
}

impl<EvictPolicy> AdjacencyGraph<EvictPolicy, FlatEditableSet>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
{
    /// Appends a node and returns its id, which stays stable for the graph's lifetime.
    ///
    /// The first node pushed into an empty graph becomes the starting node.
    ///
    /// # Arguments
    /// * `payload` - The node's vector, as aligned blocks of the graph's dimension
    /// * `neighbors` - The node's initial out-neighbors (may be empty)
    pub fn push_node(&mut self, payload: Box<[AlignedBlock]>, neighbors: Vec<usize>) -> NodeId {
        let id = NodeId {
            internal: self.adjacency.len(),
        };
        self.adjacency.push(Node {
            neighbors: FlatEditableSet::new(neighbors),
            payload,
        });
        if id.internal == 0 {
            self.starter.set_starting_node(id);
        }
        id
    }

    /// Mutable access to the out-neighbors of node `id`, for graph construction.
    pub fn neighbors_mut(&mut self, id: NodeId) -> &mut FlatEditableSet {
        &mut self.adjacency[id.internal].neighbors
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        numerics::SIMD_LANECOUNT,
        search::{
            SearchStrategy,
            hash_start::{EngineStarter, EngineStarterParams, zorder_index::ZOrderIndex},
        },
        sets::{catapults::LruSet, fixed::FlatFixedSet},
    };

    pub type TestEngineStarter = EngineStarter<LruSet>;

    use super::*;

    // A simple graph:
    // Nodes: 0 (pos 0), 1 (pos 10), 2 (pos 20), 3 (pos 30), 4 (pos 40)
    // Edges: 0 -> 1 -> 2 -> 3 -> 4
    // Query: 11.0
    fn setup_simple_graph(catapults_enabled: bool) -> AdjacencyGraph<LruSet> {
        let nodes = vec![
            // 0: Pos 0.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
            // 1: Pos 10.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 2: Pos 20.0, Neighbors: [1, 3]
            Node {
                payload: vec![AlignedBlock::new([20.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1, 3]),
            },
            // 3: Pos 30.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([30.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![4]),
            },
            // 4: Pos 40.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([40.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
        ];
        let strategy = if catapults_enabled {
            SearchStrategy::Catapult
        } else {
            SearchStrategy::Vanilla
        };
        // Start from node 0
        let params = EngineStarterParams::new(
            4,
            40,
            SIMD_LANECOUNT,
            NodeId { internal: 0 },
            42,
            catapults_enabled,
        );
        AdjacencyGraph::new_flat(nodes, EngineStarter::new(params), strategy)
    }

    #[test]
    fn lshapg() {
        let nodes = vec![
            // 0: Pos 0.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
            // 1: Pos 10.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 2: Pos 20.0, Neighbors: [1, 3]
            Node {
                payload: vec![AlignedBlock::new([20.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1, 3]),
            },
            // 3: Pos 30.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([30.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![4]),
            },
            // 4: Pos 40.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([40.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
        ];
        let strategy = SearchStrategy::LshApg([ZOrderIndex::new(2, 16, 0, 1.0)]);
        // Start from node 0
        let params =
            EngineStarterParams::new(4, 40, SIMD_LANECOUNT, NodeId { internal: 0 }, 42, false);
        let lshapg: AdjacencyGraph<LruSet> =
            AdjacencyGraph::new_flat(nodes, EngineStarter::new(params), strategy);
        let mut stats = Stats::new();

        assert_eq!(
            lshapg.beam_search(
                &[AlignedBlock::new([40.0; SIMD_LANECOUNT])],
                5,
                5,
                &mut stats
            )[0]
            .index,
            NodeId { internal: 4 }
        );
    }

    #[test]
    fn test_basic_search_path() {
        let graph = setup_simple_graph(true);
        let query = vec![AlignedBlock::new([11.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;

        // Distances: 0(121), 1(1), 2(81), 3(361), 4(841)
        let mut stats = crate::statistics::Stats::new();
        stats.enable_adv_tracking();

        let results1 = graph.beam_search(&query, k, beam_width, &mut stats);
        let results2 = graph.beam_search(&query, k, beam_width, &mut stats);

        assert_eq!(stats.get_searches_with_catapults(), 1);

        assert_eq!(
            results1.iter().map(|e| e.index).collect::<Vec<_>>(),
            results2.iter().map(|e| e.index).collect::<Vec<_>>()
        );

        // Final candidates (in order of distance): 1 (1), 2 (81), 0 (121)
        // Top K=2 results: [1, 2]
        assert_eq!(results1.len(), k);
        assert_eq!(results1[0].index.internal, 1);
        assert_eq!(results1[1].index.internal, 2);

        // Verify the graph structure - count total edges
        let total_edges: usize = graph.adjacency.iter().map(|n| n.neighbors.len()).sum();
        assert_eq!(total_edges, graph.total_edge_count());
    }

    #[test]
    fn test_first_query_same_results_with_and_without_catapults() {
        // Test that the first query returns identical results regardless of catapult setting
        // This is expected because catapults only affect subsequent queries after being established
        let graph_with_catapults = setup_simple_graph(true);
        let graph_without_catapults = setup_simple_graph(false);

        let query = vec![AlignedBlock::new([11.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;

        let mut stats_with = crate::statistics::Stats::new();
        let mut stats_without = crate::statistics::Stats::new();

        let results_with = {
            graph_with_catapults.beam_search(&query, k, beam_width, &mut stats_with);
            graph_with_catapults.beam_search(&query, k, beam_width, &mut stats_with); // only this one has access to the catapult
            graph_with_catapults.clear_all_catapults();
            graph_with_catapults.beam_search(&query, k, beam_width, &mut stats_with)
        };
        let results_without =
            graph_without_catapults.beam_search(&query, k, beam_width, &mut stats_without);

        // Both should return the same indices
        assert_eq!(
            results_with.iter().map(|e| e.index).collect::<Vec<_>>(),
            results_without.iter().map(|e| e.index).collect::<Vec<_>>(),
            "First query should return same results with and without catapults"
        );

        // Both should return the same distances
        assert_eq!(
            results_with.iter().map(|e| e.distance).collect::<Vec<_>>(),
            results_without
                .iter()
                .map(|e| e.distance)
                .collect::<Vec<_>>(),
            "First query should return same distances with and without catapults"
        );

        // Graph without catapults should never use catapults
        assert_eq!(stats_without.get_searches_with_catapults(), 0);
        assert_eq!(stats_with.get_searches_with_catapults(), 1);
    }

    #[test]
    fn test_multiple_starting_points() {
        let nodes = vec![
            // 0: Pos 100.0, Dist 10000
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 1: Pos 0.0, Dist 1. (BEST of all)
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![0]),
            },
            // 2: Pos 5.0, Dist 36. (Worst starting point)
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
        ];
        // Start points: 0, 2
        let params =
            EngineStarterParams::new(4, 40, SIMD_LANECOUNT, NodeId { internal: 0 }, 42, true);
        let graph = AdjacencyGraph::new_flat(
            nodes,
            TestEngineStarter::new(params),
            SearchStrategy::Catapult,
        );
        let query = vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;
        let mut stats = crate::statistics::Stats::new();

        let results = graph.beam_search(&query, k, beam_width, &mut stats);
        // Top K=2 results: [1, 2]
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index.internal, 1);
        assert_eq!(results[1].index.internal, 2);
    }

    #[test]
    fn test_complex_search_divergence() {
        // Query target: [0.0]
        let nodes: Vec<Node> = vec![
            // 0: Pos 10.0, Dist 100. N: [1, 5]. (Start point)
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1, 5]),
            },
            // 1: Pos 8.0, Dist 64. N: [2].
            Node {
                payload: vec![AlignedBlock::new([8.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 2: Pos 5.0, Dist 25. N: [3].
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![3]),
            },
            // 3: Pos 2.0, Dist 4. N: [4].
            Node {
                payload: vec![AlignedBlock::new([2.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![4]),
            },
            // 4: Pos 1.0, Dist 1. N: []. (The globally BEST node)
            Node {
                payload: vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![]),
            },
            // 5: Pos 100.0, Dist 10000. N: []. (A distant dead end)
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![]),
            },
        ];
        let params =
            EngineStarterParams::new(4, 40, SIMD_LANECOUNT, NodeId { internal: 1 }, 42, true);
        let graph = AdjacencyGraph::new_flat(
            nodes,
            TestEngineStarter::new(params),
            SearchStrategy::Catapult,
        );
        let query = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])];
        let k = 1;
        let beam_width = 2; // Tight beam width forces early pruning
        let mut stats = crate::statistics::Stats::new();

        let results = graph.beam_search(&query, k, beam_width, &mut stats);

        // The globally best result (Node 4) must be found and returned.
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index.internal, 4);
    }

    fn block(value: f32) -> Box<[AlignedBlock]> {
        vec![AlignedBlock::new([value; SIMD_LANECOUNT])].into_boxed_slice()
    }

    fn empty_graph() -> AdjacencyGraph<LruSet, FlatEditableSet> {
        let params =
            EngineStarterParams::new(4, 40, SIMD_LANECOUNT, NodeId { internal: 0 }, 42, false);
        AdjacencyGraph::new_flat(vec![], EngineStarter::new(params), SearchStrategy::Vanilla)
    }

    #[test]
    fn test_empty_graph_search_returns_nothing() {
        let graph = empty_graph();
        let mut stats = Stats::new();

        assert!(graph.is_empty());
        assert!(graph.beam_search(&block(1.0), 1, 1, &mut stats).is_empty());
        assert_eq!(stats.get_computed_dists(), 0);
    }

    #[test]
    fn test_first_pushed_node_becomes_starting_node() {
        let mut graph = empty_graph();
        let mut stats = Stats::new();

        let first = graph.push_node(block(5.0), vec![]);
        let second = graph.push_node(block(7.0), vec![]);

        assert_eq!((first.internal, second.internal), (0, 1));
        assert_eq!(graph.starting_node(), first);
        let results = graph.beam_search(&block(7.0), 1, 1, &mut stats);
        assert_eq!(results[0].index, first); // node 1 is unreachable without edges
    }

    #[test]
    fn test_pushed_graph_matches_constructed_graph() {
        let mut pushed = empty_graph();
        for (value, neighbors) in [
            (0.0, vec![1]),
            (10.0, vec![2]),
            (20.0, vec![1, 3]),
            (30.0, vec![4]),
            (40.0, vec![1]),
        ] {
            pushed.push_node(block(value), neighbors);
        }
        let constructed = setup_simple_graph(false);

        for query in [0.0, 11.0, 26.0, 39.0] {
            let mut pushed_stats = Stats::new();
            let mut constructed_stats = Stats::new();
            let pushed_results = pushed.beam_search(&block(query), 2, 3, &mut pushed_stats);
            let constructed_results =
                constructed.beam_search(&block(query), 2, 3, &mut constructed_stats);

            assert_eq!(
                pushed_results.iter().map(|e| e.index).collect::<Vec<_>>(),
                constructed_results
                    .iter()
                    .map(|e| e.index)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                pushed_stats.get_computed_dists(),
                constructed_stats.get_computed_dists()
            );
        }
    }

    #[test]
    fn test_edited_neighbors_change_reachability() {
        let mut graph = empty_graph();
        let mut stats = Stats::new();
        let start = graph.push_node(block(0.0), vec![]);
        let far = graph.push_node(block(50.0), vec![]);

        assert_eq!(
            graph.beam_search(&block(50.0), 1, 1, &mut stats)[0].index,
            start
        );

        graph.neighbors_mut(start).push(far);
        assert_eq!(graph.neighbors(start), &[far]);
        assert_eq!(
            graph.beam_search(&block(50.0), 1, 1, &mut stats)[0].index,
            far
        );

        graph.neighbors_mut(start).replace(vec![]);
        assert_eq!(
            graph.beam_search(&block(50.0), 1, 1, &mut stats)[0].index,
            start
        );
    }

    #[test]
    fn test_set_starting_node_moves_search_entry() {
        let mut graph = empty_graph();
        let mut stats = Stats::new();
        graph.push_node(block(0.0), vec![]);
        let other = graph.push_node(block(50.0), vec![]);

        graph.set_starting_node(other);

        assert_eq!(graph.starting_node(), other);
        assert_eq!(
            graph.payload(other)[0],
            AlignedBlock::new([50.0; SIMD_LANECOUNT])
        );
        assert_eq!(
            graph.beam_search(&block(1.0), 1, 1, &mut stats)[0].index,
            other
        );
    }

    #[test]
    #[should_panic(expected = "starting node out of range")]
    fn test_set_starting_node_rejects_missing_node() {
        let mut graph = empty_graph();
        graph.set_starting_node(NodeId { internal: 0 });
    }
}
