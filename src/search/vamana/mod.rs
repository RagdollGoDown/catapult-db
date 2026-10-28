//! Vamana graph construction (DiskANN, Subramanya et al. 2019) for editable graphs.
//!
//! Pruning is DiskANN's own implementation (see [`prune`]). The code around it follows
//! the in-memory index of Microsoft DiskANN (`diskann/src/graph/index.rs` and the C++
//! `Index::link`): candidate pools capped at `max_occlusion_size`, reverse edges
//! appended until a list exceeds `R` times the slack factor, a medoid starting node for
//! bulk builds, and a final pass trimming every list to `R`.
//!
//! Deletion follows FreshDiskANN: [`delete`](AdjacencyGraph::delete) marks a node, and
//! [`consolidate_deletes`](AdjacencyGraph::consolidate_deletes) (adapted from DiskANN)
//! later rewires the edges around deleted nodes.

mod consolidate;
mod delete;
mod prune;

use crate::{
    numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike},
    search::{AdjacencyGraph, NodeId},
    sets::{
        candidates::CandidateEntry, catapults::CatapultEvictionPolicy, editable::FlatEditableSet,
    },
    statistics::Stats,
};

/// Construction parameters for a Vamana graph. Defaults follow DiskANN's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VamanaParams {
    /// Target out-degree `R`: prunes keep at most this many neighbors.
    pub max_degree: usize,

    /// Beam width `L` of the search run to find candidate neighbors for a node.
    pub build_beam_width: usize,

    /// Pruning factor `α ≥ 1`, applied to ratios of squared L2 distances as in DiskANN.
    /// Larger values keep more long-range edges, giving a denser graph with shorter
    /// search paths.
    pub alpha: f32,

    /// Only the closest `max_occlusion_size` candidates are considered by a prune
    /// (DiskANN default: 750), since pruning is quadratic in the number of candidates.
    pub max_occlusion_size: usize,

    /// Reverse edges are appended without pruning until a list holds more than
    /// `R · slack_factor` neighbors (DiskANN default: 1.3).
    pub slack_factor: f32,
}

impl VamanaParams {
    /// Creates construction parameters with DiskANN's defaults for
    /// `max_occlusion_size` (750) and `slack_factor` (1.3).
    ///
    /// # Panics
    /// Panics if `max_degree` or `build_beam_width` is zero, or `alpha < 1`.
    pub fn new(max_degree: usize, build_beam_width: usize, alpha: f32) -> Self {
        assert!(max_degree > 0, "max_degree must be positive");
        assert!(build_beam_width > 0, "build_beam_width must be positive");
        assert!(alpha >= 1.0, "alpha must be at least 1");
        Self {
            max_degree,
            build_beam_width,
            alpha,
            max_occlusion_size: 750,
            slack_factor: 1.3,
        }
    }

    /// The most neighbors a list may hold before reverse-edge insertion prunes it.
    pub fn max_degree_with_slack(&self) -> usize {
        (self.max_degree as f32 * self.slack_factor) as usize
    }
}

impl<EvictPolicy> AdjacencyGraph<EvictPolicy, FlatEditableSet>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
{
    /// Inserts a vector into the graph and links it with Vamana's insertion procedure:
    ///
    /// 1. Greedy search for the new node from the starting node with beam width `L`,
    ///    collecting every node expanded on the way.
    /// 2. RobustPrune those nodes into at most `R` out-neighbors of the new node.
    /// 3. Add the reverse edge to each chosen neighbor, pruning a neighbor's list back to
    ///    `R` only once it exceeds `R · slack_factor`.
    ///
    /// Lists can therefore hold up to `R · slack_factor` neighbors after inserts. The
    /// insertion search starts from the starting node only and never reads or writes
    /// catapults. Its distance computations are counted in `stats`. The new node never
    /// links to deleted nodes.
    ///
    /// # Returns
    /// The id of the new node. The first node inserted into an empty graph, or into one
    /// whose nodes are all deleted, becomes the starting node.
    pub fn insert(
        &mut self,
        payload: Box<[AlignedBlock]>,
        params: &VamanaParams,
        stats: &mut Stats,
    ) -> NodeId {
        let id = self.push_node(payload, vec![]);
        if self.is_deleted(self.starting_node()) {
            self.set_starting_node(id);
        }
        if self.len() > 1 {
            self.link(id, params, stats);
        }
        id
    }

    /// Adds `payloads` to the graph and links them in the given order, as DiskANN's
    /// in-memory index build does:
    ///
    /// 1. All vectors are added first, without edges.
    /// 2. If the graph was empty (or all its nodes are deleted), the starting node
    ///    becomes the medoid of the new vectors (the one closest to their centroid).
    /// 3. Each new node is linked as in [`insert`](Self::insert).
    /// 4. Every list longer than `R` is pruned back to `R`.
    ///
    /// Vamana links points in random order, so shuffle `payloads` first if they come
    /// sorted or clustered.
    ///
    /// # Returns
    /// The ids of the new nodes, in the order of `payloads`.
    pub fn build(
        &mut self,
        payloads: impl IntoIterator<Item = Box<[AlignedBlock]>>,
        params: &VamanaParams,
        stats: &mut Stats,
    ) -> Vec<NodeId> {
        let needs_start = self.is_empty() || self.is_deleted(self.starting_node());
        let ids: Vec<NodeId> = payloads
            .into_iter()
            .map(|payload| self.push_node(payload, vec![]))
            .collect();
        if ids.is_empty() {
            return ids;
        }
        if needs_start {
            let medoid = self.medoid(&ids, stats);
            self.set_starting_node(medoid);
        }

        for &id in &ids {
            self.link(id, params, stats);
        }

        for internal in 0..self.len() {
            let id = NodeId { internal };
            if self.neighbors(id).len() > params.max_degree {
                self.prune_neighbors(id, params, stats);
            }
        }
        ids
    }

    /// Sets the out-neighbors of node `p` from a fresh search and adds the reverse edges.
    fn link(&mut self, p: NodeId, params: &VamanaParams, stats: &mut Stats) {
        let pool = {
            let query = self.payload(p);
            let start = self.starting_node();
            let start_entry = self.entries_for(p, &[start], stats);

            let (_, mut expanded) =
                self.beam_search_collecting(query, &start_entry, 1, params.build_beam_width, stats);
            expanded.retain(|entry| !self.is_deleted(entry.index));
            expanded
        };

        let chosen = self.robust_prune(p, pool, params, stats);
        self.neighbors_mut(p).replace(chosen.clone());

        for j in chosen {
            if self.neighbors(j).contains(&p) {
                continue;
            }
            self.neighbors_mut(j).push(p);
            if self.neighbors(j).len() > params.max_degree_with_slack() {
                self.prune_neighbors(j, params, stats);
            }
        }
    }

    /// Prunes the current out-neighbors of `p` back to at most `R`.
    fn prune_neighbors(&mut self, p: NodeId, params: &VamanaParams, stats: &mut Stats) {
        let pool = self.entries_for(p, self.neighbors(p), stats);
        let pruned = self.robust_prune(p, pool, params, stats);
        self.neighbors_mut(p).replace(pruned);
    }

    /// Candidate entries for `ids`, with their distances to node `p`.
    fn entries_for(&self, p: NodeId, ids: &[NodeId], stats: &mut Stats) -> Vec<CandidateEntry> {
        self.entries_for_query(self.payload(p), ids, stats)
    }

    /// RobustPrune(p, pool, α, R) with DiskANN's [`prune::robust_prune`].
    ///
    /// `pool` holds candidates with their squared distance to `p`, without duplicates;
    /// `p` itself is ignored. Only the closest `max_occlusion_size` are considered.
    fn robust_prune(
        &self,
        p: NodeId,
        mut pool: Vec<CandidateEntry>,
        params: &VamanaParams,
        stats: &mut Stats,
    ) -> Vec<NodeId> {
        pool.sort();
        pool.truncate(params.max_occlusion_size.min(u16::MAX as usize));

        let candidates: Vec<prune::Candidate<&[AlignedBlock]>> = pool
            .iter()
            .map(|entry| prune::Candidate {
                distance: entry.distance.0,
                vector: (entry.index != p).then(|| self.payload(entry.index)),
            })
            .collect();
        let mut states = vec![prune::State::default(); candidates.len()];

        let found = prune::robust_prune(
            &candidates,
            &mut states,
            params.max_degree,
            params.alpha,
            |a: &&[AlignedBlock], b: &&[AlignedBlock]| {
                stats.bump_computed_dists(1);
                a.l2_squared(b)
            },
        );

        states[..found]
            .iter()
            .map(|state| pool[state.neighbor as usize].index)
            .collect()
    }

    /// The node among `ids` closest to their centroid.
    fn medoid(&self, ids: &[NodeId], stats: &mut Stats) -> NodeId {
        let blocks = self.payload(ids[0]).len();
        let mut centroid = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT]); blocks];
        for &id in ids {
            for (sum, block) in centroid.iter_mut().zip(self.payload(id)) {
                for (s, x) in sum.data.iter_mut().zip(block.data) {
                    *s += x;
                }
            }
        }
        for sum in &mut centroid {
            for s in &mut sum.data {
                *s /= ids.len() as f32;
            }
        }

        self.entries_for_query(&centroid, ids, stats)
            .into_iter()
            .min()
            .map(|entry| entry.index)
            .expect("medoid of an empty set")
    }

    /// Candidate entries for `ids`, with their distances to `query`.
    fn entries_for_query(
        &self,
        query: &[AlignedBlock],
        ids: &[NodeId],
        stats: &mut Stats,
    ) -> Vec<CandidateEntry> {
        stats.bump_computed_dists(ids.len());
        ids.iter()
            .map(|&index| CandidateEntry {
                distance: self.payload(index).l2_squared(query).into(),
                index,
                has_catapult_ancestor: false,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
    use rand_distr::StandardNormal;

    use super::*;
    use crate::search::{
        SearchStrategy,
        hash_start::{EngineStarter, EngineStarterParams},
    };
    use crate::sets::catapults::LruSet;

    pub(super) type EditableGraph = AdjacencyGraph<LruSet, FlatEditableSet>;

    pub(super) fn empty_graph(dim: usize) -> EditableGraph {
        let params = EngineStarterParams::new(4, 40, dim, NodeId { internal: 0 }, 42, false);
        AdjacencyGraph::new_flat(vec![], EngineStarter::new(params), SearchStrategy::Vanilla)
    }

    /// A vector whose first coordinate is `x` and the rest zero.
    pub(super) fn on_line(x: f32) -> Box<[AlignedBlock]> {
        let mut data = [0.0; SIMD_LANECOUNT];
        data[0] = x;
        vec![AlignedBlock::new(data)].into_boxed_slice()
    }

    pub(super) fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Box<[AlignedBlock]>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                AlignedBlock::allocate_padded(v.into_iter().map(|x| x / norm).collect())
                    .into_boxed_slice()
            })
            .collect()
    }

    fn brute_force_nearest(points: &[Box<[AlignedBlock]>], query: &[AlignedBlock]) -> usize {
        (0..points.len())
            .min_by(|&a, &b| {
                points[a]
                    .l2_squared(query)
                    .total_cmp(&points[b].l2_squared(query))
            })
            .unwrap()
    }

    fn reachable_from_start(graph: &EditableGraph) -> usize {
        let mut seen = HashSet::new();
        let mut stack = vec![graph.starting_node()];
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                stack.extend(graph.neighbors(id).iter().copied());
            }
        }
        seen.len()
    }

    fn assert_well_formed(graph: &EditableGraph, max_degree: usize) {
        for internal in 0..graph.len() {
            let id = NodeId { internal };
            let neighbors = graph.neighbors(id);
            assert!(
                neighbors.len() <= max_degree,
                "degree {} at {id:?}",
                neighbors.len()
            );
            assert!(!neighbors.contains(&id), "self loop at {id:?}");
            let unique: HashSet<_> = neighbors.iter().collect();
            assert_eq!(unique.len(), neighbors.len(), "duplicate edge at {id:?}");
        }
        assert_eq!(reachable_from_start(graph), graph.len());
    }

    #[test]
    fn test_params_reject_alpha_below_one() {
        let result = std::panic::catch_unwind(|| VamanaParams::new(8, 16, 0.9));
        assert!(result.is_err());
    }

    #[test]
    fn test_params_slack_degree() {
        assert_eq!(VamanaParams::new(10, 16, 1.2).max_degree_with_slack(), 13);
        assert_eq!(VamanaParams::new(64, 100, 1.2).max_degree_with_slack(), 83);
    }

    #[test]
    fn test_first_insert_has_no_edges() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let mut stats = Stats::new();
        let params = VamanaParams::new(4, 8, 1.2);

        let id = graph.insert(on_line(1.0), &params, &mut stats);

        assert_eq!(graph.starting_node(), id);
        assert!(graph.neighbors(id).is_empty());
        assert_eq!(stats.get_computed_dists(), 0);
    }

    #[test]
    fn test_second_insert_links_both_ways() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let mut stats = Stats::new();
        let params = VamanaParams::new(4, 8, 1.2);

        let a = graph.insert(on_line(0.0), &params, &mut stats);
        let b = graph.insert(on_line(1.0), &params, &mut stats);

        assert_eq!(graph.neighbors(a), &[b]);
        assert_eq!(graph.neighbors(b), &[a]);
    }

    #[test]
    fn test_robust_prune_uses_squared_distance_ratios() {
        // p = 0 on a line, candidates at 1, 2, 3 and -1 (squared distances 1, 4, 9, 1).
        // A candidate c is dropped when d²(p, c) / d²(s, c) > α for a chosen s.
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let p = graph.push_node(on_line(0.0), vec![]);
        let ids: Vec<NodeId> = [1.0, 2.0, 3.0, -1.0]
            .iter()
            .map(|&x| graph.push_node(on_line(x), vec![]))
            .collect();
        let mut stats = Stats::new();
        let pool = graph.entries_for(p, &ids, &mut stats);

        // α = 1: node 1 occludes 2 (4/1) and 3 (9/4); -1 is on the other side (1/4).
        let strict = graph.robust_prune(p, pool.clone(), &VamanaParams::new(8, 8, 1.0), &mut stats);
        assert_eq!(strict, vec![ids[0], ids[3]]);

        // α = 2.3: node 3's factor against node 1 is 9/4 = 2.25, so it is admitted, while
        // node 2's is 4/1 = 4 and stays out. On plain (not squared) distances, node 2's
        // ratio would be 2/1 and it would be admitted too.
        let loose = graph.robust_prune(p, pool, &VamanaParams::new(8, 8, 2.3), &mut stats);
        assert_eq!(loose, vec![ids[0], ids[3], ids[2]]);
    }

    #[test]
    fn test_robust_prune_ignores_self() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let p = graph.push_node(on_line(0.0), vec![]);
        let q = graph.push_node(on_line(1.0), vec![]);
        let mut stats = Stats::new();
        let pool = graph.entries_for(p, &[p, q], &mut stats);

        let chosen = graph.robust_prune(p, pool, &VamanaParams::new(8, 8, 1.2), &mut stats);
        assert_eq!(chosen, vec![q]);
    }

    #[test]
    fn test_robust_prune_respects_max_degree() {
        // Points on a circle around p are not occluded by each other at α = 1.2,
        // so only the degree bound limits the result.
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let p = graph.push_node(on_line(0.0), vec![]);
        let ids: Vec<NodeId> = (0..8)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::TAU / 8.0;
                let mut data = [0.0; SIMD_LANECOUNT];
                data[0] = angle.cos();
                data[1] = angle.sin();
                graph.push_node(vec![AlignedBlock::new(data)].into(), vec![])
            })
            .collect();
        let mut stats = Stats::new();
        let pool = graph.entries_for(p, &ids, &mut stats);

        let chosen = graph.robust_prune(p, pool, &VamanaParams::new(3, 8, 1.2), &mut stats);
        assert_eq!(chosen.len(), 3);
    }

    #[test]
    fn test_robust_prune_caps_candidates() {
        // Only the closest `max_occlusion_size` candidates are considered.
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let p = graph.push_node(on_line(0.0), vec![]);
        let near = graph.push_node(on_line(1.0), vec![]);
        let far = graph.push_node(on_line(-5.0), vec![]);
        let mut stats = Stats::new();
        let pool = graph.entries_for(p, &[near, far], &mut stats);
        let mut params = VamanaParams::new(8, 8, 1.2);

        assert_eq!(
            graph.robust_prune(p, pool.clone(), &params, &mut stats),
            vec![near, far]
        );
        params.max_occlusion_size = 1;
        assert_eq!(graph.robust_prune(p, pool, &params, &mut stats), vec![near]);
    }

    #[test]
    fn test_build_starts_from_medoid() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let mut stats = Stats::new();
        let mut xs: Vec<f32> = (0..=10).map(|x| x as f32).collect();
        xs.shuffle(&mut StdRng::seed_from_u64(1));

        let ids = graph.build(
            xs.iter().map(|&x| on_line(x)),
            &VamanaParams::new(4, 8, 1.2),
            &mut stats,
        );

        let start = graph.starting_node();
        assert_eq!(xs[ids.iter().position(|&id| id == start).unwrap()], 5.0);
    }

    #[test]
    fn test_built_graph_invariants() {
        let dim = 2 * SIMD_LANECOUNT;
        let params = VamanaParams::new(12, 32, 1.2);
        let mut graph = empty_graph(dim);
        let mut stats = Stats::new();

        let ids = graph.build(random_unit_vectors(500, dim, 7), &params, &mut stats);

        assert_eq!(ids.len(), 500);
        assert!(ids.iter().enumerate().all(|(i, id)| id.internal == i));
        assert_well_formed(&graph, params.max_degree);
    }

    #[test]
    fn test_inserted_graph_invariants() {
        // Without build's final trim, lists may use the slack but never exceed it.
        let dim = 2 * SIMD_LANECOUNT;
        let params = VamanaParams::new(12, 32, 1.2);
        let mut graph = empty_graph(dim);
        let mut stats = Stats::new();

        for payload in random_unit_vectors(500, dim, 8) {
            graph.insert(payload, &params, &mut stats);
        }

        assert_well_formed(&graph, params.max_degree_with_slack());
    }

    #[test]
    fn test_recall_against_brute_force() {
        let dim = 2 * SIMD_LANECOUNT;
        let n = 2000;
        let params = VamanaParams::new(16, 48, 1.2);
        let points = random_unit_vectors(n, dim, 11);
        let queries = random_unit_vectors(200, dim, 12);

        let mut graph = empty_graph(dim);
        let mut build_stats = Stats::new();
        graph.build(points.iter().cloned(), &params, &mut build_stats);

        let mut search_stats = Stats::new();
        let hits = queries
            .iter()
            .filter(|query| {
                let found = graph.beam_search(query, 1, 32, &mut search_stats)[0].index;
                found.internal == brute_force_nearest(&points, query)
            })
            .count();
        let recall = hits as f64 / queries.len() as f64;
        let dists_per_query = search_stats.get_computed_dists() / queries.len();

        assert!(recall >= 0.95, "recall@1 = {recall}");
        assert!(
            dists_per_query < n / 2,
            "{dists_per_query} distance computations per query for {n} points"
        );
    }

    #[test]
    fn test_graph_search_finds_every_inserted_point() {
        // Each point, used as a query, must find itself: a basic navigability check.
        let dim = SIMD_LANECOUNT;
        let params = VamanaParams::new(8, 24, 1.2);
        let points = random_unit_vectors(300, dim, 3);
        let mut graph = empty_graph(dim);
        let mut stats = Stats::new();
        let ids = graph.build(points.iter().cloned(), &params, &mut stats);

        for (id, point) in ids.iter().zip(&points) {
            let found = graph.beam_search(point, 1, 24, &mut stats)[0];
            assert_eq!(found.index, *id);
            assert_eq!(found.distance.0, 0.0);
        }
    }
}
