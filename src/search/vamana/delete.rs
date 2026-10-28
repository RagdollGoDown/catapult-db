//! Lazy deletion for Vamana graphs, as in FreshDiskANN (Singh et al. 2021).
//!
//! A deleted node is only marked: it keeps its edges and stays navigable, but searches
//! no longer return it or start from it. [`consolidate_deletes`] later rewires the edges
//! around deleted nodes and disconnects them.
//!
//! DiskANN never deletes its start points (they are extra "frozen" points). Here the
//! starting node is a data point, so deleting it moves the start to the closest live
//! node instead.
//!
//! [`consolidate_deletes`]: AdjacencyGraph::consolidate_deletes

use super::VamanaParams;
use crate::{
    search::{AdjacencyGraph, NodeId},
    sets::{catapults::CatapultEvictionPolicy, editable::FlatEditableSet},
    statistics::Stats,
};

impl<EvictPolicy> AdjacencyGraph<EvictPolicy, FlatEditableSet>
where
    EvictPolicy: CatapultEvictionPolicy<Item = NodeId>,
{
    /// Deletes node `id` lazily: it is no longer returned by searches or used as a
    /// catapult, but its edges stay until [`consolidate_deletes`] runs.
    ///
    /// If `id` is the starting node, the start moves to the closest live node, found by a
    /// search with beam width `L` (or a scan of all nodes if that search finds none).
    /// Its distance computations are counted in `stats`. Deleting the last live node
    /// leaves the start on a deleted node, and searches return nothing.
    ///
    /// # Returns
    /// `false` if the node was already deleted.
    ///
    /// # Panics
    /// Panics if `id` is not a node of the graph.
    ///
    /// [`consolidate_deletes`]: AdjacencyGraph::consolidate_deletes
    pub fn delete(&mut self, id: NodeId, params: &VamanaParams, stats: &mut Stats) -> bool {
        if !self.mark_deleted(id) {
            return false;
        }
        if id == self.starting_node() {
            self.move_starting_node(params, stats);
        }
        true
    }

    /// Moves the (deleted) starting node to the live node closest to it, if any.
    fn move_starting_node(&mut self, params: &VamanaParams, stats: &mut Stats) {
        let old = self.starting_node();
        let start_entry = self.entries_for(old, &[old], stats);
        let (closest, _) = self.beam_search_collecting(
            self.payload(old),
            &start_entry,
            1,
            params.build_beam_width,
            stats,
        );

        let replacement = closest.first().map(|entry| entry.index).or_else(|| {
            let live: Vec<NodeId> = (0..self.len())
                .map(|internal| NodeId { internal })
                .filter(|&node| !self.is_deleted(node))
                .collect();
            self.entries_for(old, &live, stats)
                .into_iter()
                .min()
                .map(|entry| entry.index)
        });
        if let Some(new_start) = replacement {
            self.set_starting_node(new_start);
        }
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    use super::*;
    use crate::{
        numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike},
        search::{
            SearchStrategy,
            hash_start::{EngineStarter, EngineStarterParams},
            vamana::tests::{EditableGraph, empty_graph, on_line, random_unit_vectors},
        },
    };

    /// A graph built from `n` random points, with those points and its parameters.
    fn built_graph(n: usize, seed: u64) -> (EditableGraph, Vec<Box<[AlignedBlock]>>, VamanaParams) {
        let dim = 2 * SIMD_LANECOUNT;
        let params = VamanaParams::new(16, 48, 1.2);
        let points = random_unit_vectors(n, dim, seed);
        let mut graph = empty_graph(dim);
        graph.build(points.iter().cloned(), &params, &mut Stats::new());
        (graph, points, params)
    }

    fn delete_random(
        graph: &mut EditableGraph,
        count: usize,
        params: &VamanaParams,
        seed: u64,
    ) -> HashSet<NodeId> {
        let mut ids: Vec<NodeId> = (0..graph.len())
            .map(|internal| NodeId { internal })
            .collect();
        ids.shuffle(&mut StdRng::seed_from_u64(seed));
        let mut stats = Stats::new();
        ids.into_iter()
            .take(count)
            .inspect(|&id| assert!(graph.delete(id, params, &mut stats)))
            .collect()
    }

    fn brute_force_live_nearest(
        points: &[Box<[AlignedBlock]>],
        deleted: &HashSet<NodeId>,
        query: &[AlignedBlock],
    ) -> NodeId {
        (0..points.len())
            .map(|internal| NodeId { internal })
            .filter(|id| !deleted.contains(id))
            .min_by(|a, b| {
                points[a.internal]
                    .l2_squared(query)
                    .total_cmp(&points[b.internal].l2_squared(query))
            })
            .unwrap()
    }

    fn live_reachable_from_start(graph: &EditableGraph) -> usize {
        let mut seen = HashSet::new();
        let mut stack = vec![graph.starting_node()];
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                stack.extend(graph.neighbors(id).iter().copied());
            }
        }
        seen.into_iter().filter(|&id| !graph.is_deleted(id)).count()
    }

    #[test]
    fn test_delete_twice_returns_false() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let params = VamanaParams::new(4, 8, 1.2);
        let mut stats = Stats::new();
        graph.insert(on_line(0.0), &params, &mut stats);
        let b = graph.insert(on_line(1.0), &params, &mut stats);

        assert!(graph.delete(b, &params, &mut stats));
        assert!(!graph.delete(b, &params, &mut stats));
        assert!(graph.is_deleted(b));
        assert_eq!(graph.num_deleted(), 1);
    }

    #[test]
    fn test_deleting_start_moves_it_to_closest_live_node() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let params = VamanaParams::new(4, 8, 1.2);
        let mut stats = Stats::new();
        let ids = graph.build([0.0, 1.0, 2.0, 3.0, 4.0].map(on_line), &params, &mut stats);
        let start = graph.starting_node();
        assert_eq!(start, ids[2]);

        graph.delete(start, &params, &mut stats);

        let new_start = graph.starting_node();
        assert!(new_start == ids[1] || new_start == ids[3]);
    }

    #[test]
    fn test_deleting_every_node_empties_results() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let params = VamanaParams::new(4, 8, 1.2);
        let mut stats = Stats::new();
        let ids = graph.build([0.0, 1.0, 2.0].map(on_line), &params, &mut stats);

        for &id in &ids {
            graph.delete(id, &params, &mut stats);
        }

        assert!(
            graph
                .beam_search(&on_line(1.0), 1, 8, &mut stats)
                .is_empty()
        );
        graph.consolidate_deletes(&params, &mut stats);
        assert!(
            graph
                .beam_search(&on_line(1.0), 1, 8, &mut stats)
                .is_empty()
        );
    }

    #[test]
    fn test_insert_after_deleting_everything_becomes_start() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let params = VamanaParams::new(4, 8, 1.2);
        let mut stats = Stats::new();
        let old = graph.insert(on_line(0.0), &params, &mut stats);
        graph.delete(old, &params, &mut stats);

        let new = graph.insert(on_line(5.0), &params, &mut stats);

        assert_eq!(graph.starting_node(), new);
        assert_eq!(
            graph.beam_search(&on_line(0.0), 1, 8, &mut stats)[0].index,
            new
        );
    }

    #[test]
    fn test_search_never_returns_deleted_nodes() {
        let (mut graph, _, params) = built_graph(500, 21);
        let deleted = delete_random(&mut graph, 150, &params, 22);
        let queries = random_unit_vectors(100, 2 * SIMD_LANECOUNT, 23);
        let mut stats = Stats::new();

        for consolidated in [false, true] {
            if consolidated {
                graph.consolidate_deletes(&params, &mut stats);
            }
            for query in &queries {
                let results = graph.beam_search(query, 10, 32, &mut stats);
                assert_eq!(results.len(), 10, "consolidated: {consolidated}");
                assert!(results.iter().all(|entry| !deleted.contains(&entry.index)));
            }
        }
    }

    #[test]
    fn test_consolidated_graph_invariants_and_recall() {
        let (mut graph, points, params) = built_graph(2000, 31);
        let deleted = delete_random(&mut graph, 1000, &params, 32);
        let mut stats = Stats::new();
        graph.consolidate_deletes(&params, &mut stats);

        assert!(!graph.is_deleted(graph.starting_node()));
        for internal in 0..graph.len() {
            let id = NodeId { internal };
            let neighbors = graph.neighbors(id);
            if deleted.contains(&id) {
                assert!(neighbors.is_empty(), "deleted node {id:?} kept edges");
            } else {
                assert!(neighbors.len() <= params.max_degree);
                assert!(neighbors.iter().all(|n| !deleted.contains(n) && *n != id));
            }
        }
        assert_eq!(
            live_reachable_from_start(&graph),
            graph.len() - deleted.len()
        );

        let queries = random_unit_vectors(200, 2 * SIMD_LANECOUNT, 33);
        let hits = queries
            .iter()
            .filter(|query| {
                let found = graph.beam_search(query, 1, 32, &mut stats)[0].index;
                found == brute_force_live_nearest(&points, &deleted, query)
            })
            .count();
        let recall = hits as f64 / queries.len() as f64;
        assert!(recall >= 0.95, "recall@1 after deleting half = {recall}");
    }

    #[test]
    fn test_inserts_after_deletes_do_not_link_to_deleted_nodes() {
        let (mut graph, _, params) = built_graph(300, 41);
        let deleted = delete_random(&mut graph, 100, &params, 42);
        let mut stats = Stats::new();

        for payload in random_unit_vectors(50, 2 * SIMD_LANECOUNT, 43) {
            let id = graph.insert(payload, &params, &mut stats);
            assert!(graph.neighbors(id).iter().all(|n| !deleted.contains(n)));
        }
    }

    #[test]
    fn test_stale_catapults_are_skipped() {
        let dim = SIMD_LANECOUNT;
        let starter = EngineStarter::new(EngineStarterParams::new(
            4,
            40,
            dim,
            NodeId { internal: 0 },
            42,
            true,
        ));
        let mut graph: EditableGraph =
            AdjacencyGraph::new_flat(vec![], starter, SearchStrategy::Catapult);
        let params = VamanaParams::new(4, 8, 1.2);
        let mut stats = Stats::new();
        let xs: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let ids = graph.build(xs.iter().map(|&x| on_line(x)), &params, &mut stats);

        // The first search stores node 17 as the catapult for this query's bucket.
        let query = on_line(17.2);
        assert_eq!(
            graph.beam_search(&query, 1, 8, &mut stats)[0].index,
            ids[17]
        );

        graph.delete(ids[17], &params, &mut stats);
        graph.consolidate_deletes(&params, &mut stats);

        let results = graph.beam_search(&query, 1, 8, &mut stats);
        assert_eq!(results[0].index, ids[18]);
    }
}
