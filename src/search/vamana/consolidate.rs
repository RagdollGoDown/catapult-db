/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Delete consolidation, adapted from Microsoft DiskANN
//! (<https://github.com/microsoft/DiskANN>, commit `8a9ff2d1fc68441297a5761c5647d1d4c16fe39e`):
//!
//! * The per-node repair is `DiskANNIndex::consolidate_vector` from
//!   `diskann/src/graph/index.rs`, which implements the delete consolidation of
//!   FreshDiskANN (Singh et al. 2021, Algorithm 4).
//! * Clearing the lists of deleted nodes afterwards is DiskANN's `drop_adj_list`.
//!
//! The license and notice are in `LICENSE-DISKANN.txt` at the repository root.
//!
//! Changes from the original:
//! * Synchronous, on [`AdjacencyGraph`] directly instead of DiskANN's data providers and
//!   accessors, and run over every live node in one call.
//! * The candidate pool is kept in first-seen order rather than in a `HashSet`, so the
//!   resulting lists are deterministic.

use hashbrown::HashSet;

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
    /// Removes deleted nodes from the graph's edges, as DiskANN's consolidation does.
    ///
    /// For every live node `p` with a deleted out-neighbor, each deleted neighbor is
    /// replaced by its own live out-neighbors; if that leaves `R` or more candidates,
    /// they are pruned with RobustPrune. Live nodes without deleted neighbors are only
    /// pruned if they hold more than `R` (e.g. after inserts used the slack). Finally the
    /// lists of deleted nodes are cleared, so no edge leads into or out of them.
    ///
    /// Deleted nodes keep their ids and payloads. Distance computations of the prunes
    /// are counted in `stats`.
    pub fn consolidate_deletes(&mut self, params: &VamanaParams, stats: &mut Stats) {
        let degree = params.max_degree;

        for internal in 0..self.len() {
            let vector_id = NodeId { internal };
            if self.is_deleted(vector_id) {
                continue;
            }

            let mut seen = HashSet::new();
            let mut pool = Vec::new();
            let mut deleted_neighbors = Vec::new();
            for &i in self.neighbors(vector_id) {
                if self.is_deleted(i) {
                    deleted_neighbors.push(i);
                } else if seen.insert(i) {
                    pool.push(i);
                }
            }

            // If nothing was deleted & prune not required, do nothing.
            if deleted_neighbors.is_empty() && pool.len() <= degree {
                continue;
            }

            // If something was deleted, get neighbors of deleted nodes and add them to the pool.
            for &id in &deleted_neighbors {
                for &i in self.neighbors(id) {
                    if !self.is_deleted(i) && seen.insert(i) {
                        pool.push(i);
                    }
                }
            }

            // Remove self-loop if it snuck in.
            pool.retain(|&i| i != vector_id);

            // Prune the new pool if it exceeds the max degree.
            let adj_list = if pool.len() < degree {
                pool
            } else {
                let entries = self.entries_for(vector_id, &pool, stats);
                self.robust_prune(vector_id, entries, params, stats)
            };
            self.neighbors_mut(vector_id).replace(adj_list);
        }

        let deleted: Vec<NodeId> = self.deleted_nodes().collect();
        for id in deleted {
            self.neighbors_mut(id).replace(vec![]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        numerics::{AlignedBlock, SIMD_LANECOUNT},
        search::vamana::tests::empty_graph,
    };

    fn point(x: f32, y: f32) -> Box<[AlignedBlock]> {
        let mut data = [0.0; SIMD_LANECOUNT];
        data[0] = x;
        data[1] = y;
        vec![AlignedBlock::new(data)].into_boxed_slice()
    }

    fn sorted_neighbors<E>(
        graph: &AdjacencyGraph<E, FlatEditableSet>,
        internal: usize,
    ) -> Vec<usize>
    where
        E: CatapultEvictionPolicy<Item = NodeId>,
    {
        let mut ids: Vec<usize> = graph
            .neighbors(NodeId { internal })
            .iter()
            .map(|id| id.internal)
            .collect();
        ids.sort();
        ids
    }

    /// DiskANN's `test_consolidate_deletes_2d`: the corners of a unit square, each linked
    /// to its two adjacent corners and to the start point in the middle.
    #[test]
    fn test_consolidate_deletes_2d() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
        let lists = [vec![4, 1, 2], vec![4, 0, 3], vec![4, 3, 0], vec![4, 2, 1]];
        for ((x, y), list) in corners.into_iter().zip(lists) {
            graph.push_node(point(x, y), list);
        }
        let start = graph.push_node(point(0.5, 0.5), vec![0, 1, 2, 3]);
        graph.set_starting_node(start);

        let mut stats = Stats::new();
        let params = VamanaParams::new(4, 10, 1.2);
        assert!(graph.delete(NodeId { internal: 3 }, &params, &mut stats));
        graph.consolidate_deletes(&params, &mut stats);

        // Vertices 1 and 2 lose their edge to 3 and gain an edge to each other; the start
        // point just loses its edge to 3.
        assert_eq!(sorted_neighbors(&graph, 0), vec![1, 2, 4]);
        assert_eq!(sorted_neighbors(&graph, 1), vec![0, 2, 4]);
        assert_eq!(sorted_neighbors(&graph, 2), vec![0, 1, 4]);
        assert_eq!(sorted_neighbors(&graph, 4), vec![0, 1, 2]);
        assert!(graph.neighbors(NodeId { internal: 3 }).is_empty());
        assert_eq!(graph.starting_node(), start);
    }

    #[test]
    fn test_consolidate_prunes_large_pools() {
        // Node 0's list holds a deleted node whose neighbors push the pool past R.
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let p = graph.push_node(point(0.0, 0.0), vec![1, 2]);
        graph.push_node(point(1.0, 0.0), vec![0]);
        graph.push_node(point(5.0, 5.0), vec![3, 4, 5, 6]);
        for i in 0..4 {
            let angle = i as f32 * std::f32::consts::FRAC_PI_2;
            graph.push_node(point(2.0 * angle.cos(), 2.0 * angle.sin()), vec![0]);
        }

        let mut stats = Stats::new();
        let params = VamanaParams::new(3, 8, 1.2);
        graph.delete(NodeId { internal: 2 }, &params, &mut stats);
        graph.consolidate_deletes(&params, &mut stats);

        let neighbors = graph.neighbors(p);
        assert!(neighbors.len() <= params.max_degree);
        assert!(neighbors.iter().all(|&id| !graph.is_deleted(id)));
        assert!(stats.get_computed_dists() > 0);
    }

    #[test]
    fn test_consolidate_without_deletes_trims_slack() {
        let mut graph = empty_graph(SIMD_LANECOUNT);
        let p = graph.push_node(point(0.0, 0.0), vec![1, 2, 3, 4]);
        for x in 1..=4 {
            graph.push_node(point(x as f32, 0.0), vec![0]);
        }

        let mut stats = Stats::new();
        graph.consolidate_deletes(&VamanaParams::new(2, 8, 1.2), &mut stats);

        assert!(graph.neighbors(p).len() <= 2);
    }
}
