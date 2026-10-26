use std::fmt::Debug;

use crate::{search::NodeId, sets::NeighborSet};

/// An editable set of neighbor indices for a flat proximity graph node.
///
/// The mutable counterpart of [`FlatFixedSet`](crate::sets::fixed::FlatFixedSet),
/// used by graphs that are built and modified in memory (e.g. Vamana insertion
/// and deletion), where edges are added, replaced and pruned after creation.
pub struct FlatEditableSet {
    neighbors: Vec<NodeId>,
}

impl FlatEditableSet {
    /// Creates a new editable set from a vector of neighbor indices.
    ///
    /// # Arguments
    /// * `initial_values` - Vector of node indices representing the neighbors
    pub fn new(initial_values: Vec<usize>) -> Self {
        FlatEditableSet {
            neighbors: initial_values
                .iter()
                .map(|&x| NodeId { internal: x })
                .collect(),
        }
    }

    /// Whether `id` is already a neighbor.
    pub fn contains(&self, id: NodeId) -> bool {
        self.neighbors.contains(&id)
    }

    /// Appends a neighbor. Callers that bound the degree check
    /// [`len`](NeighborSet::len) afterwards and prune; no deduplication is done here.
    pub fn push(&mut self, id: NodeId) {
        self.neighbors.push(id);
    }

    /// Replaces the whole neighbor list, e.g. with the result of a prune.
    pub fn replace(&mut self, ids: Vec<NodeId>) {
        self.neighbors = ids;
    }

    /// Keeps only the neighbors for which `keep` returns true.
    pub fn retain(&mut self, keep: impl FnMut(&NodeId) -> bool) {
        self.neighbors.retain(keep);
    }
}

impl NeighborSet for FlatEditableSet {
    fn as_slice(&self) -> &[NodeId] {
        &self.neighbors
    }
}

impl Debug for FlatEditableSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditableSet")
            .field("neighbors", &self.neighbors)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(values: &[usize]) -> Vec<NodeId> {
        values.iter().map(|&internal| NodeId { internal }).collect()
    }

    #[test]
    fn test_new_and_as_slice() {
        let set = FlatEditableSet::new(vec![1, 2, 3]);

        assert_eq!(set.as_slice(), ids(&[1, 2, 3]).as_slice());
        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
    }

    #[test]
    fn test_as_slice_borrows_without_copying() {
        let set = FlatEditableSet::new(vec![1, 2, 3]);

        assert_eq!(set.as_slice().as_ptr(), set.as_slice().as_ptr());
    }

    #[test]
    fn test_push_and_contains() {
        let mut set = FlatEditableSet::new(vec![]);
        assert!(set.is_empty());

        set.push(NodeId { internal: 4 });

        assert!(set.contains(NodeId { internal: 4 }));
        assert!(!set.contains(NodeId { internal: 5 }));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_replace_and_retain() {
        let mut set = FlatEditableSet::new(vec![1, 2, 3]);

        set.retain(|id| id.internal != 2);
        assert_eq!(set.as_slice(), ids(&[1, 3]).as_slice());

        set.replace(ids(&[9]));
        assert_eq!(set.as_slice(), ids(&[9]).as_slice());
    }

    #[test]
    fn test_debug_formatting() {
        let set = FlatEditableSet::new(vec![7, 8]);

        assert_eq!(format!("{set:?}"), "EditableSet { neighbors: [7, 8] }");
    }
}
