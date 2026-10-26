use crate::search::NodeId;

/// Read access to a graph node's out-neighbors, as needed by beam search.
///
/// Implemented by [`FlatFixedSet`](crate::sets::fixed::FlatFixedSet) for graphs
/// loaded from disk and by [`FlatEditableSet`](crate::sets::editable::FlatEditableSet)
/// for graphs built and modified in memory.
pub trait NeighborSet {
    /// Borrows the neighbor indices without copying them.
    fn as_slice(&self) -> &[NodeId];

    /// Number of neighbors.
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Whether the node has no neighbors.
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}
