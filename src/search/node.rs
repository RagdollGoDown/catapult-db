use std::fmt::Debug;

use crate::{numerics::AlignedBlock, sets::fixed::FlatFixedSet};

/// A type-safe wrapper for node indices in the proximity graph.
///
/// # Examples
/// ```
/// use catapult::search::NodeId;
///
/// let node_id = NodeId { internal: 42 };
/// assert_eq!(node_id.internal, 42);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId {
    /// The underlying node index value
    pub internal: usize,
}

/// A node in the proximity graph, containing its vector data and neighbor connections.
///
/// Each node stores both its connectivity information (neighbors) and the actual
/// vector embedding (payload) as SIMD-aligned blocks for efficient distance computation.
///
/// # Type Parameters
/// * `N` - The neighbor set type: immutable [`FlatFixedSet`] by default, or an
///   editable set for graphs built in memory
pub struct Node<N = FlatFixedSet> {
    /// The set of neighbor node indices.
    pub neighbors: N,

    /// The vector embedding for this node, stored as SIMD-aligned blocks of f32 values
    /// for efficient parallel distance computations.
    pub payload: Box<[AlignedBlock]>,
}

impl Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.internal.fmt(f)
    }
}

impl<N: Debug> Debug for Node<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("neighbors", &self.neighbors)
            .field("payload", &self.payload)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::SIMD_LANECOUNT;
    use crate::sets::fixed::FlatFixedSet;

    #[test]
    fn test_node_debug_format_basic() {
        let node = Node {
            payload: vec![AlignedBlock::new([1.5; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FlatFixedSet::new(vec![1, 2, 3]),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [1, 2, 3] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
            [1.5; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }

    #[test]
    fn test_node_debug_format_empty_neighbors() {
        let node = Node {
            payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FlatFixedSet::new(vec![]),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
            [0.0; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }

    #[test]
    fn test_node_debug_format_multiple_payload_blocks() {
        let node = Node {
            payload: vec![
                AlignedBlock::new([1.0; SIMD_LANECOUNT]),
                AlignedBlock::new([2.0; SIMD_LANECOUNT]),
                AlignedBlock::new([3.0; SIMD_LANECOUNT]),
            ]
            .into_boxed_slice(),
            neighbors: FlatFixedSet::new(vec![1, 2, 3, 4, 5]),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [1, 2, 3, 4, 5] }}, payload: [AlignedBlock {{ data: {:?} }}, AlignedBlock {{ data: {:?} }}, AlignedBlock {{ data: {:?} }}] }}",
            [1.0; SIMD_LANECOUNT], [2.0; SIMD_LANECOUNT], [3.0; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }
}
