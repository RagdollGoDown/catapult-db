use std::collections::VecDeque;

use crate::{search::NodeId, sets::catapults::CatapultEvictionPolicy};

/// An LRU (Least Recently Used) catapult storage structure with deduplication.
///
/// Maintains up to `CAPACITY` unique entries, evicting the oldest entry when
/// capacity is exceeded. Reinserting an existing element removes its old position
/// and adds it as the newest entry, maintaining set semantics.
///
/// # Type Parameters
/// * `K` - The stored entry type, node indices by default
///
/// # Panics
/// Creating a `LruSet` with `CAPACITY == 0` will panic
pub struct LruSet<K = NodeId> {
    capacity: usize,
    queue: VecDeque<K>,
}

impl<K> LruSet<K> {
    /// Creates a new empty FIFO set with the specified capacity.
    ///
    /// # Returns
    /// A new empty `LruSet` instance
    ///
    /// # Panics
    /// Panics if `CAPACITY == 0`
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        LruSet {
            capacity,
            queue: VecDeque::with_capacity(capacity),
        }
    }
}

impl<K: Copy + Eq> CatapultEvictionPolicy for LruSet<K> {
    type Item = K;

    fn to_vec(&self) -> Vec<K> {
        self.queue.iter().copied().collect()
    }

    fn insert(&mut self, key: K) {
        // Remove any existing occurrence of the key to maintain set behavior
        if let Some(pos) = self.queue.iter().position(|&x| x == key) {
            self.queue.remove(pos);
        }

        // If at capacity, evict the oldest element
        if self.queue.len() == self.capacity {
            self.queue.pop_front();
        }

        // Insert the new element at the back
        self.queue.push_back(key);
    }

    fn new(capacity: usize) -> Self {
        LruSet {
            capacity,
            queue: VecDeque::new(),
        }
    }

    fn clear(&mut self) {
        self.queue.clear();
    }
}

impl<K: std::fmt::Debug> std::fmt::Debug for LruSet<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FifoSet")
            .field("capacity", &self.capacity)
            .field("queue", &self.queue)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_fifo_set() {
        let fifo: LruSet = LruSet::new(5);
        assert_eq!(fifo.queue.len(), 0);
    }

    #[test]
    fn insert_single_element() {
        let mut fifo = LruSet::new(3);
        fifo.insert(NodeId { internal: 10 });
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0].internal, 10);
    }

    #[test]
    fn insert_up_to_capacity() {
        let mut fifo = LruSet::new(3);
        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 2 });
        fifo.insert(NodeId { internal: 3 });

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0].internal, 1);
        assert_eq!(fifo.queue[1].internal, 2);
        assert_eq!(fifo.queue[2].internal, 3);
    }

    #[test]
    fn insert_beyond_capacity_evicts_oldest() {
        let mut fifo = LruSet::new(3);
        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 2 });
        fifo.insert(NodeId { internal: 3 });
        fifo.insert(NodeId { internal: 4 }); // Should evict 1

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0].internal, 2);
        assert_eq!(fifo.queue[1].internal, 3);
        assert_eq!(fifo.queue[2].internal, 4);
    }

    #[test]
    fn fifo_order_maintained_over_multiple_evictions() {
        let mut fifo = LruSet::new(3);

        // Fill to capacity
        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 2 });
        fifo.insert(NodeId { internal: 3 });

        // Insert more, triggering multiple evictions
        fifo.insert(NodeId { internal: 4 }); // Evicts 1, queue: [2, 3, 4]
        fifo.insert(NodeId { internal: 5 }); // Evicts 2, queue: [3, 4, 5]
        fifo.insert(NodeId { internal: 6 }); // Evicts 3, queue: [4, 5, 6]

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0].internal, 4);
        assert_eq!(fifo.queue[1].internal, 5);
        assert_eq!(fifo.queue[2].internal, 6);
    }

    #[test]
    fn capacity_one_always_contains_last_inserted() {
        let mut fifo = LruSet::new(1);

        fifo.insert(NodeId { internal: 10 });
        assert_eq!(fifo.queue[0].internal, 10);

        fifo.insert(NodeId { internal: 20 });
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0].internal, 20);

        fifo.insert(NodeId { internal: 30 });
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0].internal, 30);
    }

    #[test]
    #[should_panic]
    fn zero_capacity_panics() {
        let _fifo: LruSet = LruSet::new(0);
    }

    #[test]
    fn large_capacity_behaves_correctly() {
        let mut fifo = LruSet::new(1000);

        // Insert 500 elements
        for i in 0..500 {
            fifo.insert(NodeId { internal: i });
        }

        assert_eq!(fifo.queue.len(), 500);
        assert_eq!(fifo.queue[0].internal, 0);
        assert_eq!(fifo.queue[499].internal, 499);
    }

    #[test]
    fn stress_test_many_insertions() {
        let mut fifo = LruSet::new(100);

        // Insert 10,000 elements
        for i in 0..10_000 {
            fifo.insert(NodeId { internal: i });
        }

        // Should contain last 100: [9900..10000)
        assert_eq!(fifo.queue.len(), 100);
        assert_eq!(fifo.queue[0].internal, 9900);
        assert_eq!(fifo.queue[99].internal, 9999);
    }

    #[test]
    fn test_debug() {
        let mut fifo = LruSet::new(3);
        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 2 });

        let debug_str = format!("{fifo:?}");
        assert_eq!(debug_str, "FifoSet { capacity: 3, queue: [1, 2] }");
    }

    #[test]
    fn duplicate_insertion_maintains_set_property() {
        let mut fifo = LruSet::new(3);

        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 1 });

        // Should only have one element
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0].internal, 1);
    }

    #[test]
    fn duplicate_with_full_capacity_maintains_set_behavior() {
        let mut fifo = LruSet::new(3);

        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 2 });
        fifo.insert(NodeId { internal: 3 });
        // Queue: [1, 2, 3]

        fifo.insert(NodeId { internal: 2 }); // Should remove 2 from middle, add at end
        // Queue: [1, 3, 2]

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0].internal, 1);
        assert_eq!(fifo.queue[1].internal, 3);
        assert_eq!(fifo.queue[2].internal, 2);
    }

    #[test]
    fn repeated_duplicates_maintain_fifo_order() {
        let mut fifo = LruSet::new(4);

        fifo.insert(NodeId { internal: 1 });
        fifo.insert(NodeId { internal: 2 });
        fifo.insert(NodeId { internal: 3 });
        fifo.insert(NodeId { internal: 4 });
        // Queue: [1, 2, 3, 4]

        fifo.insert(NodeId { internal: 2 }); // Remove 2 from position 1, add at end
        // Queue: [1, 3, 4, 2]

        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0].internal, 1);
        assert_eq!(fifo.queue[1].internal, 3);
        assert_eq!(fifo.queue[2].internal, 4);
        assert_eq!(fifo.queue[3].internal, 2);

        fifo.insert(NodeId { internal: 3 }); // Remove 3 from position 1, add at end
        // Queue: [1, 4, 2, 3]

        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0].internal, 1);
        assert_eq!(fifo.queue[1].internal, 4);
        assert_eq!(fifo.queue[2].internal, 2);
        assert_eq!(fifo.queue[3].internal, 3);

        fifo.insert(NodeId { internal: 5 }); // New element, evict 1
        // Queue: [4, 2, 3, 5]

        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0].internal, 4);
        assert_eq!(fifo.queue[1].internal, 2);
        assert_eq!(fifo.queue[2].internal, 3);
        assert_eq!(fifo.queue[3].internal, 5);
    }

    #[test]
    fn set_behavior_no_duplicates_in_final_state() {
        let mut fifo = LruSet::new(5);

        fifo.insert(NodeId { internal: 10 });
        fifo.insert(NodeId { internal: 20 });
        fifo.insert(NodeId { internal: 10 });
        fifo.insert(NodeId { internal: 30 });
        fifo.insert(NodeId { internal: 20 });

        // Should have [10, 30, 20] - no duplicates
        assert_eq!(fifo.queue.len(), 3);

        // Verify no duplicates
        let vec = fifo.to_vec();
        let mut sorted = vec.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(vec.len(), sorted.len(), "Should have no duplicates");
    }

    #[test]
    fn stores_pairs_with_whole_pair_deduplication() {
        let mut lru: LruSet<(usize, usize)> = LruSet::new(2);

        lru.insert((7, 1));
        lru.insert((8, 2));
        lru.insert((7, 1)); // Refreshes (7, 1): [(8, 2), (7, 1)]
        assert_eq!(lru.to_vec(), vec![(8, 2), (7, 1)]);

        lru.insert((9, 3)); // Evicts (8, 2)
        assert_eq!(lru.to_vec(), vec![(7, 1), (9, 3)]);
    }
}
