/// A trait for data structures that store catapults with eviction policies.
///
/// Catapult structures maintain a bounded set of entries that serve as cached
/// starting points for graph searches. When capacity is reached, implementations
/// must evict entries according to their specific policy (e.g., FIFO, LRU).
///
/// Graph searches store node indices (`Item = NodeId`); other users may store any
/// small `Copy` value, for example a `(key, payload)` pair.
pub trait CatapultEvictionPolicy {
    /// The type of entry stored in the structure. Equal entries are duplicates.
    type Item: Copy + Eq;

    /// Inserts an entry into the catapult structure.
    ///
    /// If the structure is at capacity, the implementation's eviction policy determines
    /// which existing entry to remove. Inserting a duplicate may remove the old occurrence
    /// and add it fresh (implementation-specific).
    ///
    /// # Arguments
    /// * `neighbor` - The entry to insert as a catapult
    fn insert(&mut self, neighbor: Self::Item);

    /// Creates a new empty catapult structure.
    ///
    /// # Returns
    /// A new empty instance of the implementing type
    fn new(max_cap: usize) -> Self;

    /// Returns all currently stored catapult entries as a vector.
    ///
    /// The order may be significant depending on the implementation (e.g., FIFO order).
    ///
    /// # Returns
    /// A vector containing all stored entries
    fn to_vec(&self) -> Vec<Self::Item>;

    /// Removes all stored catapults, resetting the structure to empty.
    fn clear(&mut self);
}
