use crate::{
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{
        AdjacencyGraph, Node, NodeId, SearchStrategy,
        hash_start::{EngineStarter, EngineStarterParams},
    },
    sets::{catapults::CatapultEvictionPolicy, fixed::FlatFixedSet},
};

use std::{
    fs::File,
    io::{BufReader, Error, Read},
    path::PathBuf,
    vec,
};
use tracing::info_span;

impl<T: CatapultEvictionPolicy<Item = NodeId>> AdjacencyGraph<T> {
    /// Reads the next N bytes from a byte iterator.
    ///
    /// # Arguments
    /// * `iter` - Iterator over bytes (with error handling)
    ///
    /// # Returns
    /// `Some([u8; N])` if N bytes were successfully read, `None` otherwise
    fn next_bytes<I, const N: usize>(iter: &mut I) -> Option<[u8; N]>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        let mut bytes = [0u8; N];
        for b in &mut bytes {
            let next = iter.next();
            if let Some(Ok(byte)) = next {
                *b = byte
            } else {
                return None;
            }
        }
        Some(bytes)
    }

    /// Reads and parses the next 4 bytes as a little-endian u32.
    ///
    /// # Arguments
    /// * `iter` - Iterator over bytes
    ///
    /// # Returns
    /// `Some(u32)` if 4 bytes were successfully read and parsed, `None` otherwise
    fn next_u32<I>(iter: &mut I) -> Option<u32>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 4>(iter).map(u32::from_le_bytes)
    }

    /// Reads and parses the next 8 bytes as a little-endian u64.
    ///
    /// # Arguments
    /// * `iter` - Iterator over bytes
    ///
    /// # Returns
    /// `Some(u64)` if 8 bytes were successfully read and parsed, `None` otherwise
    fn next_u64<I>(iter: &mut I) -> Option<u64>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 8>(iter).map(u64::from_le_bytes)
    }

    /// Reads and parses the next 4 bytes as a little-endian f32.
    ///
    /// # Arguments
    /// * `iter` - Iterator over bytes
    ///
    /// # Returns
    /// `Some(f32)` if 4 bytes were successfully read and parsed, `None` otherwise
    fn next_f32<I>(iter: &mut I) -> Option<f32>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 4>(iter).map(f32::from_le_bytes)
    }

    /// Reads a vector payload as a sequence of aligned blocks.
    ///
    /// Reads `size` f32 values and packs them into `AlignedBlock` instances.
    ///
    /// # Arguments
    /// * `iter` - Iterator over bytes
    /// * `size` - Number of f32 elements to read (must be multiple of `SIMD_LANECOUNT`)
    ///
    /// # Returns
    /// `Some(Vec<AlignedBlock>)` if all bytes were successfully read, `None` otherwise
    ///
    /// # Panics
    /// Panics if `size` is not a multiple of `SIMD_LANECOUNT`
    fn next_payload<I>(iter: &mut I, size: usize) -> Option<Vec<AlignedBlock>>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        assert!(size.is_multiple_of(SIMD_LANECOUNT));

        let final_length = size / SIMD_LANECOUNT;
        let mut payload = Vec::with_capacity(final_length);
        for _ in 0..final_length {
            let mut block = [0.0; SIMD_LANECOUNT];
            for entry in block.iter_mut() {
                *entry = Self::next_f32(iter)?;
            }
            payload.push(AlignedBlock::new(block));
        }
        Some(payload)
    }

    /// Loads a flat graph from binary files containing graph structure and node payloads.
    ///
    /// Reads two files: one containing the graph adjacency structure and another containing
    /// the vector payloads for each node. The files use a custom binary format with headers
    /// containing metadata followed by per-node data.
    ///
    /// # Binary Format
    /// **Graph file header:**
    /// - `full_size` (u64): Total number of nodes
    /// - `max_degree` (u32): Maximum node degree
    /// - `entry_point` (u32): Starting node index
    /// - `num_frozen` (u64): Number of frozen nodes
    ///
    /// **Per node in graph file:**
    /// - `neighbor_count` (u32): Number of neighbors
    /// - `neighbor_indices` (u32[]): Array of neighbor node indices
    ///
    /// **Payload file header:**
    /// - `npoints` (u32): Number of points
    /// - `payload_dim` (u32): Vector dimension
    ///
    /// **Per node in payload file:**
    /// - `vector_data` (f32[]): Flat array of f32 values
    ///
    /// # Arguments
    /// * `graph_path` - Path to the binary graph structure file
    /// * `payload_path` - Path to the binary payload vectors file
    /// * `num_hash` - Number of LSH hash bits (creates 2^num_hash buckets)
    /// * `seed` - Random seed for LSH hyperplane generation
    /// * `enabled_catapults` - Whether to enable catapult acceleration
    ///
    /// # Returns
    /// A new `AdjacencyGraph` with the entry point from the file used as the starting node
    ///
    /// # Panics
    /// * Panics if files cannot be opened
    /// * Panics if file format is invalid or headers are missing
    /// * Panics if vector dimension is not a multiple of `SIMD_LANECOUNT`
    /// * Panics if the number of nodes in graph and payload files don't match
    pub fn load_flat_from_path(
        graph_path: PathBuf,
        payload_path: PathBuf,
        num_hash: usize,
        bucket_cap: usize,
        seed: u64,
        running_mode: SearchStrategy,
    ) -> Self {
        let mut graph_file = BufReader::new(File::open(graph_path).expect("FNF")).bytes();
        let mut payload_file = BufReader::new(File::open(payload_path).expect("FNF")).bytes();

        let full_size = Self::next_u64(&mut graph_file).expect("Misconfigured header");
        let max_degree = Self::next_u32(&mut graph_file).expect("Misconfigured header");
        let entry_point = Self::next_u32(&mut graph_file).expect("Misconfigured header");
        let num_frozen = Self::next_u64(&mut graph_file).expect("Misconfigured header");

        let npoints = Self::next_u32(&mut payload_file).expect("Misconfigured header");
        let payload_dim = Self::next_u32(&mut payload_file).expect("Misconfigured header") as usize;

        println!(
            "size {full_size} - degree {max_degree} - entry point {entry_point} - num frozen {num_frozen} - npoints {npoints} - payload_dim {payload_dim}",
        );

        let mut adjacency = Vec::new();

        {
            let _span = info_span!("parse_nodes", full_size).entered();
            while let Some(pointsize) = Self::next_u32(&mut graph_file) {
                let mut neighs = vec![];

                for _ in 0..pointsize {
                    neighs.push(
                        Self::next_u32(&mut graph_file)
                            .expect("Graph file declared more nodes than actually found")
                            as usize,
                    );
                }

                let associated_payload = Self::next_payload(&mut payload_file, payload_dim)
                    .expect("Error while parsing payloads");

                adjacency.push(Node {
                    neighbors: FlatFixedSet::new(neighs),
                    payload: associated_payload.into_boxed_slice(),
                });
            }
        }

        // we should have read all of the file contents by now.
        assert!(graph_file.count() == 0);
        assert!(payload_file.count() == 0);

        let entry_point_id = NodeId {
            internal: entry_point as usize,
        };

        // Determine plane_dim from the first node's payload
        let plane_dim = adjacency[0].payload.len() * SIMD_LANECOUNT;

        let engine_params = EngineStarterParams::new(
            num_hash,
            bucket_cap,
            plane_dim,
            entry_point_id,
            seed,
            matches!(running_mode, SearchStrategy::Catapult),
        );

        AdjacencyGraph::new_flat(
            adjacency,
            EngineStarter::<T>::new(engine_params),
            running_mode,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        search::{
            AdjacencyGraph,
            SearchStrategy::{Catapult, LshApg, Vanilla},
            hash_start::zorder_index::ZOrderIndex,
        },
        sets::catapults::LruSet,
    };

    #[test]
    fn loading_example_graph() {
        let graph_path = "test/index/ann";
        let payload_path = "test/index/ann_vectors.bin";

        let graphed1 = AdjacencyGraph::<LruSet>::load_flat_from_path(
            graph_path.into(),
            payload_path.into(),
            4, // num_hash
            40,
            42,       // seed
            Catapult, // enabled_catapults
        );

        let graphed2 = AdjacencyGraph::<LruSet>::load_flat_from_path(
            graph_path.into(),
            payload_path.into(),
            4, // num_hash
            40,
            42,      // seed
            Vanilla, // enabled_catapults
        );

        let graphed3 = AdjacencyGraph::<LruSet>::load_flat_from_path(
            graph_path.into(),
            payload_path.into(),
            4, // num_hash
            40,
            42, // seed
            LshApg([ZOrderIndex::new(4, 16, 4, 1.0)]),
        );

        assert!(graphed1.len() == 4);
        assert!(graphed2.len() == 4);
        assert_eq!(graphed3.len(), 4);
    }
}
