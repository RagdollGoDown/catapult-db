use engine_starter::{EngineStarter, StartingPoints};
use graph::AdjacencyGraph;
use pyo3::prelude::*;

mod engine_starter;
mod graph;
mod vecpy;

/// A Python module implemented in Rust for fast approximate nearest neighbor search.
///
/// This module provides Python bindings for the Catapult library, which implements
/// efficient ANN search on flat proximity graphs using beam search with optional
/// LSH-cached starting points (catapults).
#[pymodule]
fn catapultpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AdjacencyGraph>()?;
    m.add_class::<EngineStarter>()?;
    m.add_class::<StartingPoints>()?;
    Ok(())
}
