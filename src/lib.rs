//! Hierarchical Reasoning Model (HRM) Library
//!
//! This library provides an implementation of the HRM model in the Burn framework.
//! The model supports hierarchical reasoning with two levels (H-level and L-level)
//! and includes features like adaptive computation time (ACT) and sparse embeddings.

pub mod dataset;
pub mod model;
pub mod sparse_embedding;
pub mod training;

// Re-export main types for convenience
pub use dataset::{
    SudokuBatch, SudokuBatcher, SudokuData, SudokuDataset, SudokuDatasetConfig,
    SudokuDatasetMetadata, SudokuItem, create_sudoku_dataset,
};
pub use model::{HierarchicalReasoningModel, HierarchicalReasoningModelConfig};
pub use sparse_embedding::{SparseEmbedding, SparseEmbeddingConfig};
pub use training::{HrmTrainingConfig, train};
