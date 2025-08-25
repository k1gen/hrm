//! Sparse Embedding Module for Puzzle Identifiers
//!
//! This module implements a sparse embedding system that efficiently handles
//! puzzle-specific embeddings during training and inference. Unlike regular
//! embeddings, this system only maintains gradients for the embeddings that
//! are actually used in the current batch.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::{Int, Tensor, backend::Backend};

/// Configuration for sparse embedding layer
#[derive(Config, Debug)]
pub struct SparseEmbeddingConfig {
    /// Total number of possible embeddings
    pub num_embeddings: usize,
    /// Dimension of each embedding vector
    pub embedding_dim: usize,
    /// Batch size for local embedding storage
    pub batch_size: usize,
    /// Standard deviation for initialization
    #[config(default = 0.02)]
    pub init_std: f64,
}

/// Sparse embedding layer that maintains a global embedding table
/// but only computes gradients for embeddings used in the current batch
#[derive(Module, Debug)]
pub struct SparseEmbedding<B: Backend> {
    /// Global embedding weights - the real weights table
    pub global_weights: Param<Tensor<B, 2>>,
    /// Local weights for current batch (with gradients) - not persistent
    pub local_weights: Param<Tensor<B, 2>>,
}

impl SparseEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SparseEmbedding<B> {
        // Initialize global weights with truncated normal distribution or zero
        let global_weights = if self.init_std > 0.0 {
            Tensor::random(
                [self.num_embeddings, self.embedding_dim],
                burn::tensor::Distribution::Normal(0.0, self.init_std),
                device,
            )
        } else {
            // Zero initialization for puzzle embeddings (init_std=0)
            Tensor::zeros([self.num_embeddings, self.embedding_dim], device)
        };

        // Initialize local weights (will be populated during forward pass)
        let local_weights = Tensor::zeros([self.batch_size, self.embedding_dim], device);

        SparseEmbedding {
            global_weights: Param::from_tensor(global_weights),
            local_weights: Param::from_tensor(local_weights),
        }
    }
}

impl<B: Backend> SparseEmbedding<B> {
    /// Forward pass through sparse embedding
    //
    /// - In eval mode: directly return global_weights[inputs]
    /// - In training mode: copy to local_weights and return those (with gradients)
    pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        // For now, always use inference mode (direct indexing from global weights)
        // TODO: Implement training mode with local weights when training is supported

        // Convert indices to 2D for embedding function: [batch_size] -> [batch_size, 1]
        let indices_2d = indices.unsqueeze_dim::<2>(1);

        // Use burn's embedding function: global_weights[indices]
        let embeddings_3d = burn::tensor::module::embedding(self.global_weights.val(), indices_2d);

        // Squeeze out the sequence dimension: [batch_size, 1, embedding_dim] -> [batch_size, embedding_dim]
        embeddings_3d.squeeze::<2>(1)
    }

    /// Update global weights from local gradients (placeholder for training mode)
    #[allow(dead_code)]
    pub fn update_global_weights(&mut self, _indices: &Tensor<B, 1, Int>) {
        // This would be called by a custom optimizer during training
        // For now, this is a placeholder since we're focusing on inference

        // In a full implementation, we would:
        // 1. Extract gradients from local_weights
        // 2. Apply them to the corresponding global_weights entries
        // 3. Reset local_weights for next batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_sparse_embedding_creation() {
        let device = Default::default();
        let config = SparseEmbeddingConfig::new(1000, 256, 32);
        let embedding = config.init::<TestBackend>(&device);

        assert_eq!(embedding.global_weights.dims(), [1000, 256]);
        assert_eq!(embedding.local_weights.dims(), [32, 256]);
    }

    #[test]
    fn test_sparse_embedding_forward() {
        let device = Default::default();
        let config = SparseEmbeddingConfig::new(1000, 256, 32);
        let embedding = config.init::<TestBackend>(&device);

        let indices = Tensor::from_data([0, 1, 999].as_slice(), &device);
        let output = embedding.forward(indices);

        assert_eq!(output.dims(), [3, 256]);
    }

    #[test]
    fn test_zero_initialization() {
        let device = Default::default();
        let config = SparseEmbeddingConfig::new(10, 8, 2).with_init_std(0.0);
        let embedding = config.init::<TestBackend>(&device);

        let indices = Tensor::from_data([0, 1].as_slice(), &device);
        let output = embedding.forward(indices);

        assert_eq!(output.dims(), [2, 8]);

        // Check that embeddings are zero
        let data = output.to_data();
        for &val in data.as_slice::<f32>().unwrap() {
            assert_eq!(val, 0.0);
        }
    }
}
