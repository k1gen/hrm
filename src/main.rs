//! Hierarchical Reasoning Model (HRM) - Demo Application
//!
//! This binary demonstrates the basic functionality of the HRM model implemented in Burn.
//! It initializes a model with a simple configuration and runs a forward pass to verify
//! the implementation works correctly.

mod model;

use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use model::{HierarchicalReasoningModel, HierarchicalReasoningModelConfig};

type Backend = Autodiff<Wgpu>;

fn main() {
    let device = WgpuDevice::default();

    // Create a simple config for testing
    let config = HierarchicalReasoningModelConfig {
        batch_size: 32,
        seq_len: 128,
        vocab_size: 1000,
        num_puzzle_identifiers: 10,
        hidden_size: 512,
        num_heads: 8,
        h_layers: 2,
        l_layers: 4,
        h_cycles: 2,
        l_cycles: 4,
        puzzle_emb_ndim: 0,
        expansion: 2.666,
        pos_encodings: "learned".to_string(),
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dropout: 0.1,
        halt_max_steps: 8,
        halt_exploration_prob: 0.1,
    };

    println!("Initializing model...");
    let model: HierarchicalReasoningModel<Backend> = config.init(&device);
    println!("{model}");

    // Test forward pass
    let batch_size = 2;
    let seq_len = 32;
    let input = burn::tensor::Tensor::zeros([batch_size, seq_len], &device)
        .add_scalar(42); // Use a simple constant value for testing

    let carry = model.empty_carry(batch_size, seq_len, config.hidden_size, &device);
    let (_, logits, (q_halt, q_continue)) =
        model.forward(carry, input.clone(), config.h_cycles, config.l_cycles);

    println!("Forward pass successful!");
    println!("  Logits: {:?}", logits.dims());
    println!(
        "  Q-values: halt={:?}, continue={:?}",
        q_halt.dims(),
        q_continue.dims()
    );

    let embeddings = model.input_embeddings(input);
    println!("  Embeddings: {:?}", embeddings.dims());
}
