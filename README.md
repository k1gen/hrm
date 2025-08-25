# Hierarchical Reasoning Model (HRM) in Rust

A Rust implementation of the Hierarchical Reasoning Model using the Burn deep learning framework. This is a reimplementation of [@sapientinc's HRM](https://github.com/sapientinc/HRM) architecture, designed for complex reasoning tasks like Sudoku solving.

## Disclaimer

This is my first AI/ML project and was developed with substantial assistance from large language models. While I've strived to maintain architectural fidelity to the original PyTorch implementation, a lot of suboptimal design choices were made and will be made along the way. This implementation is primarily educational and experimental - for production use, please refer to the [original PyTorch implementation](https://github.com/sapientinc/HRM). All credits for the model architecture and training procedures go to the original authors.

## Architecture

The HRM implements a hierarchical reasoning system with two levels:

- **H-level (High-level)**: Handles global reasoning and strategic planning
- **L-level (Low-level)**: Performs detailed local computations and pattern recognition

### Key Features

- **Multi-cycle hierarchical reasoning** with configurable H and L cycles
- **Adaptive Computation Time (ACT)** via Q-learning halting mechanism
- **Rotary Position Encoding (RoPE)** or learned position embeddings
- **Custom SwiGLU MLP** implementation matching PyTorch version
- **Post-norm transformer blocks** with RMS normalization
- **Data augmentation** with Sudoku-specific transformations

## Model Components

### Custom Attention

- Multi-head attention with RoPE support
- Grouped query attention capability
- Causal masking option

### Custom SwiGLU

- Expansion factor of 2.666 (matching PyTorch HRM)
- Combined gate/up projection for efficiency
- Intermediate size rounded to multiple of 256

### Hierarchical Processing

- Initial states with truncated normal initialization
- Input injection at each reasoning cycle
- Gradient control for forward/backward iterations

## Dataset

Uses Sudoku puzzles from HuggingFace dataset `sapientinc/sudoku-extreme`:

- Direct CSV download with caching
- Configurable difficulty filtering
- Data augmentation with digit permutation and grid transformations
- 9x9 grid flattened to 81-element sequences

## Training

The model is trained to predict filled Sudoku cells:

- **Input**: Partially filled grid (empty cells = 1, digits 1-9 = 2-10)
- **Target**: Complete solution (digits 1-9 = 2-10, mapped to classes 0-8)
- **Loss**: Cross-entropy matching PyTorch HRM implementation

## Usage

### Training

```bash
cargo run --bin train -- \
    --output-dir ./trained_model \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 7e-5 \
    --hidden-size 256 \
    --num-heads 4
```

### Command Line Options

- `--output-dir`: Directory for training artifacts (default: `/tmp/hrm_sudoku`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 7e-5)
- `--hidden-size`: Model hidden dimension (default: 256)
- `--num-heads`: Number of attention heads (default: 4)
- `--h-layers`: H-level transformer layers (default: 2)
- `--l-layers`: L-level transformer layers (default: 2)
- `--cache-dir`: HuggingFace dataset cache directory
- `--seed`: Random seed for reproducibility (default: 42)
- `--num-aug`: Data augmentations per puzzle (default: 4)

## Configuration

### Model Configuration

```rust
HierarchicalReasoningModelConfig {
    batch_size: 32,
    seq_len: 81,           // 9x9 Sudoku grid
    vocab_size: 9,         // Digits 1-9 (classes 0-8)
    hidden_size: 256,
    num_heads: 4,
    h_layers: 2,           // H-level layers
    l_layers: 2,           // L-level layers
    h_cycles: 2,           // H-level reasoning cycles
    l_cycles: 2,           // L-level reasoning cycles
    expansion: 2.666,      // SwiGLU expansion factor
    pos_encodings: "rope", // "rope" or "learned"
    dropout: 0.1,
    // ... other options
}
```

### Training Configuration

```rust
HrmTrainingConfig {
    model: model_config,
    optimizer: AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1.0)))
        .with_beta_1(0.9)
        .with_beta_2(0.95),
    num_epochs: 100,
    learning_rate: 7e-5,
    // ... other options
}
```

## Performance

The model uses several optimizations:

- **Gradient control**: Forward iterations run without gradients for efficiency
- **Combined projections**: QKV and gate/up projections are fused
- **Memory efficient attention**: Optimized scaled dot-product attention
- **Fast RNG**: nanorand for augmentation performance

## Differences from PyTorch HRM

While maintaining architectural fidelity, some implementation differences exist:

1. **Initialization**: Default Burn initialization instead of custom PyTorch schemes
2. **Q-head bias**: Uses default initialization instead of -5.0 bias for halt bootstrapping
3. **Gradient control**: Rust-specific approach to detaching gradients during forward cycles
4. **Backend**: WGPU instead of CUDA (though both use GPU acceleration)

These differences are noted in code comments and will be addressed in the near future.

## License

Apache 2.0 License, same as the original implementation - see LICENSE file for details.

## Contributing

Contributions welcome. Please ensure tests pass and follow Rust formatting standards:

```bash
cargo test
cargo fmt
cargo clippy
```
