//! Training binary for the Hierarchical Reasoning Model (HRM)
//!
//! This binary downloads Sudoku data from HuggingFace and trains the HRM model
//! using Burn's training framework with built-in TUI support.

use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
use hrm::{
    HierarchicalReasoningModelConfig, HrmTrainingConfig, SudokuDatasetConfig,
    create_sudoku_dataset, train,
};
use std::path::PathBuf;

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output directory for training artifacts
    #[arg(short, long, default_value = "/tmp/hrm_sudoku")]
    output_dir: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    learning_rate: f64,

    /// Hidden size
    #[arg(long, default_value = "512")]
    hidden_size: usize,

    /// Number of attention heads
    #[arg(long, default_value = "8")]
    num_heads: usize,

    /// Number of H-level layers
    #[arg(long, default_value = "4")]
    h_layers: usize,

    /// Number of L-level layers
    #[arg(long, default_value = "4")]
    l_layers: usize,

    /// Cache directory for HuggingFace datasets
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    println!("🚀 Starting HRM Sudoku training");
    println!("📁 Output directory: {:?}", args.output_dir);
    println!("⚙️  Configuration:");
    println!("   - Epochs: {}", args.epochs);
    println!("   - Batch size: {}", args.batch_size);
    println!("   - Learning rate: {}", args.learning_rate);
    println!("   - Hidden size: {}", args.hidden_size);

    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("🔧 Using device: {:?}", device);

    // Dataset configurations
    let train_config = SudokuDatasetConfig {
        repo: "sapientinc/sudoku-extreme".to_string(),
        split: "train".to_string(),
        cache_dir: args
            .cache_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        subsample_size: Some(1000), // Limit training data for testing
        min_difficulty: None,
    };

    let test_config = SudokuDatasetConfig {
        split: "test".to_string(),
        subsample_size: Some(100), // Limit test data for testing
        ..train_config.clone()
    };

    // Load datasets directly with CSV download
    println!("📥 Loading training dataset...");
    let train_dataset = create_sudoku_dataset(train_config)?;
    println!("📥 Loading test dataset...");
    let test_dataset = create_sudoku_dataset(test_config)?;

    println!("📊 Dataset statistics:");
    println!("   - Training samples: {}", train_dataset.len());
    println!("   - Test samples: {}", test_dataset.len());

    // Create model configuration
    let model_config = HierarchicalReasoningModelConfig {
        batch_size: args.batch_size,
        seq_len: 81,    // 9x9 Sudoku grid
        vocab_size: 11, // PAD + empty + digits 1-9
        num_puzzle_identifiers: 1,
        hidden_size: args.hidden_size,
        num_heads: args.num_heads,
        h_layers: args.h_layers,
        l_layers: args.l_layers,
        h_cycles: 2,
        l_cycles: 2,
        puzzle_emb_ndim: 0, // Disable puzzle embeddings for Sudoku
        expansion: 2.666,
        pos_encodings: "rope".to_string(),
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dropout: 0.1,
        halt_max_steps: 8,
        halt_exploration_prob: 0.1,
    };

    // Create training configuration
    let training_config = HrmTrainingConfig {
        model: model_config,
        optimizer: burn::optim::AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4)))
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.95),
        num_epochs: args.epochs,
        batch_size: args.batch_size,
        num_workers: 4,
        learning_rate: args.learning_rate,
        seed: args.seed,
        h_cycles: 2,
        l_cycles: 2,
    };

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    println!("🎯 Starting training with TUI...");

    // Start training (TUI will be automatically enabled)
    train::<MyAutodiffBackend>(
        &args.output_dir.to_string_lossy(),
        training_config,
        device,
        train_dataset,
        test_dataset,
    );

    println!("✅ Training completed!");
    println!("📁 Model saved to: {:?}", args.output_dir);

    Ok(())
}
