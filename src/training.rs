//! Training module for the Hierarchical Reasoning Model (HRM)
//!
//! This module provides training configuration and implementation for the HRM model
//! on Sudoku puzzles, following Burn's standard training patterns with built-in TUI support.

use crate::{
    dataset::{SudokuBatch, SudokuBatcher, SudokuData, SudokuDatasetMetadata},
    model::{HierarchicalReasoningModel, HierarchicalReasoningModelConfig},
};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

/// Training configuration for HRM
#[derive(Config)]
pub struct HrmTrainingConfig {
    /// Model configuration
    pub model: HierarchicalReasoningModelConfig,

    /// Optimizer configuration
    pub optimizer: AdamConfig,

    /// Number of training epochs
    #[config(default = 100)]
    pub num_epochs: usize,

    /// Batch size
    #[config(default = 32)]
    pub batch_size: usize,

    /// Number of data loader workers
    #[config(default = 4)]
    pub num_workers: usize,

    /// Learning rate
    #[config(default = 1e-4)]
    pub learning_rate: f64,

    /// Random seed for reproducibility
    #[config(default = 42)]
    pub seed: u64,

    /// H-level reasoning cycles during training
    #[config(default = 2)]
    pub h_cycles: usize,

    /// L-level reasoning cycles during training
    #[config(default = 2)]
    pub l_cycles: usize,
}

impl Default for HrmTrainingConfig {
    fn default() -> Self {
        Self {
            model: HierarchicalReasoningModelConfig {
                batch_size: 32,
                seq_len: 81,
                vocab_size: 11,

                hidden_size: 512,
                num_heads: 8,
                h_layers: 4,
                l_layers: 4,
                h_cycles: 2,
                l_cycles: 2,

                expansion: 2.666,
                pos_encodings: "rope".to_string(),
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                dropout: 0.1,
                halt_max_steps: 8,
                halt_exploration_prob: 0.1,
            },
            optimizer: AdamConfig::new()
                .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4)))
                .with_epsilon(1e-8)
                .with_beta_1(0.9)
                .with_beta_2(0.95),
            num_epochs: 100,
            batch_size: 32,
            num_workers: 4,
            learning_rate: 1e-4,
            seed: 42,
            h_cycles: 2,
            l_cycles: 2,
        }
    }
}

/// Forward pass method for HRM model that returns ClassificationOutput
impl<B: Backend> HierarchicalReasoningModel<B> {
    /// Forward pass for training with loss computation
    pub fn forward_classification(
        &self,
        batch: SudokuBatch<B>,
        _metadata: &SudokuDatasetMetadata,
        h_cycles: usize,
        l_cycles: usize,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.inputs.dims();

        // Create empty carry state
        let carry = self.empty_carry(
            batch_size,
            seq_len,
            self.embed_tokens.weight.dims()[1], // hidden_size
            &batch.inputs.device(),
        );

        // Forward pass through the model
        let (_, logits, _) = self.forward(carry, batch.inputs.clone(), h_cycles, l_cycles);

        // Convert targets for cross-entropy loss computation
        // Original format: empty=1, digits=2-10 (solution has no empty cells, only digits 2-10)
        // Target classes: digits 1-9 map to classes 0-8
        let targets_final = batch.targets.clone().add_scalar(-2); // 2->0, 3->1, ..., 10->8

        // Compute PyTorch-style cross-entropy loss
        let loss = compute_pytorch_cross_entropy_loss(logits.clone(), targets_final.clone());

        // Reshape for ClassificationOutput compatibility
        let logits_flat = logits.flatten::<2>(0, 1);
        let targets_flat = targets_final.flatten::<1>(0, 1);

        ClassificationOutput {
            loss,
            output: logits_flat,
            targets: targets_flat,
        }
    }
}

/// Compute cross-entropy loss matching PyTorch HRM implementation
fn compute_pytorch_cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,       // [batch, seq, vocab_size]
    targets: Tensor<B, 2, Int>, // [batch, seq]
) -> Tensor<B, 1> {
    let [batch_size, seq_len, _vocab_size] = logits.dims();

    // Flatten for cross-entropy computation
    let logits_flat = logits.flatten::<2>(0, 1); // [batch*seq, vocab_size]
    let targets_flat = targets.flatten::<1>(0, 1); // [batch*seq]

    // Compute per-element cross-entropy losses
    let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
    let targets_expanded = targets_flat.clone().unsqueeze_dim::<2>(1); // [batch*seq, 1]
    let selected_log_probs = log_probs.gather(1, targets_expanded).squeeze::<1>(1); // [batch*seq]
    let element_losses = -selected_log_probs; // [batch*seq]

    // Reshape back to [batch, seq]
    let losses = element_losses.reshape([batch_size, seq_len]);

    // PyTorch HRM computes mean loss per sequence, then sums across sequences
    let sequence_losses = losses.mean_dim(1); // [batch] - mean loss per sequence
    sequence_losses.sum() // scalar - sum across all sequences
}

impl<B: AutodiffBackend> TrainStep<SudokuBatch<B>, ClassificationOutput<B>>
    for HierarchicalReasoningModel<B>
{
    fn step(&self, batch: SudokuBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let metadata = SudokuDatasetMetadata::default();
        let output = self.forward_classification(batch, &metadata, 2, 2);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<SudokuBatch<B>, ClassificationOutput<B>>
    for HierarchicalReasoningModel<B>
{
    fn step(&self, batch: SudokuBatch<B>) -> ClassificationOutput<B> {
        let metadata = SudokuDatasetMetadata::default();
        self.forward_classification(batch, &metadata, 2, 2)
    }
}

/// Create artifact directory for training outputs (only if it doesn't exist)
fn create_artifact_dir(artifact_dir: &str) {
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Check if checkpoints exist and return the latest checkpoint epoch
fn find_latest_checkpoint(artifact_dir: &str) -> Option<usize> {
    let checkpoint_dir = format!("{}/checkpoint", artifact_dir);
    if !std::path::Path::new(&checkpoint_dir).exists() {
        return None;
    }

    let mut latest_epoch = None;
    if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
        for entry in entries.flatten() {
            if let Some(filename) = entry.file_name().to_str() {
                if filename.starts_with("model-") && filename.ends_with(".mpk") {
                    if let Some(epoch_str) = filename
                        .strip_prefix("model-")
                        .and_then(|s| s.strip_suffix(".mpk"))
                    {
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            latest_epoch =
                                Some(latest_epoch.map_or(epoch, |prev: usize| prev.max(epoch)));
                        }
                    }
                }
            }
        }
    }
    latest_epoch
}

/// Main training function with TUI support
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: HrmTrainingConfig,
    device: B::Device,
    train_dataset: impl Dataset<SudokuData> + 'static,
    val_dataset: impl Dataset<SudokuData> + 'static,
) {
    create_artifact_dir(artifact_dir);

    // Check if we can resume from checkpoint
    let latest_checkpoint = find_latest_checkpoint(artifact_dir);

    if let Some(epoch) = latest_checkpoint {
        println!(
            "🔄 Found existing checkpoint at epoch {}. Resuming training...",
            epoch
        );
    } else {
        println!("🆕 No existing checkpoints found. Starting fresh training...");

        // Save configuration only when starting fresh
        config
            .save(format!("{artifact_dir}/config.json"))
            .expect("Config should be saved successfully");
    }

    // Set random seed
    B::seed(config.seed);

    // Create batchers - different types for training and validation backends
    let train_batcher = SudokuBatcher::<B>::new();
    let val_batcher = SudokuBatcher::<B::InnerBackend>::new();

    // Create data loaders
    let dataloader_train = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_val = DataLoaderBuilder::new(val_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(val_dataset);

    // Initialize model
    let model = config.model.init::<B>(&device);

    // Build learner with basic metrics
    let learner_builder = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device.clone()));

    // Either resume from checkpoint or start fresh
    let learner = if let Some(checkpoint_epoch) = latest_checkpoint {
        learner_builder.checkpoint(checkpoint_epoch).build(
            model,
            config.optimizer.init(),
            config.learning_rate,
        )
    } else {
        learner_builder.build(model, config.optimizer.init(), config.learning_rate)
    };

    // Start training (TUI will automatically be used if available)
    let model_trained = learner.fit(dataloader_train, dataloader_val);

    // Save final model
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    println!("Training completed! Model saved to: {}/model", artifact_dir);
}
