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
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric},
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
                num_puzzle_identifiers: 1,
                hidden_size: 512,
                num_heads: 8,
                h_layers: 4,
                l_layers: 4,
                h_cycles: 2,
                l_cycles: 2,
                puzzle_emb_ndim: 512,
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
        let (_, logits, _) = self.forward_with_puzzle(
            carry,
            batch.inputs.clone(),
            Some(batch.puzzle_identifiers.clone()),
            h_cycles,
            l_cycles,
        );

        // Compute cross-entropy loss
        let loss_config = CrossEntropyLossConfig::new();
        let loss_fn = loss_config.init(&logits.device());

        // Reshape for loss computation: [batch, seq, vocab] -> [batch*seq, vocab]
        let logits_flat = logits.clone().flatten::<2>(0, 1);
        let targets_flat = batch.targets.clone().flatten::<1>(0, 1);

        // Compute loss (we'll let the accuracy metric handle PAD token ignoring)
        let loss = loss_fn.forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: logits_flat,
            targets: targets_flat,
        }
    }
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

/// Create artifact directory for training outputs
fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
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

    // Save configuration
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

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

    // Build learner with TUI metrics
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(0))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(0))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device.clone()))
        .build(model, config.optimizer.init(), config.learning_rate);

    // Start training (TUI will automatically be used if available)
    let model_trained = learner.fit(dataloader_train, dataloader_val);

    // Save final model
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    println!("Training completed! Model saved to: {}/model", artifact_dir);
}
