//! Training module for the Hierarchical Reasoning Model (HRM)
//!
//! This module provides training configuration and implementation for the HRM model
//! on Sudoku puzzles, following Burn's standard training patterns with built-in TUI support.

use crate::{
    act::{ActLossHead, ActModel},
    dataset::{SudokuBatch, SudokuBatcher, SudokuData},
};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::Module,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        ClassificationOutput, LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

/// Training wrapper for HRM model that maintains carry state
#[derive(Module, Debug)]
pub struct HrmTrainer<B: Backend> {
    /// The HRM model
    pub model: ActModel<B>,
}

impl<B: Backend> HrmTrainer<B> {
    /// Create a new HRM trainer
    pub fn new(model: ActModel<B>) -> Self {
        Self { model }
    }
}

impl<B: AutodiffBackend> TrainStep<SudokuBatch<B>, ClassificationOutput<B>> for HrmTrainer<B> {
    fn step(&self, item: SudokuBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Simple single-step forward pass
        let mut loss_head = ActLossHead::new(self.model.clone());
        let (classification_output, _, _) = loss_head.forward_loss(item, true);

        let grads = classification_output.loss.backward();
        TrainOutput::new(self, grads, classification_output)
    }
}

impl<B: Backend> ValidStep<SudokuBatch<B>, ClassificationOutput<B>> for HrmTrainer<B> {
    fn step(&self, item: SudokuBatch<B>) -> ClassificationOutput<B> {
        let mut loss_head = ActLossHead::new(self.model.clone());
        let (classification_output, _, _) = loss_head.forward_loss(item, false);
        classification_output
    }
}

/// Configuration for HRM training
#[derive(Config)]
pub struct HrmTrainingConfig {
    /// Inner model configuration
    pub model: crate::model::HierarchicalReasoningModelConfig,
    /// Optimizer configuration
    #[config(default = "burn::optim::AdamConfig::new()")]
    pub optimizer: burn::optim::AdamConfig,
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
            model: crate::model::HierarchicalReasoningModelConfig {
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
            optimizer: burn::optim::AdamConfig::new()
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

/// Train HRM using Burn's standard training framework
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: HrmTrainingConfig,
    device: B::Device,
    train_dataset: impl Dataset<SudokuData> + 'static,
    val_dataset: impl Dataset<SudokuData> + 'static,
) {
    // Create artifact directory
    create_artifact_dir(artifact_dir);

    // Check for existing checkpoints
    let latest_checkpoint = find_latest_checkpoint(artifact_dir);

    if let Some(epoch) = latest_checkpoint {
        println!(
            "🔄 Found existing checkpoint at epoch {}. Resuming training...",
            epoch
        );
    } else {
        println!("🆕 No existing checkpoints found. Starting fresh training...");
    }

    // Save configuration
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Set random seed
    B::seed(config.seed);

    // Initialize inner HRM model
    let inner_model = config.model.init::<B>(&device);

    // Create HRM wrapper
    let hrm_model = ActModel::new(
        inner_model,
        config.model.halt_max_steps,
        config.model.halt_exploration_prob,
        config.h_cycles,
        config.l_cycles,
    );

    // Create HRM trainer
    let model = HrmTrainer::new(hrm_model);

    // Create batchers
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

    // Build learner with TUI and checkpoint support
    let learner_builder = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .learning_strategy(LearningStrategy::SingleDevice(device));

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

    println!(
        "✅ Training completed! Model saved to: {}/model",
        artifact_dir
    );
}
