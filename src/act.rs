//! Adaptive Computation Time (ACT) implementation for HRM
//!
//! This module implements ACT exactly like the PyTorch version, with a wrapper model
//! that maintains carry state and a loss head that drives the training loop.

use crate::dataset::SudokuBatch;
use crate::model::{HierarchicalReasoningModel, InnerCarry};
use burn::module::Module;

use burn::tensor::{Bool, ElementConversion, Int, Tensor, backend::Backend};
use burn::train::ClassificationOutput;
use std::collections::HashMap;

/// Inner carry state for ACT (matches PyTorch HierarchicalReasoningModel_ACTV1InnerCarry)
#[derive(Debug, Clone)]
pub struct ActInnerCarry<B: Backend> {
    pub z_h: Tensor<B, 3>,
    pub z_l: Tensor<B, 3>,
}

impl<B: Backend> From<InnerCarry<B>> for ActInnerCarry<B> {
    fn from(carry: InnerCarry<B>) -> Self {
        Self {
            z_h: carry.z_h,
            z_l: carry.z_l,
        }
    }
}

impl<B: Backend> From<ActInnerCarry<B>> for InnerCarry<B> {
    fn from(carry: ActInnerCarry<B>) -> Self {
        Self {
            z_h: carry.z_h,
            z_l: carry.z_l,
        }
    }
}

/// ACT carry state (matches PyTorch HierarchicalReasoningModel_ACTV1Carry)
#[derive(Debug, Clone)]
pub struct ActCarry<B: Backend> {
    pub inner_carry: ActInnerCarry<B>,
    pub steps: Tensor<B, 1, Int>,
    pub halted: Tensor<B, 1, Bool>,
    pub current_data: ActCurrentData<B>,
}

/// Current batch data stored in carry
#[derive(Debug, Clone)]
pub struct ActCurrentData<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

/// ACT model outputs (matches PyTorch outputs dict)
#[derive(Debug)]
pub struct ActOutputs<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub q_halt_logits: Tensor<B, 1>,
    pub q_continue_logits: Tensor<B, 1>,
    pub target_q_continue: Option<Tensor<B, 1>>,
}

/// ACT wrapper model (matches PyTorch HierarchicalReasoningModel_ACTV1)
#[derive(Module, Debug)]
pub struct ActModel<B: Backend> {
    pub inner: HierarchicalReasoningModel<B>,
    // Configuration stored as regular fields (not modules)
    pub halt_max_steps: usize,
    pub halt_exploration_prob: f64,
    pub h_cycles: usize,
    pub l_cycles: usize,
}

impl<B: Backend> ActModel<B> {
    /// Create new ACT model
    pub fn new(
        inner: HierarchicalReasoningModel<B>,
        halt_max_steps: usize,
        halt_exploration_prob: f64,
        h_cycles: usize,
        l_cycles: usize,
    ) -> Self {
        Self {
            inner,
            halt_max_steps,
            halt_exploration_prob,
            h_cycles,
            l_cycles,
        }
    }

    /// Create initial carry state (matches PyTorch initial_carry)
    pub fn initial_carry(&self, batch: &SudokuBatch<B>) -> ActCarry<B> {
        let [batch_size, seq_len] = batch.inputs.dims();
        let hidden_size = self.inner.embed_tokens.weight.dims()[1];
        let device = batch.inputs.device();

        // Create empty inner carry (will be reset with halted=true immediately)
        let inner_carry = self
            .inner
            .empty_carry(batch_size, seq_len, hidden_size, &device)
            .into();

        // Create current data with zeros - will be filled when halted sequences start
        let current_data = ActCurrentData {
            inputs: Tensor::zeros_like(&batch.inputs),
            targets: Tensor::zeros_like(&batch.targets),
        };

        ActCarry {
            inner_carry,
            steps: Tensor::zeros([batch_size], &device),
            halted: {
                let mut data = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    data.push(true);
                }
                Tensor::from_bool(burn::tensor::TensorData::from(data.as_slice()), &device)
            }, // Start halted
            current_data,
        }
    }

    /// Forward pass (matches PyTorch forward exactly)
    pub fn forward_act(
        &self,
        carry: ActCarry<B>,
        batch: SudokuBatch<B>,
        training: bool,
    ) -> (ActCarry<B>, ActOutputs<B>) {
        let device = batch.inputs.device();

        // Update data, carry (removing halted sequences) - matches PyTorch line 242-246
        let new_inner_carry = self.reset_carry(&carry.halted, carry.inner_carry);

        // Reset steps for halted sequences: new_steps = where(halted, 0, steps)
        let steps_zeros = Tensor::zeros_like(&carry.steps);
        let new_steps = carry.steps.mask_where(carry.halted.clone(), steps_zeros);

        // Update current_data where halted - matches PyTorch line 248
        let halted_2d = carry
            .halted
            .clone()
            .unsqueeze_dim::<2>(1)
            .repeat_dim(1, batch.inputs.dims()[1]);
        let current_inputs = carry
            .current_data
            .inputs
            .mask_where(halted_2d.clone(), batch.inputs.clone());
        let current_targets = carry
            .current_data
            .targets
            .mask_where(halted_2d, batch.targets.clone());

        // Forward inner model - matches PyTorch line 251
        let (new_inner_carry, logits, (q_halt_logits, q_continue_logits)) = self.inner.forward(
            new_inner_carry.into(),
            current_inputs.clone(),
            self.h_cycles,
            self.l_cycles,
        );

        // Prepare outputs - matches PyTorch line 253-257
        let mut outputs = ActOutputs {
            logits,
            q_halt_logits,
            q_continue_logits,
            target_q_continue: None,
        };

        // Step computation (with torch.no_grad equivalent) - matches PyTorch line 259-283
        let new_steps = new_steps + 1;
        let is_last_step = new_steps
            .clone()
            .greater_equal_elem(self.halt_max_steps as i64);

        let mut halted = is_last_step.clone();

        // Training with ACT enabled - matches PyTorch line 264-281
        if training && self.halt_max_steps > 1 {
            // Halt signal - matches PyTorch line 267-268
            let should_halt = outputs
                .q_halt_logits
                .clone()
                .greater(outputs.q_continue_logits.clone());
            // Workaround for WGPU boolean operation issues: use float operations
            let halted_float = halted.float();
            let should_halt_float = should_halt.float();
            let combined = (halted_float + should_halt_float).greater_elem(0.5);
            halted = combined;

            // Exploration - matches PyTorch line 270-274
            if self.halt_exploration_prob > 0.0 {
                let random_vals: Tensor<B, 1> = Tensor::random(
                    new_steps.dims(),
                    burn::tensor::Distribution::Uniform(0.0, 1.0),
                    &device,
                );
                let explore_mask = random_vals.lower_elem(self.halt_exploration_prob as f32);

                let min_halt_steps: Tensor<B, 1> = Tensor::random(
                    new_steps.dims(),
                    burn::tensor::Distribution::Uniform(2.0, (self.halt_max_steps + 1) as f64),
                    &device,
                );

                let meets_min_steps = new_steps
                    .clone()
                    .greater_equal(explore_mask.clone().int() * min_halt_steps.int() + 2);
                // Workaround for WGPU boolean operation issues: use float operations
                let halted_float = halted.float();
                let meets_min_steps_float = meets_min_steps.float();
                let combined = (halted_float * meets_min_steps_float).greater_elem(0.5);
                halted = combined;
            }

            // Compute target Q - matches PyTorch line 276-281
            let (_, _, (next_q_halt, next_q_continue)) = self.inner.forward(
                new_inner_carry.clone().into(),
                current_inputs.clone(),
                self.h_cycles,
                self.l_cycles,
            );

            // target_q = where(is_last_step, next_q_halt, max(next_q_halt, next_q_continue))
            let target_q = next_q_halt.clone().max_pair(next_q_continue.clone());
            outputs.target_q_continue = Some(burn::tensor::activation::sigmoid(target_q));
        }

        // Update carry state - matches PyTorch line 283
        let updated_carry = ActCarry {
            inner_carry: new_inner_carry.into(),
            steps: new_steps,
            halted,
            current_data: ActCurrentData {
                inputs: current_inputs,
                targets: current_targets,
            },
        };

        (updated_carry, outputs)
    }

    /// Reset carry for halted sequences (matches PyTorch inner.reset_carry)
    fn reset_carry(
        &self,
        halted: &Tensor<B, 1, Bool>,
        carry: ActInnerCarry<B>,
    ) -> ActInnerCarry<B> {
        let [batch_size, seq_len, hidden_size] = carry.z_h.dims();

        // Expand initial states to match carry dimensions
        let h_init_expanded = self
            .inner
            .h_init
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch_size)
            .repeat_dim(1, seq_len);

        let l_init_expanded = self
            .inner
            .l_init
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch_size)
            .repeat_dim(1, seq_len);

        // Apply reset using mask_where: where(halted, init_states, current_states)
        let halted_3d = halted
            .clone()
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(1, seq_len)
            .repeat_dim(2, hidden_size);

        ActInnerCarry {
            z_h: carry.z_h.mask_where(halted_3d.clone(), h_init_expanded),
            z_l: carry.z_l.mask_where(halted_3d, l_init_expanded),
        }
    }
}

/// ACT Loss Head (matches PyTorch ACTLossHead)
pub struct ActLossHead<B: Backend> {
    pub model: ActModel<B>,
    pub carry: Option<ActCarry<B>>,
}

impl<B: Backend> ActLossHead<B> {
    /// Create new ACT loss head
    pub fn new(model: ActModel<B>) -> Self {
        Self { model, carry: None }
    }

    /// Initialize carry state
    pub fn initial_carry(&mut self, batch: &SudokuBatch<B>) {
        self.carry = Some(self.model.initial_carry(batch));
    }

    /// Forward pass with loss computation (matches PyTorch ACTLossHead.forward)
    pub fn forward_loss(
        &mut self,
        batch: SudokuBatch<B>,
        training: bool,
    ) -> (ClassificationOutput<B>, HashMap<String, f32>, bool) {
        // Initialize carry if needed
        if self.carry.is_none() {
            self.initial_carry(&batch);
        }

        let carry = self.carry.take().unwrap();

        // Forward through ACT model
        let (new_carry, outputs) = self.model.forward_act(carry, batch.clone(), training);

        // Check if all halted
        let all_halted = new_carry.halted.clone().all().into_scalar().elem::<bool>();

        // Store updated carry
        self.carry = Some(new_carry.clone());

        // Get labels from current data
        let labels = &new_carry.current_data.targets;

        // Compute correctness (matches PyTorch line 62-71)
        let predictions = outputs.logits.clone().argmax(2).squeeze::<2>(2); // [batch, seq, 1] -> [batch, seq]
        let is_correct = predictions.equal(labels.clone()); // [batch, seq]
        let seq_is_correct = is_correct
            .clone()
            .float()
            .mean_dim(1)
            .squeeze::<1>(1)
            .equal_elem(1.0); // [batch, 1] -> [batch]

        // Compute losses (matches PyTorch line 74-85)
        let lm_loss = self.compute_lm_loss(&outputs.logits, labels);
        let q_halt_loss = self.compute_q_halt_loss(&outputs.q_halt_logits, &seq_is_correct.clone());

        let mut total_loss = lm_loss.clone() + q_halt_loss.clone() * 0.5;

        // Q continue loss (matches PyTorch line 87-92)
        if let Some(ref target_q) = outputs.target_q_continue {
            let q_continue_loss =
                self.compute_q_continue_loss(&outputs.q_continue_logits, target_q);
            total_loss = total_loss + q_continue_loss * 0.5;
        }

        // Compute metrics
        let mut metrics = HashMap::new();
        metrics.insert(
            "lm_loss".to_string(),
            lm_loss.clone().into_scalar().elem::<f32>(),
        );
        metrics.insert(
            "q_halt_loss".to_string(),
            q_halt_loss.clone().into_scalar().elem::<f32>(),
        );

        let accuracy = is_correct.float().mean().into_scalar().elem::<f32>();
        metrics.insert("accuracy".to_string(), accuracy);

        let exact_accuracy = seq_is_correct.float().mean().into_scalar().elem::<f32>();
        metrics.insert("exact_accuracy".to_string(), exact_accuracy);

        let avg_steps = new_carry.steps.float().mean().into_scalar().elem::<f32>();
        metrics.insert("avg_steps".to_string(), avg_steps);

        // Create classification output
        let classification_output = ClassificationOutput {
            loss: total_loss,
            output: outputs.logits.flatten::<2>(0, 1),
            targets: labels.clone().flatten::<1>(0, 1),
        };

        (classification_output, metrics, all_halted)
    }

    /// Compute language modeling loss (matches PyTorch loss computation)
    fn compute_lm_loss(&self, logits: &Tensor<B, 3>, targets: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [batch_size, seq_len, _vocab_size] = logits.dims();

        // Flatten for cross-entropy
        let logits_flat = logits.clone().flatten::<2>(0, 1);
        let targets_flat = targets.clone().flatten::<1>(0, 1);

        // Cross-entropy loss
        let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
        let targets_expanded = targets_flat.unsqueeze_dim::<2>(1);
        let selected_log_probs = log_probs.gather(1, targets_expanded).squeeze::<1>(1);
        let element_losses = -selected_log_probs;

        // Normalize by sequence length and sum (matches PyTorch)
        let losses = element_losses.reshape([batch_size, seq_len]);
        let normalized_losses = losses / (seq_len as f32);
        normalized_losses.sum()
    }

    /// Compute Q-halt loss (binary cross-entropy with correctness)
    fn compute_q_halt_loss(
        &self,
        q_halt: &Tensor<B, 1>,
        seq_correct: &Tensor<B, 1, Bool>,
    ) -> Tensor<B, 1> {
        let targets = seq_correct.clone().float();
        let q_halt_probs = burn::tensor::activation::sigmoid(q_halt.clone());
        let ones = Tensor::ones_like(&targets);
        let loss = -(targets.clone() * q_halt_probs.clone().log()
            + (ones.clone() - targets) * (ones - q_halt_probs).log());
        loss.sum()
    }

    /// Compute Q-continue loss (bootstrapping)
    fn compute_q_continue_loss(
        &self,
        q_continue: &Tensor<B, 1>,
        target_q: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let q_continue_probs = burn::tensor::activation::sigmoid(q_continue.clone());
        let ones = Tensor::ones_like(target_q);
        let loss = -(target_q.clone() * q_continue_probs.clone().log()
            + (ones.clone() - target_q.clone()) * (ones - q_continue_probs).log());
        loss.sum()
    }
}
