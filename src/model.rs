//! Hierarchical Reasoning Model (HRM) implementation in Burn
//!
//! This module implements a hierarchical reasoning architecture with two levels:
//! - H-level (High-level): Handles global reasoning and planning
//! - L-level (Low-level): Handles local reasoning and detailed processing
//!
//! The model supports:
//! - Multi-cycle hierarchical reasoning
//! - Adaptive computation time (ACT) via Q-learning halting mechanism
//! - Learned or rotary position encodings
//! - Configurable transformer blocks with post-norm architecture

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, SwiGlu, SwiGluConfig,
    attention::{MultiHeadAttention, MultiHeadAttentionConfig},
};
use burn::tensor::{Int, Tensor, backend::Backend};

/// Configuration for the Hierarchical Reasoning Model
#[derive(Config, Debug)]
pub struct HierarchicalReasoningModelConfig {
    /// Batch size for training
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of puzzle identifiers
    pub num_puzzle_identifiers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of H-level layers
    pub h_layers: usize,
    /// Number of L-level layers
    pub l_layers: usize,
    /// Number of H-level cycles
    pub h_cycles: usize,
    /// Number of L-level cycles
    pub l_cycles: usize,
    /// Puzzle embedding dimension (0 to disable)
    #[config(default = 0)]
    pub puzzle_emb_ndim: usize,
    /// MLP expansion factor
    #[config(default = 2.666)]
    pub expansion: f64,
    /// Position encoding type ("rope" or "learned")
    pub pos_encodings: String,
    /// RMS norm epsilon
    #[config(default = 1e-5)]
    pub rms_norm_eps: f64,
    /// RoPE theta
    #[config(default = 10000.0)]
    pub rope_theta: f32,
    /// Dropout probability
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Maximum halting steps for ACT
    #[config(default = 8)]
    pub halt_max_steps: usize,
    /// Exploration probability for halting
    #[config(default = 0.1)]
    pub halt_exploration_prob: f64,
}

impl HierarchicalReasoningModelConfig {}

/// A single transformer block
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention mechanism
    pub self_attn: MultiHeadAttention<B>,
    /// MLP layer
    pub mlp: SwiGlu<B>,
    /// RMS normalization after attention
    pub norm1: RmsNorm<B>,
    /// RMS normalization after MLP
    pub norm2: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &HierarchicalReasoningModelConfig, device: &B::Device) -> Self {
        let _head_dim = config.hidden_size / config.num_heads;

        let self_attn = MultiHeadAttentionConfig::new(config.hidden_size, config.num_heads)
            .with_dropout(config.dropout)
            .init(device);

        let mlp = SwiGluConfig::new(config.hidden_size, config.hidden_size).init(device);

        let norm1 = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        let norm2 = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        Self {
            self_attn,
            mlp,
            norm1,
            norm2,
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // Post-norm architecture: residual -> layer -> norm

        // Self-attention with residual connection
        let attn_input = hidden_states.clone();
        let mha_input = burn::nn::attention::MhaInput::self_attn(attn_input);
        let attn_output = self.self_attn.forward(mha_input);
        let hidden_states = self.norm1.forward(hidden_states + attn_output.context);

        // MLP with residual connection
        let mlp_output = self.mlp.forward(hidden_states.clone());
        let hidden_states = self.norm2.forward(hidden_states + mlp_output);

        hidden_states
    }
}

/// A reasoning module containing multiple transformer blocks
#[derive(Module, Debug)]
pub struct ReasoningModule<B: Backend> {
    /// Stack of transformer blocks
    pub layers: Vec<TransformerBlock<B>>,
}

impl<B: Backend> ReasoningModule<B> {
    pub fn new(
        num_layers: usize,
        config: &HierarchicalReasoningModelConfig,
        device: &B::Device,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();

        Self { layers }
    }

    pub fn forward(
        &self,
        mut hidden_states: Tensor<B, 3>,
        input_injection: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // Add input injection
        hidden_states = hidden_states + input_injection;

        // Pass through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }

        hidden_states
    }
}

/// Inner carry state for the model
#[derive(Debug, Clone)]
pub struct InnerCarry<B: Backend> {
    pub z_h: Tensor<B, 3>, // [batch_size, seq_len, hidden_size]
    pub z_l: Tensor<B, 3>, // [batch_size, seq_len, hidden_size]
}

/// The main hierarchical reasoning model
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct HierarchicalReasoningModel<B: Backend> {
    /// Token embeddings
    pub embed_tokens: Embedding<B>,
    /// Position embeddings (for learned positional encoding)
    pub embed_pos: Option<Embedding<B>>,
    /// H-level reasoning module
    pub h_level: ReasoningModule<B>,
    /// L-level reasoning module
    pub l_level: ReasoningModule<B>,
    /// Language modeling head
    pub lm_head: Linear<B>,
    /// Q-value head for halting decision
    pub q_head: Linear<B>,
    /// Initial H state
    pub h_init: Param<Tensor<B, 1>>,
    /// Initial L state
    pub l_init: Param<Tensor<B, 1>>,
}

impl<B: Backend> ModuleDisplay for HierarchicalReasoningModel<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(true)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [vocab_size, hidden_size] = self.embed_tokens.weight.dims();
        content
            .add("vocab_size", &vocab_size)
            .add("hidden_size", &hidden_size)
            .add("h_layers", &self.h_level.layers.len())
            .add("l_layers", &self.l_level.layers.len())
            .optional()
    }
}

impl HierarchicalReasoningModelConfig {
    /// Initialize the hierarchical reasoning model
    pub fn init<B: Backend>(&self, device: &B::Device) -> HierarchicalReasoningModel<B> {
        // Token embeddings
        let embed_tokens = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);

        // Position embeddings (if using learned positional encoding)
        let _puzzle_emb_len = if self.puzzle_emb_ndim > 0 {
            (self.puzzle_emb_ndim + self.hidden_size - 1) / self.hidden_size // ceil division
        } else {
            0
        };

        let embed_pos = if self.pos_encodings == "learned" {
            Some(EmbeddingConfig::new(self.seq_len, self.hidden_size).init(device))
        } else {
            None
        };

        // Reasoning modules
        let h_level = ReasoningModule::new(self.h_layers, self, device);
        let l_level = ReasoningModule::new(self.l_layers, self, device);

        // Output heads
        let lm_head = LinearConfig::new(self.hidden_size, self.vocab_size)
            .with_bias(false)
            .init(device);

        let q_head = LinearConfig::new(self.hidden_size, 2)
            .with_bias(true)
            .init(device);

        // Initial states - use normal initialization
        let h_init: Tensor<B, 1> = Tensor::random(
            [self.hidden_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let l_init: Tensor<B, 1> = Tensor::random(
            [self.hidden_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        HierarchicalReasoningModel {
            embed_tokens,
            embed_pos,
            h_level,
            l_level,
            lm_head,
            q_head,
            h_init: Param::from_tensor(h_init),
            l_init: Param::from_tensor(l_init),
        }
    }
}

impl<B: Backend> HierarchicalReasoningModel<B> {
    /// Create empty carry state
    pub fn empty_carry(
        &self,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        device: &B::Device,
    ) -> InnerCarry<B> {
        InnerCarry {
            z_h: Tensor::zeros([batch_size, seq_len, hidden_size], device),
            z_l: Tensor::zeros([batch_size, seq_len, hidden_size], device),
        }
    }

    /// Reset carry state for halted sequences (currently unused but kept for future ACT implementation)
    #[allow(dead_code)]
    pub fn reset_carry(
        &self,
        reset_flag: Tensor<B, 1, Int>,
        carry: InnerCarry<B>,
    ) -> InnerCarry<B> {
        let batch_size = reset_flag.dims()[0];
        let [_, seq_len, hidden_size] = carry.z_h.dims();

        // Expand initial states to batch size
        let h_init_expanded = self
            .h_init
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch_size)
            .repeat_dim(1, seq_len);

        let l_init_expanded = self
            .l_init
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch_size)
            .repeat_dim(1, seq_len);

        // Convert reset_flag to float for masking
        let reset_mask = reset_flag
            .float()
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(1, seq_len)
            .repeat_dim(2, hidden_size);

        InnerCarry {
            z_h: reset_mask.clone() * h_init_expanded + (1.0 - reset_mask.clone()) * carry.z_h,
            z_l: reset_mask.clone() * l_init_expanded + (1.0 - reset_mask) * carry.z_l,
        }
    }

    /// Input embedding computation
    pub fn input_embeddings(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Token embeddings
        let mut embeddings = self.embed_tokens.forward(input);

        // Position embeddings (if using learned)
        if let Some(ref embed_pos) = self.embed_pos {
            let seq_len = embeddings.dims()[1];
            let batch_size = embeddings.dims()[0];

            // Create position indices
            let positions = Tensor::arange(0..seq_len as i64, &embeddings.device())
                .unsqueeze_dim::<2>(0)
                .repeat_dim(0, batch_size);

            let pos_embeddings = embed_pos.forward(positions);

            // Scale by 1/sqrt(2) to maintain variance
            embeddings = embeddings * 0.707106781 + pos_embeddings * 0.707106781;
        }

        // Scale embeddings
        let embed_scale = (embeddings.dims()[2] as f64).sqrt();
        embeddings * embed_scale
    }

    /// Forward pass
    pub fn forward(
        &self,
        carry: InnerCarry<B>,
        input: Tensor<B, 2, Int>,
        h_cycles: usize,
        l_cycles: usize,
    ) -> (InnerCarry<B>, Tensor<B, 3>, (Tensor<B, 1>, Tensor<B, 1>)) {
        // Input embeddings
        let input_embeddings = self.input_embeddings(input);

        // Get current states
        let mut z_h = carry.z_h;
        let mut z_l = carry.z_l;

        // Forward iterations (no grad for efficiency)
        {
            let z_h_no_grad = z_h.clone().detach();
            let z_l_no_grad = z_l.clone().detach();

            z_h = z_h_no_grad;
            z_l = z_l_no_grad;

            // Hierarchical reasoning cycles
            for _h_step in 0..h_cycles {
                for _l_step in 0..l_cycles {
                    if !(_h_step == h_cycles - 1 && _l_step == l_cycles - 1) {
                        z_l = self
                            .l_level
                            .forward(z_l, z_h.clone() + input_embeddings.clone());
                    }
                }

                if _h_step < h_cycles - 1 {
                    z_h = self.h_level.forward(z_h, z_l.clone());
                }
            }
        }

        // Final step with gradients
        z_l = self.l_level.forward(z_l, z_h.clone() + input_embeddings);
        z_h = self.h_level.forward(z_h, z_l.clone());

        // Output predictions
        let lm_output = self.lm_head.forward(z_h.clone());

        // Q-values from first token
        let first_token_repr: Tensor<B, 2> = z_h
            .clone()
            .slice([0..z_h.dims()[0], 0..1, 0..z_h.dims()[2]])
            .flatten::<2>(1, 2);
        let q_logits = self.q_head.forward(first_token_repr);
        let q_halt: Tensor<B, 1> = q_logits
            .clone()
            .slice([0..q_logits.dims()[0], 0..1])
            .flatten::<1>(0, 1);
        let q_continue: Tensor<B, 1> = q_logits
            .clone()
            .slice([0..q_logits.dims()[0], 1..2])
            .flatten::<1>(0, 1);

        // New carry without gradients
        let new_carry = InnerCarry {
            z_h: z_h.detach(),
            z_l: z_l.detach(),
        };

        (new_carry, lm_output, (q_halt, q_continue))
    }
}
