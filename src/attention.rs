//! Custom Multi-Head Attention with RoPE support and custom SwiGLU
//!
//! This module implements a custom attention mechanism that applies Rotary Position Encoding (RoPE)
//! to the query and key tensors only, matching the PyTorch HRM implementation.
//! Also includes a custom SwiGLU implementation that matches the PyTorch version.

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay, Param};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};

use burn::tensor::{Tensor, activation, backend::Backend};

/// Configuration for custom multi-head attention with RoPE
#[derive(Config, Debug)]
pub struct CustomAttentionConfig {
    /// Hidden size (d_model)
    pub hidden_size: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for grouped query attention)
    pub num_key_value_heads: usize,
    /// Whether to use causal masking
    #[config(default = false)]
    pub causal: bool,
    /// Dropout probability
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Rotary Position Encoding
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Cached cosine values [max_seq_len, head_dim]
    pub cos_cached: Param<Tensor<B, 2>>,
    /// Cached sine values [max_seq_len, head_dim]
    pub sin_cached: Param<Tensor<B, 2>>,
}

/// Configuration for RoPE
#[derive(Config, Debug)]
pub struct RotaryEmbeddingConfig {
    /// Head dimension
    pub dim: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Base frequency
    #[config(default = 10000.0)]
    pub base: f32,
}

impl RotaryEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RotaryEmbedding<B> {
        // Create inverse frequencies: 1.0 / (base^(i/dim)) for i in [0, 2, 4, ..., dim-2]
        let inv_freq_data: Vec<f32> = (0..self.dim)
            .step_by(2)
            .map(|i| 1.0 / self.base.powf(i as f32 / self.dim as f32))
            .collect();

        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq_data.as_slice(), device);

        // Create position tensor: [0, 1, 2, ..., max_pos-1]
        let positions = Tensor::<B, 1, burn::tensor::Int>::arange(
            0..self.max_position_embeddings as i64,
            device,
        )
        .float();

        // Compute frequencies: outer product of positions and inv_freq
        // positions: [max_pos], inv_freq: [dim/2] -> freqs: [max_pos, dim/2]
        let freqs = positions
            .unsqueeze_dim::<2>(1)
            .matmul(inv_freq.unsqueeze_dim::<2>(0));

        // Duplicate frequencies: [max_pos, dim/2] -> [max_pos, dim]
        let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);

        let cos_cached = emb.clone().cos();
        let sin_cached = emb.sin();

        RotaryEmbedding {
            cos_cached: Param::from_tensor(cos_cached),
            sin_cached: Param::from_tensor(sin_cached),
        }
    }
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Get cos and sin tensors for the full sequence
    pub fn forward(&self) -> (Tensor<B, 2>, Tensor<B, 2>) {
        (self.cos_cached.val(), self.sin_cached.val())
    }
}

/// Rotate half of the tensor's last dimension
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, seq_len, num_heads, head_dim] = x.dims();
    let half_dim = head_dim / 2;

    // Split into two halves
    let x1 = x
        .clone()
        .slice([0..batch_size, 0..seq_len, 0..num_heads, 0..half_dim]);
    let x2 = x.slice([0..batch_size, 0..seq_len, 0..num_heads, half_dim..head_dim]);

    // Concatenate [-x2, x1]
    Tensor::cat(vec![-x2, x1], 3)
}

/// Apply rotary position embedding to query and key tensors
pub fn apply_rotary_pos_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_batch_size, seq_len, _num_heads, head_dim] = q.dims();

    // Slice cos and sin to match sequence length: [seq_len, head_dim]
    let cos = cos.slice([0..seq_len, 0..head_dim]);
    let sin = sin.slice([0..seq_len, 0..head_dim]);

    // Expand to match q/k dimensions: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2);
    let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2);

    // Apply RoPE: q_embed = q * cos + rotate_half(q) * sin
    let q_embed = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
    let k_embed = k.clone() * cos + rotate_half(k) * sin;

    (q_embed, k_embed)
}

/// Custom multi-head attention with RoPE support
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct CustomAttention<B: Backend> {
    /// Combined QKV projection
    pub qkv_proj: Linear<B>,
    /// Output projection
    pub o_proj: Linear<B>,
    /// Dropout layer
    pub dropout: Dropout,

    /// Configuration
    pub hidden_size: usize,
    pub head_dim: usize,
    pub output_size: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub causal: bool,
}

impl<B: Backend> ModuleDisplay for CustomAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(true)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("hidden_size", &self.hidden_size)
            .add("num_heads", &self.num_heads)
            .add("head_dim", &self.head_dim)
            .add("causal", &self.causal)
            .optional()
    }
}

impl CustomAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CustomAttention<B> {
        let output_size = self.head_dim * self.num_heads;

        // Combined QKV projection
        let qkv_size = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim;
        let qkv_proj = LinearConfig::new(self.hidden_size, qkv_size)
            .with_bias(false)
            .init(device);

        // Output projection
        let o_proj = LinearConfig::new(output_size, self.hidden_size)
            .with_bias(false)
            .init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        CustomAttention {
            qkv_proj,
            o_proj,
            dropout,
            hidden_size: self.hidden_size,
            head_dim: self.head_dim,
            output_size,
            num_heads: self.num_heads,
            num_key_value_heads: self.num_key_value_heads,
            causal: self.causal,
        }
    }
}

impl<B: Backend> CustomAttention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos_sin: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();

        // QKV projection
        let qkv = self.qkv_proj.forward(hidden_states);

        // Reshape and split into Q, K, V
        let expected_qkv_size = self.num_heads + 2 * self.num_key_value_heads;
        let qkv = qkv.reshape([batch_size, seq_len, expected_qkv_size, self.head_dim]);

        let query = qkv.clone().slice([
            0..batch_size,
            0..seq_len,
            0..self.num_heads,
            0..self.head_dim,
        ]);

        let key = qkv.clone().slice([
            0..batch_size,
            0..seq_len,
            self.num_heads..self.num_heads + self.num_key_value_heads,
            0..self.head_dim,
        ]);

        let value = qkv.slice([
            0..batch_size,
            0..seq_len,
            self.num_heads + self.num_key_value_heads
                ..self.num_heads + 2 * self.num_key_value_heads,
            0..self.head_dim,
        ]);

        // Apply RoPE if provided
        let (query, key) = if let Some((cos, sin)) = cos_sin {
            apply_rotary_pos_emb(query, key, cos, sin)
        } else {
            (query, key)
        };

        // Compute attention
        self.scaled_dot_product_attention(query, key, value)
    }

    fn scaled_dot_product_attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _, _] = query.dims();

        // Transpose key for matmul: [batch, seq, heads, head_dim] -> [batch, heads, head_dim, seq]
        let key_t = key.swap_dims(1, 2).swap_dims(2, 3);
        let query = query.swap_dims(1, 2); // [batch, heads, seq, head_dim]
        let value = value.swap_dims(1, 2); // [batch, heads, seq, head_dim]

        // Compute attention scores
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = query.matmul(key_t) * scale;

        // Apply causal mask if needed
        let scores = if self.causal {
            let mask = self.create_causal_mask(seq_len, &scores.device());
            scores + mask
        } else {
            scores
        };

        // Apply softmax
        let attn_weights = activation::softmax(scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        // Apply attention to values
        let attn_output = attn_weights.matmul(value);

        // Transpose back and reshape
        let attn_output = attn_output
            .swap_dims(1, 2) // [batch, seq, heads, head_dim]
            .reshape([batch_size, seq_len, self.output_size]);

        // Final output projection
        self.o_proj.forward(attn_output)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        // Create lower triangular mask
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }

        Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
            .reshape([seq_len, seq_len])
            .unsqueeze_dim::<4>(0)
            .unsqueeze_dim::<4>(0)
    }
}

/// Custom SwiGLU implementation matching PyTorch HRM
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct CustomSwiGlu<B: Backend> {
    /// Gate and up projection: hidden_size -> intermediate_size * 2
    pub gate_up_proj: Linear<B>,
    /// Down projection: intermediate_size -> hidden_size
    pub down_proj: Linear<B>,
    /// Configuration
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl<B: Backend> ModuleDisplay for CustomSwiGlu<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(true)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("hidden_size", &self.hidden_size)
            .add("intermediate_size", &self.intermediate_size)
            .optional()
    }
}

/// Configuration for custom SwiGLU
#[derive(Config, Debug)]
pub struct CustomSwiGluConfig {
    /// Hidden size (input/output dimension)
    pub hidden_size: usize,
    /// Expansion factor
    pub expansion: f64,
}

impl CustomSwiGluConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CustomSwiGlu<B> {
        // Calculate intermediate size like PyTorch: round(expansion * hidden_size * 2 / 3)
        // Then find multiple of 256 (like _find_multiple in PyTorch)
        let base_intermediate =
            (self.expansion * self.hidden_size as f64 * 2.0 / 3.0).round() as usize;
        let intermediate_size = base_intermediate.div_ceil(256) * 256; // Round up to multiple of 256

        // Gate and up projection combined
        let gate_up_proj = LinearConfig::new(self.hidden_size, intermediate_size * 2)
            .with_bias(false)
            .init(device);

        // Down projection
        let down_proj = LinearConfig::new(intermediate_size, self.hidden_size)
            .with_bias(false)
            .init(device);

        CustomSwiGlu {
            gate_up_proj,
            down_proj,
            hidden_size: self.hidden_size,
            intermediate_size,
        }
    }
}

impl<B: Backend> CustomSwiGlu<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Combined gate and up projection
        let gate_up = self.gate_up_proj.forward(input);

        // Split into gate and up using chunk operation
        let chunks = gate_up.chunk(2, D - 1);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();

        // Apply SiLU to gate and multiply with up
        let gate_activated = activation::silu(gate);
        let combined = gate_activated * up;

        // Down projection
        self.down_proj.forward(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Default::default();
        let config = RotaryEmbeddingConfig::new(64, 128);
        let rope = config.init::<TestBackend>(&device);

        let (cos, sin) = rope.forward();
        assert_eq!(cos.dims(), [128, 64]);
        assert_eq!(sin.dims(), [128, 64]);
    }

    #[test]
    fn test_custom_attention() {
        let device = Default::default();
        let config = CustomAttentionConfig::new(256, 64, 4, 4);
        let attention = config.init::<TestBackend>(&device);

        let hidden_states = Tensor::<TestBackend, 3>::random(
            [2, 10, 256],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = attention.forward(hidden_states, None);

        assert_eq!(output.dims(), [2, 10, 256]);
    }

    #[test]
    fn test_custom_swiglu() {
        let device = Default::default();
        let config = CustomSwiGluConfig::new(64, 2.666);
        let swiglu = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = swiglu.forward(input);

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_rotate_half() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::random(
            [1, 4, 2, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let rotated = rotate_half(x.clone());

        assert_eq!(rotated.dims(), x.dims());
    }
}
