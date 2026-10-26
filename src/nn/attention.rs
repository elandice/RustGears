//! Attention and Transformer blocks.

use super::{prefixed, Dropout, LayerNorm, Linear, Module};
use crate::tensor::Tensor;

/// Multi-head scaled dot-product self-attention.
///
/// Input / output: `[batch, seq_len, d_model]`.
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub heads: usize,
    pub d_model: usize,
    /// Apply a causal (look-ahead) mask, as in GPT-style decoders.
    pub causal: bool,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, heads: usize) -> Self {
        assert_eq!(d_model % heads, 0, "d_model must be divisible by heads");
        MultiHeadAttention {
            wq: Linear::new(d_model, d_model),
            wk: Linear::new(d_model, d_model),
            wv: Linear::new(d_model, d_model),
            wo: Linear::new(d_model, d_model),
            heads,
            d_model,
            causal: false,
        }
    }

    pub fn causal(mut self, c: bool) -> Self {
        self.causal = c;
        self
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.heads;
        let dh = d / h;
        // [B,T,D] -> [B*H, T, Dh]
        let split = |z: Tensor| z.reshape(&[b, t, h, dh]).permute(&[0, 2, 1, 3]).reshape(&[b * h, t, dh]);
        let q = split(self.wq.forward(x));
        let k = split(self.wk.forward(x));
        let v = split(self.wv.forward(x));

        let mut scores = q.bmm(&k.transpose(1, 2)).mul_scalar(1.0 / (dh as f32).sqrt()); // [B*H,T,T]
        if self.causal {
            let mut m = vec![0.0f32; t * t];
            for i in 0..t {
                for j in (i + 1)..t {
                    m[i * t + j] = -1e9;
                }
            }
            scores = scores.add(&Tensor::from_vec(m, &[t, t]).on(x.device()));
        }
        let attn = scores.softmax(-1);
        let out = attn.bmm(&v); // [B*H,T,Dh]
        let merged = out.reshape(&[b, h, t, dh]).permute(&[0, 2, 1, 3]).reshape(&[b, t, d]);
        self.wo.forward(&merged)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("q_proj", self.wq.named_parameters());
        v.extend(prefixed("k_proj", self.wk.named_parameters()));
        v.extend(prefixed("v_proj", self.wv.named_parameters()));
        v.extend(prefixed("out_proj", self.wo.named_parameters()));
        v
    }

    fn name(&self) -> String {
        format!("MultiHeadAttention(d={}, heads={}, causal={})", self.d_model, self.heads, self.causal)
    }
}

/// Pre-norm Transformer encoder block:
/// `x + MHA(LN(x))` followed by `x + FFN(LN(x))` with GELU.
pub struct TransformerEncoderLayer {
    pub attn: MultiHeadAttention,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub dropout: Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, heads: usize, ff_dim: usize, dropout: f32) -> Self {
        TransformerEncoderLayer {
            attn: MultiHeadAttention::new(d_model, heads),
            ln1: LayerNorm::new(d_model),
            ln2: LayerNorm::new(d_model),
            ff1: Linear::new(d_model, ff_dim),
            ff2: Linear::new(ff_dim, d_model),
            dropout: Dropout::new(dropout),
        }
    }

    pub fn causal(mut self, c: bool) -> Self {
        self.attn.causal = c;
        self
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, x: &Tensor) -> Tensor {
        let a = self.dropout.forward(&self.attn.forward(&self.ln1.forward(x)));
        let x = x.add(&a);
        let f = self.ff2.forward(&self.ff1.forward(&self.ln2.forward(&x)).gelu());
        x.add(&self.dropout.forward(&f))
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("attn", self.attn.named_parameters());
        v.extend(prefixed("ln1", self.ln1.named_parameters()));
        v.extend(prefixed("ln2", self.ln2.named_parameters()));
        v.extend(prefixed("ff1", self.ff1.named_parameters()));
        v.extend(prefixed("ff2", self.ff2.named_parameters()));
        v
    }

    fn name(&self) -> String {
        format!("TransformerEncoderLayer(d={}, heads={})", self.attn.d_model, self.attn.heads)
    }

    fn set_training(&self, t: bool) {
        self.dropout.set_training(t);
    }
}
