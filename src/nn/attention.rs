//! 어텐션 및 트랜스포머(Transformer) 빌딩 블록 모듈.

use super::{prefixed, Dropout, LayerNorm, Linear, Module};
use crate::autograd::no_grad;
use crate::tensor::Tensor;

/// LLaMA 등에서 사용하는 분할 반전 쌍(Split-half pairs) 방식의 회전 위치 임베딩 (RoPE: Rotary Position Embedding).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryEmbedding {
    dim: usize,
    base: f32,
}

impl RotaryEmbedding {
    /// 지정된 헤드 차원과 기저값(base, 일반적으로 10000.0)으로 새 RoPE 임베딩을 생성합니다.
    pub fn new(dim: usize, base: f32) -> Self {
        assert!(dim > 0 && dim.is_multiple_of(2), "RoPE requires a positive even head dimension");
        assert!(base.is_finite() && base > 0.0, "RoPE base must be finite and positive");
        Self { dim, base }
    }

    /// 절대 토큰 위치 오프셋(`position_offset`)을 기준으로 `[..., sequence, head_dim]` 텐서를 회전 변환합니다.
    /// 디바이스 위치를 보존하며 역전파를 완벽히 지원합니다.
    pub fn forward(&self, x: &Tensor, position_offset: usize) -> Tensor {
        assert!(x.ndim() >= 2, "RoPE expects at least two dimensions");
        assert_eq!(x.size(-1), self.dim, "RoPE head dimension mismatch");
        let t = x.size(-2);
        position_offset.checked_add(t).expect("RoPE position overflow");
        let half = self.dim / 2;
        let angles: Vec<(f32, f32)> = (0..t).flat_map(|position| {
            (0..half).map(move |i| {
                let angle = (position_offset + position) as f32 / self.base.powf((2 * i) as f32 / self.dim as f32);
                let (sin, cos) = angle.sin_cos();
                (sin, cos)
            })
        }).collect();
        let mut data = x.to_vec();
        for (row, values) in data.chunks_exact_mut(self.dim).enumerate() {
            for i in 0..half {
                let (sin, cos) = angles[(row % t) * half + i];
                let (a, b) = (values[i], values[i + half]);
                values[i] = a * cos - b * sin;
                values[i + half] = a * sin + b * cos;
            }
        }
        let dim = self.dim;
        Tensor::from_op(data, x.shape().to_vec(), x.device(), vec![x.clone()], "RoPEBackward", Box::new(move |g| {
            let mut dx = g.to_vec();
            for (row, values) in dx.chunks_exact_mut(dim).enumerate() {
                for i in 0..half {
                    let (sin, cos) = angles[(row % t) * half + i];
                    let (a, b) = (values[i], values[i + half]);
                    values[i] = a * cos + b * sin;
                    values[i + half] = b * cos - a * sin;
                }
            }
            vec![Some(dx)]
        }))
    }
}

/// 레이어별 추론 가속용 Key-Value 캐시 (KV Cache).
/// 어텐션 레이어마다 별도의 캐시를 사용해야 하며, 새로운 시퀀스 생성을 시작하거나 모델 가중치를 바꿀 때는 [`clear`](Self::clear)를 호출해야 합니다.
#[derive(Default)]
pub struct KvCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
    layer_id: Option<usize>,
    batch: usize,
    heads: usize,
    rotary: Option<RotaryEmbedding>,
}

impl KvCache {
    /// 비어 있는 새 KV 캐시 인스턴스를 생성합니다.
    pub fn new() -> Self {
        Self::default()
    }

    /// 현재 캐시된 토큰 시퀀스 길이(토큰 위치 개수)를 반환합니다.
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.shape()[1])
    }

    /// 캐시가 비어 있는지 확인합니다.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 캐시를 완전히 비우고 초기화합니다.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// 이전 K/V 텐서 뒤에 새로 생성된 현재 토큰의 K/V 텐서를 덧붙입니다.
    fn append(previous: Option<&Tensor>, current: &Tensor) -> Tensor {
        let Some(previous) = previous else { return current.clone() };
        assert_eq!(previous.device(), current.device(), "KV cache device changed; clear the cache first");
        let (bh, old_t, dim) = (previous.shape()[0], previous.shape()[1], previous.shape()[2]);
        let t = current.shape()[1];
        assert_eq!(current.shape(), &[bh, t, dim], "KV cache shape changed; clear the cache first");
        let old = previous.data();
        let new = current.data();
        let mut data = Vec::with_capacity(bh * (old_t + t) * dim);
        for head in 0..bh {
            data.extend_from_slice(&old[head * old_t * dim..(head + 1) * old_t * dim]);
            data.extend_from_slice(&new[head * t * dim..(head + 1) * t * dim]);
        }
        Tensor::from_vec(data, &[bh, old_t + t, dim]).on(current.device())
    }
}

/// 다중 헤드 스케일드 닷 프로덕트 셀프 어텐션 (Multi-Head Attention: MHA).
///
/// 입력 / 출력 형태: `[batch, seq_len, d_model]`.
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub heads: usize,
    pub d_model: usize,
    /// GPT 스타일 디코더처럼 미래 토큰 참조를 차단하는 인과적(causal / look-ahead) 마스크 적용 여부
    pub causal: bool,
    rotary: Option<RotaryEmbedding>,
}

impl MultiHeadAttention {
    /// 임베딩 차원 `d_model`과 어텐션 헤드 개수 `heads`로 새 어텐션 모듈을 생성합니다.
    pub fn new(d_model: usize, heads: usize) -> Self {
        assert!(d_model > 0 && heads > 0, "d_model and heads must be positive");
        assert_eq!(d_model % heads, 0, "d_model must be divisible by heads");
        MultiHeadAttention {
            wq: Linear::new(d_model, d_model),
            wk: Linear::new(d_model, d_model),
            wv: Linear::new(d_model, d_model),
            wo: Linear::new(d_model, d_model),
            heads,
            d_model,
            causal: false,
            rotary: None,
        }
    }

    /// 인과적(Causal) 마스킹 사용 여부를 설정합니다 (빌더 패턴).
    pub fn causal(mut self, c: bool) -> Self {
        self.causal = c;
        self
    }

    /// Query 및 Key에 적용할 RoPE(Rotary Position Embedding)를 활성화합니다 (`10000.0`이 일반적인 base 값).
    pub fn rope(mut self, base: f32) -> Self {
        self.rotary = Some(RotaryEmbedding::new(self.d_model / self.heads, base));
        self
    }

    fn validate_input(&self, x: &Tensor) {
        assert_eq!(x.ndim(), 3, "attention expects [batch, sequence, d_model]");
        assert!(self.heads > 0 && self.d_model > 0 && self.d_model.is_multiple_of(self.heads), "invalid attention dimensions");
        assert_eq!(x.shape()[2], self.d_model, "attention d_model mismatch");
        assert!(x.shape()[0] > 0 && x.shape()[1] > 0, "attention requires a nonempty batch and sequence");
    }

    /// 입력을 Q, K, V 헤드로 투영하고 필요한 경우 RoPE를 적용합니다.
    fn projections(&self, x: &Tensor, position_offset: usize) -> (Tensor, Tensor, Tensor) {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (h, dh) = (self.heads, d / self.heads);
        let split = |z: Tensor| z.reshape(&[b, t, h, dh]).permute(&[0, 2, 1, 3]).reshape(&[b * h, t, dh]);
        let mut q = split(self.wq.forward(x));
        let mut k = split(self.wk.forward(x));
        let v = split(self.wv.forward(x));
        if let Some(rotary) = &self.rotary {
            q = rotary.forward(&q, position_offset);
            k = rotary.forward(&k, position_offset);
        }
        (q, k, v)
    }

    /// 스케일드 닷 프로덕트 어텐션 점수를 계산하고 Softmax 및 가중치 합을 수행합니다.
    fn attend(&self, x: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor, position_offset: usize) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], self.d_model);
        let (h, dh, key_t) = (self.heads, d / self.heads, k.shape()[1]);
        let mut scores = q.bmm(&k.transpose(1, 2)).mul_scalar(1.0 / (dh as f32).sqrt());
        if self.causal && t > 1 {
            let mut mask = vec![0.0; t * key_t];
            for i in 0..t {
                for j in (position_offset + i + 1)..key_t {
                    mask[i * key_t + j] = f32::NEG_INFINITY;
                }
            }
            scores = scores.add(&Tensor::from_vec(mask, &[t, key_t]).on(x.device()));
        }
        let out = scores.softmax(-1).bmm(v);
        let merged = out.reshape(&[b, h, t, dh]).permute(&[0, 2, 1, 3]).reshape(&[b, t, d]);
        self.wo.forward(&merged)
    }

    /// 추론 전용: 새로운 입력 토큰에 대해 기존 KV 캐시를 재활용하여 인과적 어텐션을 고속 수행합니다.
    /// 토큰 위치는 캐시 길이에 맞춰 자동으로 누적 전진합니다.
    pub fn forward_cached(&self, x: &Tensor, cache: &mut KvCache) -> Tensor {
        self.validate_input(x);
        assert!(self.causal, "KV caching requires causal attention");
        if let Some(layer_id) = cache.layer_id {
            assert_eq!(layer_id, self.wk.weight.id(), "KV cache belongs to another attention layer");
            assert_eq!(cache.batch, x.shape()[0], "KV cache batch changed; clear the cache first");
            assert_eq!(cache.heads, self.heads, "KV cache head count changed; clear the cache first");
            assert_eq!(cache.rotary, self.rotary, "KV cache RoPE changed; clear the cache first");
        }
        let position_offset = cache.len();
        position_offset.checked_add(x.shape()[1]).expect("KV cache position overflow");
        no_grad(|| {
            let (q, k, v) = self.projections(x, position_offset);
            let keys = KvCache::append(cache.keys.as_ref(), &k);
            let values = KvCache::append(cache.values.as_ref(), &v);
            let output = self.attend(x, &q, &keys, &values, position_offset);
            cache.keys = Some(keys);
            cache.values = Some(values);
            cache.layer_id = Some(self.wk.weight.id());
            cache.batch = x.shape()[0];
            cache.heads = self.heads;
            cache.rotary = self.rotary;
            output
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.validate_input(x);
        let (q, k, v) = self.projections(x, 0);
        self.attend(x, &q, &k, &v, 0)
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

/// Pre-LayerNorm 구조의 트랜스포머 인코더 블록:
/// `x + Dropout(MHA(LN1(x)))` 및 `x + Dropout(FFN(LN2(x)))` (GELU 활성화 함수 적용).
pub struct TransformerEncoderLayer {
    pub attn: MultiHeadAttention,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub dropout: Dropout,
}

impl TransformerEncoderLayer {
    /// 새 트랜스포머 인코더 블록을 생성합니다.
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

    /// 인과적(Causal) 마스킹 사용 여부를 설정합니다.
    pub fn causal(mut self, c: bool) -> Self {
        self.attn.causal = c;
        self
    }

    /// RoPE 위치 임베딩을 활성화합니다.
    pub fn rope(mut self, base: f32) -> Self {
        self.attn = self.attn.rope(base);
        self
    }

    /// 드롭아웃을 끄고 KV 캐시를 활용하여 고속 추론을 수행합니다.
    pub fn forward_cached(&self, x: &Tensor, cache: &mut KvCache) -> Tensor {
        no_grad(|| {
            let a = self.attn.forward_cached(&self.ln1.forward(x), cache);
            let x = x.add(&a);
            let f = self.ff2.forward(&self.ff1.forward(&self.ln2.forward(&x)).gelu());
            x.add(&f)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tolerance, "{a} != {e}");
        }
    }

    fn sequence_slice(x: &Tensor, start: usize, len: usize) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let values = x.data();
        let mut data = Vec::with_capacity(b * len * d);
        for batch in 0..b {
            data.extend_from_slice(&values[(batch * t + start) * d..(batch * t + start + len) * d]);
        }
        Tensor::from_vec(data, &[b, len, d]).on(x.device())
    }

    #[test]
    fn rope_absolute_positions_and_gradients() {
        let rope = RotaryEmbedding::new(4, 10_000.0);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).requires_grad_(true);
        assert_close(&rope.forward(&x, 0).to_vec(), &x.to_vec(), 1e-6);
        let rotated = rope.forward(&x, 3);
        let (s, c) = 3.0f32.sin_cos();
        let (s2, c2) = 0.03f32.sin_cos();
        assert_close(&rotated.to_vec(), &[c - 3.0 * s, 2.0 * c2 - 4.0 * s2, s + 3.0 * c, 2.0 * s2 + 4.0 * c2], 1e-6);
        let weights = Tensor::from_slice(&[0.7, -0.3, 1.1, 0.2], &[1, 1, 4]);
        rotated.mul(&weights).sum().backward();
        let gradient = x.grad_vec().unwrap();
        for i in 0..x.numel() {
            let mut plus = x.to_vec();
            let mut minus = plus.clone();
            plus[i] += 1e-3;
            minus[i] -= 1e-3;
            let eval = |data| rope.forward(&Tensor::from_vec(data, x.shape()), 3).mul(&weights).sum().item();
            assert_close(&[gradient[i]], &[(eval(plus) - eval(minus)) / 2e-3], 5e-4);
        }
    }

    #[test]
    fn cached_attention_matches_full_sequence_with_and_without_rope() {
        let x = Tensor::from_vec((0..80).map(|i| (i as f32 * 0.3).sin()).collect(), &[2, 5, 8]);
        for use_rope in [false, true] {
            let mut attention = MultiHeadAttention::new(8, 2).causal(true);
            if use_rope {
                attention = attention.rope(10_000.0);
            }
            let expected = no_grad(|| attention.forward(&x));
            let mut cache = KvCache::new();
            let mut actual = vec![0.0; x.numel()];
            let mut start = 0;
            for len in [2, 2, 1] {
                let chunk = attention.forward_cached(&sequence_slice(&x, start, len), &mut cache);
                assert!(!chunk.requires_grad());
                let data = chunk.data();
                for batch in 0..2 {
                    actual[(batch * 5 + start) * 8..(batch * 5 + start + len) * 8]
                        .copy_from_slice(&data[batch * len * 8..(batch + 1) * len * 8]);
                }
                start += len;
                assert_eq!(cache.len(), start);
            }
            assert_close(&actual, &expected.to_vec(), 2e-5);
            cache.clear();
            assert!(cache.is_empty());
            assert_close(&attention.forward_cached(&x, &mut cache).to_vec(), &expected.to_vec(), 2e-5);
        }
    }

    #[test]
    fn cached_transformer_matches_evaluation() {
        let block = TransformerEncoderLayer::new(8, 2, 12, 0.5).causal(true).rope(10_000.0);
        block.eval();
        let x = Tensor::from_vec((0..32).map(|i| (i as f32 * 0.4).cos()).collect(), &[1, 4, 8]);
        let expected = no_grad(|| block.forward(&x));
        let mut cache = KvCache::new();
        let mut actual = Vec::new();
        for position in 0..4 {
            actual.extend(block.forward_cached(&sequence_slice(&x, position, 1), &mut cache).to_vec());
        }
        assert_close(&actual, &expected.to_vec(), 2e-5);
    }

    #[test]
    fn causal_mask_excludes_arbitrarily_large_future_scores() {
        let attention = MultiHeadAttention::new(2, 1).causal(true);
        for projection in [&attention.wq, &attention.wk, &attention.wv, &attention.wo] {
            projection.weight.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            projection.bias.as_ref().unwrap().copy_from_slice(&[0.0, 0.0]);
        }
        let x = Tensor::from_slice(&[1.0, 0.0, 1e11, 0.0], &[1, 2, 2]);
        assert_close(&attention.forward(&x).to_vec()[..2], &[1.0, 0.0], 1e-6);
    }

    #[test]
    #[should_panic(expected = "belongs to another attention layer")]
    fn rejects_reusing_cache_for_another_layer() {
        let a = MultiHeadAttention::new(4, 1).causal(true);
        let b = MultiHeadAttention::new(4, 1).causal(true);
        let x = Tensor::ones(&[1, 1, 4]);
        let mut cache = KvCache::new();
        a.forward_cached(&x, &mut cache);
        b.forward_cached(&x, &mut cache);
    }

    #[test]
    #[should_panic(expected = "batch changed")]
    fn rejects_cache_batch_changes() {
        let attention = MultiHeadAttention::new(4, 1).causal(true);
        let mut cache = KvCache::new();
        attention.forward_cached(&Tensor::ones(&[1, 1, 4]), &mut cache);
        attention.forward_cached(&Tensor::ones(&[2, 1, 4]), &mut cache);
    }

    #[test]
    #[should_panic(expected = "requires causal attention")]
    fn rejects_noncausal_cache() {
        MultiHeadAttention::new(4, 1).forward_cached(&Tensor::ones(&[1, 1, 4]), &mut KvCache::new());
    }
}
