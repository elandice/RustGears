//! 소형 트랜스포머 인코더(`Embedding` + `TransformerEncoderLayer`, `MultiHeadAttention`, `LayerNorm`, `GELU`)를
//! 활용한 시퀀스 정렬(Sequence Sorting) 신경망 학습 예제.
//!
//! 실행 방법:
//! ```bash
//! cargo run --example transformer
//! ```

use rustgears::data::datasets;
use rustgears::prelude::*;

/// 정수 시퀀스를 오름차순으로 정렬하는 소형 트랜스포머 인코더 모델.
struct SortTransformer {
    tok_emb: Embedding,
    pos_emb: Embedding,
    block: TransformerEncoderLayer,
    ln: LayerNorm,
    head: Linear,
    seq_len: usize,
}

impl SortTransformer {
    /// 어휘 크기, 시퀀스 길이, 은닉 차원(`d_model`), 어텐션 헤드 수를 받아 모델을 구성합니다.
    fn new(vocab: usize, seq_len: usize, d_model: usize, heads: usize) -> Self {
        SortTransformer {
            tok_emb: Embedding::new(vocab, d_model),
            pos_emb: Embedding::new(seq_len, d_model),
            block: TransformerEncoderLayer::new(d_model, heads, d_model * 2, 0.0),
            ln: LayerNorm::new(d_model),
            head: Linear::new(d_model, vocab),
            seq_len,
        }
    }
}

impl Module for SortTransformer {
    /// 시퀀스 입력에 대해 토큰 임베딩과 위치 임베딩을 더한 뒤, 인코더 블록과 분류 헤드를 통과시킵니다.
    fn forward(&self, x: &Tensor) -> Tensor {
        let pos = Tensor::arange(self.seq_len).reshape(&[1, self.seq_len]).on(x.device());
        let h = self.tok_emb.forward(x).add(&self.pos_emb.forward(&pos));
        let h = self.ln.forward(&self.block.forward(&h));
        self.head.forward(&h)
    }

    /// 계층적 파라미터 이름을 지정하여 반환합니다.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = rustgears::nn::prefixed("tok_emb", self.tok_emb.named_parameters());
        v.extend(rustgears::nn::prefixed("pos_emb", self.pos_emb.named_parameters()));
        v.extend(rustgears::nn::prefixed("block", self.block.named_parameters()));
        v.extend(rustgears::nn::prefixed("ln", self.ln.named_parameters()));
        v.extend(rustgears::nn::prefixed("head", self.head.named_parameters()));
        v
    }
}

fn main() {
    println!("=== RustGears Transformer Sequence-Sorting Demo ===");
    manual_seed(10);

    // 0~7 범위의 토큰으로 구성된 길이 5의 시퀀스 정렬 데이터셋 생성
    let (vocab, seq_len) = (8, 5);
    let ds = datasets::sort_sequences(400, seq_len, vocab, 10);
    let (train_ds, test_ds) = ds.train_test_split(0.2, 10);
    let (x_test, y_test) = test_ds.tensors();

    let model = SortTransformer::new(vocab, seq_len, 32, 4);
    println!("Trainable parameters: {}", model.num_parameters());

    let mut opt = AdamW::new(model.parameters(), 0.01, 1e-4);
    let mut loader = DataLoader::new(&train_ds, 32).shuffle(true);

    // 20 에포크 학습
    for epoch in 1..=20 {
        for (xb, yb) in loader.iter() {
            opt.zero_grad();
            let logits = model.forward(&xb).reshape(&[xb.shape()[0] * seq_len, vocab]);
            let targets = yb.reshape(&[yb.numel()]);
            let loss = cross_entropy(&logits, &targets);
            loss.backward();
            // 안정적인 트랜스포머 학습을 위한 기울기 노름 클리핑
            clip_grad_norm(&model.parameters(), 1.0);
            opt.step();
        }
        if epoch == 1 || epoch % 5 == 0 {
            let acc = no_grad(|| {
                let logits = model.forward(&x_test).reshape(&[x_test.shape()[0] * seq_len, vocab]);
                accuracy(&logits, &y_test.reshape(&[y_test.numel()]))
            });
            println!("Epoch {:>2}/20 | token sorting accuracy: {:.1}%", epoch, acc * 100.0);
        }
    }

    // 학습된 모델의 정렬 추론 테스트
    let sample = Tensor::from_slice(&[6.0, 1.0, 4.0, 0.0, 3.0], &[1, seq_len]);
    let pred = no_grad(|| model.forward(&sample).reshape(&[seq_len, vocab]).argmax());
    println!("Input tokens : {:?}", sample.to_vec().iter().map(|&v| v as usize).collect::<Vec<_>>());
    println!("Sorted output: {:?}", pred);
}
