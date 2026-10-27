//! RustGears의 최신 딥러닝 기능 종합 시연:
//! - 멀티스레드 비동기 데이터 프리패치 (`iter_prefetch`)
//! - 모델 가중치 지수 이동 평균 (`ModelEMA`)
//! - 터미널 프로그레스 바 (`ProgressBar`)
//! - Hugging Face 호환 Safetensors 저장 및 복원
//! - RoPE 회전 위치 인코딩 및 KV 캐시 지원 인과적 어텐션 (`MultiHeadAttention` + `KvCache`)
//! - 전치 합성곱 디코더 (`ConvTranspose2d`) 및 전역 평균 풀링 (`AdaptiveAvgPool2d`)
//!
//! 실행 방법:
//! ```bash
//! cargo run --example modern_features --no-default-features
//! ```

use rustgears::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    manual_seed(42);
    // 소형 MLP 빌더 클로저
    let build = || Sequential::new().add(Linear::new(2, 8)).add(Silu).add(Linear::new(8, 1));
    let model = build();
    let evaluation = build();
    let x = Tensor::from_vec((0..64).map(|i| i as f32 / 64.0).collect(), &[32, 2]);
    let y = x.sum_dim(1, true);
    let dataset = TensorDataset::new(&x, &y);
    let mut loader = DataLoader::new(&dataset, 8).shuffle(true);
    let mut optimizer = Adam::new(model.parameters(), 0.01);
    let mut ema = ModelEMA::new(&model, 0.9);

    // 1. 프로그레스 바 및 백그라운드 프리패치(iter_prefetch) 학습 루프
    for _ in 0..3 {
        let mut progress = ProgressBar::new(loader.num_batches());
        std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            for (inputs, targets) in loader.iter_prefetch(scope, 2) {
                optimizer.zero_grad();
                let loss = mse_loss(&model.forward(&inputs), &targets);
                loss.backward();
                optimizer.step();
                ema.update(&model)?;
                progress.set_message(format!("loss={:.4}", loss.item()));
                progress.inc(1)?;
            }
            Ok(())
        })?;
        progress.finish()?;
        println!("{}", progress.render());
    }

    // 2. EMA 추적 가중치를 평가용 모델로 전송
    ema.copy_to(&evaluation)?;
    evaluation.eval();

    // 3. Safetensors 포맷으로 가중치 저장 및 복원 라운드트립 검증
    let path = std::env::temp_dir().join(format!("rustgears_modern_{}.safetensors", std::process::id()));
    save_safetensors(&evaluation, &path)?;
    let restored = build();
    load_safetensors(&restored, &path)?;
    std::fs::remove_file(path)?;
    let expected = no_grad(|| evaluation.forward(&x)).to_vec();
    let actual = no_grad(|| restored.forward(&x)).to_vec();
    assert_eq!(actual, expected);
    println!("Safetensors roundtrip and {} EMA updates verified", ema.num_updates());

    // 4. RoPE 회전 위치 인코딩 및 KV 캐시를 활용한 점진적 토큰 생성 추론
    let attention = MultiHeadAttention::new(8, 2).causal(true).rope(10_000.0);
    let mut cache = KvCache::new();
    let _prompt = attention.forward_cached(&Tensor::randn(&[1, 2, 8]), &mut cache);
    let token = attention.forward_cached(&Tensor::randn(&[1, 1, 8]), &mut cache);
    assert_eq!(cache.len(), 3);
    println!("Cached attention output: {:?}, positions: {}", token.shape(), cache.len());

    // 5. 전치 합성곱(ConvTranspose2d) 기반 업샘플링 디코더 및 AdaptiveAvgPool2d
    let decoder = ConvTranspose2d::new(1, 2, 3).stride(2).padding(1).output_padding(1);
    let image = decoder.forward(&Tensor::ones(&[1, 1, 4, 4]));
    let pooled = AdaptiveAvgPool2d::new((1, 1)).forward(&image);
    assert_eq!(image.shape(), &[1, 2, 8, 8]);
    assert_eq!(pooled.shape(), &[1, 2, 1, 1]);
    println!("Decoder: {:?}; global average pool: {:?}", image.shape(), pooled.shape());
    Ok(())
}
