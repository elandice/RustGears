//! Run with `cargo run --example modern_features --no-default-features`.
use rustorch::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    manual_seed(42);
    let build = || Sequential::new().add(Linear::new(2, 8)).add(Silu).add(Linear::new(8, 1));
    let model = build();
    let evaluation = build();
    let x = Tensor::from_vec((0..64).map(|i| i as f32 / 64.0).collect(), &[32, 2]);
    let y = x.sum_dim(1, true);
    let dataset = TensorDataset::new(&x, &y);
    let mut loader = DataLoader::new(&dataset, 8).shuffle(true);
    let mut optimizer = Adam::new(model.parameters(), 0.01);
    let mut ema = ModelEMA::new(&model, 0.9);

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
    ema.copy_to(&evaluation)?;
    evaluation.eval();

    let path = std::env::temp_dir().join(format!("rustorch_modern_{}.safetensors", std::process::id()));
    save_safetensors(&evaluation, &path)?;
    let restored = build();
    load_safetensors(&restored, &path)?;
    std::fs::remove_file(path)?;
    let expected = no_grad(|| evaluation.forward(&x)).to_vec();
    let actual = no_grad(|| restored.forward(&x)).to_vec();
    assert_eq!(actual, expected);
    println!("Safetensors roundtrip and {} EMA updates verified", ema.num_updates());

    let attention = MultiHeadAttention::new(8, 2).causal(true).rope(10_000.0);
    let mut cache = KvCache::new();
    let _prompt = attention.forward_cached(&Tensor::randn(&[1, 2, 8]), &mut cache);
    let token = attention.forward_cached(&Tensor::randn(&[1, 1, 8]), &mut cache);
    assert_eq!(cache.len(), 3);
    println!("Cached attention output: {:?}, positions: {}", token.shape(), cache.len());

    let decoder = ConvTranspose2d::new(1, 2, 3).stride(2).padding(1).output_padding(1);
    let image = decoder.forward(&Tensor::ones(&[1, 1, 4, 4]));
    let pooled = AdaptiveAvgPool2d::new((1, 1)).forward(&image);
    assert_eq!(image.shape(), &[1, 2, 8, 8]);
    assert_eq!(pooled.shape(), &[1, 2, 1, 1]);
    println!("Decoder: {:?}; global average pool: {:?}", image.shape(), pooled.shape());
    Ok(())
}
