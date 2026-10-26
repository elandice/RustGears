//! End-to-end showcase of RustGears:
//! 1. Hardware discovery & device selection (`CPU`, `GPU:0`, ...)
//! 2. Multi-GPU Data-Parallel training (`DataParallel` + `ring_all_reduce`)
//! 3. Checkpointing & portable model export (`.rtim`)
//! 4. Inference on the **NPU** (`ExecutionProvider::Npu(Int8 / Fp16)`) with
//!    operator fusion and systolic-array telemetry.
//!
//! Run with:
//! ```bash
//! cargo run --example end_to_end
//! ```

use rustgears::data::datasets;
use rustgears::prelude::*;

fn build_mlp() -> Sequential {
    Sequential::new()
        .add(Linear::new(2, 64))
        .add(BatchNorm1d::new(64))
        .add(Relu)
        .add(Linear::new(64, 64))
        .add(BatchNorm1d::new(64))
        .add(Relu)
        .add(Linear::new(64, 3))
}

fn main() {
    println!("======================================================================");
    println!("  RustGears - End-to-End Deep Learning Pipeline (CPU / GPU / NPU)");
    println!("======================================================================\n");

    // ------------------------------------------------------------------
    // 1. Hardware & device selection
    // ------------------------------------------------------------------
    println!("[1] Discovered training devices:");
    for info in list_devices() {
        println!("    * {info}");
    }
    let primary = Device::best_available();
    println!("    -> Selected primary device: {primary} ({})\n", primary.backend().name());

    // ------------------------------------------------------------------
    // 2. Dataset preparation (3-class non-linear spiral classification)
    // ------------------------------------------------------------------
    manual_seed(42);
    let dataset = datasets::spirals(240, 3, 0.18, 42);
    let (train_ds, test_ds) = dataset.train_test_split(0.2, 42);
    let (x_test, y_test) = test_ds.tensors();
    println!(
        "[2] Spiral dataset: {} train samples, {} test samples (3 classes)\n",
        train_ds.len(),
        test_ds.len()
    );

    // ------------------------------------------------------------------
    // 3. Multi-GPU Data-Parallel training (Ring All-Reduce)
    // ------------------------------------------------------------------
    let dp_devices = vec![Device::Gpu(0), Device::Gpu(1), Device::Gpu(2), Device::Gpu(3)];
    println!("[3] Initialising DataParallel across {} devices: {:?}", dp_devices.len(), dp_devices);
    let dp = DataParallel::new(dp_devices, build_mlp);
    println!("{}", dp.module().summary());

    let epochs = 35;
    let mut loader = DataLoader::new(&train_ds, 96).shuffle(true).seed(42);
    let mut opt = AdamW::new(dp.parameters(), 0.025, 1e-4);
    let mut sched = CosineAnnealingLr::new(0.025, epochs, 1e-3);

    for epoch in 1..=epochs {
        dp.train();
        let mut epoch_loss = 0.0;
        let mut steps = 0;
        let mut last_report = None;
        for (xb, yb) in loader.iter() {
            let rep = dp.fit_batch(&xb, &yb, cross_entropy, &mut opt);
            epoch_loss += rep.loss;
            steps += 1;
            last_report = Some(rep);
        }
        sched.step(&mut opt);

        if epoch == 1 || epoch % 7 == 0 || epoch == epochs {
            dp.eval();
            let test_acc = no_grad(|| accuracy(&dp.forward(&x_test), &y_test));
            let rep = last_report.unwrap();
            println!(
                "    Epoch {:>2}/{epochs} | loss: {:.4} | test acc: {:>5.1}% | lr: {:.4} | ring all-reduce: {} steps ({} B)",
                epoch,
                epoch_loss / steps as f32,
                test_acc * 100.0,
                sched.current_lr(),
                rep.allreduce.steps,
                rep.allreduce.bytes_sent
            );
        }
    }

    // ------------------------------------------------------------------
    // 4. Save checkpoint & portable inference model
    // ------------------------------------------------------------------
    let model = dp.module();
    let rtim_path = std::env::temp_dir().join("spiral_mlp.rtim");
    save_inference_model(model, &rtim_path).expect("export model");
    println!("\n[4] Exported portable inference graph to: {}", rtim_path.display());

    // ------------------------------------------------------------------
    // 5. Execute on NPU (INT8 & FP16) vs GPU vs CPU
    // ------------------------------------------------------------------
    println!("\n[5] Running inference across execution providers:");
    let layers = load_inference_layers(&rtim_path).expect("load rtim");
    let _ = std::fs::remove_file(&rtim_path);

    for provider in [
        ExecutionProvider::Cpu,
        ExecutionProvider::Gpu(0),
        ExecutionProvider::npu_fp16(),
        ExecutionProvider::npu_int8(),
    ] {
        let sess = InferenceSession::from_layers(layers.clone(), provider);
        println!("\n  {}", sess.compilation_report());
        let logits = sess.run(&x_test);
        let acc = accuracy(&logits, &y_test);
        println!("  -> Test accuracy on {provider}: {:.2}%", acc * 100.0);
        if let Some(prof) = sess.last_npu_profile() {
            print!("{prof}");
        }
    }
}
