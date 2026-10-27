//! RustGears 엔드투엔드(End-to-End) 딥러닝 워크플로 쇼케이스:
//! 1. 하드웨어 탐색 및 디바이스 선택 (`CPU`, `GPU:0`, ...)
//! 2. 다중 가상 디바이스 기반 데이터 병렬 학습 (`DataParallel` + `ring_all_reduce`)
//! 3. 체크포인트 저장 및 이식 가능한 추론 모델 내보내기 (`.rtim`)
//! 4. 연산자 융합 및 시스톨릭 어레이 텔레메트리를 지원하는 **NPU** 추론 (`ExecutionProvider::Npu(Int8 / Fp16)`)
//!
//! 실행 방법:
//! ```bash
//! cargo run --example end_to_end
//! ```

use rustgears::data::datasets;
use rustgears::prelude::*;

/// 3개 클래스 분류를 위한 다층 퍼셉트론(MLP) 모델 빌더 함수.
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
    // 1. 하드웨어 탐색 및 연산 디바이스 선택
    // ------------------------------------------------------------------
    println!("[1] Discovered training devices:");
    for info in list_devices() {
        println!("    * {info}");
    }
    let primary = Device::best_available();
    println!("    -> Selected primary device: {primary} ({})\n", primary.backend().name());

    // ------------------------------------------------------------------
    // 2. 데이터셋 준비 (3-클래스 비선형 나선형 데이터)
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
    // 3. 다중 디바이스 데이터 병렬 학습 (Ring All-Reduce)
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
            // fit_batch: scatter -> parallel forward/backward -> ring all-reduce -> opt.step -> broadcast
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
    // 4. 체크포인트 및 이식 가능한 추론 전용 모델(.rtim) 내보내기
    // ------------------------------------------------------------------
    let model = dp.module();
    let rtim_path = std::env::temp_dir().join("spiral_mlp.rtim");
    save_inference_model(model, &rtim_path).expect("export model");
    println!("\n[4] Exported portable inference graph to: {}", rtim_path.display());

    // ------------------------------------------------------------------
    // 5. 다양한 실행 공급자(CPU, GPU, NPU FP16, NPU INT8)에서 추론 실행 비교
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
