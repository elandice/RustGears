//! 2D 합성곱 신경망(CNN)을 합성 획(Stroke) 이미지 데이터셋으로 학습하고,
//! `Conv2d + BatchNorm2d + ReLU` 연산자 융합 및 INT8 양자화를 적용하여
//! NPU 런타임에 배포하는 전체 파이프라인 예제.
//!
//! 실행 방법:
//! ```bash
//! cargo run --example cnn_npu
//! ```

use rustgears::data::datasets;
use rustgears::prelude::*;

fn main() {
    println!("=== RustGears CNN Training + NPU INT8 Deployment ===");
    // 난수 재현성을 위한 시드 설정
    manual_seed(7);

    // 최적의 연산 디바이스 선택 (GPU 우선, 없을 시 CPU 폴백)
    let device = Device::best_available();
    println!("Training device: {device} ({})", device.backend().name());

    // 4개 클래스 합성 획(Stroke) 이미지 320개 생성 (8x8 해상도)
    let ds = datasets::stroke_images(320, 8, 0.08, 7);
    let (train_ds, test_ds) = ds.train_test_split(0.25, 7);
    let (x_test, y_test) = test_ds.tensors();

    // CNN 모델 정의: Conv2d + BatchNorm2d + Relu + MaxPool2d 블록 스택
    let model = Sequential::new()
        .add(Conv2d::new(1, 8, 3).padding(1))
        .add(BatchNorm2d::new(8))
        .add(Relu)
        .add(MaxPool2d::new(2)) // 8x8 -> 4x4
        .add(Conv2d::new(8, 16, 3).padding(1))
        .add(BatchNorm2d::new(16))
        .add(Relu)
        .add(MaxPool2d::new(2)) // 4x4 -> 2x2
        .add(Flatten)
        .add(Linear::new(16 * 2 * 2, 4));
    model.to(device);
    println!("{}", model.summary());

    let mut opt = Adam::new(model.parameters(), 0.015);
    let mut loader = DataLoader::new(&train_ds, 32).shuffle(true).device(device);

    // 15 에포크 학습 수행
    for epoch in 1..=15 {
        model.train();
        let mut total = 0.0;
        let mut n = 0;
        for (xb, yb) in loader.iter() {
            opt.zero_grad();
            let loss = cross_entropy(&model.forward(&xb), &yb);
            loss.backward();
            opt.step();
            total += loss.item();
            n += 1;
        }
        if epoch == 1 || epoch % 5 == 0 {
            model.eval();
            let acc = no_grad(|| accuracy(&model.forward(&x_test.to(device)), &y_test.to(device)));
            println!("Epoch {:>2}/15 | loss: {:.4} | test acc: {:.1}%", epoch, total / n as f32, acc * 100.0);
        }
    }

    // 학습 완료된 모델을 NPU (INT8 대칭 양자화) 세션으로 컴파일
    let npu_session = InferenceSession::from_module(&model, ExecutionProvider::npu_int8()).unwrap();
    println!("\n{}", npu_session.compilation_report());
    
    // NPU 시뮬레이션 환경에서 테스트 세트 추론 수행
    let preds = npu_session.run(&x_test);
    println!("NPU INT8 Test Accuracy: {:.1}%", accuracy(&preds, &y_test) * 100.0);
    // NPU 시스톨릭 MAC 어레이 텔레메트리 프로파일 출력
    println!("{}", npu_session.last_npu_profile().unwrap());
}
