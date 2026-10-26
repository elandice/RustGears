//! 2-D Convolutional Neural Network trained on synthetic stroke images and
//! deployed to the NPU with `Conv2d + BatchNorm2d + ReLU` operator fusion and
//! INT8 quantisation.
//!
//! Run with:
//! ```bash
//! cargo run --example cnn_npu
//! ```

use rustorch::data::datasets;
use rustorch::prelude::*;

fn main() {
    println!("=== RusTorch CNN Training + NPU INT8 Deployment ===");
    manual_seed(7);

    let device = Device::best_available();
    println!("Training device: {device} ({})", device.backend().name());

    let ds = datasets::stroke_images(320, 8, 0.08, 7);
    let (train_ds, test_ds) = ds.train_test_split(0.25, 7);
    let (x_test, y_test) = test_ds.tensors();

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

    // Compile and run on the NPU (INT8)
    let npu_session = InferenceSession::from_module(&model, ExecutionProvider::npu_int8()).unwrap();
    println!("\n{}", npu_session.compilation_report());
    let preds = npu_session.run(&x_test);
    println!("NPU INT8 Test Accuracy: {:.1}%", accuracy(&preds, &y_test) * 100.0);
    println!("{}", npu_session.last_npu_profile().unwrap());
}
