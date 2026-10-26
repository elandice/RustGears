//! Repeatable CPU mini-batch benchmarks; no additional dependencies.
use rustgears::prelude::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn measure(name: &str, mut work: impl FnMut()) {
    if std::env::var("MINIBATCH_FILTER").is_ok_and(|filter| !name.contains(&filter)) {
        return;
    }
    for _ in 0..3 {
        work();
    }
    let start = Instant::now();
    let mut iterations = 0;
    while start.elapsed() < Duration::from_millis(20) {
        work();
        iterations += 1;
    }
    let iterations = (iterations * 5).max(1);
    let mut timings = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iterations {
            work();
        }
        timings.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    timings.sort_by(f64::total_cmp);
    println!("{name},{:.0},{:.0},{:.0}", timings[3], timings[0], timings[6]);
}

fn loader_case(samples: usize, width: usize, batch: usize, shuffle: bool, prefetch: bool) {
    let x = Tensor::ones(&[samples, width]);
    let y = Tensor::arange(samples);
    let dataset = TensorDataset::new(&x, &y);
    let mut loader = DataLoader::new(&dataset, batch).shuffle(shuffle).seed(42);
    let name = format!(
        "loader_n{samples}_d{width}_b{batch}_{}_{}",
        if shuffle { "shuffle" } else { "ordered" },
        if prefetch { "prefetch" } else { "sync" }
    );
    measure(&name, || {
        let mut seen = 0;
        let mut consume = |(x, y): (Tensor, Tensor)| {
            seen += x.shape()[0];
            black_box((x, y));
        };
        if prefetch {
            std::thread::scope(|scope| loader.iter_prefetch(scope, 2).for_each(&mut consume));
        } else {
            loader.iter().for_each(&mut consume);
        }
        assert_eq!(seen, samples);
    });
}

struct TransformedDataset;
impl Dataset for TransformedDataset {
    fn len(&self) -> usize {
        512
    }
    fn input_shape(&self) -> Vec<usize> {
        vec![16]
    }
    fn target_shape(&self) -> Vec<usize> {
        vec![]
    }
    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        let mut value = index as f32;
        for _ in 0..256 {
            value = black_box(value).sin() + 0.1;
        }
        (vec![value; 16], vec![0.0])
    }
}

fn main() {
    if cfg!(debug_assertions) || std::env::args().any(|argument| argument == "--test") {
        return;
    }
    // Fixed CPU worker count keeps before/after comparisons reproducible.
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    println!(
        "# CPU-only; rayon_threads=4; hardware_threads={}; 7 samples, ~100ms each; loader units=epoch, others=call",
        std::thread::available_parallelism().unwrap()
    );
    println!("case,median_ns,min_ns,max_ns");
    for &(samples, width, batch, shuffle, prefetch) in &[
        (8192, 16, 32, false, false),
        (8192, 16, 32, true, false),
        (8192, 16, 256, true, false),
        (4096, 784, 64, false, false),
        (4096, 784, 64, true, false),
        (8192, 16, 32, true, true),
    ] {
        loader_case(samples, width, batch, shuffle, prefetch);
    }

    let mut loader = DataLoader::new(&TransformedDataset, 32);
    measure("loader_custom_cpu_transform", || {
        loader.iter().for_each(|batch| {
            black_box(batch);
        });
    });

    for &(rows, width, count) in &[(4096, 16, 32), (512, 784, 256)] {
        let x = Tensor::ones(&[rows, width]);
        measure(&format!("narrow_rows_d{width}_b{count}"), || {
            black_box(x.narrow_rows(16, count));
        });
    }

    for &(batch, workers) in &[(32, 1), (1, 4), (96, 4)] {
        let model = DataParallel::new(vec![Device::Cpu; workers], || Linear::no_bias(16, 4));
        model.module().weight.copy_from_slice(&[0.1; 64]);
        model.broadcast_parameters();
        let x = Tensor::ones(&[batch, 16]);
        let y = Tensor::zeros(&[batch, 4]);
        measure(&format!("parallel_b{batch}_workers{workers}"), || {
            black_box(model.train_step(&x, &y, mse_loss));
        });
    }

    let dataset = TensorDataset::new(&Tensor::ones(&[1024, 16]), &Tensor::zeros(&[1024, 4]));
    let mut loader = DataLoader::new(&dataset, 32);
    let model = Sequential::new().add(Linear::new(16, 32)).add(Relu).add(Linear::new(32, 4));
    let mut optimizer = Sgd::new(model.parameters(), 1e-5);
    measure("training_mlp_epoch_n1024_b32", || {
        for (x, y) in loader.iter() {
            optimizer.zero_grad();
            let loss = mse_loss(&model.forward(&x), &y);
            loss.backward();
            optimizer.step();
            black_box(loss.item());
        }
    });
}
