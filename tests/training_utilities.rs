use rustgears::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn batches(loader: &mut DataLoader<'_>) -> Vec<(Vec<f32>, Vec<f32>)> {
    loader.iter().map(|(x, y)| (x.to_vec(), y.to_vec())).collect()
}

#[test]
fn prefetch_matches_sync_for_multiple_shuffled_epochs() {
    let ds = TensorDataset::new(&Tensor::arange(21).reshape(&[7, 3]), &Tensor::arange(7));
    for drop_last in [false, true] {
        let mut sync = DataLoader::new(&ds, 3).shuffle(true).seed(77).drop_last(drop_last);
        let mut asynchronous =
            DataLoader::new(&ds, 3).shuffle(true).seed(77).drop_last(drop_last).device(Device::Gpu(99));
        for _ in 0..2 {
            let expected = batches(&mut sync);
            let actual: Vec<_> = std::thread::scope(|scope| {
                asynchronous
                    .iter_prefetch(scope, 2)
                    .map(|(x, y)| {
                        assert_eq!(x.device(), Device::Gpu(99));
                        assert_eq!(y.device(), Device::Gpu(99));
                        (x.to_vec(), y.to_vec())
                    })
                    .collect()
            });
            assert_eq!(actual, expected);
        }
    }
    let empty = TensorDataset::new(&Tensor::zeros(&[0, 3]), &Tensor::zeros(&[0]));
    let mut loader = DataLoader::new(&empty, 2);
    std::thread::scope(|scope| assert!(loader.iter_prefetch(scope, 1).next().is_none()));
}

struct CountingDataset {
    reads: AtomicUsize,
    fail: bool,
}

impl Dataset for CountingDataset {
    fn len(&self) -> usize {
        100
    }
    fn input_shape(&self) -> Vec<usize> {
        vec![1]
    }
    fn target_shape(&self) -> Vec<usize> {
        vec![]
    }
    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(!self.fail, "dataset failure must reach the caller");
        self.reads.fetch_add(1, Ordering::SeqCst);
        (vec![index as f32], vec![index as f32])
    }
}

#[test]
fn prefetch_drop_cancels_a_full_queue_and_worker_errors_propagate() {
    let ds = CountingDataset { reads: AtomicUsize::new(0), fail: false };
    let mut loader = DataLoader::new(&ds, 1);
    std::thread::scope(|scope| {
        let mut iter = loader.iter_prefetch(scope, 1);
        assert_eq!(iter.next().unwrap().0.to_vec(), vec![0.0]);
        drop(iter);
    });
    assert!(ds.reads.load(Ordering::SeqCst) <= 4, "prefetch must be bounded");

    let failure = std::panic::catch_unwind(|| {
        let ds = CountingDataset { reads: AtomicUsize::new(0), fail: true };
        let mut loader = DataLoader::new(&ds, 1);
        std::thread::scope(|scope| {
            loader.iter_prefetch(scope, 1).next();
        });
    });
    assert!(failure.is_err());
}

#[test]
fn prefetch_can_be_consumed_inside_a_single_rayon_worker() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let ds = TensorDataset::new(&Tensor::arange(4).reshape(&[4, 1]), &Tensor::arange(4));
        let mut loader = DataLoader::new(&ds, 2);
        let actual: Vec<_> =
            std::thread::scope(|scope| loader.iter_prefetch(scope, 1).flat_map(|(x, _)| x.to_vec()).collect());
        assert_eq!(actual, vec![0.0, 1.0, 2.0, 3.0]);
    });
}

#[test]
fn prefetch_prepares_the_next_batch_while_consumer_is_busy() {
    struct Signals(std::sync::mpsc::Sender<()>);
    impl Dataset for Signals {
        fn len(&self) -> usize {
            2
        }
        fn input_shape(&self) -> Vec<usize> {
            vec![1]
        }
        fn target_shape(&self) -> Vec<usize> {
            vec![]
        }
        fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
            if index == 1 {
                self.0.send(()).unwrap();
            }
            (vec![index as f32], vec![0.0])
        }
    }
    let (sender, receiver) = std::sync::mpsc::channel();
    let dataset = Signals(sender);
    let mut loader = DataLoader::new(&dataset, 1);
    std::thread::scope(|scope| {
        let mut iter = loader.iter_prefetch(scope, 1);
        assert_eq!(iter.next().unwrap().0.item(), 0.0);
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("second batch should be prepared before next()");
        assert_eq!(iter.next().unwrap().0.item(), 1.0);
    });
}

#[test]
fn loader_rejects_each_invalid_sample_even_when_flattened_sizes_cancel() {
    struct InvalidSamples;
    impl Dataset for InvalidSamples {
        fn len(&self) -> usize {
            2
        }
        fn input_shape(&self) -> Vec<usize> {
            vec![1]
        }
        fn target_shape(&self) -> Vec<usize> {
            vec![]
        }
        fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
            (vec![0.0; 2 * index], vec![0.0])
        }
    }
    assert!(std::panic::catch_unwind(|| {
        DataLoader::new(&InvalidSamples, 2).iter().next();
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| {
        let mut loader = DataLoader::new(&InvalidSamples, 2);
        std::thread::scope(|scope| {
            loader.iter_prefetch(scope, 1).next();
        });
    })
    .is_err());
}

#[test]
fn ema_averages_parameters_and_buffers_without_aliasing_or_gradients() {
    let model = BatchNorm1d::new(1);
    let mut ema = ModelEMA::new(&model, 0.5);
    model.gamma.copy_from_slice(&[5.0]);
    model.beta.copy_from_slice(&[4.0]);
    model.running_mean.copy_from_slice(&[8.0]);
    model.running_var.copy_from_slice(&[9.0]);
    assert_eq!(ema.state_dict()[0].1.to_vec(), vec![1.0]);
    ema.update(&model).unwrap();
    assert_eq!(ema.num_updates(), 1);
    let target = BatchNorm1d::new(1);
    target.to(Device::Gpu(99));
    target.gamma.set_grad(Some(vec![42.0]));
    ema.copy_to(&target).unwrap();
    assert_eq!(target.gamma.to_vec(), vec![3.0]);
    assert_eq!(target.beta.to_vec(), vec![2.0]);
    assert_eq!(target.running_mean.to_vec(), vec![4.0]);
    assert_eq!(target.running_var.to_vec(), vec![5.0]);
    assert_eq!(target.gamma.device(), Device::Gpu(99));
    assert_eq!(target.gamma.grad_vec().unwrap(), vec![42.0]);
    assert!(ema.state_dict().iter().all(|(_, t)| !t.requires_grad()));
    assert_eq!(model.gamma.to_vec(), vec![5.0]);
}

#[test]
fn ema_checks_all_shapes_before_modifying_any_state() {
    struct State(Vec<(String, Tensor)>);
    impl Module for State {
        fn forward(&self, x: &Tensor) -> Tensor {
            x.clone()
        }
        fn named_parameters(&self) -> Vec<(String, Tensor)> {
            self.0.clone()
        }
    }
    let original = State(vec![("a".into(), Tensor::ones(&[1])), ("b".into(), Tensor::ones(&[2]))]);
    let mut ema = ModelEMA::new(&original, 0.9);
    let wrong = State(vec![("b".into(), Tensor::zeros(&[1, 2])), ("a".into(), Tensor::full(&[1], 99.0))]);
    assert!(ema.update(&wrong).is_err());
    assert!(ema.copy_to(&wrong).is_err());
    assert_eq!(wrong.0[1].1.to_vec(), vec![99.0]);
    assert_eq!(ema.state_dict()[0].1.to_vec(), vec![1.0]);
    assert_eq!(ema.num_updates(), 0);
}

#[test]
fn progress_bar_handles_empty_and_overflowing_progress() {
    let mut progress = ProgressBar::new(2);
    progress.set_message("loss=0.25");
    progress.inc(1).unwrap();
    assert!(progress.render().contains("1/2  50%"));
    progress.inc(usize::MAX).unwrap();
    let rendered = progress.render();
    assert!(rendered.contains("2/2 100%"));
    assert!(rendered.contains("loss=0.25"));
    assert!(!rendered.contains("NaN"));
    progress.finish().unwrap();
    let mut empty = ProgressBar::new(0);
    empty.finish().unwrap();
    assert!(empty.render().contains("0/0 100%"));
}
