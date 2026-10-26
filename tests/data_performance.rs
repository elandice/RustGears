use rustgears::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn tensor_batch_gather_matches_individual_samples() {
    let dataset = TensorDataset::new(&Tensor::arange(20).reshape(&[5, 2, 2]), &Tensor::arange(5));
    for indices in [vec![1, 2, 3], vec![4, 0, 2], vec![2, 2, 0], vec![], vec![4]] {
        let (x, y) = dataset.get_batch(&indices).unwrap();
        let mut expected_x = Vec::new();
        let mut expected_y = Vec::new();
        for index in indices {
            let (a, b) = dataset.get(index);
            expected_x.extend(a);
            expected_y.extend(b);
        }
        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}

struct SampleOnly(TensorDataset);

impl Dataset for SampleOnly {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn input_shape(&self) -> Vec<usize> {
        self.0.input_shape()
    }
    fn target_shape(&self) -> Vec<usize> {
        self.0.target_shape()
    }
    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        self.0.get(index)
    }
}

#[test]
fn fast_batch_loader_preserves_shuffle_epochs_last_batch_and_device() {
    let x = Tensor::arange(28).reshape(&[7, 2, 2]);
    let y = Tensor::arange(7);
    let fast = TensorDataset::new(&x, &y);
    let sample_only = SampleOnly(TensorDataset::new(&x, &y));
    for shuffle in [false, true] {
        for drop_last in [false, true] {
            let mut reference = DataLoader::new(&sample_only, 3).shuffle(shuffle).seed(29).drop_last(drop_last);
            let mut sync =
                DataLoader::new(&fast, 3).shuffle(shuffle).seed(29).drop_last(drop_last).device(Device::Gpu(99));
            let mut asynchronous =
                DataLoader::new(&fast, 3).shuffle(shuffle).seed(29).drop_last(drop_last).device(Device::Gpu(99));
            for _ in 0..2 {
                let expected: Vec<_> =
                    reference.iter().map(|(a, b)| (a.shape().to_vec(), a.to_vec(), b.to_vec())).collect();
                let actual: Vec<_> = sync
                    .iter()
                    .map(|(a, b)| {
                        assert_eq!(a.device(), Device::Gpu(99));
                        assert_eq!(b.device(), Device::Gpu(99));
                        (a.shape().to_vec(), a.to_vec(), b.to_vec())
                    })
                    .collect();
                assert_eq!(actual, expected);
                let prefetched: Vec<_> = std::thread::scope(|scope| {
                    asynchronous
                        .iter_prefetch(scope, 2)
                        .map(|(a, b)| {
                            assert_eq!(a.device(), Device::Gpu(99));
                            assert_eq!(b.device(), Device::Gpu(99));
                            (a.shape().to_vec(), a.to_vec(), b.to_vec())
                        })
                        .collect()
                });
                assert_eq!(prefetched, expected);
                assert_eq!(actual.len(), if drop_last { 2 } else { 3 });
            }
        }
    }
}

#[test]
fn loader_uses_optional_batch_gather_without_per_sample_reads() {
    struct BatchOnly(AtomicUsize);
    impl Dataset for BatchOnly {
        fn len(&self) -> usize {
            5
        }
        fn input_shape(&self) -> Vec<usize> {
            vec![1]
        }
        fn target_shape(&self) -> Vec<usize> {
            vec![]
        }
        fn get(&self, _: usize) -> (Vec<f32>, Vec<f32>) {
            panic!("batch gather must bypass per-sample reads")
        }
        fn get_batch(&self, indices: &[usize]) -> Option<(Vec<f32>, Vec<f32>)> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Some((indices.iter().map(|&i| i as f32).collect(), vec![1.0; indices.len()]))
        }
    }
    let dataset = BatchOnly(AtomicUsize::new(0));
    let mut loader = DataLoader::new(&dataset, 2);
    let x: Vec<_> = loader.iter().flat_map(|(x, _)| x.to_vec()).collect();
    assert_eq!(x, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dataset.0.load(Ordering::Relaxed), 3);
    std::thread::scope(|scope| assert_eq!(loader.iter_prefetch(scope, 1).count(), 3));
    assert_eq!(dataset.0.load(Ordering::Relaxed), 6);
}

#[test]
fn loader_validates_batch_override_lengths() {
    struct InvalidBatch;
    impl Dataset for InvalidBatch {
        fn len(&self) -> usize {
            2
        }
        fn input_shape(&self) -> Vec<usize> {
            vec![1]
        }
        fn target_shape(&self) -> Vec<usize> {
            vec![]
        }
        fn get(&self, _: usize) -> (Vec<f32>, Vec<f32>) {
            unreachable!()
        }
        fn get_batch(&self, _: &[usize]) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![0.0], vec![0.0; 2]))
        }
    }
    assert!(std::panic::catch_unwind(|| {
        DataLoader::new(&InvalidBatch, 2).iter().next();
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| {
        let mut loader = DataLoader::new(&InvalidBatch, 2);
        std::thread::scope(|scope| {
            loader.iter_prefetch(scope, 1).next();
        });
    })
    .is_err());
}

#[test]
fn batch_gather_checks_indices_when_samples_are_empty() {
    let empty = TensorDataset::new(&Tensor::zeros(&[0, usize::MAX, usize::MAX]), &Tensor::zeros(&[0]));
    assert_eq!(empty.get_batch(&[]).unwrap(), (vec![], vec![]));
    let dataset = TensorDataset::new(&Tensor::zeros(&[2, 0]), &Tensor::zeros(&[2, 0]));
    assert_eq!(dataset.get_batch(&[1, 0]).unwrap(), (vec![], vec![]));
    assert!(std::panic::catch_unwind(|| dataset.get_batch(&[2])).is_err());
}

#[test]
fn malformed_sample_is_rejected_before_allocating_declared_batch_capacity() {
    struct InvalidShape;
    impl Dataset for InvalidShape {
        fn len(&self) -> usize {
            1
        }
        fn input_shape(&self) -> Vec<usize> {
            vec![usize::MAX]
        }
        fn target_shape(&self) -> Vec<usize> {
            vec![]
        }
        fn get(&self, _: usize) -> (Vec<f32>, Vec<f32>) {
            (vec![], vec![0.0])
        }
    }
    let panic = std::panic::catch_unwind(|| {
        DataLoader::new(&InvalidShape, 1).iter().next();
    })
    .unwrap_err();
    let message = panic.downcast_ref::<String>().unwrap();
    assert!(message.contains("Dataset::get returned the wrong input size"));
}
