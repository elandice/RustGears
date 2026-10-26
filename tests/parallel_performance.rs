use rustgears::autograd::is_grad_enabled;
use rustgears::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

struct Tracked {
    used: Tensor,
    unused: Tensor,
    resets: Arc<Mutex<Vec<ThreadId>>>,
}

impl Module for Tracked {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.mul(&self.used)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("used".into(), self.used.clone()), ("unused".into(), self.unused.clone())]
    }

    fn zero_grad(&self) {
        self.resets.lock().unwrap().push(thread::current().id());
        for parameter in self.parameters() {
            parameter.zero_grad();
        }
    }
}

fn tracked(devices: usize, resets: Arc<Mutex<Vec<ThreadId>>>) -> DataParallel<Tracked> {
    DataParallel::new(vec![Device::Cpu; devices], || {
        let used = Tensor::scalar(2.0).requires_grad_(true);
        let unused = Tensor::scalar(7.0).requires_grad_(true);
        used.set_grad(Some(vec![f32::NAN]));
        unused.set_grad(Some(vec![f32::NAN]));
        Tracked { used, unused, resets: resets.clone() }
    })
}

#[test]
fn one_device_runs_inline_and_preserves_unused_gradients_and_detached_inputs() {
    let resets = Arc::new(Mutex::new(Vec::new()));
    let model = tracked(1, resets.clone());
    let x = Tensor::from_slice(&[1.0, 2.0], &[2, 1]).requires_grad_(true);
    let y = Tensor::zeros_like(&x).requires_grad_(true);
    let report = model.train_step(&x, &y, mse_loss);
    assert_eq!(*resets.lock().unwrap(), vec![thread::current().id()]);
    assert_eq!(report.loss, 10.0);
    assert_eq!(model.module().used.grad_vec().unwrap(), vec![10.0]);
    assert_eq!(model.module().unused.grad_vec().unwrap(), vec![0.0]);
    assert!(x.grad().is_none() && y.grad().is_none());
    assert_eq!(report.allreduce.workers, 1);
    assert_eq!(report.allreduce.elements, 2);
    assert_eq!(report.allreduce.steps, 0);
    assert_eq!(report.allreduce.bytes_sent, 0);
}

#[test]
fn one_device_under_no_grad_keeps_worker_recording_and_caller_state() {
    let resets = Arc::new(Mutex::new(Vec::new()));
    let model = tracked(1, resets.clone());
    no_grad(|| {
        let report = model.train_step(&Tensor::ones(&[1, 1]), &Tensor::zeros(&[1, 1]), mse_loss);
        assert_eq!(report.loss, 4.0);
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
    assert_ne!(resets.lock().unwrap()[0], thread::current().id());
    assert_eq!(model.module().used.grad_vec().unwrap(), vec![4.0]);
    assert_eq!(model.module().unused.grad_vec().unwrap(), vec![0.0]);
}

#[test]
fn empty_replicas_reset_inline_without_spawning_or_stale_gradients() {
    let resets = Arc::new(Mutex::new(Vec::new()));
    let model = tracked(4, resets.clone());
    let report = model.train_step(&Tensor::ones(&[1, 1]), &Tensor::zeros(&[1, 1]), mse_loss);
    let resets = resets.lock().unwrap();
    assert_eq!(resets.len(), 4);
    assert_eq!(resets.iter().filter(|&&id| id == thread::current().id()).count(), 3);
    assert_eq!(report.replicas.iter().map(|replica| replica.samples).collect::<Vec<_>>(), vec![1, 0, 0, 0]);
    assert_eq!(report.loss, 4.0);
    assert_eq!(model.module().used.grad_vec().unwrap(), vec![4.0]);
    assert_eq!(model.module().unused.grad_vec().unwrap(), vec![0.0]);
    assert_eq!(report.allreduce.workers, 4);
    assert_eq!(report.allreduce.elements, 2);
    assert_eq!(report.allreduce.steps, 6);
}
