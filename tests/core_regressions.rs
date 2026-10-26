use rustgears::autograd::is_grad_enabled;
use rustgears::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rejects(f: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
}

#[test]
fn tiny_parallel_batch_updates_master_batchnorm_statistics() {
    let model = DataParallel::new(vec![Device::Cpu; 4], || BatchNorm2d::new(1));
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[1, 1, 2, 2]);
    let y = Tensor::zeros_like(&x);
    let mut optimizer = Sgd::new(model.parameters(), 0.01);
    let report = model.fit_batch(&x, &y, mse_loss, &mut optimizer);
    assert_eq!(report.replicas.iter().map(|r| r.samples).collect::<Vec<_>>(), vec![1, 0, 0, 0]);
    assert!((model.module().running_mean.item() - 0.25).abs() < 1e-6);
    assert!(report.loss.is_finite());
}

#[test]
fn uneven_and_tiny_parallel_gradients_are_sample_weighted() {
    for (samples, workers) in [(5, 3), (2, 4)] {
        let model = DataParallel::new(vec![Device::Cpu; workers], || Linear::no_bias(1, 1));
        model.module().weight.copy_from_slice(&[2.]);
        model.broadcast_parameters();
        let x = Tensor::from_vec((1..=samples).map(|i| i as f32).collect(), &[samples, 1]);
        let y = Tensor::zeros_like(&x);
        let report = model.train_step(&x, &y, mse_loss);
        let expected = 4. * (1..=samples).map(|i| (i * i) as f32).sum::<f32>() / samples as f32;
        assert!((report.loss - expected).abs() < 1e-5);
        assert!((model.parameters()[0].grad_vec().unwrap()[0] - expected).abs() < 1e-5);
    }
}

#[test]
fn parallel_training_rejects_invalid_batch_shapes_before_work() {
    let model = DataParallel::new(vec![Device::Cpu; 2], || Linear::new(1, 1));
    rejects(|| {
        model.train_step(&Tensor::zeros(&[0, 1]), &Tensor::zeros(&[0, 1]), mse_loss);
    });
    rejects(|| {
        model.train_step(&Tensor::zeros(&[2, 1]), &Tensor::zeros(&[1, 1]), mse_loss);
    });
    rejects(|| {
        model.train_step(&Tensor::scalar(1.), &Tensor::scalar(1.), mse_loss);
    });
    assert!(model.parameters().iter().all(|p| p.grad().is_none()));
}

#[test]
fn normalization_rejects_broadcasted_feature_counts_and_zero_dimensions() {
    let layers: Vec<Box<dyn Module>> =
        vec![Box::new(BatchNorm1d::new(1)), Box::new(LayerNorm::new(1)), Box::new(RMSNorm::new(1))];
    for layer in layers {
        rejects(|| {
            layer.forward(&Tensor::ones(&[2, 2]));
        });
    }
    rejects(|| {
        BatchNorm2d::new(1).forward(&Tensor::ones(&[2, 2, 2, 2]));
    });
    rejects(|| {
        BatchNorm1d::new(0);
    });
    rejects(|| {
        BatchNorm2d::new(0);
    });
    rejects(|| {
        LayerNorm::new(0);
    });
    rejects(|| {
        RMSNorm::new(0);
    });
}

#[test]
fn empty_batchnorm_training_does_not_corrupt_running_statistics() {
    let layer = BatchNorm1d::new(2);
    rejects(|| {
        layer.forward(&Tensor::zeros(&[0, 2]));
    });
    assert_eq!(layer.running_mean.to_vec(), vec![0., 0.]);
    assert_eq!(layer.running_var.to_vec(), vec![1., 1.]);
    let spatial = BatchNorm2d::new(1);
    rejects(|| {
        spatial.forward(&Tensor::zeros(&[1, 1, 0, 2]));
    });
    assert_eq!(spatial.running_mean.item(), 0.);
    assert_eq!(spatial.running_var.item(), 1.);
}

#[test]
fn no_grad_nested_guards_restore_state_after_panic() {
    assert!(is_grad_enabled());
    no_grad(|| {
        assert!(!is_grad_enabled());
        no_grad(|| assert!(!is_grad_enabled()));
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
    rejects(|| {
        no_grad(|| {
            let _inner = NoGradGuard::new();
            assert!(!is_grad_enabled());
            panic!("test unwind restoration");
        })
    });
    assert!(is_grad_enabled());
    let x = Tensor::scalar(2.).requires_grad_(true);
    x.square().backward();
    assert_eq!(x.grad_vec().unwrap(), vec![4.]);
}
