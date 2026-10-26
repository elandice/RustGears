//! Integration tests covering autograd, layers, GPU dispatch, DataParallel
//! (Ring All-Reduce), NPU inference (operator fusion + INT8/FP16), and model
//! serialisation.

use rustorch::prelude::*;

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "mismatch at index {i}: {x} vs {y} (diff {diff} > {tol})");
    }
}

#[test]
fn autograd_numerical_gradient_check() {
    manual_seed(7);
    let x = Tensor::randn(&[4, 5]).requires_grad_(true);
    let w = Tensor::randn(&[5, 3]).requires_grad_(true);
    let b = Tensor::randn(&[1, 3]).requires_grad_(true);

    let forward = |x: &Tensor, w: &Tensor, b: &Tensor| {
        let h = x.matmul(w).add(b).gelu();
        h.square().mean()
    };

    let loss = forward(&x, &w, &b);
    loss.backward();
    let gw_analytic = w.grad_vec().unwrap();

    // Central finite difference check on w
    let eps = 1e-3f32;
    let mut w_data = w.to_vec();
    let mut gw_num = vec![0.0; w_data.len()];
    for i in 0..w_data.len() {
        let orig = w_data[i];
        w_data[i] = orig + eps;
        let wp = Tensor::from_slice(&w_data, &[5, 3]);
        let lp = no_grad(|| forward(&x, &wp, &b).item());

        w_data[i] = orig - eps;
        let wm = Tensor::from_slice(&w_data, &[5, 3]);
        let lm = no_grad(|| forward(&x, &wm, &b).item());

        w_data[i] = orig;
        gw_num[i] = (lp - lm) / (2.0 * eps);
    }
    assert_close(&gw_analytic, &gw_num, 2e-3);
}

#[test]
fn gpu_and_cpu_matmul_and_unary_match() {
    manual_seed(123);
    // Force even small workloads onto the physical GPU if one is present
    rustorch::device::gpu::set_offload_threshold(0, 0);

    let a_cpu = Tensor::randn(&[17, 19]).requires_grad_(true);
    let b_cpu = Tensor::randn(&[19, 13]).requires_grad_(true);
    let out_cpu = a_cpu.matmul(&b_cpu).gelu().sum();
    out_cpu.backward();

    let dev = Device::Gpu(0);
    let a_gpu = Tensor::from_vec(a_cpu.to_vec(), &[17, 19]).on(dev).requires_grad_(true);
    let b_gpu = Tensor::from_vec(b_cpu.to_vec(), &[19, 13]).on(dev).requires_grad_(true);
    let out_gpu = a_gpu.matmul(&b_gpu).gelu().sum();
    out_gpu.backward();

    assert_close(&out_cpu.to_vec(), &out_gpu.to_vec(), 1e-3);
    assert_close(&a_cpu.grad_vec().unwrap(), &a_gpu.grad_vec().unwrap(), 1e-3);
    assert_close(&b_cpu.grad_vec().unwrap(), &b_gpu.grad_vec().unwrap(), 1e-3);
}

#[test]
fn ring_all_reduce_matches_exact_sum() {
    let mut buffers = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
    ];
    let expected = vec![10.5, 21.5, 32.5, 43.5, 54.5, 65.5, 76.5];
    let stats = ring_all_reduce(&mut buffers);
    assert_eq!(stats.workers, 4);
    assert_eq!(stats.steps, 6); // 2 * (N - 1)
    for b in &buffers {
        assert_close(b, &expected, 1e-5);
    }
}

#[test]
fn data_parallel_gradients_match_single_device() {
    manual_seed(99);
    let build = || {
        Sequential::new()
            .add(Linear::new(6, 12))
            .add(Relu)
            .add(Linear::new(12, 3))
    };

    // Single-device reference model
    manual_seed(99);
    let single = build();
    // Other tests share the global RNG, so copy the reference weights explicitly.
    let dp = DataParallel::new(
        vec![Device::Gpu(0), Device::Gpu(1), Device::Gpu(2), Device::Gpu(3)],
        build,
    );
    for (source, target) in single.parameters().iter().zip(dp.parameters()) {
        target.copy_from_slice(&source.to_vec());
    }
    dp.broadcast_parameters();

    let x = Tensor::randn(&[16, 6]);
    let y = Tensor::from_slice(&[0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0], &[16]);

    let loss_single = cross_entropy(&single.forward(&x), &y);
    loss_single.backward();

    let report = dp.train_step(&x, &y, cross_entropy);
    assert!((loss_single.item() - report.loss).abs() < 1e-4);

    for (ps, pd) in single.parameters().iter().zip(dp.parameters()) {
        assert_close(&ps.grad_vec().unwrap(), &pd.grad_vec().unwrap(), 1e-4);
    }
}

#[test]
fn npu_fusion_and_quantised_inference_match_fp32() {
    manual_seed(2026);
    let model = Sequential::new()
        .add(Conv2d::new(1, 4, 3).padding(1))
        .add(BatchNorm2d::new(4))
        .add(Relu)
        .add(MaxPool2d::new(2))
        .add(Flatten)
        .add(Linear::new(4 * 4 * 4, 16))
        .add(BatchNorm1d::new(16))
        .add(Relu)
        .add(Linear::new(16, 4));

    // Warm up BatchNorm running stats
    let x = Tensor::randn(&[8, 1, 8, 8]);
    let _ = model.forward(&x);
    model.eval();

    let ref_out = no_grad(|| model.forward(&x));

    let cpu_sess = InferenceSession::from_module(&model, ExecutionProvider::Cpu).unwrap();
    let cpu_out = cpu_sess.run(&x);
    assert_close(&ref_out.to_vec(), &cpu_out.to_vec(), 1e-4);

    let npu_fp16 = InferenceSession::from_module(&model, ExecutionProvider::npu_fp16()).unwrap();
    let fp16_out = npu_fp16.run(&x);
    assert_close(&ref_out.to_vec(), &fp16_out.to_vec(), 2e-2);

    let npu_int8 = InferenceSession::from_module(&model, ExecutionProvider::npu_int8()).unwrap();
    let int8_out = npu_int8.run(&x);
    let prof = npu_int8.last_npu_profile().unwrap();
    // 9 raw layers -> Conv+BN+ReLU (1) + MaxPool (1) + Flatten (1) + Linear+BN+ReLU (1) + Linear (1) = 5 fused ops
    assert_eq!(prof.original_layers, 9);
    assert_eq!(prof.fused_ops, 5);
    assert!(prof.compression_ratio() > 3.0);
    assert_close(&ref_out.to_vec(), &int8_out.to_vec(), 0.15);
}

#[test]
fn transformer_and_checkpoint_roundtrip() {
    manual_seed(11);
    let emb = Embedding::new(16, 8);
    let enc = TransformerEncoderLayer::new(8, 2, 16, 0.0).causal(true);
    let head = Linear::new(8, 16);

    let tokens = Tensor::from_slice(&[1.0, 5.0, 3.0, 2.0, 7.0, 0.0, 4.0, 6.0], &[2, 4]);
    let h = enc.forward(&emb.forward(&tokens));
    let logits = head.forward(&h);
    assert_eq!(logits.shape(), &[2, 4, 16]);
    let loss = logits.square().mean();
    loss.backward();
    assert!(emb.weight.grad().is_some());

    // Save & load state dict on a Sequential model
    let net = Sequential::new().add(Linear::new(4, 8)).add(Relu).add(Linear::new(8, 2));
    let probe = Tensor::randn(&[3, 4]);
    let before = net.forward(&probe).to_vec();

    let tmp = std::env::temp_dir().join("rustorch_test_ckpt.rtsc");
    save_state_dict(&net, &tmp).unwrap();

    let net2 = Sequential::new().add(Linear::new(4, 8)).add(Relu).add(Linear::new(8, 2));
    load_state_dict(&net2, &tmp).unwrap();
    let after = net2.forward(&probe).to_vec();
    let _ = std::fs::remove_file(&tmp);
    assert_close(&before, &after, 1e-6);
}

#[test]
fn test_early_stopping_and_checkpoint() {
    manual_seed(42);
    let net = Sequential::new().add(Linear::new(2, 4)).add(Relu).add(Linear::new(4, 1));
    let mut es = EarlyStopping::new(2).mode(EarlyStoppingMode::Min).restore_best_weights(true);

    // Epoch 1: Loss 1.0 -> improved, best=1.0
    assert!(es.step(1.0, &net));
    assert_eq!(es.counter(), 0);
    assert!(!es.should_stop());

    // Alter weights intentionally
    for p in net.parameters() {
        p.copy_from_slice(&vec![99.0; p.numel()]);
    }

    // Epoch 2: Loss 1.2 -> bad epoch 1
    assert!(!es.step(1.2, &net));
    assert_eq!(es.counter(), 1);
    assert!(!es.should_stop());

    // Epoch 3: Loss 1.5 -> bad epoch 2 >= patience(2) -> triggers early stop & restores
    assert!(!es.step(1.5, &net));
    assert!(es.should_stop());

    // Verify weights were restored to the best snapshot (not 99.0)
    for p in net.parameters() {
        assert!(p.to_vec().iter().all(|&v| (v - 99.0).abs() > 1.0));
    }

    // ModelCheckpoint test
    let tmp = std::env::temp_dir().join("test_model_checkpoint.rtsc");
    let mut ckpt = ModelCheckpoint::new(&tmp).mode(EarlyStoppingMode::Min).save_best_only(true);
    assert!(ckpt.step(0.8, &net).unwrap()); // saved
    assert!(!ckpt.step(0.9, &net).unwrap()); // not saved (score degraded)
    assert!(ckpt.step(0.5, &net).unwrap()); // saved (score improved)
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_reduce_lr_on_plateau_and_gradient_accumulator() {
    let mut opt = Sgd::new(vec![Tensor::zeros(&[2])], 0.1);
    let mut sched = ReduceLROnPlateau::new(0.1)
        .factor(0.5)
        .patience(2)
        .threshold(1e-3)
        .cooldown(1);

    sched.step(&mut opt, 1.0); // Epoch 1: best=1.0
    assert_eq!(sched.current_lr(), 0.1);

    sched.step(&mut opt, 1.05); // Bad epoch 1
    assert_eq!(sched.current_lr(), 0.1);

    sched.step(&mut opt, 1.02); // Bad epoch 2 -> triggers decay: 0.1 * 0.5 = 0.05
    assert_eq!(sched.current_lr(), 0.05);
    assert_eq!(opt.lr(), 0.05);

    // GradientAccumulator test
    let mut accum = GradientAccumulator::new(4);
    let loss = Tensor::from_slice(&[4.0], &[1]);
    let scaled = accum.scale_loss(&loss);
    assert_close(&scaled.to_vec(), &[1.0], 1e-6);

    assert!(!accum.step()); // 1
    assert!(!accum.step()); // 2
    assert!(!accum.step()); // 3
    assert!(accum.step());  // 4 -> triggers step
}

#[test]
fn test_rmsnorm_and_swiglu_autograd() {
    manual_seed(101);
    let rms = RMSNorm::new(4);
    let x = Tensor::randn(&[2, 4]).requires_grad_(true);
    let y = rms.forward(&x);
    assert_eq!(y.shape(), &[2, 4]);

    let loss = y.square().sum();
    loss.backward();
    assert!(x.grad().is_some());
    assert!(rms.weight.grad().is_some());

    // SwiGLU forward & backward
    let swiglu = SwiGLU::new(4, 8, 4);
    let x2 = Tensor::randn(&[3, 4]).requires_grad_(true);
    let out = swiglu.forward(&x2);
    assert_eq!(out.shape(), &[3, 4]);
    let loss2 = out.sum();
    loss2.backward();
    assert!(x2.grad().is_some());
    assert!(swiglu.gate.weight.grad().is_some());
    assert!(swiglu.up.weight.grad().is_some());
    assert!(swiglu.down.weight.grad().is_some());
}

#[test]
fn test_metrics_precision_recall_topk() {
    let preds = Tensor::from_slice(&[0.0, 1.0, 1.0, 2.0], &[4]);
    let targets = Tensor::from_slice(&[0.0, 1.0, 0.0, 2.0], &[4]);
    let (prec, rec, f1, macro_f1) = precision_recall_f1(&preds, &targets, 3);

    // Class 0: True=2 (idx 0,2), Pred=1 (idx 0). TP=1, FP=0 -> Prec=1.0, Rec=0.5, F1=2*(1*0.5)/(1.5)=2/3=0.6667
    assert_close(&[prec[0]], &[1.0], 1e-4);
    assert_close(&[rec[0]], &[0.5], 1e-4);
    assert_close(&[f1[0]], &[0.66667], 1e-4);

    // Class 1: True=1 (idx 1), Pred=2 (idx 1,2). TP=1, FP=1 -> Prec=0.5, Rec=1.0, F1=2/3=0.6667
    assert_close(&[prec[1]], &[0.5], 1e-4);
    assert_close(&[rec[1]], &[1.0], 1e-4);

    // Class 2: True=1 (idx 3), Pred=1 (idx 3). TP=1, FP=0 -> Prec=1.0, Rec=1.0, F1=1.0
    assert_close(&[prec[2]], &[1.0], 1e-4);
    assert_close(&[rec[2]], &[1.0], 1e-4);
    assert_close(&[macro_f1], &[(0.66667 + 0.66667 + 1.0) / 3.0], 1e-4);

    // Top-k accuracy
    let logits = Tensor::from_slice(&[
        10.0, 2.0, 1.0, // top-1 is 0
        1.0, 5.0, 8.0,  // top-1 is 2, top-2 is 1
    ], &[2, 3]);
    let tgts = Tensor::from_slice(&[0.0, 1.0], &[2]);
    assert_close(&[top_k_accuracy(&logits, &tgts, 1)], &[0.5], 1e-4);
    assert_close(&[top_k_accuracy(&logits, &tgts, 2)], &[1.0], 1e-4);

    // Confusion matrix
    let cm = confusion_matrix(&preds, &targets, 3);
    assert_eq!(cm[0][0], 1); // target 0, pred 0
    assert_eq!(cm[0][1], 1); // target 0, pred 1
    assert_eq!(cm[1][1], 1); // target 1, pred 1
    assert_eq!(cm[2][2], 1); // target 2, pred 2
}
