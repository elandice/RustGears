//! Loss functions.

use crate::tensor::Tensor;

/// Mean squared error.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).square().mean()
}

/// Mean absolute error.
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).abs().mean()
}

/// Smooth L1 / Huber loss.
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Tensor {
    let diff = pred.sub(target).to_vec();
    let n = diff.len().max(1) as f32;
    let loss: f32 = diff
        .iter()
        .map(|d| if d.abs() <= delta { 0.5 * d * d } else { delta * (d.abs() - 0.5 * delta) })
        .sum::<f32>()
        / n;
    let (p, t) = (pred.clone(), target.clone());
    Tensor::from_op(
        vec![loss],
        vec![],
        pred.device(),
        vec![pred.clone(), target.clone()],
        "HuberBackward",
        Box::new(move |g| {
            let gd: Vec<f32> = diff.iter().map(|d| g[0] * d.clamp(-delta, delta) / n).collect();
            vec![p.requires_grad().then(|| gd.clone()), t.requires_grad().then(|| gd.iter().map(|v| -v).collect())]
        }),
    )
}

/// Cross-entropy for class logits `[N, C]` and integer targets `[N]`.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    cross_entropy_smooth(logits, targets, 0.0)
}

/// Cross-entropy with label smoothing `eps` (fused, numerically stable).
pub fn cross_entropy_smooth(logits: &Tensor, targets: &Tensor, eps: f32) -> Tensor {
    assert_eq!(logits.ndim(), 2, "cross_entropy expects logits [N, C], got {:?}", logits.shape());
    let (n, c) = (logits.shape()[0], logits.shape()[1]);
    assert_eq!(targets.numel(), n, "cross_entropy: {} targets for batch of {}", targets.numel(), n);
    let tg: Vec<usize> = targets.data().iter().map(|&v| v as usize).collect();
    let x = logits.data();
    let mut probs = vec![0.0f32; n * c];
    let mut loss = 0.0f32;
    for i in 0..n {
        let row = &x[i * c..(i + 1) * c];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let se: f32 = row.iter().map(|v| (v - mx).exp()).sum();
        let lse = mx + se.ln();
        assert!(tg[i] < c, "cross_entropy: target {} out of range ({c} classes)", tg[i]);
        for j in 0..c {
            probs[i * c + j] = (row[j] - lse).exp();
            let q = if j == tg[i] { 1.0 - eps + eps / c as f32 } else { eps / c as f32 };
            if q > 0.0 {
                loss -= q * (row[j] - lse);
            }
        }
    }
    drop(x);
    loss /= n as f32;
    Tensor::from_op(
        vec![loss],
        vec![],
        logits.device(),
        vec![logits.clone()],
        "CrossEntropyBackward",
        Box::new(move |g| {
            let scale = g[0] / n as f32;
            let mut gx = probs.clone();
            for i in 0..n {
                for j in 0..c {
                    let q = if j == tg[i] { 1.0 - eps + eps / c as f32 } else { eps / c as f32 };
                    gx[i * c + j] = (gx[i * c + j] - q) * scale;
                }
            }
            vec![Some(gx)]
        }),
    )
}

/// Negative log-likelihood on log-probabilities `[N, C]`.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    let (n, c) = (log_probs.shape()[0], log_probs.shape()[1]);
    let tg: Vec<usize> = targets.data().iter().map(|&v| v as usize).collect();
    let lp = log_probs.data();
    let loss = -tg.iter().enumerate().map(|(i, &t)| lp[i * c + t]).sum::<f32>() / n as f32;
    drop(lp);
    Tensor::from_op(
        vec![loss],
        vec![],
        log_probs.device(),
        vec![log_probs.clone()],
        "NllLossBackward",
        Box::new(move |g| {
            let mut gx = vec![0.0; n * c];
            for (i, &t) in tg.iter().enumerate() {
                gx[i * c + t] = -g[0] / n as f32;
            }
            vec![Some(gx)]
        }),
    )
}

/// Binary cross-entropy on raw logits (sigmoid fused for stability).
pub fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Tensor {
    assert_eq!(logits.shape(), targets.shape(), "bce_with_logits: shape mismatch");
    let x = logits.to_vec();
    let y = targets.to_vec();
    let n = x.len() as f32;
    let loss: f32 = x.iter().zip(&y).map(|(&x, &y)| x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln()).sum::<f32>() / n;
    Tensor::from_op(
        vec![loss],
        vec![],
        logits.device(),
        vec![logits.clone()],
        "BceWithLogitsBackward",
        Box::new(move |g| {
            vec![Some(x.iter().zip(&y).map(|(&x, &y)| g[0] * (1.0 / (1.0 + (-x).exp()) - y) / n).collect())]
        }),
    )
}

/// Classification accuracy of logits `[N, C]` against integer targets `[N]`.
pub fn accuracy(logits: &Tensor, targets: &Tensor) -> f32 {
    let pred = logits.argmax();
    let tg = targets.data();
    let correct = pred.iter().zip(tg.iter()).filter(|(p, t)| **p == **t as usize).count();
    correct as f32 / pred.len().max(1) as f32
}
