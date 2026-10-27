//! 신경망 손실 함수(Loss functions) 구현 모듈.

use crate::tensor::Tensor;

/// 평균 제곱 오차 (Mean Squared Error: MSE) 손실을 계산합니다.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).square().mean()
}

/// 평균 절대 오차 (Mean Absolute Error: L1) 손실을 계산합니다.
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).abs().mean()
}

/// Smooth L1 / Huber 손실 함수: 오차가 `delta` 이하이면 제곱 오차를, 크면 선형 오차를 적용하여 이상치에 강건합니다.
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

/// 로짓 `[N, C]`과 정수 정답 라벨 `[N]` 간의 교차 엔트로피(Cross-Entropy) 손실을 계산합니다.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    cross_entropy_smooth(logits, targets, 0.0)
}

/// 레이블 스무딩(Label Smoothing: `eps`)이 적용된 수치적으로 안정적인 융합 교차 엔트로피 손실 함수.
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

/// 로그 확률 텐서 `[N, C]`에 대한 음의 로그 우도(Negative Log-Likelihood: NLL) 손실을 계산합니다.
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

/// 원시 로짓에 대한 이진 교차 엔트로피(Binary Cross-Entropy with Logits) 손실 (수치적 안정성을 위해 시그모이드와 융합).
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

/// 예측 로짓 `[N, C]`과 정수 정답 라벨 `[N]` 간의 분류 정확도(Accuracy)를 계산합니다.
pub fn accuracy(logits: &Tensor, targets: &Tensor) -> f32 {
    let pred = logits.argmax();
    let tg = targets.data();
    let correct = pred.iter().zip(tg.iter()).filter(|(p, t)| **p == **t as usize).count();
    correct as f32 / pred.len().max(1) as f32
}
