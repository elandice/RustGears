# 추가 기능과 점검 결과

다섯 기능은 모두 `use RustGears::prelude::*;`로 사용할 수 있습니다. 실행 가능한 전체 예제는 [modern_features.rs](../examples/modern_features.rs)입니다.

```sh
cargo run --example modern_features --no-default-features
```

## 1. Hugging Face Safetensors

공식 [Safetensors](https://huggingface.co/docs/safetensors/index) Rust 파서를 사용해 파일 구조, shape, 데이터 범위를 검증합니다. 저장은 F32, 읽기는 F32/F16/BF16을 지원하며 내부에서는 f32를 사용합니다.

```rust
save_safetensors(&model, "model.safetensors")?;
load_safetensors(&model, "model.safetensors")?;
// 기존 API: 확장자로 저장 형식 선택, 파일 내용으로 읽기 형식 판별
save_state_dict(&model, "model.safetensors")?;
load_state_dict(&model, "model.safetensors")?;
```

이름과 shape가 일치해야 하며, 모델의 모든 상태를 검증한 후 반영합니다. 파일에만 있는 추가 키는 무시합니다. `read_safetensors(path)`는 이름별 CPU 텐서를 반환해 외부 가중치를 직접 매핑할 수 있습니다.

예를 들어 PyTorch/Hugging Face Linear의 `[out, in]` 가중치는 RustGears의 `[in, out]`에 맞춰 전치해야 합니다. 이름이 같은 정사각 가중치도 레이아웃 확인이 필요합니다.

```rust
let weights = read_safetensors("model.safetensors")?;
let layer = Linear::new(768, 768);
let external = weights.get("encoder.layer.0.attention.self.query.weight")
    .ok_or("missing query weight")?;
let mapped = external.transpose(0, 1);
assert_eq!(mapped.shape(), layer.weight.shape());
layer.weight.copy_from_slice(&mapped.to_vec());
```

모델 config·토크나이저·다운로드·sharded index 해석과 정수/양자화 dtype 로드는 포함하지 않습니다. 실제 사전학습 모델을 실행하려면 해당 모델의 구조, 이름, 레이아웃을 맞춰야 합니다. `read_safetensors`는 파일의 모든 텐서를 읽으므로 미지원 dtype이 하나라도 있으면 오류를 반환합니다.

Safetensors·RTSC·RTIM은 같은 디렉터리의 임시 파일에 기록을 마친 뒤 대상 파일을 교체합니다. 저장이 중간에 실패해도 이전 파일은 유지됩니다.

## 2. KV-Cache와 RoPE

```rust
let block = TransformerEncoderLayer::new(64, 8, 256, 0.1)
    .causal(true).rope(10_000.0);
block.eval();
let mut cache = KvCache::new();
let prompt = Tensor::randn(&[1, 12, 64]);
let _prefill = block.forward_cached(&prompt, &mut cache);
let next = Tensor::randn(&[1, 1, 64]);
let output = block.forward_cached(&next, &mut cache);
assert_eq!(cache.len(), 13);
cache.clear();
```

`MultiHeadAttention`에도 같은 builder와 `forward_cached`를 제공합니다. 캐시에는 새 토큰만 전달하고 레이어마다 별도의 캐시를 사용합니다. 기존 K/V projection을 재사용하며, 이전 토큰 수를 기준으로 RoPE의 절대 위치와 chunk causal mask를 계산합니다. 캐시 추론은 gradient 기록과 dropout을 사용하지 않습니다. 새 시퀀스나 가중치 변경 후 `clear()`가 필요합니다.

RoPE는 [RoFormer](https://arxiv.org/abs/2104.09864)의 회전을 Llama식 split-half 쌍으로 적용하며 head dimension은 짝수여야 합니다. 독립 `RotaryEmbedding::new(head_dim, base).forward(&tensor, offset)`는 역전파를 지원합니다. 현재는 일반 MHA와 동적 캐시이며, 긴 문맥에서 append마다 K/V를 복사합니다. GQA, RoPE scaling, sliding window는 포함하지 않습니다.

## 3. 비동기 멀티스레드 DataLoader

```rust
let mut loader = DataLoader::new(&dataset, 32).shuffle(true).seed(42);
std::thread::scope(|scope| {
    for (inputs, targets) in loader.iter_prefetch(scope, 2) {
        // 현재 배치를 학습하는 동안 다음 배치를 준비합니다.
    }
});
```

`iter()`는 동기 로더이고 `iter_prefetch(scope, capacity)`는 별도 producer 스레드를 사용합니다. `TensorDataset`은 배치를 일괄 읽어 샘플별 임시 버퍼를 생략하며, 일반 Dataset은 Rayon 샘플 병렬화를 사용합니다. 최대 `capacity`개 완성 배치를 큐에 두고 producer가 준비 중인 한 배치와 소비 중인 배치는 별도입니다. shuffle·drop_last·device와 배치 순서는 동기 로더와 같습니다. [미니배치 성능 분석과 측정 결과](MINIBATCH_PERFORMANCE.md)에서 경로별 개선 폭과 사용 시 판단 기준을 설명합니다.

borrowed Dataset을 안전하게 유지하기 위해 표준 scoped thread 안에서 사용합니다. 조기 중단 시 큐를 닫고 producer를 join하며, Dataset의 panic을 소비자에 전달합니다. Rayon 작업 내부에서 소비할 때는 스레드 풀 교착을 피하도록 producer의 샘플 읽기를 순차 실행합니다. 실제 CPU/GPU 전송 중첩이나 GPU 처리량 향상은 별도 하드웨어 측정이 필요합니다.

## 4. 비전 레이어

```rust
let up = ConvTranspose2d::new(32, 16, 3)
    .stride(2).padding(1).output_padding(1);
let pool = AdaptiveAvgPool2d::new((1, 1));
let image = up.forward(&Tensor::randn(&[2, 32, 8, 8]));
assert_eq!(image.shape(), &[2, 16, 16, 16]);
assert_eq!(pool.forward(&image).shape(), &[2, 16, 1, 1]);
```

두 레이어 모두 NCHW 입력, device 유지와 역전파를 지원합니다. Tensor에는 `conv_transpose2d(&weight, stride, padding, output_padding)`와 `adaptive_avg_pool2d((h, w))`가 있습니다. Adaptive pooling은 나누어떨어지지 않는 크기와 더 큰 출력도 지원합니다.

ConvTranspose2d는 정사각 커널, scalar stride/padding, groups=1, dilation=1을 지원합니다. 가중치는 `[in_channels, out_channels, k, k]`이고 `output_padding < stride`여야 합니다. 의미는 [PyTorch ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)와 [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html)를 따릅니다. 현재 `.rtim`/NPU 그래프 export는 지원하지 않으며 명시적인 오류를 반환합니다.

## 5. ModelEMA와 프로그레스 바

```rust
let mut ema = ModelEMA::new(&model, 0.999);
let mut progress = ProgressBar::new(loader.num_batches());
// 각 optimizer.step() 직후:
ema.update(&model)?;
progress.set_message(format!("loss={:.4}", loss.item()));
progress.inc(1)?;
// epoch 종료:
progress.finish()?;
// 동일한 구조의 별도 평가 모델에 반영:
ema.copy_to(&evaluation_model)?;
evaluation_model.eval();
```

EMA는 초기 상태를 복사한 뒤 `decay * average + (1-decay) * current`로 파라미터와 BatchNorm을 포함한 f32 buffer를 평균냅니다. 이름과 shape가 다르면 평균이나 대상 모델을 변경하지 않습니다. `state_dict()`는 독립 snapshot, `num_updates()`는 업데이트 횟수입니다. 평가 모델로 복사하면 학습 가중치를 유지하면서 평가와 저장을 할 수 있습니다.

ProgressBar는 완료율·경과 시간·ETA·메시지를 stderr에 표시하며 출력 빈도를 제한합니다. redirected stderr에서는 자동 표시를 끄고 `render()`로 로그 문자열을 얻을 수 있습니다. 출력 오류는 `io::Result`로 전달합니다.

## 관련 분석과 보완

Tensor, Module, Device, LayerSpec의 연결 구조와 호출부를 분석하고 기존 테스트를 먼저 실행했습니다. 다음 결함을 기능 구현과 함께 보완했습니다.

- 체크포인트 로드 중 후반 텐서 검증 실패로 앞선 가중치만 변경되는 문제: 전체 검증 후 적용.
- RTSC/RTIM을 덮어쓰는 중 I/O 오류로 이전 파일이 손상될 수 있는 문제: 임시 파일 기록 후 교체.
- RTSC의 잘못된 이름·중복 이름·shape/data 길이, RTIM의 알 수 없는 활성화 태그: 오류 처리.
- 유한한 causal mask가 큰 미래 score를 차단하지 못하는 문제: 음의 무한대로 mask.
- MaxPool2d의 음의 무한대 윈도 gradient 위치 및 NaN 전파: 윈도 내부 argmax 보장.
- 외부 Dataset이 샘플마다 다른 길이를 반환하는 문제: 개별 샘플 검증.
- 체크포인트 저장 실패 후 같은 점수 재시도 누락, 비유한 metric, 불완전한 early stopping 복원: 상태 검증과 갱신 순서 보완.
- 장치 수보다 작은 배치에서 master replica가 빈 shard를 받는 문제, 빈 배치·샘플 수 불일치: 분배와 검증 보완.
- 정규화의 잘못된 feature 수 broadcasting과 빈 BatchNorm 통계 오염: 명시적 입력 검증.
- NoGradGuard가 다른 스레드에서 해제돼 thread-local 상태를 손상하는 문제: 스레드 이동 금지.
- 병렬 테스트의 전역 RNG 간섭: 비교 대상 가중치를 직접 복사해 재현성 확보.

회귀 검증은 외부 dtype fixture, 캐시/전체 추론 동등성, 비전 수치 미분, bounded queue 취소와 panic 전파, EMA 독립성과 실패 시 무변경, 콜백 실패 재시도, 작은 배치·정규화 입력 검증을 포함합니다. 실물 GPU 성능, 전체 Hugging Face 모델 및 대형 데이터셋 처리량은 별도 검증 범위입니다.
