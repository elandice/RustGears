# 미니배치 성능 분석과 개선

메모리 기반 데이터셋의 샘플별 할당·복사와 작은 배치의 스레드 생성 비용을 줄였습니다. 기존 DataLoader 호출법, 셔플·마지막 배치 처리, device, 손실 및 gradient의 샘플 수 가중치는 유지합니다. 의존성은 추가하지 않았습니다.

## 측정 조건

- Windows, 사용 가능한 논리 프로세서 12개, Rayon 작업 스레드 4개.
- `release`, CPU backend, `--no-default-features`. 실제 GPU나 디스크 I/O는 측정하지 않았습니다.
- 변경 전 소스 기준: `b83f86e`. 동일한 benchmark를 수정 전에 빌드하고 실행 파일을 보관해 비교했습니다.
- 데이터와 모델은 측정 루프 밖에서 생성하고 결과는 `black_box`로 소비합니다. 예열 및 반복 횟수 보정 후 약 100ms씩 7회 측정한 중앙값입니다.
- 초기 비교와 반복 비교 후, 빈 shape 및 잘못된 Dataset의 검증 순서를 보완하고 최종 비교를 실행했습니다. 아래 표는 최종 실행 결과입니다. 세 비교의 중앙값·최솟값·최댓값은 [원시 결과 CSV](MINIBATCH_BENCHMARK.csv)에 남겼습니다.
- loader 항목은 한 epoch 전체, slicing과 DataParallel은 한 호출의 시간입니다. MLP는 1,024개 샘플의 실제 forward/backward/SGD 갱신을 포함한 짧은 epoch이며 수렴 속도·최종 정확도 실험은 아닙니다.

## 최종 결과

`N`은 샘플 수, `D`는 샘플의 입력 원소 수, `B`는 미니배치 크기입니다. 배수는 변경 전 시간 / 변경 후 시간입니다.

| 측정 항목 | 변경 전 | 변경 후 | 배수 |
|---|---:|---:|---:|
| TensorDataset N8192/D16/B32, 순서대로 | 1.561 ms | 0.122 ms | 12.8× |
| TensorDataset N8192/D16/B32, 셔플 | 1.575 ms | 0.175 ms | 9.0× |
| TensorDataset N8192/D16/B256, 셔플 | 1.680 ms | 0.070 ms | 24.0× |
| TensorDataset N4096/D784/B64, 순서대로 | 2.835 ms | 0.327 ms | 8.7× |
| TensorDataset N4096/D784/B64, 셔플 | 2.774 ms | 0.463 ms | 6.0× |
| TensorDataset N8192/D16/B32, 셔플·prefetch 2 | 1.952 ms | 0.286 ms | 6.8× |
| 사용자 정의 Dataset, CPU 변환 포함 | 0.918 ms | 0.911 ms | 1.0× |
| 연속 행 복사 D16/B32 | 238 ns | 124 ns | 1.9× |
| 연속 행 복사 D784/B256 | 13.646 µs | 12.916 µs | 1.1× |
| DataParallel CPU 1장치/B32 | 65.772 µs | 5.323 µs | 12.4× |
| DataParallel CPU 4장치/B1 | 217.012 µs | 123.857 µs | 1.8× |
| DataParallel CPU 4장치/B96 | 230.540 µs | 227.793 µs | 1.0× |
| MLP 16→32→4, SGD, N1024/B32 | 1.614 ms | 1.420 ms | 1.14× |

이 로더 위주의 측정에서 얻은 배수를 전체 학습 속도에 그대로 적용할 수는 없습니다. 작은 MLP epoch는 약 12% 단축됐으며, 네 장치를 모두 사용하는 B96 경로는 반복 비교에서도 일관된 개선이 없었습니다. 큰 연속 텐서는 payload 복사 자체가 대부분의 비용이라 개선 폭이 작습니다.

## 바뀐 경로

### 1. TensorDataset 배치 일괄 읽기

이전에는 각 샘플마다 입력·정답 Vec를 만들고, 중간 샘플 목록을 거쳐 최종 배치 Vec로 다시 복사했습니다. `TensorDataset`은 이제 최종 입력·정답 버퍼에 직접 복사합니다. 연속 인덱스는 각 텐서의 연속 구간을 한 번에 복사하고, 셔플 인덱스는 순서를 유지하며 행을 모읍니다. 샘플별 임시 Vec `2 × B`개와 중간 목록을 제거했습니다. 최종 배치와 tensor storage의 할당·복사는 여전히 있습니다.

`Dataset`에는 기본값이 `None`인 선택적 `get_batch(indices)` 메서드를 추가했습니다. 기존 사용자 정의 Dataset은 계속 `get()`의 Rayon 병렬 경로를 사용합니다. I/O나 CPU 변환 비용을 알 수 없는 사용자 코드에 임의의 순차 실행 기준을 적용하지 않았습니다. 직접 일괄 읽기를 구현할 경우 인덱스 순서와 선언된 샘플 shape를 지켜야 하며 DataLoader는 반환된 입력·정답의 전체 길이를 검증합니다.

일반 `get()` 경로도 개별 샘플을 먼저 검증한 뒤 최종 버퍼 용량을 확보합니다. 잘못된 큰 shape 선언이 검증 전에 불필요한 대량 할당을 유발하지 않도록 했습니다.

### 2. 연속 행 slicing

`Tensor::narrow_rows`는 연속 범위인데도 인덱스 Vec를 만들고 행별 복사를 했습니다. 이제 범위를 검증한 뒤 연속 slice를 한 번 복사합니다. 중복 shape 할당도 제거했습니다. 결과는 기존처럼 원본과 독립된 복사이며 autograd에서 분리되고 device를 유지합니다. 빈 배치와 폭이 0인 샘플도 처리합니다.

### 3. 작은 DataParallel 배치

- 단일 장치에서는 별도 OS 스레드와 all-reduce 버킷 생성·복사·재적용을 생략합니다.
- 단일 장치라도 외부 `no_grad` 안에서 호출하면 기존 worker 경로를 유지해 gradient 기록과 호출자의 상태를 보존합니다.
- 빈 shard는 gradient를 초기화하지만 OS 스레드를 만들지 않습니다. 전체 replica 보고와 gradient·parameter·buffer 동기화는 유지합니다.
- 다중 장치의 gradient 버킷은 정확한 크기로 할당하고 gradient를 잠금 안에서 읽어 채웁니다. 파라미터마다 `grad_vec()`로 전체 gradient를 복사하는 비용을 제거했습니다.

단일 장치의 일반 학습 콜백(`zero_grad`, `forward`, loss)은 이제 호출자 스레드에서 실행됩니다. 빈 replica의 `zero_grad`도 호출자 스레드에서 실행됩니다. 별도 worker의 스레드 로컬 상태에 의존하는 custom Module은 이 차이를 고려해야 합니다. 일반적인 학습 수식, 입력 detach, 미사용 파라미터의 0 gradient와 통계 형식은 회귀 테스트로 확인합니다.

## 사용 시 판단 기준

`DataLoader::new(&dataset, 32)`는 전체 미니배치 크기가 32라는 뜻입니다. 이를 4장치 DataParallel에 전달하면 보통 장치마다 8개를 처리합니다. `iter_prefetch(scope, 2)`의 2는 대기 큐의 배치 개수입니다.

이번 메모리 전용 셔플 B32 측정에서는 동기 로더가 0.175ms, prefetch가 0.286ms였습니다. 데이터 준비가 이미 저렴하면 producer 스레드와 채널 비용이 더 큽니다. prefetch를 항상 켜기보다는 실제 학습 연산과 데이터 준비를 함께 측정해야 합니다.

gradient 누적의 잔여 배치 처리, 서로 다른 크기의 microbatch 가중치, DataParallel과의 자동 누적 연계는 이번 성능 변경에 포함하지 않았습니다. 현재 누적 helper의 동작은 유지하며, 동일 크기의 배치 32를 4회 누적할 때 유효 배치 128이 되는 기본 경우와 경계 처리를 구분해야 합니다. SyncBatchNorm, GPU 상주 텐서, 비동기 GPU 전송과 실제 다중 GPU 처리량도 별도 작업입니다.

## 재실행

```powershell
cargo bench --bench minibatch --no-default-features
```

선택적으로 `MINIBATCH_FILTER` 환경변수에 case 이름 일부를 넣으면 해당 항목만 측정합니다. 일반 debug 테스트에서는 benchmark 본문을 실행하지 않습니다. 새로운 의존성 없이 표준 라이브러리로 측정하며, benchmark 소스는 [benches/minibatch.rs](../benches/minibatch.rs)에 있습니다.
