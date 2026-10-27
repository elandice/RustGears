//! 다중 디바이스 분산 병렬 처리 모듈.
//!
//! * [`DataParallel`] – 여러 디바이스(물리 GPU, 가상 GPU, CPU)에 걸친 동기식 데이터 병렬 학습 (PyTorch DDP 스타일).
//! * [`ring_all_reduce`] – DataParallel에서 사용하는 대역폭 최적화 그래디언트 평균화 알고리즘 (NCCL / Horovod의 핵심 알고리즘).

mod allreduce;
mod data_parallel;

pub use allreduce::{ring_all_reduce, AllReduceStats};
pub use data_parallel::{DataParallel, ReplicaStats, StepReport};
