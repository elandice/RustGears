//! Multi-device parallelism.
//!
//! * [`DataParallel`] – synchronous data-parallel training across several
//!   devices (GPUs, virtual GPUs and/or the CPU), DDP style.
//! * [`ring_all_reduce`] – bandwidth-optimal gradient averaging used by
//!   DataParallel (the algorithm behind NCCL / Horovod).

mod allreduce;
mod data_parallel;

pub use allreduce::{ring_all_reduce, AllReduceStats};
pub use data_parallel::{DataParallel, ReplicaStats, StepReport};
