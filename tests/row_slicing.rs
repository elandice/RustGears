use rustgears::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn contiguous_rows_match_gather_and_remain_independent_detached_copies() {
    let source = Tensor::arange(24).reshape(&[4, 2, 3]).detach().requires_grad_(true).on(Device::Gpu(99));
    let sliced = source.narrow_rows(1, 2);
    assert_eq!(sliced.shape(), &[2, 2, 3]);
    assert_eq!(sliced.to_vec(), source.index_rows(&[1, 2]).to_vec());
    assert_eq!(sliced.device(), Device::Gpu(99));
    assert!(!sliced.requires_grad());
    assert!(sliced.grad_fn().is_none());
    assert!(sliced.is_leaf());
    sliced.copy_from_slice(&[99.0; 12]);
    assert_eq!(source.to_vec(), (0..24).map(|value| value as f32).collect::<Vec<_>>());
}

#[test]
fn contiguous_rows_preserve_empty_batch_and_zero_width_shapes() {
    let empty = Tensor::zeros(&[0, usize::MAX, usize::MAX]).narrow_rows(0, 0);
    assert_eq!(empty.shape(), &[0, usize::MAX, usize::MAX]);
    for (shape, start, len, expected) in
        [(vec![0, 3], 0, 0, vec![0, 3]), (vec![3, 0, 2], 1, 2, vec![2, 0, 2]), (vec![4], 4, 0, vec![0])]
    {
        let output = Tensor::zeros(&shape).narrow_rows(start, len);
        assert_eq!(output.shape(), expected);
        assert!(output.to_vec().is_empty());
    }
    assert_eq!(Tensor::arange(4).narrow_rows(1, 2).to_vec(), vec![1.0, 2.0]);
}

#[test]
fn contiguous_rows_reject_invalid_and_overflowing_ranges() {
    for shape in [vec![4, 2], vec![4, 0]] {
        let tensor = Tensor::zeros(&shape);
        for (start, len) in [(5, 0), (4, 1), (3, 2), (usize::MAX, 1), (1, usize::MAX)] {
            assert!(catch_unwind(AssertUnwindSafe(|| tensor.narrow_rows(start, len))).is_err());
        }
    }
    assert!(catch_unwind(|| Tensor::scalar(1.0).narrow_rows(0, 1)).is_err());
}
