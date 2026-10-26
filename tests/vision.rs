use rustgears::nn::layers::{AdaptiveAvgPool2d, ConvTranspose2d};
use rustgears::prelude::*;

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance, "index {i}: {a} != {b}");
    }
}

fn check_gradient(tensor: &Tensor, forward: impl Fn() -> Tensor) {
    let gradient = tensor.grad_vec().unwrap();
    let original = tensor.to_vec();
    let mut values = original.clone();
    let epsilon = 1e-3;
    for i in 0..values.len() {
        values[i] = original[i] + epsilon;
        tensor.copy_from_slice(&values);
        let plus = no_grad(|| forward().item());
        values[i] = original[i] - epsilon;
        tensor.copy_from_slice(&values);
        let minus = no_grad(|| forward().item());
        values[i] = original[i];
        close(&[gradient[i]], &[(plus - minus) / (2.0 * epsilon)], 2e-3);
    }
    tensor.copy_from_slice(&original);
}

#[test]
fn adaptive_pool_handles_non_divisible_sizes_upsampling_and_gradients() {
    let x = Tensor::from_vec((1..=12).map(|v| v as f32).collect(), &[1, 1, 3, 4]).requires_grad_(true);
    let pool = AdaptiveAvgPool2d::new((2, 3));
    let y = pool.forward(&x);
    assert_eq!(y.shape(), &[1, 1, 2, 3]);
    close(&y.to_vec(), &[3.5, 4.5, 5.5, 7.5, 8.5, 9.5], 1e-6);
    y.mean().backward();
    check_gradient(&x, || pool.forward(&x).mean());
    close(&x.adaptive_avg_pool2d((1, 1)).to_vec(), &[6.5], 1e-6);

    let small = Tensor::from_slice(&[1., 2., 3., 4.], &[1, 1, 2, 2]);
    close(&small.adaptive_avg_pool2d((3, 3)).to_vec(), &[1., 1.5, 2., 2., 2.5, 3., 3., 3.5, 4.], 1e-6);
    let moved = small.detach().on(Device::Gpu(0));
    assert_eq!(moved.adaptive_avg_pool2d((1, 1)).device(), Device::Gpu(0));
}

#[test]
fn transposed_convolution_known_result_and_layer_bias() {
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[1, 1, 2, 2]);
    let layer = ConvTranspose2d::new(1, 1, 2);
    layer.weight.copy_from_slice(&[1., 1., 1., 1.]);
    layer.bias.copy_from_slice(&[0.5]);
    let y = layer.forward(&x);
    assert_eq!(y.shape(), &[1, 1, 3, 3]);
    close(&y.to_vec(), &[1.5, 3.5, 2.5, 4.5, 10.5, 6.5, 3.5, 7.5, 4.5], 1e-6);
    assert_eq!(layer.named_parameters().len(), 2);
    y.sum().backward();
    close(&layer.bias.grad_vec().unwrap(), &[9.], 1e-6);
}

#[test]
fn transposed_convolution_matches_scatter_with_padding_and_has_correct_gradients() {
    let x = Tensor::from_vec((0..16).map(|i| (i as f32 - 7.) / 20.).collect(), &[2, 2, 2, 2]).requires_grad_(true);
    let weights: Vec<_> = (0..24).map(|i| (i as f32 - 11.) / 30.).collect();
    let weight = Tensor::from_vec(weights.clone(), &[2, 3, 2, 2]).requires_grad_(true);
    let y = x.conv_transpose2d(&weight, 2, 1, 1);
    assert_eq!(y.shape(), &[2, 3, 3, 3]);
    let mut expected = vec![0.; y.numel()];
    let input = x.to_vec();
    for b in 0..2 {
        for c in 0..2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    for o in 0..3 {
                        for ky in 0..2 {
                            for kx in 0..2 {
                                let oy = (iy * 2 + ky) as isize - 1;
                                let ox = (ix * 2 + kx) as isize - 1;
                                if (0..3).contains(&oy) && (0..3).contains(&ox) {
                                    expected[((b * 3 + o) * 3 + oy as usize) * 3 + ox as usize] += input
                                        [((b * 2 + c) * 2 + iy) * 2 + ix]
                                        * weights[((c * 3 + o) * 2 + ky) * 2 + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    close(&y.to_vec(), &expected, 1e-6);
    y.square().sum().backward();
    let forward = || x.conv_transpose2d(&weight, 2, 1, 1).square().sum();
    check_gradient(&x, forward);
    check_gradient(&weight, forward);
    let gpu_x = x.detach().on(Device::Gpu(0));
    let gpu_weight = weight.detach().on(Device::Gpu(0));
    let gpu_y = gpu_x.conv_transpose2d(&gpu_weight, 2, 1, 1);
    assert_eq!(gpu_y.device(), Device::Gpu(0));
    close(&gpu_y.to_vec(), &expected, 1e-5);
}

#[test]
fn max_pool_negative_infinity_gradients_stay_in_their_windows() {
    let x = Tensor::full(&[2, 2, 2, 2], f32::NEG_INFINITY).requires_grad_(true);
    x.max_pool2d(1, 1).sum().backward();
    close(&x.grad_vec().unwrap(), &[1.; 16], 0.);
    let nan = Tensor::from_slice(&[f32::NAN, 1.], &[1, 1, 1, 2]);
    assert!(nan.max_pool2d(1, 1).to_vec()[0].is_nan());
}

#[test]
#[should_panic(expected = "output_padding < stride")]
fn transposed_convolution_rejects_invalid_output_padding() {
    Tensor::ones(&[1, 1, 1, 1]).conv_transpose2d(&Tensor::ones(&[1, 1, 1, 1]), 1, 0, 1);
}

#[test]
#[should_panic(expected = "spatial sizes must be positive")]
fn adaptive_pool_rejects_empty_output() {
    Tensor::ones(&[1, 1, 2, 2]).adaptive_avg_pool2d((0, 1));
}

#[test]
#[should_panic(expected = "kernel and stride must be positive")]
fn pooling_rejects_zero_stride() {
    Tensor::ones(&[1, 1, 2, 2]).max_pool2d(1, 0);
}
