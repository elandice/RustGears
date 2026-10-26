//! Hugging Face Safetensors weight interchange (native tensors store `f32`).

use super::{apply_state_dict, invalid_data};
use crate::{nn::Module, tensor::Tensor};
use ::safetensors::tensor::{serialize_to_file, Dtype, SafeTensorError, SafeTensors, TensorView};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::{io, path::Path};

/// Save parameters and buffers as interoperable `F32` Safetensors tensors.
/// Tensor names and layouts are preserved, including RustGears's `[in, out]`
/// Linear weights; PyTorch Linear weights use `[out, in]` instead.
pub fn save_safetensors(module: &dyn Module, path: impl AsRef<Path>) -> io::Result<()> {
    let mut owned = BTreeMap::new();
    for (name, tensor) in module.state_dict() {
        if name == "__metadata__" || owned.contains_key(&name) {
            return Err(invalid_data(format!("reserved or duplicate tensor name '{name}'")));
        }
        let bytes: Vec<u8> = tensor.data().iter().flat_map(|v| v.to_le_bytes()).collect();
        owned.insert(name, (tensor.shape().to_vec(), bytes));
    }
    let views = owned
        .iter()
        .map(|(name, (shape, data))| {
            TensorView::new(Dtype::F32, shape.clone(), data).map(|view| (name.as_str(), view)).map_err(invalid_data)
        })
        .collect::<io::Result<Vec<_>>>()?;
    serialize_to_file(views, None, path.as_ref()).map_err(|error| match error {
        SafeTensorError::IoError(error) => error,
        error => invalid_data(error),
    })
}

/// Read named `F32`, `F16`, or `BF16` weights into CPU `f32` tensors.
///
/// Use this to map external parameter names and transpose external Linear
/// weights before copying them into a RustGears model. Unsupported dtypes are
/// reported as errors; no integer or quantized weights are silently cast.
pub fn read_safetensors(path: impl AsRef<Path>) -> io::Result<HashMap<String, Tensor>> {
    let bytes = std::fs::read(path)?;
    decode(&bytes, None)
}

/// Load weights by exact name and shape, preserving the model's devices and
/// gradient flags. All tensors are checked before any model state is changed.
/// Additional tensors in the file are ignored, as with `load_state_dict`.
pub fn load_safetensors(module: &dyn Module, path: impl AsRef<Path>) -> io::Result<()> {
    let names: HashSet<_> = module.state_dict().into_iter().map(|(name, _)| name).collect();
    let bytes = std::fs::read(path)?;
    apply_state_dict(module, &decode(&bytes, Some(&names))?)
}

fn decode(bytes: &[u8], names: Option<&HashSet<String>>) -> io::Result<HashMap<String, Tensor>> {
    let file = SafeTensors::deserialize(bytes).map_err(invalid_data)?;
    file.iter()
        .filter(|(name, _)| names.is_none_or(|names| names.contains(*name)))
        .map(|(name, view)| {
            let values = match view.dtype() {
                Dtype::F32 => view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
                Dtype::F16 => {
                    view.data().as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect()
                }
                Dtype::BF16 => view
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                    .collect(),
                dtype => return Err(invalid_data(format!("unsupported dtype {dtype:?} for tensor '{name}'"))),
            };
            Ok((name.to_owned(), Tensor::from_vec(values, view.shape())))
        })
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    match exponent {
        0 => {
            let value = mantissa as f32 * (1.0 / 16_777_216.0);
            if sign != 0 {
                -value
            } else {
                value
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{BatchNorm1d, Linear, Sequential};
    use crate::save::{load_state_dict, save_state_dict};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn path(extension: &str) -> std::path::PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "rustgears_safetensors_{}_{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            extension
        ))
    }

    fn fixture(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn safetensors_roundtrip_parameters_and_buffers() {
        let model = Sequential::new().add(Linear::new(2, 3)).add(BatchNorm1d::new(3));
        let restored = Sequential::new().add(Linear::new(2, 3)).add(BatchNorm1d::new(3));
        let file = path("safetensors");
        save_state_dict(&model, &file).unwrap();
        load_state_dict(&restored, &file).unwrap();
        std::fs::remove_file(file).unwrap();
        for ((name, a), (other, b)) in model.state_dict().into_iter().zip(restored.state_dict()) {
            assert_eq!(name, other);
            assert_eq!(a.to_vec(), b.to_vec());
            assert_eq!(a.requires_grad(), b.requires_grad());
        }
    }

    #[test]
    fn reads_external_mixed_dtype_fixture() {
        let header = r#"{"f32":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"f16":{"dtype":"F16","shape":[7],"data_offsets":[4,18]},"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[18,22]},"__metadata__":{"format":"pt"}}"#;
        let mut data = 3.5f32.to_le_bytes().to_vec();
        for v in [0x3c00u16, 0xc000, 0x0001, 0x7c00, 0xfc00, 0x7e00, 0x8000, 0x3f80, 0xc020] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let decoded = decode(&fixture(header, &data), None).unwrap();
        assert_eq!(decoded["f32"].shape(), &[] as &[usize]);
        assert_eq!(decoded["f32"].to_vec(), vec![3.5]);
        let half = decoded["f16"].to_vec();
        assert_eq!(&half[..3], &[1.0, -2.0, 1.0 / 16_777_216.0]);
        assert_eq!(&half[3..5], &[f32::INFINITY, f32::NEG_INFINITY]);
        assert!(half[5].is_nan());
        assert_eq!(half[6].to_bits(), (-0.0f32).to_bits());
        assert_eq!(decoded["bf16"].to_vec(), vec![1.0, -2.5]);
    }

    #[test]
    fn rejects_malformed_offsets_and_unsupported_dtypes() {
        let malformed = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#;
        assert_eq!(decode(&fixture(malformed, &[0; 8]), None).unwrap_err().kind(), io::ErrorKind::InvalidData);
        let unsupported = r#"{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#;
        assert!(decode(&fixture(unsupported, &[0; 8]), None).unwrap_err().to_string().contains("I64"));
    }

    #[test]
    fn module_load_ignores_unused_dtypes_but_rejects_required_ones() {
        let header = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"unused":{"dtype":"I64","shape":[1],"data_offsets":[24,32]}}"#;
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        data.extend_from_slice(&0i64.to_le_bytes());
        let file = path("safetensors");
        std::fs::write(&file, fixture(header, &data)).unwrap();
        let model = Linear::no_bias(2, 3);
        load_safetensors(&model, &file).unwrap();
        assert_eq!(model.weight.to_vec(), expected);
        assert!(read_safetensors(&file).unwrap_err().to_string().contains("I64"));
        let malformed = header.replace("[24,32]", "[24,33]");
        std::fs::write(&file, fixture(&malformed, &data)).unwrap();
        assert!(load_safetensors(&model, &file).is_err());
        assert_eq!(model.weight.to_vec(), expected);

        let header = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"bias":{"dtype":"I64","shape":[3],"data_offsets":[24,48]}}"#;
        data.extend_from_slice(&[0; 16]);
        std::fs::write(&file, fixture(header, &data)).unwrap();
        let model = Linear::new(2, 3);
        let before = model.weight.to_vec();
        let before_bias = model.bias.as_ref().unwrap().to_vec();
        assert!(load_safetensors(&model, &file).unwrap_err().to_string().contains("I64"));
        assert_eq!(model.weight.to_vec(), before);
        assert_eq!(model.bias.as_ref().unwrap().to_vec(), before_bias);
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn failed_load_does_not_partly_overwrite_model() {
        for extension in ["rtsc", "safetensors"] {
            let source = Linear::new(2, 3);
            source.weight.copy_from_slice(&[42.0; 6]);
            let target = Linear::no_bias(2, 3);
            let missing_bias = Linear::new(2, 3);
            let before = missing_bias.weight.to_vec();
            let file = path(extension);
            save_state_dict(&target, &file).unwrap();
            assert!(load_state_dict(&missing_bias, &file).is_err());
            assert_eq!(missing_bias.weight.to_vec(), before);
            save_state_dict(&source, &file).unwrap();
            let wrong_shape = Linear::new(2, 4);
            let before = wrong_shape.weight.to_vec();
            assert!(load_state_dict(&wrong_shape, &file).is_err());
            assert_eq!(wrong_shape.weight.to_vec(), before);
            std::fs::remove_file(file).unwrap();
        }
    }

    #[test]
    fn rtsc_rejects_data_length_mismatch() {
        let mut bytes = b"RTSC".to_vec();
        for n in [1u32, 6] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(b"weight");
        for n in [2u32, 2, 3, 1] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let file = path("rtsc");
        std::fs::write(&file, bytes).unwrap();
        assert_eq!(load_state_dict(&Linear::no_bias(2, 3), &file).unwrap_err().kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn rtim_rejects_unknown_activation() {
        let mut bytes = b"RTIM".to_vec();
        for n in [1u32, 6, 99] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let file = path("rtim");
        std::fs::write(&file, bytes).unwrap();
        assert_eq!(crate::save::load_inference_layers(&file).unwrap_err().kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(file).unwrap();
    }
}
