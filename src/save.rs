//! Model state dicts (`RTSC` / Hugging Face Safetensors) and portable inference
//! models (`RTIM`).

mod safetensors;
pub use safetensors::{load_safetensors, read_safetensors, save_safetensors};

use crate::inference::{Activation, LayerSpec};
use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const STATE_MAGIC: &[u8; 4] = b"RTSC";
const MODEL_MAGIC: &[u8; 4] = b"RTIM";

fn write_atomic(path: impl AsRef<Path>, write: impl FnOnce(&mut BufWriter<&File>) -> io::Result<()>) -> io::Result<()> {
    let path = path.as_ref();
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let temp = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(temp.as_file());
        write(&mut writer)?;
        writer.flush()?;
    }
    temp.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_f32_slice(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    write_u32(w, v.len() as u32)?;
    for &x in v {
        write_f32(w, x)?;
    }
    Ok(())
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let n = read_u32(r)? as usize;
    let mut v = Vec::new();
    for _ in 0..n {
        v.push(read_f32(r)?);
    }
    Ok(v)
}

/// Save a module's `state_dict` (parameters + buffers) to a file.
/// The `.safetensors` extension selects Safetensors; other paths use `RTSC`.
pub fn save_state_dict(module: &dyn Module, path: impl AsRef<Path>) -> io::Result<()> {
    if path.as_ref().extension().is_some_and(|ext| ext == "safetensors") {
        return save_safetensors(module, path);
    }
    write_atomic(path, |mut w| {
        w.write_all(STATE_MAGIC)?;
        let sd = module.state_dict();
        write_u32(&mut w, sd.len() as u32)?;
        for (name, t) in sd {
            let nb = name.as_bytes();
            write_u32(&mut w, nb.len() as u32)?;
            w.write_all(nb)?;
            write_u32(&mut w, t.ndim() as u32)?;
            for &d in t.shape() {
                write_u32(&mut w, d as u32)?;
            }
            write_f32_slice(&mut w, &t.data())?;
        }
        Ok(())
    })
}

/// Load `RTSC` or Safetensors weights into `module` by exact name and shape.
pub fn load_state_dict(module: &dyn Module, path: impl AsRef<Path>) -> io::Result<()> {
    let mut r = BufReader::new(File::open(path.as_ref())?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != STATE_MAGIC {
        return load_safetensors(module, path);
    }
    let count = read_u32(&mut r)? as usize;
    let mut stored = HashMap::new();
    for _ in 0..count {
        let nlen = read_u32(&mut r)? as usize;
        let mut nb = Vec::new();
        r.by_ref().take(nlen as u64).read_to_end(&mut nb)?;
        if nb.len() != nlen {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated parameter name"));
        }
        let name = String::from_utf8(nb).map_err(invalid_data)?;
        let ndim = read_u32(&mut r)? as usize;
        let mut shape = Vec::new();
        for _ in 0..ndim {
            shape.push(read_u32(&mut r)? as usize);
        }
        let data = read_f32_vec(&mut r)?;
        let numel = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if numel != Some(data.len()) {
            return Err(invalid_data(format!("invalid data length for '{name}'")));
        }
        if stored.insert(name.clone(), Tensor::from_vec(data, &shape)).is_some() {
            return Err(invalid_data(format!("duplicate parameter '{name}'")));
        }
    }
    apply_state_dict(module, &stored)
}

fn invalid_data(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

fn apply_state_dict(module: &dyn Module, stored: &HashMap<String, Tensor>) -> io::Result<()> {
    let state = module.state_dict();
    // Validate the entire checkpoint before changing any live parameters.
    for (name, tensor) in &state {
        let Some(value) = stored.get(name) else {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("missing parameter '{name}' in checkpoint")));
        };
        if value.shape() != tensor.shape() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("shape mismatch for '{name}': expected {:?}, got {:?}", tensor.shape(), value.shape()),
            ));
        }
    }
    for (name, tensor) in state {
        tensor.copy_from_slice(&stored[&name].data());
    }
    Ok(())
}

/// Export and save a portable inference model (`.rtim`) for deployment on
/// CPU, GPU or NPU.
pub fn save_inference_model(module: &dyn Module, path: impl AsRef<Path>) -> Result<(), String> {
    module.eval();
    let layers = module.export()?;
    save_layers(&layers, path).map_err(|e| e.to_string())
}

pub fn save_layers(layers: &[LayerSpec], path: impl AsRef<Path>) -> io::Result<()> {
    write_atomic(path, |mut w| {
        w.write_all(MODEL_MAGIC)?;
        write_u32(&mut w, layers.len() as u32)?;
        for l in layers {
            match l {
                LayerSpec::Linear { in_features, out_features, weight, bias } => {
                    write_u32(&mut w, 1)?;
                    write_u32(&mut w, *in_features as u32)?;
                    write_u32(&mut w, *out_features as u32)?;
                    write_f32_slice(&mut w, weight)?;
                    write_f32_slice(&mut w, bias)?;
                }
                LayerSpec::Conv2d { in_channels, out_channels, kernel, stride, padding, weight, bias } => {
                    write_u32(&mut w, 2)?;
                    write_u32(&mut w, *in_channels as u32)?;
                    write_u32(&mut w, *out_channels as u32)?;
                    write_u32(&mut w, *kernel as u32)?;
                    write_u32(&mut w, *stride as u32)?;
                    write_u32(&mut w, *padding as u32)?;
                    write_f32_slice(&mut w, weight)?;
                    write_f32_slice(&mut w, bias)?;
                }
                LayerSpec::ChannelAffine { scale, shift } => {
                    write_u32(&mut w, 3)?;
                    write_f32_slice(&mut w, scale)?;
                    write_f32_slice(&mut w, shift)?;
                }
                LayerSpec::MaxPool2d { kernel, stride } => {
                    write_u32(&mut w, 4)?;
                    write_u32(&mut w, *kernel as u32)?;
                    write_u32(&mut w, *stride as u32)?;
                }
                LayerSpec::Flatten => write_u32(&mut w, 5)?,
                LayerSpec::Activation(act) => {
                    write_u32(&mut w, 6)?;
                    let (code, param) = match act {
                        Activation::None => (0, 0.0),
                        Activation::Relu => (1, 0.0),
                        Activation::Sigmoid => (2, 0.0),
                        Activation::Tanh => (3, 0.0),
                        Activation::Gelu => (4, 0.0),
                        Activation::LeakyRelu(s) => (5, *s),
                    };
                    write_u32(&mut w, code)?;
                    write_f32(&mut w, param)?;
                }
                LayerSpec::Softmax => write_u32(&mut w, 7)?,
            }
        }
        Ok(())
    })
}

/// Load portable [`LayerSpec`]s from a `.rtim` file.
pub fn load_inference_layers(path: impl AsRef<Path>) -> io::Result<Vec<LayerSpec>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MODEL_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid RTIM model header"));
    }
    let count = read_u32(&mut r)? as usize;
    let mut out = Vec::new();
    for _ in 0..count {
        let tag = read_u32(&mut r)?;
        let spec = match tag {
            1 => LayerSpec::Linear {
                in_features: read_u32(&mut r)? as usize,
                out_features: read_u32(&mut r)? as usize,
                weight: read_f32_vec(&mut r)?,
                bias: read_f32_vec(&mut r)?,
            },
            2 => LayerSpec::Conv2d {
                in_channels: read_u32(&mut r)? as usize,
                out_channels: read_u32(&mut r)? as usize,
                kernel: read_u32(&mut r)? as usize,
                stride: read_u32(&mut r)? as usize,
                padding: read_u32(&mut r)? as usize,
                weight: read_f32_vec(&mut r)?,
                bias: read_f32_vec(&mut r)?,
            },
            3 => LayerSpec::ChannelAffine { scale: read_f32_vec(&mut r)?, shift: read_f32_vec(&mut r)? },
            4 => LayerSpec::MaxPool2d { kernel: read_u32(&mut r)? as usize, stride: read_u32(&mut r)? as usize },
            5 => LayerSpec::Flatten,
            6 => {
                let code = read_u32(&mut r)?;
                let p = read_f32(&mut r)?;
                let act = match code {
                    0 => Activation::None,
                    1 => Activation::Relu,
                    2 => Activation::Sigmoid,
                    3 => Activation::Tanh,
                    4 => Activation::Gelu,
                    5 => Activation::LeakyRelu(p),
                    _ => return Err(invalid_data(format!("unknown activation tag {code}"))),
                };
                LayerSpec::Activation(act)
            }
            7 => LayerSpec::Softmax,
            other => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown layer tag {other}")));
            }
        };
        out.push(spec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_preserves_existing_file_and_cleans_up_on_failure() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.rtsc");
        std::fs::write(&path, b"original").unwrap();
        let error = write_atomic(&path, |writer| {
            writer.write_all(b"partial replacement")?;
            Err(io::Error::new(io::ErrorKind::WriteZero, "injected write failure"))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        write_atomic(&path, |writer| writer.write_all(b"replacement")).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"replacement");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
