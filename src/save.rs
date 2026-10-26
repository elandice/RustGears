//! Binary serialisation for model state dicts (`RTSC`) and portable inference
//! models (`RTIM`).

use crate::inference::{Activation, LayerSpec};
use crate::nn::Module;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const STATE_MAGIC: &[u8; 4] = b"RTSC";
const MODEL_MAGIC: &[u8; 4] = b"RTIM";

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
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(read_f32(r)?);
    }
    Ok(v)
}

/// Save a module's `state_dict` (parameters + buffers) to a file.
pub fn save_state_dict(module: &dyn Module, path: impl AsRef<Path>) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
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
    w.flush()
}

/// Load a `state_dict` file into `module` (matched by parameter name).
pub fn load_state_dict(module: &dyn Module, path: impl AsRef<Path>) -> io::Result<()> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != STATE_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid state_dict magic header"));
    }
    let count = read_u32(&mut r)? as usize;
    let mut stored = std::collections::HashMap::with_capacity(count);
    for _ in 0..count {
        let nlen = read_u32(&mut r)? as usize;
        let mut nb = vec![0u8; nlen];
        r.read_exact(&mut nb)?;
        let name = String::from_utf8_lossy(&nb).into_owned();
        let ndim = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(&mut r)? as usize);
        }
        let data = read_f32_vec(&mut r)?;
        stored.insert(name, (shape, data));
    }

    for (name, tensor) in module.state_dict() {
        let Some((shape, data)) = stored.get(&name) else {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("missing parameter '{name}' in checkpoint")));
        };
        if shape.as_slice() != tensor.shape() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("shape mismatch for '{name}': expected {:?}, got {:?}", tensor.shape(), shape),
            ));
        }
        tensor.copy_from_slice(data);
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
    let mut w = BufWriter::new(File::create(path)?);
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
    w.flush()
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
    let mut out = Vec::with_capacity(count);
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
            3 => LayerSpec::ChannelAffine {
                scale: read_f32_vec(&mut r)?,
                shift: read_f32_vec(&mut r)?,
            },
            4 => LayerSpec::MaxPool2d {
                kernel: read_u32(&mut r)? as usize,
                stride: read_u32(&mut r)? as usize,
            },
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
                    _ => Activation::LeakyRelu(p),
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
