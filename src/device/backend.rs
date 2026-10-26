//! Compute backend abstraction.
//!
//! Every [`Device`](super::Device) resolves to a `Backend` that implements the
//! heavy numerical kernels. Tensors store their data in host memory; the
//! backend decides *where* the kernel is executed (CPU threads, GPU compute
//! shaders, ...).

/// Element-wise binary operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    #[inline]
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        }
    }

    pub fn code(self) -> u32 {
        self as u32
    }
}

/// Element-wise unary operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Relu,
    Sigmoid,
    Tanh,
    Sqrt,
    Gelu,
    Abs,
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)

impl UnaryOp {
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            UnaryOp::Neg => -x,
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Gelu => 0.5 * x * (1.0 + (GELU_C * (x + 0.044715 * x * x * x)).tanh()),
            UnaryOp::Abs => x.abs(),
        }
    }

    /// Derivative `dy/dx` given input `x` and output `y`.
    #[inline]
    pub fn derivative(self, x: f32, y: f32) -> f32 {
        match self {
            UnaryOp::Neg => -1.0,
            UnaryOp::Exp => y,
            UnaryOp::Log => 1.0 / x,
            UnaryOp::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            UnaryOp::Sigmoid => y * (1.0 - y),
            UnaryOp::Tanh => 1.0 - y * y,
            UnaryOp::Sqrt => 0.5 / y,
            UnaryOp::Gelu => {
                let inner = GELU_C * (x + 0.044715 * x * x * x);
                let t = inner.tanh();
                let dinner = GELU_C * (1.0 + 3.0 * 0.044715 * x * x);
                0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner
            }
            UnaryOp::Abs => {
                if x >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    pub fn code(self) -> u32 {
        // binary ops occupy 0..4 in the shared GPU shader
        10 + self as u32
    }
}

/// A compute backend. Implementations must be thread-safe so that several
/// training replicas can run concurrently (data parallelism).
pub trait Backend: Send + Sync {
    /// Human readable name, e.g. `"cpu (rayon x16)"` or `"gpu:0 AMD Radeon (Vulkan)"`.
    fn name(&self) -> String;

    /// Row-major matrix multiply: `C[m,n] = A[m,k] @ B[k,n]`.
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

    /// Element-wise binary op on equally sized slices.
    fn binary(&self, op: BinaryOp, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// Element-wise unary op.
    fn unary(&self, op: UnaryOp, a: &[f32]) -> Vec<f32>;
}
