//! 연산 백엔드 추상화 인터페이스.
//!
//! 모든 [`Device`](super::Device)는 고성능 수치 연산 커널을 구현하는 `Backend` 트레이트로 연결됩니다.
//! 텐서는 호스트 메모리에 기본 데이터를 저장하며, 백엔드는 커널이 실행될 위치(Rayon CPU 멀티스레드, GPU 컴퓨트 셰이더 등)를 결정합니다.

/// 요소별 이항 연산자 (사칙연산).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// 덧셈 (`+`)
    Add,
    /// 뺄셈 (`-`)
    Sub,
    /// 곱셈 (`*`)
    Mul,
    /// 나눗셈 (`/`)
    Div,
}

impl BinaryOp {
    /// 두 스칼라 값에 연산자를 적용합니다.
    #[inline]
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        }
    }

    /// GPU 셰이더에서 사용하는 정수 연산자 식별 코드
    pub fn code(self) -> u32 {
        self as u32
    }
}

/// 요소별 단항 연산자 (수학 함수 및 활성화 함수).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// 부호 반전 (`-x`)
    Neg,
    /// 자연 지수 함수 (`e^x`)
    Exp,
    /// 자연로그 함수 (`ln(x)`)
    Log,
    /// ReLU 함수 (`max(0, x)`)
    Relu,
    /// 시그모이드 함수 (`1 / (1 + e^-x)`)
    Sigmoid,
    /// 쌍곡탄젠트 함수 (`tanh(x)`)
    Tanh,
    /// 제곱근 (`sqrt(x)`)
    Sqrt,
    /// GELU 근사 함수
    Gelu,
    /// 절댓값 (`|x|`)
    Abs,
}

/// GELU 계산 상수: sqrt(2/pi)
const GELU_C: f32 = 0.797_884_6;

impl UnaryOp {
    /// 스칼라 값에 단항 연산자를 적용합니다.
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

    /// 입력 `x`와 출력 `y`가 주어졌을 때의 도함수 `dy/dx`를 계산합니다 (역전파용).
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

    /// GPU 공유 셰이더 내에서 사용하는 단항 연산자 식별 코드 (이항 연산자 0..3과 구분하기 위해 10부터 시작)
    pub fn code(self) -> u32 {
        10 + self as u32
    }
}

/// 연산 백엔드 트레이트.
/// 다중 GPU 데이터 병렬 처리에서 여러 학습 레플리카가 동시 실행될 수 있도록 스레드 안전(`Send + Sync`)해야 합니다.
pub trait Backend: Send + Sync {
    /// 사람이 읽을 수 있는 백엔드 이름 (예: `"cpu (rayon x16)"` 또는 `"gpu:0 AMD Radeon (Vulkan)"`).
    fn name(&self) -> String;

    /// 행 우선(Row-major) 행렬 곱셈: `C[m,n] = A[m,k] @ B[k,n]`.
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

    /// 동일한 크기의 두 슬라이스에 대한 요소별 이항 연산.
    fn binary(&self, op: BinaryOp, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// 슬라이스에 대한 요소별 단항 연산.
    fn unary(&self, op: UnaryOp, a: &[f32]) -> Vec<f32>;
}
