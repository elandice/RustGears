//! 자동 미분(Autograd)을 지원하는 N차원 다차원 텐서(Tensor) 모듈.

pub(crate) mod ops;
pub mod shape;

use crate::autograd::{self, BackwardFn, Node};
use crate::device::{Backend, Device};
use crate::rng;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// 전역 텐서 ID 생성기
static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// 텐서의 내부 데이터 저장소 구조체
pub(crate) struct TensorInner {
    /// 텐서 고유 식별 번호
    id: usize,
    /// 텐서의 평탄화된 f32 요소 데이터를 저장하는 버퍼 (스레드 안전한 읽기/쓰기 지원)
    data: RwLock<Vec<f32>>,
    /// 텐서의 차원 형태(Shape)
    shape: Vec<usize>,
    /// 텐서가 상주하는 디바이스 (CPU 또는 GPU)
    device: RwLock<Device>,
    /// 역방향 자동 미분 시 기울기 계산 필요 여부 플래그
    requires_grad: AtomicBool,
    /// 역전파 과정에서 누적되는 기울기 버퍼
    grad: Mutex<Option<Vec<f32>>>,
    /// 연산 그래프 상에서 이 텐서를 생성한 연산 노드 (리프 텐서일 경우 None)
    node: Option<Node>,
}

/// 참조 카운팅(`Arc`) 기반의 스레드 안전한 N차원 `f32` 텐서.
///
/// 텐서를 클론(clone)하는 것은 매우 가볍습니다 (`Arc` 참조 복사만 수행).
/// 두 핸들은 동일한 기본 메모리를 가리키며, Python의 `torch.Tensor` 참조 방식과 동일하게 동작합니다.
#[derive(Clone)]
pub struct Tensor(Arc<TensorInner>);

impl Tensor {
    // ------------------------------------------------------------------
    // 생성자 (Construction)
    // ------------------------------------------------------------------

    /// 내부 속성들을 조립하여 새 텐서를 구축합니다.
    fn build(data: Vec<f32>, shape: Vec<usize>, device: Device, requires_grad: bool, node: Option<Node>) -> Tensor {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length {} does not match shape {:?}", data.len(), shape);
        Tensor(Arc::new(TensorInner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: RwLock::new(data),
            shape,
            device: RwLock::new(device),
            requires_grad: AtomicBool::new(requires_grad),
            grad: Mutex::new(None),
            node,
        }))
    }

    /// 데이터 벡터와 형태(Shape)를 지정하여 CPU 텐서를 생성합니다.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Self::build(data, shape.to_vec(), Device::Cpu, false, None)
    }

    /// 데이터 슬라이스와 형태를 지정하여 텐서를 생성합니다.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Tensor {
        Self::from_vec(data.to_vec(), shape)
    }

    /// 2차원 중첩 행 벡터들로부터 2차원 텐서를 생성합니다.
    pub fn from_rows(rows: &[Vec<f32>]) -> Tensor {
        let r = rows.len();
        let c = rows.first().map(|x| x.len()).unwrap_or(0);
        let data: Vec<f32> = rows
            .iter()
            .flat_map(|row| {
                assert_eq!(row.len(), c, "ragged rows");
                row.iter().copied()
            })
            .collect();
        Self::from_vec(data, &[r, c])
    }

    /// 0차원 스칼라(Scalar) 텐서를 생성합니다.
    pub fn scalar(v: f32) -> Tensor {
        Self::from_vec(vec![v], &[])
    }

    /// 지정된 `value` 값으로 채워진 주어진 형태의 텐서를 생성합니다.
    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Self::from_vec(vec![value; shape.iter().product()], shape)
    }

    /// 모든 원소가 0.0으로 초기화된 텐서를 생성합니다.
    pub fn zeros(shape: &[usize]) -> Tensor {
        Self::full(shape, 0.0)
    }

    /// 모든 원소가 1.0으로 초기화된 텐서를 생성합니다.
    pub fn ones(shape: &[usize]) -> Tensor {
        Self::full(shape, 1.0)
    }

    /// 대상 텐서 `t`와 동일한 형상 및 디바이스를 가지는 0 텐서를 생성합니다.
    pub fn zeros_like(t: &Tensor) -> Tensor {
        Self::zeros(t.shape()).on(t.device())
    }

    /// 대상 텐서 `t`와 동일한 형상 및 디바이스를 가지는 1 텐서를 생성합니다.
    pub fn ones_like(t: &Tensor) -> Tensor {
        Self::ones(t.shape()).on(t.device())
    }

    /// 표준 정규 분포 `N(0, 1)`를 따르는 난수로 채워진 텐서를 생성합니다.
    pub fn randn(shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Self::from_vec(rng::fill_with(n, |r| r.normal()), shape)
    }

    /// 균등 분포 `U[0, 1)`를 따르는 난수로 채워진 텐서를 생성합니다.
    pub fn rand(shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Self::from_vec(rng::fill_with(n, |r| r.uniform()), shape)
    }

    /// 범위 `U[lo, hi)`의 균등 분포 난수로 채워진 텐서를 생성합니다.
    pub fn uniform(shape: &[usize], lo: f32, hi: f32) -> Tensor {
        let n = shape.iter().product();
        Self::from_vec(rng::fill_with(n, |r| r.uniform_range(lo, hi)), shape)
    }

    /// `[0, 1, ..., n-1]`의 연속된 정수로 채워진 1차원 텐서를 생성합니다.
    pub fn arange(n: usize) -> Tensor {
        Self::from_vec((0..n).map(|i| i as f32).collect(), &[n])
    }

    /// 크기 `n x n`의 단위 행렬(Identity matrix) 텐서를 생성합니다.
    pub fn eye(n: usize) -> Tensor {
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            d[i * n + i] = 1.0;
        }
        Self::from_vec(d, &[n, n])
    }

    /// 빌더 패턴: 학습 가능한 리프 파라미터로 표시합니다 (`requires_grad` 설정).
    pub fn requires_grad_(self, flag: bool) -> Tensor {
        assert!(self.0.node.is_none() || !flag, "requires_grad_ can only be set on leaf tensors");
        self.0.requires_grad.store(flag, Ordering::Relaxed);
        self
    }

    /// 빌더 패턴: 이 텐서의 대상 디바이스를 지정합니다 (복사 없이 메타데이터 변경).
    pub fn on(self, device: Device) -> Tensor {
        self.set_device(device);
        self
    }

    /// 미분 가능한 연산자(ops)에서 새 결과 텐서를 생성할 때 사용하는 내부 생성자.
    pub(crate) fn from_op(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
        inputs: Vec<Tensor>,
        name: &'static str,
        backward: BackwardFn,
    ) -> Tensor {
        let needs = autograd::is_grad_enabled() && inputs.iter().any(|t| t.requires_grad());
        if needs {
            let node = Node { name, inputs, backward };
            Self::build(data, shape, device, true, Some(node))
        } else {
            Self::build(data, shape, device, false, None)
        }
    }

    // ------------------------------------------------------------------
    // 속성 접근자 (Accessors)
    // ------------------------------------------------------------------

    /// 텐서 고유 식별자 ID를 반환합니다.
    pub fn id(&self) -> usize {
        self.0.id
    }

    /// 텐서의 형태(Shape) 슬라이스를 반환합니다.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// 특정 축(차원 `d`)의 크기를 반환합니다 (음수 인덱스 지원: -1은 마지막 차원).
    pub fn size(&self, d: isize) -> usize {
        self.0.shape[self.dim_index(d)]
    }

    /// 텐서의 총 차원 수(Rank)를 반환합니다.
    pub fn ndim(&self) -> usize {
        self.0.shape.len()
    }

    /// 텐서의 총 요소(원소) 개수를 반환합니다.
    pub fn numel(&self) -> usize {
        self.0.shape.iter().product()
    }

    /// 현재 텐서가 위치한 디바이스를 반환합니다.
    pub fn device(&self) -> Device {
        *self.0.device.read().unwrap()
    }

    /// 텐서(및 향후 연산)의 디바이스를 내부에서 변경합니다.
    /// `Module::to` 등에서 매개변수의 디바이스를 이동시킬 때 사용됩니다.
    pub fn set_device(&self, device: Device) {
        *self.0.device.write().unwrap() = device;
    }

    /// 현재 디바이스에 연결된 연산 백엔드(Backend) 인스턴스를 반환합니다.
    pub(crate) fn backend(&self) -> Arc<dyn Backend> {
        self.device().backend()
    }

    /// 기울기 계산이 활성화되어 있는지 여부를 반환합니다.
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

    /// 이 텐서가 연산의 결과가 아닌 리프(Leaf) 노드인지 확인합니다.
    pub fn is_leaf(&self) -> bool {
        self.0.node.is_none()
    }

    /// 연산 그래프 노드에 대한 참조를 반환합니다.
    pub(crate) fn node(&self) -> Option<&Node> {
        self.0.node.as_ref()
    }

    /// 이 텐서를 생성한 연산자의 이름(예: "AddBackward")을 반환합니다 (리프인 경우 None).
    pub fn grad_fn(&self) -> Option<&'static str> {
        self.0.node.as_ref().map(|n| n.name)
    }

    /// 기본 데이터 버퍼에 대한 스레드 안전한 읽기 잠금 가드를 획득합니다.
    pub fn data(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.0.data.read().unwrap()
    }

    /// 기본 데이터 버퍼에 대한 쓰기 잠금 가드를 획득합니다 (옵티마이저 가중치 업데이트용).
    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.0.data.write().unwrap()
    }

    /// 내부 데이터를 새로운 `Vec<f32>`로 복제하여 반환합니다.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data().clone()
    }

    /// 단일 원소를 갖는 스칼라 텐서에서 f32 값을 추출합니다.
    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() requires a single-element tensor, got {:?}", self.shape());
        self.data()[0]
    }

    /// 다차원 인덱스 위치에 해당하는 원소의 값을 반환합니다.
    pub fn at(&self, idx: &[usize]) -> f32 {
        let strides = shape::strides(self.shape());
        let off: usize = idx.iter().zip(&strides).map(|(i, s)| i * s).sum();
        self.data()[off]
    }

    /// 데이터 슬라이스로부터 내부 버퍼를 덮어씁니다 (형상은 동일해야 함).
    pub fn copy_from_slice(&self, src: &[f32]) {
        let mut d = self.data_mut();
        assert_eq!(d.len(), src.len(), "copy_from_slice: size mismatch");
        d.copy_from_slice(src);
    }

    // ------------------------------------------------------------------
    // 기울기 관리 (Gradients)
    // ------------------------------------------------------------------

    /// 누적된 기울기를 `Tensor` 형태로 반환합니다.
    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.lock().unwrap().as_ref().map(|g| Tensor::from_vec(g.clone(), self.shape()).on(self.device()))
    }

    /// 누적된 기울기를 원시 벡터(`Vec<f32>`)로 반환합니다.
    pub fn grad_vec(&self) -> Option<Vec<f32>> {
        self.0.grad.lock().unwrap().clone()
    }

    /// 기울기 버퍼를 명시적으로 교체합니다 (all-reduce 또는 그래디언트 클리핑 시 활용).
    pub fn set_grad(&self, g: Option<Vec<f32>>) {
        if let Some(g) = &g {
            assert_eq!(g.len(), self.numel(), "set_grad: size mismatch");
        }
        *self.0.grad.lock().unwrap() = g;
    }

    /// 전달받은 기울기를 기존 누적 기울기 버퍼에 요소별로 더합니다.
    pub(crate) fn accumulate_grad(&self, g: &[f32]) {
        let mut slot = self.0.grad.lock().unwrap();
        match slot.as_mut() {
            Some(acc) => acc.iter_mut().zip(g).for_each(|(a, b)| *a += b),
            None => *slot = Some(g.to_vec()),
        }
    }

    /// 누적된 기울기 벡터가 존재할 경우 클로저를 통해 직접 수정합니다.
    pub fn with_grad_mut(&self, f: impl FnOnce(&mut Vec<f32>)) {
        if let Some(g) = self.0.grad.lock().unwrap().as_mut() {
            f(g)
        }
    }

    /// 누적된 기울기를 `None`으로 초기화(0으로 리셋)합니다.
    pub fn zero_grad(&self) {
        *self.0.grad.lock().unwrap() = None;
    }

    /// 이 스칼라 텐서를 기점으로 계산 그래프의 역전파(Backpropagation)를 수행합니다.
    pub fn backward(&self) {
        assert_eq!(self.numel(), 1, "backward() without arguments requires a scalar output (got {:?})", self.shape());
        assert!(
            self.requires_grad(),
            "backward(): tensor does not require grad (no trainable inputs or no_grad active)"
        );
        autograd::run_backward(self, vec![1.0]);
    }

    /// 명시적으로 지정된 외부 출력 기울기를 사용하여 역전파를 수행합니다.
    pub fn backward_with(&self, grad: &Tensor) {
        assert_eq!(grad.shape(), self.shape(), "backward_with: gradient shape mismatch");
        autograd::run_backward(self, grad.to_vec());
    }

    /// 계산 그래프 이력(graph history)과의 연결을 끊고 데이터만 복사한 새 리프 텐서를 생성합니다.
    pub fn detach(&self) -> Tensor {
        Tensor::from_vec(self.to_vec(), self.shape()).on(self.device())
    }

    /// 음수 차원 인덱스를 양수 인덱스로 정규화합니다.
    pub(crate) fn dim_index(&self, d: isize) -> usize {
        let n = self.ndim() as isize;
        let i = if d < 0 { n + d } else { d };
        assert!(i >= 0 && i < n, "dimension {d} out of range for tensor with {n} dims");
        i as usize
    }

    /// 행(0번 차원) 인덱스들을 기준으로 부분 선택합니다 (데이터 로딩용, 미분 불가).
    pub fn index_rows(&self, rows: &[usize]) -> Tensor {
        let row = self.numel() / self.shape()[0].max(1);
        let d = self.data();
        let mut out = Vec::with_capacity(rows.len() * row);
        for &r in rows {
            out.extend_from_slice(&d[r * row..(r + 1) * row]);
        }
        let mut shape = self.shape().to_vec();
        shape[0] = rows.len();
        Tensor::from_vec(out, &shape).on(self.device())
    }

    /// 0번 축(배치 차원)에 대해 `[start, start+len)` 구간의 연속된 행들을 슬라이싱합니다 (미분 불가).
    pub fn narrow_rows(&self, start: usize, len: usize) -> Tensor {
        assert!(self.ndim() > 0, "narrow_rows requires a batch dimension");
        let rows = self.shape()[0];
        assert!(start <= rows && len <= rows - start, "narrow_rows range exceeds the batch dimension");
        let row = self.numel() / rows.max(1);
        let data = self.data()[start * row..(start + len) * row].to_vec();
        let mut shape = self.shape().to_vec();
        shape[0] = len;
        Self::build(data, shape, self.device(), false, None)
    }

    /// 마지막 차원을 따라 각 행에서 최댓값을 갖는 위치의 인덱스를 벡터로 반환합니다.
    pub fn argmax(&self) -> Vec<usize> {
        let c = *self.shape().last().expect("argmax on scalar");
        self.data()
            .chunks(c)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                    .0
            })
            .collect()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.data();
        let shown: Vec<String> = d.iter().take(8).map(|v| format!("{v:.4}")).collect();
        let more = if d.len() > 8 { ", ..." } else { "" };
        write!(f, "tensor([{}{}], shape={:?}, device={}", shown.join(", "), more, self.shape(), self.device())?;
        if let Some(n) = self.grad_fn() {
            write!(f, ", grad_fn={n}")?;
        } else if self.requires_grad() {
            write!(f, ", requires_grad=true")?;
        }
        write!(f, ")")
    }
}
