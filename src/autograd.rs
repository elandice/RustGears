//! 역방향 모드 자동 미분 (Reverse-mode Autograd) 엔진.
//!
//! 미분 가능한 모든 연산은 입력 텐서들과 역전파 클로저를 담은 [`Node`]를 기록합니다.
//! [`Tensor::backward`]를 호출하면 계산 그래프를 위상 정렬(topological sort)하고
//! 연쇄 법칙(chain rule)에 따라 출력 노드에서 잎(leaf) 노드로 기울기(gradient)를 역전파합니다.

use crate::tensor::Tensor;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// 역전파 클로저 타입:
/// 상위 노드로부터 `dL/d(output)`을 전달받아 각 입력에 대한 `dL/d(input_i)`를 반환합니다.
/// (해당 입력이 기울기를 필요로 하지 않는 경우 `None` 반환)
pub type BackwardFn = Box<dyn Fn(&[f32]) -> Vec<Option<Vec<f32>>> + Send + Sync>;

/// 연산 그래프(DAG)의 단일 연산 노드.
pub struct Node {
    /// 연산 식별용 이름 (예: "matmul", "add", "relu" 등)
    pub name: &'static str,
    /// 해당 연산의 입력으로 사용된 텐서들
    pub inputs: Vec<Tensor>,
    /// 출력 기울기를 입력 텐서들의 기울기로 변환하는 역전파 함수
    pub backward: BackwardFn,
}

thread_local! {
    /// 스레드별 기울기 추적 활성화 상태 플래그
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// 현재 스레드에서 연산 그래프 기록(기울기 추적)이 활성화되어 있는지 확인합니다.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// 현재 스레드에서 연산 그래프 기록 활성화 여부를 설정합니다.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|g| g.set(enabled));
}

/// 드롭될 때까지 기울기 추적을 비활성화하는 RAII 가드.
/// 스레드 로컬 상태를 복구해야 하므로 다른 스레드로 전송할 수 없습니다.
///
/// ```compile_fail
/// let guard = rustgears::autograd::NoGradGuard::new();
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct NoGradGuard {
    prev: bool,
    thread_bound: PhantomData<*mut ()>,
}

impl NoGradGuard {
    /// 새로운 NoGradGuard를 생성하여 현재 스레드의 기울기 계산을 끕니다.
    pub fn new() -> Self {
        let prev = is_grad_enabled();
        set_grad_enabled(false);
        NoGradGuard { prev, thread_bound: PhantomData }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        // 가드가 해제될 때 이전 기울기 추적 활성화 상태로 복원
        set_grad_enabled(self.prev);
    }
}

/// 연산 그래프를 기록하지 않고 클로저 `f`를 실행합니다 (`Python`의 `with torch.no_grad():`와 동일).
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _g = NoGradGuard::new();
    f()
}

/// 텐서 ID별 기울기 맵에 새로운 기울기를 요소별로 누적(add)합니다.
fn accumulate(map: &mut HashMap<usize, Vec<f32>>, id: usize, g: Vec<f32>) {
    match map.get_mut(&id) {
        Some(acc) => {
            for (a, b) in acc.iter_mut().zip(g.iter()) {
                *a += b;
            }
        }
        None => {
            map.insert(id, g);
        }
    }
}

/// `root` 텐서를 시작점으로 하여 연산 그래프 전체에 시드 기울기(`seed` = dL/d(root))를 역전파합니다.
pub(crate) fn run_backward(root: &Tensor, seed: Vec<f32>) {
    // 1. 위상 정렬 (반복적 깊이 우선 탐색 DFS, 후위 순회 방식)
    let mut order: Vec<Tensor> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(Tensor, bool)> = vec![(root.clone(), false)];
    while let Some((t, expanded)) = stack.pop() {
        if expanded {
            order.push(t);
            continue;
        }
        if !visited.insert(t.id()) {
            continue;
        }
        stack.push((t.clone(), true));
        if let Some(node) = t.node() {
            for inp in &node.inputs {
                if inp.requires_grad() && !visited.contains(&inp.id()) {
                    stack.push((inp.clone(), false));
                }
            }
        }
    }

    // 2. 역위상 순서(reverse topological order)로 기울기 전파
    let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
    grads.insert(root.id(), seed);
    for t in order.iter().rev() {
        let g = match grads.remove(&t.id()) {
            Some(g) => g,
            None => continue,
        };
        match t.node() {
            Some(node) => {
                // 연산 노드의 역전파 함수 실행 -> 입력 텐서별 기울기 산출
                let input_grads = (node.backward)(&g);
                for (inp, ig) in node.inputs.iter().zip(input_grads) {
                    if let Some(ig) = ig {
                        if inp.requires_grad() {
                            debug_assert_eq!(ig.len(), inp.numel(), "grad size mismatch in {}", node.name);
                            accumulate(&mut grads, inp.id(), ig);
                        }
                    }
                }
            }
            // 리프(Leaf) 파라미터 노드에 도달하면 최종 기울기 누적
            None => t.accumulate_grad(&g),
        }
    }
}
