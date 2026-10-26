//! Reverse-mode automatic differentiation engine.
//!
//! Every differentiable operation records a [`Node`] holding its inputs and a
//! backward closure. Calling [`Tensor::backward`] topologically sorts the graph
//! and propagates gradients from the output to the leaves (chain rule).

use crate::tensor::Tensor;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// Backward closure: receives `dL/d(output)` and returns `dL/d(input_i)` for
/// every input (or `None` if that input does not need a gradient).
pub type BackwardFn = Box<dyn Fn(&[f32]) -> Vec<Option<Vec<f32>>> + Send + Sync>;

/// A node of the computation graph.
pub struct Node {
    pub name: &'static str,
    pub inputs: Vec<Tensor>,
    pub backward: BackwardFn,
}

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Is graph recording enabled on this thread?
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Enable / disable graph recording on this thread.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|g| g.set(enabled));
}

/// RAII guard that disables gradient tracking until dropped.
/// The guard stays on its creating thread because it restores thread-local state.
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
        set_grad_enabled(self.prev);
    }
}

/// Run `f` without recording a graph (like `with torch.no_grad():`).
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _g = NoGradGuard::new();
    f()
}

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

/// Back-propagate `seed` (= dL/d(root)) through the graph rooted at `root`.
pub(crate) fn run_backward(root: &Tensor, seed: Vec<f32>) {
    // 1. topological order (iterative DFS, post-order)
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

    // 2. propagate in reverse topological order
    let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
    grads.insert(root.id(), seed);
    for t in order.iter().rev() {
        let g = match grads.remove(&t.id()) {
            Some(g) => g,
            None => continue,
        };
        match t.node() {
            Some(node) => {
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
            None => t.accumulate_grad(&g),
        }
    }
}
