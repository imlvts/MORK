//! Kernel-backed execution: a fast path that runs *alongside* the
//! tree-walking evaluator in [`super::eval`], never instead of it.
//!
//! # Contract
//!
//! The tree-walker is the semantics oracle. Everything here is required to
//! be **bit-identical** to it (LANGUAGE.md §10 E1 — deterministic per
//! artifact), not merely close, and to fall back to it silently for
//! anything it does not recognise. The design follows from that:
//!
//! - **The loop nest is the oracle's loop nest.** A reduction iterates its
//!   free indices outermost (ascending index id — exactly what
//!   `eval::lower` computes) and its bound indices innermost in *binder
//!   order*, folding left-to-right from the operator's identity. Nothing is
//!   reassociated, nothing is reordered, no algebraic identity is applied.
//! - **Reductions still materialize into temps** (§10 E3), in the same
//!   post-order, so a softmax denominator stays a single O(n) pass and a
//!   registered function is called exactly as often as before.
//! - **Sparse storage stays on the oracle.** Skipping structural zeros
//!   changes an addition sequence (`-0.0 + 0.0` is observable), so any
//!   input without a contiguous row-major image
//!   ([`NDIndex::as_flat_slice`]) sends the whole run to the tree-walker.
//!
//! The one thing *not* covered by that promise is a **NaN's payload**.
//! IEEE-754 leaves the choice of which input NaN an operation propagates
//! unspecified, and LLVM commutes `fadd`/`fmul` operands freely, so
//! `NaN_a + NaN_b` can return either operand depending only on how the
//! surrounding loop was compiled. Whether a result *is* NaN is
//! deterministic; its sign and payload are not. Every other value —
//! `±0.0` and `±∞` included — is exact. (`tests/lang_sweep.rs` measures
//! the size of this hole: 1 element in 72 284 across 20 000 random
//! programs.)
//!
//! # What it buys
//!
//! The v1 evaluator costs ~21–25 ns per element visit: a recursive `EExpr`
//! walk, a `Vec` index scratch rebuilt per load, and a `dyn NDIndex::get`
//! vtable call per operand. This module removes all three.
//!
//! 1. **Flat storage.** Every tensor is a `&[T]`; every load is a
//!    precomputed stride dot-product, resolved once per outer iteration.
//! 2. **A staged tape.** Each region compiles to a linear, register-numbered
//!    tape in two stages: an *outer* scalar tape holding the maximal
//!    subexpressions that do not depend on the innermost loop index
//!    (loop-invariant code motion — this is what collapses rmsnorm's
//!    `sqrt(sum(j: x[j]*x[j])/n + eps)` to one scalar), and an *inner* tape
//!    run in blocks of [`BLOCK`] elements. Block granularity amortises the
//!    per-op dispatch over 64 elements and lets each op body autovectorize.
//! 3. **The JIT, where it is provably order-identical.** A reduction whose
//!    body is a left-leaning chain of tensor loads under `*` or `+`, folded
//!    with the built-in `sum`/`prod`, is a classic einsum contraction: it
//!    is lowered to [`crate::jit::EinsumF32Jit`] (f32 only, cached per
//!    spec+shapes) after checking that the backend's contraction order —
//!    first appearance across the input patterns — is exactly the binder
//!    order. `max`/`min` reductions and `max`/`min` combines are *not*
//!    routed: Cranelift's `fmax`/`fmin` disagree with the language's
//!    comparison fold on `±0.0` and NaN.

use std::collections::HashMap;

use super::ast::BinOp;
use super::check::{CExpr, Checked, IndexId};
use super::eval::for_each_index;
use super::{Elem, Registry};
use crate::tensor::NDIndex;

/// Elements processed per inner-tape step. Big enough to amortise the
/// per-op dispatch and fill a few vector registers, small enough that the
/// whole block register file stays in L1.
const BLOCK: usize = 64;

// ─────────────────────────────────────────────────────────────────────────
// Buffers
// ─────────────────────────────────────────────────────────────────────────

/// A tensor in flat row-major form: borrowed from an input, or owned
/// because we computed it (a reduction temp or a statement result).
enum Buf<'a, T> {
    Slice(&'a [T]),
    Owned(Vec<T>),
}

impl<T> Buf<'_, T> {
    fn as_slice(&self) -> &[T] {
        match self {
            Buf::Slice(s) => s,
            Buf::Owned(v) => v,
        }
    }
}

/// Row-major strides of `shape`, in elements (empty for rank 0).
fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * shape[k + 1];
    }
    s
}

fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ─────────────────────────────────────────────────────────────────────────
// Reduce-free expression over flat buffers
// ─────────────────────────────────────────────────────────────────────────

/// What is left of a statement RHS once every reduction node has been
/// materialized — the same shape as `eval::EExpr`, but naming flat buffers
/// instead of `dyn NDIndex` handles.
enum FExpr<T> {
    Num(T),
    Load { buf: usize, indices: Vec<IndexId>, strides: Vec<usize> },
    Call { func: usize, args: Vec<FExpr<T>> },
    Binary { op: BinOp, lhs: Box<FExpr<T>>, rhs: Box<FExpr<T>> },
}

impl<T> FExpr<T> {
    fn collect_ids(&self, out: &mut Vec<IndexId>) {
        match self {
            FExpr::Num(_) => {}
            FExpr::Load { indices, .. } => out.extend_from_slice(indices),
            FExpr::Call { args, .. } => args.iter().for_each(|a| a.collect_ids(out)),
            FExpr::Binary { lhs, rhs, .. } => {
                lhs.collect_ids(out);
                rhs.collect_ids(out);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Compiled region: a loop nest plus a two-stage tape
// ─────────────────────────────────────────────────────────────────────────

/// One tensor read by a region, resolved against the region's loop nest.
struct LoadDesc {
    buf: usize,
    /// Stride contributed by each loop level (0 where the level's index
    /// does not occur in this reference; a repeated index — `M[i,i]` —
    /// contributes the sum of its positional strides).
    strides: Vec<usize>,
}

/// Scalar tape: the maximal subexpressions independent of the innermost
/// loop level, evaluated once per outer iteration. Operands are earlier
/// registers of this same tape.
enum OutOp<T> {
    Const(T),
    Load { load: usize },
    Bin { op: BinOp, a: usize, b: usize },
    Call { func: usize, args: Vec<usize> },
}

/// Block tape: everything that varies with the innermost loop level, run
/// [`BLOCK`] elements at a time. `Splat` lifts an outer scalar into a block.
enum InOp {
    Splat { outer: usize },
    Gather { load: usize },
    Bin { op: BinOp, a: usize, b: usize },
    Call { func: usize, args: Vec<usize> },
}

/// A compiled loop nest. `dims` is outermost-first; the last level is the
/// blocked one (absent only when the nest is empty — a scalar result).
struct Kernel<T> {
    dims: Vec<usize>,
    loads: Vec<LoadDesc>,
    outer: Vec<OutOp<T>>,
    inner: Vec<InOp>,
    /// Register holding the region's value: an `inner` register when the
    /// nest is non-empty, an `outer` register otherwise.
    root: usize,
}

/// Where a subexpression's value lives while the tape is being built.
#[derive(Clone, Copy)]
enum Src {
    Outer(usize),
    Inner(usize),
}

struct Builder<T> {
    /// Loop level of each index id, or `usize::MAX` if not a level here.
    level_of: Vec<usize>,
    n_levels: usize,
    /// Index id of the innermost (blocked) level, if the nest has one.
    inner_id: Option<IndexId>,
    loads: Vec<LoadDesc>,
    outer: Vec<OutOp<T>>,
    inner: Vec<InOp>,
}

impl<T: Elem> Builder<T> {
    fn new(levels: &[IndexId], n_ids: usize) -> Self {
        let mut level_of = vec![usize::MAX; n_ids];
        for (k, &id) in levels.iter().enumerate() {
            level_of[id as usize] = k;
        }
        Builder {
            level_of,
            n_levels: levels.len(),
            inner_id: levels.last().copied(),
            loads: Vec::new(),
            outer: Vec::new(),
            inner: Vec::new(),
        }
    }

    fn add_load(&mut self, buf: usize, indices: &[IndexId], strides: &[usize]) -> usize {
        let mut lv = vec![0usize; self.n_levels];
        for (p, &id) in indices.iter().enumerate() {
            let l = self.level_of[id as usize];
            debug_assert!(l != usize::MAX, "load index is not a loop level of this region");
            lv[l] += strides[p];
        }
        self.loads.push(LoadDesc { buf, strides: lv });
        self.loads.len() - 1
    }

    /// Does this subtree read the innermost loop index?
    fn varies(&self, e: &FExpr<T>) -> bool {
        match e {
            FExpr::Num(_) => false,
            FExpr::Load { indices, .. } => match self.inner_id {
                Some(id) => indices.contains(&id),
                None => false,
            },
            FExpr::Call { args, .. } => args.iter().any(|a| self.varies(a)),
            FExpr::Binary { lhs, rhs, .. } => self.varies(lhs) || self.varies(rhs),
        }
    }

    fn to_inner(&mut self, s: Src) -> usize {
        match s {
            Src::Inner(i) => i,
            Src::Outer(o) => {
                self.inner.push(InOp::Splat { outer: o });
                self.inner.len() - 1
            }
        }
    }

    /// Emit `e`, hoisting every maximal innermost-invariant subtree into
    /// the outer tape.
    fn build(&mut self, e: &FExpr<T>) -> Src {
        if !self.varies(e) {
            return Src::Outer(self.build_outer(e));
        }
        match e {
            FExpr::Load { buf, indices, strides } => {
                let load = self.add_load(*buf, indices, strides);
                self.inner.push(InOp::Gather { load });
                Src::Inner(self.inner.len() - 1)
            }
            FExpr::Call { func, args } => {
                let srcs: Vec<Src> = args.iter().map(|a| self.build(a)).collect();
                let args: Vec<usize> = srcs.into_iter().map(|s| self.to_inner(s)).collect();
                self.inner.push(InOp::Call { func: *func, args });
                Src::Inner(self.inner.len() - 1)
            }
            FExpr::Binary { op, lhs, rhs } => {
                let ls = self.build(lhs);
                let rs = self.build(rhs);
                let a = self.to_inner(ls);
                let b = self.to_inner(rs);
                self.inner.push(InOp::Bin { op: *op, a, b });
                Src::Inner(self.inner.len() - 1)
            }
            // `varies` is false for a literal, so it never reaches here.
            FExpr::Num(_) => unreachable!("a literal never varies"),
        }
    }

    fn build_outer(&mut self, e: &FExpr<T>) -> usize {
        let op = match e {
            FExpr::Num(v) => OutOp::Const(*v),
            FExpr::Load { buf, indices, strides } => {
                let load = self.add_load(*buf, indices, strides);
                OutOp::Load { load }
            }
            FExpr::Call { func, args } => {
                let args = args.iter().map(|a| self.build_outer(a)).collect();
                OutOp::Call { func: *func, args }
            }
            FExpr::Binary { op, lhs, rhs } => {
                let a = self.build_outer(lhs);
                let b = self.build_outer(rhs);
                OutOp::Bin { op: *op, a, b }
            }
        };
        self.outer.push(op);
        self.outer.len() - 1
    }

    fn finish(mut self, e: &FExpr<T>, dims: Vec<usize>) -> Kernel<T> {
        let root = self.build(e);
        let root = if self.inner_id.is_some() {
            self.to_inner(root)
        } else {
            match root {
                Src::Outer(o) => o,
                Src::Inner(_) => unreachable!("an empty nest has no innermost level"),
            }
        };
        Kernel { dims, loads: self.loads, outer: self.outer, inner: self.inner, root }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Execution
// ─────────────────────────────────────────────────────────────────────────

#[inline]
fn bin_apply<T: Elem>(op: BinOp, a: T, b: T) -> T {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

/// Row-major odometer over levels `lo..hi` of `pos`; `false` on wrap.
fn bump(pos: &mut [usize], dims: &[usize], lo: usize, hi: usize) -> bool {
    for k in (lo..hi).rev() {
        pos[k] += 1;
        if pos[k] < dims[k] {
            return true;
        }
        pos[k] = 0;
    }
    false
}

/// Scratch reused across every region of one run.
struct Regs<T> {
    outer: Vec<T>,
    /// `inner.len()` blocks of [`BLOCK`] elements, laid out back to back.
    blocks: Vec<T>,
    load_base: Vec<usize>,
    pos: Vec<usize>,
    args: Vec<T>,
}

impl<T: Elem> Regs<T> {
    fn new() -> Self {
        Regs {
            outer: Vec::new(),
            blocks: Vec::new(),
            load_base: Vec::new(),
            pos: Vec::new(),
            args: Vec::new(),
        }
    }

    fn fit(&mut self, k: &Kernel<T>) {
        self.outer.clear();
        self.outer.resize(k.outer.len(), T::ZERO);
        self.blocks.clear();
        self.blocks.resize(k.inner.len() * BLOCK, T::ZERO);
        self.load_base.clear();
        self.load_base.resize(k.loads.len(), 0);
        self.pos.clear();
        self.pos.resize(k.dims.len(), 0);
    }
}

/// Resolve each load's offset for the current outer position, then run the
/// scalar tape.
fn run_outer<T: Elem>(k: &Kernel<T>, bufs: &[&[T]], reg: &Registry<T>, r: &mut Regs<T>) {
    let n_outer = k.dims.len().saturating_sub(1);
    for (i, ld) in k.loads.iter().enumerate() {
        let mut off = 0usize;
        for l in 0..n_outer {
            off += r.pos[l] * ld.strides[l];
        }
        r.load_base[i] = off;
    }
    for i in 0..k.outer.len() {
        let v = match &k.outer[i] {
            OutOp::Const(v) => *v,
            OutOp::Load { load } => bufs[k.loads[*load].buf][r.load_base[*load]],
            OutOp::Bin { op, a, b } => bin_apply(*op, r.outer[*a], r.outer[*b]),
            OutOp::Call { func, args } => {
                r.args.clear();
                for &a in args {
                    let v = r.outer[a];
                    r.args.push(v);
                }
                (reg.fns[*func].eval)(&r.args)
            }
        };
        r.outer[i] = v;
    }
}

/// Run the block tape over `start..start+len` of the innermost level.
fn run_inner<T: Elem>(
    k: &Kernel<T>,
    bufs: &[&[T]],
    reg: &Registry<T>,
    r: &mut Regs<T>,
    start: usize,
    len: usize,
) {
    let inner_lvl = k.dims.len() - 1;
    for (i, op) in k.inner.iter().enumerate() {
        // Registers are SSA and numbered in emission order, so every
        // operand lives strictly before `i` — `prev` and `dst` are disjoint.
        let (prev, rest) = r.blocks.split_at_mut(i * BLOCK);
        let dst = &mut rest[..len];
        match op {
            InOp::Splat { outer } => dst.fill(r.outer[*outer]),
            InOp::Gather { load } => {
                let ld = &k.loads[*load];
                let step = ld.strides[inner_lvl];
                let base = r.load_base[*load] + start * step;
                let src = bufs[ld.buf];
                if step == 1 {
                    dst.copy_from_slice(&src[base..base + len]);
                } else {
                    for (j, d) in dst.iter_mut().enumerate() {
                        *d = src[base + j * step];
                    }
                }
            }
            InOp::Bin { op, a, b } => {
                let x = &prev[a * BLOCK..a * BLOCK + len];
                let y = &prev[b * BLOCK..b * BLOCK + len];
                match op {
                    BinOp::Add => {
                        for (d, (p, q)) in dst.iter_mut().zip(x.iter().zip(y)) {
                            *d = *p + *q;
                        }
                    }
                    BinOp::Sub => {
                        for (d, (p, q)) in dst.iter_mut().zip(x.iter().zip(y)) {
                            *d = *p - *q;
                        }
                    }
                    BinOp::Mul => {
                        for (d, (p, q)) in dst.iter_mut().zip(x.iter().zip(y)) {
                            *d = *p * *q;
                        }
                    }
                    BinOp::Div => {
                        for (d, (p, q)) in dst.iter_mut().zip(x.iter().zip(y)) {
                            *d = *p / *q;
                        }
                    }
                }
            }
            InOp::Call { func, args } => {
                let f = reg.fns[*func].eval;
                r.args.clear();
                r.args.resize(args.len(), T::ZERO);
                for (j, d) in dst.iter_mut().enumerate() {
                    for (s, &a) in r.args.iter_mut().zip(args) {
                        *s = prev[a * BLOCK + j];
                    }
                    *d = f(&r.args);
                }
            }
        }
    }
}

/// Evaluate a whole statement RHS into a fresh row-major buffer.
fn exec_map<T: Elem>(
    k: &Kernel<T>,
    bufs: &[&[T]],
    reg: &Registry<T>,
    r: &mut Regs<T>,
    out: &mut [T],
) {
    r.fit(k);
    let n = k.dims.len();
    if n == 0 {
        run_outer(k, bufs, reg, r);
        out[0] = r.outer[k.root];
        return;
    }
    if k.dims.contains(&0) {
        return; // no elements
    }
    let inner_dim = k.dims[n - 1];
    let root = k.root * BLOCK;
    let mut out_base = 0usize;
    loop {
        run_outer(k, bufs, reg, r);
        let mut start = 0usize;
        while start < inner_dim {
            let len = BLOCK.min(inner_dim - start);
            run_inner(k, bufs, reg, r, start, len);
            out[out_base + start..out_base + start + len]
                .copy_from_slice(&r.blocks[root..root + len]);
            start += len;
        }
        out_base += inner_dim;
        if !bump(&mut r.pos, &k.dims, 0, n - 1) {
            break;
        }
    }
}

/// Fold a reduction. Levels `0..n_free` are the free indices (one
/// accumulator each, in ascending-id row-major order); the rest are the
/// bound indices in binder order, innermost last.
#[allow(clippy::too_many_arguments)]
fn exec_reduce<T: Elem>(
    k: &Kernel<T>,
    bufs: &[&[T]],
    reg: &Registry<T>,
    r: &mut Regs<T>,
    n_free: usize,
    identity: T,
    fold: fn(T, T) -> T,
    out: &mut [T],
) {
    r.fit(k);
    let n = k.dims.len();
    debug_assert!(n > n_free, "a binder always binds at least one index");
    if k.dims[..n_free].contains(&0) {
        return; // no output elements
    }
    let inner_dim = k.dims[n - 1];
    let bound_empty = k.dims[n_free..].contains(&0);
    let root = k.root * BLOCK;
    let mut o = 0usize;
    loop {
        let mut acc = identity;
        if !bound_empty {
            r.pos[n_free..].fill(0);
            loop {
                run_outer(k, bufs, reg, r);
                let mut start = 0usize;
                while start < inner_dim {
                    let len = BLOCK.min(inner_dim - start);
                    run_inner(k, bufs, reg, r, start, len);
                    for v in &r.blocks[root..root + len] {
                        acc = fold(acc, *v);
                    }
                    start += len;
                }
                if !bump(&mut r.pos, &k.dims, n_free, n - 1) {
                    break;
                }
            }
        }
        out[o] = acc;
        o += 1;
        if !bump(&mut r.pos, &k.dims, 0, n_free) {
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────────

/// How much of a run the kernel path handled — so a differential test can
/// tell "agrees with the oracle" from "silently fell back to it".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Statements in the program.
    pub statements: usize,
    /// Statements executed by the kernel path. Either 0 or `statements`:
    /// the gate is input storage, a property of the whole run.
    pub kernel_statements: usize,
    /// Reduction nodes lowered to the einsum JIT.
    pub jit_reductions: usize,
    /// Reduction nodes run by the blocked tape kernel.
    pub tape_reductions: usize,
}

impl RunReport {
    /// Did the kernel path execute the whole program?
    pub fn all_kernel(&self) -> bool {
        self.kernel_statements == self.statements
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Driver
// ─────────────────────────────────────────────────────────────────────────

struct Fast<'a, T: Elem> {
    reg: &'a Registry<T>,
    extents: &'a [usize],
    n_ids: usize,
    bufs: Vec<Buf<'a, T>>,
    shapes: Vec<Vec<usize>>,
    regs: Regs<T>,
    report: RunReport,
}

impl<'a, T: Elem> Fast<'a, T> {
    fn slices(&self) -> Vec<&[T]> {
        self.bufs.iter().map(|b| b.as_slice()).collect()
    }

    fn push(&mut self, data: Vec<T>, shape: Vec<usize>) -> usize {
        self.bufs.push(Buf::Owned(data));
        self.shapes.push(shape);
        self.bufs.len() - 1
    }

    /// Lower one RHS, materializing each reduction node (innermost first,
    /// left to right) into its own buffer — the same order and the same
    /// arity as `eval::lower`, which is what keeps a registered function's
    /// call count identical between the two paths.
    fn lower(&mut self, e: &CExpr, name_buf: &HashMap<String, usize>) -> FExpr<T> {
        match e {
            CExpr::Num(v) => FExpr::Num((self.reg.lit)(*v)),
            CExpr::ScalarRef(name) => FExpr::Load {
                buf: name_buf[name.as_str()],
                indices: Vec::new(),
                strides: Vec::new(),
            },
            CExpr::Load { tensor, indices } => {
                let buf = name_buf[tensor.as_str()];
                FExpr::Load {
                    buf,
                    indices: indices.clone(),
                    strides: strides_of(&self.shapes[buf]),
                }
            }
            CExpr::Call { func, args } => FExpr::Call {
                func: *func,
                args: args.iter().map(|a| self.lower(a, name_buf)).collect(),
            },
            CExpr::Binary { op, lhs, rhs } => FExpr::Binary {
                op: *op,
                lhs: Box::new(self.lower(lhs, name_buf)),
                rhs: Box::new(self.lower(rhs, name_buf)),
            },
            CExpr::Reduce { op, indices, body } => {
                let body = self.lower(body, name_buf);
                let mut free = Vec::new();
                body.collect_ids(&mut free);
                free.retain(|id| !indices.contains(id));
                free.sort_unstable();
                free.dedup();
                let buf = self.reduce(*op, indices, &free, &body);
                let strides = strides_of(&self.shapes[buf]);
                FExpr::Load { buf, indices: free, strides }
            }
        }
    }

    /// Materialize one reduction node into a fresh buffer.
    fn reduce(&mut self, op: usize, bound: &[IndexId], free: &[IndexId], body: &FExpr<T>) -> usize {
        let shape: Vec<usize> = free.iter().map(|&i| self.extents[i as usize]).collect();

        #[cfg(feature = "jit")]
        if let Some(data) = self.try_jit(op, bound, free, body, &shape) {
            self.report.jit_reductions += 1;
            return self.push(data, shape);
        }

        let levels: Vec<IndexId> = free.iter().chain(bound).copied().collect();
        let dims: Vec<usize> = levels.iter().map(|&i| self.extents[i as usize]).collect();
        let kernel = Builder::new(&levels, self.n_ids).finish(body, dims);

        let rdef = &self.reg.reduces[op];
        let (identity, fold) = (rdef.identity, rdef.fold);
        let mut out = vec![identity; elem_count(&shape)];
        let slices: Vec<&[T]> = self.bufs.iter().map(|b| b.as_slice()).collect();
        exec_reduce(
            &kernel,
            &slices,
            self.reg,
            &mut self.regs,
            free.len(),
            identity,
            fold,
            &mut out,
        );
        drop(slices);
        self.report.tape_reductions += 1;
        self.push(out, shape)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// JIT lowering
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "jit")]
mod jit_route {
    use super::{Elem, FExpr, IndexId};
    use crate::dense::Dense;
    use crate::einsum::{Combine, Reduce};
    use crate::jit::{EinsumF32Jit, JitInput};
    use std::any::{Any, TypeId};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Reinterpret `s` as `&[f32]` — `None` unless `T` really is `f32`.
    pub(super) fn as_f32_slice<T: 'static>(s: &[T]) -> Option<&[f32]> {
        if TypeId::of::<T>() != TypeId::of::<f32>() {
            return None;
        }
        // SAFETY: the TypeId check above proves `T == f32`, so `[T]` and
        // `[f32]` have identical layout, alignment and validity, and the
        // returned slice borrows exactly the input's memory.
        Some(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len()) })
    }

    /// Move an `f32` buffer back into the program's element type. Safe —
    /// the downcast is `TypeId`-checked by `Box<dyn Any>`.
    pub(super) fn from_f32_vec<T: 'static>(v: Vec<f32>) -> Vec<T> {
        *(Box::new(v) as Box<dyn Any>).downcast::<Vec<T>>().expect("checked: T is f32")
    }

    /// Everything the generated code bakes in.
    #[derive(PartialEq, Eq)]
    pub(super) struct Key {
        pub spec: String,
        pub in_shapes: Vec<Vec<usize>>,
        pub out_shape: Vec<usize>,
        pub reduce: Reduce,
        pub combine: Combine,
    }

    thread_local! {
        /// Compiled kernels. A `None` payload records a pattern the
        /// backend refused, so we never re-pay ~400 µs of Cranelift to be
        /// told "unsupported" a second time.
        static CACHE: RefCell<Vec<(Key, Option<Rc<EinsumF32Jit>>)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Bound on cached kernels: JIT pages stay mapped as long as they are
    /// held, and the language targets a handful of hot programs.
    const CACHE_CAP: usize = 256;

    pub(super) fn compiled(
        key: Key,
        inputs: &[JitInput],
        out_shape: &[usize],
    ) -> Option<Rc<EinsumF32Jit>> {
        CACHE.with(|c| {
            if let Some((_, hit)) = c.borrow().iter().find(|(k, _)| *k == key) {
                return hit.clone();
            }
            let built = EinsumF32Jit::compile_reduce(
                &key.spec,
                key.reduce,
                key.combine,
                inputs,
                &[out_shape.to_vec()],
            )
            .ok()
            .map(Rc::new);
            let mut c = c.borrow_mut();
            if c.len() >= CACHE_CAP {
                let half = c.len() / 2;
                c.drain(..half);
            }
            c.push((key, built.clone()));
            built
        })
    }

    pub(super) fn run_into(
        jit: &EinsumF32Jit,
        inputs: &[JitInput],
        out_shape: &[usize],
        identity: f32,
    ) -> Vec<f32> {
        let mut out = Dense::<f32>::zeros(out_shape.to_vec());
        out.data.fill(identity);
        jit.run(inputs, &mut [&mut out]);
        out.data
    }

    /// A left-leaning chain `((l₀ ⊗ l₁) ⊗ l₂) ⊗ …` of tensor loads under a
    /// single `*` or `+`, or a bare load. Returns the leaves in the order
    /// the backend combines them (`emit_contribution` folds left to right
    /// in input order, so any other association would change the bits).
    ///
    /// Every leaf must carry at least one index (the spec parser rejects an
    /// empty input pattern) and no index twice (a diagonal read has no
    /// einsum-input form here).
    pub(super) fn flatten_chain<T: Elem>(
        e: &FExpr<T>,
    ) -> Option<(Combine, Vec<(&[IndexId], usize)>)> {
        fn leaf<T: Elem>(e: &FExpr<T>) -> Option<(&[IndexId], usize)> {
            let FExpr::Load { buf, indices, .. } = e else { return None };
            if indices.is_empty() {
                return None;
            }
            for (k, id) in indices.iter().enumerate() {
                if indices[..k].contains(id) {
                    return None;
                }
            }
            Some((indices.as_slice(), *buf))
        }

        fn walk<'e, T: Elem>(
            e: &'e FExpr<T>,
            want: super::BinOp,
            out: &mut Vec<(&'e [IndexId], usize)>,
        ) -> bool {
            if let FExpr::Binary { op, lhs, rhs } = e
                && *op == want
            {
                return walk(lhs, want, out)
                    && match leaf(rhs) {
                        Some(l) => {
                            out.push(l);
                            true
                        }
                        None => false,
                    };
            }
            match leaf(e) {
                Some(l) => {
                    out.push(l);
                    true
                }
                None => false,
            }
        }

        match e {
            // Single operand: the combine never fires, so its choice is
            // immaterial.
            FExpr::Load { .. } => Some((Combine::Mul, vec![leaf(e)?])),
            FExpr::Binary { op, .. } => {
                let combine = match op {
                    super::BinOp::Mul => Combine::Mul,
                    super::BinOp::Add => Combine::Add,
                    _ => return None,
                };
                let mut leaves = Vec::new();
                walk(e, *op, &mut leaves).then_some((combine, leaves))
            }
            _ => None,
        }
    }
}

#[cfg(feature = "jit")]
impl<T: Elem> Fast<'_, T> {
    /// Recognise `reduce(bound: load ⊗ load ⊗ …)` as a classic einsum
    /// contraction and run it on [`crate::jit::EinsumF32Jit`].
    ///
    /// Every condition below exists to keep the result bit-identical to the
    /// tree-walker (see the module docs). Returns `None` — silently, having
    /// computed nothing — whenever one does not hold; the blocked tape
    /// kernel then walks the same nest itself.
    fn try_jit(
        &self,
        op: usize,
        bound: &[IndexId],
        free: &[IndexId],
        body: &FExpr<T>,
        out_shape: &[usize],
    ) -> Option<Vec<T>> {
        use crate::einsum::Reduce;
        use crate::jit::JitInput;

        // ── f32 only: that is the backend's element type. ──
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            return None;
        }

        // ── The fold must be a built-in `sum`/`prod` (identity included).
        // `max`/`min` map to Cranelift fmax/fmin, which disagree with the
        // language's comparison fold on ±0.0 and NaN — those stay on the
        // tape, where the fold function itself is what runs. ──
        let reduce = self.reg.builtin_reduce(op)?;
        if !matches!(reduce, Reduce::Sum | Reduce::Prod) {
            return None;
        }

        // ── Body must be a left-leaning chain of tensor loads. ──
        let (combine, leaves) = jit_route::flatten_chain(body)?;

        // ── Assign a spec letter per index, in first-appearance order. ──
        let mut slot_of: HashMap<IndexId, u8> = HashMap::new();
        let mut appearance: Vec<IndexId> = Vec::new();
        for (indices, _) in &leaves {
            for &id in *indices {
                if !slot_of.contains_key(&id) {
                    let n = slot_of.len();
                    if n >= 26 {
                        return None;
                    }
                    slot_of.insert(id, n as u8);
                    appearance.push(id);
                }
            }
        }
        // Every bound index must be carried by some input, and the
        // backend's contraction order — first appearance across the input
        // patterns — must be exactly the binder order, or the sum would
        // associate differently.
        let contracted: Vec<IndexId> =
            appearance.iter().copied().filter(|id| bound.contains(id)).collect();
        if contracted != bound {
            return None;
        }
        // A zero extent would hand the backend an empty buffer for no gain.
        if out_shape.contains(&0) || appearance.iter().any(|&id| self.extents[id as usize] == 0) {
            return None;
        }

        let letters = |ids: &[IndexId]| -> String {
            ids.iter().map(|id| (b'a' + slot_of[id]) as char).collect()
        };
        let spec = format!(
            "{}->{}",
            leaves.iter().map(|(ix, _)| letters(ix)).collect::<Vec<_>>().join(","),
            letters(free)
        );

        let bufs = self.slices();
        let mut inputs: Vec<JitInput> = Vec::with_capacity(leaves.len());
        let mut in_shapes: Vec<Vec<usize>> = Vec::with_capacity(leaves.len());
        for (indices, buf) in &leaves {
            let data = jit_route::as_f32_slice(bufs[*buf])?;
            let shape = self.shapes[*buf].clone();
            debug_assert_eq!(shape.len(), indices.len());
            in_shapes.push(shape.clone());
            inputs.push(JitInput::DenseSlice { data, shape });
        }

        let key =
            jit_route::Key { spec, in_shapes, out_shape: out_shape.to_vec(), reduce, combine };
        let jit = jit_route::compiled(key, &inputs, out_shape)?;
        let identity = if reduce == Reduce::Sum { 0.0f32 } else { 1.0f32 };
        Some(jit_route::from_f32_vec(jit_route::run_into(
            &jit, &inputs, out_shape, identity,
        )))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────

/// Try to execute `checked` on the kernel path.
///
/// Returns `None` — having written nothing — when the fast path declines,
/// which today means "some input has no contiguous row-major image"
/// (CSR, blocked, or any other exotic [`NDIndex`] storage). The caller then
/// runs the tree-walking evaluator.
///
/// `extents` must already have been resolved by the shared binding phase,
/// so this function performs no validation: everything it could reject has
/// been rejected before it is called.
pub(super) fn try_run<T: Elem>(
    checked: &Checked,
    reg: &Registry<T>,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
    extents: &[usize],
) -> Option<RunReport> {
    let mut bufs: Vec<Buf<T>> = Vec::with_capacity(inputs.len());
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(inputs.len());
    let mut name_buf: HashMap<String, usize> = HashMap::new();
    for (i, (name, t)) in inputs.iter().enumerate() {
        bufs.push(Buf::Slice(t.as_flat_slice()?));
        shapes.push((0..t.ndim()).map(|a| t.dim(a)).collect());
        name_buf.insert((*name).to_string(), i);
    }

    let n_ids = checked.index_names.len();
    let mut f = Fast {
        reg,
        extents,
        n_ids,
        bufs,
        shapes,
        regs: Regs::new(),
        report: RunReport {
            statements: checked.stmts.len(),
            kernel_statements: checked.stmts.len(),
            ..RunReport::default()
        },
    };

    for stmt in &checked.stmts {
        let rhs = f.lower(&stmt.rhs, &name_buf);
        let shape: Vec<usize> = stmt.lhs.iter().map(|&i| extents[i as usize]).collect();
        // `y[…] = <one reduction over exactly the LHS indices>` needs no
        // copy: the reduction already wrote that buffer, in that layout.
        let buf = match &rhs {
            FExpr::Load { buf, indices, .. } if *indices == stmt.lhs => *buf,
            _ => {
                let kernel = Builder::new(&stmt.lhs, n_ids).finish(&rhs, shape.clone());
                let mut out = vec![T::ZERO; elem_count(&shape)];
                let slices: Vec<&[T]> = f.bufs.iter().map(|b| b.as_slice()).collect();
                exec_map(&kernel, &slices, reg, &mut f.regs, &mut out);
                drop(slices);
                f.push(out, shape)
            }
        };
        name_buf.insert(stmt.name.clone(), buf);
    }

    for (name, dst) in outputs.iter_mut() {
        let buf = name_buf[*name];
        let data = f.bufs[buf].as_slice();
        // A dense destination is the same row-major image we just built,
        // so the whole write-back is one memcpy instead of one vtable
        // `set` per element — which at 64 Ki elements is most of the cost
        // of an elementwise statement.
        if let Some(out) = dst.as_flat_slice_mut()
            && out.len() == data.len()
        {
            out.copy_from_slice(data);
            continue;
        }
        let shape = f.shapes[buf].clone();
        let mut k = 0usize;
        for_each_index(&shape, |ix| {
            dst.set(ix, data[k]);
            k += 1;
        });
    }
    Some(f.report)
}
