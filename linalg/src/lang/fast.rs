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
//! the size of this hole: it reports how many output elements differed in
//! NaN payload alone — 0 of 72 242 on the current generator.)
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
//! 4. **Load and store maps** (RESUME.md step 4). Built-in scalar
//!    functions are recognised by a [`FnKind`] tag the [`Registry`]
//!    records — never by name and never by function pointer — and become
//!    inline block code instead of an indirect `fn(&[T]) -> T` call per
//!    element; that call was most of naive softmax's cost, and it also
//!    kept the surrounding loop from vectorizing. A unary chain *wrapping*
//!    a reduction is applied at the flush (`sqrt(sum(l: M[l,j]))` writes
//!    the finished value straight out, with no second pass), and both ends
//!    reach the JIT — per-input maps at load, the chain at store — for the
//!    four unaries Cranelift emits exactly ([`crate::jit::UnaryMap`]).
//!    Every inline form is required to be bit-identical to the
//!    [`super::FnDef::eval`] it replaces; a caller-registered function is
//!    never inlined, because nothing here can know its bits.
//! 5. **Dynamic axes** (RESUME.md step 6). By default every extent is a
//!    constant in the generated code, so a compiled kernel is keyed by —
//!    and only valid for — the exact shapes it saw. A caller whose shapes
//!    move (a KV cache growing one position per token) then pays a fresh
//!    Cranelift compile per shape. [`super::RunOptions::dynamic`] names
//!    index axes whose extent should instead be passed to the kernel as a
//!    **runtime argument**; those axes drop out of the cache key
//!    ([`jit_route::Key`]), so one kernel serves every extent of them
//!    while every other extent stays baked. Nothing about the plan, the
//!    loop nest, the iteration order or the arithmetic changes — the
//!    marking decides what is a constant and nothing else, which is why a
//!    dynamic axis is bit-identical to the same axis static.

use std::collections::HashMap;

use super::ast::BinOp;
use super::check::{CExpr, Checked, IndexId};
use super::eval::for_each_index;
use super::{Elem, FnKind, Registry};
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
    /// A built-in unary applied over the whole block by inline code —
    /// see [`unary_block`]. `func` is kept only as the fallback for an
    /// element type the inline form does not cover.
    Un { kind: FnKind, func: usize, a: usize },
    /// Built-in elementwise `max`/`min` over two blocks.
    MaxMin { is_max: bool, a: usize, b: usize },
    /// Anything else: an indirect `fn(&[T]) -> T` call per element.
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
    /// Block-tape calls compiled to inline code vs left indirect —
    /// diagnostics only, surfaced through [`RunReport`].
    n_inline: usize,
    n_indirect: usize,
}

/// Where a subexpression's value lives while the tape is being built.
#[derive(Clone, Copy)]
enum Src {
    Outer(usize),
    Inner(usize),
}

struct Builder<'r, T> {
    /// Loop level of each index id, or `usize::MAX` if not a level here.
    level_of: Vec<usize>,
    n_levels: usize,
    /// Index id of the innermost (blocked) level, if the nest has one.
    inner_id: Option<IndexId>,
    /// The registry's built-in tags, parallel to its function table.
    kinds: &'r [Option<FnKind>],
    loads: Vec<LoadDesc>,
    outer: Vec<OutOp<T>>,
    inner: Vec<InOp>,
    n_inline: usize,
    n_indirect: usize,
}

impl<'r, T: Elem> Builder<'r, T> {
    fn new(levels: &[IndexId], n_ids: usize, kinds: &'r [Option<FnKind>]) -> Self {
        let mut level_of = vec![usize::MAX; n_ids];
        for (k, &id) in levels.iter().enumerate() {
            level_of[id as usize] = k;
        }
        Builder {
            level_of,
            n_levels: levels.len(),
            inner_id: levels.last().copied(),
            kinds,
            loads: Vec::new(),
            outer: Vec::new(),
            inner: Vec::new(),
            n_inline: 0,
            n_indirect: 0,
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
                // A *built-in* becomes inline block code — one indirect
                // `fn(&[T]) -> T` call per element is what softmax's `exp`
                // was spending most of its time on, and it also blocks
                // autovectorization of everything around it. Anything the
                // caller registered stays on the portable call path.
                let op = match self.kinds[*func] {
                    Some(FnKind::Max2) => InOp::MaxMin { is_max: true, a: args[0], b: args[1] },
                    Some(FnKind::Min2) => InOp::MaxMin { is_max: false, a: args[0], b: args[1] },
                    Some(kind) => InOp::Un { kind, func: *func, a: args[0] },
                    None => InOp::Call { func: *func, args },
                };
                if matches!(op, InOp::Call { .. }) {
                    self.n_indirect += 1;
                } else {
                    self.n_inline += 1;
                }
                self.inner.push(op);
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
        Kernel {
            dims,
            loads: self.loads,
            outer: self.outer,
            inner: self.inner,
            root,
            n_inline: self.n_inline,
            n_indirect: self.n_indirect,
        }
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

/// Reinterpret a block as `U`, if `T` really is `U`.
///
/// The same device as `jit_route::as_f32_slice`, for the same reason: the
/// tape is generic over the element type but the transcendental built-ins
/// only exist for the concrete float types, and a generic `fn(&[T]) -> T`
/// call is exactly what we are trying to get rid of.
#[inline]
fn cast_slice<T: 'static, U: 'static>(s: &[T]) -> Option<&[U]> {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<U>() {
        return None;
    }
    // SAFETY: `T == U` by the `TypeId` check, so the layouts, alignments
    // and validity invariants are the same and the result borrows exactly
    // the input's memory for the input's lifetime.
    Some(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const U, s.len()) })
}

#[inline]
fn cast_slice_mut<T: 'static, U: 'static>(s: &mut [T]) -> Option<&mut [U]> {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<U>() {
        return None;
    }
    // SAFETY: as `cast_slice`, and the exclusive borrow is preserved.
    Some(unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut U, s.len()) })
}

/// Apply a float-only built-in over a block, monomorphized to the concrete
/// element type so the call is direct (and, for `sqrt`/`abs`, an
/// instruction the loop can be vectorized around).
///
/// `g32`/`g64` are generic parameters, not `fn` pointers, so each
/// instantiation inlines one concrete function — using `fn(f32) -> f32`
/// here would just trade one indirect call for another. `f` is the
/// registry's own evaluator, used for element types that have no inline
/// form (none today: these kinds are registered for `f32`/`f64` only).
#[inline]
fn float_block<T: Elem>(
    x: &[T],
    dst: &mut [T],
    f: fn(&[T]) -> T,
    g32: impl Fn(f32) -> f32,
    g64: impl Fn(f64) -> f64,
) {
    if let Some(src) = cast_slice::<T, f32>(x) {
        let out = cast_slice_mut::<T, f32>(dst).expect("T is f32");
        for (d, s) in out.iter_mut().zip(src) {
            *d = g32(*s);
        }
    } else if let Some(src) = cast_slice::<T, f64>(x) {
        let out = cast_slice_mut::<T, f64>(dst).expect("T is f64");
        for (d, s) in out.iter_mut().zip(src) {
            *d = g64(*s);
        }
    } else {
        for (d, s) in dst.iter_mut().zip(x) {
            *d = f(std::slice::from_ref(s));
        }
    }
}

/// Apply one built-in unary over a block.
///
/// **Every arm must be bit-identical to the [`super::FnDef::eval`] its
/// [`FnKind`] tags** — that is the whole contract of the tag. `Neg` and
/// `Relu` are written generically in exactly the form `super::fn_neg` /
/// the `relu` registration use (`T::ZERO` *is* `+0.0` for both float
/// types, so `relu(-0.0) == +0.0` and `relu(NaN) == +0.0` here too); the
/// rest delegate to the same `f32`/`f64` methods the registry entries call.
fn unary_block<T: Elem>(kind: FnKind, f: fn(&[T]) -> T, x: &[T], dst: &mut [T]) {
    match kind {
        FnKind::Neg => {
            for (d, s) in dst.iter_mut().zip(x) {
                *d = -*s;
            }
        }
        FnKind::Relu => {
            for (d, s) in dst.iter_mut().zip(x) {
                *d = if *s > T::ZERO { *s } else { T::ZERO };
            }
        }
        FnKind::Exp => float_block(x, dst, f, f32::exp, f64::exp),
        FnKind::Ln => float_block(x, dst, f, f32::ln, f64::ln),
        FnKind::Sqrt => float_block(x, dst, f, f32::sqrt, f64::sqrt),
        FnKind::Abs => float_block(x, dst, f, f32::abs, f64::abs),
        FnKind::Tanh => float_block(x, dst, f, f32::tanh, f64::tanh),
        // Arity 2: the builder emits `InOp::MaxMin` for these instead.
        FnKind::Max2 | FnKind::Min2 => unreachable!("arity-2 built-in is not a unary block op"),
    }
}

/// Elementwise `max`/`min` over two blocks, in the comparison form the
/// registry's `fn_max2`/`fn_min2` use — *not* `f32::max`, which differs on
/// `±0.0` and NaN.
fn maxmin_block<T: Elem>(is_max: bool, x: &[T], y: &[T], dst: &mut [T]) {
    if is_max {
        for (d, (a, b)) in dst.iter_mut().zip(x.iter().zip(y)) {
            *d = if *b > *a { *b } else { *a };
        }
    } else {
        for (d, (a, b)) in dst.iter_mut().zip(x.iter().zip(y)) {
            *d = if *b < *a { *b } else { *a };
        }
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
            InOp::Un { kind, func, a } => {
                unary_block(*kind, reg.fns[*func].eval, &prev[a * BLOCK..a * BLOCK + len], dst);
            }
            InOp::MaxMin { is_max, a, b } => maxmin_block(
                *is_max,
                &prev[a * BLOCK..a * BLOCK + len],
                &prev[b * BLOCK..b * BLOCK + len],
                dst,
            ),
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

/// Apply a **store map** — a chain of arity-1 registry functions, left to
/// right — over a reduction's finished output buffer, in place.
///
/// Exactly one application per output element, on the same value the
/// unfused lowering would have mapped, and in the same order relative to
/// the fold (the oracle also completes the whole reduction before its
/// enclosing expression maps it). What is saved is the intermediate buffer
/// and the extra kernel: this walks the result once, in blocks, so a
/// built-in gets the same inline, vectorizable code it would have got from
/// the map kernel.
fn apply_store_maps<T: Elem>(
    reg: &Registry<T>,
    maps: &[usize],
    buf: &mut [T],
    scratch: &mut Vec<T>,
) {
    for &f in maps {
        let eval = reg.fns[f].eval;
        match reg.builtin_fn(f) {
            Some(kind) if kind.arity() == 1 => {
                for chunk in buf.chunks_mut(BLOCK) {
                    scratch.clear();
                    scratch.extend_from_slice(chunk);
                    unary_block(kind, eval, scratch, chunk);
                }
            }
            _ => {
                for v in buf.iter_mut() {
                    *v = eval(std::slice::from_ref(v));
                }
            }
        }
    }
}

/// Fold a reduction. Levels `0..n_free` are the free indices (one
/// accumulator each, in ascending-id row-major order); the rest are the
/// bound indices in binder order, innermost last.
///
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
    /// Reduction nodes whose enclosing unary chain became a **store map**:
    /// applied to the reduction's own output buffer rather than turned
    /// into a separate temporary plus a separate map kernel.
    pub store_maps: usize,
    /// Input loads carrying a unary chain fused into a JIT'd contraction
    /// (a **load map**), counted per mapped leaf.
    pub jit_load_maps: usize,
    /// Store maps handed to the JIT rather than applied on the tape.
    pub jit_store_maps: usize,
    /// Built-in scalar functions compiled to inline block code on the
    /// tape, and calls left as an indirect `fn(&[T]) -> T` per element
    /// (a caller-registered function, always).
    pub inline_fn_ops: usize,
    pub indirect_fn_ops: usize,
    /// Index axes actually marked **dynamic** for this run: the entries of
    /// [`RunOptions::dynamic`](super::RunOptions::dynamic) that named an
    /// index this program has. A name that matches nothing is ignored, so
    /// comparing this against what you asked for is how to catch a typo.
    pub dynamic_axes: usize,
    /// Reductions lowered to the JIT with at least one axis compiled as a
    /// runtime argument — i.e. reductions whose compiled kernel is shared
    /// across every extent of that axis instead of one kernel per shape.
    pub jit_dynamic_reductions: usize,
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
    /// Per index id: is this axis a runtime argument to generated code?
    /// Affects *only* what the JIT bakes in and how kernels are keyed —
    /// never an extent, an iteration order or an arithmetic operation.
    /// **Empty means "none"**, so the default all-static path allocates
    /// nothing; read it through [`Fast::is_dynamic`], never by index.
    dynamic: &'a [bool],
    n_ids: usize,
    bufs: Vec<Buf<'a, T>>,
    shapes: Vec<Vec<usize>>,
    regs: Regs<T>,
    report: RunReport,
}

impl<'a, T: Elem> Fast<'a, T> {
    /// Is index `id`'s extent a runtime argument? See [`Fast::dynamic`].
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    fn is_dynamic(&self, id: IndexId) -> bool {
        self.dynamic.get(id as usize).copied().unwrap_or(false)
    }

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
            CExpr::Call { .. } => {
                // A unary chain wrapping a reduction is a **store map**:
                // `sqrt(sum(l: M[l,j]))` applies `sqrt` once per element of
                // the reduction's own result, so it belongs to the
                // reduction, not to the expression around it. Same
                // function, same arguments, same number of calls, same
                // order — the temporary and the extra kernel disappear
                // (and on the JIT it is emitted before the single store).
                if let CExpr::Reduce { op, indices, body } = unary_chain_target(e) {
                    let body = self.lower(body, name_buf);
                    let free = free_ids(&body, indices);
                    // The chain is peeled outermost-first; applying it to
                    // the accumulator runs innermost-first.
                    let mut maps = unary_chain_funcs(e);
                    maps.reverse();
                    let buf = self.reduce(*op, indices, &free, &body, &maps);
                    self.report.store_maps += 1;
                    let strides = strides_of(&self.shapes[buf]);
                    return FExpr::Load { buf, indices: free, strides };
                }
                let CExpr::Call { func, args } = e else { unreachable!("matched above") };
                FExpr::Call {
                    func: *func,
                    args: args.iter().map(|a| self.lower(a, name_buf)).collect(),
                }
            }
            CExpr::Binary { op, lhs, rhs } => FExpr::Binary {
                op: *op,
                lhs: Box::new(self.lower(lhs, name_buf)),
                rhs: Box::new(self.lower(rhs, name_buf)),
            },
            CExpr::Reduce { op, indices, body } => {
                let body = self.lower(body, name_buf);
                let free = free_ids(&body, indices);
                let buf = self.reduce(*op, indices, &free, &body, &[]);
                let strides = strides_of(&self.shapes[buf]);
                FExpr::Load { buf, indices: free, strides }
            }
        }
    }

    /// Roll a compiled kernel's inline/indirect call counts into the report.
    fn account(&mut self, k: &Kernel<T>) {
        self.report.inline_fn_ops += k.n_inline;
        self.report.indirect_fn_ops += k.n_indirect;
    }

    /// Materialize one reduction node into a fresh buffer, applying
    /// `store_maps` to each accumulator at the flush.
    fn reduce(
        &mut self,
        op: usize,
        bound: &[IndexId],
        free: &[IndexId],
        body: &FExpr<T>,
        store_maps: &[usize],
    ) -> usize {
        let shape: Vec<usize> = free.iter().map(|&i| self.extents[i as usize]).collect();

        #[cfg(feature = "jit")]
        if let Some((data, n_load_maps, any_dynamic, has_store_map)) =
            self.try_jit(op, bound, free, body, store_maps, &shape)
        {
            self.report.jit_reductions += 1;
            self.report.jit_load_maps += n_load_maps;
            self.report.jit_store_maps += usize::from(has_store_map);
            self.report.jit_dynamic_reductions += usize::from(any_dynamic);
            return self.push(data, shape);
        }

        let levels: Vec<IndexId> = free.iter().chain(bound).copied().collect();
        let dims: Vec<usize> = levels.iter().map(|&i| self.extents[i as usize]).collect();
        let kernel = Builder::new(&levels, self.n_ids, &self.reg.fn_kind).finish(body, dims);
        self.account(&kernel);

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
        apply_store_maps(self.reg, store_maps, &mut out, &mut self.regs.args);
        self.report.tape_reductions += 1;
        self.push(out, shape)
    }
}

/// Free indices of a reduction body: everything it reads that the binder
/// does not bind, ascending — the temporary's axis order.
fn free_ids<T>(body: &FExpr<T>, bound: &[IndexId]) -> Vec<IndexId> {
    let mut free = Vec::new();
    body.collect_ids(&mut free);
    free.retain(|id| !bound.contains(id));
    free.sort_unstable();
    free.dedup();
    free
}

/// Look through a chain of single-argument calls: the first node that is
/// not one. Allocation-free, so the overwhelmingly common "this call does
/// not wrap a reduction" answer costs nothing.
fn unary_chain_target(e: &CExpr) -> &CExpr {
    let mut cur = e;
    while let CExpr::Call { func: _, args } = cur
        && args.len() == 1
    {
        cur = &args[0];
    }
    cur
}

/// The function ids of that chain, outermost-first.
fn unary_chain_funcs(e: &CExpr) -> Vec<usize> {
    let mut chain = Vec::new();
    let mut cur = e;
    while let CExpr::Call { func, args } = cur
        && args.len() == 1
    {
        chain.push(*func);
        cur = &args[0];
    }
    chain
}

// ─────────────────────────────────────────────────────────────────────────
// JIT lowering
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "jit")]
mod jit_route {
    use super::{Elem, FExpr, FnKind, IndexId};
    use crate::dense::Dense;
    use crate::einsum::{Combine, Reduce};
    use crate::jit::{EinsumF32Jit, JitInput, UnaryMap};
    use std::any::{Any, TypeId};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    /// The built-ins this backend can emit *exactly* — see
    /// [`UnaryMap`]'s table. `exp`/`ln`/`tanh` are absent on purpose:
    /// emitting them would need a libcall whose rounding we cannot pin to
    /// the `libm` the tape uses, so a mapped `exp` stays on the tape.
    pub(super) fn jit_map(kind: FnKind) -> Option<UnaryMap> {
        Some(match kind {
            FnKind::Neg => UnaryMap::Neg,
            FnKind::Abs => UnaryMap::Abs,
            FnKind::Sqrt => UnaryMap::Sqrt,
            FnKind::Relu => UnaryMap::Relu,
            _ => return None,
        })
    }

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

    /// Everything the generated code bakes in — and *only* that.
    ///
    /// Two calls that agree on this key must be servable by the same
    /// machine code, so every axis whose extent is a compile-time constant
    /// has to appear here and every axis that is a runtime argument has to
    /// *not*. Hence:
    ///
    /// - `spec` — the contraction's shape as a string, which fixes the loop
    ///   nest, the per-input index patterns and the output pattern;
    /// - `sparse_mask` — the per-input **layout** (bit `i` set = input `i`
    ///   is sparse), because dense addressing and CSR row iteration are
    ///   different code. `lang` never sets a bit (a non-contiguous input
    ///   sends the whole run to the tree-walker), but the key states the
    ///   dependency rather than relying on that;
    /// - `in_dims` / `out_dims` — the **static** extents, per axis, with
    ///   `None` at every axis compiled as a runtime argument. The `None`
    ///   pattern is what makes two differently-marked calls distinct keys,
    ///   so no separate "which axes are dynamic" field is needed;
    /// - `reduce` / `combine` and the map chains — the arithmetic emitted
    ///   inside the nest.
    ///
    /// A fully static call therefore keys exactly as it did before dynamic
    /// axes existed (`in_dims` is all `Some`), and marking an axis dynamic
    /// collapses every extent of it onto one entry.
    #[derive(PartialEq, Eq)]
    pub(super) struct Key {
        pub spec: String,
        pub sparse_mask: u32,
        pub in_dims: Vec<Vec<Option<usize>>>,
        pub out_dims: Vec<Option<usize>>,
        pub reduce: Reduce,
        pub combine: Combine,
        pub load_maps: Vec<Vec<UnaryMap>>,
        pub store_map: Vec<UnaryMap>,
    }

    /// One cache slot. `jit` is `None` for a pattern the backend refused,
    /// so we never re-pay ~400 µs of Cranelift to be told "unsupported" a
    /// second time. `used` is a logical clock stamp for eviction; it is a
    /// [`Cell`] so a hit stays on the shared `borrow()` path.
    pub(super) struct Slot {
        key: Key,
        jit: Option<Rc<EinsumF32Jit>>,
        used: Cell<u64>,
    }

    thread_local! {
        static CACHE: RefCell<Vec<Slot>> = const { RefCell::new(Vec::new()) };
        static CLOCK: Cell<u64> = const { Cell::new(0) };
    }

    /// Bound on cached kernels: JIT pages stay mapped as long as they are
    /// held, and the language targets a handful of hot programs.
    const CACHE_CAP: usize = 256;

    fn tick() -> u64 {
        CLOCK.with(|c| {
            let t = c.get() + 1;
            c.set(t);
            t
        })
    }

    pub(super) fn compiled(
        key: Key,
        inputs: &[JitInput],
        out_shape: &[usize],
        dynamic: &[char],
    ) -> Option<Rc<EinsumF32Jit>> {
        let now = tick();
        CACHE.with(|c| {
            if let Some(slot) = c.borrow().iter().find(|s| s.key == key) {
                slot.used.set(now);
                return slot.jit.clone();
            }
            let loads: Vec<&[UnaryMap]> = key.load_maps.iter().map(|m| m.as_slice()).collect();
            let built = EinsumF32Jit::compile_reduce_mapped_dyn(
                &key.spec,
                key.reduce,
                key.combine,
                &loads,
                &key.store_map,
                inputs,
                &[out_shape.to_vec()],
                dynamic,
            )
            .ok()
            .map(Rc::new);
            drop(loads);
            let mut c = c.borrow_mut();
            // Evict one entry — the least recently used — rather than half
            // the cache. Dropping half means a program whose working set
            // merely *touches* the cap re-compiles its own hot kernels
            // mid-run; dropping the coldest single entry keeps the hot set
            // resident as long as it fits. O(n) at n ≤ CACHE_CAP, and only
            // on a miss, which already costs a Cranelift compile.
            if c.len() >= CACHE_CAP
                && let Some(victim) = (0..c.len()).min_by_key(|&i| c[i].used.get())
            {
                c.swap_remove(victim);
            }
            c.push(Slot { key, jit: built.clone(), used: Cell::new(now) });
            built
        })
    }

    pub(super) fn cache_len() -> usize {
        CACHE.with(|c| c.borrow().len())
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

    /// One operand of a contraction: a tensor load, plus the **load map**
    /// applied to every element as it is read.
    pub(super) struct Leaf<'e> {
        pub indices: &'e [IndexId],
        pub buf: usize,
        /// Built-in unaries wrapping the load, in application order.
        pub maps: Vec<UnaryMap>,
    }

    /// A left-leaning chain `((l₀ ⊗ l₁) ⊗ l₂) ⊗ …` under a single `*` or
    /// `+`, or a lone operand. Returns the leaves in the order the backend
    /// combines them (`emit_contribution` folds left to right in input
    /// order, so any other association would change the bits).
    ///
    /// Each leaf is a tensor load optionally wrapped in built-in unaries —
    /// `sum(j: relu(a[i,j]) * b[j,k])` is still one contraction, with
    /// `relu` fused into the load. The map must be one the JIT emits
    /// exactly ([`jit_map`]); anything else declines and stays on the tape.
    ///
    /// Every leaf must carry at least one index (the spec parser rejects an
    /// empty input pattern) and no index twice (a diagonal read has no
    /// einsum-input form here).
    pub(super) fn flatten_chain<'e, T: Elem>(
        e: &'e FExpr<T>,
        kinds: &[Option<FnKind>],
    ) -> Option<(Combine, Vec<Leaf<'e>>)> {
        fn leaf<'e, T: Elem>(e: &'e FExpr<T>, kinds: &[Option<FnKind>]) -> Option<Leaf<'e>> {
            // Peel the unary wrappers outermost-first; they apply to the
            // loaded value innermost-first.
            let mut maps = Vec::new();
            let mut cur = e;
            while let FExpr::Call { func, args } = cur
                && args.len() == 1
            {
                maps.push(jit_map(kinds[*func]?)?);
                cur = &args[0];
            }
            maps.reverse();

            let FExpr::Load { buf, indices, .. } = cur else { return None };
            if indices.is_empty() {
                return None;
            }
            for (k, id) in indices.iter().enumerate() {
                if indices[..k].contains(id) {
                    return None;
                }
            }
            Some(Leaf { indices: indices.as_slice(), buf: *buf, maps })
        }

        fn walk<'e, T: Elem>(
            e: &'e FExpr<T>,
            want: super::BinOp,
            kinds: &[Option<FnKind>],
            out: &mut Vec<Leaf<'e>>,
        ) -> bool {
            if let FExpr::Binary { op, lhs, rhs } = e
                && *op == want
            {
                return walk(lhs, want, kinds, out)
                    && match leaf(rhs, kinds) {
                        Some(l) => {
                            out.push(l);
                            true
                        }
                        None => false,
                    };
            }
            match leaf(e, kinds) {
                Some(l) => {
                    out.push(l);
                    true
                }
                None => false,
            }
        }

        match e {
            FExpr::Binary { op, .. } => {
                let combine = match op {
                    super::BinOp::Mul => Combine::Mul,
                    super::BinOp::Add => Combine::Add,
                    _ => return None,
                };
                let mut leaves = Vec::new();
                walk(e, *op, kinds, &mut leaves).then_some((combine, leaves))
            }
            // A lone operand (possibly mapped): the combine never fires,
            // so its choice is immaterial.
            _ => Some((Combine::Mul, vec![leaf(e, kinds)?])),
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
    #[allow(clippy::too_many_arguments)]
    fn try_jit(
        &self,
        op: usize,
        bound: &[IndexId],
        free: &[IndexId],
        body: &FExpr<T>,
        store_maps: &[usize],
        out_shape: &[usize],
    ) -> Option<(Vec<T>, usize, bool, bool)> {
        use crate::einsum::Reduce;
        use crate::jit::{JitInput, UnaryMap};

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

        // ── An enclosing unary chain becomes the store map, but only if
        // this backend emits every link of it exactly. ──
        let store_map: Vec<UnaryMap> = store_maps
            .iter()
            .map(|&f| self.reg.builtin_fn(f).and_then(jit_route::jit_map))
            .collect::<Option<_>>()?;

        // ── Body must be a left-leaning chain of (possibly mapped) loads. ──
        let (combine, leaves) = jit_route::flatten_chain(body, &self.reg.fn_kind)?;

        // ── Assign a spec letter per index, in first-appearance order. ──
        let mut slot_of: HashMap<IndexId, u8> = HashMap::new();
        let mut appearance: Vec<IndexId> = Vec::new();
        for leaf in &leaves {
            for &id in leaf.indices {
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
            leaves.iter().map(|l| letters(l.indices)).collect::<Vec<_>>().join(","),
            letters(free)
        );

        // ── Dynamic axes: spec letters whose extent the generated code
        // takes as an argument instead of baking. Marked per *index*, so a
        // letter is dynamic exactly when the index it stands for is. The
        // extents themselves are unchanged — this only decides what is a
        // constant, and therefore what the cache key has to distinguish. ──
        let dynamic: Vec<char> = appearance
            .iter()
            .filter(|&&id| self.is_dynamic(id))
            .map(|id| (b'a' + slot_of[id]) as char)
            .collect();
        let is_dyn = |id: IndexId| self.is_dynamic(id);

        let bufs = self.slices();
        let mut inputs: Vec<JitInput> = Vec::with_capacity(leaves.len());
        let mut in_dims: Vec<Vec<Option<usize>>> = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            let data = jit_route::as_f32_slice(bufs[leaf.buf])?;
            let shape = self.shapes[leaf.buf].clone();
            debug_assert_eq!(shape.len(), leaf.indices.len());
            in_dims.push(
                leaf.indices
                    .iter()
                    .zip(&shape)
                    .map(|(&id, &d)| if is_dyn(id) { None } else { Some(d) })
                    .collect(),
            );
            inputs.push(JitInput::DenseSlice { data, shape });
        }
        // An empty `load_maps` *is* "no maps" for the backend, so an
        // unmapped contraction — the common case — stays exactly as cheap
        // to key and compile as it was before load maps existed.
        let n_mapped = leaves.iter().filter(|l| !l.maps.is_empty()).count();
        let load_maps: Vec<Vec<UnaryMap>> = if n_mapped == 0 {
            Vec::new()
        } else {
            leaves.iter().map(|l| l.maps.clone()).collect()
        };
        let n_store = store_map.len();

        let key = jit_route::Key {
            spec,
            // Every leaf is a `DenseSlice`: `try_run` bails out before this
            // point if any input lacks a contiguous row-major image.
            sparse_mask: 0,
            in_dims,
            out_dims: free
                .iter()
                .zip(out_shape)
                .map(|(&id, &d)| if is_dyn(id) { None } else { Some(d) })
                .collect(),
            reduce,
            combine,
            load_maps,
            store_map,
        };
        let jit = jit_route::compiled(key, &inputs, out_shape, &dynamic)?;
        let identity = if reduce == Reduce::Sum { 0.0f32 } else { 1.0f32 };
        let out = jit_route::run_into(&jit, &inputs, out_shape, identity);
        Some((jit_route::from_f32_vec(out), n_mapped, !dynamic.is_empty(), n_store > 0))
    }
}

/// Compiled-kernel cache entries held by this thread — i.e. how many
/// distinct cache keys ([`jit_route::Key`]) have been seen, one Cranelift
/// compile each.
///
/// Diagnostics only; it says nothing about results. It is the number
/// [`RunOptions::dynamic`](super::RunOptions::dynamic) exists to shrink:
/// a `t`-dependent contraction run at 32 different `t` is 32 entries with
/// `t` static and 1 with `t` dynamic. The cache is per thread and bounded
/// (least-recently-used eviction above 256 entries), so this saturates
/// rather than growing without limit.
#[cfg(feature = "jit")]
pub fn jit_cache_len() -> usize {
    jit_route::cache_len()
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
    dynamic: &[bool],
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
        dynamic,
        n_ids,
        bufs,
        shapes,
        regs: Regs::new(),
        report: RunReport {
            statements: checked.stmts.len(),
            kernel_statements: checked.stmts.len(),
            dynamic_axes: dynamic.iter().filter(|&&d| d).count(),
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
                let kernel =
                    Builder::new(&stmt.lhs, n_ids, &reg.fn_kind).finish(&rhs, shape.clone());
                f.account(&kernel);
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
