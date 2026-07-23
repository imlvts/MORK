//! Cranelift JIT backend for Einstein summation, dense and CSR-sparse.
//!
//! This is the native-code counterpart to the interpreted [`crate::einsum`]
//! VM. A spec like `"ab,bc->ac"` plus concrete inputs is compiled — once —
//! into a machine-code loop nest that reads and writes the tensors' backing
//! storage directly (no per-element trait dispatch). The compiled
//! [`EinsumF32Jit`] is then reused across many executions of the same spec and
//! shapes.
//!
//! # Inputs and outputs
//!
//! Inputs are passed as [`JitInput`] and may be mixed freely:
//! - [`Dense<f32>`] of any rank;
//! - a square 2D [`Csr<u32, f32>`] (the graph matrix type);
//! - a general [`SparseView`] — a batched, possibly **rectangular** CSR-style
//!   tensor whose last axis is the stored column and whose leading axes form
//!   the compound row index. This covers non-square sparse matrices and
//!   higher-rank batched sparse tensors (e.g. `"bij,bjk->bik"`).
//!
//! Outputs are always [`Dense<f32>`] (sparse storage is immutable, so it
//! can't be written into).
//!
//! # Design
//!
//! - **Direct memory.** The codegen emits native loads/stores against the raw
//!   data pointers. Dense tensors get random-access addressing with row-major
//!   strides baked in as constants ([`DenseLayout`]); CSR inputs are walked
//!   with native **sparse row iteration** — for a fixed row, the inner loop
//!   visits only that row's stored non-zeros (over `row_ptr`/`col_idx`/
//!   `values`), exactly like the VM's sparse loop, so structural zeros are
//!   skipped at native speed.
//! - **Shape-specialized, except where you ask otherwise.** Dimensions are
//!   baked in as constants by default, so a given [`EinsumF32Jit`] is valid
//!   only for the exact shapes (and per-input dense/sparse kinds) it was
//!   compiled for; [`run`](EinsumF32Jit::run) asserts this. Naming a spec
//!   letter in [`compile_reduce_mapped_dyn`](EinsumF32Jit::compile_reduce_mapped_dyn)'s
//!   `dynamic` list instead makes that axis a **runtime argument**: its
//!   loop bound, and any stride that depends on it, are read from a dims
//!   vector on entry, so one compiled program serves every extent of that
//!   axis. Everything else stays a constant. This is a pure
//!   *specialization* choice — the emitted floating-point operation
//!   sequence is identical either way, so results are bit-identical to the
//!   fully-baked form.
//! - **Layout seam.** [`Layout`] abstracts random-access element addressing.
//!   `DenseLayout` is the only implementor (CSR has no constant-time random
//!   address, so it participates through sparse iteration instead).
//!
//! The call boundary is type-erased to arrays of raw pointers, so the
//! Rust-side call is a single monomorphic `extern "C"` invocation regardless
//! of arity or per-input layout. A dense input contributes one pointer
//! (its `f32` data); a CSR input contributes three (`row_ptr`, `col_idx`,
//! `values`); the third argument carries the dynamic axes' extents, one
//! `usize` per dynamic spec letter in ascending letter order (unread, and
//! null, when there are none):
//!
//! ```text
//! extern "C" fn(ins: *const *const u8, outs: *const *mut u8, dims: *const usize)
//! ```
//!
//! # Coverage limits
//!
//! A CSR input is only handled when its row axis can be fixed by an outer
//! loop before its column axis is needed — true for the common cases
//! (CSR × Dense, CSR × CSR chains). Patterns where a sparse input's column
//! index is otherwise constrained (e.g. `"ab,cb->ac"` with both operands
//! sparse, or a sparse trace `"aa->"`) are rejected at compile time with
//! [`JitError::Unsupported`]; use the [`crate::einsum`] VM for those.
//!
//! # Generalized reductions
//!
//! [`EinsumF32Jit::compile_reduce`] / [`einsum_jit_reduce`] accept a
//! ([`Reduce`], [`Combine`]) operator pair, compiling e.g. max-product or
//! min-plus (tropical) contractions to the same loop nests with `fmax` /
//! `fmin` / `fadd` / `fmul` in place of the sum-product instructions —
//! the native counterpart of [`crate::einsum::einsum_reduce`]. Sparse
//! inputs are only accepted for (`Sum`, `Mul`), since sparse row iteration
//! skips structural zeros and that is unsound for any other semiring
//! (a missing entry is a real 0 that must compete in the reduction);
//! other pairs return [`JitError::Unsupported`] — use the VM.
//!
//! # Output convention
//!
//! Outputs are **folded into** with the reduce operator and must be
//! pre-filled by the caller with [`Reduce::identity`] — for the plain
//! sum-product entry points that is the usual zeroing (same convention as
//! [`crate::einsum`]); the [`einsum_jit_reduce`] wrapper does the fill for
//! you. For an all-dense single output the contraction is held in a
//! register and stored once per free-index tuple; otherwise outputs are
//! read-modify-write.
//!
//! # Example
//!
//! ```
//! use linalg::jit::{EinsumF32Jit, JitInput};
//! use linalg::dense::Dense;
//! use linalg::tensor::NDIndex;
//!
//! let mut a = Dense::<f32>::zeros(vec![2, 3]);
//! a.fill_from(&[1., 2., 3., 4., 5., 6.]);
//! let mut b = Dense::<f32>::zeros(vec![3, 2]);
//! b.fill_from(&[7., 8., 9., 10., 11., 12.]);
//! let mut c = Dense::<f32>::zeros(vec![2, 2]);
//!
//! let jit = EinsumF32Jit::compile(
//!     "ab,bc->ac",
//!     &[JitInput::Dense(&a), JitInput::Dense(&b)],
//!     &[vec![2, 2]],
//! ).unwrap();
//! jit.run(&[JitInput::Dense(&a), JitInput::Dense(&b)], &mut [&mut c]);
//!
//! assert_eq!(c.get(&[0, 0]), 58.0);
//! assert_eq!(c.get(&[1, 1]), 154.0);
//! ```

use std::fmt;
use std::mem;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, Block, InstBuilder, MemFlags, Type, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::csr::Csr;
use crate::dense::Dense;
use crate::einsum::{parse_spec, Combine, InvalidSpec, Reduce};

// ─────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────

/// Failure compiling an einsum spec for the JIT backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitError {
    /// The spec itself is malformed or inconsistent with the shapes.
    Spec(InvalidSpec),
    /// The spec is valid but this backend cannot generate code for it
    /// (see the module-level "Coverage limits"). The string explains why.
    Unsupported(&'static str),
}

impl From<InvalidSpec> for JitError {
    fn from(e: InvalidSpec) -> Self {
        JitError::Spec(e)
    }
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::Spec(e) => write!(f, "{e}"),
            JitError::Unsupported(why) => write!(f, "unsupported by JIT backend: {why}"),
        }
    }
}

impl std::error::Error for JitError {}

// ─────────────────────────────────────────────────────────────────────────
// Load / store maps
// ─────────────────────────────────────────────────────────────────────────

/// An elementwise unary applied *inside* a compiled contraction: to an
/// input value as it is loaded (a **load map**), or to the finished
/// accumulator just before it is stored (a **store map**). Fusing these
/// into the kernel is what turns `sqrt(sum(j: relu(a[i,j]) * b[j]))` into
/// a single loop nest with no intermediate buffers.
///
/// Deliberately restricted to unaries that lower to a *single, exactly
/// specified* float operation, so a mapped kernel is bit-identical to
/// applying the same function elementwise in Rust:
///
/// | variant | emitted | Rust equivalent |
/// |---|---|---|
/// | `Neg`  | `fneg` (sign-bit flip) | `-x` |
/// | `Abs`  | `fabs` (sign-bit clear) | `x.abs()` |
/// | `Sqrt` | `sqrt` (IEEE, correctly rounded) | `x.sqrt()` |
/// | `Relu` | `select(x > 0.0, x, +0.0)` | `if x > 0.0 { x } else { 0.0 }` |
///
/// Transcendentals (`exp`, `ln`, `tanh`) are *not* here: they would need a
/// libcall whose rounding this backend cannot pin to the host `libm` the
/// rest of the program uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryMap {
    /// `-x`.
    Neg,
    /// `|x|`.
    Abs,
    /// `√x` (`NaN` for a negative argument, `-0.0` for `-0.0`).
    Sqrt,
    /// `max(x, 0)` written as `if x > 0.0 { x } else { 0.0 }` — so
    /// `relu(-0.0)` and `relu(NaN)` are both `+0.0`.
    Relu,
}

// ─────────────────────────────────────────────────────────────────────────
// Public input handle
// ─────────────────────────────────────────────────────────────────────────

/// A borrowed batched / rectangular sparse tensor in CSR-style layout.
///
/// The **last** axis is the sparse stored column; all preceding axes form the
/// compound row index, flattened row-major (C-order) into `row_ptr`. This
/// represents a stack of `rows × cols` sparse matrices (one per leading-index
/// tuple) — so it covers non-square matrices and batched sparse tensors
/// (e.g. `"bij"`). A square 2D [`Csr`] is the special case `B = 1, rows = cols`.
#[derive(Clone)]
pub struct SparseView<'a> {
    shape: Vec<usize>,
    row_ptr: &'a [usize],
    col_idx: &'a [u32],
    values: &'a [f32],
}

impl<'a> SparseView<'a> {
    /// Build a view from raw CSR-style arrays.
    ///
    /// - `shape`: full logical shape, `ndim >= 2`. Last axis = sparse column;
    ///   the product of the preceding axes is the number of compound rows.
    /// - `row_ptr`: length `compound_rows + 1`; `row_ptr[r]..row_ptr[r+1]` is
    ///   the stored-entry range for compound row `r`.
    /// - `col_idx` / `values`: parallel arrays of length `nnz`; each
    ///   `col_idx[k] < shape[last]`.
    ///
    /// Panics if the lengths are inconsistent with `shape`.
    pub fn new(
        shape: Vec<usize>,
        row_ptr: &'a [usize],
        col_idx: &'a [u32],
        values: &'a [f32],
    ) -> Self {
        assert!(shape.len() >= 2, "sparse view needs ndim >= 2");
        let rows: usize = shape[..shape.len() - 1].iter().product();
        assert_eq!(
            row_ptr.len(),
            rows + 1,
            "row_ptr length {} must be product(leading dims)+1 = {}",
            row_ptr.len(),
            rows + 1
        );
        assert_eq!(col_idx.len(), values.len(), "col_idx and values length differ");
        Self { shape, row_ptr, col_idx, values }
    }
}

/// A tensor passed to the JIT: dense, a [`Csr`] (any shape), or a general
/// (batched / rectangular) [`SparseView`]. `Csr` and `SparseView` share the
/// same CSR-style layout and code path; `Csr` is the convenient owning type.
#[derive(Clone)]
pub enum JitInput<'a> {
    Dense(&'a Dense<f32>),
    /// A borrowed contiguous row-major buffer with an explicit shape —
    /// the same layout as [`JitInput::Dense`], for callers that hold the
    /// storage as a plain slice rather than an owning [`Dense`].
    /// `data.len()` must equal the product of `shape` (1 for rank 0).
    DenseSlice { data: &'a [f32], shape: Vec<usize> },
    Csr(&'a Csr<u32, f32>),
    Sparse(SparseView<'a>),
}

impl JitInput<'_> {
    fn is_sparse(&self) -> bool {
        !matches!(self, JitInput::Dense(_) | JitInput::DenseSlice { .. })
    }

    /// Logical einsum shape.
    fn shape(&self) -> Vec<usize> {
        match self {
            JitInput::Dense(d) => d.shape.clone(),
            JitInput::DenseSlice { shape, .. } => shape.clone(),
            JitInput::Csr(c) => c.shape.clone(),
            JitInput::Sparse(v) => v.shape.clone(),
        }
    }

    /// Append this input's backing pointers in codegen slot order. Sparse
    /// inputs contribute `row_ptr`, `col_idx`, `values` (in that order).
    fn push_ptrs(&self, out: &mut Vec<*const u8>) {
        match self {
            JitInput::Dense(d) => out.push(d.data.as_ptr() as *const u8),
            JitInput::DenseSlice { data, .. } => out.push(data.as_ptr() as *const u8),
            JitInput::Csr(c) => {
                out.push(c.row_ptr.as_ptr() as *const u8);
                out.push(c.col_idx.as_ptr() as *const u8);
                out.push(c.values.as_ptr() as *const u8);
            }
            JitInput::Sparse(v) => {
                out.push(v.row_ptr.as_ptr() as *const u8);
                out.push(v.col_idx.as_ptr() as *const u8);
                out.push(v.values.as_ptr() as *const u8);
            }
        }
    }
}

/// What this program was compiled to expect for one input.
#[derive(Clone, PartialEq, Eq)]
struct InputSpec {
    is_sparse: bool,
    /// The spec slot of each axis, so a dynamic axis can be resolved
    /// against the dims table at run time.
    pattern: Vec<u8>,
    /// Compiled extent per axis; `None` where the axis is **dynamic** and
    /// so is whatever the caller passes.
    shape: Vec<Option<usize>>,
}

// ─────────────────────────────────────────────────────────────────────────
// Dimensions: baked constants and runtime arguments
// ─────────────────────────────────────────────────────────────────────────

/// An extent (or a stride derived from extents) inside generated code: a
/// compile-time constant, or a value read from the runtime dims argument.
///
/// A [`Dim::Var`] is always materialized in the entry block, which
/// dominates the whole nest, so it is usable from any later block.
#[derive(Clone, Copy)]
enum Dim {
    Const(i64),
    Var(Value),
}

impl Dim {
    fn value(self, b: &mut FunctionBuilder, ptr_ty: Type) -> Value {
        match self {
            Dim::Const(c) => b.ins().iconst(ptr_ty, c),
            Dim::Var(v) => v,
        }
    }

    /// `self * other`, folding when both sides are known. Constant folding
    /// here is what keeps a fully-static program's code *identical* to what
    /// it was before dynamic axes existed: nothing is emitted at all.
    fn mul(self, other: Dim, b: &mut FunctionBuilder, ptr_ty: Type) -> Dim {
        match (self, other) {
            (Dim::Const(a), Dim::Const(c)) => Dim::Const(a * c),
            _ => {
                let x = self.value(b, ptr_ty);
                let y = other.value(b, ptr_ty);
                Dim::Var(b.ins().imul(x, y))
            }
        }
    }

    /// `factor * self`, as an addend of an address computation.
    fn mul_index(self, idx: Value, b: &mut FunctionBuilder) -> Value {
        match self {
            Dim::Const(c) => b.ins().imul_imm(idx, c),
            Dim::Var(v) => b.ins().imul(idx, v),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Layout seam (dense random-access addressing)
// ─────────────────────────────────────────────────────────────────────────

/// How to compute the byte address of a randomly-indexed element of a tensor
/// struct. CSR has no constant-time random address, so it does not implement
/// this — it participates through sparse row iteration instead.
trait Layout {
    fn emit_elem_addr(
        &self,
        b: &mut FunctionBuilder,
        ptr_ty: Type,
        base: Value,
        indices: &[Value],
    ) -> Value;
}

/// Row-major dense layout. A stride is baked in as a constant unless some
/// extent it is a product of is dynamic, in which case it is computed once
/// in the entry block.
struct DenseLayout {
    /// Element stride per axis (in elements, not bytes).
    strides: Vec<Dim>,
}

impl DenseLayout {
    fn new(b: &mut FunctionBuilder, ptr_ty: Type, axis_dims: &[Dim]) -> Self {
        let n = axis_dims.len();
        let mut strides = vec![Dim::Const(1); n];
        for j in (0..n.saturating_sub(1)).rev() {
            strides[j] = strides[j + 1].mul(axis_dims[j + 1], b, ptr_ty);
        }
        Self { strides }
    }
}

impl Layout for DenseLayout {
    fn emit_elem_addr(
        &self,
        b: &mut FunctionBuilder,
        ptr_ty: Type,
        base: Value,
        indices: &[Value],
    ) -> Value {
        let mut off = b.ins().iconst(ptr_ty, 0);
        for (j, &idx) in indices.iter().enumerate() {
            let contrib = self.strides[j].mul_index(idx, b);
            off = b.ins().iadd(off, contrib);
        }
        // f32 == 4 bytes; offset (elements) << 2 == byte offset.
        let byte_off = b.ins().ishl_imm(off, 2);
        b.ins().iadd(base, byte_off)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Compiled program
// ─────────────────────────────────────────────────────────────────────────

/// A compiled, shape-specialized einsum producing [`Dense<f32>`] output(s).
///
/// Construct with [`compile`](Self::compile); execute with
/// [`run`](Self::run). Holds the owning [`JITModule`] so the generated code
/// stays mapped for the lifetime of this value; the code memory is released
/// on drop.
pub struct EinsumF32Jit {
    /// `Option` only so [`Drop`] can `take()` it and call the consuming
    /// `JITModule::free_memory(self)`. Always `Some` between construction
    /// and drop; the [`Drop`] impl is the only place it goes to `None`.
    module: Option<JITModule>,
    func: extern "C" fn(*const *const u8, *const *mut u8, *const usize),
    inputs: Vec<InputSpec>,
    /// Spec slot per axis of each output, and the compiled extent (`None`
    /// where the axis is dynamic).
    outputs: Vec<(Vec<u8>, Vec<Option<usize>>)>,
    /// Spec slots compiled as runtime arguments, ascending. The generated
    /// code reads their extents from the dims argument in this order.
    dyn_slots: Vec<u8>,
    /// Parallel to `dyn_slots`: `(input, axis)` to read each dynamic
    /// extent from at run time. Every dynamic slot is required at compile
    /// time to occur in some input pattern, so this always exists.
    dyn_src: Vec<(usize, usize)>,
    /// Extent per spec slot as compiled; entries for dynamic slots are
    /// placeholders that [`run`](Self::run) overwrites.
    static_dims: [usize; 26],
}

impl Drop for EinsumF32Jit {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: we own the only reference to the JIT'd code (`func`
            // lives in this same struct and is dropped with us, and the
            // function pointer never escapes — `run` takes `&self`). So
            // nothing else can be running or about to call the code.
            unsafe { m.free_memory() };
        }
    }
}

impl EinsumF32Jit {
    /// Compile `spec` for the given inputs and output shapes.
    ///
    /// Only each input's shape and kind (dense vs sparse) are read here, not
    /// its data — but those are baked in, so the returned program is valid
    /// only for inputs matching exactly. Shapes are outermost-axis-first.
    pub fn compile(
        spec: &str,
        inputs: &[JitInput],
        output_shapes: &[Vec<usize>],
    ) -> Result<Self, JitError> {
        Self::compile_reduce(spec, Reduce::Sum, Combine::Mul, inputs, output_shapes)
    }

    /// [`compile`](Self::compile) generalized to an arbitrary
    /// ([`Reduce`], [`Combine`]) semiring — the native counterpart of
    /// [`crate::einsum::einsum_reduce`].
    ///
    /// The compiled program *folds into* the outputs with the reduce
    /// operator, so the caller must pre-fill them with
    /// [`Reduce::identity`] (for `Sum` that is the usual zeroing) — or use
    /// the [`einsum_jit_reduce`] wrapper which does the fill.
    ///
    /// Sparse inputs are only supported for (`Sum`, `Mul`): sparse row
    /// iteration skips structural zeros, which is unsound for any other
    /// semiring, and this backend has no dense-fallback addressing for
    /// CSR. Other op pairs with sparse inputs return
    /// [`JitError::Unsupported`]; use the VM for those.
    ///
    /// Float `Max`/`Min` compile to Cranelift `fmax`/`fmin`, which are
    /// NaN-propagating — the VM's comparison-based fold agrees for
    /// non-NaN data, but NaN behaviour is unspecified in both.
    pub fn compile_reduce(
        spec: &str,
        reduce: Reduce,
        combine: Combine,
        inputs: &[JitInput],
        output_shapes: &[Vec<usize>],
    ) -> Result<Self, JitError> {
        Self::compile_reduce_mapped(spec, reduce, combine, &[], &[], inputs, output_shapes)
    }

    /// [`compile_reduce`](Self::compile_reduce) with elementwise
    /// [`UnaryMap`] chains fused into the loop nest.
    ///
    /// - `load_maps` is either empty (no maps) or one chain per input,
    ///   applied left to right to each value as it is loaded.
    /// - `store_map` is applied left to right to the finished accumulator
    ///   immediately before the single store per output element.
    ///
    /// Restrictions, both reported as [`JitError::Unsupported`]:
    ///
    /// - a **store map** needs the register-accumulator loop nest, i.e. all
    ///   inputs dense and exactly one output — otherwise outputs are
    ///   read-modify-written and the map would be re-applied per
    ///   contraction step;
    /// - a **load map** on a sparse input is rejected outright: sparse
    ///   iteration skips structural zeros, so fusing `f` would silently
    ///   substitute `0` for `f(0)` on every skipped entry.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_reduce_mapped(
        spec: &str,
        reduce: Reduce,
        combine: Combine,
        load_maps: &[&[UnaryMap]],
        store_map: &[UnaryMap],
        inputs: &[JitInput],
        output_shapes: &[Vec<usize>],
    ) -> Result<Self, JitError> {
        Self::compile_reduce_mapped_dyn(
            spec,
            reduce,
            combine,
            load_maps,
            store_map,
            inputs,
            output_shapes,
            &[],
        )
    }

    /// [`compile_reduce_mapped`](Self::compile_reduce_mapped) with some
    /// axes compiled as **runtime arguments** instead of baked constants.
    ///
    /// `dynamic` names spec letters (`'t'`, …). For each one, the generated
    /// code reads the extent from a dims vector on entry rather than
    /// materializing it as a constant, and any stride that is a product
    /// involving it becomes a multiply instead of a folded constant.
    /// Everything else — every other loop bound, every other stride — is
    /// baked exactly as before, and a `dynamic` list that is empty
    /// reproduces the old code instruction for instruction.
    ///
    /// The point is reuse: one program then serves *every* extent of that
    /// axis, so a caller whose shapes grow (a KV cache, a batch dimension)
    /// pays Cranelift once instead of once per shape.
    /// [`run`](Self::run) reads the actual extents off the inputs and
    /// checks that every occurrence agrees, exactly as it checks a static
    /// axis against its compiled value.
    ///
    /// The **floating-point** instruction sequence is unaffected — same
    /// loop nest, same order, same operations — so a dynamic axis is
    /// bit-identical to the same axis baked static.
    ///
    /// Errors ([`JitError::Unsupported`]) if a letter is not a lowercase
    /// ASCII letter, does not occur in the spec, or occurs only in an
    /// output (there would be nothing to read its extent from).
    #[allow(clippy::too_many_arguments)]
    pub fn compile_reduce_mapped_dyn(
        spec: &str,
        reduce: Reduce,
        combine: Combine,
        load_maps: &[&[UnaryMap]],
        store_map: &[UnaryMap],
        inputs: &[JitInput],
        output_shapes: &[Vec<usize>],
        dynamic: &[char],
    ) -> Result<Self, JitError> {
        if !load_maps.is_empty() && load_maps.len() != inputs.len() {
            return Err(JitError::Unsupported("load_maps must be empty or one chain per input"));
        }
        let parsed = parse_spec(spec, inputs.len())?;
        let in_shapes: Vec<Vec<usize>> = inputs.iter().map(|i| i.shape()).collect();
        let is_sparse: Vec<bool> = inputs.iter().map(|i| i.is_sparse()).collect();

        // Collect and validate per-slot dimensions from input shapes.
        let mut dims = [0usize; 26];
        let mut dim_set = [false; 26];
        for (pi, pattern) in parsed.inputs.iter().enumerate() {
            if in_shapes[pi].len() != pattern.len() {
                return Err(InvalidSpec::InputNdimMismatch {
                    input: pi,
                    array_ndim: in_shapes[pi].len(),
                    spec_ndim: pattern.len(),
                }
                .into());
            }
            for (pos, &s) in pattern.iter().enumerate() {
                let si = s as usize;
                let d = in_shapes[pi][pos];
                if dim_set[si] {
                    if dims[si] != d {
                        return Err(InvalidSpec::DimensionMismatch {
                            index: (s + b'a') as char,
                            expected: dims[si],
                            got: d,
                        }
                        .into());
                    }
                } else {
                    dims[si] = d;
                    dim_set[si] = true;
                }
            }
        }

        // ── Dynamic axes: spec letters whose extent becomes an argument. ──
        let mut dyn_mask = [false; 26];
        for &c in dynamic {
            if !c.is_ascii_lowercase() {
                return Err(JitError::Unsupported("a dynamic axis must be a lowercase letter"));
            }
            dyn_mask[(c as u8 - b'a') as usize] = true;
        }
        // Where to read each dynamic extent from at run time: the first
        // input axis carrying it. An axis that no input carries has no such
        // source, so it cannot be dynamic.
        let mut dyn_src_of: [Option<(usize, usize)>; 26] = [None; 26];
        for (pi, pattern) in parsed.inputs.iter().enumerate() {
            for (pos, &s) in pattern.iter().enumerate() {
                let si = s as usize;
                if dyn_mask[si] && dyn_src_of[si].is_none() {
                    dyn_src_of[si] = Some((pi, pos));
                }
            }
        }
        for si in 0..26 {
            if dyn_mask[si] && dyn_src_of[si].is_none() {
                return Err(JitError::Unsupported(
                    "a dynamic axis must occur in at least one input",
                ));
            }
        }
        let dyn_slots: Vec<u8> = (0..26u8).filter(|&s| dyn_mask[s as usize]).collect();
        let dyn_src: Vec<(usize, usize)> =
            dyn_slots.iter().map(|&s| dyn_src_of[s as usize].expect("checked above")).collect();

        // Validate output shapes against the resolved dims.
        if output_shapes.len() != parsed.outputs.len() {
            return Err(InvalidSpec::WrongInputCount {
                expected: parsed.outputs.len(),
                got: output_shapes.len(),
            }
            .into());
        }
        for (oi, pattern) in parsed.outputs.iter().enumerate() {
            if output_shapes[oi].len() != pattern.len() {
                return Err(InvalidSpec::OutputNdimMismatch {
                    array_ndim: output_shapes[oi].len(),
                    spec_ndim: pattern.len(),
                }
                .into());
            }
            for (pos, &s) in pattern.iter().enumerate() {
                if output_shapes[oi][pos] != dims[s as usize] {
                    return Err(InvalidSpec::OutputDimMismatch {
                        axis: pos,
                        expected: dims[s as usize],
                        got: output_shapes[oi][pos],
                    }
                    .into());
                }
            }
        }

        let no_maps: Vec<&[UnaryMap]> = vec![&[]; inputs.len()];
        let load_maps = if load_maps.is_empty() { &no_maps[..] } else { load_maps };
        let (module, func) = codegen(
            &parsed.inputs,
            &parsed.outputs,
            &is_sparse,
            &dims,
            &dyn_slots,
            reduce,
            combine,
            load_maps,
            store_map,
        )?;

        // A dynamic axis is recorded as `None` rather than as its sample
        // extent, so `run` accepts any extent there and asserts the rest.
        let erase = |pattern: &[u8], shape: &[usize]| -> Vec<Option<usize>> {
            pattern
                .iter()
                .zip(shape)
                .map(|(&s, &d)| if dyn_mask[s as usize] { None } else { Some(d) })
                .collect()
        };

        Ok(Self {
            module: Some(module),
            func,
            inputs: parsed
                .inputs
                .iter()
                .zip(&in_shapes)
                .zip(is_sparse)
                .map(|((pattern, shape), is_sparse)| InputSpec {
                    is_sparse,
                    pattern: pattern.clone(),
                    shape: erase(pattern, shape),
                })
                .collect(),
            outputs: parsed
                .outputs
                .iter()
                .zip(output_shapes)
                .map(|(pattern, shape)| (pattern.clone(), erase(pattern, shape)))
                .collect(),
            dyn_slots,
            dyn_src,
            static_dims: dims,
        })
    }

    /// Execute against concrete tensors.
    ///
    /// Panics if any input/output count, kind, rank, or **static** extent
    /// does not match what this program was compiled for. Dynamic axes
    /// (see [`compile_reduce_mapped_dyn`](Self::compile_reduce_mapped_dyn))
    /// take their extent from the inputs, and every occurrence of one —
    /// inputs and outputs alike — must agree. Outputs must be pre-zeroed.
    pub fn run(&self, inputs: &[JitInput], outputs: &mut [&mut Dense<f32>]) {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "input count mismatch: got {}, compiled for {}",
            inputs.len(),
            self.inputs.len()
        );
        assert_eq!(
            outputs.len(),
            self.outputs.len(),
            "output count mismatch: got {}, compiled for {}",
            outputs.len(),
            self.outputs.len()
        );

        // Kind and rank first, so the dynamic-extent lookups below index
        // shapes that are known to be long enough.
        let in_shapes: Vec<Vec<usize>> = inputs.iter().map(|i| i.shape()).collect();
        for (i, inp) in inputs.iter().enumerate() {
            let spec = &self.inputs[i];
            assert_eq!(
                inp.is_sparse(), spec.is_sparse,
                "input {i} kind mismatch (dense vs sparse)"
            );
            assert_eq!(
                in_shapes[i].len(), spec.shape.len(),
                "input {i} rank mismatch: got {}, compiled for {}",
                in_shapes[i].len(), spec.shape.len()
            );
        }

        // Resolve the dynamic axes from the inputs, then check *every*
        // occurrence against the resolved table — which subsumes the static
        // check, since a static slot's entry is its compiled extent.
        let mut dims = self.static_dims;
        let mut dyn_vals = [0usize; 26];
        for (k, &slot) in self.dyn_slots.iter().enumerate() {
            let (i, axis) = self.dyn_src[k];
            dims[slot as usize] = in_shapes[i][axis];
            dyn_vals[k] = in_shapes[i][axis];
        }
        for (i, spec) in self.inputs.iter().enumerate() {
            for (pos, &s) in spec.pattern.iter().enumerate() {
                assert_eq!(
                    in_shapes[i][pos], dims[s as usize],
                    "input {i} axis {pos} ('{}') is {}, expected {}",
                    (s + b'a') as char, in_shapes[i][pos], dims[s as usize]
                );
            }
        }
        for (o, out) in outputs.iter().enumerate() {
            let (pattern, _) = &self.outputs[o];
            assert_eq!(
                out.shape.len(), pattern.len(),
                "output {o} rank mismatch: got {}, compiled for {}",
                out.shape.len(), pattern.len()
            );
            for (pos, &s) in pattern.iter().enumerate() {
                assert_eq!(
                    out.shape[pos], dims[s as usize],
                    "output {o} axis {pos} ('{}') is {}, expected {}",
                    (s + b'a') as char, out.shape[pos], dims[s as usize]
                );
            }
        }

        let mut in_ptrs: Vec<*const u8> = Vec::new();
        for inp in inputs {
            inp.push_ptrs(&mut in_ptrs);
        }
        let out_ptrs: Vec<*mut u8> =
            outputs.iter_mut().map(|d| d.data.as_mut_ptr() as *mut u8).collect();
        // SAFETY: the generated code reads exactly the pointer slots that the
        // inputs above produce (kinds asserted to match what was compiled),
        // reads `dyn_slots.len()` entries of the dims vector (a 26-element
        // stack array, and there are at most 26 slots), and addresses only
        // elements within the validated shapes.
        (self.func)(in_ptrs.as_ptr(), out_ptrs.as_ptr(), dyn_vals.as_ptr());
    }
}

/// One-shot compile + run + free, matching the shape of
/// [`crate::einsum::einsum`]. Output shapes are taken from the passed-in
/// `Dense` outputs.
///
/// JIT compile is ~hundreds of µs, so this amortizes badly across repeated
/// calls with the same spec — prefer [`EinsumF32Jit::compile`] +
/// [`run`](EinsumF32Jit::run) when reusing a program. This wrapper exists
/// for one-off / scripty use where the convenience matters more than the
/// per-call compile cost.
///
/// # Example
///
/// ```
/// use linalg::jit::{einsum_jit, JitInput};
/// use linalg::dense::Dense;
/// use linalg::tensor::NDIndex;
///
/// let mut a = Dense::<f32>::zeros(vec![2, 3]);
/// a.fill_from(&[1., 2., 3., 4., 5., 6.]);
/// let mut b = Dense::<f32>::zeros(vec![3, 2]);
/// b.fill_from(&[7., 8., 9., 10., 11., 12.]);
/// let mut c = Dense::<f32>::zeros(vec![2, 2]);
///
/// einsum_jit("ab,bc->ac", &[JitInput::Dense(&a), JitInput::Dense(&b)], &mut [&mut c]).unwrap();
/// assert_eq!(c.get(&[0, 0]), 58.0);
/// ```
pub fn einsum_jit(
    spec: &str,
    inputs: &[JitInput],
    outputs: &mut [&mut Dense<f32>],
) -> Result<(), JitError> {
    let output_shapes: Vec<Vec<usize>> =
        outputs.iter().map(|o| o.shape.clone()).collect();
    let jit = EinsumF32Jit::compile(spec, inputs, &output_shapes)?;
    jit.run(inputs, outputs);
    // jit is dropped here — its Drop impl frees the JIT'd code memory.
    Ok(())
}

/// One-shot generalized-reduction einsum, matching the shape of
/// [`crate::einsum::einsum_reduce`]: outputs are **pre-filled with the
/// reduction identity** (overwrite semantics), then compiled, run, and the
/// code freed. Same amortization caveat as [`einsum_jit`] — prefer
/// [`EinsumF32Jit::compile_reduce`] + [`run`](EinsumF32Jit::run) for
/// repeated use (pre-filling the outputs yourself).
///
/// # Example
///
/// ```
/// use linalg::jit::{einsum_jit_reduce, JitInput};
/// use linalg::einsum::{Reduce, Combine};
/// use linalg::dense::Dense;
/// use linalg::tensor::NDIndex;
///
/// let mut a = Dense::<f32>::zeros(vec![2, 2]);
/// a.fill_from(&[1., -2., 3., 4.]);
/// let mut b = Dense::<f32>::zeros(vec![2, 2]);
/// b.fill_from(&[5., 6., -7., 8.]);
/// let mut c = Dense::<f32>::zeros(vec![2, 2]);
///
/// einsum_jit_reduce(
///     "ab,bc->ac",
///     Reduce::Max,
///     Combine::Mul,
///     &[JitInput::Dense(&a), JitInput::Dense(&b)],
///     &mut [&mut c],
/// ).unwrap();
/// assert_eq!(c.get(&[0, 0]), 14.0); // max(1·5, -2·-7)
/// ```
pub fn einsum_jit_reduce(
    spec: &str,
    reduce: Reduce,
    combine: Combine,
    inputs: &[JitInput],
    outputs: &mut [&mut Dense<f32>],
) -> Result<(), JitError> {
    let output_shapes: Vec<Vec<usize>> =
        outputs.iter().map(|o| o.shape.clone()).collect();
    let jit = EinsumF32Jit::compile_reduce(spec, reduce, combine, inputs, &output_shapes)?;
    let ident = reduce.identity::<f32>();
    for out in outputs.iter_mut() {
        out.data.fill(ident);
    }
    jit.run(inputs, outputs);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Loop scheduling
// ─────────────────────────────────────────────────────────────────────────

/// The sparse axes of one input: the trailing stored-column slot, and the
/// preceding slots that together form the compound row index (row-major).
struct SparseAxes {
    leading: Vec<u8>,
    col: u8,
}

/// One loop in the generated nest, outermost-first.
enum LoopOp {
    /// Dense counted loop `for slot in 0..dim`.
    Dense { slot: u8 },
    /// Iterate the stored non-zeros of `inputs[input_idx]`'s compound row
    /// (formed from `leading`), binding `col_slot` to each column index.
    Sparse { input_idx: usize, leading: Vec<u8>, col_slot: u8 },
}

/// Greedy schedule mirroring the VM: prefer a sparse row loop whose leading
/// (row) axes are all fixed; otherwise add a dense loop, favouring a sparse
/// input's leading axis so it can be unlocked next.
fn schedule(
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    sparse_axes: &[Option<SparseAxes>],
) -> Vec<LoopOp> {
    let mut all_slots = Vec::new();
    let mut seen = [false; 26];
    for pat in inputs.iter().chain(outputs.iter()) {
        for &s in pat {
            if !seen[s as usize] {
                seen[s as usize] = true;
                all_slots.push(s);
            }
        }
    }

    let mut fixed = [false; 26];
    let mut n_fixed = 0;
    let mut plan = Vec::new();

    while n_fixed < all_slots.len() {
        // Try to emit a sparse row loop whose leading axes are all fixed.
        let mut found = false;
        'scan: for &s in &all_slots {
            if fixed[s as usize] {
                continue;
            }
            for (idx, axes) in sparse_axes.iter().enumerate() {
                if let Some(ax) = axes {
                    let leads_fixed = ax.leading.iter().all(|&l| fixed[l as usize]);
                    if ax.col == s && !ax.leading.contains(&s) && leads_fixed {
                        plan.push(LoopOp::Sparse {
                            input_idx: idx,
                            leading: ax.leading.clone(),
                            col_slot: s,
                        });
                        fixed[s as usize] = true;
                        n_fixed += 1;
                        found = true;
                        break 'scan;
                    }
                }
            }
        }
        if found {
            continue;
        }

        // Otherwise a dense loop; prefer a sparse input's leading axis.
        let mut pick = None;
        for &s in &all_slots {
            if fixed[s as usize] {
                continue;
            }
            let is_leading = sparse_axes
                .iter()
                .any(|a| matches!(a, Some(ax) if ax.leading.contains(&s)));
            if is_leading {
                pick = Some(s);
                break;
            }
            if pick.is_none() {
                pick = Some(s);
            }
        }
        let s = pick.expect("unfixed slot exists");
        plan.push(LoopOp::Dense { slot: s });
        fixed[s as usize] = true;
        n_fixed += 1;
    }

    plan
}

// ─────────────────────────────────────────────────────────────────────────
// Code generation
// ─────────────────────────────────────────────────────────────────────────

/// How to emit a row-iteration loop for one tensor format in JIT'd code.
///
/// The format decides how to walk a row's stored entries given its own base
/// pointers (in `bases`, in the same order the format pushes them across the
/// call boundary). The codegen passes the already-computed compound row
/// index and the Cranelift variables to def with each entry's column and
/// value.
///
/// Same convention as [`open_dense_loop`]: returns `(induction_var, header,
/// exit)`; afterwards the builder sits in the (sealed) loop body with
/// `col_var`/`val_var` already defined. The caller closes the loop via
/// [`close_loop`].
trait RowLayout {
    /// Number of pointer-slots this format takes at the call boundary.
    fn n_ptr_slots(&self) -> usize;

    fn emit_row_open(
        &self,
        b: &mut FunctionBuilder,
        ptr_ty: Type,
        bases: &[Value],
        compound_row: Value,
        col_var: Variable,
        val_var: Variable,
    ) -> (Variable, Block, Block);
}

/// `Csr<u32, f32>` layout: three pointers (`row_ptr: usize`, `col_idx: u32`,
/// `values: f32`); the row's entries are `row_ptr[r]..row_ptr[r+1]`.
struct CsrRowLayout;

impl RowLayout for CsrRowLayout {
    fn n_ptr_slots(&self) -> usize {
        3
    }

    fn emit_row_open(
        &self,
        b: &mut FunctionBuilder,
        ptr_ty: Type,
        bases: &[Value],
        compound_row: Value,
        col_var: Variable,
        val_var: Variable,
    ) -> (Variable, Block, Block) {
        let row_ptr_base = bases[0];
        let col_idx_base = bases[1];
        let values_base = bases[2];

        // start = row_ptr[r], end = row_ptr[r + 1]  (usize / i64).
        let row_byte = b.ins().ishl_imm(compound_row, 3); // * 8
        let start_addr = b.ins().iadd(row_ptr_base, row_byte);
        let start = b.ins().load(types::I64, MemFlags::trusted(), start_addr, 0);
        let end = b.ins().load(types::I64, MemFlags::trusted(), start_addr, 8);

        let ei = b.declare_var(ptr_ty);
        b.def_var(ei, start);

        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.ins().jump(header, &[]);
        b.switch_to_block(header);
        let ei_v = b.use_var(ei);
        let cond = b.ins().icmp(IntCC::UnsignedLessThan, ei_v, end);
        b.ins().brif(cond, body, &[], exit, &[]);
        b.switch_to_block(body);
        b.seal_block(body);

        // col = col_idx[ei] (u32 -> index), val = values[ei] (f32).
        let ei_b = b.use_var(ei);
        let off4 = b.ins().ishl_imm(ei_b, 2); // * 4
        let col_addr = b.ins().iadd(col_idx_base, off4);
        let col32 = b.ins().load(types::I32, MemFlags::trusted(), col_addr, 0);
        let col = b.ins().uextend(ptr_ty, col32);
        b.def_var(col_var, col);
        let val_addr = b.ins().iadd(values_base, off4);
        let val = b.ins().load(types::F32, MemFlags::trusted(), val_addr, 0);
        b.def_var(val_var, val);

        (ei, header, exit)
    }
}

/// Base pointer(s) plus the layout for one input, as loaded at function entry.
enum InputBase {
    Dense { layout: DenseLayout, base: Value },
    Row { layout: Box<dyn RowLayout>, bases: Vec<Value> },
}

/// Open a dense counted loop `for v in 0..dim`. Returns `(header, exit)`;
/// afterwards the builder sits in the (sealed) loop body. A [`Dim::Const`]
/// bound emits the same `icmp_imm` it always did; a dynamic one compares
/// against the value read from the dims argument.
fn open_dense_loop(b: &mut FunctionBuilder, ptr_ty: Type, v: Variable, dim: Dim) -> (Block, Block) {
    let zero = b.ins().iconst(ptr_ty, 0);
    b.def_var(v, zero);
    let header = b.create_block();
    let body = b.create_block();
    let exit = b.create_block();
    b.ins().jump(header, &[]);
    b.switch_to_block(header);
    let idx = b.use_var(v);
    let cond = match dim {
        Dim::Const(c) => b.ins().icmp_imm(IntCC::UnsignedLessThan, idx, c),
        Dim::Var(d) => b.ins().icmp(IntCC::UnsignedLessThan, idx, d),
    };
    b.ins().brif(cond, body, &[], exit, &[]);
    b.switch_to_block(body);
    b.seal_block(body);
    (header, exit)
}

/// Row-major strides (in compound-row units) for the leading axes of a sparse
/// input, so `compound_row = Σ leading_val[k] * strides[k]`.
fn leading_strides(
    b: &mut FunctionBuilder,
    ptr_ty: Type,
    leading: &[u8],
    dims: &[Dim; 26],
) -> Vec<Dim> {
    let n = leading.len();
    let mut strides = vec![Dim::Const(1); n];
    for k in (0..n.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1].mul(dims[leading[k + 1] as usize], b, ptr_ty);
    }
    strides
}

/// Emit the row-major compound row index for a sparse input from its leading
/// slot vars and dims (`compound = Σ leading_val[k] · stride[k]`). Format-
/// agnostic — used by every row-layout call site.
fn emit_compound_row(
    b: &mut FunctionBuilder,
    ptr_ty: Type,
    leading: &[u8],
    leading_strides: &[Dim],
    vars: &[Variable; 26],
) -> Value {
    let mut compound = b.ins().iconst(ptr_ty, 0);
    for (k, &s) in leading.iter().enumerate() {
        let lv = b.use_var(vars[s as usize]);
        let contrib = leading_strides[k].mul_index(lv, b);
        compound = b.ins().iadd(compound, contrib);
    }
    compound
}

/// Close a loop: increment its induction variable, back-edge, seal the header,
/// then position the builder in the (sealed) exit block.
fn close_loop(b: &mut FunctionBuilder, iv: Variable, header: Block, exit: Block) {
    let idx = b.use_var(iv);
    let next = b.ins().iadd_imm(idx, 1);
    b.def_var(iv, next);
    b.ins().jump(header, &[]);
    b.seal_block(header);
    b.switch_to_block(exit);
    b.seal_block(exit);
}

/// Emit the reduce-op fold of the accumulator with one contribution.
///
/// NaN caveat: Cranelift's `fmax`/`fmin` are NaN-propagating, while the
/// VM's `Reduce::apply` is comparison-based — results agree for non-NaN
/// data; behaviour with NaN operands is unspecified in both.
fn emit_reduce(b: &mut FunctionBuilder, reduce: Reduce, acc: Value, v: Value) -> Value {
    match reduce {
        Reduce::Sum => b.ins().fadd(acc, v),
        Reduce::Prod => b.ins().fmul(acc, v),
        Reduce::Max => b.ins().fmax(acc, v),
        Reduce::Min => b.ins().fmin(acc, v),
    }
}

/// Emit one [`UnaryMap`], as the doc table on that type specifies.
fn emit_map(b: &mut FunctionBuilder, map: UnaryMap, v: Value) -> Value {
    match map {
        UnaryMap::Neg => b.ins().fneg(v),
        UnaryMap::Abs => b.ins().fabs(v),
        UnaryMap::Sqrt => b.ins().sqrt(v),
        UnaryMap::Relu => {
            // `if x > 0.0 { x } else { 0.0 }`: an *ordered* compare, so
            // NaN takes the else branch and yields +0.0 — same as the Rust
            // form, and unlike `fmax`.
            let zero = b.ins().f32const(0.0);
            let gt = b.ins().fcmp(FloatCC::GreaterThan, v, zero);
            b.ins().select(gt, v, zero)
        }
    }
}

/// Emit a whole map chain, left to right.
fn emit_map_chain(b: &mut FunctionBuilder, maps: &[UnaryMap], mut v: Value) -> Value {
    for &m in maps {
        v = emit_map(b, m, v);
    }
    v
}

/// Emit the combine ⊗ of all input elements at the current index values
/// (left-to-right in input order), each first passed through its load-map
/// chain. Dense inputs are loaded by computed address; sparse-covered
/// inputs read their cached per-iteration value variable.
#[allow(clippy::too_many_arguments)]
fn emit_contribution(
    b: &mut FunctionBuilder,
    ptr_ty: Type,
    combine: Combine,
    inputs: &[Vec<u8>],
    bases: &[InputBase],
    val_vars: &[Option<Variable>],
    vars: &[Variable; 26],
    load_maps: &[&[UnaryMap]],
) -> Value {
    let mut contrib: Option<Value> = None;
    for (i, pattern) in inputs.iter().enumerate() {
        let v = if let Some(vv) = val_vars[i] {
            b.use_var(vv)
        } else {
            match &bases[i] {
                InputBase::Dense { layout, base } => {
                    let idx_vals: Vec<Value> =
                        pattern.iter().map(|&s| b.use_var(vars[s as usize])).collect();
                    let addr = layout.emit_elem_addr(b, ptr_ty, *base, &idx_vals);
                    b.ins().load(types::F32, MemFlags::trusted(), addr, 0)
                }
                InputBase::Row { .. } => unreachable!("row input must be sparse-covered"),
            }
        };
        let v = emit_map_chain(b, load_maps[i], v);
        contrib = Some(match contrib {
            None => v,
            Some(p) => match combine {
                Combine::Mul => b.ins().fmul(p, v),
                Combine::Add => b.ins().fadd(p, v),
                Combine::Min => b.ins().fmin(p, v),
                Combine::Max => b.ins().fmax(p, v),
            },
        });
    }
    contrib.expect("einsum input list is non-empty")
}

/// Build the host ISA and a fresh JIT module.
fn new_module() -> JITModule {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_native::builder().expect("host machine is not supported");
    let isa = isa_builder
        .finish(settings::Flags::new(flags))
        .expect("failed to build ISA");
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    JITModule::new(builder)
}

/// Generate native code for `inputs -> outputs` with the given per-slot dims.
///
/// `dyn_slots` (ascending) are the slots whose extent is read from the
/// third argument at entry instead of baked from `dims`; their entries in
/// `dims` are never consulted.
#[allow(clippy::too_many_arguments)]
fn codegen(
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    is_sparse: &[bool],
    dims: &[usize; 26],
    dyn_slots: &[u8],
    reduce: Reduce,
    combine: Combine,
    load_maps: &[&[UnaryMap]],
    store_map: &[UnaryMap],
) -> Result<(JITModule, extern "C" fn(*const *const u8, *const *mut u8, *const usize)), JitError> {
    let any_sparse = is_sparse.iter().any(|&c| c);

    // A load map on a sparse input would be applied only to the *stored*
    // entries; every skipped structural zero would contribute 0 where the
    // dense-equivalent contributes f(0). Reject rather than silently
    // restrict this to zero-preserving maps.
    if is_sparse.iter().zip(load_maps).any(|(&sp, m)| sp && !m.is_empty()) {
        return Err(JitError::Unsupported("a load map on a sparse input skips structural zeros"));
    }
    // The store map is applied once, to a finished accumulator. That only
    // exists in the register-accumulator nest below; the general path
    // read-modify-writes the output, where the map would compound.
    let register_acc = !any_sparse && outputs.len() == 1;
    if !store_map.is_empty() && !register_acc {
        return Err(JitError::Unsupported(
            "a store map requires all-dense inputs and exactly one output",
        ));
    }

    // Sparse row iteration skips structural zeros, which is only sound for
    // sum-of-products (a zero factor annihilates the product and adding 0
    // is a no-op). Under any other semiring a missing entry is a real 0
    // that must still compete in the reduction, and this backend has no
    // dense-fallback addressing for CSR — use the VM for those.
    if any_sparse && !(reduce == Reduce::Sum && combine == Combine::Mul) {
        return Err(JitError::Unsupported(
            "sparse inputs require sum-of-products (Reduce::Sum, Combine::Mul)",
        ));
    }

    // Sparse axes per input: the last slot is the stored column; all preceding
    // slots form the (row-major) compound row index.
    let sparse_axes: Vec<Option<SparseAxes>> = inputs
        .iter()
        .zip(is_sparse)
        .map(|(pat, &sp)| {
            if sp {
                let n = pat.len();
                Some(SparseAxes { leading: pat[..n - 1].to_vec(), col: pat[n - 1] })
            } else {
                None
            }
        })
        .collect();

    // Free (in some output) vs contracted (inputs only) slots, first-appearance.
    let mut in_output = [false; 26];
    for out in outputs {
        for &s in out {
            in_output[s as usize] = true;
        }
    }
    let mut order = Vec::new();
    let mut seen = [false; 26];
    for pat in inputs.iter().chain(outputs.iter()) {
        for &s in pat {
            if !seen[s as usize] {
                seen[s as usize] = true;
                order.push(s);
            }
        }
    }
    let free: Vec<u8> = order.iter().copied().filter(|&s| in_output[s as usize]).collect();
    let contracted: Vec<u8> =
        order.iter().copied().filter(|&s| !in_output[s as usize]).collect();

    // Sparse-aware plan (used whenever any input is sparse, or for multi-output).
    let plan = schedule(inputs, outputs, &sparse_axes);

    // Which inputs are value-sourced from a sparse loop in the plan?
    let mut sparse_covered = vec![false; inputs.len()];
    for op in &plan {
        if let LoopOp::Sparse { input_idx, .. } = op {
            sparse_covered[*input_idx] = true;
        }
    }
    // Every sparse input must be covered, else we'd need a (binary-search)
    // random access we don't generate.
    for (i, &sp) in is_sparse.iter().enumerate() {
        if sp && !sparse_covered[i] {
            return Err(JitError::Unsupported(
                "a sparse input's column index could not be reached by row iteration",
            ));
        }
    }

    let mut module = new_module();
    let ptr_ty = module.target_config().pointer_type();
    let ptr_bytes = ptr_ty.bytes() as i32;

    let mut ctx = module.make_context();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // ins:  *const *const u8
    sig.params.push(AbiParam::new(ptr_ty)); // outs: *const *mut  u8
    sig.params.push(AbiParam::new(ptr_ty)); // dims: *const usize
    ctx.func.signature = sig;

    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

        // One index variable per slot, an accumulator register, and a value
        // register per sparse input. Unused variables are harmless.
        let vars: [Variable; 26] = std::array::from_fn(|_| b.declare_var(ptr_ty));
        let acc = b.declare_var(types::F32);
        let val_vars: Vec<Option<Variable>> = is_sparse
            .iter()
            .map(|&sp| if sp { Some(b.declare_var(types::F32)) } else { None })
            .collect();

        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let ins_ptr = b.block_params(entry)[0];
        let outs_ptr = b.block_params(entry)[1];
        let dims_ptr = b.block_params(entry)[2];

        // Per-slot extents: a constant for a static axis, a value read once
        // from the dims argument for a dynamic one. Read in the entry
        // block, which dominates the whole nest.
        let mut slot_dim: [Dim; 26] = std::array::from_fn(|s| Dim::Const(dims[s] as i64));
        for (k, &s) in dyn_slots.iter().enumerate() {
            let v = b.ins().load(ptr_ty, MemFlags::trusted(), dims_ptr, k as i32 * ptr_bytes);
            slot_dim[s as usize] = Dim::Var(v);
        }
        let axis_dims =
            |pat: &[u8]| -> Vec<Dim> { pat.iter().map(|&s| slot_dim[s as usize]).collect() };

        // Load base pointer(s) per input. Each input's layout decides how
        // many pointer-slots it consumes; we just walk them in order.
        let mut bases: Vec<InputBase> = Vec::with_capacity(inputs.len());
        let mut slot = 0i32;
        for (i, &sp) in is_sparse.iter().enumerate() {
            if sp {
                let layout: Box<dyn RowLayout> = Box::new(CsrRowLayout);
                let n = layout.n_ptr_slots();
                let input_bases: Vec<Value> = (0..n)
                    .map(|j| {
                        b.ins().load(
                            ptr_ty,
                            MemFlags::trusted(),
                            ins_ptr,
                            (slot + j as i32) * ptr_bytes,
                        )
                    })
                    .collect();
                bases.push(InputBase::Row { layout, bases: input_bases });
                slot += n as i32;
            } else {
                let base =
                    b.ins().load(ptr_ty, MemFlags::trusted(), ins_ptr, slot * ptr_bytes);
                let ad = axis_dims(&inputs[i]);
                let layout = DenseLayout::new(&mut b, ptr_ty, &ad);
                bases.push(InputBase::Dense { layout, base });
                slot += 1;
            }
        }
        let out_bases: Vec<Value> = (0..outputs.len())
            .map(|i| b.ins().load(ptr_ty, MemFlags::trusted(), outs_ptr, i as i32 * ptr_bytes))
            .collect();
        // Row-major dense layouts for outputs (outputs are always dense).
        let out_layouts: Vec<DenseLayout> = outputs
            .iter()
            .map(|p| {
                let ad = axis_dims(p);
                DenseLayout::new(&mut b, ptr_ty, &ad)
            })
            .collect();

        if register_acc {
            // ── All-dense single output: register accumulator. ──
            // for free: { acc = identity; for contracted { acc = acc ⊕ contrib }; out = acc }
            let mut free_loops = Vec::new();
            for &s in &free {
                free_loops.push((
                    vars[s as usize],
                    open_dense_loop(&mut b, ptr_ty, vars[s as usize], slot_dim[s as usize]),
                ));
            }
            let ident = b.ins().f32const(reduce.identity::<f32>());
            b.def_var(acc, ident);
            let mut c_loops = Vec::new();
            for &s in &contracted {
                c_loops.push((
                    vars[s as usize],
                    open_dense_loop(&mut b, ptr_ty, vars[s as usize], slot_dim[s as usize]),
                ));
            }

            let contrib = emit_contribution(
                &mut b, ptr_ty, combine, inputs, &bases, &val_vars, &vars, load_maps,
            );
            let cur = b.use_var(acc);
            let folded = emit_reduce(&mut b, reduce, cur, contrib);
            b.def_var(acc, folded);

            for (iv, (h, e)) in c_loops.into_iter().rev() {
                close_loop(&mut b, iv, h, e);
            }
            let acc_val = b.use_var(acc);
            let acc_val = emit_map_chain(&mut b, store_map, acc_val);
            let idx_vals: Vec<Value> =
                outputs[0].iter().map(|&s| b.use_var(vars[s as usize])).collect();
            let addr = out_layouts[0].emit_elem_addr(&mut b, ptr_ty, out_bases[0], &idx_vals);
            b.ins().store(MemFlags::trusted(), acc_val, addr, 0);
            for (iv, (h, e)) in free_loops.into_iter().rev() {
                close_loop(&mut b, iv, h, e);
            }
        } else {
            // ── General path: sparse-aware loop nest, read-modify-write. ──
            let mut opened: Vec<(Variable, Block, Block)> = Vec::with_capacity(plan.len());
            for op in &plan {
                match op {
                    LoopOp::Dense { slot } => {
                        let (h, e) = open_dense_loop(
                            &mut b,
                            ptr_ty,
                            vars[*slot as usize],
                            slot_dim[*slot as usize],
                        );
                        opened.push((vars[*slot as usize], h, e));
                    }
                    LoopOp::Sparse { input_idx, leading, col_slot } => {
                        let strides = leading_strides(&mut b, ptr_ty, leading, &slot_dim);
                        let compound =
                            emit_compound_row(&mut b, ptr_ty, leading, &strides, &vars);
                        let (layout, input_bases) = match &bases[*input_idx] {
                            InputBase::Row { layout, bases } => {
                                (layout.as_ref(), bases.as_slice())
                            }
                            InputBase::Dense { .. } => {
                                unreachable!("sparse op on dense input")
                            }
                        };
                        let (ei, h, e) = layout.emit_row_open(
                            &mut b,
                            ptr_ty,
                            input_bases,
                            compound,
                            vars[*col_slot as usize],
                            val_vars[*input_idx].expect("sparse input has value var"),
                        );
                        opened.push((ei, h, e));
                    }
                }
            }

            let contrib = emit_contribution(
                &mut b, ptr_ty, combine, inputs, &bases, &val_vars, &vars, load_maps,
            );
            for (oi, pattern) in outputs.iter().enumerate() {
                let idx_vals: Vec<Value> =
                    pattern.iter().map(|&s| b.use_var(vars[s as usize])).collect();
                let addr =
                    out_layouts[oi].emit_elem_addr(&mut b, ptr_ty, out_bases[oi], &idx_vals);
                let cur = b.ins().load(types::F32, MemFlags::trusted(), addr, 0);
                let folded = emit_reduce(&mut b, reduce, cur, contrib);
                b.ins().store(MemFlags::trusted(), folded, addr, 0);
            }

            for (iv, h, e) in opened.into_iter().rev() {
                close_loop(&mut b, iv, h, e);
            }
        }

        b.ins().return_(&[]);
        b.finalize();
    }

    let id = module
        .declare_function("einsum", Linkage::Export, &ctx.func.signature)
        .expect("declare_function");
    module.define_function(id, &mut ctx).expect("define_function");
    module.clear_context(&mut ctx);
    module.finalize_definitions().expect("finalize_definitions");

    let code = module.get_finalized_function(id);
    // SAFETY: `code` is a finalized function with exactly the declared
    // signature (two pointer params, no return); `module` is returned and
    // kept alive by the caller so the code stays mapped.
    let func = unsafe {
        mem::transmute::<
            *const u8,
            extern "C" fn(*const *const u8, *const *mut u8, *const usize),
        >(code)
    };
    Ok((module, func))
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::NDIndex;

    fn dense(shape: Vec<usize>, data: &[f32]) -> Dense<f32> {
        let mut d = Dense::<f32>::zeros(shape);
        d.fill_from(data);
        d
    }

    fn d(x: &Dense<f32>) -> JitInput<'_> {
        JitInput::Dense(x)
    }

    #[test]
    fn matmul() {
        let a = dense(vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = dense(vec![3, 2], &[7., 8., 9., 10., 11., 12.]);
        let jit = EinsumF32Jit::compile("ab,bc->ac", &[d(&a), d(&b)], &[vec![2, 2]]).unwrap();
        let mut c = Dense::<f32>::zeros(vec![2, 2]);
        jit.run(&[d(&a), d(&b)], &mut [&mut c]);
        assert_eq!(c.data, vec![58., 64., 139., 154.]);
    }

    /// A small deterministic LCG so the dynamic-vs-static comparison below
    /// sees awkward values (`±0.0`, `±∞`, denormals) rather than pretty ones.
    fn awkward(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                match (s >> 60) & 7 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    4 => 1.0e-38,
                    _ => ((s >> 20) as i32 as f32) / 7.0e5,
                }
            })
            .collect()
    }

    fn dyn_compile<'a>(
        spec: &str,
        inputs: &[JitInput<'a>],
        out: &[Vec<usize>],
        dynamic: &[char],
    ) -> Result<EinsumF32Jit, JitError> {
        EinsumF32Jit::compile_reduce_mapped_dyn(
            spec,
            Reduce::Sum,
            Combine::Mul,
            &[],
            &[],
            inputs,
            out,
            dynamic,
        )
    }

    /// One program, compiled once, run at several extents of the dynamic
    /// axis — each result **bit-identical** to a program specialized to that
    /// extent. This is the whole promise of a dynamic axis: it changes what
    /// is a constant, never what arithmetic happens.
    #[test]
    fn dynamic_axis_is_bit_identical_to_static() {
        // `b` (the contracted axis) and `c` (a free axis, so it also drives
        // the output's stride) are both runtime arguments.
        let seeds = [3u64, 9, 17];
        let dynamic = ['b', 'c'];
        let mut shared: Option<EinsumF32Jit> = None;
        for (nb, nc) in [(1usize, 1usize), (5, 3), (7, 4), (16, 9)] {
            let a = dense(vec![2, nb], &awkward(2 * nb, seeds[0]));
            let bb = dense(vec![nb, nc], &awkward(nb * nc, seeds[1]));
            let ins = [d(&a), d(&bb)];
            let oshape = vec![vec![2, nc]];

            let jit = shared.get_or_insert_with(|| {
                dyn_compile("ab,bc->ac", &ins, &oshape, &dynamic).unwrap()
            });
            let mut got = Dense::<f32>::zeros(vec![2, nc]);
            jit.run(&ins, &mut [&mut got]);

            let stat = EinsumF32Jit::compile("ab,bc->ac", &ins, &oshape).unwrap();
            let mut want = Dense::<f32>::zeros(vec![2, nc]);
            stat.run(&ins, &mut [&mut want]);

            let g: Vec<u32> = got.data.iter().map(|v| v.to_bits()).collect();
            let w: Vec<u32> = want.data.iter().map(|v| v.to_bits()).collect();
            assert_eq!(g, w, "dynamic vs static bits at ({nb}, {nc}) — seed {}", seeds[2]);
        }
    }

    /// Dynamic axes also work under the general (read-modify-write) nest
    /// and with a batch axis whose extent drives every leading stride.
    #[test]
    fn dynamic_batch_axis_matches_static() {
        for nb in [1usize, 2, 5] {
            let a = dense(vec![nb, 3, 4], &awkward(nb * 12, 11));
            let bb = dense(vec![nb, 4, 2], &awkward(nb * 8, 23));
            let ins = [d(&a), d(&bb)];
            let oshape = vec![vec![nb, 3, 2]];
            let jit = dyn_compile("aij,ajk->aik", &ins, &oshape, &['a']).unwrap();
            let mut got = Dense::<f32>::zeros(vec![nb, 3, 2]);
            jit.run(&ins, &mut [&mut got]);
            let stat = EinsumF32Jit::compile("aij,ajk->aik", &ins, &oshape).unwrap();
            let mut want = Dense::<f32>::zeros(vec![nb, 3, 2]);
            stat.run(&ins, &mut [&mut want]);
            assert_eq!(
                got.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn dynamic_axis_rejected_when_unreadable() {
        let a = dense(vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = dense(vec![3, 2], &[7., 8., 9., 10., 11., 12.]);
        let ins = [d(&a), d(&b)];
        // Not a lowercase letter.
        assert!(matches!(
            dyn_compile("ab,bc->ac", &ins, &[vec![2, 2]], &['Z']),
            Err(JitError::Unsupported("a dynamic axis must be a lowercase letter"))
        ));
        // A letter no input carries has nothing to read its extent from.
        assert!(matches!(
            dyn_compile("ab,bc->ac", &ins, &[vec![2, 2]], &['z']),
            Err(JitError::Unsupported("a dynamic axis must occur in at least one input"))
        ));
    }

    /// A run whose inputs disagree about a dynamic axis is a caller bug,
    /// caught with the same assertion a static mismatch gets.
    #[test]
    #[should_panic(expected = "axis 0 ('b')")]
    fn dynamic_axis_inconsistent_between_inputs_panics() {
        let a = dense(vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = dense(vec![3, 2], &[7., 8., 9., 10., 11., 12.]);
        let jit = dyn_compile("ab,bc->ac", &[d(&a), d(&b)], &[vec![2, 2]], &['b']).unwrap();
        let wrong = dense(vec![4, 2], &[0.; 8]);
        let mut c = Dense::<f32>::zeros(vec![2, 2]);
        jit.run(&[d(&a), d(&wrong)], &mut [&mut c]);
    }

    #[test]
    fn transpose() {
        let a = dense(vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        let jit = EinsumF32Jit::compile("ab->ba", &[d(&a)], &[vec![3, 2]]).unwrap();
        let mut t = Dense::<f32>::zeros(vec![3, 2]);
        jit.run(&[d(&a)], &mut [&mut t]);
        assert_eq!(t.data, vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn dot_to_scalar() {
        let a = dense(vec![4], &[1., 2., 3., 4.]);
        let b = dense(vec![4], &[5., 6., 7., 8.]);
        let jit = EinsumF32Jit::compile("i,i->", &[d(&a), d(&b)], &[vec![]]).unwrap();
        let mut s = Dense::<f32>::zeros(vec![]);
        jit.run(&[d(&a), d(&b)], &mut [&mut s]);
        assert_eq!(s.get(&[]), 70.0);
    }

    #[test]
    fn trace() {
        let m = dense(vec![3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let jit = EinsumF32Jit::compile("aa->", &[d(&m)], &[vec![]]).unwrap();
        let mut s = Dense::<f32>::zeros(vec![]);
        jit.run(&[d(&m)], &mut [&mut s]);
        assert_eq!(s.get(&[]), 15.0);
    }

    #[test]
    fn three_input_chain() {
        let a = dense(vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = dense(vec![3, 4], &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]);
        let c = dense(vec![4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let jit = EinsumF32Jit::compile("ab,bc,cd->ad", &[d(&a), d(&b), d(&c)], &[vec![2, 2]])
            .unwrap();
        let mut out = Dense::<f32>::zeros(vec![2, 2]);
        jit.run(&[d(&a), d(&b), d(&c)], &mut [&mut out]);
        assert_eq!(out.data, vec![22., 28., 49., 64.]);
    }

    #[test]
    fn multi_output() {
        let a = dense(vec![2, 2], &[1., 2., 3., 4.]);
        let b = dense(vec![2, 2], &[5., 6., 7., 8.]);
        let jit = EinsumF32Jit::compile(
            "ab,bc->ac,ca",
            &[d(&a), d(&b)],
            &[vec![2, 2], vec![2, 2]],
        )
        .unwrap();
        let mut ac = Dense::<f32>::zeros(vec![2, 2]);
        let mut ca = Dense::<f32>::zeros(vec![2, 2]);
        jit.run(&[d(&a), d(&b)], &mut [&mut ac, &mut ca]);
        assert_eq!(ac.data, vec![19., 22., 43., 50.]);
        assert_eq!(ca.data, vec![19., 43., 22., 50.]);
    }

    #[test]
    fn reused_across_executions() {
        let b = dense(vec![2, 2], &[1., 2., 3., 4.]);
        let a0 = dense(vec![2, 2], &[1., 0., 0., 1.]);
        let jit = EinsumF32Jit::compile("ab,bc->ac", &[d(&a0), d(&b)], &[vec![2, 2]]).unwrap();
        for k in 1..4u32 {
            let f = k as f32;
            let a = dense(vec![2, 2], &[f, 0., 0., f]);
            let mut c = Dense::<f32>::zeros(vec![2, 2]);
            jit.run(&[d(&a), d(&b)], &mut [&mut c]);
            assert_eq!(c.data, vec![f, f * 2., f * 3., f * 4.]);
        }
    }

    // ── Generalized reductions ──

    #[test]
    fn reduce_max_product_matches_vm() {
        // Register-accumulator path (all-dense single output), mixed signs.
        let a = dense(vec![2, 3], &[1., -2., 3., -4., 5., 0.]);
        let b = dense(vec![3, 2], &[-7., 8., 9., -10., 11., 12.]);
        let mut got = Dense::<f32>::zeros(vec![2, 2]);
        einsum_jit_reduce(
            "ab,bc->ac",
            Reduce::Max,
            Combine::Mul,
            &[d(&a), d(&b)],
            &mut [&mut got],
        )
        .unwrap();

        let mut expect = Dense::<f32>::zeros(vec![2, 2]);
        crate::einsum::einsum_reduce::<f32>(
            "ab,bc->ac",
            Reduce::Max,
            Combine::Mul,
            &[&a as &dyn crate::tensor::NDIndex<f32>, &b],
            &mut [&mut expect as &mut dyn crate::tensor::NDIndex<f32>],
        )
        .unwrap();
        assert_eq!(got.data, expect.data);
    }

    #[test]
    fn reduce_min_plus_matches_vm() {
        let a = dense(vec![2, 2], &[0., 3., 7., 0.]);
        let b = dense(vec![2, 2], &[0., 4., 1., 0.]);
        let mut got = Dense::<f32>::zeros(vec![2, 2]);
        einsum_jit_reduce(
            "ij,jk->ik",
            Reduce::Min,
            Combine::Add,
            &[d(&a), d(&b)],
            &mut [&mut got],
        )
        .unwrap();
        assert_eq!(got.data, vec![0., 3., 1., 0.]);
    }

    #[test]
    fn reduce_multi_output_rmw_path() {
        // Multi-output forces the read-modify-write codegen path; outputs
        // must land on the identity prefill correctly for max.
        let a = dense(vec![2, 2], &[1., -2., 3., 4.]);
        let b = dense(vec![2, 2], &[5., 6., -7., 8.]);
        let mut ac = Dense::<f32>::zeros(vec![2, 2]);
        let mut ca = Dense::<f32>::zeros(vec![2, 2]);
        einsum_jit_reduce(
            "ab,bc->ac,ca",
            Reduce::Max,
            Combine::Mul,
            &[d(&a), d(&b)],
            &mut [&mut ac, &mut ca],
        )
        .unwrap();

        let mut vac = Dense::<f32>::zeros(vec![2, 2]);
        let mut vca = Dense::<f32>::zeros(vec![2, 2]);
        crate::einsum::einsum_reduce::<f32>(
            "ab,bc->ac,ca",
            Reduce::Max,
            Combine::Mul,
            &[&a as &dyn crate::tensor::NDIndex<f32>, &b],
            &mut [
                &mut vac as &mut dyn crate::tensor::NDIndex<f32>,
                &mut vca,
            ],
        )
        .unwrap();
        assert_eq!(ac.data, vac.data);
        assert_eq!(ca.data, vca.data);
    }

    #[test]
    fn reduce_empty_range_yields_identity() {
        let a = dense(vec![2, 0], &[]);
        let mut m = Dense::<f32>::zeros(vec![2]);
        einsum_jit_reduce("iq->i", Reduce::Max, Combine::Mul, &[d(&a)], &mut [&mut m])
            .unwrap();
        assert_eq!(m.data, vec![f32::NEG_INFINITY; 2]);
    }

    // ── Load / store maps ──

    /// A mapped contraction must equal applying the same unaries in Rust,
    /// **bit for bit** — that is the whole premise of [`UnaryMap`].
    #[test]
    fn load_and_store_maps_match_scalar_rust() {
        let a = dense(vec![2, 3], &[1., -2., 0., -0.0, 4., -5.]);
        let b = dense(vec![3], &[-1., 2., -3.]);
        let maps: [&[UnaryMap]; 2] = [&[UnaryMap::Relu], &[UnaryMap::Neg, UnaryMap::Abs]];
        let mut out = Dense::<f32>::zeros(vec![2]);
        let jit = EinsumF32Jit::compile_reduce_mapped(
            "ab,b->a",
            Reduce::Sum,
            Combine::Mul,
            &maps,
            &[UnaryMap::Sqrt],
            &[d(&a), d(&b)],
            &[vec![2]],
        )
        .unwrap();
        jit.run(&[d(&a), d(&b)], &mut [&mut out]);

        let relu = |x: f32| if x > 0.0 { x } else { 0.0 };
        for i in 0..2 {
            let mut acc = 0.0f32;
            for j in 0..3 {
                acc += relu(a.data[i * 3 + j]) * (-b.data[j]).abs();
            }
            assert_eq!(out.data[i].to_bits(), acc.sqrt().to_bits(), "row {i}");
        }
    }

    /// `relu` must be the `if x > 0.0 { x } else { 0.0 }` form: `-0.0` and
    /// `NaN` both come out as `+0.0`, which `fmax(x, 0.0)` would not give.
    #[test]
    fn relu_map_normalizes_negative_zero_and_nan() {
        let a = dense(vec![1, 2], &[-0.0, f32::NAN]);
        let b = dense(vec![2], &[1.0, 1.0]);
        let mut out = Dense::<f32>::zeros(vec![1]);
        let jit = EinsumF32Jit::compile_reduce_mapped(
            "ab,b->a",
            Reduce::Sum,
            Combine::Mul,
            &[&[UnaryMap::Relu], &[]],
            &[],
            &[d(&a), d(&b)],
            &[vec![1]],
        )
        .unwrap();
        jit.run(&[d(&a), d(&b)], &mut [&mut out]);
        assert_eq!(out.data[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn maps_rejected_where_they_would_be_unsound() {
        let a = Csr::<u32, f32>::from_coo(3, &mut vec![(0, 1, 1.0), (1, 2, 1.0)]);
        let x = dense(vec![3, 2], &[1., 2., 3., 4., 5., 6.]);
        // A load map on a sparse input would silently skip `f(0)` on every
        // structural zero.
        let res = EinsumF32Jit::compile_reduce_mapped(
            "ab,bc->ac",
            Reduce::Sum,
            Combine::Mul,
            &[&[UnaryMap::Neg], &[]],
            &[],
            &[JitInput::Csr(&a), d(&x)],
            &[vec![3, 2]],
        );
        assert!(matches!(res, Err(JitError::Unsupported(_))));

        // A store map needs the register accumulator; the sparse nest
        // read-modify-writes and would compound the map.
        let res = EinsumF32Jit::compile_reduce_mapped(
            "ab,bc->ac",
            Reduce::Sum,
            Combine::Mul,
            &[],
            &[UnaryMap::Sqrt],
            &[JitInput::Csr(&a), d(&x)],
            &[vec![3, 2]],
        );
        assert!(matches!(res, Err(JitError::Unsupported(_))));

        // One chain per input, or none at all.
        let y = dense(vec![2, 2], &[1., 2., 3., 4.]);
        let res = EinsumF32Jit::compile_reduce_mapped(
            "ab,bc->ac",
            Reduce::Sum,
            Combine::Mul,
            &[&[UnaryMap::Abs]],
            &[],
            &[d(&y), d(&y)],
            &[vec![2, 2]],
        );
        assert!(matches!(res, Err(JitError::Unsupported(_))));
    }

    #[test]
    fn reduce_sparse_input_unsupported() {
        let a = Csr::<u32, f32>::from_coo(3, &mut vec![(0, 1, 1.0), (1, 2, 1.0)]);
        let x = dense(vec![3, 2], &[1., 2., 3., 4., 5., 6.]);
        let res = EinsumF32Jit::compile_reduce(
            "ab,bc->ac",
            Reduce::Max,
            Combine::Mul,
            &[JitInput::Csr(&a), d(&x)],
            &[vec![3, 2]],
        );
        match res {
            Err(JitError::Unsupported(_)) => {}
            Err(e) => panic!("expected Unsupported, got {e:?}"),
            Ok(_) => panic!("expected Unsupported, got Ok"),
        }
    }

    // ── CSR-sparse inputs ──

    /// 3×3 cyclic permutation: edges (0,1),(1,2),(2,0) with value 1.
    fn cyclic_perm() -> Csr<u32, f32> {
        Csr::<u32, f32>::from_coo(3, &mut vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    }

    #[test]
    fn csr_times_dense() {
        let a = cyclic_perm();
        let x = dense(vec![3, 2], &[1., 2., 3., 4., 5., 6.]);
        let jit =
            EinsumF32Jit::compile("ab,bc->ac", &[JitInput::Csr(&a), d(&x)], &[vec![3, 2]]).unwrap();
        let mut y = Dense::<f32>::zeros(vec![3, 2]);
        jit.run(&[JitInput::Csr(&a), d(&x)], &mut [&mut y]);
        // row i picks up row (i+1 mod 3) of x.
        assert_eq!(y.data, vec![3., 4., 5., 6., 1., 2.]);
    }

    #[test]
    fn csr_times_csr() {
        let a = cyclic_perm();
        let b = cyclic_perm();
        let jit = EinsumF32Jit::compile(
            "ab,bc->ac",
            &[JitInput::Csr(&a), JitInput::Csr(&b)],
            &[vec![3, 3]],
        )
        .unwrap();
        let mut c = Dense::<f32>::zeros(vec![3, 3]);
        jit.run(&[JitInput::Csr(&a), JitInput::Csr(&b)], &mut [&mut c]);
        // shift twice: edges (0,2),(1,0),(2,1).
        assert_eq!(c.data, vec![0., 0., 1., 1., 0., 0., 0., 1., 0.]);
    }

    #[test]
    fn csr_dense_matches_einsum_vm() {
        use crate::einsum::einsum;

        // Random-ish but structured sparse A (5×5) and dense X (5×3).
        let a = Csr::<u32, f32>::from_coo(
            5,
            &mut vec![
                (0, 1, 2.0),
                (0, 4, -1.0),
                (1, 1, 3.0),
                (2, 0, 1.5),
                (2, 3, 0.5),
                (4, 2, -2.0),
            ],
        );
        let mut state = 0x1234_5678u64;
        let mut x = Dense::<f32>::zeros(vec![5, 3]);
        for v in x.data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let jit =
            EinsumF32Jit::compile("ab,bc->ac", &[JitInput::Csr(&a), d(&x)], &[vec![5, 3]]).unwrap();
        let mut jit_out = Dense::<f32>::zeros(vec![5, 3]);
        jit.run(&[JitInput::Csr(&a), d(&x)], &mut [&mut jit_out]);

        let mut vm_out = Dense::<f32>::zeros(vec![5, 3]);
        einsum::<f32>(
            "ab,bc->ac",
            &[&a as &dyn NDIndex<f32>, &x],
            &mut [&mut vm_out as &mut dyn NDIndex<f32>],
        )
        .unwrap();

        for (j, v) in jit_out.data.iter().zip(vm_out.data.iter()) {
            assert!((j - v).abs() <= 1e-5 * (1.0 + v.abs()), "jit {j} vs vm {v}");
        }
    }

    #[test]
    fn rectangular_csr_input() {
        // A non-square Csr (2×3) passed via the JitInput::Csr path.
        let a = Csr::<u32, f32>::from_parts(
            vec![2, 3],
            vec![0, 2, 3],
            vec![1, 2, 0],
            vec![2.0, 3.0, 1.0],
        );
        let a_dense = dense(vec![2, 3], &[0., 2., 3., 1., 0., 0.]);
        let x = dense(vec![3, 4], &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

        let sp =
            EinsumF32Jit::compile("ab,bc->ac", &[JitInput::Csr(&a), d(&x)], &[vec![2, 4]]).unwrap();
        let mut y_sp = Dense::<f32>::zeros(vec![2, 4]);
        sp.run(&[JitInput::Csr(&a), d(&x)], &mut [&mut y_sp]);

        let de = EinsumF32Jit::compile("ab,bc->ac", &[d(&a_dense), d(&x)], &[vec![2, 4]]).unwrap();
        let mut y_de = Dense::<f32>::zeros(vec![2, 4]);
        de.run(&[d(&a_dense), d(&x)], &mut [&mut y_de]);

        assert_eq!(y_sp.data, y_de.data);
    }

    #[test]
    fn rectangular_sparse_times_dense() {
        // Non-square sparse A (2×3): A[0,1]=2, A[0,2]=3, A[1,0]=1.
        let row_ptr = vec![0usize, 2, 3];
        let col_idx = vec![1u32, 2, 0];
        let values = vec![2.0f32, 3.0, 1.0];
        let a = SparseView::new(vec![2, 3], &row_ptr, &col_idx, &values);
        let a_dense = dense(vec![2, 3], &[0., 2., 3., 1., 0., 0.]);
        let x = dense(vec![3, 4], &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

        let sp =
            EinsumF32Jit::compile("ab,bc->ac", &[JitInput::Sparse(a.clone()), d(&x)], &[vec![2, 4]])
                .unwrap();
        let mut y_sp = Dense::<f32>::zeros(vec![2, 4]);
        sp.run(&[JitInput::Sparse(a), d(&x)], &mut [&mut y_sp]);

        // Cross-check against the dense-equivalent JIT (engine already
        // validated against the VM for dense).
        let de = EinsumF32Jit::compile("ab,bc->ac", &[d(&a_dense), d(&x)], &[vec![2, 4]]).unwrap();
        let mut y_de = Dense::<f32>::zeros(vec![2, 4]);
        de.run(&[d(&a_dense), d(&x)], &mut [&mut y_de]);

        assert_eq!(y_sp.data, y_de.data);
    }

    #[test]
    fn batched_sparse_times_dense() {
        // 3D sparse A "bij" with shape [2,2,2]; dense B "bjk" [2,2,3];
        // out "bik" [2,2,3]. Compound row index = b*2 + i.
        let row_ptr = vec![0usize, 2, 3, 4, 5];
        let col_idx = vec![0u32, 1, 1, 0, 0];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let a = SparseView::new(vec![2, 2, 2], &row_ptr, &col_idx, &values);
        // Dense equivalent of A.
        let a_dense = dense(vec![2, 2, 2], &[1., 2., 0., 3., 4., 0., 5., 0.]);
        let bb = dense(
            vec![2, 2, 3],
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );

        let sp = EinsumF32Jit::compile(
            "bij,bjk->bik",
            &[JitInput::Sparse(a.clone()), d(&bb)],
            &[vec![2, 2, 3]],
        )
        .unwrap();
        let mut out_sp = Dense::<f32>::zeros(vec![2, 2, 3]);
        sp.run(&[JitInput::Sparse(a), d(&bb)], &mut [&mut out_sp]);

        let de = EinsumF32Jit::compile("bij,bjk->bik", &[d(&a_dense), d(&bb)], &[vec![2, 2, 3]])
            .unwrap();
        let mut out_de = Dense::<f32>::zeros(vec![2, 2, 3]);
        de.run(&[d(&a_dense), d(&bb)], &mut [&mut out_de]);

        assert_eq!(out_sp.data, out_de.data);
        // Spot-check one hand value: out[0,1,:] = 3 * B[0,1,:] = [12,15,18].
        assert_eq!(&out_sp.data[3..6], &[12., 15., 18.]);
    }

    #[test]
    fn unsupported_sparse_pattern_errs() {
        // Both operands sparse and sharing the contracted column index 'b':
        // the second's column can't be reached by row iteration.
        let a = cyclic_perm();
        let b = cyclic_perm();
        let err = EinsumF32Jit::compile(
            "ab,cb->ac",
            &[JitInput::Csr(&a), JitInput::Csr(&b)],
            &[vec![3, 3]],
        )
        .err();
        assert!(matches!(err, Some(JitError::Unsupported(_))), "got {err:?}");
    }

    /// Cross-check the all-dense paths against the interpreted VM.
    #[test]
    fn matches_vm_on_random() {
        use crate::einsum::einsum_homogenous;

        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut rand_dense = |shape: Vec<usize>| {
            let n: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
            let data: Vec<f32> = (0..n).map(|_| next()).collect();
            dense(shape, &data)
        };

        let cases: &[(&str, &[&[usize]], &[&[usize]])] = &[
            ("ab,bc->ac", &[&[5, 7], &[7, 3]], &[&[5, 3]]),
            ("abc,acd->abd", &[&[2, 4, 3], &[2, 3, 5]], &[&[2, 4, 5]]),
            ("ab->ba", &[&[6, 4]], &[&[4, 6]]),
            ("ij,ij->", &[&[4, 5], &[4, 5]], &[&[]]),
            ("ab,bc,cd->ad", &[&[2, 3], &[3, 4], &[4, 2]], &[&[2, 2]]),
            ("ab,bc->ac,ca", &[&[3, 4], &[4, 3]], &[&[3, 3], &[3, 3]]),
        ];

        for (spec, in_shapes, out_shapes) in cases {
            let ins: Vec<Dense<f32>> = in_shapes.iter().map(|s| rand_dense(s.to_vec())).collect();
            let jit_inputs: Vec<JitInput> = ins.iter().map(d).collect();
            let in_refs: Vec<&Dense<f32>> = ins.iter().collect();

            let jit = EinsumF32Jit::compile(
                spec,
                &jit_inputs,
                &out_shapes.iter().map(|s| s.to_vec()).collect::<Vec<_>>(),
            )
            .unwrap();

            let mut jit_outs: Vec<Dense<f32>> =
                out_shapes.iter().map(|s| Dense::<f32>::zeros(s.to_vec())).collect();
            let mut jit_refs: Vec<&mut Dense<f32>> = jit_outs.iter_mut().collect();
            jit.run(&jit_inputs, &mut jit_refs);

            let mut vm_outs: Vec<Dense<f32>> =
                out_shapes.iter().map(|s| Dense::<f32>::zeros(s.to_vec())).collect();
            let mut vm_refs: Vec<&mut Dense<f32>> = vm_outs.iter_mut().collect();
            einsum_homogenous::<f32, _, _>(spec, &in_refs, &mut vm_refs).unwrap();

            for (oi, (j, v)) in jit_outs.iter().zip(vm_outs.iter()).enumerate() {
                for (k, (jv, vv)) in j.data.iter().zip(v.data.iter()).enumerate() {
                    assert!(
                        (jv - vv).abs() <= 1e-4 * (1.0 + vv.abs()),
                        "spec {spec} output {oi} elem {k}: jit {jv} vs vm {vv}"
                    );
                }
            }
        }
    }

    #[test]
    fn spec_errors_surface() {
        let a = dense(vec![2, 3], &[0.; 6]);
        let b = dense(vec![3, 2], &[0.; 6]);
        let b_bad = dense(vec![4, 2], &[0.; 8]);
        assert_eq!(
            EinsumF32Jit::compile("ab,bc", &[d(&a), d(&b)], &[vec![2, 2]]).err(),
            Some(JitError::Spec(InvalidSpec::MissingArrow)),
        );
        assert_eq!(
            EinsumF32Jit::compile("ab,bc->az", &[d(&a), d(&b)], &[vec![2, 2]]).err(),
            Some(JitError::Spec(InvalidSpec::UnboundOutputIndex { index: 'z' })),
        );
        assert_eq!(
            EinsumF32Jit::compile("ab,bc->ac", &[d(&a), d(&b_bad)], &[vec![2, 2]]).err(),
            Some(JitError::Spec(InvalidSpec::DimensionMismatch {
                index: 'b',
                expected: 3,
                got: 4
            })),
        );
    }
}
