//! Binding and execution.
//!
//! [`run`] takes a [`Checked`] program, resolves names against the given
//! inputs/outputs, infers every index's extent (which must agree across
//! *all* of its use sites — the bound output tensors count as use sites,
//! which is how broadcast-only LHS indices get extents), and executes.
//!
//! Execution is the v1 materializing strategy (LANGUAGE.md §10 E3): every
//! reduction node computes once into a temporary indexed by exactly the
//! free indices occurring under it — a softmax denominator is therefore a
//! 0-dim temp computed once, never re-derived per output element — and
//! statements evaluate elementwise over their LHS extents. Iteration
//! order is a fixed row-major odometer, so results are deterministic
//! (§10 E1). Sparse inputs are read through [`NDIndex::get`], i.e. the
//! dense-equivalent fallback (§6.4); kernel-backed execution slots in
//! behind these semantics later.

use std::collections::HashMap;

use super::ast::BinOp;
use super::check::{CExpr, Checked, IndexId};
use super::{Elem, LangError, Registry};
use crate::dense::Dense;
use crate::tensor::NDIndex;

pub use super::fast::RunReport;

fn dims_of<T>(t: &dyn NDIndex<T>) -> Vec<usize> {
    (0..t.ndim()).map(|a| t.dim(a)).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Odometer: row-major iteration over a set of index ids
// ─────────────────────────────────────────────────────────────────────────

/// Iterates the Cartesian product of `ids`' extents by mutating their
/// positions in a shared index buffer. Last id varies fastest. With no
/// ids it yields exactly one (empty) tuple; with any zero extent, none.
struct Odo<'a> {
    ids: &'a [IndexId],
    dims: Vec<usize>,
    started: bool,
}

impl<'a> Odo<'a> {
    fn new(ids: &'a [IndexId], extents: &[usize]) -> Self {
        Odo { ids, dims: ids.iter().map(|&i| extents[i as usize]).collect(), started: false }
    }

    fn advance(&mut self, idx: &mut [usize]) -> bool {
        if !self.started {
            if self.dims.iter().any(|&d| d == 0) {
                return false;
            }
            for &id in self.ids {
                idx[id as usize] = 0;
            }
            self.started = true;
            return true;
        }
        for k in (0..self.ids.len()).rev() {
            let p = self.ids[k] as usize;
            idx[p] += 1;
            if idx[p] < self.dims[k] {
                return true;
            }
            idx[p] = 0;
        }
        false
    }
}

/// Row-major visit of every index tuple of `shape` (one empty tuple for
/// rank 0, none if any extent is 0).
pub(super) fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut ix = vec![0usize; shape.len()];
    'outer: loop {
        f(&ix);
        for k in (0..ix.len()).rev() {
            ix[k] += 1;
            if ix[k] < shape[k] {
                continue 'outer;
            }
            ix[k] = 0;
        }
        return;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 1: bind — name resolution and extent inference
// ─────────────────────────────────────────────────────────────────────────

struct Binder<'a> {
    index_names: &'a [String],
    /// Shapes of every name defined so far (inputs, then statements as
    /// they are processed).
    shapes: HashMap<String, Vec<usize>>,
    extents: Vec<Option<usize>>,
}

impl Binder<'_> {
    fn unify(&mut self, id: IndexId, d: usize, site: &str) -> Result<(), LangError> {
        match self.extents[id as usize] {
            None => {
                self.extents[id as usize] = Some(d);
                Ok(())
            }
            Some(prev) if prev == d => Ok(()),
            Some(prev) => Err(LangError::ExtentMismatch {
                index: self.index_names[id as usize].clone(),
                expected: prev,
                got: d,
                site: site.to_string(),
            }),
        }
    }

    fn shape_of(&self, name: &str, stmt: &str) -> Result<&Vec<usize>, LangError> {
        self.shapes
            .get(name)
            .ok_or_else(|| LangError::UnknownName { name: name.to_string(), stmt: stmt.to_string() })
    }

    fn bind_expr(
        &mut self,
        e: &CExpr,
        stmt: &str,
        used: &mut Vec<IndexId>,
    ) -> Result<(), LangError> {
        match e {
            CExpr::Num(_) => Ok(()),
            CExpr::ScalarRef(name) => {
                let shape = self.shape_of(name, stmt)?;
                if !shape.is_empty() {
                    return Err(LangError::RankMismatch {
                        name: name.clone(),
                        expected: shape.len(),
                        got: 0,
                    });
                }
                Ok(())
            }
            CExpr::Load { tensor, indices } => {
                let shape = self.shape_of(tensor, stmt)?.clone();
                if shape.len() != indices.len() {
                    return Err(LangError::RankMismatch {
                        name: tensor.clone(),
                        expected: shape.len(),
                        got: indices.len(),
                    });
                }
                for (&id, &d) in indices.iter().zip(&shape) {
                    self.unify(id, d, tensor)?;
                }
                used.extend_from_slice(indices);
                Ok(())
            }
            CExpr::Call { args, .. } => {
                for a in args {
                    self.bind_expr(a, stmt, used)?;
                }
                Ok(())
            }
            CExpr::Binary { lhs, rhs, .. } => {
                self.bind_expr(lhs, stmt, used)?;
                self.bind_expr(rhs, stmt, used)
            }
            CExpr::Reduce { indices, body, .. } => {
                used.extend_from_slice(indices);
                self.bind_expr(body, stmt, used)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 2: evaluate
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum SrcRef {
    /// Position in the caller's `inputs` slice.
    Input(usize),
    /// Position in the arena of computed statement results and reduction
    /// temporaries.
    Arena(usize),
}

/// Reduce-free elementwise expression: what remains of a statement RHS
/// once every reduction node has been materialized into an arena temp.
enum EExpr<T> {
    Num(T),
    Load { src: SrcRef, indices: Vec<IndexId> },
    Call { func: usize, args: Vec<EExpr<T>> },
    Binary { op: BinOp, lhs: Box<EExpr<T>>, rhs: Box<EExpr<T>> },
}

fn collect_ids<T>(e: &EExpr<T>, out: &mut Vec<IndexId>) {
    match e {
        EExpr::Num(_) => {}
        EExpr::Load { indices, .. } => out.extend_from_slice(indices),
        EExpr::Call { args, .. } => {
            for a in args {
                collect_ids(a, out);
            }
        }
        EExpr::Binary { lhs, rhs, .. } => {
            collect_ids(lhs, out);
            collect_ids(rhs, out);
        }
    }
}

fn bin_apply<T: Elem>(op: BinOp, a: T, b: T) -> T {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

fn eval_elem<T: Elem>(
    e: &EExpr<T>,
    idx: &[usize],
    inputs: &[(&str, &dyn NDIndex<T>)],
    arena: &[Dense<T>],
    reg: &Registry<T>,
    scratch: &mut Vec<usize>,
) -> T {
    match e {
        EExpr::Num(v) => *v,
        EExpr::Load { src, indices } => {
            scratch.clear();
            scratch.extend(indices.iter().map(|&i| idx[i as usize]));
            match src {
                SrcRef::Input(i) => inputs[*i].1.get(&scratch[..]),
                SrcRef::Arena(i) => arena[*i].get(&scratch[..]),
            }
        }
        EExpr::Call { func, args } => {
            let vals: Vec<T> =
                args.iter().map(|a| eval_elem(a, idx, inputs, arena, reg, scratch)).collect();
            (reg.fns[*func].eval)(&vals)
        }
        EExpr::Binary { op, lhs, rhs } => {
            let a = eval_elem(lhs, idx, inputs, arena, reg, scratch);
            let b = eval_elem(rhs, idx, inputs, arena, reg, scratch);
            bin_apply(*op, a, b)
        }
    }
}

/// Lower one statement RHS to a reduce-free [`EExpr`], materializing each
/// reduction node (innermost first) into an arena temp indexed by the
/// free indices occurring under it.
#[allow(clippy::too_many_arguments)]
fn lower<T: Elem>(
    e: &CExpr,
    reg: &Registry<T>,
    extents: &[usize],
    inputs: &[(&str, &dyn NDIndex<T>)],
    name_src: &HashMap<String, SrcRef>,
    arena: &mut Vec<Dense<T>>,
    idx: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
) -> EExpr<T> {
    match e {
        CExpr::Num(v) => EExpr::Num((reg.lit)(*v)),
        CExpr::ScalarRef(name) => {
            EExpr::Load { src: name_src[name.as_str()], indices: Vec::new() }
        }
        CExpr::Load { tensor, indices } => {
            EExpr::Load { src: name_src[tensor.as_str()], indices: indices.clone() }
        }
        CExpr::Call { func, args } => EExpr::Call {
            func: *func,
            args: args
                .iter()
                .map(|a| lower(a, reg, extents, inputs, name_src, arena, idx, scratch))
                .collect(),
        },
        CExpr::Binary { op, lhs, rhs } => EExpr::Binary {
            op: *op,
            lhs: Box::new(lower(lhs, reg, extents, inputs, name_src, arena, idx, scratch)),
            rhs: Box::new(lower(rhs, reg, extents, inputs, name_src, arena, idx, scratch)),
        },
        CExpr::Reduce { op, indices, body } => {
            let body_e = lower(body, reg, extents, inputs, name_src, arena, idx, scratch);
            let mut free = Vec::new();
            collect_ids(&body_e, &mut free);
            free.retain(|id| !indices.contains(id));
            free.sort_unstable();
            free.dedup();

            let shape: Vec<usize> = free.iter().map(|&i| extents[i as usize]).collect();
            let mut tmp = Dense::<T>::zeros(shape);
            let rdef = &reg.reduces[*op];
            let mut outer = Odo::new(&free, extents);
            while outer.advance(idx) {
                let mut acc = rdef.identity;
                let mut inner = Odo::new(indices, extents);
                while inner.advance(idx) {
                    let v = eval_elem(&body_e, idx, inputs, &*arena, reg, scratch);
                    acc = (rdef.fold)(acc, v);
                }
                scratch.clear();
                scratch.extend(free.iter().map(|&i| idx[i as usize]));
                tmp.set(scratch, acc);
            }
            let src = SrcRef::Arena(arena.len());
            arena.push(tmp);
            EExpr::Load { src, indices: free }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// run
// ─────────────────────────────────────────────────────────────────────────

/// Bind a checked program to an environment and execute it.
///
/// `inputs` are the free names the program reads (tensors of any
/// [`NDIndex`] implementation — 0-dim for scalar parameters); `outputs`
/// is the explicit output set (LANGUAGE.md §10 D8): every listed name
/// must be defined by a statement, its provided tensor supplies/confirms
/// the output shape, and every statement LHS *not* listed is a temporary.
/// Pass the same registry used for [`check`](super::check).
///
/// Execution takes the kernel-backed fast path where it applies and the
/// tree-walking evaluator otherwise; the two are required to agree bit
/// for bit, and [`run_reference`] runs the tree-walker unconditionally if
/// you want to check that yourself.
pub fn run<T: Elem>(
    checked: &Checked,
    reg: &Registry<T>,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
) -> Result<(), LangError> {
    run_reported(checked, reg, inputs, outputs).map(|_| ())
}

/// [`run`], additionally reporting which path executed the program.
///
/// The counts are diagnostics — how much of the program the kernel path
/// took, and how many reductions reached the JIT — not part of the
/// semantics. Results are identical to [`run`] either way.
pub fn run_reported<T: Elem>(
    checked: &Checked,
    reg: &Registry<T>,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
) -> Result<RunReport, LangError> {
    run_with(checked, reg, inputs, outputs, RunOptions::default())
}

/// Per-run execution options. Purely a **specialization** choice: nothing
/// here changes what a program computes, only how the backend is asked to
/// compile it. `RunOptions::default()` is what [`run`] uses.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunOptions<'a> {
    /// Index names whose extent should be a **runtime argument** to
    /// generated code rather than a baked-in constant.
    ///
    /// These are *index* names — the `t` in `sum(t: …)` or in `s[h,t]`,
    /// not a tensor name — and marking one marks every occurrence of that
    /// name in the program. The plan and the AST stay shape-polymorphic
    /// either way; what changes is that a JIT'd contraction touching a
    /// dynamic axis compiles **one** kernel serving every extent of it,
    /// instead of one kernel per extent. Mark the axes that *move* between
    /// runs (a KV-cache length, a batch size); leaving an axis static keeps
    /// its extent a constant, which is what lets Cranelift fold strides.
    ///
    /// Results are bit-identical either way: a dynamic axis changes what is
    /// a constant, never the loop nest, the iteration order, or a single
    /// floating-point operation.
    ///
    /// Names that no index of the program matches are ignored;
    /// [`RunReport::dynamic_axes`] reports how many did match, which is how
    /// to detect a typo.
    pub dynamic: &'a [&'a str],
}

/// [`run_reported`] with explicit [`RunOptions`].
pub fn run_with<T: Elem>(
    checked: &Checked,
    reg: &Registry<T>,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
    opts: RunOptions<'_>,
) -> Result<RunReport, LangError> {
    let extents = bind(checked, inputs, outputs)?;
    // Index names are per-*occurrence* (a binder gets a fresh id, alpha-
    // renaming and all), so "the axis named t" is every id with that name.
    // Nothing marked stays allocation-free — that is what plain `run` does,
    // and it should cost exactly what it did before this option existed.
    let dynamic: Vec<bool> = if opts.dynamic.is_empty() {
        Vec::new()
    } else {
        checked.index_names.iter().map(|n| opts.dynamic.contains(&n.as_str())).collect()
    };
    if let Some(report) = super::fast::try_run(checked, reg, inputs, outputs, &extents, &dynamic) {
        return Ok(report);
    }
    tree_walk(checked, reg, inputs, outputs, &extents);
    Ok(RunReport {
        statements: checked.stmts.len(),
        dynamic_axes: dynamic.iter().filter(|&&d| d).count(),
        ..RunReport::default()
    })
}

/// [`run`] on the tree-walking evaluator only — the semantics oracle.
///
/// Slow by construction (recursive per-element evaluation through
/// [`NDIndex::get`]); it exists so the kernel path can be differentially
/// tested against a definition of the language that has no fast cases.
pub fn run_reference<T: Elem>(
    checked: &Checked,
    reg: &Registry<T>,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
) -> Result<(), LangError> {
    let extents = bind(checked, inputs, outputs)?;
    tree_walk(checked, reg, inputs, outputs, &extents);
    Ok(())
}

/// Phase 1, shared by both execution paths: validate the name sets and
/// infer every index's extent. The error surface lives entirely here, so
/// which path runs afterwards can never change what a program rejects.
fn bind<T: Elem>(
    checked: &Checked,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
) -> Result<Vec<usize>, LangError> {
    // ── Name sets ──
    let mut input_ix: HashMap<&str, usize> = HashMap::new();
    for (i, (name, _)) in inputs.iter().enumerate() {
        if input_ix.insert(name, i).is_some() {
            return Err(LangError::RedefinedName { name: name.to_string() });
        }
    }
    let mut output_shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    for (name, t) in outputs.iter() {
        if input_ix.contains_key(name) || output_shapes.insert(name, dims_of(&**t)).is_some() {
            return Err(LangError::RedefinedName { name: name.to_string() });
        }
    }
    for (name, _) in outputs.iter() {
        if !checked.stmts.iter().any(|s| s.name == *name) {
            return Err(LangError::MissingOutput { name: name.to_string() });
        }
    }

    // ── Phase 1: extents and shapes ──
    let mut binder = Binder {
        index_names: &checked.index_names,
        shapes: inputs.iter().map(|(n, t)| (n.to_string(), dims_of(*t))).collect(),
        extents: vec![None; checked.index_names.len()],
    };
    for stmt in &checked.stmts {
        if input_ix.contains_key(stmt.name.as_str()) {
            return Err(LangError::RedefinedName { name: stmt.name.clone() });
        }
        let mut used: Vec<IndexId> = stmt.lhs.clone();
        binder.bind_expr(&stmt.rhs, &stmt.name, &mut used)?;
        if let Some(oshape) = output_shapes.get(stmt.name.as_str()) {
            if oshape.len() != stmt.lhs.len() {
                return Err(LangError::RankMismatch {
                    name: stmt.name.clone(),
                    expected: oshape.len(),
                    got: stmt.lhs.len(),
                });
            }
            let oshape = oshape.clone();
            for (&id, &d) in stmt.lhs.iter().zip(&oshape) {
                binder.unify(id, d, &stmt.name)?;
            }
        }
        for &id in &used {
            if binder.extents[id as usize].is_none() {
                return Err(LangError::UnresolvedExtent {
                    index: checked.index_names[id as usize].clone(),
                    stmt: stmt.name.clone(),
                });
            }
        }
        let shape: Vec<usize> =
            stmt.lhs.iter().map(|&id| binder.extents[id as usize].unwrap()).collect();
        binder.shapes.insert(stmt.name.clone(), shape);
    }
    Ok(binder.extents.iter().map(|o| o.unwrap_or(0)).collect())
}

/// Phase 2, the reference strategy: materialize every reduction into an
/// arena temp and evaluate the rest elementwise through `NDIndex::get`.
/// Runs only after [`bind`] has succeeded, so it cannot fail.
fn tree_walk<T: Elem>(
    checked: &Checked,
    reg: &Registry<T>,
    inputs: &[(&str, &dyn NDIndex<T>)],
    outputs: &mut [(&str, &mut dyn NDIndex<T>)],
    extents: &[usize],
) {
    let mut arena: Vec<Dense<T>> = Vec::new();
    let mut name_src: HashMap<String, SrcRef> =
        inputs.iter().enumerate().map(|(i, (n, _))| (n.to_string(), SrcRef::Input(i))).collect();
    let mut idx = vec![0usize; checked.index_names.len()];
    let mut scratch: Vec<usize> = Vec::new();

    for stmt in &checked.stmts {
        let elem =
            lower(&stmt.rhs, reg, extents, inputs, &name_src, &mut arena, &mut idx, &mut scratch);
        let shape: Vec<usize> = stmt.lhs.iter().map(|&id| extents[id as usize]).collect();
        let mut out = Dense::<T>::zeros(shape);
        let mut o = Odo::new(&stmt.lhs, extents);
        while o.advance(&mut idx) {
            let v = eval_elem(&elem, &idx, inputs, &arena, reg, &mut scratch);
            scratch.clear();
            scratch.extend(stmt.lhs.iter().map(|&i| idx[i as usize]));
            out.set(&scratch, v);
        }
        name_src.insert(stmt.name.clone(), SrcRef::Arena(arena.len()));
        arena.push(out);
    }

    // ── Copy requested outputs out of the arena ──
    for (name, dst) in outputs.iter_mut() {
        let Some(SrcRef::Arena(i)) = name_src.get(*name).copied() else {
            unreachable!("output presence validated above");
        };
        let src = &arena[i];
        for_each_index(&src.shape, |ix| dst.set(ix, src.get(ix)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odometer_orders_and_edge_cases() {
        // ids 0,1 with extents 2,3 — row-major, last fastest.
        let extents = [2usize, 3usize];
        let ids = [0u32, 1u32];
        let mut idx = vec![0usize; 2];
        let mut seen = Vec::new();
        let mut o = Odo::new(&ids, &extents);
        while o.advance(&mut idx) {
            seen.push((idx[0], idx[1]));
        }
        assert_eq!(seen, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);

        // No ids: exactly one empty tuple.
        let mut o = Odo::new(&[], &extents);
        assert!(o.advance(&mut idx));
        assert!(!o.advance(&mut idx));

        // Zero extent: no tuples.
        let mut o = Odo::new(&[0], &[0usize]);
        assert!(!o.advance(&mut idx));
    }

    #[test]
    fn for_each_index_rank0_and_empty() {
        let mut n = 0;
        for_each_index(&[], |ix| {
            assert!(ix.is_empty());
            n += 1;
        });
        assert_eq!(n, 1);
        for_each_index(&[3, 0], |_| panic!("empty tensor has no indices"));
    }
}
