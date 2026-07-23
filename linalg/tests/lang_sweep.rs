//! Differential sweep: kernel-backed execution vs the tree-walking
//! evaluator, **bit for bit**.
//!
//! `lang::run` picks the fast path in `lang::fast` where it applies and
//! the tree-walker otherwise; `lang::run_reference` is the tree-walker
//! unconditionally and is the language's semantics oracle. LANGUAGE.md
//! §10 E1 makes a compiled artifact's results reproducible to the bit, so
//! "close enough" is not the bar here: every element is compared as raw
//! `f32` bits, which also pins down `±0.0` and NaN payloads that an
//! approximate comparison would hide.
//!
//! Same shape as `einsum_sweep.rs` / `reduce_sweep.rs`: a deterministic
//! generator, a bounded default run, and an `#[ignore]`d wide one.
//!
//!   * `sweep_random_programs` — a few hundred random programs over 2–3
//!     free indices, mixing every operator, every built-in function, and
//!     nested/sibling reductions.
//!   * `sweep_random_programs_wide` (`--ignored`) — the same generator,
//!     two orders of magnitude more cases.
//!
//! Every case additionally asserts that the fast path *ran* — a
//! differential test that silently compares the tree-walker with itself
//! would pass forever.

#![cfg(feature = "dense")]

use std::collections::HashMap;

use linalg::dense::Dense;
use linalg::lang::{
    BinOp, Expr, LangError, Program, Registry, Stmt, check, parse, run_reference, run_reported,
};
use linalg::tensor::NDIndex;

// ─── deterministic RNG ──────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
    /// True with probability `num / den`.
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.next() % den < num
    }
}

/// Values chosen to make the edge cases likely: both signed zeros (whose
/// ordering under `max`/`min` is fold-order sensitive), exact small
/// integers, fractions, and magnitudes that overflow to ±∞ under `prod`.
const POOL: [f32; 14] = [
    0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.25, 3.0, -4.0, 0.0, 1e19, -1e19, 7.0,
];

// ─── program generator ──────────────────────────────────────────────────

/// Free-index names a statement may bind, plus the broadcast-only spare.
const FREE_NAMES: [&str; 2] = ["i", "j"];
const SPARE_NAME: &str = "z";

struct Gen {
    rng: Rng,
    /// Extent of every index name ever used (binder names are unique
    /// program-wide, so one map is enough).
    extent: HashMap<String, usize>,
    /// Generated input tensors: name, index list, data.
    inputs: Vec<(String, Vec<String>, Dense<f32>)>,
    /// Statement results defined so far: name, index list.
    defined: Vec<(String, Vec<String>)>,
    n_tensors: usize,
    n_binders: usize,
}

impl Gen {
    fn new(seed: u64) -> Self {
        let mut extent = HashMap::new();
        extent.insert("i".to_string(), 3usize);
        extent.insert("j".to_string(), 4usize);
        extent.insert(SPARE_NAME.to_string(), 2usize);
        Gen {
            rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xDEAD_BEEF)),
            extent,
            inputs: Vec::new(),
            defined: Vec::new(),
            n_tensors: 0,
            n_binders: 0,
        }
    }

    fn fresh_binder(&mut self) -> String {
        let name = format!("b{}", self.n_binders);
        self.n_binders += 1;
        // 2..=4, so a binder's extent is sometimes shorter and sometimes
        // longer than the block size boundary logic cares about.
        let e = 2 + self.rng.below(3);
        self.extent.insert(name.clone(), e);
        name
    }

    fn extent_of(&self, name: &str) -> usize {
        self.extent[name]
    }

    fn value(&mut self) -> f32 {
        POOL[self.rng.below(POOL.len())]
    }

    /// A tensor reference over `indices` — reusing an existing input with
    /// the identical index list about half the time, so the same buffer
    /// gets read twice in one expression (`x[j] * x[j]`).
    fn tensor(&mut self, indices: Vec<String>) -> Expr {
        if indices.is_empty() {
            // 0-dim: a scalar parameter.
            let existing: Vec<usize> = (0..self.inputs.len())
                .filter(|&k| self.inputs[k].1.is_empty())
                .collect();
            if !existing.is_empty() && self.rng.chance(1, 2) {
                let k = existing[self.rng.below(existing.len())];
                return Expr::Scalar(self.inputs[k].0.clone());
            }
            let name = format!("t{}", self.n_tensors);
            self.n_tensors += 1;
            let mut d = Dense::<f32>::zeros(vec![]);
            let v = self.value();
            d.data[0] = v;
            self.inputs.push((name.clone(), Vec::new(), d));
            return Expr::Scalar(name);
        }

        let existing: Vec<usize> =
            (0..self.inputs.len()).filter(|&k| self.inputs[k].1 == indices).collect();
        if !existing.is_empty() && self.rng.chance(1, 2) {
            let k = existing[self.rng.below(existing.len())];
            let name = self.inputs[k].0.clone();
            return Expr::Tensor { name, indices };
        }

        let name = format!("t{}", self.n_tensors);
        self.n_tensors += 1;
        let shape: Vec<usize> = indices.iter().map(|n| self.extent_of(n)).collect();
        let mut d = Dense::<f32>::zeros(shape);
        for k in 0..d.data.len() {
            d.data[k] = self.value();
        }
        self.inputs.push((name.clone(), indices.clone(), d));
        Expr::Tensor { name, indices }
    }

    /// A leaf whose index list contains every name in `must` (so a binder
    /// always has a shape-bearing use site) plus a random tail drawn from
    /// `scope` — including deliberate repeats, which read a diagonal.
    fn leaf(&mut self, scope: &[String], must: &[String]) -> Expr {
        let mut ix: Vec<String> = must.to_vec();
        let extra = self.rng.below(3);
        for _ in 0..extra {
            if scope.is_empty() {
                break;
            }
            let pick = scope[self.rng.below(scope.len())].clone();
            // A repeat is legal and reads the diagonal; keep it rare.
            if ix.contains(&pick) && !self.rng.chance(1, 4) {
                continue;
            }
            ix.push(pick);
        }
        // Rotate so `must` is not always leading.
        if !ix.is_empty() && self.rng.chance(1, 2) {
            let r = self.rng.below(ix.len());
            ix.rotate_left(r);
        }
        self.tensor(ix)
    }

    fn expr(&mut self, scope: &[String], depth: usize) -> Expr {
        if depth == 0 || self.rng.chance(1, 3) {
            return if self.rng.chance(1, 6) {
                Expr::Num(POOL[self.rng.below(POOL.len())] as f64)
            } else {
                self.leaf(scope, &[])
            };
        }
        match self.rng.below(10) {
            0..=3 => {
                let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div][self.rng.below(4)];
                let lhs = self.expr(scope, depth - 1);
                let rhs = self.expr(scope, depth - 1);
                Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            4..=5 => {
                const UNARY: [&str; 7] = ["exp", "ln", "sqrt", "abs", "relu", "tanh", "neg"];
                let f = UNARY[self.rng.below(UNARY.len())];
                let a = self.expr(scope, depth - 1);
                Expr::Call { func: f.to_string(), args: vec![a] }
            }
            6 => {
                let f = if self.rng.chance(1, 2) { "max" } else { "min" };
                let a = self.expr(scope, depth - 1);
                let b = self.expr(scope, depth - 1);
                Expr::Call { func: f.to_string(), args: vec![a, b] }
            }
            _ => self.reduction(scope, depth),
        }
    }

    fn reduction(&mut self, scope: &[String], depth: usize) -> Expr {
        let nb = 1 + self.rng.below(2);
        let bound: Vec<String> = (0..nb).map(|_| self.fresh_binder()).collect();
        let mut inner: Vec<String> = scope.to_vec();
        inner.extend(bound.iter().cloned());

        // The core leaf carries every bound index, so extents resolve; the
        // optional partner is what turns `sum(k: a[i,k])` into the classic
        // contraction `sum(k: a[i,k] * b[k,j])`.
        let core = self.leaf(&inner, &bound);
        let body = if depth > 1 && self.rng.chance(2, 3) {
            let op = [BinOp::Mul, BinOp::Add, BinOp::Sub][self.rng.below(3)];
            let partner = if self.rng.chance(1, 2) {
                self.leaf(&inner, &[])
            } else {
                self.expr(&inner, depth - 1)
            };
            Expr::Binary { op, lhs: Box::new(core), rhs: Box::new(partner) }
        } else {
            core
        };

        const REDUCERS: [&str; 4] = ["sum", "prod", "max", "min"];
        Expr::Reduce {
            op: REDUCERS[self.rng.below(REDUCERS.len())].to_string(),
            indices: bound,
            body: Box::new(body),
        }
    }

    /// Names in `scope` actually indexed somewhere in `e`.
    fn used_names(e: &Expr, scope: &[String], out: &mut Vec<String>) {
        match e {
            Expr::Num(_) | Expr::Scalar(_) => {}
            Expr::Tensor { indices, .. } => {
                for ix in indices {
                    if scope.contains(ix) && !out.contains(ix) {
                        out.push(ix.clone());
                    }
                }
            }
            Expr::Call { args, .. } => {
                args.iter().for_each(|a| Self::used_names(a, scope, out))
            }
            Expr::Binary { lhs, rhs, .. } => {
                Self::used_names(lhs, scope, out);
                Self::used_names(rhs, scope, out);
            }
            Expr::Reduce { body, .. } => Self::used_names(body, scope, out),
        }
    }

    /// A whole program: 1–3 statements, the last one the requested output.
    fn program(&mut self) -> (Program, String, Vec<usize>) {
        let n_stmts = 1 + self.rng.below(3);
        let mut stmts = Vec::new();
        let mut last: (String, Vec<String>) = (String::new(), Vec::new());

        for s in 0..n_stmts {
            let is_last = s + 1 == n_stmts;
            let n_free = self.rng.below(3);
            let scope: Vec<String> =
                FREE_NAMES[..n_free].iter().map(|s| s.to_string()).collect();

            let depth = 2 + self.rng.below(2);
            let mut rhs = self.expr(&scope, depth);

            // Chain onto an earlier statement result when its index list
            // is in scope here.
            let reusable: Vec<usize> = (0..self.defined.len())
                .filter(|&k| self.defined[k].1.iter().all(|n| scope.contains(n)))
                .collect();
            if !reusable.is_empty() && self.rng.chance(1, 3) {
                let k = reusable[self.rng.below(reusable.len())];
                let (name, ix) = self.defined[k].clone();
                let prev = if ix.is_empty() {
                    Expr::Scalar(name)
                } else {
                    Expr::Tensor { name, indices: ix }
                };
                rhs = Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(rhs),
                    rhs: Box::new(prev),
                };
            }

            // Every LHS index of an intermediate statement must have a
            // shape-bearing use site on the RHS (D5); only the requested
            // output can take an extent from the bound tensor, so the
            // broadcast-only index is added there and nowhere else.
            let mut lhs = Vec::new();
            Self::used_names(&rhs, &scope, &mut lhs);
            lhs.sort_by_key(|n| FREE_NAMES.iter().position(|f| f == n).unwrap_or(usize::MAX));
            if is_last && self.rng.chance(1, 4) {
                lhs.push(SPARE_NAME.to_string());
            }

            let name = format!("s{s}");
            stmts.push(Stmt { name: name.clone(), indices: lhs.clone(), rhs });
            self.defined.push((name.clone(), lhs.clone()));
            last = (name, lhs);
        }

        let shape: Vec<usize> = last.1.iter().map(|n| self.extent_of(n)).collect();
        (Program { stmts }, last.0, shape)
    }
}

// ─── differential driver ────────────────────────────────────────────────

/// Bit equality, with the one documented exception: **NaN payloads**.
///
/// IEEE-754 does not specify which input NaN an operation propagates, and
/// LLVM freely commutes `fadd`/`fmul` operands — so `NaN_a + NaN_b` can
/// hand back either operand depending only on how a loop was compiled.
/// Every other value, `±0.0` and `±∞` included, must match exactly; a NaN
/// facing a non-NaN is still a failure.
fn bits_eq(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    if a.len() != b.len() {
        return Some((usize::MAX, 0.0, 0.0));
    }
    a.iter()
        .zip(b)
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()))
        .map(|(i, (x, y))| (i, *x, *y))
}

#[derive(Default)]
struct Stats {
    checked: usize,
    skipped: usize,
    jit_reductions: usize,
    tape_reductions: usize,
    /// Output elements compared, and how many of those were NaN (whose
    /// payload `bits_eq` deliberately ignores) — a sweep whose outputs
    /// were all NaN would prove nothing.
    elems: usize,
    nan_elems: usize,
    /// Elements whose raw bits differed *and* were NaN on both sides —
    /// the exact size of the carve-out `bits_eq` documents.
    nan_payload_diffs: usize,
}

/// Run one generated program down both paths and demand bit equality.
fn differential(g: &mut Gen, stats: &mut Stats) {
    let reg = Registry::<f32>::builtins();
    let (prog, out_name, out_shape) = g.program();
    let Ok(checked) = check(&prog, &reg) else {
        stats.skipped += 1;
        return;
    };

    let inputs: Vec<(&str, &dyn NDIndex<f32>)> =
        g.inputs.iter().map(|(n, _, d)| (n.as_str(), d as &dyn NDIndex<f32>)).collect();

    let mut fast_out = Dense::<f32>::zeros(out_shape.clone());
    let mut ref_out = Dense::<f32>::zeros(out_shape);
    let fast = run_reported(
        &checked,
        &reg,
        &inputs,
        &mut [(out_name.as_str(), &mut fast_out as &mut dyn NDIndex<f32>)],
    );
    let refr = run_reference(
        &checked,
        &reg,
        &inputs,
        &mut [(out_name.as_str(), &mut ref_out as &mut dyn NDIndex<f32>)],
    );

    match (fast, refr) {
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "the two paths disagree about rejecting a program");
            stats.skipped += 1;
        }
        (Ok(report), Ok(())) => {
            assert!(
                report.all_kernel(),
                "all-Dense inputs must take the kernel path, got {report:?}"
            );
            if let Some((i, x, y)) = bits_eq(&fast_out.data, &ref_out.data) {
                panic!(
                    "bit mismatch at flat index {i}: kernel {x:?} ({:#010x}) vs \
                     tree-walker {y:?} ({:#010x})\nprogram:\n{}",
                    x.to_bits(),
                    y.to_bits(),
                    describe(&prog),
                );
            }
            stats.checked += 1;
            stats.jit_reductions += report.jit_reductions;
            stats.tape_reductions += report.tape_reductions;
            stats.elems += ref_out.data.len();
            stats.nan_elems += ref_out.data.iter().filter(|v| v.is_nan()).count();
            stats.nan_payload_diffs += fast_out
                .data
                .iter()
                .zip(&ref_out.data)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
        }
        (a, b) => panic!("paths disagree on success: kernel {a:?} vs reference {b:?}"),
    }
}

/// Best-effort rendering of a generated program for failure messages.
fn describe(p: &Program) -> String {
    fn e(x: &Expr) -> String {
        match x {
            Expr::Num(v) => format!("{v}"),
            Expr::Scalar(n) => n.clone(),
            Expr::Tensor { name, indices } => format!("{name}[{}]", indices.join(",")),
            Expr::Call { func, args } => {
                format!("{func}({})", args.iter().map(e).collect::<Vec<_>>().join(", "))
            }
            Expr::Binary { op, lhs, rhs } => {
                let o = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                };
                format!("({} {o} {})", e(lhs), e(rhs))
            }
            Expr::Reduce { op, indices, body } => {
                format!("{op}({}: {})", indices.join(","), e(body))
            }
        }
    }
    p.stmts
        .iter()
        .map(|s| {
            let lhs = if s.indices.is_empty() {
                s.name.clone()
            } else {
                format!("{}[{}]", s.name, s.indices.join(","))
            };
            format!("{lhs} = {}", e(&s.rhs))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn sweep(n: usize, seed0: u64) -> Stats {
    let mut stats = Stats::default();
    for k in 0..n {
        let mut g = Gen::new(seed0 + k as u64);
        differential(&mut g, &mut stats);
    }
    stats
}

// ─── tests ──────────────────────────────────────────────────────────────

fn report(tag: &str, s: &Stats) {
    println!(
        "[{tag}] {} programs bit-identical, {} skipped (bind errors); \
         reductions: {} on the JIT, {} on the tape; \
         {} output elements compared ({} NaN, of which {} differed only in \
         NaN payload)",
        s.checked,
        s.skipped,
        s.jit_reductions,
        s.tape_reductions,
        s.elems,
        s.nan_elems,
        s.nan_payload_diffs
    );
    assert!(
        s.elems - s.nan_elems > s.elems / 2,
        "most outputs were NaN — the sweep would not be testing much"
    );
}

#[test]
fn sweep_random_programs() {
    let s = sweep(400, 1);
    report("lang_sweep", &s);
    assert!(s.checked > 250, "generator produced too few runnable programs: {}", s.checked);
    assert!(s.tape_reductions > 0, "no reduction exercised the tape kernel");
    #[cfg(feature = "jit")]
    assert!(s.jit_reductions > 0, "no reduction reached the JIT");
}

#[test]
#[ignore = "wide sweep; run explicitly"]
fn sweep_random_programs_wide() {
    let s = sweep(20_000, 1_000_003);
    report("lang_sweep wide", &s);
    assert!(s.checked > 12_000);
}

// ─── the named examples, both paths, bit for bit ────────────────────────

/// Deterministic LCG fill matching `tests/lang.rs`.
fn fill(t: &mut Dense<f32>, seed: u64) {
    let mut s = seed;
    for v in t.data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
}

fn both_paths_agree(
    src: &str,
    inputs: &[(&str, &Dense<f32>)],
    out_name: &str,
    out_shape: Vec<usize>,
) -> linalg::lang::RunReport {
    let reg = Registry::<f32>::builtins();
    let checked = check(&parse(src).unwrap(), &reg).unwrap();
    let ins: Vec<(&str, &dyn NDIndex<f32>)> =
        inputs.iter().map(|(n, d)| (*n, *d as &dyn NDIndex<f32>)).collect();

    let mut a = Dense::<f32>::zeros(out_shape.clone());
    let mut b = Dense::<f32>::zeros(out_shape);
    let report =
        run_reported(&checked, &reg, &ins, &mut [(out_name, &mut a as &mut dyn NDIndex<f32>)])
            .unwrap();
    run_reference(&checked, &reg, &ins, &mut [(out_name, &mut b as &mut dyn NDIndex<f32>)])
        .unwrap();
    if let Some((i, x, y)) = bits_eq(&a.data, &b.data) {
        panic!("{src}\nbit mismatch at {i}: kernel {x:?} vs tree-walker {y:?}");
    }
    report
}

#[test]
fn language_examples_are_bit_identical() {
    let mut a = Dense::<f32>::zeros(vec![5, 6]);
    let mut b = Dense::<f32>::zeros(vec![6, 4]);
    fill(&mut a, 7);
    fill(&mut b, 8);
    let r = both_paths_agree(
        "c[i,k] = sum(j: a[i,j] * b[j,k])",
        &[("a", &a), ("b", &b)],
        "c",
        vec![5, 4],
    );
    #[cfg(feature = "jit")]
    assert_eq!(r.jit_reductions, 1, "matmul must lower to the JIT");
    #[cfg(not(feature = "jit"))]
    assert_eq!(r.tape_reductions, 1);

    // Tropical: max/min folds stay off the JIT (fmax/fmin disagree with
    // the language's comparison fold on ±0 and NaN).
    let mut m = Dense::<f32>::zeros(vec![6, 6]);
    let mut n = Dense::<f32>::zeros(vec![6, 6]);
    fill(&mut m, 9);
    fill(&mut n, 10);
    let r = both_paths_agree(
        "d[i,k] = min(j: a[i,j] + b[j,k])",
        &[("a", &m), ("b", &n)],
        "d",
        vec![6, 6],
    );
    assert_eq!(r.jit_reductions, 0);
    assert_eq!(r.tape_reductions, 1);

    let mut v = Dense::<f32>::zeros(vec![7]);
    fill(&mut v, 42);
    both_paths_agree("soft[i] = exp(v[i]) / sum(j: exp(v[j]))", &[("v", &v)], "soft", vec![7]);
    both_paths_agree(
        "m       = max(j: v[j])\n\
         soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))",
        &[("v", &v)],
        "soft",
        vec![7],
    );

    let mut x = Dense::<f32>::zeros(vec![8]);
    fill(&mut x, 12);
    let mut eps = Dense::<f32>::zeros(vec![]);
    eps.set(&[], 1e-5);
    let mut count = Dense::<f32>::zeros(vec![]);
    count.set(&[], 8.0);
    let r = both_paths_agree(
        "y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps)",
        &[("x", &x), ("n", &count), ("eps", &eps)],
        "y",
        vec![8],
    );
    #[cfg(feature = "jit")]
    assert_eq!(r.jit_reductions, 1, "sum(j: x[j]*x[j]) is a contraction");
    let _ = r;

    let mut q = Dense::<f32>::zeros(vec![5]);
    let mut xs = Dense::<f32>::zeros(vec![6]);
    let mut mm = Dense::<f32>::zeros(vec![6, 4]);
    fill(&mut q, 1);
    fill(&mut xs, 2);
    fill(&mut mm, 3);
    for t in mm.data.iter_mut() {
        *t = t.abs() + 0.1;
    }
    both_paths_agree(
        "y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))",
        &[("q", &q), ("x", &xs), ("M", &mm)],
        "y",
        vec![4, 5],
    );

    // Multi-index binder: the JIT is only taken when the backend's
    // contraction order matches the binder order.
    let mut t3 = Dense::<f32>::zeros(vec![3, 4, 2]);
    let mut aa = Dense::<f32>::zeros(vec![4, 5]);
    let mut bb = Dense::<f32>::zeros(vec![2, 5]);
    fill(&mut t3, 21);
    fill(&mut aa, 22);
    fill(&mut bb, 23);
    both_paths_agree(
        "s[i,j] = sum(k,l: T[i,k,l] * A[k,j] * B[l,j])",
        &[("T", &t3), ("A", &aa), ("B", &bb)],
        "s",
        vec![3, 5],
    );

    // Diagonal read and a broadcast-only output index.
    let mut sq = Dense::<f32>::zeros(vec![5, 5]);
    fill(&mut sq, 14);
    both_paths_agree("t = sum(i: M[i,i])", &[("M", &sq)], "t", vec![]);
    let mut vv = Dense::<f32>::zeros(vec![3]);
    fill(&mut vv, 13);
    both_paths_agree("y[i,j] = v[i]", &[("v", &vv)], "y", vec![3, 4]);
}

/// Storage without a contiguous row-major image keeps the whole run on
/// the tree-walker — and still agrees with the dense-equivalent program
/// (LANGUAGE.md S8).
#[cfg(feature = "csr")]
#[test]
fn sparse_input_falls_back_silently() {
    use linalg::csr::Csr;

    let reg = Registry::<f32>::builtins();
    let checked = check(&parse("c[i,k] = sum(j: a[i,j] * b[j,k])").unwrap(), &reg).unwrap();

    let n = 4usize;
    let a = Csr::<u32, f32>::from_coo(
        n as u32,
        &mut vec![(0, 1, 2.0), (1, 3, -1.5), (2, 0, 0.5), (3, 3, 4.0)],
    );
    let mut dense_a = Dense::<f32>::zeros(vec![n, n]);
    for r in 0..n {
        for c in 0..n {
            dense_a.set(&[r, c], a.get(r as u32, c as u32));
        }
    }
    let mut b = Dense::<f32>::zeros(vec![n, n]);
    fill(&mut b, 11);

    let mut from_sparse = Dense::<f32>::zeros(vec![n, n]);
    let sparse_report = run_reported(
        &checked,
        &reg,
        &[("a", &a as &dyn NDIndex<f32>), ("b", &b)],
        &mut [("c", &mut from_sparse as &mut dyn NDIndex<f32>)],
    )
    .unwrap();
    assert_eq!(sparse_report.kernel_statements, 0, "CSR input must fall back");

    let mut from_dense = Dense::<f32>::zeros(vec![n, n]);
    let dense_report = run_reported(
        &checked,
        &reg,
        &[("a", &dense_a as &dyn NDIndex<f32>), ("b", &b)],
        &mut [("c", &mut from_dense as &mut dyn NDIndex<f32>)],
    )
    .unwrap();
    assert!(dense_report.all_kernel());

    assert!(bits_eq(&from_sparse.data, &from_dense.data).is_none());
}

/// A custom reduction is not a built-in fold, so it must never be handed
/// to the einsum backend — and must still agree bit for bit.
#[test]
fn custom_reduction_stays_on_the_tape() {
    use linalg::lang::ReduceDef;
    let mut reg = Registry::<f32>::builtins();
    // Deliberately named "sum2" with the same fold as `sum`: recognition
    // is by fold identity, not by name.
    reg.register_reduce(ReduceDef { name: "sum2", identity: 0.0, fold: |a, b| a + b });

    let mut a = Dense::<f32>::zeros(vec![4, 5]);
    let mut b = Dense::<f32>::zeros(vec![5, 3]);
    fill(&mut a, 31);
    fill(&mut b, 32);
    let checked = check(&parse("c[i,k] = sum2(j: a[i,j] * b[j,k])").unwrap(), &reg).unwrap();

    let ins: Vec<(&str, &dyn NDIndex<f32>)> = vec![("a", &a), ("b", &b)];
    let mut x = Dense::<f32>::zeros(vec![4, 3]);
    let mut y = Dense::<f32>::zeros(vec![4, 3]);
    let report =
        run_reported(&checked, &reg, &ins, &mut [("c", &mut x as &mut dyn NDIndex<f32>)]).unwrap();
    run_reference(&checked, &reg, &ins, &mut [("c", &mut y as &mut dyn NDIndex<f32>)]).unwrap();
    assert_eq!(report.jit_reductions, 0);
    assert_eq!(report.tape_reductions, 1);
    assert!(bits_eq(&x.data, &y.data).is_none());
}

/// The materialize-once rule (§10 E3) has to survive the rewrite: a
/// reduction temp is computed at its own arity, not per output element.
#[test]
fn kernel_path_materializes_reductions_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    fn counted(args: &[f32]) -> f32 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        args[0].exp()
    }

    let mut reg = Registry::<f32>::builtins();
    reg.register_fn(linalg::lang::FnDef {
        name: "cexp",
        arity: 1,
        eval: counted,
        zero_preserving: false,
    });

    let n = 32;
    let mut v = Dense::<f32>::zeros(vec![n]);
    fill(&mut v, 15);
    let checked =
        check(&parse("soft[i] = cexp(v[i]) / sum(j: cexp(v[j]))").unwrap(), &reg).unwrap();

    let mut soft = Dense::<f32>::zeros(vec![n]);
    CALLS.store(0, Ordering::Relaxed);
    let report = run_reported(
        &checked,
        &reg,
        &[("v", &v as &dyn NDIndex<f32>)],
        &mut [("soft", &mut soft as &mut dyn NDIndex<f32>)],
    )
    .unwrap();
    assert!(report.all_kernel());
    assert_eq!(CALLS.load(Ordering::Relaxed), 2 * n);
}

/// Bind-time errors must be reported identically whichever path would
/// have run — the validation phase is shared, and this pins that down.
#[test]
fn error_surface_is_path_independent() {
    let reg = Registry::<f32>::builtins();
    let v3 = Dense::<f32>::zeros(vec![3]);
    let v4 = Dense::<f32>::zeros(vec![4]);
    let checked = check(&parse("y[i] = a[i] + b[i]").unwrap(), &reg).unwrap();
    let ins: Vec<(&str, &dyn NDIndex<f32>)> = vec![("a", &v3), ("b", &v4)];
    let mut out = Dense::<f32>::zeros(vec![3]);
    let fast = run_reported(&checked, &reg, &ins, &mut [("y", &mut out)]).unwrap_err();
    let refr = run_reference(&checked, &reg, &ins, &mut [("y", &mut out)]).unwrap_err();
    assert_eq!(fast, refr);
    assert!(matches!(fast, LangError::ExtentMismatch { .. }));
}
