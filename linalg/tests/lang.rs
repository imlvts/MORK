//! Integration tests for the tensor-expression language: the motivating
//! examples from LANGUAGE.md, differential checks against manual
//! references and the einsum VM, and the bind-time error surface.

#![cfg(feature = "dense")]

use linalg::dense::Dense;
use linalg::lang::{LangError, Registry, check, parse, parse_sexpr, run};
use linalg::tensor::NDIndex;

fn reg() -> Registry<f32> {
    Registry::<f32>::builtins()
}

/// Deterministic pseudo-random fill in [-1, 1).
fn fill(t: &mut Dense<f32>, seed: u64) {
    let mut s = seed;
    for v in t.data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
}

fn run_src(
    src: &str,
    inputs: &[(&str, &dyn NDIndex<f32>)],
    outputs: &mut [(&str, &mut dyn NDIndex<f32>)],
) -> Result<(), LangError> {
    let r = reg();
    let checked = check(&parse(src).unwrap(), &r)?;
    run(&checked, &r, inputs, outputs)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "mismatch at flat index {i}: got {g}, want {w} (tol {tol})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The motivating examples
// ─────────────────────────────────────────────────────────────────────────

/// `y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))` vs manual loops.
#[test]
fn mixed_reductions_under_a_product() {
    let (nj, nk, nl) = (4, 5, 6);
    let mut q = Dense::<f32>::zeros(vec![nk]);
    let mut x = Dense::<f32>::zeros(vec![nl]);
    let mut m = Dense::<f32>::zeros(vec![nl, nj]);
    fill(&mut q, 1);
    fill(&mut x, 2);
    fill(&mut m, 3);
    // Keep the column sums positive so sqrt stays real.
    for v in m.data.iter_mut() {
        *v = v.abs() + 0.1;
    }

    let mut y = Dense::<f32>::zeros(vec![nj, nk]);
    run_src(
        "y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))",
        &[("q", &q), ("x", &x), ("M", &m)],
        &mut [("y", &mut y)],
    )
    .unwrap();

    let xmin = x.data.iter().copied().fold(f32::INFINITY, f32::min);
    let mut want = vec![0.0f32; nj * nk];
    for j in 0..nj {
        let col: f32 = (0..nl).map(|l| m.get(&[l, j])).sum();
        for k in 0..nk {
            want[j * nk + k] = q.get(&[k]) * xmin * col.sqrt();
        }
    }
    assert_close(&y.data, &want, 1e-5);
}

/// Naive and numerically-stable softmax, both vs a manual reference.
#[test]
fn softmax_both_forms() {
    let n = 7;
    let mut v = Dense::<f32>::zeros(vec![n]);
    fill(&mut v, 42);
    for (i, x) in v.data.iter_mut().enumerate() {
        *x = *x * 3.0 + i as f32 * 0.25;
    }

    let m = v.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.data.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    let want: Vec<f32> = e.iter().map(|x| x / z).collect();

    let mut naive = Dense::<f32>::zeros(vec![n]);
    run_src(
        "soft[i] = exp(v[i]) / sum(j: exp(v[j]))",
        &[("v", &v)],
        &mut [("soft", &mut naive)],
    )
    .unwrap();
    assert_close(&naive.data, &want, 1e-6);

    let mut stable = Dense::<f32>::zeros(vec![n]);
    run_src(
        "m       = max(j: v[j])\n\
         soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))",
        &[("v", &v)],
        &mut [("soft", &mut stable)],
    )
    .unwrap();
    assert_close(&stable.data, &want, 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────
// Differential checks against the einsum VM
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn matmul_matches_einsum() {
    let (ni, nj, nk) = (5, 6, 4);
    let mut a = Dense::<f32>::zeros(vec![ni, nj]);
    let mut b = Dense::<f32>::zeros(vec![nj, nk]);
    fill(&mut a, 7);
    fill(&mut b, 8);

    let mut via_lang = Dense::<f32>::zeros(vec![ni, nk]);
    run_src(
        "c[i,k] = sum(j: a[i,j] * b[j,k])",
        &[("a", &a), ("b", &b)],
        &mut [("c", &mut via_lang)],
    )
    .unwrap();

    let mut via_einsum = Dense::<f32>::zeros(vec![ni, nk]);
    linalg::einsum::einsum_homogenous::<f32, Dense<f32>, Dense<f32>>(
        "ab,bc->ac",
        &[&a, &b],
        &mut [&mut via_einsum],
    )
    .unwrap();

    assert_eq!(via_lang.data, via_einsum.data);
}

/// Tropical (min-plus) matmul against `einsum_reduce`.
#[test]
fn tropical_matches_einsum_reduce() {
    let n = 6;
    let mut a = Dense::<f32>::zeros(vec![n, n]);
    let mut b = Dense::<f32>::zeros(vec![n, n]);
    fill(&mut a, 9);
    fill(&mut b, 10);

    let mut via_lang = Dense::<f32>::zeros(vec![n, n]);
    run_src(
        "d[i,k] = min(j: a[i,j] + b[j,k])",
        &[("a", &a), ("b", &b)],
        &mut [("d", &mut via_lang)],
    )
    .unwrap();

    let mut via_reduce = Dense::<f32>::zeros(vec![n, n]);
    linalg::einsum::einsum_reduce::<f32>(
        "ab,bc->ac",
        linalg::einsum::Reduce::Min,
        linalg::einsum::Combine::Add,
        &[&a as &dyn NDIndex<f32>, &b],
        &mut [&mut via_reduce as &mut dyn NDIndex<f32>],
    )
    .unwrap();

    assert_eq!(via_lang.data, via_reduce.data);
}

#[cfg(feature = "csr")]
#[test]
fn sparse_input_reads_as_dense_semantics() {
    use linalg::csr::Csr;
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

    let src = "c[i,k] = sum(j: a[i,j] * b[j,k])";
    let mut from_sparse = Dense::<f32>::zeros(vec![n, n]);
    run_src(src, &[("a", &a), ("b", &b)], &mut [("c", &mut from_sparse)]).unwrap();
    let mut from_dense = Dense::<f32>::zeros(vec![n, n]);
    run_src(src, &[("a", &dense_a), ("b", &b)], &mut [("c", &mut from_dense)]).unwrap();

    assert_eq!(from_sparse.data, from_dense.data);
}

// ─────────────────────────────────────────────────────────────────────────
// Semantics details
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_with_scalar_params() {
    let n = 8;
    let mut x = Dense::<f32>::zeros(vec![n]);
    fill(&mut x, 12);
    let mut eps = Dense::<f32>::zeros(vec![]);
    eps.set(&[], 1e-5);
    let mut count = Dense::<f32>::zeros(vec![]);
    count.set(&[], n as f32);

    let mut y = Dense::<f32>::zeros(vec![n]);
    run_src(
        "y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps)",
        &[("x", &x), ("n", &count), ("eps", &eps)],
        &mut [("y", &mut y)],
    )
    .unwrap();

    let ms = x.data.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let want: Vec<f32> = x.data.iter().map(|v| v * inv).collect();
    assert_close(&y.data, &want, 1e-6);
}

#[test]
fn empty_reductions_yield_identities() {
    let empty = Dense::<f32>::zeros(vec![0]);
    let mut s = Dense::<f32>::zeros(vec![]);
    run_src("s = sum(i: v[i])", &[("v", &empty)], &mut [("s", &mut s)]).unwrap();
    assert_eq!(s.get(&[]), 0.0);

    let mut m = Dense::<f32>::zeros(vec![]);
    run_src("m = max(i: v[i])", &[("v", &empty)], &mut [("m", &mut m)]).unwrap();
    assert_eq!(m.get(&[]), f32::NEG_INFINITY);

    let mut p = Dense::<f32>::zeros(vec![]);
    run_src("p = prod(i: v[i])", &[("v", &empty)], &mut [("p", &mut p)]).unwrap();
    assert_eq!(p.get(&[]), 1.0);
}

/// Broadcast-only LHS index takes its extent from the bound output (D5).
#[test]
fn broadcast_from_output_shape() {
    let ni = 3;
    let mut v = Dense::<f32>::zeros(vec![ni]);
    fill(&mut v, 13);
    let mut y = Dense::<f32>::zeros(vec![ni, 4]);
    run_src("y[i,j] = v[i]", &[("v", &v)], &mut [("y", &mut y)]).unwrap();
    for i in 0..ni {
        for j in 0..4 {
            assert_eq!(y.get(&[i, j]), v.get(&[i]));
        }
    }
}

/// Trace: repeated index within one tensor reference reads the diagonal.
#[test]
fn trace_via_diagonal_read() {
    let n = 5;
    let mut m = Dense::<f32>::zeros(vec![n, n]);
    fill(&mut m, 14);
    let mut t = Dense::<f32>::zeros(vec![]);
    run_src("t = sum(i: M[i,i])", &[("M", &m)], &mut [("t", &mut t)]).unwrap();
    let want: f32 = (0..n).map(|i| m.get(&[i, i])).sum();
    assert!((t.get(&[]) - want).abs() < 1e-6);
}

/// Temps are computed once at their own arity: the softmax denominator
/// must not be O(n²). Detect recomputation by wall-clock-free proxy — a
/// custom counting function.
#[test]
fn reductions_materialize_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    fn counted(args: &[f32]) -> f32 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        args[0].exp()
    }

    let mut r = reg();
    r.register_fn(linalg::lang::FnDef {
        name: "cexp",
        arity: 1,
        eval: counted,
        zero_preserving: false,
    });

    let n = 32;
    let mut v = Dense::<f32>::zeros(vec![n]);
    fill(&mut v, 15);
    let mut soft = Dense::<f32>::zeros(vec![n]);
    let checked = check(
        &parse("soft[i] = cexp(v[i]) / sum(j: cexp(v[j]))").unwrap(),
        &r,
    )
    .unwrap();
    CALLS.store(0, Ordering::Relaxed);
    run(&checked, &r, &[("v", &v)], &mut [("soft", &mut soft)]).unwrap();
    // Numerator: n calls. Denominator: n calls (once total, not once per i).
    assert_eq!(CALLS.load(Ordering::Relaxed), 2 * n);
}

/// Statement chains: temps feed later statements; only requested outputs
/// are written back.
#[test]
fn multi_statement_program() {
    let n = 6;
    let mut v = Dense::<f32>::zeros(vec![n]);
    fill(&mut v, 16);
    let mut y = Dense::<f32>::zeros(vec![n]);
    run_src(
        "s     = sum(i: v[i])\n\
         mean  = s / n\n\
         y[i]  = v[i] - mean",
        &[("v", &v), ("n", &{
            let mut c = Dense::<f32>::zeros(vec![]);
            c.set(&[], n as f32);
            c
        })],
        &mut [("y", &mut y)],
    )
    .unwrap();
    let mean = v.data.iter().sum::<f32>() / n as f32;
    let want: Vec<f32> = v.data.iter().map(|x| x - mean).collect();
    assert_close(&y.data, &want, 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────
// Bind-time errors
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn bind_error_surface() {
    let v3 = Dense::<f32>::zeros(vec![3]);
    let v4 = Dense::<f32>::zeros(vec![4]);
    let m = Dense::<f32>::zeros(vec![3, 4]);

    // Extent mismatch across use sites.
    let mut out = Dense::<f32>::zeros(vec![3]);
    let err = run_src(
        "y[i] = a[i] + b[i]",
        &[("a", &v3), ("b", &v4)],
        &mut [("y", &mut out)],
    )
    .unwrap_err();
    assert!(matches!(err, LangError::ExtentMismatch { .. }), "got {err:?}");

    // Unknown RHS name.
    let err = run_src("y[i] = q[i]", &[("a", &v3)], &mut [("y", &mut out)]).unwrap_err();
    assert!(matches!(err, LangError::UnknownName { .. }), "got {err:?}");

    // Rank mismatch.
    let err = run_src("y[i] = M[i]", &[("M", &m)], &mut [("y", &mut out)]).unwrap_err();
    assert!(matches!(err, LangError::RankMismatch { .. }), "got {err:?}");

    // Output not defined by any statement.
    let err = run_src("y[i] = a[i]", &[("a", &v3)], &mut [("z", &mut out)]).unwrap_err();
    assert!(matches!(err, LangError::MissingOutput { .. }), "got {err:?}");

    // Statement name colliding with an input.
    let err = run_src("a[i] = a[i]", &[("a", &v3)], &mut [("a", &mut out)]).unwrap_err();
    assert!(matches!(err, LangError::RedefinedName { .. }), "got {err:?}");

    // Binder index with no shape-bearing use site.
    let mut s = Dense::<f32>::zeros(vec![]);
    let err = run_src("s = sum(j: a[i0]) ", &[("a", &v3)], &mut [("s", &mut s)]);
    // `i0` is unbound at check time already — use a well-scoped variant:
    assert!(err.is_err());
    let err =
        run_src("s = sum(j: 1.0)", &[("a", &v3)], &mut [("s", &mut s)]).unwrap_err();
    assert!(matches!(err, LangError::UnresolvedExtent { .. }), "got {err:?}");

    // Output shape disagreeing with inferred extents.
    let mut wrong = Dense::<f32>::zeros(vec![5]);
    let err = run_src("y[i] = a[i]", &[("a", &v3)], &mut [("y", &mut wrong)]).unwrap_err();
    assert!(matches!(err, LangError::ExtentMismatch { .. }), "got {err:?}");
}

// ─────────────────────────────────────────────────────────────────────────
// Front-end isomorphism on a full program
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn sexpr_and_infix_agree_end_to_end() {
    let infix = parse(
        "m       = max(j: v[j])\n\
         soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))",
    )
    .unwrap();
    let sexpr = parse_sexpr(
        "(= m (max (j) (@ v j)))\n\
         (= (soft i) (/ (exp (- (@ v i) m)) (sum (j) (exp (- (@ v j) m)))))",
    )
    .unwrap();
    assert_eq!(infix, sexpr);
}
