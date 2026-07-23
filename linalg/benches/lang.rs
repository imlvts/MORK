//! Baseline benchmarks for the v1 tensor-expression language evaluator
//! (`linalg::lang`) — RESUME.md step 2.
//!
//! Same house style as `benches/perf.rs`: no criterion, hand-rolled
//! `Instant` timing, mean µs/iter after a warmup. Iteration counts are
//! auto-calibrated to roughly a fixed wall-clock budget per arm so the
//! very slow tree-walker configurations (256² matmul) stay bounded.
//!
//! Every group cross-checks its arms against each other on the *same*
//! inputs before timing anything: a benchmark of a wrong computation is
//! worthless. Disagreements beyond tolerance are collected and reported
//! at the end (and set a non-zero exit status).
//!
//! Sections:
//!   1. matmul `c[i,k] = sum(j: a[i,j]*b[j,k])` — lang vs einsum VM vs JIT
//!   2. softmax / rmsnorm — lang vs the gpt2 example's hand-rolled loops
//!   3. mixed reduction `y[j,k] = q[k]*min(l: x[l])*sqrt(sum(l: M[l,j]))`
//!   4. multi-statement stable softmax — statement/temp overhead
//!
//! Run with `cargo bench --bench lang` (add `--features jit` for the JIT
//! arm of section 1).

use std::time::{Duration, Instant};

use linalg::dense::Dense;
use linalg::einsum::einsum_homogenous;
use linalg::lang::{Checked, Registry, RunOptions, check, parse, run, run_with};
use linalg::tensor::NDIndex;

// ─── timing ─────────────────────────────────────────────────────────────

/// Wall-clock budget per timed arm (after calibration).
const TARGET: Duration = Duration::from_millis(300);
/// Budget for the calibration/warmup phase.
const CALIBRATE: Duration = Duration::from_millis(60);

/// Time `f`, reporting the mean µs/iter like `benches/perf.rs` does.
///
/// The calibration loop doubles as the warmup: it runs `f` for at least
/// `CALIBRATE` (or one iteration, whichever is longer), then picks an
/// iteration count aiming at `TARGET` — minimum 3 iterations, so a very
/// slow arm costs about `4 ×` one call rather than a fixed large count.
fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
    let warm = Instant::now();
    let mut n = 0u64;
    loop {
        f();
        n += 1;
        if warm.elapsed() >= CALIBRATE || n >= 2000 {
            break;
        }
    }
    let per = warm.elapsed().as_secs_f64() / n as f64;
    let iters = ((TARGET.as_secs_f64() / per) as u64).clamp(3, 200_000);

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per_iter_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
    println!("  {name:44} {per_iter_us:12.3} µs/iter  ({iters} iters)");
    per_iter_us
}

// ─── data ───────────────────────────────────────────────────────────────

/// Same LCG fill as `tests/lang.rs` — values in [-1, 1).
fn fill(t: &mut Dense<f32>, seed: u64) {
    let mut s = seed;
    for v in t.data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
}

fn filled(shape: Vec<usize>, seed: u64) -> Dense<f32> {
    let mut t = Dense::<f32>::zeros(shape);
    fill(&mut t, seed);
    t
}

fn scalar(v: f32) -> Dense<f32> {
    let mut t = Dense::<f32>::zeros(vec![]);
    t.set(&[], v);
    t
}

// ─── correctness cross-checks ───────────────────────────────────────────

/// Max absolute and max relative deviation between two result buffers.
fn deviation(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        let scale = x.abs().max(y.abs());
        if scale > 0.0 {
            max_rel = max_rel.max(d / scale);
        }
    }
    (max_abs, max_rel)
}

/// Compare two arms' outputs; record (don't panic on) a disagreement so
/// the rest of the run still produces numbers.
fn compare(label: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32, bad: &mut Vec<String>) {
    if a.len() != b.len() {
        let msg = format!("{label}: length mismatch {} vs {}", a.len(), b.len());
        println!("  !! {msg}");
        bad.push(msg);
        return;
    }
    let (max_abs, max_rel) = deviation(a, b);
    let ok = a
        .iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs().max(x.abs()));
    let tag = if ok { "ok " } else { "!! " };
    println!("  {tag}{label:40} max|Δ| {max_abs:.3e}  max rel {max_rel:.3e}");
    if !ok {
        bad.push(format!(
            "{label}: max|Δ| {max_abs:.3e}, max rel {max_rel:.3e} (atol {atol:.1e}, rtol {rtol:.1e})"
        ));
    }
}

// ─── the gpt2 example's hand-rolled reference loops ─────────────────────
// Copied verbatim from `examples/gpt2/main.rs` (the example is not
// refactored — these are the baselines section 2 measures against).

fn map(x: &Dense<f32>, f: impl Fn(f32) -> f32) -> Dense<f32> {
    Dense { data: x.data.iter().map(|&v| f(v)).collect(), shape: x.shape.clone() }
}

/// Row-vector RMSNorm: `x / sqrt(mean(x²) + eps)`. No learned gain.
fn rmsnorm(x: &Dense<f32>, eps: f32) -> Dense<f32> {
    let ms = x.data.iter().map(|&v| v * v).sum::<f32>() / x.data.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    map(x, |v| v * inv)
}

/// Softmax over the last axis (each contiguous run of `last` elements).
fn softmax_last(x: &Dense<f32>) -> Dense<f32> {
    let last = *x.shape.last().unwrap();
    let mut data = x.data.clone();
    for row in data.chunks_mut(last) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    Dense { data, shape: x.shape.clone() }
}

// ─── lang driver helper ─────────────────────────────────────────────────

fn compile(src: &str, reg: &Registry<f32>) -> Checked {
    check(&parse(src).unwrap(), reg).unwrap()
}

// ─── 1. matmul: lang vs einsum VM vs JIT ────────────────────────────────

fn section_matmul(bad: &mut Vec<String>) {
    println!("\n=== 1. matmul  c[i,k] = sum(j: a[i,j] * b[j,k])  (einsum \"ab,bc->ac\") ===");
    let reg = Registry::<f32>::builtins();
    let prog = compile("c[i,k] = sum(j: a[i,j] * b[j,k])", &reg);

    for &n in &[16usize, 64, 256] {
        println!("\n--- {n}×{n} ---");
        let a = filled(vec![n, n], 7);
        let b = filled(vec![n, n], 8);

        // Reference results, computed once, outside any timing.
        let mut c_lang = Dense::<f32>::zeros(vec![n, n]);
        run(&prog, &reg, &[("a", &a), ("b", &b)], &mut [("c", &mut c_lang)]).unwrap();
        let mut c_vm = Dense::<f32>::zeros(vec![n, n]);
        einsum_homogenous::<f32, _, _>("ab,bc->ac", &[&a, &b], &mut [&mut c_vm]).unwrap();

        // Contraction length n over values in [-1,1): |c| ~ sqrt(n)/3.
        let atol = 1e-6 * n as f32;
        compare(
            &format!("matmul {n}²: lang vs VM"),
            &c_lang.data,
            &c_vm.data,
            atol,
            1e-5,
            bad,
        );

        let lang_us = bench("lang evaluator (run)", || {
            let mut c = Dense::<f32>::zeros(vec![n, n]);
            run(&prog, &reg, &[("a", &a), ("b", &b)], &mut [("c", &mut c)]).unwrap();
            std::hint::black_box(&c);
        });
        let vm_us = bench("einsum VM (einsum_homogenous)", || {
            let mut c = Dense::<f32>::zeros(vec![n, n]);
            einsum_homogenous::<f32, _, _>("ab,bc->ac", &[&a, &b], &mut [&mut c]).unwrap();
            std::hint::black_box(&c);
        });
        println!("  lang / VM: {:.2}×", lang_us / vm_us);

        #[cfg(feature = "jit")]
        {
            use linalg::jit::{EinsumF32Jit, JitInput};
            let jit = EinsumF32Jit::compile(
                "ab,bc->ac",
                &[JitInput::Dense(&a), JitInput::Dense(&b)],
                &[vec![n, n]],
            )
            .unwrap();
            let mut c_jit = Dense::<f32>::zeros(vec![n, n]);
            jit.run(&[JitInput::Dense(&a), JitInput::Dense(&b)], &mut [&mut c_jit]);
            compare(
                &format!("matmul {n}²: lang vs JIT"),
                &c_lang.data,
                &c_jit.data,
                atol,
                1e-5,
                bad,
            );
            let jit_us = bench("einsum JIT (EinsumF32Jit::run)", || {
                let mut c = Dense::<f32>::zeros(vec![n, n]);
                jit.run(&[JitInput::Dense(&a), JitInput::Dense(&b)], &mut [&mut c]);
                std::hint::black_box(&c);
            });
            println!("  lang / JIT: {:.2}×", lang_us / jit_us);
        }
    }
}

// ─── 2. softmax and rmsnorm vs the hand-rolled loops ────────────────────

fn section_softmax(bad: &mut Vec<String>) {
    println!("\n=== 2a. softmax  soft[i] = exp(v[i]) / sum(j: exp(v[j])) ===");
    let reg = Registry::<f32>::builtins();
    let prog = compile("soft[i] = exp(v[i]) / sum(j: exp(v[j]))", &reg);

    for &n in &[64usize, 1024, 65536] {
        println!("\n--- n = {n} ---");
        let v = filled(vec![n], 42);

        let mut s_lang = Dense::<f32>::zeros(vec![n]);
        run(&prog, &reg, &[("v", &v)], &mut [("soft", &mut s_lang)]).unwrap();
        let s_ref = softmax_last(&v);
        // The lang form is the *naive* softmax, the gpt2 loop subtracts the
        // row max first; on inputs in [-1,1) they agree to float rounding.
        compare(
            &format!("softmax n={n}: lang vs hand-rolled"),
            &s_lang.data,
            &s_ref.data,
            1e-9,
            1e-5,
            bad,
        );

        let lang_us = bench("lang evaluator (run)", || {
            let mut s = Dense::<f32>::zeros(vec![n]);
            run(&prog, &reg, &[("v", &v)], &mut [("soft", &mut s)]).unwrap();
            std::hint::black_box(&s);
        });
        let ref_us = bench("hand-rolled softmax_last (gpt2)", || {
            let s = softmax_last(&v);
            std::hint::black_box(&s);
        });
        println!("  lang / hand-rolled: {:.2}×", lang_us / ref_us);
    }

    println!("\n=== 2b. rmsnorm  y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps) ===");
    let prog = compile("y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps)", &reg);
    const EPS: f32 = 1e-5;

    for &n in &[64usize, 1024, 65536] {
        println!("\n--- n = {n} ---");
        let x = filled(vec![n], 43);
        let count = scalar(n as f32);
        let eps = scalar(EPS);

        let mut y_lang = Dense::<f32>::zeros(vec![n]);
        run(
            &prog,
            &reg,
            &[("x", &x), ("n", &count), ("eps", &eps)],
            &mut [("y", &mut y_lang)],
        )
        .unwrap();
        let y_ref = rmsnorm(&x, EPS);
        // Lang divides per element; the hand-rolled loop multiplies by a
        // precomputed reciprocal — a ~1 ulp difference.
        compare(
            &format!("rmsnorm n={n}: lang vs hand-rolled"),
            &y_lang.data,
            &y_ref.data,
            1e-9,
            1e-5,
            bad,
        );

        let lang_us = bench("lang evaluator (run)", || {
            let mut y = Dense::<f32>::zeros(vec![n]);
            run(
                &prog,
                &reg,
                &[("x", &x), ("n", &count), ("eps", &eps)],
                &mut [("y", &mut y)],
            )
            .unwrap();
            std::hint::black_box(&y);
        });
        let ref_us = bench("hand-rolled rmsnorm (gpt2)", || {
            let y = rmsnorm(&x, EPS);
            std::hint::black_box(&y);
        });
        println!("  lang / hand-rolled: {:.2}×", lang_us / ref_us);
    }
}

// ─── 3. mixed reduction (new capability, no library baseline) ───────────

fn section_mixed(bad: &mut Vec<String>) {
    // Sizes: nj = 256 (M columns / y rows), nk = 256 (q / y cols),
    // nl = 1024 (both reduction extents). Work: one nl-scalar min, an
    // nl×nj sum into an nj temp, then nj×nk output elements.
    const NJ: usize = 256;
    const NK: usize = 256;
    const NL: usize = 1024;
    println!(
        "\n=== 3. mixed reduction  y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j])) ===\
         \n--- nj={NJ} nk={NK} nl={NL} ---"
    );

    let reg = Registry::<f32>::builtins();
    let prog = compile("y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))", &reg);

    let q = filled(vec![NK], 1);
    let x = filled(vec![NL], 2);
    let mut m = filled(vec![NL, NJ], 3);
    // Keep the column sums positive so sqrt stays real.
    for v in m.data.iter_mut() {
        *v = v.abs() + 0.1;
    }

    let mut y_lang = Dense::<f32>::zeros(vec![NJ, NK]);
    run(
        &prog,
        &reg,
        &[("q", &q), ("x", &x), ("M", &m)],
        &mut [("y", &mut y_lang)],
    )
    .unwrap();

    // No library baseline exists for this shape of expression; the manual
    // loop below is written here purely as the correctness oracle. It is
    // also timed, clearly labelled, as an "ideal hand-written" reference.
    let manual = || {
        let xmin = x.data.iter().copied().fold(f32::INFINITY, f32::min);
        let mut out = Dense::<f32>::zeros(vec![NJ, NK]);
        for j in 0..NJ {
            let col: f32 = (0..NL).map(|l| m.data[l * NJ + j]).sum();
            let s = xmin * col.sqrt();
            for k in 0..NK {
                out.data[j * NK + k] = q.data[k] * s;
            }
        }
        out
    };
    let y_ref = manual();
    compare(
        "mixed: lang vs manual loops",
        &y_lang.data,
        &y_ref.data,
        1e-5,
        1e-4,
        bad,
    );

    let lang_us = bench("lang evaluator (run)", || {
        let mut y = Dense::<f32>::zeros(vec![NJ, NK]);
        run(
            &prog,
            &reg,
            &[("q", &q), ("x", &x), ("M", &m)],
            &mut [("y", &mut y)],
        )
        .unwrap();
        std::hint::black_box(&y);
    });
    let ref_us = bench("manual loops (correctness oracle only)", || {
        std::hint::black_box(manual());
    });
    println!("  lang / manual: {:.2}×", lang_us / ref_us);
}

// ─── 4. multi-statement program: stable softmax ─────────────────────────

fn section_multi_statement(bad: &mut Vec<String>) {
    println!("\n=== 4. statement/temp overhead: stable softmax ===");
    println!("  naive-1stmt : soft[i] = exp(v[i]) / sum(j: exp(v[j]))");
    println!("  stable-2stmt: m = max(j: v[j]) ; soft[i] = exp(v[i]-m) / sum(j: exp(v[j]-m))");
    println!("  stable-1stmt: same, inlined (recomputes max(j: v[j]) twice — no CSE yet)");

    let reg = Registry::<f32>::builtins();
    let naive = compile("soft[i] = exp(v[i]) / sum(j: exp(v[j]))", &reg);
    let stable2 = compile(
        "m       = max(j: v[j])\n\
         soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))",
        &reg,
    );
    let stable1 = compile(
        "soft[i] = exp(v[i] - max(j: v[j])) / sum(j: exp(v[j] - max(k: v[k])))",
        &reg,
    );

    for &n in &[64usize, 1024, 65536] {
        println!("\n--- n = {n} ---");
        let v = filled(vec![n], 42);

        let mut s_naive = Dense::<f32>::zeros(vec![n]);
        run(&naive, &reg, &[("v", &v)], &mut [("soft", &mut s_naive)]).unwrap();
        let mut s2 = Dense::<f32>::zeros(vec![n]);
        run(&stable2, &reg, &[("v", &v)], &mut [("soft", &mut s2)]).unwrap();
        let mut s1 = Dense::<f32>::zeros(vec![n]);
        run(&stable1, &reg, &[("v", &v)], &mut [("soft", &mut s1)]).unwrap();
        let s_ref = softmax_last(&v);

        compare(
            &format!("stable-2stmt n={n}: vs hand-rolled"),
            &s2.data,
            &s_ref.data,
            1e-9,
            1e-5,
            bad,
        );
        compare(
            &format!("stable-1stmt n={n}: vs stable-2stmt"),
            &s1.data,
            &s2.data,
            1e-9,
            1e-5,
            bad,
        );
        compare(
            &format!("naive-1stmt  n={n}: vs stable-2stmt"),
            &s_naive.data,
            &s2.data,
            1e-9,
            1e-5,
            bad,
        );

        let naive_us = bench("lang naive softmax (1 stmt, 1 reduce)", || {
            let mut s = Dense::<f32>::zeros(vec![n]);
            run(&naive, &reg, &[("v", &v)], &mut [("soft", &mut s)]).unwrap();
            std::hint::black_box(&s);
        });
        let s2_us = bench("lang stable softmax (2 stmts, 2 reduces)", || {
            let mut s = Dense::<f32>::zeros(vec![n]);
            run(&stable2, &reg, &[("v", &v)], &mut [("soft", &mut s)]).unwrap();
            std::hint::black_box(&s);
        });
        let s1_us = bench("lang stable softmax (1 stmt, 3 reduces)", || {
            let mut s = Dense::<f32>::zeros(vec![n]);
            run(&stable1, &reg, &[("v", &v)], &mut [("soft", &mut s)]).unwrap();
            std::hint::black_box(&s);
        });
        println!("  stable-2stmt / naive-1stmt: {:.2}×", s2_us / naive_us);
        println!("  stable-1stmt / stable-2stmt: {:.2}×", s1_us / s2_us);
    }
}

// ─── main ───────────────────────────────────────────────────────────────

// ─── 5. dynamic axes (RESUME step 6) ────────────────────────────────────

/// What a *runtime* extent costs on a hot kernel.
///
/// Marking an axis dynamic buys one compiled kernel per contraction site
/// instead of one per shape; the price is that its loop bound is a loaded
/// value rather than an immediate, and every stride that is a product
/// involving it becomes a multiply instead of a folded constant. matmul is
/// the worst case in this file for that — a tight three-deep nest with
/// nothing else to hide behind — so this is where to look for a regression.
///
/// Bit equality is asserted, not merely tolerated: a dynamic axis must
/// change what is a constant and nothing else.
fn section_dynamic(bad: &mut Vec<String>) {
    println!("\n=== 5. dynamic axes: c[i,k] = sum(j: a[i,j] * b[j,k]) with runtime extents ===");
    let reg = Registry::<f32>::builtins();
    let prog = compile("c[i,k] = sum(j: a[i,j] * b[j,k])", &reg);

    // `j` alone is the KV-cache case (the contraction length grows); all
    // three is the pessimal case, where no extent is a constant anywhere.
    let cases: [(&str, &[&str]); 3] =
        [("static (baked extents)", &[]), ("dynamic j", &["j"]), ("dynamic i,j,k", &["i", "j", "k"])];

    for &n in &[16usize, 64, 256] {
        println!("\n--- {n}×{n} ---");
        let a = filled(vec![n, n], 7);
        let b = filled(vec![n, n], 8);

        let mut base: Vec<u32> = Vec::new();
        let mut static_us = 0.0;
        for (label, dynamic) in cases {
            let mut c = Dense::<f32>::zeros(vec![n, n]);
            run_with(
                &prog,
                &reg,
                &[("a", &a), ("b", &b)],
                &mut [("c", &mut c)],
                RunOptions { dynamic },
            )
            .unwrap();
            let bits: Vec<u32> = c.data.iter().map(|v| v.to_bits()).collect();
            if base.is_empty() {
                base = bits;
            } else if bits != base {
                let msg = format!("matmul {n}²: {label} is not bit-identical to the static run");
                println!("  !! {msg}");
                bad.push(msg);
            }

            let us = bench(label, || {
                let mut c = Dense::<f32>::zeros(vec![n, n]);
                run_with(
                    &prog,
                    &reg,
                    &[("a", &a), ("b", &b)],
                    &mut [("c", &mut c)],
                    RunOptions { dynamic },
                )
                .unwrap();
                std::hint::black_box(&c);
            });
            if dynamic.is_empty() {
                static_us = us;
            } else {
                println!("  {label} / static: {:.2}×", us / static_us);
            }
        }
    }
}

fn main() {
    println!("=== linalg::lang v1 evaluator bench ===");
    println!("(mean µs/iter, iteration counts auto-calibrated to ~300 ms per arm)");
    #[cfg(feature = "jit")]
    println!("(jit feature: ON)");
    #[cfg(not(feature = "jit"))]
    println!("(jit feature: OFF — section 1 JIT arm skipped)");

    let mut bad: Vec<String> = Vec::new();
    section_matmul(&mut bad);
    section_softmax(&mut bad);
    section_mixed(&mut bad);
    section_multi_statement(&mut bad);
    section_dynamic(&mut bad);

    println!("\n=== correctness summary ===");
    if bad.is_empty() {
        println!("  all arms agree within tolerance");
    } else {
        println!("  {} DISAGREEMENT(S):", bad.len());
        for b in &bad {
            println!("    - {b}");
        }
        std::process::exit(1);
    }
    println!("\n=== done ===");
}
