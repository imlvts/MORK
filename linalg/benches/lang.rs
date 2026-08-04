//! Benchmarks for the tensor-expression language (`linalg::lang`) — the
//! kernel-backed fast path of RESUME.md steps 3–6, measured against
//! hand-written Rust and against the machine's own limits.
//!
//! House style is `benches/perf.rs`: no criterion, hand-rolled `Instant`
//! timing. Each arm is warmed once (untimed — this is where a Cranelift
//! compile and the first-touch page faults happen), calibrated for
//! ~60 ms, then run as several timed batches; the **median** batch is
//! reported, with the batch count, the iteration count per batch and the
//! fastest batch, so a two-iteration statistic is visibly a
//! two-iteration statistic.
//!
//! Two things this file is careful about, because the numbers are
//! otherwise easy to misread:
//!
//! * **Size.** Microsecond-scale arms are dominated by the ~1 µs per-`run`
//!   fixed cost (bind + lower + build + kernel-cache lookup). Section 1
//!   measures that constant directly, at `n = 1`, once per program, and
//!   every later table subtracts it explicitly. The large sizes (matmul up
//!   to 1024², elementwise up to 4 Mi elements) exist so that there is at
//!   least one column where the constant is irrelevant.
//! * **Matched programs.** A ratio between two arms is only meaningful if
//!   they compute the same thing. `x / d` is not `x * (1/d)`, and
//!   `exp(v[i]) / sum(j: exp(v[j]))` calls `exp` `2n` times where a
//!   hand-rolled softmax calls it `n` times. Section 4 therefore runs
//!   *both* forms on *both* sides and decomposes the total gap into
//!   setup / divide / double-`exp` / buffer discipline / leftover.
//!
//! Every group cross-checks its arms against each other on the same
//! inputs before timing anything; the summary at the end says which pairs
//! were **bit-identical** (every `f32` bit pattern equal) and which were
//! merely close, so a loosened tolerance cannot hide silently.
//!
//! Sections:
//!   0. machine reference kernels — copy / scale / divide / sum / `exp`
//!   1. per-`run` fixed cost, per program, at `n = 1`
//!   2. matmul `c[i,k] = sum(j: a[i,j]*b[j,k])` — lang vs einsum VM vs JIT
//!   3. mixed reduction `y[j,k] = q[k]*min(l: x[l])*sqrt(sum(l: M[l,j]))`
//!   4a. rmsnorm, divide vs hoisted reciprocal, gap decomposed
//!   4b. softmax, `2n` vs `n` `exp`, gap decomposed
//!   4c. single-kernel breakdown — where 4a/4b's leftover actually is
//!   5. multi-statement stable softmax — statement/temp overhead
//!   6. dynamic axes (RESUME step 6)
//!   7. every benchmarked program as nested loops (`Checked::explain`)
//!   1b. the fixed cost re-measured after the suite (cache-growth check)
//!
//! Run with `cargo bench --bench lang` (add `--features jit` for the JIT
//! arm of section 2). `LANG_BENCH_SECTIONS=0,4` runs a subset;
//! `LANG_BENCH_SECTIONS=7` prints the program listing and measures
//! nothing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use linalg::dense::Dense;
use linalg::einsum::einsum_homogenous;
use linalg::lang::{Checked, Registry, RunOptions, check, parse, run, run_reported, run_with};
use linalg::tensor::NDIndex;

// ─── timing ─────────────────────────────────────────────────────────────

/// Wall-clock budget per timed arm, split across batches.
const TARGET: Duration = Duration::from_millis(300);
/// Budget for the calibration phase (which is also a second warmup).
const CALIBRATE: Duration = Duration::from_millis(60);
/// An arm slower than this gets fewer, larger batches.
const SLOW: f64 = 0.02;

struct Timing {
    /// Median batch, µs per iteration.
    us: f64,
    /// Fastest batch, µs per iteration.
    min: f64,
    iters: u64,
    batches: usize,
}

/// Warm, calibrate, then time `batches` batches and take the median.
///
/// The untimed warmup call matters at these sizes: it pays the Cranelift
/// compile, the first-touch faults on a freshly allocated output and the
/// kernel-cache miss, none of which recur.
fn measure<F: FnMut()>(f: &mut F) -> Timing {
    f();

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

    let batches = if per >= SLOW { 3 } else { 5 };
    let iters = ((TARGET.as_secs_f64() / batches as f64 / per) as u64).clamp(1, 200_000);

    let mut runs = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        runs.push(start.elapsed().as_secs_f64() / iters as f64 * 1e6);
    }
    runs.sort_by(f64::total_cmp);
    Timing { us: runs[batches / 2], min: runs[0], iters, batches }
}

fn line(name: &str, t: &Timing, extra: &str) {
    println!(
        "  {name:46} {:11.3} µs  [{}×{}, min {:9.3}]{extra}",
        t.us, t.batches, t.iters, t.min
    );
}

/// Time an arm, reporting the median µs/iter.
fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
    let t = measure(&mut f);
    line(name, &t, "");
    t.us
}

/// Time an arm over `n` elements, additionally reporting ns/element and
/// effective bandwidth against a stated bytes-per-element model.
fn bench_bw<F: FnMut()>(name: &str, n: usize, bytes_per_elem: f64, mut f: F) -> f64 {
    let t = measure(&mut f);
    let ns_elem = t.us * 1000.0 / n as f64;
    let gbs = bytes_per_elem * n as f64 / (t.us * 1e-6) / 1e9;
    line(name, &t, &format!("  {ns_elem:7.3} ns/elem  {gbs:7.2} GB/s"));
    t.us
}

/// Time an arm doing `flops` floating-point operations, reporting GFLOP/s.
fn bench_flops<F: FnMut()>(name: &str, flops: f64, mut f: F) -> f64 {
    let t = measure(&mut f);
    let gflops = flops / (t.us * 1e-6) / 1e9;
    line(name, &t, &format!("  {gflops:8.3} GFLOP/s"));
    t.us
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

#[derive(Default)]
struct Checks {
    bad: Vec<String>,
    identical: Vec<String>,
    close: Vec<String>,
}

impl Checks {
    /// Compare two arms' outputs. Records — rather than panics on — a
    /// disagreement, so the rest of the run still produces numbers, and
    /// records *how* they agreed: bit-for-bit, or only within tolerance.
    fn cmp(&mut self, label: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
        if a.len() != b.len() {
            let msg = format!("{label}: length mismatch {} vs {}", a.len(), b.len());
            println!("  !! {msg}");
            self.bad.push(msg);
            return;
        }
        let bits = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
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
        if bits == 0 {
            println!("  ok  {label:52} bit-identical ({} elems)", a.len());
            self.identical.push(label.to_string());
            return;
        }
        let ok = a
            .iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs().max(x.abs()));
        let tag = if ok { "ok " } else { "!! " };
        println!(
            "  {tag} {label:52} {bits}/{} differ, max|Δ| {max_abs:.3e}, max rel {max_rel:.3e}",
            a.len()
        );
        if ok {
            self.close.push(format!("{label} (max rel {max_rel:.2e}, {bits}/{} elems)", a.len()));
        } else {
            self.bad.push(format!(
                "{label}: max|Δ| {max_abs:.3e}, max rel {max_rel:.3e} (atol {atol:.1e}, rtol {rtol:.1e})"
            ));
        }
    }
}

// ─── hand-written reference loops ───────────────────────────────────────
//
// The gpt2 example's helpers, rewritten to take a caller-owned output
// buffer. Timing an arm that allocates its own output measures the
// allocator (16 MB of fresh pages at n = 4 Mi), so every arm in this file
// — language and hand-written alike — writes into a buffer allocated once,
// outside the timed region. `*_langbuf` variants deliberately put the
// allocation back, to price the language's buffer discipline.

/// `x * (1/sqrt(mean(x²) + eps))` — the gpt2 `rmsnorm`, reciprocal hoisted.
fn rmsnorm_mul_into(x: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|&v| v * v).sum::<f32>();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v * inv;
    }
}

/// The same, dividing per element — matched to the language's `x[i] / …`.
fn rmsnorm_div_into(x: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|&v| v * v).sum::<f32>();
    let d = (ss / x.len() as f32 + eps).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v / d;
    }
}

/// [`rmsnorm_mul_into`] with the language's buffer discipline: the result
/// is built in a freshly allocated, zero-filled `Vec` and then copied to
/// the destination (`fast.rs` allocates `vec![T::ZERO; n]` per statement
/// and finishes with one `copy_from_slice`).
fn rmsnorm_mul_langbuf(x: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|&v| v * v).sum::<f32>();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    let mut tmp = vec![0.0f32; x.len()];
    for (o, &v) in tmp.iter_mut().zip(x) {
        *o = v * inv;
    }
    out.copy_from_slice(&tmp);
}

/// `out[i] = x[i] * c` written the way `fast.rs`'s tape actually executes
/// it, op for op (`exec_map` + `run_inner`, read off the source):
///
/// 1. one SSA block register per tape op, 64 elements each;
/// 2. `InOp::Gather` — `copy_from_slice` the input block;
/// 3. `InOp::Splat` — `fill` a whole block with the outer-tape scalar,
///    once per block (the scalar is not an operand, it is a register);
/// 4. `InOp::Bin{Mul}` — read two blocks, write a third;
/// 5. `copy_from_slice` the root block into a freshly allocated,
///    zero-filled statement buffer; and finally
/// 6. `copy_from_slice` that buffer into the destination.
///
/// None of this is interpretation overhead — it is the *shape* of the
/// work the design generates. If this arm lands on the language's number,
/// the gap is explained by the shape and not by codegen quality.
fn scale_tape_shape(x: &[f32], c: f32, out: &mut [f32]) {
    const BLK: usize = 64;
    let mut tmp = vec![0.0f32; x.len()];
    let mut gather = [0.0f32; BLK];
    let mut splat = [0.0f32; BLK];
    let mut prod = [0.0f32; BLK];
    for (bi, chunk) in x.chunks(BLK).enumerate() {
        let m = chunk.len();
        gather[..m].copy_from_slice(chunk);
        splat[..m].fill(c);
        for (d, (p, q)) in prod[..m].iter_mut().zip(gather[..m].iter().zip(&splat[..m])) {
            *d = *p * *q;
        }
        tmp[bi * BLK..bi * BLK + m].copy_from_slice(&prod[..m]);
    }
    out.copy_from_slice(&tmp);
}

/// Naive softmax, `n` calls to `exp`, per-element **divide**.
fn softmax_naive_div_into(x: &[f32], out: &mut [f32]) {
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(x) {
        let e = v.exp();
        *o = e;
        sum += e;
    }
    for o in out.iter_mut() {
        *o = *o / sum;
    }
}

/// Naive softmax, `n` calls to `exp`, hoisted reciprocal.
fn softmax_naive_mul_into(x: &[f32], out: &mut [f32]) {
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(x) {
        let e = v.exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// The gpt2 `softmax_last` (stable: max pass, `n` calls to `exp`, hoisted
/// reciprocal), over a single row, without the input clone.
fn softmax_stable_into(x: &[f32], out: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(x) {
        let e = (v - m).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// [`softmax_stable_into`] with the *pass structure* a three-statement
/// program has: the language materializes `e[i] = exp(v[i]-m)` before the
/// reduction over it can start, so the `exp` pass and the summing pass
/// cannot be fused the way a hand-written loop fuses them. Four passes
/// (max, exp, sum, scale) instead of three. Buffers are still the
/// caller's, so this isolates *fusion* from *allocation*.
fn softmax_stable_unfused(x: &[f32], e: &mut [f32], out: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for (o, &v) in e.iter_mut().zip(x) {
        *o = (v - m).exp();
    }
    let mut sum = 0.0f32;
    for &v in e.iter() {
        sum += v;
    }
    let inv = 1.0 / sum;
    for (o, &v) in out.iter_mut().zip(e.iter()) {
        *o = v * inv;
    }
}

/// [`softmax_stable_unfused`] plus the language's buffer discipline: one
/// freshly allocated, zero-filled `Vec` per statement (`fast.rs` does
/// `vec![T::ZERO; n]` for every statement result) and one final
/// `copy_from_slice` into the destination.
fn softmax_stable_unfused_langbuf(x: &[f32], out: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut e = vec![0.0f32; x.len()];
    for (o, &v) in e.iter_mut().zip(x) {
        *o = (v - m).exp();
    }
    let mut sum = 0.0f32;
    for &v in e.iter() {
        sum += v;
    }
    let inv = 1.0 / sum;
    let mut scaled = vec![0.0f32; x.len()];
    for (o, &v) in scaled.iter_mut().zip(e.iter()) {
        *o = v * inv;
    }
    out.copy_from_slice(&scaled);
}

// ─── the program sources, in one place ──────────────────────────────────

/// Every program this file benchmarks. Each compile site below draws its
/// source from here, and section 7 prints the loop-nest lowering of
/// [`SOURCES`] — so "which programs does this bench run?" has exactly one
/// answer, and adding an arm means adding an entry.
mod src {
    pub const TRIVIAL: &str = "y[i] = x[i]";
    pub const MATMUL: &str = "c[i,k] = sum(j: a[i,j] * b[j,k])";
    pub const RMS_DIV: &str = "y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps)";
    pub const RMS_MUL: &str = "y[i] = x[i] * (1.0 / sqrt(sum(j: x[j]*x[j]) / n + eps))";
    pub const SM_2N: &str = "soft[i] = exp(v[i]) / sum(j: exp(v[j]))";
    pub const SM_N_DIV: &str = "e[i] = exp(v[i])\nsoft[i] = e[i] / sum(j: e[j])";
    pub const SM_N_MUL: &str = "e[i] = exp(v[i])\nsoft[i] = e[i] * (1.0 / sum(j: e[j]))";
    pub const SM_STABLE: &str = "m       = max(j: v[j])\n\
                                 e[i]    = exp(v[i] - m)\n\
                                 soft[i] = e[i] * (1.0 / sum(j: e[j]))";
    pub const RED_SUMSQ: &str = "s = sum(j: x[j]*x[j])";
    pub const MAP_SCALE: &str = "y[i] = x[i] * c";
    pub const MAP_EXP: &str = "y[i] = exp(x[i])";
    pub const MIXED: &str = "y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))";
    /// Section 5's stable softmax written as two statements — the same
    /// math as [`SM_STABLE`], one statement fewer and dividing rather
    /// than multiplying by the reciprocal.
    pub const SM_STABLE_2: &str = "m       = max(j: v[j])\n\
                                   soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))";
    /// The same inlined into one statement, which recomputes the maximum
    /// twice — there is no CSE, and section 7's print shows the two
    /// separate temporaries that result.
    pub const SM_STABLE_1: &str =
        "soft[i] = exp(v[i] - max(j: v[j])) / sum(j: exp(v[j] - max(k: v[k])))";

    /// Every source above, labelled — the section 7 listing.
    pub const SOURCES: &[(&str, &str)] = &[
        ("trivial (map)", TRIVIAL),
        ("matmul", MATMUL),
        ("rmsnorm, divide", RMS_DIV),
        ("rmsnorm, hoisted reciprocal", RMS_MUL),
        ("softmax, 2n exp, divide", SM_2N),
        ("softmax, n exp, divide (2 stmts)", SM_N_DIV),
        ("softmax, n exp, reciprocal (2 stmts)", SM_N_MUL),
        ("softmax, stable, matched (3 stmts)", SM_STABLE),
        ("softmax, stable (2 stmts)", SM_STABLE_2),
        ("softmax, stable, inlined (1 stmt)", SM_STABLE_1),
        ("reduction only, sum of squares", RED_SUMSQ),
        ("map only, scale", MAP_SCALE),
        ("map only, exp", MAP_EXP),
        ("mixed reduction", MIXED),
    ];
}

// ─── lang driver helpers ────────────────────────────────────────────────

fn compile(src: &str, reg: &Registry<f32>) -> Checked {
    check(&parse(src).unwrap(), reg).unwrap()
}

/// Print which execution path a program actually took — the evidence for
/// any claim about *why* an arm costs what it costs.
fn print_report(label: &str, rep: &linalg::lang::RunReport) {
    println!(
        "  path {label:34} stmts {}/{} on kernels, jit-red {}, tape-red {}, \
         store-maps {}, inline-fn {}, indirect-fn {}",
        rep.kernel_statements,
        rep.statements,
        rep.jit_reductions,
        rep.tape_reductions,
        rep.store_maps,
        rep.inline_fn_ops,
        rep.indirect_fn_ops
    );
}

fn show_path_softmax(label: &str, checked: &Checked, reg: &Registry<f32>, n: usize) {
    let v = filled(vec![n], 42);
    let mut out = Dense::<f32>::zeros(vec![n]);
    let rep = run_reported(checked, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
    print_report(label, &rep);
}

fn show_path_reduce(label: &str, checked: &Checked, reg: &Registry<f32>, n: usize) {
    let x = filled(vec![n], 43);
    let mut s = Dense::<f32>::zeros(vec![]);
    let rep = run_reported(checked, reg, &[("x", &x)], &mut [("s", &mut s)]).unwrap();
    print_report(label, &rep);
}

fn show_path_rms(label: &str, checked: &Checked, reg: &Registry<f32>, n: usize) {
    let x = filled(vec![n], 43);
    let count = scalar(n as f32);
    let eps = scalar(1e-5);
    let mut out = Dense::<f32>::zeros(vec![n]);
    let rep = run_reported(
        checked,
        reg,
        &[("x", &x), ("n", &count), ("eps", &eps)],
        &mut [("y", &mut out)],
    )
    .unwrap();
    print_report(label, &rep);
}

// ─── 0. machine reference kernels ───────────────────────────────────────

/// What the hardware does on the shapes the elementwise arms have, so
/// section 4's decomposition can be checked against a prediction rather
/// than asserted.
#[derive(Clone, Copy, Default)]
struct MachineRef {
    n: usize,
    copy_ns: f64,
    scale_ns: f64,
    div_ns: f64,
    exp_ns: f64,
    sumsq_ns: f64,
    alloc_ns: f64,
}

fn section_reference() -> Vec<MachineRef> {
    println!("\n=== 0. machine reference kernels (hand-written, one core) ===");
    println!("  GB/s is against each kernel's own traffic; ns/elem is the number to compare.");
    let mut out = Vec::new();
    for &n in &[65_536usize, 1_048_576, 4_194_304] {
        println!("\n--- n = {n} ({} MiB per f32 buffer) ---", n * 4 / (1 << 20));
        let x = filled(vec![n], 11);
        let x = &x.data;
        let mut dst = vec![0.0f32; n];
        let c = black_box(1.000_001f32);

        let ns = |us: f64| us * 1000.0 / n as f64;

        let copy = bench_bw("copy            out[i] = x[i]", n, 8.0, || {
            dst.copy_from_slice(x);
            black_box(&dst);
        });
        let scale = bench_bw("scale           out[i] = x[i] * c", n, 8.0, || {
            for (o, &v) in dst.iter_mut().zip(x) {
                *o = v * c;
            }
            black_box(&dst);
        });
        let div = bench_bw("divide          out[i] = x[i] / c", n, 8.0, || {
            for (o, &v) in dst.iter_mut().zip(x) {
                *o = v / c;
            }
            black_box(&dst);
        });
        let expa = bench_bw("exp             out[i] = exp(x[i])", n, 8.0, || {
            for (o, &v) in dst.iter_mut().zip(x) {
                *o = v.exp();
            }
            black_box(&dst);
        });
        bench_bw("sum             s += x[i]  (serial fadd)", n, 4.0, || {
            let mut s = 0.0f32;
            for &v in x.iter() {
                s += v;
            }
            black_box(s);
        });
        let sumsq = bench_bw("sum of squares  s += x[i]*x[i]  (serial)", n, 4.0, || {
            black_box(x.iter().map(|&v| v * v).sum::<f32>());
        });
        let alloc = bench_bw("alloc+zero      vec![0f32; n]", n, 4.0, || {
            black_box(vec![0.0f32; n]);
        });

        println!(
            "  → divide premium {:.3} ns/elem, exp {:.3} ns/elem above a copy, \
             fresh-buffer alloc+zero {:.3} ns/elem",
            ns(div) - ns(scale),
            ns(expa) - ns(copy),
            ns(alloc)
        );
        out.push(MachineRef {
            n,
            copy_ns: ns(copy),
            scale_ns: ns(scale),
            div_ns: ns(div),
            exp_ns: ns(expa),
            sumsq_ns: ns(sumsq),
            alloc_ns: ns(alloc),
        });
    }
    out
}

fn machine_for(refs: &[MachineRef], n: usize) -> Option<MachineRef> {
    refs.iter().copied().find(|m| m.n == n)
}

// ─── 1. per-run fixed cost ──────────────────────────────────────────────

/// The constant every `run` pays regardless of size: name binding, extent
/// inference, lowering, tape/loop-nest building, kernel-cache lookup, and
/// the write-back of a one-element output.
///
/// Measured as **the same program at `n = 1`**, per program, so later
/// tables can subtract the constant belonging to the arm they are about
/// rather than a single global guess.
#[derive(Clone, Copy, Default)]
struct Fixed {
    trivial: f64,
    rms_div: f64,
    rms_mul: f64,
    sm_2n: f64,
    sm_n_div: f64,
    sm_n_mul: f64,
    sm_stable: f64,
    matmul: f64,
}

fn section_fixed(programs: &Programs, reg: &Registry<f32>) -> Fixed {
    println!("\n=== 1. per-`run` fixed cost (same programs at n = 1) ===");
    println!("  bind + extent inference + lower + build + cache lookup + write-back.");
    println!("  Everything below is per `run`, so a program at n = 64 is mostly this.");

    let one = filled(vec![1], 5);
    let count = scalar(1.0);
    let eps = scalar(1e-5);
    let mut out1 = Dense::<f32>::zeros(vec![1]);

    let trivial = bench("y[i] = x[i]                       @ n=1", || {
        run(&programs.trivial, reg, &[("x", &one)], &mut [("y", &mut out1)]).unwrap();
        black_box(&out1);
    });
    let rms_div = bench("rmsnorm, divide                   @ n=1", || {
        run(
            &programs.rms_div,
            reg,
            &[("x", &one), ("n", &count), ("eps", &eps)],
            &mut [("y", &mut out1)],
        )
        .unwrap();
        black_box(&out1);
    });
    let rms_mul = bench("rmsnorm, hoisted reciprocal       @ n=1", || {
        run(
            &programs.rms_mul,
            reg,
            &[("x", &one), ("n", &count), ("eps", &eps)],
            &mut [("y", &mut out1)],
        )
        .unwrap();
        black_box(&out1);
    });
    let sm_2n = bench("softmax, 2n exp, divide           @ n=1", || {
        run(&programs.sm_2n, reg, &[("v", &one)], &mut [("soft", &mut out1)]).unwrap();
        black_box(&out1);
    });
    let sm_n_div = bench("softmax, n exp, divide (2 stmts)  @ n=1", || {
        run(&programs.sm_n_div, reg, &[("v", &one)], &mut [("soft", &mut out1)]).unwrap();
        black_box(&out1);
    });
    let sm_n_mul = bench("softmax, n exp, reciprocal        @ n=1", || {
        run(&programs.sm_n_mul, reg, &[("v", &one)], &mut [("soft", &mut out1)]).unwrap();
        black_box(&out1);
    });
    let sm_stable = bench("softmax, stable, matched (3 stmts)@ n=1", || {
        run(&programs.sm_stable, reg, &[("v", &one)], &mut [("soft", &mut out1)]).unwrap();
        black_box(&out1);
    });

    let a1 = filled(vec![1, 1], 7);
    let b1 = filled(vec![1, 1], 8);
    let mut c1 = Dense::<f32>::zeros(vec![1, 1]);
    let matmul = bench("matmul c[i,k]=sum(j: …)           @ 1×1", || {
        run(&programs.matmul, reg, &[("a", &a1), ("b", &b1)], &mut [("c", &mut c1)]).unwrap();
        black_box(&c1);
    });

    println!(
        "\n  A one-statement program costs {trivial:.3} µs per `run` before it does any work;\n  \
         the multi-statement softmax forms cost {:.3}–{:.3} µs.",
        sm_n_div.min(sm_stable),
        sm_n_div.max(sm_stable)
    );
    Fixed { trivial, rms_div, rms_mul, sm_2n, sm_n_div, sm_n_mul, sm_stable, matmul }
}

// ─── the programs, compiled once ────────────────────────────────────────

struct Programs {
    trivial: Checked,
    matmul: Checked,
    rms_div: Checked,
    rms_mul: Checked,
    sm_2n: Checked,
    sm_n_div: Checked,
    sm_n_mul: Checked,
    sm_stable: Checked,
    red_sumsq: Checked,
    map_scale: Checked,
    map_exp: Checked,
}

impl Programs {
    fn new(reg: &Registry<f32>) -> Self {
        Programs {
            trivial: compile(src::TRIVIAL, reg),
            matmul: compile(src::MATMUL, reg),
            rms_div: compile(src::RMS_DIV, reg),
            rms_mul: compile(src::RMS_MUL, reg),
            sm_2n: compile(src::SM_2N, reg),
            sm_n_div: compile(src::SM_N_DIV, reg),
            sm_n_mul: compile(src::SM_N_MUL, reg),
            sm_stable: compile(src::SM_STABLE, reg),
            red_sumsq: compile(src::RED_SUMSQ, reg),
            map_scale: compile(src::MAP_SCALE, reg),
            map_exp: compile(src::MAP_EXP, reg),
        }
    }
}

// ─── 2. matmul: lang vs einsum VM vs JIT ────────────────────────────────

fn section_matmul(ck: &mut Checks, reg: &Registry<f32>, programs: &Programs, fixed: &Fixed) {
    println!("\n=== 2. matmul  c[i,k] = sum(j: a[i,j] * b[j,k])  (einsum \"ab,bc->ac\") ===");
    println!("  GFLOP/s counts 2n³ (one multiply + one add per contraction element).");
    println!("  Output buffers are allocated once; the VM and JIT arms zero theirs per");
    println!("  iteration (their contract is accumulate-into-a-prepared-output).");
    println!(
        "  Per-`run` fixed cost for this program is {:.3} µs — subtract it from the lang arm.",
        fixed.matmul
    );

    // The einsum VM is ~0.14 GFLOP/s; at 512² one iteration would take
    // ~2 s and at 1024² ~15 s, so it stops at 256². Nothing else does.
    const VM_MAX: usize = 256;

    for &n in &[16usize, 64, 256, 512, 1024] {
        println!("\n--- {n}×{n} ---");
        let a = filled(vec![n, n], 7);
        let b = filled(vec![n, n], 8);
        let flops = 2.0 * (n as f64).powi(3);

        let mut c_lang = Dense::<f32>::zeros(vec![n, n]);
        run(&programs.matmul, reg, &[("a", &a), ("b", &b)], &mut [("c", &mut c_lang)]).unwrap();
        // Contraction length n over values in [-1,1): |c| ~ sqrt(n)/3.
        let atol = 1e-6 * n as f32;

        if n <= VM_MAX {
            let mut c_vm = Dense::<f32>::zeros(vec![n, n]);
            einsum_homogenous::<f32, _, _>("ab,bc->ac", &[&a, &b], &mut [&mut c_vm]).unwrap();
            ck.cmp(&format!("matmul {n}²: lang vs einsum VM"), &c_lang.data, &c_vm.data, atol, 1e-5);
        }

        let mut c = Dense::<f32>::zeros(vec![n, n]);
        let lang_us = bench_flops("lang evaluator (run)", flops, || {
            run(&programs.matmul, reg, &[("a", &a), ("b", &b)], &mut [("c", &mut c)]).unwrap();
            black_box(&c);
        });
        println!(
            "  lang minus fixed cost: {:.3} µs ({:.1} % of the arm is per-`run` setup)",
            lang_us - fixed.matmul,
            100.0 * fixed.matmul / lang_us
        );

        if n <= VM_MAX {
            let vm_us = bench_flops("einsum VM (einsum_homogenous)", flops, || {
                c.data.fill(0.0);
                einsum_homogenous::<f32, _, _>("ab,bc->ac", &[&a, &b], &mut [&mut c]).unwrap();
                black_box(&c);
            });
            println!(
                "  lang / VM: {:.2}×  (the language is {:.1}× faster than the einsum VM here)",
                lang_us / vm_us,
                vm_us / lang_us
            );
        }

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
            ck.cmp(&format!("matmul {n}²: lang vs JIT"), &c_lang.data, &c_jit.data, atol, 1e-5);

            let jit_us = bench_flops("einsum JIT (EinsumF32Jit::run)", flops, || {
                c.data.fill(0.0);
                jit.run(&[JitInput::Dense(&a), JitInput::Dense(&b)], &mut [&mut c]);
                black_box(&c);
            });
            println!(
                "  lang / JIT: {:.2}×   (lang minus fixed cost / JIT: {:.2}×)",
                lang_us / jit_us,
                (lang_us - fixed.matmul) / jit_us
            );
        }
    }
}

// ─── 3. mixed reduction (new capability, no library baseline) ───────────

fn section_mixed(ck: &mut Checks, reg: &Registry<f32>) {
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

    let prog = compile(src::MIXED, reg);

    let q = filled(vec![NK], 1);
    let x = filled(vec![NL], 2);
    let mut m = filled(vec![NL, NJ], 3);
    // Keep the column sums positive so sqrt stays real.
    for v in m.data.iter_mut() {
        *v = v.abs() + 0.1;
    }

    let mut y_lang = Dense::<f32>::zeros(vec![NJ, NK]);
    run(&prog, reg, &[("q", &q), ("x", &x), ("M", &m)], &mut [("y", &mut y_lang)]).unwrap();

    // No library baseline exists for this shape of expression; the manual
    // loop below is written here purely as the correctness oracle. It is
    // also timed, clearly labelled, as an "ideal hand-written" reference.
    let manual = |out: &mut Dense<f32>| {
        let xmin = x.data.iter().copied().fold(f32::INFINITY, f32::min);
        for j in 0..NJ {
            let col: f32 = (0..NL).map(|l| m.data[l * NJ + j]).sum();
            let s = xmin * col.sqrt();
            for k in 0..NK {
                out.data[j * NK + k] = q.data[k] * s;
            }
        }
    };
    let mut y_ref = Dense::<f32>::zeros(vec![NJ, NK]);
    manual(&mut y_ref);
    ck.cmp("mixed: lang vs manual loops", &y_lang.data, &y_ref.data, 1e-5, 1e-4);

    let mut y = Dense::<f32>::zeros(vec![NJ, NK]);
    let lang_us = bench("lang evaluator (run)", || {
        run(&prog, reg, &[("q", &q), ("x", &x), ("M", &m)], &mut [("y", &mut y)]).unwrap();
        black_box(&y);
    });
    let ref_us = bench("manual loops (correctness oracle only)", || {
        manual(&mut y);
        black_box(&y);
    });
    println!("  lang / manual: {:.2}×", lang_us / ref_us);
}

// ─── 4. rmsnorm and softmax, with the gap decomposed ────────────────────

#[derive(Clone, Copy, Default)]
struct RmsRow {
    n: usize,
    lang_div: f64,
    lang_mul: f64,
    hand_div: f64,
    hand_mul: f64,
    hand_langbuf: f64,
}

#[derive(Clone, Copy, Default)]
struct SmRow {
    n: usize,
    lang_2n: f64,
    lang_n_div: f64,
    lang_n_mul: f64,
    lang_stable: f64,
    hand_n_div: f64,
    hand_n_mul: f64,
    hand_stable: f64,
    hand_unfused: f64,
    hand_langbuf: f64,
}

/// Elementwise traffic model used for the GB/s column: 2 reads + 1 write
/// of the vector, which is the least any of these arms could move. An arm
/// that makes more passes shows a lower "effective" bandwidth — that is
/// the point of holding the model fixed.
const ELEM_BYTES: f64 = 12.0;

const SIZES: [usize; 5] = [64, 1024, 65_536, 1_048_576, 4_194_304];

fn section_rmsnorm(
    ck: &mut Checks,
    reg: &Registry<f32>,
    programs: &Programs,
    fixed: &Fixed,
    machine: &[MachineRef],
) -> Vec<RmsRow> {
    println!("\n=== 4a. rmsnorm ===");
    println!("  lang  divide     y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps)");
    println!("  lang  reciprocal y[i] = x[i] * (1.0 / sqrt(sum(j: x[j]*x[j]) / n + eps))");
    println!("  hand  reciprocal = the gpt2 `rmsnorm` (matches the lang reciprocal form)");
    println!("  hand  divide     = the same, dividing per element (matches the lang divide form)");
    println!("  hand  lang-bufs  = hand reciprocal + a fresh zeroed temp + copy_from_slice");
    println!("  GB/s is against a fixed {ELEM_BYTES} B/element model (2 reads + 1 write).");
    const EPS: f32 = 1e-5;
    show_path_rms("rmsnorm, divide", &programs.rms_div, reg, 1024);
    show_path_rms("rmsnorm, reciprocal", &programs.rms_mul, reg, 1024);

    let mut rows = Vec::new();
    for &n in &SIZES {
        println!("\n--- n = {n} ---");
        let x = filled(vec![n], 43);
        let count = scalar(n as f32);
        let eps = scalar(EPS);
        let mut out = Dense::<f32>::zeros(vec![n]);

        // Reference results (into the buffer that timing will reuse, so a
        // stale-buffer bug in an arm shows up here too).
        let mut y_lang_div = vec![0.0f32; n];
        run(
            &programs.rms_div,
            reg,
            &[("x", &x), ("n", &count), ("eps", &eps)],
            &mut [("y", &mut out)],
        )
        .unwrap();
        y_lang_div.copy_from_slice(&out.data);

        let mut y_lang_mul = vec![0.0f32; n];
        run(
            &programs.rms_mul,
            reg,
            &[("x", &x), ("n", &count), ("eps", &eps)],
            &mut [("y", &mut out)],
        )
        .unwrap();
        y_lang_mul.copy_from_slice(&out.data);

        let mut y_hand_mul = vec![0.0f32; n];
        rmsnorm_mul_into(&x.data, EPS, &mut y_hand_mul);
        let mut y_hand_div = vec![0.0f32; n];
        rmsnorm_div_into(&x.data, EPS, &mut y_hand_div);
        let mut y_hand_buf = vec![0.0f32; n];
        rmsnorm_mul_langbuf(&x.data, EPS, &mut y_hand_buf);

        ck.cmp(&format!("rmsnorm n={n}: lang mul vs hand mul"), &y_lang_mul, &y_hand_mul, 1e-9, 1e-6);
        ck.cmp(&format!("rmsnorm n={n}: lang div vs hand div"), &y_lang_div, &y_hand_div, 1e-9, 1e-6);
        ck.cmp(&format!("rmsnorm n={n}: hand buf vs hand mul"), &y_hand_buf, &y_hand_mul, 0.0, 0.0);
        ck.cmp(&format!("rmsnorm n={n}: lang div vs lang mul"), &y_lang_div, &y_lang_mul, 1e-9, 1e-6);

        let mut row = RmsRow { n, ..Default::default() };
        row.lang_div = bench_bw("lang, divide", n, ELEM_BYTES, || {
            run(
                &programs.rms_div,
                reg,
                &[("x", &x), ("n", &count), ("eps", &eps)],
                &mut [("y", &mut out)],
            )
            .unwrap();
            black_box(&out);
        });
        row.lang_mul = bench_bw("lang, hoisted reciprocal", n, ELEM_BYTES, || {
            run(
                &programs.rms_mul,
                reg,
                &[("x", &x), ("n", &count), ("eps", &eps)],
                &mut [("y", &mut out)],
            )
            .unwrap();
            black_box(&out);
        });
        row.hand_div = bench_bw("hand-rolled, divide", n, ELEM_BYTES, || {
            rmsnorm_div_into(&x.data, EPS, &mut y_hand_div);
            black_box(&y_hand_div);
        });
        row.hand_mul = bench_bw("hand-rolled, reciprocal (gpt2)", n, ELEM_BYTES, || {
            rmsnorm_mul_into(&x.data, EPS, &mut y_hand_mul);
            black_box(&y_hand_mul);
        });
        row.hand_langbuf = bench_bw("hand-rolled + lang's buffer discipline", n, ELEM_BYTES, || {
            rmsnorm_mul_langbuf(&x.data, EPS, &mut y_hand_buf);
            black_box(&y_hand_buf);
        });

        print_rms_decomposition(&row, fixed, machine);
        rows.push(row);
    }
    rows
}

/// Terms are differences of **net** times (each lang arm minus that
/// program's own `n = 1` cost), so a term never smuggles in a difference
/// in setup cost between two programs. They sum exactly to the total gap.
fn print_rms_decomposition(r: &RmsRow, fixed: &Fixed, machine: &[MachineRef]) {
    let net_div = r.lang_div - fixed.rms_div;
    let net_mul = r.lang_mul - fixed.rms_mul;
    let total = r.lang_div - r.hand_mul;
    let setup = fixed.rms_div;
    let divide = net_div - net_mul;
    let bufs = r.hand_langbuf - r.hand_mul;
    let leftover = net_mul - r.hand_langbuf;
    let pct = |v: f64| 100.0 * v / total;
    println!("  gap decomposition (lang divide-form vs hand-rolled reciprocal-form):");
    println!("    total gap                     {total:10.3} µs  ({:.2}×)", r.lang_div / r.hand_mul);
    println!("    per-`run` fixed cost          {setup:10.3} µs  ({:5.1} %)", pct(setup));
    println!("    divide vs multiply            {divide:10.3} µs  ({:5.1} %)", pct(divide));
    println!("    lang's buffer discipline      {bufs:10.3} µs  ({:5.1} %)", pct(bufs));
    println!("    leftover (codegen)            {leftover:10.3} µs  ({:5.1} %)", pct(leftover));
    if let Some(m) = machine_for(machine, r.n) {
        let pred = (m.div_ns - m.scale_ns) * r.n as f64 / 1000.0;
        println!(
            "    ↳ divide term predicted from section 0: {pred:.3} µs \
             ({:.3} ns/elem × {} elems)",
            m.div_ns - m.scale_ns,
            r.n
        );
        let floor = (m.sumsq_ns + m.scale_ns) * r.n as f64 / 1000.0;
        println!(
            "    ↳ hand-rolled arm vs its own kernel floor (sum-of-squares + scale): \
             {:.3} µs measured, {floor:.3} µs predicted",
            r.hand_mul
        );
        let pred_buf = (m.alloc_ns + m.copy_ns) * r.n as f64 / 1000.0;
        println!(
            "    ↳ buffer term predicted from section 0 (alloc+zero {:.3} + copy {:.3} \
             ns/elem): {pred_buf:.3} µs",
            m.alloc_ns, m.copy_ns
        );
    }
    println!(
        "    matched pair, setup removed:  lang reciprocal {net_mul:.3} µs vs hand {:.3} µs = {:.2}×",
        r.hand_mul,
        net_mul / r.hand_mul
    );
}

fn section_softmax(
    ck: &mut Checks,
    reg: &Registry<f32>,
    programs: &Programs,
    fixed: &Fixed,
    machine: &[MachineRef],
) -> Vec<SmRow> {
    println!("\n=== 4b. softmax ===");
    println!("  lang 2n-exp   soft[i] = exp(v[i]) / sum(j: exp(v[j]))          (exp called 2n×)");
    println!("  lang n-exp /  e[i] = exp(v[i]) ; soft[i] = e[i] / sum(j: e[j])");
    println!("  lang n-exp *  e[i] = exp(v[i]) ; soft[i] = e[i] * (1.0 / sum(j: e[j]))");
    println!("  lang stable   m = max(j: v[j]) ; e[i] = exp(v[i]-m) ; soft[i] = e[i]*(1.0/sum(j: e[j]))");
    println!("  hand stable   = the gpt2 `softmax_last` (same arithmetic as the lang stable form)");
    println!("  hand n-exp /, hand n-exp *  = naive hand-written forms matching the lang ones");
    println!("  hand unfused  = hand stable in four passes (max, exp, sum, scale) — the pass");
    println!("                  structure a three-statement program is forced into");
    println!("  hand lang-bufs = hand unfused + a fresh zeroed temp per statement + copy_from_slice");
    show_path_softmax("softmax, 2n exp", &programs.sm_2n, reg, 1024);
    show_path_softmax("softmax, n exp, divide", &programs.sm_n_div, reg, 1024);
    show_path_softmax("softmax, n exp, reciprocal", &programs.sm_n_mul, reg, 1024);
    show_path_softmax("softmax, stable", &programs.sm_stable, reg, 1024);

    let mut rows = Vec::new();
    for &n in &SIZES {
        println!("\n--- n = {n} ---");
        let v = filled(vec![n], 42);
        let mut out = Dense::<f32>::zeros(vec![n]);

        let mut lang_2n = vec![0.0f32; n];
        run(&programs.sm_2n, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        lang_2n.copy_from_slice(&out.data);
        let mut lang_n_div = vec![0.0f32; n];
        run(&programs.sm_n_div, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        lang_n_div.copy_from_slice(&out.data);
        let mut lang_n_mul = vec![0.0f32; n];
        run(&programs.sm_n_mul, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        lang_n_mul.copy_from_slice(&out.data);
        let mut lang_stable = vec![0.0f32; n];
        run(&programs.sm_stable, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        lang_stable.copy_from_slice(&out.data);

        let mut hand_n_div = vec![0.0f32; n];
        softmax_naive_div_into(&v.data, &mut hand_n_div);
        let mut hand_n_mul = vec![0.0f32; n];
        softmax_naive_mul_into(&v.data, &mut hand_n_mul);
        let mut hand_stable = vec![0.0f32; n];
        softmax_stable_into(&v.data, &mut hand_stable);
        let mut hand_e = vec![0.0f32; n];
        let mut hand_unfused = vec![0.0f32; n];
        softmax_stable_unfused(&v.data, &mut hand_e, &mut hand_unfused);
        let mut hand_buf = vec![0.0f32; n];
        softmax_stable_unfused_langbuf(&v.data, &mut hand_buf);

        ck.cmp(&format!("softmax n={n}: lang stable vs hand stable"), &lang_stable, &hand_stable, 1e-9, 1e-6);
        ck.cmp(&format!("softmax n={n}: lang n-exp * vs hand n-exp *"), &lang_n_mul, &hand_n_mul, 1e-9, 1e-6);
        ck.cmp(&format!("softmax n={n}: lang n-exp / vs hand n-exp /"), &lang_n_div, &hand_n_div, 1e-9, 1e-6);
        ck.cmp(&format!("softmax n={n}: lang 2n-exp vs lang n-exp /"), &lang_2n, &lang_n_div, 1e-9, 1e-6);
        ck.cmp(&format!("softmax n={n}: lang stable vs lang 2n-exp"), &lang_stable, &lang_2n, 1e-9, 1e-5);
        ck.cmp(&format!("softmax n={n}: hand unfused vs hand stable"), &hand_unfused, &hand_stable, 0.0, 0.0);
        ck.cmp(&format!("softmax n={n}: hand buf vs hand stable"), &hand_buf, &hand_stable, 0.0, 0.0);

        let mut row = SmRow { n, ..Default::default() };
        row.lang_2n = bench_bw("lang, 2n exp, divide", n, ELEM_BYTES, || {
            run(&programs.sm_2n, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        row.lang_n_div = bench_bw("lang, n exp, divide", n, ELEM_BYTES, || {
            run(&programs.sm_n_div, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        row.lang_n_mul = bench_bw("lang, n exp, reciprocal", n, ELEM_BYTES, || {
            run(&programs.sm_n_mul, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        row.lang_stable = bench_bw("lang, stable (matched to hand-rolled)", n, ELEM_BYTES, || {
            run(&programs.sm_stable, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        row.hand_n_div = bench_bw("hand-rolled, n exp, divide", n, ELEM_BYTES, || {
            softmax_naive_div_into(&v.data, &mut hand_n_div);
            black_box(&hand_n_div);
        });
        row.hand_n_mul = bench_bw("hand-rolled, n exp, reciprocal", n, ELEM_BYTES, || {
            softmax_naive_mul_into(&v.data, &mut hand_n_mul);
            black_box(&hand_n_mul);
        });
        row.hand_stable = bench_bw("hand-rolled, stable (gpt2 softmax_last)", n, ELEM_BYTES, || {
            softmax_stable_into(&v.data, &mut hand_stable);
            black_box(&hand_stable);
        });
        row.hand_unfused = bench_bw("hand-rolled, stable, 4 unfused passes", n, ELEM_BYTES, || {
            softmax_stable_unfused(&v.data, &mut hand_e, &mut hand_unfused);
            black_box(&hand_unfused);
        });
        row.hand_langbuf = bench_bw("hand-rolled unfused + lang's buffers", n, ELEM_BYTES, || {
            softmax_stable_unfused_langbuf(&v.data, &mut hand_buf);
            black_box(&hand_buf);
        });

        print_sm_decomposition(&row, fixed, machine);
        rows.push(row);
    }
    rows
}

fn print_sm_decomposition(r: &SmRow, fixed: &Fixed, machine: &[MachineRef]) {
    let net_2n = r.lang_2n - fixed.sm_2n;
    let net_n_div = r.lang_n_div - fixed.sm_n_div;
    let net_n_mul = r.lang_n_mul - fixed.sm_n_mul;
    let net_stable = r.lang_stable - fixed.sm_stable;
    let total = r.lang_2n - r.hand_stable;
    let setup = fixed.sm_2n;
    let double_exp = net_2n - net_n_div;
    let divide = net_n_div - net_n_mul;
    let stability = net_n_mul - net_stable; // negative: the stable form does more
    let fusion = r.hand_unfused - r.hand_stable;
    let bufs = r.hand_langbuf - r.hand_unfused;
    let leftover = net_stable - r.hand_langbuf;
    let pct = |v: f64| 100.0 * v / total;
    println!("  gap decomposition (lang 2n-exp form vs the gpt2 hand-rolled loop):");
    println!("    total gap                     {total:10.3} µs  ({:.2}×)", r.lang_2n / r.hand_stable);
    println!("    per-`run` fixed cost          {setup:10.3} µs  ({:5.1} %)", pct(setup));
    println!("    second n exp calls            {double_exp:10.3} µs  ({:5.1} %)", pct(double_exp));
    println!("    divide vs multiply            {divide:10.3} µs  ({:5.1} %)", pct(divide));
    println!(
        "    max-pass the naive form skips {stability:10.3} µs  ({:5.1} %)  [credit: the \
         hand-rolled arm is the *stable* form]",
        pct(stability)
    );
    println!("    exp/sum pass not fused        {fusion:10.3} µs  ({:5.1} %)", pct(fusion));
    println!("    lang's buffer discipline      {bufs:10.3} µs  ({:5.1} %)", pct(bufs));
    println!("    leftover (codegen)            {leftover:10.3} µs  ({:5.1} %)", pct(leftover));
    if let Some(m) = machine_for(machine, r.n) {
        println!(
            "    ↳ second-exp term predicted from section 0: {:.3} µs ({:.3} ns/elem × {})",
            m.exp_ns * r.n as f64 / 1000.0,
            m.exp_ns,
            r.n
        );
        println!(
            "    ↳ divide term predicted from section 0: {:.3} µs ({:.3} ns/elem × {})",
            (m.div_ns - m.scale_ns) * r.n as f64 / 1000.0,
            m.div_ns - m.scale_ns,
            r.n
        );
    }
    println!(
        "    matched pair, setup removed:  lang stable {net_stable:.3} µs vs hand stable \
         {:.3} µs = {:.2}×  (vs the unfused hand form: {:.2}×)",
        r.hand_stable,
        net_stable / r.hand_stable,
        net_stable / r.hand_unfused
    );
}

// ─── 4c. single-kernel breakdown ────────────────────────────────────────

/// Where the "leftover" of 4a/4b actually lives: one statement, one
/// kernel, against the identical hand-written loop.
///
/// - `s = sum(j: x[j]*x[j])` is one reduction, scalar output, no
///   write-back — the reduction kernel on its own.
/// - `y[i] = x[i] * c` is one elementwise map — the map kernel plus the
///   language's per-statement buffer (`vec![0f32; n]` + `copy_from_slice`).
/// - `y[i] = exp(x[i])` is the same map with an inlined built-in, so it
///   says whether `exp` costs the language more than it costs Rust.
///
/// Each arm is reported net of its own `n = 1` cost.
fn section_kernels(ck: &mut Checks, reg: &Registry<f32>, programs: &Programs) {
    println!("\n=== 4c. single-kernel breakdown (one statement, one kernel) ===");
    println!("  `net` subtracts the same program's n = 1 cost, measured just below.");

    let one = filled(vec![1], 5);
    let c1 = scalar(1.000_001);
    let mut s1 = Dense::<f32>::zeros(vec![]);
    let mut o1 = Dense::<f32>::zeros(vec![1]);
    let f_red = bench("s = sum(j: x[j]*x[j])             @ n=1", || {
        run(&programs.red_sumsq, reg, &[("x", &one)], &mut [("s", &mut s1)]).unwrap();
        black_box(&s1);
    });
    let f_scale = bench("y[i] = x[i] * c                   @ n=1", || {
        run(&programs.map_scale, reg, &[("x", &one), ("c", &c1)], &mut [("y", &mut o1)]).unwrap();
        black_box(&o1);
    });
    let f_exp = bench("y[i] = exp(x[i])                  @ n=1", || {
        run(&programs.map_exp, reg, &[("x", &one)], &mut [("y", &mut o1)]).unwrap();
        black_box(&o1);
    });
    show_path_reduce("s = sum(j: x[j]*x[j])", &programs.red_sumsq, reg, 1024);

    for &n in &[65_536usize, 1_048_576, 4_194_304] {
        println!("\n--- n = {n} ---");
        let x = filled(vec![n], 43);
        let c = scalar(1.000_001);
        let mut sc = Dense::<f32>::zeros(vec![]);
        let mut out = Dense::<f32>::zeros(vec![n]);
        let mut hand = vec![0.0f32; n];

        // reduction
        run(&programs.red_sumsq, reg, &[("x", &x)], &mut [("s", &mut sc)]).unwrap();
        let hand_ss = x.data.iter().map(|&v| v * v).sum::<f32>();
        ck.cmp(&format!("sum(x²) n={n}: lang vs hand"), &sc.data, &[hand_ss], 0.0, 1e-6);
        let l_red = bench_bw("lang  s = sum(j: x[j]*x[j])", n, 4.0, || {
            run(&programs.red_sumsq, reg, &[("x", &x)], &mut [("s", &mut sc)]).unwrap();
            black_box(&sc);
        });
        let h_red = bench_bw("hand  s += x[i]*x[i]", n, 4.0, || {
            black_box(x.data.iter().map(|&v| v * v).sum::<f32>());
        });
        println!(
            "  → reduction kernel: lang net {:.3} µs vs hand {:.3} µs = {:.2}×  \
             ({:.3} vs {:.3} ns/elem)",
            l_red - f_red,
            h_red,
            (l_red - f_red) / h_red,
            (l_red - f_red) * 1000.0 / n as f64,
            h_red * 1000.0 / n as f64
        );

        // elementwise map
        run(&programs.map_scale, reg, &[("x", &x), ("c", &c)], &mut [("y", &mut out)]).unwrap();
        let cv = c.data[0];
        for (o, &v) in hand.iter_mut().zip(&x.data) {
            *o = v * cv;
        }
        ck.cmp(&format!("y=x*c n={n}: lang vs hand"), &out.data, &hand, 0.0, 0.0);
        let l_scale = bench_bw("lang  y[i] = x[i] * c", n, 8.0, || {
            run(&programs.map_scale, reg, &[("x", &x), ("c", &c)], &mut [("y", &mut out)]).unwrap();
            black_box(&out);
        });
        let h_scale = bench_bw("hand  out[i] = x[i] * c", n, 8.0, || {
            for (o, &v) in hand.iter_mut().zip(&x.data) {
                *o = v * cv;
            }
            black_box(&hand);
        });
        let mut hand_blk = vec![0.0f32; n];
        scale_tape_shape(&x.data, cv, &mut hand_blk);
        ck.cmp(&format!("y=x*c n={n}: tape-shaped vs hand"), &hand_blk, &hand, 0.0, 0.0);
        let h_blk = bench_bw("hand  written in the tape's op shape", n, 8.0, || {
            scale_tape_shape(&x.data, cv, &mut hand_blk);
            black_box(&hand_blk);
        });
        println!(
            "  → map kernel: lang net {:.3} µs vs hand {:.3} µs = {:.2}×  \
             ({:.3} vs {:.3} ns/elem)",
            l_scale - f_scale,
            h_scale,
            (l_scale - f_scale) / h_scale,
            (l_scale - f_scale) * 1000.0 / n as f64,
            h_scale * 1000.0 / n as f64
        );
        println!(
            "    of which the tape's op shape (blocks + splat + buffers) explains {:.3} µs \
             ({:.3} ns/elem); unexplained {:.3} µs ({:.3} ns/elem)",
            h_blk - h_scale,
            (h_blk - h_scale) * 1000.0 / n as f64,
            l_scale - f_scale - h_blk,
            (l_scale - f_scale - h_blk) * 1000.0 / n as f64
        );

        // elementwise map with a built-in
        run(&programs.map_exp, reg, &[("x", &x)], &mut [("y", &mut out)]).unwrap();
        for (o, &v) in hand.iter_mut().zip(&x.data) {
            *o = v.exp();
        }
        ck.cmp(&format!("y=exp(x) n={n}: lang vs hand"), &out.data, &hand, 0.0, 0.0);
        let l_exp = bench_bw("lang  y[i] = exp(x[i])", n, 8.0, || {
            run(&programs.map_exp, reg, &[("x", &x)], &mut [("y", &mut out)]).unwrap();
            black_box(&out);
        });
        let h_exp = bench_bw("hand  out[i] = exp(x[i])", n, 8.0, || {
            for (o, &v) in hand.iter_mut().zip(&x.data) {
                *o = v.exp();
            }
            black_box(&hand);
        });
        println!(
            "  → exp map: lang net {:.3} µs vs hand {:.3} µs = {:.2}×  \
             ({:.3} vs {:.3} ns/elem)",
            l_exp - f_exp,
            h_exp,
            (l_exp - f_exp) / h_exp,
            (l_exp - f_exp) * 1000.0 / n as f64,
            h_exp * 1000.0 / n as f64
        );
    }
}

// ─── 5. multi-statement program: stable softmax ─────────────────────────

fn section_multi_statement(ck: &mut Checks, reg: &Registry<f32>) {
    println!("\n=== 5. statement/temp overhead: stable softmax ===");
    println!("  naive-1stmt : soft[i] = exp(v[i]) / sum(j: exp(v[j]))");
    println!("  stable-2stmt: m = max(j: v[j]) ; soft[i] = exp(v[i]-m) / sum(j: exp(v[j]-m))");
    println!("  stable-1stmt: same, inlined (recomputes max(j: v[j]) twice — no CSE yet)");

    let naive = compile(src::SM_2N, reg);
    let stable2 = compile(src::SM_STABLE_2, reg);
    let stable1 = compile(src::SM_STABLE_1, reg);

    for &n in &[64usize, 1024, 65536] {
        println!("\n--- n = {n} ---");
        let v = filled(vec![n], 42);
        let mut out = Dense::<f32>::zeros(vec![n]);

        let mut s_naive = vec![0.0f32; n];
        run(&naive, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        s_naive.copy_from_slice(&out.data);
        let mut s2 = vec![0.0f32; n];
        run(&stable2, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        s2.copy_from_slice(&out.data);
        let mut s1 = vec![0.0f32; n];
        run(&stable1, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
        s1.copy_from_slice(&out.data);
        let mut s_ref = vec![0.0f32; n];
        softmax_stable_into(&v.data, &mut s_ref);

        ck.cmp(&format!("stable-2stmt n={n}: vs hand-rolled"), &s2, &s_ref, 1e-9, 1e-6);
        ck.cmp(&format!("stable-1stmt n={n}: vs stable-2stmt"), &s1, &s2, 1e-9, 1e-6);
        ck.cmp(&format!("naive-1stmt  n={n}: vs stable-2stmt"), &s_naive, &s2, 1e-9, 1e-5);

        let naive_us = bench("lang naive softmax (1 stmt, 1 reduce)", || {
            run(&naive, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        let s2_us = bench("lang stable softmax (2 stmts, 2 reduces)", || {
            run(&stable2, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        let s1_us = bench("lang stable softmax (1 stmt, 3 reduces)", || {
            run(&stable1, reg, &[("v", &v)], &mut [("soft", &mut out)]).unwrap();
            black_box(&out);
        });
        println!("  stable-2stmt / naive-1stmt: {:.2}×", s2_us / naive_us);
        println!("  stable-1stmt / stable-2stmt: {:.2}×", s1_us / s2_us);
    }
}

// ─── 6. dynamic axes (RESUME step 6) ────────────────────────────────────

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
fn section_dynamic(ck: &mut Checks, reg: &Registry<f32>, programs: &Programs) {
    println!("\n=== 6. dynamic axes: c[i,k] = sum(j: a[i,j] * b[j,k]) with runtime extents ===");

    // `j` alone is the KV-cache case (the contraction length grows); all
    // three is the pessimal case, where no extent is a constant anywhere.
    let cases: [(&str, &[&str]); 3] = [
        ("static (baked extents)", &[]),
        ("dynamic j", &["j"]),
        ("dynamic i,j,k", &["i", "j", "k"]),
    ];

    for &n in &[16usize, 64, 256, 512] {
        println!("\n--- {n}×{n} ---");
        let a = filled(vec![n, n], 7);
        let b = filled(vec![n, n], 8);
        let flops = 2.0 * (n as f64).powi(3);

        let mut base: Vec<u32> = Vec::new();
        let mut static_us = 0.0;
        let mut c = Dense::<f32>::zeros(vec![n, n]);
        for (label, dynamic) in cases {
            run_with(
                &programs.matmul,
                reg,
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
                ck.bad.push(msg);
            } else {
                ck.identical.push(format!("matmul {n}²: {label} vs static"));
            }

            let us = bench_flops(label, flops, || {
                run_with(
                    &programs.matmul,
                    reg,
                    &[("a", &a), ("b", &b)],
                    &mut [("c", &mut c)],
                    RunOptions { dynamic },
                )
                .unwrap();
                black_box(&c);
            });
            if dynamic.is_empty() {
                static_us = us;
            } else {
                println!("  {label} / static: {:.2}×", us / static_us);
            }
        }
    }
}

// ─── 1b. is the fixed cost still the fixed cost? ────────────────────────

/// Section 1 runs first, with the JIT kernel cache nearly empty; the
/// cache is a linear scan over string-keyed slots, so the constant every
/// later table subtracts could in principle have grown underneath it.
/// Re-measure two of the arms at the end and print the drift.
fn recheck_fixed(programs: &Programs, reg: &Registry<f32>, fixed: &Fixed) {
    println!("\n=== 1b. per-`run` fixed cost, re-measured after the whole suite ===");
    #[cfg(feature = "jit")]
    println!("  kernel-cache entries now held by this thread: {}", linalg::lang::jit_cache_len());

    let one = filled(vec![1], 5);
    let count = scalar(1.0);
    let eps = scalar(1e-5);
    let mut out1 = Dense::<f32>::zeros(vec![1]);

    let rms_div = bench("rmsnorm, divide                   @ n=1", || {
        run(
            &programs.rms_div,
            reg,
            &[("x", &one), ("n", &count), ("eps", &eps)],
            &mut [("y", &mut out1)],
        )
        .unwrap();
        black_box(&out1);
    });
    let sm_2n = bench("softmax, 2n exp, divide           @ n=1", || {
        run(&programs.sm_2n, reg, &[("v", &one)], &mut [("soft", &mut out1)]).unwrap();
        black_box(&out1);
    });
    println!(
        "  drift vs section 1: rmsnorm {:+.3} µs ({:+.1} %), softmax {:+.3} µs ({:+.1} %)",
        rms_div - fixed.rms_div,
        100.0 * (rms_div - fixed.rms_div) / fixed.rms_div,
        sm_2n - fixed.sm_2n,
        100.0 * (sm_2n - fixed.sm_2n) / fixed.sm_2n,
    );
}

// ─── final summary tables ───────────────────────────────────────────────

fn summarize(rms: &[RmsRow], sm: &[SmRow], fixed: &Fixed) {
    if rms.is_empty() && sm.is_empty() {
        return;
    }
    println!("\n=== summary: where the elementwise gap goes ===");
    println!(
        "  Terms are µs and sum exactly to the total gap. `setup` is the same program\n  \
         at n = 1; `bufs` is the hand-written kernel re-run with the language's\n  \
         allocate-a-fresh-zeroed-buffer-per-statement + copy_from_slice discipline;\n  \
         `leftover` is what neither explains — the real codegen gap."
    );

    if !rms.is_empty() {
        println!("\n  rmsnorm: lang(divide) vs hand(reciprocal).  Terms sum to `total`.");
        println!(
            "  {:>9}  {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>9} {:>9}",
            "n", "lang÷", "lang×", "hand×", "total", "setup", "divide", "bufs", "leftover"
        );
        for r in rms {
            println!(
                "  {:>9}  {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3}",
                r.n,
                r.lang_div,
                r.lang_mul,
                r.hand_mul,
                r.lang_div - r.hand_mul,
                fixed.rms_div,
                (r.lang_div - fixed.rms_div) - (r.lang_mul - fixed.rms_mul),
                r.hand_langbuf - r.hand_mul,
                (r.lang_mul - fixed.rms_mul) - r.hand_langbuf,
            );
        }
        println!("  matched pair (both hoist the reciprocal), each side's setup removed:");
        for r in rms {
            let net = r.lang_mul - fixed.rms_mul;
            println!(
                "    n={:<9} lang {:>10.3} µs  hand {:>10.3} µs   {:.2}×",
                r.n, net, r.hand_mul, net / r.hand_mul
            );
        }
    }

    if !sm.is_empty() {
        println!("\n  softmax: lang(2n exp, divide) vs hand(stable, n exp, reciprocal).");
        println!(
            "  {:>9}  {:>10} {:>10} {:>10} {:>9} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9}",
            "n", "lang 2n", "lang stbl", "hand stbl", "total", "2nd exp", "divide", "max-pass",
            "fusion", "bufs", "leftover"
        );
        for r in sm {
            let net_2n = r.lang_2n - fixed.sm_2n;
            let net_n_div = r.lang_n_div - fixed.sm_n_div;
            let net_n_mul = r.lang_n_mul - fixed.sm_n_mul;
            let net_stable = r.lang_stable - fixed.sm_stable;
            println!(
                "  {:>9}  {:>10.3} {:>10.3} {:>10.3} {:>9.3} {:>9.3} {:>8.3} {:>9.3} {:>9.3} \
                 {:>9.3} {:>9.3}",
                r.n,
                r.lang_2n,
                r.lang_stable,
                r.hand_stable,
                r.lang_2n - r.hand_stable,
                net_2n - net_n_div,
                net_n_div - net_n_mul,
                net_n_mul - net_stable,
                r.hand_unfused - r.hand_stable,
                r.hand_langbuf - r.hand_unfused,
                net_stable - r.hand_langbuf,
            );
        }
        println!("  (`setup` = {:.3} µs, the same program at n=1, is the missing column)", fixed.sm_2n);
        println!("  matched pair (both stable, n exp, hoisted reciprocal), setup removed:");
        for r in sm {
            let net = r.lang_stable - fixed.sm_stable;
            println!(
                "    n={:<9} lang {:>10.3} µs  hand {:>10.3} µs   {:.2}×  \
                 (vs hand unfused {:>10.3} µs: {:.2}×)",
                r.n,
                net,
                r.hand_stable,
                net / r.hand_stable,
                r.hand_unfused,
                net / r.hand_unfused
            );
        }
    }
}

// ─── 7. the programs, as nested loops ───────────────────────────────────

/// Print every benchmarked program's loop-nest lowering.
///
/// This is what the tables below are timings *of*: `Checked::explain`
/// renders the materializing lowering the language is defined by — where
/// each reduction's temporary is allocated, which loops it sits inside,
/// and in what order the folds run. Reading an arm's cost against its
/// nest answers most "why is this one slower?" questions before any
/// profiling: `2n` vs `n` `exp` calls, a hoisted reciprocal, a maximum
/// recomputed into two separate temporaries.
///
/// Extents stay symbolic (`0..|j|`) — every program here runs at several
/// sizes, and the nest is the same shape at all of them.
fn section_programs(reg: &Registry<f32>) {
    println!("\n=== 7. programs, as nested loops ===");
    println!("  `Checked::explain`: the lowering each timing below is of. `%k` is a");
    println!("  materialized reduction temporary, `acck` its accumulator.");
    for (label, source) in src::SOURCES {
        println!("\n--- {label} ---");
        for line in source.lines() {
            println!("  {}", line.trim_end());
        }
        println!();
        for line in compile(source, reg).explain(reg).lines() {
            // Indent the nest under the section, without leaving trailing
            // whitespace on its blank separator lines.
            if line.is_empty() {
                println!();
            } else {
                println!("  {line}");
            }
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────

fn wanted(section: usize) -> bool {
    match std::env::var("LANG_BENCH_SECTIONS") {
        Ok(s) => s.split(',').any(|p| p.trim().parse::<usize>() == Ok(section)),
        Err(_) => true,
    }
}

fn main() {
    println!("=== linalg::lang evaluator bench ===");
    println!("(median of several timed batches; [batches×iters, min] shown per arm)");
    #[cfg(feature = "jit")]
    println!("(jit feature: ON)");
    #[cfg(not(feature = "jit"))]
    println!("(jit feature: OFF — section 2's JIT arm skipped)");

    let reg = Registry::<f32>::builtins();

    if wanted(7) {
        section_programs(&reg);
    }
    // The listing is documentation, not a measurement — asking for it
    // alone should not cost the fixed-cost section's several seconds.
    if ![0, 2, 3, 4, 5, 6].iter().any(|&s| wanted(s)) {
        println!("\n=== done (listing only) ===");
        return;
    }

    let programs = Programs::new(&reg);
    let mut ck = Checks::default();

    let machine = if wanted(0) { section_reference() } else { Vec::new() };
    // Section 1 is not optional: every later table subtracts it.
    let fixed = section_fixed(&programs, &reg);
    if wanted(2) {
        section_matmul(&mut ck, &reg, &programs, &fixed);
    }
    if wanted(3) {
        section_mixed(&mut ck, &reg);
    }
    let (rms, sm) = if wanted(4) {
        let r = section_rmsnorm(&mut ck, &reg, &programs, &fixed, &machine);
        let s = section_softmax(&mut ck, &reg, &programs, &fixed, &machine);
        section_kernels(&mut ck, &reg, &programs);
        (r, s)
    } else {
        (Vec::new(), Vec::new())
    };
    if wanted(5) {
        section_multi_statement(&mut ck, &reg);
    }
    if wanted(6) {
        section_dynamic(&mut ck, &reg, &programs);
    }

    recheck_fixed(&programs, &reg, &fixed);
    summarize(&rms, &sm, &fixed);

    println!("\n=== correctness summary ===");
    println!("  bit-identical pairs ({}):", ck.identical.len());
    for c in &ck.identical {
        println!("    = {c}");
    }
    println!("  close but not bit-identical ({}):", ck.close.len());
    for c in &ck.close {
        println!("    ~ {c}");
    }
    if ck.bad.is_empty() {
        println!("  no disagreement beyond tolerance");
    } else {
        println!("  {} DISAGREEMENT(S):", ck.bad.len());
        for b in &ck.bad {
            println!("    - {b}");
        }
        std::process::exit(1);
    }
    println!("\n=== done ===");
}
