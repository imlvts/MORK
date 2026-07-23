//! A GPT-2-shaped decoder-only transformer, run one token at a time on top of
//! `linalg` — loading weights trained by `train.py` and checking its output
//! against the NumPy reference in `gpt2_reference.py`.
//!
//! The same forward pass is implemented **twice**, and the example runs both
//! and compares them:
//!
//! - [`Backend::Einsum`] — every contraction through the einsum VM
//!   ([`einsum_homogenous`]) with the same specs as the Python reference
//!   (`"hjd,d->hj"`, `"hj,thj->ht"`, `"ohj,hj->o"`, …); the pieces einsum does
//!   not cover (RMSNorm, ReLU, softmax, residual adds, the attention scale)
//!   are hand-rolled loops over the flat `Dense<f32>` storage.
//! - [`Backend::Lang`] — *all* of the per-step math, contractions and
//!   elementwise alike, as five [`linalg::lang`] programs (see [`Programs`]),
//!   parsed and checked once at load time and `run` per decode step.
//!
//! ## Architecture
//!
//! GPT-2 skeleton (learned token + absolute position embeddings, pre-norm
//! blocks, causal self-attention with a KV cache, output projection, MLP) with
//! **RMSNorm** (no gain) instead of LayerNorm, **ReLU** instead of GELU, and no
//! biases anywhere.
//!
//! ## Workflow
//!
//! ```sh
//! uv run examples/gpt2/train.py            # train + export weights/
//! uv run examples/gpt2/gpt2_reference.py   # NumPy reference + ref_logits.bin
//! cargo run --release --example gpt2       # this: same decode, compared
//! cargo run --release --example gpt2 -- bench   # time both backends
//! ```
//!
//! With no `weights/` directory present it falls back to deterministic random
//! weights (a self-contained smoke test) and skips the comparison.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use linalg::dense::Dense;
use linalg::einsum::einsum_homogenous;
use linalg::lang::{Checked, Registry, RunReport, check, parse, run, run_reported};
use linalg::tensor::NDIndex;

// ─────────────────────────────────────────────────────────────────────────
// Config (loaded at runtime from weights/config.txt)
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Config {
    vocab: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    head_dim: usize,
    mlp_hidden: usize,
    block_size: usize,
    n_generate: usize,
    rms_eps: f32,
}

impl Config {
    /// Fallback config used when no trained weights are present.
    fn default_demo() -> Self {
        Config {
            vocab: 128,
            n_embd: 64,
            n_head: 4,
            n_layer: 3,
            head_dim: 16,
            mlp_hidden: 256,
            block_size: 64,
            n_generate: 12,
            rms_eps: 1e-5,
        }
    }

    fn parse(text: &str) -> Self {
        let mut c = Config::default_demo();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let (Some(k), Some(v)) = (it.next(), it.next()) else { continue };
            match k {
                "vocab" => c.vocab = v.parse().unwrap(),
                "n_embd" => c.n_embd = v.parse().unwrap(),
                "n_head" => c.n_head = v.parse().unwrap(),
                "n_layer" => c.n_layer = v.parse().unwrap(),
                "head_dim" => c.head_dim = v.parse().unwrap(),
                "mlp_hidden" => c.mlp_hidden = v.parse().unwrap(),
                "block_size" => c.block_size = v.parse().unwrap(),
                "n_generate" => c.n_generate = v.parse().unwrap(),
                "rms_eps" => c.rms_eps = v.parse().unwrap(),
                _ => {}
            }
        }
        c
    }
}

// ─────────────────────────────────────────────────────────────────────────
// einsum helper — allocate the output and run the VM, mirroring Python's `E`.
// ─────────────────────────────────────────────────────────────────────────

fn e(spec: &str, inputs: &[&Dense<f32>], out_shape: Vec<usize>) -> Dense<f32> {
    let mut out = Dense::<f32>::zeros(out_shape);
    einsum_homogenous::<f32, Dense<f32>, Dense<f32>>(spec, inputs, &mut [&mut out])
        .unwrap_or_else(|err| panic!("einsum {spec:?} failed: {err}"));
    out
}

// ─────────────────────────────────────────────────────────────────────────
// Elementwise ops the einsum VM doesn't cover (all over flat storage).
// Used by `Backend::Einsum` only — `Backend::Lang` expresses all of this.
// ─────────────────────────────────────────────────────────────────────────

/// Build a new tensor by applying `f` elementwise (shape preserved).
fn map(x: &Dense<f32>, f: impl Fn(f32) -> f32) -> Dense<f32> {
    Dense { data: x.data.iter().map(|&v| f(v)).collect(), shape: x.shape.clone() }
}

/// Row-vector RMSNorm: `x / sqrt(mean(x²) + eps)`. No learned gain.
fn rmsnorm(x: &Dense<f32>, eps: f32) -> Dense<f32> {
    let ms = x.data.iter().map(|&v| v * v).sum::<f32>() / x.data.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    map(x, |v| v * inv)
}

fn relu(x: &Dense<f32>) -> Dense<f32> {
    map(x, |v| v.max(0.0))
}

fn add(a: &Dense<f32>, b: &Dense<f32>) -> Dense<f32> {
    debug_assert_eq!(a.shape, b.shape);
    Dense {
        data: a.data.iter().zip(&b.data).map(|(&x, &y)| x + y).collect(),
        shape: a.shape.clone(),
    }
}

fn scale(x: &Dense<f32>, s: f32) -> Dense<f32> {
    map(x, |v| v * s)
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

// ─────────────────────────────────────────────────────────────────────────
// The same math as five `linalg::lang` programs
// ─────────────────────────────────────────────────────────────────────────

/// RMSNorm as one language statement, written so that it performs *exactly*
/// the float operations [`rmsnorm`] does: the reciprocal is loop-invariant, so
/// the kernel path hoists `1 / sqrt(…)` out of the element loop and multiplies
/// — it does not divide per element, which would be a different result.
fn rmsnorm_stmt(src: &str, dst: &str, n_embd: usize, eps: f32) -> String {
    format!(
        "{dst}[d] = {src}[d] * (1.0 / sqrt(sum(e: {src}[e] * {src}[e]) / {n_embd}.0 + {eps:e}))"
    )
}

/// The decode-step math, parsed and scope-checked once.
///
/// Split into five programs by where the KV cache has to be touched and where
/// a residual has to be captured — not for any expressive reason. Everything
/// each program does (contractions, RMSNorm, the attention scale, softmax,
/// ReLU, residual adds) is in the language; the surrounding Rust only moves
/// tensors around.
struct Programs {
    reg: Registry<f32>,
    /// `wte` row + `wpe` row → normed block input.
    embed: Checked,
    /// pre-norm + the three head projections.
    qkv: Checked,
    /// scores, scaled softmax, head readout, output projection, residual.
    attn: Checked,
    /// pre-norm + fc1 + ReLU + fc2 + residual.
    mlp: Checked,
    /// LM head.
    head: Checked,
    /// Source of each program, in the order they run (for `-- programs`).
    sources: Vec<(&'static str, String)>,
}

impl Programs {
    fn new(cfg: Config) -> Self {
        let (n, eps) = (cfg.n_embd, cfg.rms_eps);
        let inv_scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let embed = format!("j[d] = tok[d] + pos[d]\n{}\n", rmsnorm_stmt("j", "x", n, eps));

        let qkv = format!(
            "{}\n\
             q[h,j] = sum(d: wq[h,j,d] * nx[d])\n\
             k[h,j] = sum(d: wk[h,j,d] * nx[d])\n\
             v[h,j] = sum(d: wv[h,j,d] * nx[d])\n",
            rmsnorm_stmt("x", "nx", n, eps)
        );

        // Softmax is written in its stable form and, like `rmsnorm_stmt`,
        // multiplies by a hoisted reciprocal rather than dividing per element,
        // so it matches `softmax_last` operation for operation.
        let attn = format!(
            "s[h,t]  = sum(j: q[h,j] * keys[t,h,j]) * {inv_scale:e}\n\
             mx[h]   = max(t: s[h,t])\n\
             ex[h,t] = exp(s[h,t] - mx[h])\n\
             zs[h]   = sum(t: ex[h,t])\n\
             w[h,t]  = ex[h,t] * (1.0 / zs[h])\n\
             hd[h,j] = sum(t: w[h,t] * vals[t,h,j])\n\
             pr[o]   = sum(h, j: wo[o,h,j] * hd[h,j])\n\
             y[o]    = pr[o] + x[o]\n"
        );

        let mlp = format!(
            "{}\n\
             h1[o] = relu(sum(d: fc1[o,d] * nx[d]))\n\
             y[e]  = sum(m: fc2[e,m] * h1[m]) + x[e]\n",
            rmsnorm_stmt("x", "nx", n, eps)
        );

        let head = "logits[v] = sum(d: lm[v,d] * x[d])\n".to_string();

        let sources = vec![
            ("embed", embed),
            ("qkv", qkv),
            ("attn", attn),
            ("mlp", mlp),
            ("head", head),
        ];
        let reg = Registry::<f32>::builtins();
        let compile = |src: &str| -> Checked {
            check(&parse(src).unwrap_or_else(|err| panic!("parse: {err}\n{src}")), &reg)
                .unwrap_or_else(|err| panic!("check: {err}\n{src}"))
        };
        Programs {
            embed: compile(&sources[0].1),
            qkv: compile(&sources[1].1),
            attn: compile(&sources[2].1),
            mlp: compile(&sources[3].1),
            head: compile(&sources[4].1),
            reg,
            sources,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Fast-path accounting
// ─────────────────────────────────────────────────────────────────────────

/// Sum of the [`RunReport`]s of every `run` in a decode — how much of the
/// program the kernel path took and where its reductions went.
#[derive(Default, Clone, Copy)]
struct RunStats {
    runs: usize,
    kernel_runs: usize,
    statements: usize,
    kernel_statements: usize,
    jit_reductions: usize,
    tape_reductions: usize,
    store_maps: usize,
    inline_fn_ops: usize,
    indirect_fn_ops: usize,
}

impl RunStats {
    fn add(&mut self, r: RunReport) {
        self.runs += 1;
        self.kernel_runs += usize::from(r.all_kernel());
        self.statements += r.statements;
        self.kernel_statements += r.kernel_statements;
        self.jit_reductions += r.jit_reductions;
        self.tape_reductions += r.tape_reductions;
        self.store_maps += r.store_maps;
        self.inline_fn_ops += r.inline_fn_ops;
        self.indirect_fn_ops += r.indirect_fn_ops;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Weights
// ─────────────────────────────────────────────────────────────────────────

struct Layer {
    wq: Dense<f32>,  // [h, j, d]
    wk: Dense<f32>,  // [h, j, d]
    wv: Dense<f32>,  // [h, j, d]
    wo: Dense<f32>,  // [o, h, j]
    fc1: Dense<f32>, // [mlp_hidden, n_embd]
    fc2: Dense<f32>, // [n_embd, mlp_hidden]
}

struct Model {
    cfg: Config,
    wte: Dense<f32>,     // [vocab, n_embd]
    wpe: Dense<f32>,     // [block_size, n_embd]
    layers: Vec<Layer>,
    lm_head: Dense<f32>, // [vocab, n_embd]
}

/// Sequential f32 reader over `weights.bin`, matching the export order.
struct Blob {
    data: Vec<f32>,
    off: usize,
}
impl Blob {
    fn take(&mut self, shape: Vec<usize>) -> Dense<f32> {
        let n: usize = shape.iter().product();
        let slice = &self.data[self.off..self.off + n];
        self.off += n;
        Dense { data: slice.to_vec(), shape }
    }
}

/// Deterministic small-magnitude weight fill (fallback demo mode).
struct Lcg(u64);
impl Lcg {
    fn tensor(&mut self, shape: Vec<usize>) -> Dense<f32> {
        let mut t = Dense::<f32>::zeros(shape);
        for v in t.data.iter_mut() {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.02;
        }
        t
    }
}

impl Model {
    fn from_blob(cfg: Config, mut b: Blob) -> Self {
        let (h, j, d, o, mlp) = (cfg.n_head, cfg.head_dim, cfg.n_embd, cfg.n_embd, cfg.mlp_hidden);
        let wte = b.take(vec![cfg.vocab, d]);
        let wpe = b.take(vec![cfg.block_size, d]);
        let layers = (0..cfg.n_layer)
            .map(|_| Layer {
                wq: b.take(vec![h, j, d]),
                wk: b.take(vec![h, j, d]),
                wv: b.take(vec![h, j, d]),
                wo: b.take(vec![o, h, j]),
                fc1: b.take(vec![mlp, d]),
                fc2: b.take(vec![d, mlp]),
            })
            .collect();
        let lm_head = b.take(vec![cfg.vocab, d]);
        assert_eq!(b.off, b.data.len(), "weight blob length mismatch");
        Model { cfg, wte, wpe, layers, lm_head }
    }

    fn random(cfg: Config, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let (h, j, d, o, mlp) = (cfg.n_head, cfg.head_dim, cfg.n_embd, cfg.n_embd, cfg.mlp_hidden);
        let layers = (0..cfg.n_layer)
            .map(|_| Layer {
                wq: r.tensor(vec![h, j, d]),
                wk: r.tensor(vec![h, j, d]),
                wv: r.tensor(vec![h, j, d]),
                wo: r.tensor(vec![o, h, j]),
                fc1: r.tensor(vec![mlp, d]),
                fc2: r.tensor(vec![d, mlp]),
            })
            .collect();
        Model {
            cfg,
            wte: r.tensor(vec![cfg.vocab, d]),
            wpe: r.tensor(vec![cfg.block_size, d]),
            layers,
            lm_head: r.tensor(vec![cfg.vocab, d]),
        }
    }
}

/// Row `i` of a `[rows, cols]` dense tensor as a fresh `[cols]` vector.
fn row(t: &Dense<f32>, i: usize) -> Dense<f32> {
    let cols = t.shape[1];
    Dense { data: t.data[i * cols..(i + 1) * cols].to_vec(), shape: vec![cols] }
}

// ─────────────────────────────────────────────────────────────────────────
// KV cache — per layer, a flat [T, n_head, head_dim] buffer that grows by one
// position each decode step.
// ─────────────────────────────────────────────────────────────────────────

struct Cache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
}
impl Cache {
    fn new(n_layer: usize) -> Self {
        Cache { keys: vec![Vec::new(); n_layer], values: vec![Vec::new(); n_layer], len: 0 }
    }
    fn push(&mut self, li: usize, k: &Dense<f32>, v: &Dense<f32>) {
        self.keys[li].extend_from_slice(&k.data);
        self.values[li].extend_from_slice(&v.data);
    }
    fn prefix(&self, li: usize, n_head: usize, head_dim: usize) -> (Dense<f32>, Dense<f32>) {
        let shape = vec![self.len, n_head, head_dim];
        (
            Dense { data: self.keys[li].clone(), shape: shape.clone() },
            Dense { data: self.values[li].clone(), shape },
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Forward: one token at position `pos`, returns logits `[vocab]`.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    /// einsum VM contractions + hand-rolled elementwise loops.
    Einsum,
    /// Everything as `linalg::lang` programs.
    Lang,
}

fn calculate_step_einsum(
    m: &Model,
    cache: &mut Cache,
    token_id: usize,
    pos: usize,
) -> Dense<f32> {
    let cfg = m.cfg;
    let head_scale = (cfg.head_dim as f32).sqrt();

    let tok_emb = row(&m.wte, token_id);
    let pos_emb = row(&m.wpe, pos);
    let joint = add(&tok_emb, &pos_emb);
    let mut x = rmsnorm(&joint, cfg.rms_eps); // block_input_0

    cache.len = pos + 1;

    for (li, layer) in m.layers.iter().enumerate() {
        // ── Attention ──
        let attn_residual = x.clone();
        let normed = rmsnorm(&x, cfg.rms_eps);

        let q = e("hjd,d->hj", &[&layer.wq, &normed], vec![cfg.n_head, cfg.head_dim]);
        let k = e("hjd,d->hj", &[&layer.wk, &normed], vec![cfg.n_head, cfg.head_dim]);
        let v = e("hjd,d->hj", &[&layer.wv, &normed], vec![cfg.n_head, cfg.head_dim]);

        cache.push(li, &k, &v);
        let (key_prefix, value_prefix) = cache.prefix(li, cfg.n_head, cfg.head_dim);
        let t = cache.len;

        let logits = e("hj,thj->ht", &[&q, &key_prefix], vec![cfg.n_head, t]);
        let logits = scale(&logits, 1.0 / head_scale);
        let weights = softmax_last(&logits);
        let head_output =
            e("ht,thj->hj", &[&weights, &value_prefix], vec![cfg.n_head, cfg.head_dim]);

        let attn_projected = e("ohj,hj->o", &[&layer.wo, &head_output], vec![cfg.n_embd]);
        x = add(&attn_projected, &attn_residual);

        // ── MLP ──
        let mlp_residual = x.clone();
        let mlp_norm = rmsnorm(&x, cfg.rms_eps);
        let fc1_out = e("od,d->o", &[&layer.fc1, &mlp_norm], vec![cfg.mlp_hidden]);
        let hidden = relu(&fc1_out);
        let fc2_out = e("od,d->o", &[&layer.fc2, &hidden], vec![cfg.n_embd]);
        x = add(&fc2_out, &mlp_residual);
    }

    // No final norm (the reference just copies the last block output).
    e("od,d->o", &[&m.lm_head, &x], vec![cfg.vocab])
}

/// The same forward pass, with every arithmetic step expressed in the
/// language. The Rust here allocates outputs, slices embedding rows, and
/// grows the KV cache — it performs no arithmetic on model values.
fn calculate_step_lang(
    m: &Model,
    p: &Programs,
    cache: &mut Cache,
    stats: &mut RunStats,
    token_id: usize,
    pos: usize,
) -> Dense<f32> {
    let cfg = m.cfg;
    let tok_emb = row(&m.wte, token_id);
    let pos_emb = row(&m.wpe, pos);

    let mut x = Dense::<f32>::zeros(vec![cfg.n_embd]);
    stats.add(
        run_reported(
            &p.embed,
            &p.reg,
            &[("tok", &tok_emb as &dyn NDIndex<f32>), ("pos", &pos_emb)],
            &mut [("x", &mut x as &mut dyn NDIndex<f32>)],
        )
        .unwrap(),
    );

    cache.len = pos + 1;

    for (li, layer) in m.layers.iter().enumerate() {
        // ── Attention ──
        let mut q = Dense::<f32>::zeros(vec![cfg.n_head, cfg.head_dim]);
        let mut k = Dense::<f32>::zeros(vec![cfg.n_head, cfg.head_dim]);
        let mut v = Dense::<f32>::zeros(vec![cfg.n_head, cfg.head_dim]);
        stats.add(
            run_reported(
                &p.qkv,
                &p.reg,
                &[
                    ("x", &x as &dyn NDIndex<f32>),
                    ("wq", &layer.wq),
                    ("wk", &layer.wk),
                    ("wv", &layer.wv),
                ],
                &mut [
                    ("q", &mut q as &mut dyn NDIndex<f32>),
                    ("k", &mut k),
                    ("v", &mut v),
                ],
            )
            .unwrap(),
        );

        cache.push(li, &k, &v);
        let (key_prefix, value_prefix) = cache.prefix(li, cfg.n_head, cfg.head_dim);

        let mut y = Dense::<f32>::zeros(vec![cfg.n_embd]);
        stats.add(
            run_reported(
                &p.attn,
                &p.reg,
                &[
                    ("q", &q as &dyn NDIndex<f32>),
                    ("keys", &key_prefix),
                    ("vals", &value_prefix),
                    ("wo", &layer.wo),
                    ("x", &x),
                ],
                &mut [("y", &mut y as &mut dyn NDIndex<f32>)],
            )
            .unwrap(),
        );
        x = y;

        // ── MLP ──
        let mut y = Dense::<f32>::zeros(vec![cfg.n_embd]);
        stats.add(
            run_reported(
                &p.mlp,
                &p.reg,
                &[("x", &x as &dyn NDIndex<f32>), ("fc1", &layer.fc1), ("fc2", &layer.fc2)],
                &mut [("y", &mut y as &mut dyn NDIndex<f32>)],
            )
            .unwrap(),
        );
        x = y;
    }

    // No final norm (the reference just copies the last block output).
    let mut logits = Dense::<f32>::zeros(vec![cfg.vocab]);
    stats.add(
        run_reported(
            &p.head,
            &p.reg,
            &[("lm", &m.lm_head as &dyn NDIndex<f32>), ("x", &x)],
            &mut [("logits", &mut logits as &mut dyn NDIndex<f32>)],
        )
        .unwrap(),
    );
    logits
}

fn calculate_step(
    m: &Model,
    p: &Programs,
    backend: Backend,
    cache: &mut Cache,
    stats: &mut RunStats,
    token_id: usize,
    pos: usize,
) -> Dense<f32> {
    match backend {
        Backend::Einsum => calculate_step_einsum(m, cache, token_id, pos),
        Backend::Lang => calculate_step_lang(m, p, cache, stats, token_id, pos),
    }
}

fn argmax_slice(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0
}

/// Full greedy decode from a fresh KV cache. Returns the token stream (prompt
/// + generated) and the per-step logits recorded for the reference comparison.
/// One forward pass runs per token position.
fn run_decode(
    model: &Model,
    progs: &Programs,
    backend: Backend,
    prompt: &[usize],
    stats: &mut RunStats,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let cfg = model.cfg;
    let mut cache = Cache::new(cfg.n_layer);
    let mut tokens = prompt.to_vec();
    let mut pos = 0usize;

    let mut logits = Dense::<f32>::zeros(vec![cfg.vocab]);
    for &tok in prompt {
        logits = calculate_step(model, progs, backend, &mut cache, stats, tok, pos);
        pos += 1;
    }

    let mut step_logits: Vec<Vec<f32>> = Vec::new();
    for _ in 0..cfg.n_generate {
        if pos >= cfg.block_size {
            break;
        }
        step_logits.push(logits.data.clone());
        let next = argmax_slice(&logits.data);
        tokens.push(next);
        logits = calculate_step(model, progs, backend, &mut cache, stats, next, pos);
        pos += 1;
    }
    (tokens, step_logits)
}

// ─────────────────────────────────────────────────────────────────────────
// Comparison
// ─────────────────────────────────────────────────────────────────────────

struct Diff {
    steps: usize,
    max_abs: f32,
    argmax_matches: usize,
    bit_identical: usize,
    elements: usize,
}

fn diff_rows(a: &[&[f32]], b: &[&[f32]]) -> Diff {
    let steps = a.len().min(b.len());
    let mut d =
        Diff { steps, max_abs: 0.0, argmax_matches: 0, bit_identical: 0, elements: 0 };
    for s in 0..steps {
        for (&x, &y) in a[s].iter().zip(b[s]) {
            d.max_abs = d.max_abs.max((x - y).abs());
            d.bit_identical += usize::from(x.to_bits() == y.to_bits());
            d.elements += 1;
        }
        d.argmax_matches += usize::from(argmax_slice(a[s]) == argmax_slice(b[s]));
    }
    d
}

// ─────────────────────────────────────────────────────────────────────────
// Benchmark
// ─────────────────────────────────────────────────────────────────────────

/// Seconds per call of `f`, best of several batches after a warmup.
fn time_call<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..20_000 {
        f();
    }
    let iters = 100_000u32;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / iters as f64);
    }
    best
}

/// Per-`run` fixed cost: bind, lower, and build the kernel tape, on a program
/// small enough that its arithmetic is noise. Every `run` pays this, and the
/// decode cannot amortize it because the KV extent changes every step, so no
/// bound plan survives to the next one.
fn probe_map() -> f64 {
    let reg = Registry::<f32>::builtins();
    let checked = check(&parse("y[i] = x[i] + 1.0").unwrap(), &reg).unwrap();
    let x = Dense::<f32>::zeros(vec![2]);
    let mut y = Dense::<f32>::zeros(vec![2]);
    time_call(|| {
        run(
            &checked,
            &reg,
            &[("x", &x as &dyn NDIndex<f32>)],
            &mut [("y", &mut y as &mut dyn NDIndex<f32>)],
        )
        .unwrap();
    })
}

/// [`probe_map`] plus one `n`×`n` contraction — so it also pays a lookup in
/// the shape-keyed compiled-kernel cache, which is a linear scan over every
/// distinct (spec, shapes, ops) the process has compiled. A decode adds one
/// entry per contraction site *per KV length*, so running this at two
/// different `n` — one whose key was seeded before the decode and so sits at
/// the front of the scan, one first seen after it and so sitting at the back
/// — brackets what shape-rebinding costs per reduction.
fn probe_reduce(n: usize) -> f64 {
    let reg = Registry::<f32>::builtins();
    let checked = check(&parse("y[i] = sum(j: a[i,j] * b[j])").unwrap(), &reg).unwrap();
    let a = Dense::<f32>::zeros(vec![n, n]);
    let b = Dense::<f32>::zeros(vec![n]);
    let mut y = Dense::<f32>::zeros(vec![n]);
    time_call(|| {
        run(
            &checked,
            &reg,
            &[("a", &a as &dyn NDIndex<f32>), ("b", &b)],
            &mut [("y", &mut y as &mut dyn NDIndex<f32>)],
        )
        .unwrap();
    })
}

/// Time a full greedy decode from a cold cache on both backends and report
/// per-token latency, throughput, and the language's fixed per-step costs.
/// Best of several batches, to suppress scheduler noise.
fn bench(model: &Model, progs: &Programs, prompt: &[usize]) {
    use std::hint::black_box;

    let cfg = model.cfg;

    // Seed the compiled-kernel cache with the 2×2 probe's key before the
    // decode fills it, so that at the end of the run that key is at the front
    // of the scan and the 3×3 probe's key is at the back.
    let _ = probe_reduce(2);

    // The very first language decode in this process: every JIT kernel is
    // compiled here, one per (contraction site, KV length) pair, because the
    // cache key includes the shapes. Must be measured before any warmup.
    let mut cold_stats = RunStats::default();
    let t0 = Instant::now();
    let (tokens, _) = run_decode(model, progs, Backend::Lang, prompt, &mut cold_stats);
    let cold = t0.elapsed().as_secs_f64();
    let steps = tokens.len(); // forwards per decode (prompt + generated)

    // Best of five batches, each calibrated to about a second of work, so the
    // two backends get comparable statistics despite differing by ~12×.
    let mut time = |backend: Backend| -> f64 {
        let mut stats = RunStats::default();
        let warm = Instant::now();
        let mut n = 0u32;
        while warm.elapsed() < Duration::from_millis(200) {
            black_box(run_decode(model, progs, backend, prompt, &mut stats));
            n += 1;
        }
        let iters = ((n as f64 / warm.elapsed().as_secs_f64()) as u32).clamp(3, 100_000);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                black_box(run_decode(model, progs, backend, prompt, &mut stats));
            }
            best = best.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        best
    };
    let vm = time(Backend::Einsum);
    let lang = time(Backend::Lang);

    let runs_per_token = cold_stats.runs as f64 / steps as f64;
    let reduces_per_token =
        (cold_stats.jit_reductions + cold_stats.tape_reductions) as f64 / steps as f64;
    let floor = probe_map();
    let probe_front = probe_reduce(2);
    let probe_back = probe_reduce(3);

    println!("── decode benchmark ──");
    println!(
        "model              : n_embd={} n_head={} n_layer={} mlp={} vocab={}",
        cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.mlp_hidden, cfg.vocab
    );
    println!("forwards / decode  : {steps} (ctx up to block_size={})", cfg.block_size);
    println!(
        "lang runs / token  : {runs_per_token:.0}  ({} statements / token)",
        cold_stats.statements / steps
    );
    println!();
    println!("  {:<16}{:>14}{:>16}{:>14}", "backend", "full decode", "per token", "tokens/s");
    for (name, t) in [("einsum VM", vm), ("lang", lang)] {
        let per_tok_us = t / steps as f64 * 1e6;
        println!(
            "  {name:<16}{:>11.3} ms{per_tok_us:>13.2} µs{:>14.0}",
            t * 1e3,
            1e6 / per_tok_us
        );
    }
    println!("  {:<16}{:>14.2}×", "lang speedup", vm / lang);
    println!();
    println!("── what the growing KV cache costs (RESUME step 6) ──");
    let lang_per_token_us = lang / steps as f64 * 1e6;
    println!(
        "cold decode        : {:.3} ms, {:.3} ms above warm ({:.1}×) — one-off kernel build,\n\
         \x20                    once per (contraction site, KV length) pair because the compiled\n\
         \x20                    kernel is cached by shape (dominated by Cranelift under \
         `--features jit`)",
        cold * 1e3,
        (cold - lang) * 1e3,
        cold / lang
    );
    let setup_us = floor * runs_per_token * 1e6;
    println!(
        "per-run fixed cost : {:.3} µs (bind + lower + build) × {runs_per_token:.0} runs/token \
         = {setup_us:.2} µs/token,\n\
         \x20                    {:.1}% of the warm per-token time",
        floor * 1e6,
        100.0 * setup_us / lang_per_token_us
    );
    let lookup_ns = (probe_back - probe_front).abs() * 1e9;
    let lookup_us = (probe_back - probe_front).abs() * reduces_per_token * 1e6;
    println!(
        "kernel-cache scan  : one contraction costs {:.3} µs with its key at the front of the\n\
         \x20                    cache and {:.3} µs at the back, after a decode has filled it with\n\
         \x20                    one entry per site per KV length — |Δ| {lookup_ns:.0} ns, so at most\n\
         \x20                    {lookup_us:.2} µs/token over {reduces_per_token:.0} reductions ({:.1}% of the warm per-token\n\
         \x20                    time). The scan is not what shape-rebinding costs; compilation is.",
        probe_front * 1e6,
        probe_back * 1e6,
        100.0 * lookup_us / lang_per_token_us
    );
}

// ─────────────────────────────────────────────────────────────────────────
// I/O helpers
// ─────────────────────────────────────────────────────────────────────────

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/gpt2/weights")
}

fn read_f32_le(path: &std::path::Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() {
    let dir = weights_dir();
    let cfg_path = dir.join("config.txt");

    // ── Trained mode if weights exist, else a self-contained random demo. ──
    let (model, prompt_tokens, itos, ref_logits) = if cfg_path.exists() {
        let cfg = Config::parse(&std::fs::read_to_string(&cfg_path).unwrap());
        let blob = Blob { data: read_f32_le(&dir.join("weights.bin")).unwrap(), off: 0 };
        let model = Model::from_blob(cfg, blob);

        let vocab_bytes = std::fs::read(dir.join("vocab.bin")).unwrap();
        let itos: Vec<char> = vocab_bytes.iter().map(|&b| b as char).collect();
        let stoi: std::collections::HashMap<char, usize> =
            itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).unwrap();
        let prompt_tokens: Vec<usize> = prompt.chars().map(|c| stoi[&c]).collect();

        // Reference logits: [n_step, vocab], row-major.
        let ref_logits = read_f32_le(&dir.join("ref_logits.bin")).ok();

        println!(
            "gpt2 (RMSNorm+ReLU) — trained weights, decode on the linalg\n\
             expression language (and on the einsum VM, for comparison)\n\
             config: {:?}\n",
            cfg
        );
        println!("prompt: {prompt:?}");
        (model, prompt_tokens, Some(itos), ref_logits)
    } else {
        let cfg = Config::default_demo();
        println!(
            "gpt2 (RMSNorm+ReLU) — NO trained weights found, using random demo weights\n\
             (run examples/gpt2/train.py to train and compare against the reference)\n"
        );
        (Model::random(cfg, 0x9E3779B97F4A7C15), vec![7, 42, 13, 99], None, None)
    };

    let cfg = model.cfg;
    let progs = Programs::new(cfg);

    // `cargo run --release --example gpt2 -- programs` prints the source of
    // every language program the decode runs.
    if std::env::args().any(|a| a == "programs") {
        for (name, src) in &progs.sources {
            println!("── program {name} ──\n{src}");
        }
        return;
    }

    // `cargo run --release --example gpt2 -- bench` times the decode instead.
    if std::env::args().any(|a| a == "bench") {
        bench(&model, &progs, &prompt_tokens);
        return;
    }

    let mut stats = RunStats::default();
    let (tokens, lang_logits) =
        run_decode(&model, &progs, Backend::Lang, &prompt_tokens, &mut stats);
    let (vm_tokens, vm_logits) =
        run_decode(&model, &progs, Backend::Einsum, &prompt_tokens, &mut RunStats::default());

    // ── Report ──
    if let Some(itos) = &itos {
        let text: String = tokens.iter().map(|&t| itos[t]).collect();
        println!("output: {text:?}");
    }
    println!("tokens: {tokens:?}");

    // ── How the language executed ──
    println!("\n── language execution ──");
    println!(
        "runs               : {} ({} statements), {} on the kernel fast path",
        stats.runs, stats.statements, stats.kernel_statements
    );
    println!(
        "reductions         : {} JIT / {} tape ({} with a fused store map)",
        stats.jit_reductions, stats.tape_reductions, stats.store_maps
    );
    println!(
        "scalar fn calls    : {} inline / {} indirect",
        stats.inline_fn_ops, stats.indirect_fn_ops
    );

    // ── The two backends against each other ──
    let lang_rows: Vec<&[f32]> = lang_logits.iter().map(|r| r.as_slice()).collect();
    let vm_rows: Vec<&[f32]> = vm_logits.iter().map(|r| r.as_slice()).collect();
    let d = diff_rows(&lang_rows, &vm_rows);
    println!("\n── language vs einsum VM ──");
    println!("token streams      : {}", if tokens == vm_tokens { "identical" } else { "DIFFER" });
    println!("max abs logit diff : {:.3e}", d.max_abs);
    println!("bit-identical      : {}/{} logits", d.bit_identical, d.elements);

    // ── Compare against the NumPy reference ──
    if let Some(reference) = ref_logits {
        let vocab = cfg.vocab;
        let ref_rows: Vec<&[f32]> =
            reference.chunks_exact(vocab).take(lang_logits.len()).collect();
        let d = diff_rows(&ref_rows, &lang_rows);
        println!("\n── comparison vs NumPy reference ──");
        println!("steps compared     : {}", d.steps);
        println!("max abs logit diff : {:.3e}", d.max_abs);
        println!("argmax agreement   : {}/{}", d.argmax_matches, d.steps);
        if d.argmax_matches == d.steps && d.max_abs < 1e-2 {
            println!("RESULT: MATCH ✓");
        } else {
            println!("RESULT: MISMATCH ✗");
        }
    } else if itos.is_some() {
        println!("\n(no ref_logits.bin — run gpt2_reference.py to enable the comparison)");
    }
}
