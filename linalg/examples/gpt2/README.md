# GPT-2 on `linalg`

A small GPT-2-shaped decoder-only transformer, decoded one token at a time on
`linalg`. It exists to show that a full transformer forward pass expresses in
the crate's own vocabulary, and to check the Rust implementation against a
Python reference bit-for-bit (modulo float summation order).

The example implements the **same forward pass twice** and runs both:

| backend | what it uses |
|---|---|
| `Backend::Einsum` | every contraction through the einsum VM (`einsum_homogenous`); RMSNorm, ReLU, softmax, residual adds and the `1/sqrt(head_dim)` scale as hand-rolled loops over the flat `Dense<f32>` storage |
| `Backend::Lang`   | **all** of it — contractions *and* elementwise — as five [`linalg::lang`] programs, parsed and scope-checked once at load and `run` per decode step |

The two are compared against each other (they agree bit for bit) and the
language's logits are compared against the NumPy reference.

The same architecture is written **three times against one set of einsum specs**:

| File | Backend | Role |
|---|---|---|
| `train.py` | PyTorch (`torch.einsum`) | trains the model, exports weights |
| `gpt2_reference.py` | NumPy (`np.einsum`) | reference forward, dumps logits |
| `main.rs` | `linalg` einsum VM *and* `linalg::lang` | the example, compared against the reference |

## Architecture

GPT-2 skeleton with two modern substitutions (this is why it is **not** a real
GPT-2 checkpoint and has nothing to download):

- Learned **token + absolute position** embeddings (`wte` + `wpe`), added and
  then passed through an initial RMSNorm.
- Pre-norm blocks: causal self-attention (with a KV cache) + a two-layer MLP,
  each wrapped in a residual around the *pre-norm* value.
- **RMSNorm** instead of LayerNorm — no learned gain, no bias.
- **ReLU** MLP activation instead of GELU.
- No biases anywhere; every projection is a pure einsum.
- No final norm before the LM head (the reference just copies the last block
  output).

The einsum specs, shared across all three implementations (batch/position axes
`b`,`t`,`s` added in the training/reference forwards):

| step | spec (decode) | meaning |
|---|---|---|
| Q/K/V projection | `hjd,d->hj` | per-head weight `[h,j,d]` · normed `[d]` |
| attention logits | `hj,thj->ht` | query `[h,j]` · cached keys `[t,h,j]` |
| head output | `ht,thj->hj` | weights `[h,t]` · cached values `[t,h,j]` |
| output projection | `ohj,hj->o` | `wo [o,h,j]` · heads `[h,j]` |
| MLP / LM head | `od,d->o` | dense weight `[o,d]` · `[d]` |

## The forward pass as language programs

`cargo run --release --example gpt2 -- programs` prints them. There are five,
split only by where the KV cache has to be appended to and where a residual has
to be captured — nothing about the math forced the split. The dimensions and
`rms_eps` are baked in as literals from `config.txt`:

```text
── program embed ──
j[d] = tok[d] + pos[d]
x[d] = j[d] * (1.0 / sqrt(sum(e: j[e] * j[e]) / 64.0 + 1e-5))

── program qkv ──
nx[d] = x[d] * (1.0 / sqrt(sum(e: x[e] * x[e]) / 64.0 + 1e-5))
q[h,j] = sum(d: wq[h,j,d] * nx[d])
k[h,j] = sum(d: wk[h,j,d] * nx[d])
v[h,j] = sum(d: wv[h,j,d] * nx[d])

── program attn ──
s[h,t]  = sum(j: q[h,j] * keys[t,h,j]) * 2.5e-1
mx[h]   = max(t: s[h,t])
ex[h,t] = exp(s[h,t] - mx[h])
zs[h]   = sum(t: ex[h,t])
w[h,t]  = ex[h,t] * (1.0 / zs[h])
hd[h,j] = sum(t: w[h,t] * vals[t,h,j])
pr[o]   = sum(h, j: wo[o,h,j] * hd[h,j])
y[o]    = pr[o] + x[o]

── program mlp ──
nx[d] = x[d] * (1.0 / sqrt(sum(e: x[e] * x[e]) / 64.0 + 1e-5))
h1[o] = relu(sum(d: fc1[o,d] * nx[d]))
y[e]  = sum(m: fc2[e,m] * h1[m]) + x[e]

── program head ──
logits[v] = sum(d: lm[v,d] * x[d])
```

Notes on the two idioms that look roundabout:

- RMSNorm and softmax multiply by a **hoisted reciprocal** (`… * (1.0 / z)`)
  rather than dividing per element. The kernel path hoists any subexpression
  that does not depend on the innermost loop index, so this is one reciprocal
  per row — and it is the operation sequence the hand-rolled loops perform, so
  the two backends stay bit-identical.
- Softmax is written in its stable form (subtract the row max), matching the
  reference.

Four things the surrounding Rust still does, because the language does not
express them:

| step | why |
|---|---|
| `wte[token] / wpe[pos]` row slice | an embedding **gather**: indexing by a runtime value, not by a loop index |
| appending `k`/`v` to the KV cache | a **scatter/append** into a growing buffer |
| `argmax(logits)` | reductions yield **values, not positions** |
| allocating each output | plumbing, not arithmetic |

Every floating-point operation applied to a model value is in the language.

## Running

```sh
# 1. Train (PyTorch, CPU, ~15 s after the one-time torch download) and export
#    weights/ (weights.bin, config.txt, vocab.bin, prompt.txt).
uv run examples/gpt2/train.py

# 2. NumPy reference forward: greedy-decode and write weights/ref_logits.bin.
uv run examples/gpt2/gpt2_reference.py

# 3. The Rust example: load the same weights, decode on both backends, and
#    compare against each other and against the reference.
cargo run --release --example gpt2
cargo run --release --example gpt2 -- programs   # print the programs
cargo run --release --example gpt2 -- bench      # time both backends
cargo run --release --example gpt2 -- bench static   # …with the KV length baked in
```

Add `--features jit` to compile the language's eligible contractions with
Cranelift instead of running them on the blocked tape kernel.

`uv` is used per the repo convention — no manual venv needed; dependencies are
declared inline (PEP 723) at the top of each script.

Expected tail of step 3:

```
prompt: "the spar"
output: "the sparse tensor hums a quiet t"
tokens: [21, 10, 7, 0, 20, 17, 3, 19, ...]

── language execution ──
runs               : 352 (1536 statements), 1536 on the kernel fast path
reductions         : 1120 JIT / 96 tape (96 with a fused store map)
scalar fn calls    : 96 inline / 0 indirect
dynamic axes       : ["t"], on 288 of 1120 JIT'd reductions
kernel cache keys  : 9 (one Cranelift compile each)

── language vs einsum VM ──
token streams      : identical
max abs logit diff : 0.000e0
bit-identical      : 648/648 logits

── comparison vs NumPy reference ──
steps compared     : 24
max abs logit diff : 1.717e-5
argmax agreement   : 24/24
RESULT: MATCH ✓
```

The `~1e-5` residual is float32 summation-order difference between `linalg`,
NumPy, and PyTorch; the greedy token streams are identical. The 96 tape
reductions are the softmax row maxima — `max`/`min` folds are deliberately kept
off the JIT because Cranelift's `fmax`/`fmin` disagree with the language's
comparison fold on `±0.0` and NaN.

### Benchmark

`-- bench` decodes the whole context from a cold KV cache on both backends and
reports per-token latency, plus what the *growing* KV cache costs the language.

The KV length `t` is the one extent in the decode that changes from step to
step. It is marked **dynamic** (`RunOptions::dynamic`), so the three
`t`-dependent contraction sites (`s`, `zs`, `hd`) take it as a runtime argument
and one compiled kernel serves every context length; everything else — `n_embd`,
`n_head`, `head_dim`, `mlp_hidden`, `vocab` — stays a baked constant. That is
**9 compiled kernels per decode instead of 102**, and a cold decode within
~1.3× of a warm one instead of ~4.3×. Append `static` (`-- bench static`,
`-- static`) to un-mark it and measure the difference — only the first decode in
a process is cold, so the two configurations have to be run as two processes.
The logits are bit-identical either way; the example asserts that on every run.

### Running without the trained weights

`cargo run --release --example gpt2` works with no `weights/` directory: it
falls back to deterministic random weights (a self-contained smoke test that
exercises both backends and checks them against each other) and skips the
reference comparison.

## Retraining / reconfiguring

Model size, corpus, and training length live at the top of `train.py`
(`N_EMBD`, `N_HEAD`, `N_LAYER`, `MLP_HIDDEN`, `BLOCK_SIZE`, `STEPS`, `CORPUS`,
`N_GENERATE`). The exported `config.txt` carries the dims, so after retraining
the NumPy reference and the Rust example pick up the new shape automatically —
no code changes needed (the language programs are formatted from `config.txt`
at startup). Re-run steps 2 and 3 after any retrain so `ref_logits.bin` matches
the new weights.

The checked-in `weights/` is the model produced by the committed `train.py`, so
steps 2–3 run out of the box without retraining.

## Files

```
examples/gpt2/
├── README.md            this file
├── train.py             PyTorch trainer + weight exporter
├── gpt2_reference.py    NumPy reference forward + logit dump
├── main.rs              the linalg example (loads weights, decodes, compares)
└── weights/             exported model + reference logits
    ├── config.txt       dims (vocab, n_embd, n_head, n_layer, …)
    ├── weights.bin      all parameters, row-major f32 LE, fixed order
    ├── vocab.bin        byte i = ASCII char for token i
    ├── prompt.txt       the decode prompt
    └── ref_logits.bin   NumPy per-step logits [n_step, vocab] for comparison
```

The `weights.bin` byte layout (matching `train.py`'s export and `main.rs`'s
loader): `wte`, `wpe`, then for each layer `wq, wk, wv, wo, fc1, fc2`, then
`lm_head` — each a contiguous row-major f32 block.
