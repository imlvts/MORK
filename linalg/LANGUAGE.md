# The linalg tensor-expression language

Status: **design accepted** (2026-07-23). All previously-open decisions are
resolved (§10). v1 implementation lives in `src/lang/`.

This language replaces the flat einsum spec string (`"ab,bc->ac"` + one
`(Reduce, Combine)` pair) as the primary way to describe tensor computations.
The old string specs remain as a compatibility front-end that desugars into
this language (§7). The execution machinery (VM loop nests, sparse row
iteration, Cranelift JIT, `NDIndex`/`Sparse2D` storage traits) is retained
underneath; what changes is the IR the compiler consumes.

## 1. Scope

The language describes **pointwise-and-reduction computations over tensors**:

- elementwise arithmetic (`+ - * /`, unary functions like `exp`, `sqrt`),
- reductions (`sum`, `prod`, `max`, `min`) over explicitly bound indices,
  freely nested inside expressions,
- broadcasting of lower-rank results across free indices,
- multi-statement programs with named intermediates.

Non-goals (for now): data-dependent control flow, index arithmetic
(`a[i+1]` — the grammar reserves room, §8), in-place mutation, ragged shapes.

## 2. Examples

```text
c[i,k]  = sum(j: a[i,j] * b[j,k])            # matmul
d[i,k]  = min(j: a[i,j] + b[j,k])            # tropical / shortest-path step
t       = sum(i: M[i,i])                     # trace (scalar LHS)
y[j,k]  = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))
s[i,j]  = sum(k,l: T[i,k,l] * A[k,j] * B[l,j])   # multi-index binder

# softmax, naive
softmax[i] = exp(v[i]) / sum(j: exp(v[j]))

# softmax, numerically stable — a two-statement program
m          = max(j: v[j])
softmax[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))

# rmsnorm (n, eps are scalars from the environment)
y[i] = x[i] / sqrt(sum(j: x[j]*x[j]) / n + eps)

# attention readout
o[h,d] = sum(t: w[h,t] * v[t,h,d])
```

## 3. Lexical structure

- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`. Used for tensor names, scalar
  names, and index names. Which of the three an identifier denotes is
  determined by position, never by a symbol table:
  tensor if followed by `[…]`, index if inside `[…]` or a binder list,
  scalar otherwise.
- **Reserved words**: `sum`, `prod`, `max`, `min` — reserved as *expression
  heads* only (they cannot start a call/reduction as ordinary names); they
  are still usable as index names.
- **Numbers**: decimal literals — `1`, `0.5`, `1e-5`, `2.5e3`.
- **Comments**: `#` to end of line.
- **Statements** are newline-terminated. A newline inside unbalanced
  `(…)`/`[…]` continues the statement.

## 4. Grammar

```ebnf
program   := stmt+
stmt      := lhs '=' expr NEWLINE
lhs       := ident indexlist?               (* y[j,k]  |  m  — bare = scalar *)
indexlist := '[' ident (',' ident)* ']'

expr      := term  (('+' | '-') term)*      (* two precedence levels *)
term      := unary (('*' | '/') unary)*
unary     := '-' unary | atom
atom      := number
           | ident '[' indexref (',' indexref)* ']'    (* tensor reference *)
           | ident                                     (* scalar reference *)
           | ident '(' ident (',' ident)* ':' expr ')' (* reduction binder *)
           | ident '(' expr (',' expr)* ')'            (* function call *)
           | '(' expr ')'
indexref  := ident            (* v1; slot reserved for affine index exprs, §8 *)
```

Disambiguation notes:

- The **colon marks a binder**: if the parenthesized content after
  `head(` matches `ident (',' ident)* ':'` it is a **reduction binder**,
  and `head` must name a registered reduction (built-ins: `sum`, `prod`,
  `max`, `min`; the registry is code-extensible, §10 A4). Without a colon
  it is an **elementwise call** — valid for `max`/`min` with exactly two
  arguments (elementwise maximum/minimum). `sum(a, b)` without a colon is
  an error (write `a + b`).
- Function names (`exp`, `sqrt`, …) are *not* reserved: `exp(x)` is a call,
  `exp[i]` is a tensor named `exp`. Unknown function names are rejected at
  validation, not at parse.
- Unary minus binds tighter than binary operators: `-a + b` is `(-a) + b`,
  `-a * b` is `(-a) * b`.

**Built-in scalar functions (v1)**: unary `exp`, `ln`, `sqrt`, `abs`,
`relu`, `tanh`; binary `max`, `min` (colon-less form). The set is a table,
not a language feature — growing it is adding a row (name, arity, VM impl,
JIT emission, zero-preservation flag §6.4).

## 5. Static rules (binding and validation)

Runs after parsing, before lowering. All errors carry source spans.

1. **Binding.** Every index appearing on the RHS must be bound by exactly
   one of: the statement's LHS index list, or one enclosing reduction
   binder. Unbound → error (with a "did you mean `sum(j: …)`?" hint).
   An index bound by both the LHS and a binder → error.
2. **No shadowing.** A binder may not rebind an index already in scope
   (whether from the LHS or an outer binder). Sibling binders may reuse a
   name: `sum(l: …) * min(l: …)` is two disjoint scopes and is legal.
3. **No implicit reduction.** There is no "unbound means summed" rule.
   Classic-einsum behavior exists only in the compatibility desugaring (§7).
4. **Extent inference.** Each index takes its extent from every tensor
   position it indexes (including the LHS output, if its shape is known at
   bind time). All sources must agree; a conflict is an error naming the
   index and two disagreeing sources. Repeated indices within one tensor
   reference (`M[i,i]`) are legal — diagonal access.
5. **Programs.** Statements execute top to bottom. A statement's LHS name
   becomes visible to later statements. Each name is defined at most once
   (SSA-style; see open decision D7). Free names resolve against the
   environment supplied at bind time.

Validation error taxonomy: `UnboundIndex`, `IndexBoundTwice`,
`ShadowedBinder`, `ExtentMismatch`, `UnknownName`, `UnknownFunction`,
`FunctionArity`, `RedefinedName`, `ReductionNeedsColon`.

## 6. Semantics

### 6.1 Statements

`out[i₁,…,iₙ] = e` means: for every tuple in the Cartesian product of the
index extents, evaluate `e` under that binding and **overwrite** the output
element. The output is fully defined by the statement — there is no
caller-must-pre-zero contract, and reduction identities never leak out as
initial values the caller can observe or provide. A bare-identifier LHS is
the 0-dim (scalar) case.

### 6.2 Reductions

`rop(j₁,…,jₖ: e)` folds `e` over the Cartesian product of the bound
indices' extents with the operator's fold and **identity**:

| op     | fold        | identity (float / int)   |
|--------|-------------|--------------------------|
| `sum`  | `acc + v`   | `0`                      |
| `prod` | `acc * v`   | `1`                      |
| `max`  | `v > acc ? v : acc` | `-∞` / `T::MIN`  |
| `min`  | `v < acc ? v : acc` | `+∞` / `T::MAX`  |

A reduction over an empty range yields the identity. The fold **order is
unspecified** (open decision E1 governs reproducibility guarantees).

### 6.3 Numerics

Arithmetic follows IEEE-754 for the element type: `1/0 = ∞`, `sqrt(-1) =
NaN`, `ln(0) = -∞`, etc. — no language-level domain checks. `max`/`min`
(both forms) are comparison-based as written above; their behavior when an
operand is NaN is **unspecified** (consistent with the current semiring
implementation; open decision A3 revisits this).

### 6.4 Sparse tensors

**Sparsity is a storage property, not a semantic one.** A structurally
missing entry *is* the value `0`, and every program must produce results
identical to the same program run on densified inputs. Skipping stored-zero
iteration is purely an optimization, legal exactly where analysis proves the
skipped work contributes nothing: a missing operand annihilates its
contribution under `(sum, *)`; zero-preserving unaries (`sqrt`, `relu`,
`abs`, `-`) keep that property; non-zero-preserving ones (`exp(0)=1`)
densify. This generalizes the current per-program `skip_missing` boolean
into a per-subexpression analysis, but the *contract* is fixed here:
dense-equivalence, always.

## 7. S-expression encoding

The AST has a canonical s-expr form — the machine-facing syntax (programs
as data, generation and rewriting in expression space). The infix form and
this encoding are isomorphic; two readers, one AST.

```lisp
(= (y j k) (* (@ q k)
              (min (l) (@ x l))
              (sqrt (sum (l) (@ M l j)))))

(= (softmax i) (/ (exp (@ v i))
                  (sum (j) (exp (@ v j)))))

(= m (max (j) (@ v j)))          ; scalar LHS: bare name
```

- `(@ tensor idx…)` — tensor reference. The explicit `@` keeps heads
  context-free (no symbol table needed to distinguish indexing from calls),
  playing the same role the `[…]`/`(…)` split plays in the infix form.
- `(sum (idx…) body)` — reduction; the binder list is a parenthesized list,
  always present, even when singleton. This is the one deliberate departure
  from ad-hoc forms like `(sum M l j)`: reductions bind indices over
  arbitrary subexpressions, so the bound set must be written, not inferred
  from an LHS.
- `(= lhs rhs)` — statement; a program is a sequence of `=` forms.
- Comments: `;` to end of line.

## 8. Compatibility desugaring

The legacy APIs become mechanical rewrites:

- `einsum("ab,bc->ac", …)` →
  `out[a,c] = sum(b: in0[a,b] * in1[b,c])`
  where the bound set is (indices in inputs) ∖ (indices in output).
- `einsum_reduce(spec, Reduce::Max, Combine::Add, …)` →
  the same shape with `max(…: … + …)`.
- Multi-output specs (`"ab,bc->ac,ca"`) desugar to one statement per output.
  Whether they still execute in one pass is the planner's fusion decision,
  not a language construct (open decision D8).

`Combine` disappears entirely — operand merging is just the expression tree.

## 9. Settled decisions

| # | Decision | Rationale |
|---|----------|-----------|
| S1 | Reductions are **explicit binders** (`sum(j: …)`); no implicit summation | Reductions apply to expressions, not tensors, so there's no index list to consult; nesting makes "absent from LHS" ambiguous; unbound-index errors catch real bugs |
| S2 | `t[i]` = indexing, `f(x)` = call; s-expr uses `@` for references | Parsing needs no symbol table; grammar stays LL(1)-ish |
| S3 | The colon distinguishes a reduction binder from an elementwise call; binder heads are validated against the reduction registry (built-ins `sum/prod/max/min`) | Crisp parse, crisp errors; `max`/`min` still available elementwise; registry-extensible without grammar changes |
| S4 | `=` is overwrite; output fully defined by its statement | Kills the identity-prefill vs caller-zeroed split (and the `AccFlush` bug class it produced) |
| S5 | No shadowing of in-scope indices | Buys nothing; silently wrong contractions when names collide |
| S6 | Bare identifier = scalar (0-dim); no special scalar syntax | Uniform: scalars are just rank-0 tensors |
| S7 | `indexref` is a grammar nonterminal even though v1 allows only bare identifiers | Shifts/strides (`a[i+1]`, `a[2*i]` — convolutions) become a non-breaking extension |
| S8 | Sparse storage is semantically invisible (§6.4) | Preserves the differential-testing methodology: dense reference is always the oracle |
| S9 | Programs are statement sequences; intermediates are ordinary named tensors | Sharing (stable softmax's `m`) is user-visible and CSE-friendly; no `where`-clause sublanguage |

## 10. Resolved decisions (2026-07-23)

Previously open; resolved with the project owner. **A** = language
semantics, **D** = program/API structure, **E** = execution model.

### A1. Element type policy — **monomorphic**
One scalar type per program (parameterized `T`). No promotion rules; mixed
types can come later via explicit `cast` functions without touching the
grammar.

### A2. Scalar-function extensibility — **static, in code**
Functions live in a registry extended statically (in code, at program-build
time) — no dynamic plugin surface. A registered function must supply *all*
its semantics up front: evaluation, zero-preservation flag (§6.4), and —
once the JIT backend consumes the registry — its JIT emission. No
registration without JIT semantics once that backend exists.

### A3. NaN in `max`/`min` — **status quo**
Unspecified-with-documented-implementation (`v > acc` comparison
semantics). Revisit only if a real consumer needs a guarantee.

### A4. User-defined reductions — **extensible in code**
Reductions live in the same registry: name + identity + fold, extended
statically like A2 (fold must come with JIT semantics once applicable;
associativity/commutativity is a documented contract on the registrant).
The binder grammar takes any registered name — no grammar change needed.

### D5. Broadcast-only LHS indices — **unified extent rule**
No special case: an index's extent must be equal across *all* its use
sites, and one site suffices to determine it. The bound output tensor is a
use site like any other, so `y[i,j] = v[i]` gets `j`'s extent from `y`'s
declared shape. An index with zero shape-bearing use sites is an error
(`UnresolvedExtent`); disagreeing sites are an error (`ExtentMismatch`).

### D6. Scalar parameters — **runtime-bound**
Environment scalars (`eps`, `n`) are 0-dim inputs bound at run time (one
pointer slot in the JIT ABI), never baked constants. Source *literals* are
baked.

### D7. Name rebinding — **SSA in the IR**
Each name is defined at most once in the core IR. Reassignment
(`x = relu(x)` chains) may later be added as front-end sugar that
alpha-renames; it never reaches the IR.

### D8. Program outputs — **explicit output set**
Declared at bind time; every other statement LHS is a temp the planner
owns (fuse away, free early). Multi-output einsum is subsumed: request
several outputs, fusion decides whether one pass serves them.

### E1. Reduction order — **deterministic per compiled artifact**
Same program + shapes + backend → same bits, run to run. Reassociation is
permitted only as a compile-time choice. (The exact-integer test
methodology is order-immune, so differential sweeps stay bit-exact
regardless.)

### E2. Shape specialization — **hybrid; separate task**
Per-axis `dynamic` marking at bind time: dynamic axes become runtime args,
static axes keep baked-constant codegen behind a shape-keyed compile cache.
The AST/plan is always shape-polymorphic; only codegen specializes. Not
part of v1 — scheduled as its own task.

### E3. v1 execution — **materialize reductions into temps**
Simplest-first: every reduction node materializes a temp (computed once at
its own index arity — this is what makes softmax O(n) by construction),
executed as a DAG over the existing machinery. Fused multi-accumulator
codegen comes later as a pure optimization behind identical semantics;
likely intermediate step: per-input load-maps and an output store-map on
the existing kernels to eliminate most temps.

### E4. Sparse union iteration — **TODO, out of v1**
`sparse + sparse` wants union co-iteration of row streams; v1 uses the
dense fallback (correct by S8). **TODO:** design the loop-IR iterator
interface so a union merge node slots in without reshaping the IR, then
implement union co-iteration.

## 11. Future extensions (explicitly out of v1)

- Accumulating assignments (`+=`, `max=`) — Tullio-style; a program can
  always express the same with an extra statement.
- Affine index expressions (`a[i+1]`, `a[2*i]`, `a[i+j]`) — convolution
  territory; grammar slot reserved (S7).
- `cast`/mixed element types (A1), user scalar functions (A2), user
  semirings (A4).
- Masked/conditional evaluation (`where`-style selection) — interacts with
  sparsity analysis; needs its own design pass.
