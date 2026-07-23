//! The tensor-expression language: pointwise arithmetic, scalar functions,
//! and explicitly-bound reductions over [`NDIndex`](crate::tensor::NDIndex)
//! tensors. The full design (grammar, semantics, decision log) lives in
//! `linalg/LANGUAGE.md`.
//!
//! A **program** is a sequence of statements, each defining a tensor (or
//! scalar) from an expression. Reductions are *binders*: `sum(j: …)` binds
//! `j` over an arbitrary subexpression — there is no implicit "unbound
//! means summed" rule. Two isomorphic front-ends produce the same
//! [`Program`]: the human-facing infix form ([`parse`]) and an
//! s-expression encoding ([`parse_sexpr`]).
//!
//! Pipeline: [`parse`] / [`parse_sexpr`] → [`check`] (scope + registry
//! validation, alpha-renaming) → [`run`] (bind names/extents against an
//! environment, then execute). Element type is monomorphic per program;
//! scalar functions and reduction operators live in a [`Registry`]
//! extended statically in code.
//!
//! Execution is v1-simple by design: every reduction node materializes a
//! temporary computed once at its own index arity — which is what makes
//! e.g. a softmax denominator O(n) instead of O(n²) — and everything else
//! is evaluated elementwise. Kernel/JIT-backed execution slots in behind
//! the same semantics later.
//!
//! ```
//! use linalg::dense::Dense;
//! use linalg::lang::{parse, check, run, Registry};
//!
//! let src = "
//! m       = max(j: v[j])
//! soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))
//! ";
//! let reg = Registry::<f32>::builtins();
//! let checked = check(&parse(src).unwrap(), &reg).unwrap();
//!
//! let mut v = Dense::<f32>::zeros(vec![3]);
//! v.fill_from(&[1.0, 2.0, 3.0]);
//! let mut soft = Dense::<f32>::zeros(vec![3]);
//!
//! run(&checked, &reg, &[("v", &v)], &mut [("soft", &mut soft)]).unwrap();
//!
//! let total: f32 = soft.data.iter().sum();
//! assert!((total - 1.0).abs() < 1e-6);
//! ```

use std::fmt;

use crate::tensor::Scalar;

pub mod ast;
mod check;
mod eval;
mod fast;
mod parse;
mod sexpr;

pub use ast::{BinOp, Expr, Program, Stmt};
pub use check::{Checked, check};
pub use eval::{RunReport, run, run_reference, run_reported};
pub use parse::parse;
pub use sexpr::parse_sexpr;

// ─────────────────────────────────────────────────────────────────────────
// Element trait
// ─────────────────────────────────────────────────────────────────────────

/// Element type usable by the expression language.
///
/// Extends the einsum [`Scalar`] semiring bounds with the full arithmetic
/// the language exposes (`-`, `/`, unary negation). Blanket-implemented
/// for every qualifying type — floats and signed integers out of the box;
/// unsigned integers don't qualify (no `Neg`).
///
/// The `'static` bound lets the executor recognise `f32` at run time (via
/// [`std::any::TypeId`]) and route eligible reductions to the JIT backend;
/// every primitive numeric type satisfies it.
pub trait Elem:
    Scalar
    + std::ops::Sub<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + 'static
{
}

impl<T> Elem for T where
    T: Scalar
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + 'static
{
}

// ─────────────────────────────────────────────────────────────────────────
// Registry: scalar functions and reduction operators
// ─────────────────────────────────────────────────────────────────────────

/// Which built-in scalar function a registry entry *is*.
///
/// Recorded out-of-band in [`Registry::fn_kind`] rather than on
/// [`FnDef`], for the same reason [`Registry::reduce_kind`] is: `FnDef`'s
/// fields are public and callers build it with a struct literal, so an
/// extra field would be a breaking change — and a *user*-registered
/// function must never be mistaken for a built-in just because it shares
/// a name or an implementation.
///
/// Backends use it to replace an indirect `fn(&[T]) -> T` call per
/// element with inline, monomorphic, autovectorizable code. Every kind's
/// inline form is required to be bit-identical to the [`FnDef::eval`] it
/// stands for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FnKind {
    /// `-x` (registered for every element type).
    Neg,
    /// `max(a, b)` — the comparison form `if b > a { b } else { a }`.
    Max2,
    /// `min(a, b)` — `if b < a { b } else { a }`.
    Min2,
    Exp,
    Ln,
    Sqrt,
    Abs,
    /// `if x > 0 { x } else { 0 }` — note `relu(-0.0) == +0.0`.
    Relu,
    Tanh,
}

impl FnKind {
    /// Arity of the function this kind stands for.
    pub(crate) fn arity(self) -> usize {
        match self {
            FnKind::Max2 | FnKind::Min2 => 2,
            _ => 1,
        }
    }
}

/// A scalar function usable in call position: `exp(x)`, `max(a, b)`, …
///
/// Registered statically in code via [`Registry::register_fn`]. A
/// registered function must supply *all* of its semantics: evaluation,
/// the zero-preservation flag (drives sparse-iteration legality — see
/// LANGUAGE.md §6.4), and, once the JIT backend consumes the registry,
/// its JIT emission.
pub struct FnDef<T> {
    pub name: &'static str,
    pub arity: usize,
    /// Evaluate on `arity` arguments. Called with exactly `arity` values.
    pub eval: fn(&[T]) -> T,
    /// `f(0, …, 0) == 0` — whether the function maps structural zeros to
    /// zero (e.g. `sqrt`, `relu`: yes; `exp`: no, `exp(0) = 1`).
    pub zero_preserving: bool,
}

/// A reduction operator usable in binder position: `sum(j: …)`, …
///
/// Registered statically in code via [`Registry::register_reduce`]. The
/// fold must be associative and commutative with `identity` as its
/// identity element — that contract is on the registrant; the evaluator
/// is free to pick any deterministic iteration order.
pub struct ReduceDef<T> {
    pub name: &'static str,
    pub identity: T,
    pub fold: fn(T, T) -> T,
}

/// The set of scalar functions and reduction operators a program may use.
///
/// One registry serves both [`check`] (name/arity validation) and [`run`]
/// (evaluation); pass the *same* registry to both. [`Registry::new`]
/// seeds the four built-in reductions (`sum`, `prod`, `max`, `min`) and
/// the type-generic functions (`neg`, elementwise `max`/`min`);
/// [`Registry::<f32>::builtins`] (and `f64`) adds the float function
/// table (`exp`, `ln`, `sqrt`, `abs`, `relu`, `tanh`).
pub struct Registry<T> {
    pub(crate) lit: fn(f64) -> T,
    pub(crate) fns: Vec<FnDef<T>>,
    /// Parallel to `fns`: which built-in scalar function an entry *is*, so
    /// a backend can inline it instead of calling through
    /// [`FnDef::eval`]. Only the constructors below ever set a `Some`, so
    /// a user-registered function stays `None` and keeps the portable
    /// indirect-call path. See [`FnKind`].
    pub(crate) fn_kind: Vec<Option<FnKind>>,
    pub(crate) reduces: Vec<ReduceDef<T>>,
    /// Parallel to `reduces`: which built-in semiring operator an entry
    /// *is*, for backends that speak [`crate::einsum::Reduce`] rather than
    /// a fold pointer. Only [`Registry::new`] ever sets a `Some`, so a
    /// user-registered reduction — even one named `"sum"` with an
    /// identical fold — stays `None` and keeps the portable path.
    pub(crate) reduce_kind: Vec<Option<crate::einsum::Reduce>>,
}

fn fold_sum<T: Elem>(a: T, b: T) -> T {
    a + b
}
fn fold_prod<T: Elem>(a: T, b: T) -> T {
    a * b
}
fn fold_max<T: Elem>(a: T, b: T) -> T {
    if b > a { b } else { a }
}
fn fold_min<T: Elem>(a: T, b: T) -> T {
    if b < a { b } else { a }
}
fn fn_neg<T: Elem>(args: &[T]) -> T {
    -args[0]
}
fn fn_max2<T: Elem>(args: &[T]) -> T {
    if args[1] > args[0] { args[1] } else { args[0] }
}
fn fn_min2<T: Elem>(args: &[T]) -> T {
    if args[1] < args[0] { args[1] } else { args[0] }
}

impl<T: Elem> Registry<T> {
    /// A registry with the built-in reductions and type-generic functions,
    /// converting literals via `lit`.
    pub fn new(lit: fn(f64) -> T) -> Self {
        use crate::einsum::Reduce;
        let mut r = Registry {
            lit,
            fns: Vec::new(),
            fn_kind: Vec::new(),
            reduces: Vec::new(),
            reduce_kind: Vec::new(),
        };
        r.register_reduce(ReduceDef { name: "sum", identity: T::ZERO, fold: fold_sum::<T> });
        r.register_reduce(ReduceDef { name: "prod", identity: T::ONE, fold: fold_prod::<T> });
        r.register_reduce(ReduceDef { name: "max", identity: T::LEAST, fold: fold_max::<T> });
        r.register_reduce(ReduceDef { name: "min", identity: T::GREATEST, fold: fold_min::<T> });
        // The four just-registered entries occupy ids 0..4, in this order,
        // with exactly `Reduce::identity`'s identities.
        r.reduce_kind =
            vec![Some(Reduce::Sum), Some(Reduce::Prod), Some(Reduce::Max), Some(Reduce::Min)];
        r.register_builtin_fn(
            FnDef { name: "neg", arity: 1, eval: fn_neg::<T>, zero_preserving: true },
            FnKind::Neg,
        );
        r.register_builtin_fn(
            FnDef { name: "max", arity: 2, eval: fn_max2::<T>, zero_preserving: true },
            FnKind::Max2,
        );
        r.register_builtin_fn(
            FnDef { name: "min", arity: 2, eval: fn_min2::<T>, zero_preserving: true },
            FnKind::Min2,
        );
        r
    }

    /// [`register_fn`](Self::register_fn) for a function whose semantics
    /// this crate itself defines, tagging it so backends may inline it.
    /// Private on purpose: a caller-supplied function is never a built-in.
    fn register_builtin_fn(&mut self, def: FnDef<T>, kind: FnKind) {
        assert_eq!(def.arity, kind.arity(), "built-in {:?} arity disagrees with its kind", def.name);
        self.register_fn(def);
        *self.fn_kind.last_mut().expect("just registered") = Some(kind);
    }

    /// Register a scalar function. Panics on a duplicate name — the
    /// registry is assembled statically, so a collision is a programming
    /// error, not a runtime condition.
    pub fn register_fn(&mut self, def: FnDef<T>) {
        assert!(
            self.fns.iter().all(|f| f.name != def.name),
            "function {:?} already registered",
            def.name
        );
        self.fns.push(def);
        self.fn_kind.push(None);
    }

    /// Register a reduction operator. Panics on a duplicate name.
    pub fn register_reduce(&mut self, def: ReduceDef<T>) {
        assert!(
            self.reduces.iter().all(|r| r.name != def.name),
            "reduction {:?} already registered",
            def.name
        );
        self.reduces.push(def);
        self.reduce_kind.push(None);
    }
}

impl<T> Registry<T> {
    pub(crate) fn fn_id(&self, name: &str) -> Option<usize> {
        self.fns.iter().position(|f| f.name == name)
    }

    pub(crate) fn reduce_id(&self, name: &str) -> Option<usize> {
        self.reduces.iter().position(|r| r.name == name)
    }

    /// Which built-in semiring operator reduction `id` is, if any. See
    /// [`Registry::reduce_kind`].
    pub(crate) fn builtin_reduce(&self, id: usize) -> Option<crate::einsum::Reduce> {
        self.reduce_kind[id]
    }

    /// Which built-in scalar function `id` is, if any. See [`FnKind`].
    pub(crate) fn builtin_fn(&self, id: usize) -> Option<FnKind> {
        self.fn_kind[id]
    }
}

macro_rules! float_builtins {
    ($t:ty) => {
        impl Registry<$t> {
            /// [`Registry::new`] plus the float function table:
            /// `exp`, `ln`, `sqrt`, `abs`, `relu`, `tanh`.
            pub fn builtins() -> Self {
                let mut r = Self::new(|x| x as $t);
                r.register_builtin_fn(
                    FnDef { name: "exp", arity: 1, eval: |a| a[0].exp(), zero_preserving: false },
                    FnKind::Exp,
                );
                r.register_builtin_fn(
                    FnDef { name: "ln", arity: 1, eval: |a| a[0].ln(), zero_preserving: false },
                    FnKind::Ln,
                );
                r.register_builtin_fn(
                    FnDef { name: "sqrt", arity: 1, eval: |a| a[0].sqrt(), zero_preserving: true },
                    FnKind::Sqrt,
                );
                r.register_builtin_fn(
                    FnDef { name: "abs", arity: 1, eval: |a| a[0].abs(), zero_preserving: true },
                    FnKind::Abs,
                );
                r.register_builtin_fn(
                    FnDef {
                        name: "relu",
                        arity: 1,
                        eval: |a| if a[0] > 0.0 { a[0] } else { 0.0 },
                        zero_preserving: true,
                    },
                    FnKind::Relu,
                );
                r.register_builtin_fn(
                    FnDef { name: "tanh", arity: 1, eval: |a| a[0].tanh(), zero_preserving: true },
                    FnKind::Tanh,
                );
                r
            }
        }
    };
}
float_builtins!(f32);
float_builtins!(f64);

// ─────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────

/// Any error from parsing, checking, or binding/running a program.
#[derive(Debug, Clone, PartialEq)]
pub enum LangError {
    /// Syntax error in either front-end. `line`/`col` are 1-based.
    Parse { line: usize, col: usize, msg: String },
    /// Call to a function the registry doesn't know.
    UnknownFunction { name: String },
    /// Binder head that isn't a registered reduction.
    UnknownReduction { name: String },
    /// Call with the wrong number of arguments.
    FunctionArity { name: String, expected: usize, got: usize },
    /// A registered reduction used in call position (`sum(a, b)`).
    ReductionNeedsColon { name: String },
    /// RHS index not bound by the LHS or any enclosing reduction.
    UnboundIndex { index: String, stmt: String },
    /// The same index appears twice in one LHS or one binder list.
    DuplicateIndex { index: String, stmt: String },
    /// A binder rebinds an index already in scope.
    ShadowedBinder { index: String, stmt: String },
    /// An in-scope index used as a scalar value.
    IndexAsScalar { index: String, stmt: String },
    /// A name defined more than once (statement LHS, input, or output).
    RedefinedName { name: String },
    /// RHS reference to a name that is neither an input nor a previously
    /// defined statement.
    UnknownName { name: String, stmt: String },
    /// A tensor referenced with the wrong number of indices.
    RankMismatch { name: String, expected: usize, got: usize },
    /// An index whose use sites disagree about its extent.
    ExtentMismatch { index: String, expected: usize, got: usize, site: String },
    /// An index with no shape-bearing use site to take its extent from.
    UnresolvedExtent { index: String, stmt: String },
    /// A requested output that no statement defines.
    MissingOutput { name: String },
}

impl fmt::Display for LangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { line, col, msg } => write!(f, "parse error at {line}:{col}: {msg}"),
            Self::UnknownFunction { name } => write!(f, "unknown function '{name}'"),
            Self::UnknownReduction { name } => write!(f, "unknown reduction '{name}'"),
            Self::FunctionArity { name, expected, got } => {
                write!(f, "function '{name}' takes {expected} argument(s), got {got}")
            }
            Self::ReductionNeedsColon { name } => write!(
                f,
                "'{name}' is a reduction — bind indices with a colon: {name}(j: …)"
            ),
            Self::UnboundIndex { index, stmt } => write!(
                f,
                "index '{index}' in statement '{stmt}' is not bound by the LHS or any \
                 enclosing reduction (did you mean sum({index}: …)?)"
            ),
            Self::DuplicateIndex { index, stmt } => {
                write!(f, "index '{index}' appears twice in one binding list of '{stmt}'")
            }
            Self::ShadowedBinder { index, stmt } => write!(
                f,
                "binder in statement '{stmt}' rebinds index '{index}' already in scope"
            ),
            Self::IndexAsScalar { index, stmt } => write!(
                f,
                "index '{index}' used as a scalar value in statement '{stmt}'"
            ),
            Self::RedefinedName { name } => write!(f, "name '{name}' defined more than once"),
            Self::UnknownName { name, stmt } => write!(
                f,
                "statement '{stmt}' references '{name}', which is neither an input nor a \
                 previously defined statement"
            ),
            Self::RankMismatch { name, expected, got } => {
                write!(f, "'{name}' has rank {expected} but is referenced with {got} index(es)")
            }
            Self::ExtentMismatch { index, expected, got, site } => write!(
                f,
                "extent mismatch for index '{index}': {expected} elsewhere vs {got} at '{site}'"
            ),
            Self::UnresolvedExtent { index, stmt } => write!(
                f,
                "index '{index}' in statement '{stmt}' has no use site to take its extent from"
            ),
            Self::MissingOutput { name } => {
                write!(f, "requested output '{name}' is not defined by any statement")
            }
        }
    }
}

impl std::error::Error for LangError {}
