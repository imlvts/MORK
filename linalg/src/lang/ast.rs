//! Surface AST — the common target of both front-ends.
//!
//! Names are unresolved strings here; [`super::check`] alpha-renames
//! indices to ids and validates functions/reductions against the
//! [`Registry`](super::Registry), and binding to actual tensors happens
//! in [`super::run`].

/// A whole program: an ordered sequence of statements. Later statements
/// may reference the names earlier ones define (SSA — each name at most
/// once).
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub stmts: Vec<Stmt>,
}

/// One statement: `name[indices…] = rhs` (empty `indices` = scalar LHS).
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub name: String,
    pub indices: Vec<String>,
    pub rhs: Expr,
}

/// Binary arithmetic operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric literal (converted to the element type at run time via the
    /// registry's literal function).
    Num(f64),
    /// Bare identifier — reference to a 0-dim (scalar) tensor.
    Scalar(String),
    /// `t[i,j]` / `(@ t i j)` — tensor reference.
    Tensor { name: String, indices: Vec<String> },
    /// `f(a, b, …)` — scalar function call.
    Call { func: String, args: Vec<Expr> },
    /// `l op r`.
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    /// `op(i,j: body)` / `(op (i j) body)` — reduction binder.
    Reduce { op: String, indices: Vec<String>, body: Box<Expr> },
}

impl Expr {
    /// Helper for the parsers: fold a binary op.
    pub(crate) fn binary(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    /// Helper for the parsers: unary minus. Folds literals; otherwise
    /// desugars to the built-in `neg` function.
    pub(crate) fn negate(e: Expr) -> Expr {
        match e {
            Expr::Num(x) => Expr::Num(-x),
            other => Expr::Call { func: "neg".to_string(), args: vec![other] },
        }
    }
}
