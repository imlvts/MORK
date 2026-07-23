//! Scope checking and resolution.
//!
//! Validates a [`Program`] against a [`Registry`] and produces the
//! resolved form the evaluator consumes: every index binding site gets a
//! unique id (alpha-renaming — sibling binders reusing a name become
//! distinct indices), functions and reductions become registry ids.
//!
//! Enforced here (LANGUAGE.md §5): every RHS index is bound by exactly one
//! of the LHS or an enclosing binder; no shadowing; no duplicate names in
//! a binding list; call names/arities exist in the registry; each
//! statement name is defined at most once. Name-to-tensor resolution and
//! extent checking need the environment, so they happen in
//! [`run`](super::run).

use super::ast::{BinOp, Expr, Program, Stmt};
use super::{LangError, Registry};

/// Index binding sites are numbered program-wide; `idx` values below are
/// positions in [`Checked::index_names`].
pub(crate) type IndexId = u32;

#[derive(Debug)]
pub(crate) enum CExpr {
    Num(f64),
    /// Reference to a 0-dim tensor (input or earlier statement).
    ScalarRef(String),
    Load {
        tensor: String,
        indices: Vec<IndexId>,
    },
    Call {
        func: usize,
        args: Vec<CExpr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<CExpr>,
        rhs: Box<CExpr>,
    },
    Reduce {
        op: usize,
        indices: Vec<IndexId>,
        body: Box<CExpr>,
    },
}

#[derive(Debug)]
pub(crate) struct CStmt {
    pub name: String,
    pub lhs: Vec<IndexId>,
    pub rhs: CExpr,
}

/// A scope-checked program, ready for [`run`](super::run) with the same
/// registry it was checked against.
#[derive(Debug)]
pub struct Checked {
    pub(crate) stmts: Vec<CStmt>,
    /// Original name of each index id (error messages).
    pub(crate) index_names: Vec<String>,
}

struct Ctx<'r, T> {
    reg: &'r Registry<T>,
    index_names: Vec<String>,
    /// In-scope bindings, innermost last.
    scope: Vec<(String, IndexId)>,
    stmt_name: String,
}

impl<T> Ctx<'_, T> {
    fn lookup(&self, name: &str) -> Option<IndexId> {
        self.scope.iter().rev().find(|(n, _)| n == name).map(|&(_, id)| id)
    }

    fn bind(&mut self, name: &str) -> IndexId {
        let id = self.index_names.len() as IndexId;
        self.index_names.push(name.to_string());
        self.scope.push((name.to_string(), id));
        id
    }

    /// Bind a whole list (an LHS or a binder list): rejects duplicates
    /// within the list, and — for binders — shadowing of enclosing scope.
    fn bind_list(
        &mut self,
        names: &[String],
        is_binder: bool,
    ) -> Result<Vec<IndexId>, LangError> {
        let mut ids = Vec::with_capacity(names.len());
        for (k, name) in names.iter().enumerate() {
            if names[..k].contains(name) {
                return Err(LangError::DuplicateIndex {
                    index: name.clone(),
                    stmt: self.stmt_name.clone(),
                });
            }
            if is_binder && self.lookup(name).is_some() {
                return Err(LangError::ShadowedBinder {
                    index: name.clone(),
                    stmt: self.stmt_name.clone(),
                });
            }
            ids.push(self.bind(name));
        }
        Ok(ids)
    }

    fn check_expr(&mut self, e: &Expr) -> Result<CExpr, LangError> {
        match e {
            Expr::Num(v) => Ok(CExpr::Num(*v)),
            Expr::Scalar(name) => {
                if self.lookup(name).is_some() {
                    return Err(LangError::IndexAsScalar {
                        index: name.clone(),
                        stmt: self.stmt_name.clone(),
                    });
                }
                Ok(CExpr::ScalarRef(name.clone()))
            }
            Expr::Tensor { name, indices } => {
                let mut ids = Vec::with_capacity(indices.len());
                for ix in indices {
                    match self.lookup(ix) {
                        Some(id) => ids.push(id),
                        None => {
                            return Err(LangError::UnboundIndex {
                                index: ix.clone(),
                                stmt: self.stmt_name.clone(),
                            });
                        }
                    }
                }
                Ok(CExpr::Load { tensor: name.clone(), indices: ids })
            }
            Expr::Call { func, args } => {
                let Some(id) = self.reg.fn_id(func) else {
                    return Err(if self.reg.reduce_id(func).is_some() {
                        LangError::ReductionNeedsColon { name: func.clone() }
                    } else {
                        LangError::UnknownFunction { name: func.clone() }
                    });
                };
                let expected = self.reg.fns[id].arity;
                if args.len() != expected {
                    return Err(LangError::FunctionArity {
                        name: func.clone(),
                        expected,
                        got: args.len(),
                    });
                }
                let args =
                    args.iter().map(|a| self.check_expr(a)).collect::<Result<Vec<_>, _>>()?;
                Ok(CExpr::Call { func: id, args })
            }
            Expr::Binary { op, lhs, rhs } => Ok(CExpr::Binary {
                op: *op,
                lhs: Box::new(self.check_expr(lhs)?),
                rhs: Box::new(self.check_expr(rhs)?),
            }),
            Expr::Reduce { op, indices, body } => {
                let Some(id) = self.reg.reduce_id(op) else {
                    return Err(LangError::UnknownReduction { name: op.clone() });
                };
                let depth = self.scope.len();
                let ids = self.bind_list(indices, true)?;
                let body = self.check_expr(body)?;
                self.scope.truncate(depth);
                Ok(CExpr::Reduce { op: id, indices: ids, body: Box::new(body) })
            }
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<CStmt, LangError> {
        self.stmt_name = stmt.name.clone();
        debug_assert!(self.scope.is_empty());
        let lhs = self.bind_list(&stmt.indices, false)?;
        let rhs = self.check_expr(&stmt.rhs)?;
        self.scope.clear();
        Ok(CStmt { name: stmt.name.clone(), lhs, rhs })
    }
}

/// Scope-check `prog` against `reg`, resolving indices to ids and
/// functions/reductions to registry entries. Pass the *same* registry to
/// [`run`](super::run).
pub fn check<T>(prog: &Program, reg: &Registry<T>) -> Result<Checked, LangError> {
    let mut ctx =
        Ctx { reg, index_names: Vec::new(), scope: Vec::new(), stmt_name: String::new() };
    let mut stmts = Vec::with_capacity(prog.stmts.len());
    for (k, stmt) in prog.stmts.iter().enumerate() {
        if prog.stmts[..k].iter().any(|s| s.name == stmt.name) {
            return Err(LangError::RedefinedName { name: stmt.name.clone() });
        }
        stmts.push(ctx.check_stmt(stmt)?);
    }
    Ok(Checked { stmts, index_names: ctx.index_names })
}

#[cfg(test)]
mod tests {
    use super::super::{Registry, parse};
    use super::*;

    fn reg() -> Registry<f32> {
        Registry::<f32>::builtins()
    }

    fn check_src(src: &str) -> Result<Checked, LangError> {
        check(&parse(src).unwrap(), &reg())
    }

    #[test]
    fn sibling_binders_may_reuse_a_name() {
        let c = check_src("y[j] = sum(l: M[l,j]) * min(l: x[l])").unwrap();
        // j, l (sum), l (min) — three distinct ids.
        assert_eq!(c.index_names, vec!["j", "l", "l"]);
    }

    #[test]
    fn unbound_index() {
        assert!(matches!(
            check_src("y[i] = x[j]"),
            Err(LangError::UnboundIndex { index, .. }) if index == "j"
        ));
    }

    #[test]
    fn shadowing_rejected() {
        assert!(matches!(
            check_src("y[i] = sum(i: x[i])"),
            Err(LangError::ShadowedBinder { index, .. }) if index == "i"
        ));
        assert!(matches!(
            check_src("s = sum(l: prod(l: x[l]))"),
            Err(LangError::ShadowedBinder { .. })
        ));
    }

    #[test]
    fn duplicate_in_binding_lists() {
        assert!(matches!(check_src("y[i,i] = x[i]"), Err(LangError::DuplicateIndex { .. })));
        assert!(matches!(
            check_src("s = sum(j,j: M[j,j])"),
            Err(LangError::DuplicateIndex { .. })
        ));
    }

    #[test]
    fn registry_validation() {
        assert!(matches!(
            check_src("y[i] = frobnicate(x[i])"),
            Err(LangError::UnknownFunction { name }) if name == "frobnicate"
        ));
        assert!(matches!(
            check_src("s = sum(a, b)"),
            Err(LangError::ReductionNeedsColon { name }) if name == "sum"
        ));
        assert!(matches!(
            check_src("s = median(j: x[j])"),
            Err(LangError::UnknownReduction { name }) if name == "median"
        ));
        assert!(matches!(
            check_src("y[i] = exp(x[i], x[i])"),
            Err(LangError::FunctionArity { expected: 1, got: 2, .. })
        ));
    }

    #[test]
    fn index_as_scalar_and_ssa() {
        assert!(matches!(
            check_src("y[i] = x[i] + i"),
            Err(LangError::IndexAsScalar { index, .. }) if index == "i"
        ));
        assert!(matches!(
            check_src("y[i] = x[i]\ny[i] = x[i]"),
            Err(LangError::RedefinedName { name }) if name == "y"
        ));
    }

    #[test]
    fn diagonal_read_is_legal() {
        assert!(check_src("t = sum(i: M[i,i])").is_ok());
    }
}
