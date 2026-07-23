//! S-expression front-end — the machine-facing encoding of the AST.
//!
//! A program is a sequence of `(= lhs rhs)` forms:
//!
//! ```lisp
//! (= (y j k) (* (@ q k)
//!               (min (l) (@ x l))
//!               (sqrt (sum (l) (@ M l j)))))
//! (= m (max (j) (@ v j)))          ; scalar LHS: bare name
//! ```
//!
//! - `(@ t i j)` — tensor reference (the explicit `@` keeps heads
//!   context-free; bare symbols are scalar references).
//! - `(op (i j) body)` — reduction binder; the bound list is always a
//!   parenthesized list of bare symbols, even when singleton.
//! - `+` and `*` are n-ary (left-folded); `-` and `/` are binary or,
//!   for `-`, unary. Anything else in head position is a function call.
//! - `;` comments to end of line.

use super::LangError;
use super::ast::{BinOp, Expr, Program, Stmt};

// ─────────────────────────────────────────────────────────────────────────
// Reader
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Pos {
    line: usize,
    col: usize,
}

#[derive(Debug)]
enum SExp {
    Sym(String, Pos),
    Num(f64, Pos),
    List(Vec<SExp>, Pos),
}

impl SExp {
    fn pos(&self) -> Pos {
        match self {
            SExp::Sym(_, p) | SExp::Num(_, p) | SExp::List(_, p) => *p,
        }
    }
}

fn err<T>(pos: Pos, msg: impl Into<String>) -> Result<T, LangError> {
    Err(LangError::Parse { line: pos.line, col: pos.col, msg: msg.into() })
}

struct Reader<'a> {
    src: &'a [u8],
    at: usize,
    line: usize,
    col: usize,
}

impl<'a> Reader<'a> {
    fn new(src: &'a str) -> Self {
        Reader { src: src.as_bytes(), at: 0, line: 1, col: 1 }
    }

    fn pos(&self) -> Pos {
        Pos { line: self.line, col: self.col }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.at).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.at += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b';' {
                while let Some(c) = self.peek() {
                    if c == b'\n' {
                        break;
                    }
                    self.bump();
                }
            } else if c.is_ascii_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn read_all(&mut self) -> Result<Vec<SExp>, LangError> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.peek().is_none() {
                return Ok(out);
            }
            out.push(self.read_one()?);
        }
    }

    fn read_one(&mut self) -> Result<SExp, LangError> {
        self.skip_ws();
        let pos = self.pos();
        match self.peek() {
            None => err(pos, "unexpected end of input"),
            Some(b'(') => {
                self.bump();
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek() {
                        None => return err(pos, "unclosed '('"),
                        Some(b')') => {
                            self.bump();
                            return Ok(SExp::List(items, pos));
                        }
                        Some(_) => items.push(self.read_one()?),
                    }
                }
            }
            Some(b')') => err(pos, "unexpected ')'"),
            Some(_) => {
                let start = self.at;
                while let Some(c) = self.peek() {
                    if c.is_ascii_whitespace() || c == b'(' || c == b')' || c == b';' {
                        break;
                    }
                    self.bump();
                }
                let atom = std::str::from_utf8(&self.src[start..self.at])
                    .map_err(|_| LangError::Parse {
                        line: pos.line,
                        col: pos.col,
                        msg: "invalid utf-8 in atom".to_string(),
                    })?
                    .to_string();
                if looks_numeric(&atom) {
                    match atom.parse::<f64>() {
                        Ok(v) => Ok(SExp::Num(v, pos)),
                        Err(_) => err(pos, format!("malformed number '{atom}'")),
                    }
                } else {
                    Ok(SExp::Sym(atom, pos))
                }
            }
        }
    }
}

/// Numbers start with a digit, or a sign/dot followed by a digit — so `-`
/// and `inf`/`nan` stay symbols.
fn looks_numeric(atom: &str) -> bool {
    let b = atom.as_bytes();
    match b[0] {
        b'0'..=b'9' => true,
        b'-' | b'+' | b'.' => b.len() > 1 && matches!(b[1], b'0'..=b'9' | b'.'),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// SExp → AST
// ─────────────────────────────────────────────────────────────────────────

fn as_sym(e: &SExp, what: &str) -> Result<String, LangError> {
    match e {
        SExp::Sym(s, _) => Ok(s.clone()),
        other => err(other.pos(), format!("expected {what}, got a non-symbol")),
    }
}

/// A list of bare symbols — the shape of a binder's bound-index list.
fn sym_list(e: &SExp) -> Option<Vec<String>> {
    match e {
        SExp::List(items, _) if !items.is_empty() => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    SExp::Sym(s, _) => out.push(s.clone()),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn to_expr(e: &SExp) -> Result<Expr, LangError> {
    match e {
        SExp::Num(v, _) => Ok(Expr::Num(*v)),
        SExp::Sym(s, _) => Ok(Expr::Scalar(s.clone())),
        SExp::List(items, pos) => {
            let Some(head) = items.first() else {
                return err(*pos, "empty list is not an expression");
            };
            let head_name = as_sym(head, "an operator or function in head position")?;
            let args = &items[1..];
            match head_name.as_str() {
                "@" => {
                    if args.len() < 2 {
                        return err(*pos, "(@ tensor idx…) needs a tensor and at least one \
                                          index; use a bare symbol for 0-dim references");
                    }
                    let name = as_sym(&args[0], "a tensor name after @")?;
                    let indices = args[1..]
                        .iter()
                        .map(|a| as_sym(a, "an index name"))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Expr::Tensor { name, indices })
                }
                "+" | "*" | "-" | "/" => {
                    let op = match head_name.as_str() {
                        "+" => BinOp::Add,
                        "*" => BinOp::Mul,
                        "-" => BinOp::Sub,
                        _ => BinOp::Div,
                    };
                    if op == BinOp::Sub && args.len() == 1 {
                        return Ok(Expr::negate(to_expr(&args[0])?));
                    }
                    if args.len() < 2 {
                        return err(*pos, format!("'{head_name}' needs at least two operands"));
                    }
                    let mut acc = to_expr(&args[0])?;
                    for a in &args[1..] {
                        acc = Expr::binary(op, acc, to_expr(a)?);
                    }
                    Ok(acc)
                }
                _ => {
                    // Binder shape: (op (i j…) body). A list of bare symbols
                    // is never a valid expression, so this is unambiguous.
                    if args.len() == 2 {
                        if let Some(indices) = sym_list(&args[0]) {
                            return Ok(Expr::Reduce {
                                op: head_name,
                                indices,
                                body: Box::new(to_expr(&args[1])?),
                            });
                        }
                    }
                    let args = args.iter().map(to_expr).collect::<Result<Vec<_>, _>>()?;
                    Ok(Expr::Call { func: head_name, args })
                }
            }
        }
    }
}

fn to_stmt(e: &SExp) -> Result<Stmt, LangError> {
    let SExp::List(items, pos) = e else {
        return err(e.pos(), "expected a (= lhs rhs) form");
    };
    let ok_head = matches!(items.first(), Some(SExp::Sym(s, _)) if s == "=");
    if !ok_head || items.len() != 3 {
        return err(*pos, "expected a (= lhs rhs) form");
    }
    let (name, indices) = match &items[1] {
        SExp::Sym(s, _) => (s.clone(), Vec::new()),
        SExp::List(parts, lpos) => {
            if parts.is_empty() {
                return err(*lpos, "empty LHS");
            }
            let name = as_sym(&parts[0], "an output name")?;
            let indices = parts[1..]
                .iter()
                .map(|p| as_sym(p, "an index name"))
                .collect::<Result<Vec<_>, _>>()?;
            (name, indices)
        }
        SExp::Num(_, npos) => return err(*npos, "LHS must be a name or (name idx…)"),
    };
    Ok(Stmt { name, indices, rhs: to_expr(&items[2])? })
}

/// Parse the s-expression encoding of a program. See the module docs for
/// the format; isomorphic to the infix form ([`super::parse`]).
pub fn parse_sexpr(src: &str) -> Result<Program, LangError> {
    let forms = Reader::new(src).read_all()?;
    let stmts = forms.iter().map(to_stmt).collect::<Result<Vec<_>, _>>()?;
    Ok(Program { stmts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_binder_and_tensor_ref() {
        let p = parse_sexpr("(= (y j) (sqrt (sum (l) (@ M l j))))").unwrap();
        assert_eq!(p.stmts.len(), 1);
        let s = &p.stmts[0];
        assert_eq!(s.name, "y");
        assert_eq!(s.indices, vec!["j"]);
        let Expr::Call { func, args } = &s.rhs else { panic!("expected sqrt call") };
        assert_eq!(func, "sqrt");
        let Expr::Reduce { op, indices, body } = &args[0] else { panic!("expected reduce") };
        assert_eq!(op, "sum");
        assert_eq!(indices, &vec!["l".to_string()]);
        let Expr::Tensor { name, indices } = body.as_ref() else { panic!("expected tensor") };
        assert_eq!(name, "M");
        assert_eq!(indices, &vec!["l".to_string(), "j".to_string()]);
    }

    #[test]
    fn nary_product_folds_left() {
        let p = parse_sexpr("(= s (* a b c))").unwrap();
        let Expr::Binary { op: BinOp::Mul, lhs, .. } = &p.stmts[0].rhs else {
            panic!("expected mul");
        };
        assert!(matches!(lhs.as_ref(), Expr::Binary { op: BinOp::Mul, .. }));
    }

    #[test]
    fn unary_minus_and_negative_literals() {
        let p = parse_sexpr("(= s (- x)) (= t -1.5)").unwrap();
        assert!(matches!(&p.stmts[0].rhs, Expr::Call { func, .. } if func == "neg"));
        assert_eq!(p.stmts[1].rhs, Expr::Num(-1.5));
    }

    #[test]
    fn elementwise_max_is_a_call_not_a_binder() {
        let p = parse_sexpr("(= s (max a b))").unwrap();
        assert!(matches!(&p.stmts[0].rhs, Expr::Call { func, args } if func == "max" && args.len() == 2));
    }

    #[test]
    fn comments_and_scalar_lhs() {
        let p = parse_sexpr("; a comment\n(= m (max (j) (@ v j))) ; trailing").unwrap();
        assert_eq!(p.stmts[0].name, "m");
        assert!(p.stmts[0].indices.is_empty());
    }

    #[test]
    fn rejects_unclosed_and_bad_forms() {
        assert!(matches!(parse_sexpr("(= s"), Err(LangError::Parse { .. })));
        assert!(matches!(parse_sexpr("(foo bar)"), Err(LangError::Parse { .. })));
        assert!(matches!(parse_sexpr("(= s ())"), Err(LangError::Parse { .. })));
    }
}
