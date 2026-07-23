//! Infix front-end — the human-facing syntax.
//!
//! ```text
//! m       = max(j: v[j])                       # scalar LHS, binder
//! soft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))
//! ```
//!
//! Statements are newline-terminated; a newline inside unbalanced
//! `(…)`/`[…]` continues the statement. `#` comments to end of line.
//! The colon marks a reduction binder — `head(i,j: body)`; without a
//! colon, `head(…)` is a function call. Grammar in `LANGUAGE.md` §4.

use super::LangError;
use super::ast::{BinOp, Expr, Program, Stmt};

// ─────────────────────────────────────────────────────────────────────────
// Lexer
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Eq,
    Comma,
    Colon,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Newline,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("identifier '{s}'"),
            Tok::Num(v) => format!("number {v}"),
            Tok::Newline => "end of statement".to_string(),
            Tok::Plus => "'+'".to_string(),
            Tok::Minus => "'-'".to_string(),
            Tok::Star => "'*'".to_string(),
            Tok::Slash => "'/'".to_string(),
            Tok::Eq => "'='".to_string(),
            Tok::Comma => "','".to_string(),
            Tok::Colon => "':'".to_string(),
            Tok::LBracket => "'['".to_string(),
            Tok::RBracket => "']'".to_string(),
            Tok::LParen => "'('".to_string(),
            Tok::RParen => "')'".to_string(),
        }
    }
}

fn perr<T>(line: usize, col: usize, msg: impl Into<String>) -> Result<T, LangError> {
    Err(LangError::Parse { line, col, msg: msg.into() })
}

/// Tokenize. Newlines are emitted only at bracket depth 0 (statement
/// separators); consecutive newlines collapse to one.
fn lex(src: &str) -> Result<Vec<(Tok, usize, usize)>, LangError> {
    let b = src.as_bytes();
    let mut toks = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1usize, 1usize);
    let mut depth = 0usize;
    while i < b.len() {
        let (c, tline, tcol) = (b[i], line, col);
        let mut bump = |i: &mut usize| {
            if b[*i] == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
            *i += 1;
        };
        match c {
            b'#' => {
                while i < b.len() && b[i] != b'\n' {
                    bump(&mut i);
                }
            }
            b'\n' => {
                bump(&mut i);
                if depth == 0 && !matches!(toks.last(), None | Some((Tok::Newline, _, _))) {
                    toks.push((Tok::Newline, tline, tcol));
                }
            }
            c if c.is_ascii_whitespace() => bump(&mut i),
            b'+' => {
                toks.push((Tok::Plus, tline, tcol));
                bump(&mut i);
            }
            b'-' => {
                toks.push((Tok::Minus, tline, tcol));
                bump(&mut i);
            }
            b'*' => {
                toks.push((Tok::Star, tline, tcol));
                bump(&mut i);
            }
            b'/' => {
                toks.push((Tok::Slash, tline, tcol));
                bump(&mut i);
            }
            b'=' => {
                toks.push((Tok::Eq, tline, tcol));
                bump(&mut i);
            }
            b',' => {
                toks.push((Tok::Comma, tline, tcol));
                bump(&mut i);
            }
            b':' => {
                toks.push((Tok::Colon, tline, tcol));
                bump(&mut i);
            }
            b'[' => {
                depth += 1;
                toks.push((Tok::LBracket, tline, tcol));
                bump(&mut i);
            }
            b']' => {
                depth = depth.saturating_sub(1);
                toks.push((Tok::RBracket, tline, tcol));
                bump(&mut i);
            }
            b'(' => {
                depth += 1;
                toks.push((Tok::LParen, tline, tcol));
                bump(&mut i);
            }
            b')' => {
                depth = depth.saturating_sub(1);
                toks.push((Tok::RParen, tline, tcol));
                bump(&mut i);
            }
            b'0'..=b'9' => {
                let start = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    bump(&mut i);
                }
                if i < b.len() && b[i] == b'.' {
                    bump(&mut i);
                    while i < b.len() && b[i].is_ascii_digit() {
                        bump(&mut i);
                    }
                }
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    let mut j = i + 1;
                    if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        while i < j {
                            bump(&mut i);
                        }
                        while i < b.len() && b[i].is_ascii_digit() {
                            bump(&mut i);
                        }
                    }
                }
                let text = std::str::from_utf8(&b[start..i]).unwrap();
                match text.parse::<f64>() {
                    Ok(v) => toks.push((Tok::Num(v), tline, tcol)),
                    Err(_) => return perr(tline, tcol, format!("malformed number '{text}'")),
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    bump(&mut i);
                }
                let text = std::str::from_utf8(&b[start..i]).unwrap().to_string();
                toks.push((Tok::Ident(text), tline, tcol));
            }
            other => return perr(tline, tcol, format!("unexpected character '{}'", other as char)),
        }
    }
    Ok(toks)
}

// ─────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────

struct Parser {
    toks: Vec<(Tok, usize, usize)>,
    pos: usize,
    end_line: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _, _)| t)
    }

    fn here(&self) -> (usize, usize) {
        self.toks.get(self.pos).map(|&(_, l, c)| (l, c)).unwrap_or((self.end_line, 1))
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).map(|(t, _, _)| t.clone());
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.peek() == Some(want) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, want: Tok, ctx: &str) -> Result<(), LangError> {
        let (l, c) = self.here();
        match self.bump() {
            Some(t) if t == want => Ok(()),
            Some(t) => perr(l, c, format!("expected {} {ctx}, got {}", want.describe(), t.describe())),
            None => perr(l, c, format!("expected {} {ctx}, got end of input", want.describe())),
        }
    }

    fn expect_ident(&mut self, ctx: &str) -> Result<String, LangError> {
        let (l, c) = self.here();
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            Some(t) => perr(l, c, format!("expected {ctx}, got {}", t.describe())),
            None => perr(l, c, format!("expected {ctx}, got end of input")),
        }
    }

    fn skip_newlines(&mut self) {
        while self.eat(&Tok::Newline) {}
    }

    // program := stmt+
    fn program(&mut self) -> Result<Program, LangError> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().is_none() {
                return Ok(Program { stmts });
            }
            stmts.push(self.stmt()?);
        }
    }

    // stmt := ident indexlist? '=' expr (NEWLINE | EOF)
    fn stmt(&mut self) -> Result<Stmt, LangError> {
        let name = self.expect_ident("an output name")?;
        let mut indices = Vec::new();
        if self.eat(&Tok::LBracket) {
            loop {
                indices.push(self.expect_ident("an index name")?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(Tok::RBracket, "after LHS indices")?;
        }
        self.expect(Tok::Eq, "after the statement LHS")?;
        let rhs = self.expr()?;
        let (l, c) = self.here();
        match self.bump() {
            None | Some(Tok::Newline) => Ok(Stmt { name, indices, rhs }),
            Some(t) => perr(l, c, format!("expected end of statement, got {}", t.describe())),
        }
    }

    // expr := term (('+' | '-') term)*
    fn expr(&mut self) -> Result<Expr, LangError> {
        let mut acc = self.term()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => BinOp::Add,
                Some(Tok::Minus) => BinOp::Sub,
                _ => return Ok(acc),
            };
            self.pos += 1;
            acc = Expr::binary(op, acc, self.term()?);
        }
    }

    // term := unary (('*' | '/') unary)*
    fn term(&mut self) -> Result<Expr, LangError> {
        let mut acc = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => BinOp::Mul,
                Some(Tok::Slash) => BinOp::Div,
                _ => return Ok(acc),
            };
            self.pos += 1;
            acc = Expr::binary(op, acc, self.unary()?);
        }
    }

    // unary := '-' unary | atom
    fn unary(&mut self) -> Result<Expr, LangError> {
        if self.eat(&Tok::Minus) {
            return Ok(Expr::negate(self.unary()?));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, LangError> {
        let (l, c) = self.here();
        match self.bump() {
            Some(Tok::Num(v)) => Ok(Expr::Num(v)),
            Some(Tok::LParen) => {
                let e = self.expr()?;
                self.expect(Tok::RParen, "to close the parenthesized expression")?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => match self.peek() {
                Some(Tok::LBracket) => {
                    self.pos += 1;
                    let mut indices = Vec::new();
                    loop {
                        indices.push(self.expect_ident("an index name")?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(Tok::RBracket, "after tensor indices")?;
                    Ok(Expr::Tensor { name, indices })
                }
                Some(Tok::LParen) => {
                    self.pos += 1;
                    // Binder if the content matches `ident (',' ident)* ':'`,
                    // otherwise a function call — backtrack to decide.
                    if let Some(indices) = self.try_binder_head() {
                        let body = self.expr()?;
                        self.expect(Tok::RParen, "to close the reduction")?;
                        Ok(Expr::Reduce { op: name, indices, body: Box::new(body) })
                    } else {
                        let mut args = Vec::new();
                        loop {
                            args.push(self.expr()?);
                            if !self.eat(&Tok::Comma) {
                                break;
                            }
                        }
                        self.expect(Tok::RParen, "to close the call")?;
                        Ok(Expr::Call { func: name, args })
                    }
                }
                _ => Ok(Expr::Scalar(name)),
            },
            Some(t) => perr(l, c, format!("expected an expression, got {}", t.describe())),
            None => perr(l, c, "expected an expression, got end of input"),
        }
    }

    /// After the head's `(`: consume `ident (',' ident)* ':'` if present,
    /// returning the bound indices; otherwise restore and return None.
    fn try_binder_head(&mut self) -> Option<Vec<String>> {
        let save = self.pos;
        let mut ids = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::Ident(s)) => {
                    ids.push(s.clone());
                    self.pos += 1;
                }
                _ => break,
            }
            match self.peek() {
                Some(Tok::Comma) => {
                    self.pos += 1;
                }
                Some(Tok::Colon) => {
                    self.pos += 1;
                    return Some(ids);
                }
                _ => break,
            }
        }
        self.pos = save;
        None
    }
}

/// Parse the infix form of a program. Isomorphic to the s-expression
/// front-end ([`super::parse_sexpr`]); see the module docs for the syntax.
pub fn parse(src: &str) -> Result<Program, LangError> {
    let toks = lex(src)?;
    let end_line = src.lines().count().max(1);
    let mut p = Parser { toks, pos: 0, end_line };
    p.program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_and_binder() {
        let p = parse("y[i] = a[i] + b[i] * c[i]").unwrap();
        let Expr::Binary { op: BinOp::Add, rhs, .. } = &p.stmts[0].rhs else {
            panic!("expected + at top");
        };
        assert!(matches!(rhs.as_ref(), Expr::Binary { op: BinOp::Mul, .. }));

        let p = parse("s = sum(i, j: t[i,j])").unwrap();
        let Expr::Reduce { op, indices, .. } = &p.stmts[0].rhs else { panic!("expected reduce") };
        assert_eq!(op, "sum");
        assert_eq!(indices, &vec!["i".to_string(), "j".to_string()]);
    }

    #[test]
    fn call_vs_binder_backtracks() {
        // `max(a, b)` — two idents then `)` not `:` → a call.
        let p = parse("s = max(a, b)").unwrap();
        assert!(matches!(&p.stmts[0].rhs, Expr::Call { func, args } if func == "max" && args.len() == 2));
        // `max(a: …)` → a binder even though `max` is also a function.
        let p = parse("s = max(a: v[a])").unwrap();
        assert!(matches!(&p.stmts[0].rhs, Expr::Reduce { .. }));
    }

    #[test]
    fn newline_continuation_inside_parens() {
        let p = parse("s = sum(j:\n    v[j])\nt = s").unwrap();
        assert_eq!(p.stmts.len(), 2);
    }

    #[test]
    fn comments_blank_lines_scientific_notation() {
        let p = parse("# header\n\ny[i] = x[i] * 1e-5  # eps\n").unwrap();
        assert_eq!(p.stmts.len(), 1);
        let Expr::Binary { rhs, .. } = &p.stmts[0].rhs else { panic!() };
        assert_eq!(rhs.as_ref(), &Expr::Num(1e-5));
    }

    #[test]
    fn unary_minus() {
        let p = parse("y[i] = -x[i] + 2").unwrap();
        let Expr::Binary { op: BinOp::Add, lhs, .. } = &p.stmts[0].rhs else { panic!() };
        assert!(matches!(lhs.as_ref(), Expr::Call { func, .. } if func == "neg"));
        assert!(matches!(parse("t = -2.5").unwrap().stmts[0].rhs, Expr::Num(v) if v == -2.5));
    }

    #[test]
    fn errors_have_positions() {
        let Err(LangError::Parse { line, .. }) = parse("y[i] = x[i]\nz = +") else {
            panic!("expected parse error");
        };
        assert_eq!(line, 2);
        assert!(parse("y[i = x[i]").is_err());
        assert!(parse("y[i] = x[i] x").is_err());
    }

    #[test]
    fn matches_sexpr_front_end() {
        let infix = parse(
            "y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))\n\
             soft[i] = exp(v[i]) / sum(j: exp(v[j]))",
        )
        .unwrap();
        let sexpr = crate::lang::parse_sexpr(
            "(= (y j k) (* (@ q k) (min (l) (@ x l)) (sqrt (sum (l) (@ M l j)))))\n\
             (= (soft i) (/ (exp (@ v i)) (sum (j) (exp (@ v j)))))",
        )
        .unwrap();
        assert_eq!(infix, sexpr);
    }
}
