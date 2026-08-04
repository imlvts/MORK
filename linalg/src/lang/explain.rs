//! Debug rendering: a checked program as the loop nest it means.
//!
//! The printer walks a [`Checked`] exactly the way `eval::lower` walks
//! it — same free-index computation, same emission order — so the print
//! and the reference execution cannot drift in structure.

use super::check::{CExpr, CStmt, Checked, IndexId};
use super::{Elem, LangError, Registry};
use crate::einsum::Reduce;
use crate::tensor::NDIndex;

impl Checked {
    /// Render the program as nested loops and assignments, with extents
    /// left symbolic (`0..|j|`).
    ///
    /// The output is the imperative form the language is *defined* to
    /// compute: the v1 materializing lowering (LANGUAGE.md §10 E3) that
    /// [`run_reference`](super::run_reference) executes. Every reduction
    /// node becomes a temporary `%k` with its own accumulator `acck`,
    /// allocated at exactly the free indices occurring under it and
    /// computed once, innermost reduction first — which is how you *see*
    /// that a softmax denominator is hoisted out of the element loop
    /// rather than recomputed per output.
    ///
    /// It renders the *semantics*, not whatever backend actually runs: the
    /// kernel and JIT paths are required to agree with this lowering bit
    /// for bit, so a program that surprises you reads the same here either
    /// way. Loop order is the real one (row-major, last index fastest) and
    /// binary operators are fully parenthesized, so the print also answers
    /// float-associativity questions.
    ///
    /// Pass the registry the program was [`check`](super::check)ed
    /// against — function and reduction names live there, not in the
    /// checked form. [`explain_bound`](Self::explain_bound) is the same
    /// print with real extents.
    ///
    /// ```
    /// use linalg::lang::{Registry, check, parse};
    ///
    /// let reg = Registry::<f32>::builtins();
    /// let prog = parse("s = sum(j: x[j] * x[j])").unwrap();
    /// let text = check(&prog, &reg).unwrap().explain(&reg);
    ///
    /// assert!(text.contains("for j in 0..|j| {"));
    /// assert!(text.contains("acc0 = (acc0 + (x[j] * x[j]))"));
    /// ```
    pub fn explain<T>(&self, reg: &Registry<T>) -> String {
        let mut p = Printer::new(self, reg, None);
        p.header(self, &[]);
        p.program(self);
        p.out
    }

    /// [`explain`](Self::explain) against a concrete environment: extents
    /// become numbers, temporaries get shapes, and statements that no
    /// requested output names are marked as such.
    ///
    /// Takes exactly what [`run`](super::run) takes — copy the call and
    /// change the verb — and reports the same binding errors it would,
    /// since it runs the same binding phase. The output tensors are read
    /// for their shapes only; nothing is written and nothing is executed.
    pub fn explain_bound<T: Elem>(
        &self,
        reg: &Registry<T>,
        inputs: &[(&str, &dyn NDIndex<T>)],
        outputs: &mut [(&str, &mut dyn NDIndex<T>)],
    ) -> Result<String, LangError> {
        let extents = super::eval::bind(self, inputs, outputs)?;
        let out_names: Vec<String> = outputs.iter().map(|(n, _)| n.to_string()).collect();
        let mut p = Printer::new(self, reg, Some(Env { extents, outputs: out_names }));
        p.header(self, inputs);
        p.program(self);
        Ok(p.out)
    }
}

/// What binding against an environment adds to the print.
struct Env {
    /// Extent of every index id, as [`bind`](super::eval::bind) inferred it.
    extents: Vec<usize>,
    /// Names the caller asked for; every other statement is a temporary.
    outputs: Vec<String>,
}

struct Printer<'a, T> {
    reg: &'a Registry<T>,
    /// Source name of every index id, as [`Checked`] recorded it.
    index_names: &'a [String],
    /// Printable name of every index id, recomputed per statement — see
    /// [`display_names`].
    names: Vec<String>,
    env: Option<Env>,
    out: String,
    depth: usize,
    /// How many reduction temporaries have been emitted so far; also the
    /// number of the next `%k` / `acck` pair.
    temps: usize,
}

/// A printable name per index id, for the one statement whose ids are
/// `used` (ascending). Binders are alpha-renamed per occurrence, so one
/// source name can stand for several distinct axes; where that happens
/// *within a statement* (`min(l: …) * sum(l: …)`) each gets an occurrence
/// suffix, so the block reads unambiguously. Across statements it cannot
/// be ambiguous — each is printed on its own — so the plain source name
/// stays, and `t[i] = …` followed by `y[i] = …` prints as written.
fn display_names(index_names: &[String], used: &[IndexId]) -> Vec<String> {
    let mut names: Vec<String> = index_names.to_vec();
    for (k, &id) in used.iter().enumerate() {
        let n = &index_names[id as usize];
        let dups = used.iter().filter(|&&o| index_names[o as usize] == *n).count();
        if dups > 1 {
            let ord = used[..k].iter().filter(|&&o| index_names[o as usize] == *n).count() + 1;
            names[id as usize] = format!("{n}#{ord}");
        }
    }
    names
}

/// Every index id one statement mentions, ascending.
fn stmt_ids(stmt: &CStmt) -> Vec<IndexId> {
    fn walk(e: &CExpr, out: &mut Vec<IndexId>) {
        match e {
            CExpr::Num(_) | CExpr::ScalarRef(_) => {}
            CExpr::Load { indices, .. } => out.extend_from_slice(indices),
            CExpr::Call { args, .. } => args.iter().for_each(|a| walk(a, out)),
            CExpr::Binary { lhs, rhs, .. } => {
                walk(lhs, out);
                walk(rhs, out);
            }
            CExpr::Reduce { indices, body, .. } => {
                out.extend_from_slice(indices);
                walk(body, out);
            }
        }
    }
    let mut ids = stmt.lhs.clone();
    walk(&stmt.rhs, &mut ids);
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Literals print in `Debug` form so an integral one still reads as a
/// number of the element type (`2.0`, not `2`).
fn num(v: f64) -> String {
    format!("{v:?}")
}

fn op_str(op: super::ast::BinOp) -> &'static str {
    use super::ast::BinOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
    }
}

impl<'a, T> Printer<'a, T> {
    fn new(checked: &'a Checked, reg: &'a Registry<T>, env: Option<Env>) -> Self {
        Printer {
            reg,
            index_names: &checked.index_names,
            names: checked.index_names.clone(),
            env,
            out: String::new(),
            depth: 0,
            temps: 0,
        }
    }

    // ── output primitives ──

    fn line(&mut self, s: &str) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// A blank separator line, unless one is already there.
    fn blank(&mut self) {
        if !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
    }

    fn open_loop(&mut self, id: IndexId) {
        let s = format!("for {} in 0..{} {{", self.names[id as usize], self.extent(id));
        self.line(&s);
        self.depth += 1;
    }

    fn close(&mut self, n: usize) {
        for _ in 0..n {
            self.depth -= 1;
            self.line("}");
        }
    }

    // ── naming ──

    fn extent(&self, id: IndexId) -> String {
        match &self.env {
            Some(env) => env.extents[id as usize].to_string(),
            None => format!("|{}|", self.names[id as usize]),
        }
    }

    fn ix_list(&self, ids: &[IndexId]) -> String {
        ids.iter().map(|&i| self.names[i as usize].as_str()).collect::<Vec<_>>().join(", ")
    }

    /// `t[i, j]`, or bare `t` at rank 0.
    fn subscript(&self, base: &str, ids: &[IndexId]) -> String {
        if ids.is_empty() { base.to_string() } else { format!("{base}[{}]", self.ix_list(ids)) }
    }

    /// `alloc[i, j]`, with the concrete shape appended when known.
    fn alloc(&self, ids: &[IndexId]) -> String {
        let head = format!("alloc[{}]", self.ix_list(ids));
        match &self.env {
            Some(env) => {
                let dims: Vec<String> =
                    ids.iter().map(|&i| env.extents[i as usize].to_string()).collect();
                format!("{head}    // shape [{}]", dims.join(", "))
            }
            None => head,
        }
    }

    /// The reduction's identity, as the accumulator's initial value.
    fn identity(&self, op: usize) -> String {
        match self.reg.builtin_reduce(op) {
            Some(Reduce::Sum) => "0".to_string(),
            Some(Reduce::Prod) => "1".to_string(),
            // Not `-inf`/`+inf`: the element type need not be a float.
            Some(Reduce::Max) => "LEAST".to_string(),
            Some(Reduce::Min) => "GREATEST".to_string(),
            None => format!("identity({})", self.reg.reduces[op].name),
        }
    }

    /// One fold step, in the operator's natural notation.
    fn fold(&self, op: usize, acc: &str, v: &str) -> String {
        match self.reg.builtin_reduce(op) {
            Some(Reduce::Sum) => format!("({acc} + {v})"),
            Some(Reduce::Prod) => format!("({acc} * {v})"),
            Some(Reduce::Max) => format!("max({acc}, {v})"),
            Some(Reduce::Min) => format!("min({acc}, {v})"),
            None => format!("{}({acc}, {v})", self.reg.reduces[op].name),
        }
    }

    // ── the program ──

    fn header(&mut self, checked: &Checked, inputs: &[(&str, &dyn NDIndex<T>)]) {
        let plural = |n: usize, what: &str| {
            format!("{n} {what}{}", if n == 1 { "" } else { "s" })
        };
        let s = format!(
            "// {}, {}{}",
            plural(checked.stmts.len(), "statement"),
            plural(checked.index_names.len(), "index binding"),
            if self.env.is_some() { "" } else { "; extents unbound" },
        );
        self.line(&s);
        if self.env.is_none() {
            return;
        }
        for (name, t) in inputs {
            let dims: Vec<String> = (0..t.ndim()).map(|a| t.dim(a).to_string()).collect();
            let s = format!("// input {name}[{}]", dims.join(", "));
            self.line(&s);
        }
    }

    fn program(&mut self, checked: &Checked) {
        for stmt in &checked.stmts {
            self.blank();
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &CStmt) {
        self.names = display_names(self.index_names, &stmt_ids(stmt));
        let temporary = match &self.env {
            Some(env) => !env.outputs.contains(&stmt.name),
            None => false,
        };
        let s = format!(
            "// {} = {}{}",
            self.subscript(&stmt.name, &stmt.lhs),
            self.source(&stmt.rhs),
            if temporary { "    (temporary)" } else { "" },
        );
        self.line(&s);

        // Reductions materialize first, innermost first — exactly as the
        // evaluator lowers them — leaving a reduce-free elementwise body.
        let (body, _) = self.emit(&stmt.rhs);

        let s = format!("{} = {}", stmt.name, self.alloc(&stmt.lhs));
        self.line(&s);
        for &id in &stmt.lhs {
            self.open_loop(id);
        }
        let s = format!("{} = {body}", self.subscript(&stmt.name, &stmt.lhs));
        self.line(&s);
        self.close(stmt.lhs.len());
    }

    /// Emit whatever loop nests `e` needs and return (the expression that
    /// reads its value, the index ids that expression loads at).
    ///
    /// The id list mirrors `eval::collect_ids` over the lowered form: it
    /// is what a materialized reduction takes its free indices from, so a
    /// temporary's rank here is the temporary's rank there.
    fn emit(&mut self, e: &CExpr) -> (String, Vec<IndexId>) {
        match e {
            CExpr::Num(v) => (num(*v), Vec::new()),
            CExpr::ScalarRef(name) => (name.clone(), Vec::new()),
            CExpr::Load { tensor, indices } => {
                (self.subscript(tensor, indices), indices.clone())
            }
            CExpr::Call { func, args } => {
                let mut ids = Vec::new();
                let mut parts = Vec::with_capacity(args.len());
                for a in args {
                    let (s, i) = self.emit(a);
                    parts.push(s);
                    ids.extend(i);
                }
                (format!("{}({})", self.reg.fns[*func].name, parts.join(", ")), ids)
            }
            CExpr::Binary { op, lhs, rhs } => {
                let (l, mut ids) = self.emit(lhs);
                let (r, rids) = self.emit(rhs);
                ids.extend(rids);
                (format!("({l} {} {r})", op_str(*op)), ids)
            }
            CExpr::Reduce { op, indices, body } => {
                let src = self.source(e);
                let (body_s, body_ids) = self.emit(body);

                // Free indices of the body: everything it reads that this
                // binder does not bind, ascending — the temp's axis order.
                let mut free = body_ids;
                free.retain(|id| !indices.contains(id));
                free.sort_unstable();
                free.dedup();

                let k = self.temps;
                self.temps += 1;
                let (tmp, acc) = (format!("%{k}"), format!("acc{k}"));

                // No leading separator: each block's trailing blank already
                // separates it from whatever follows, and the first block
                // sits directly under the statement's own comment.
                let s = format!("// {tmp} <- {src}");
                self.line(&s);
                let s = format!("{tmp} = {}", self.alloc(&free));
                self.line(&s);
                for &id in &free {
                    self.open_loop(id);
                }
                let s = format!("{acc} = {}", self.identity(*op));
                self.line(&s);
                for &id in indices {
                    self.open_loop(id);
                }
                let s = format!("{acc} = {}", self.fold(*op, &acc, &body_s));
                self.line(&s);
                self.close(indices.len());
                let s = format!("{} = {acc}", self.subscript(&tmp, &free));
                self.line(&s);
                self.close(free.len());
                self.blank();

                (self.subscript(&tmp, &free), free)
            }
        }
    }

    /// The statement's own expression, re-rendered in source form — the
    /// comment that heads each lowered block.
    fn source(&self, e: &CExpr) -> String {
        match e {
            CExpr::Num(v) => num(*v),
            CExpr::ScalarRef(name) => name.clone(),
            CExpr::Load { tensor, indices } => self.subscript(tensor, indices),
            CExpr::Call { func, args } => {
                let parts: Vec<String> = args.iter().map(|a| self.source(a)).collect();
                format!("{}({})", self.reg.fns[*func].name, parts.join(", "))
            }
            CExpr::Binary { op, lhs, rhs } => {
                format!("({} {} {})", self.source(lhs), op_str(*op), self.source(rhs))
            }
            CExpr::Reduce { op, indices, body } => format!(
                "{}({}: {})",
                self.reg.reduces[*op].name,
                self.ix_list(indices),
                self.source(body)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Registry, check, parse};
    use crate::dense::Dense;
    use crate::tensor::NDIndex;

    fn explain(src: &str) -> String {
        let reg = Registry::<f32>::builtins();
        let text = check(&parse(src).unwrap(), &reg).unwrap().explain(&reg);
        println!("{src}\n{text}");
        text
    }

    /// Every `{` opens a loop and every `}` closes one, and the nesting
    /// never goes negative — the print is a well-formed loop nest.
    fn assert_balanced(text: &str) {
        let mut depth = 0i32;
        for line in text.lines() {
            let t = line.trim();
            if t == "}" {
                depth -= 1;
                assert!(depth >= 0, "unbalanced close in:\n{text}");
                assert_eq!(
                    line.len() - line.trim_start().len(),
                    depth as usize * 2,
                    "close at wrong indent in:\n{text}"
                );
            } else if t.starts_with("for ") {
                assert!(t.ends_with('{'), "loop header without a brace: {t}");
                depth += 1;
            }
        }
        assert_eq!(depth, 0, "unclosed loop in:\n{text}");
    }

    #[test]
    fn matmul_is_a_three_deep_nest() {
        let text = explain("c[i,k] = sum(j: a[i,j] * b[j,k])");
        assert_balanced(&text);
        // The temp carries the reduction's free indices (i, k) and the
        // fold runs inside them.
        assert!(text.contains("%0 = alloc[i, k]"));
        assert!(text.contains("for i in 0..|i| {"));
        assert!(text.contains("acc0 = (acc0 + (a[i, j] * b[j, k]))"));
        assert!(text.contains("%0[i, k] = acc0"));
        assert!(text.contains("c[i, k] = %0[i, k]"));
    }

    #[test]
    fn scalar_reduction_has_no_outer_loop() {
        let text = explain("s = max(j: v[j])");
        assert_balanced(&text);
        assert!(text.contains("%0 = alloc[]"));
        assert!(text.contains("acc0 = LEAST"));
        assert!(text.contains("acc0 = max(acc0, v[j])"));
        // Rank 0: no brackets on the load, no loop around the store.
        assert!(text.contains("%0 = acc0"));
        assert!(text.contains("s = %0"));
    }

    #[test]
    fn sibling_binders_reusing_a_name_are_disambiguated() {
        let text = explain("y[j,k] = q[k] * min(l: x[l]) * sqrt(sum(l: M[l,j]))");
        assert_balanced(&text);
        assert!(text.contains("for l#1 in 0..|l#1| {"));
        assert!(text.contains("for l#2 in 0..|l#2| {"));
        // Two temps, the second one indexed by the free index j.
        assert!(text.contains("%0 = alloc[]"));
        assert!(text.contains("%1 = alloc[j]"));
        assert!(text.contains("acc1 = (acc1 + M[l#2, j])"));
        assert!(text.contains("y[j, k] = ((q[k] * %0) * sqrt(%1[j]))"));
    }

    /// A softmax denominator is a temp computed once, not per output
    /// element — the property that makes it O(n) rather than O(n²).
    #[test]
    fn reductions_materialize_outside_the_element_loop() {
        let text = explain("soft[i] = exp(v[i]) / sum(j: exp(v[j]))");
        assert_balanced(&text);
        let temp = text.find("%0 = alloc").unwrap();
        let loop_i = text.find("for i in").unwrap();
        // The store itself, not the source-form comment that heads the block.
        let store = text.find("\n  soft[i] = ").unwrap();
        assert!(temp < loop_i && loop_i < store, "denominator not hoisted:\n{text}");
    }

    #[test]
    fn nested_reductions_lower_innermost_first() {
        let text = explain("s = sum(i: prod(j: M[i,j]))");
        assert_balanced(&text);
        // The inner product is a temp over the outer binder's index...
        assert!(text.contains("%0 = alloc[i]"));
        assert!(text.contains("acc0 = 1"));
        // ...and the outer sum folds over that temp.
        assert!(text.contains("acc1 = (acc1 + %0[i])"));
        assert!(text.find("%0 = alloc").unwrap() < text.find("%1 = alloc").unwrap());
    }

    /// The two-statement stable softmax: each statement is printed as its
    /// own block, and `j` in one is `j` in the other even though they are
    /// different index ids.
    #[test]
    fn statements_print_independently() {
        let text = explain("m = max(j: v[j])\nsoft[i] = exp(v[i] - m) / sum(j: exp(v[j] - m))");
        assert_balanced(&text);
        assert!(!text.contains('#'), "no cross-statement disambiguation needed:\n{text}");
        assert_eq!(text.matches("for j in 0..|j| {").count(), 2);
        // The scalar `m` reads without brackets, in both roles.
        assert!(text.contains("m = %0"));
        assert!(text.contains("acc1 = (acc1 + exp((v[j] - m)))"));
    }

    #[test]
    fn binaries_are_fully_parenthesized() {
        // Left-associative, as the evaluator folds it — and that is what
        // the print has to show, since float `+` is not associative.
        let text = explain("y[i] = a[i] + b[i] + c[i]");
        assert!(text.contains("y[i] = ((a[i] + b[i]) + c[i])"));
    }

    #[test]
    fn bound_print_has_real_extents() {
        let reg = Registry::<f32>::builtins();
        let checked = check(&parse("c[i,k] = sum(j: a[i,j] * b[j,k])").unwrap(), &reg).unwrap();
        let a = Dense::<f32>::zeros(vec![2, 3]);
        let b = Dense::<f32>::zeros(vec![3, 4]);
        let mut c = Dense::<f32>::zeros(vec![2, 4]);
        let text = checked
            .explain_bound(
                &reg,
                &[("a", &a as &dyn NDIndex<f32>), ("b", &b)],
                &mut [("c", &mut c as &mut dyn NDIndex<f32>)],
            )
            .unwrap();
        println!("{text}");
        assert_balanced(&text);
        assert!(text.contains("// input a[2, 3]"));
        assert!(text.contains("for i in 0..2 {"));
        assert!(text.contains("for j in 0..3 {"));
        assert!(text.contains("for k in 0..4 {"));
        assert!(text.contains("%0 = alloc[i, k]    // shape [2, 4]"));
        // `c` was requested, so it is not flagged a temporary.
        assert!(!text.contains("(temporary)"));
    }

    #[test]
    fn bound_print_flags_temporaries_and_reports_bind_errors() {
        let reg = Registry::<f32>::builtins();
        let checked = check(&parse("t[i] = v[i] * 2\ny[i] = t[i] + 1").unwrap(), &reg).unwrap();
        let v = Dense::<f32>::zeros(vec![5]);
        let mut y = Dense::<f32>::zeros(vec![5]);
        let text = checked
            .explain_bound(
                &reg,
                &[("v", &v as &dyn NDIndex<f32>)],
                &mut [("y", &mut y as &mut dyn NDIndex<f32>)],
            )
            .unwrap();
        println!("{text}");
        assert!(text.contains("// t[i] = (v[i] * 2.0)    (temporary)"));
        assert!(text.contains("// y[i] = (t[i] + 1.0)\n"));

        // Same binding phase as `run`, so the same errors surface.
        let mut wrong = Dense::<f32>::zeros(vec![4]);
        assert!(matches!(
            checked.explain_bound(
                &reg,
                &[("v", &v as &dyn NDIndex<f32>)],
                &mut [("y", &mut wrong as &mut dyn NDIndex<f32>)],
            ),
            Err(super::LangError::ExtentMismatch { .. })
        ));
    }
}
