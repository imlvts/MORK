# Sinks emit `root ++ template` instead of `template`

`WriteZipper::move_to_path` is **relative to the zipper's root**
(`PathMap/src/zipper.rs:156` — "Moves the zipper's focus to a specific location
specified by `path`, relative to the zipper's root"). Five sinks pass it an
**absolute** path, so the emitted expression is the write root followed by the
whole template rather than the template alone.

This is pre-existing on `main` (`45bdb9a`) and is independent of PR #137. The
equivalent defect in `PureSink` was fixed there; these were left alone because
each sink has its own `request()` and the roots needed checking separately. They
have now been checked, and they are broken in the same way.

## Reproduction

Each program below emits a template holding an unbound variable. Run against a
release build of `main`:

```scheme
(foo 1) (foo 2) (foo 3)
(exec 0 (, (foo $x)) (O (count (tag $q) $ (cux $x))))
```

| program | expected | actual on `main` |
|---|---|---|
| `(count (tag $q) $ (cux $x))` | `(tag $a)` | `(tag (tag $a))` |
| `(hash  (tag $q) $ (cux $x))` | `(tag $a)` | `(tag (tag $a))` |
| `(and   (tag $q) $ (cux $x))` | `(tag $a)` | `(tag (tag $a))` |
| `(sum   (tag $q) $ $x)`       | `(tag $a)` | `(tag (tag $a))` |
| `(fmin  (tag $q) $ $x)`       | `(tag $a)` | `(tag (tag $a))` |

The literal-guard arm is affected too, not only the ignore arm. With three `foo`
facts so the count matches:

```scheme
(exec 0 (, (foo $x)) (O (count (tag $q) 3 (cux $x))))   ; => (tag (tag $a))
```

The control that pins the diagnosis is the var-ref arm, which is the one arm in
each of these sinks that already strips the root. It is correct:

```scheme
(exec 0 (, (foo $x)) (O (count (tag $c) $c (cux $x))))  ; => (tag 3)   correct
```

## Mechanism

`finalize` builds `rooted_input` by grafting the registered paths at
`wz.root_prefix_path()`, then reads it with a zipper rooted at `&[]`:

```rust
rooted_input.write_zipper_at_path(wz.root_prefix_path()).graft_map(_to_swap);
let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
```

So `prz.path()` is **absolute** — it includes the write root. Handing it
straight to `wz.move_to_path`, whose argument is measured from the write root,
appends it to the root instead of replacing it.

Byte arithmetic for the `count` case above. The template is `(tag $q)`;
`request()` is `e.prefix()[7..]`, whose constant prefix stops at `$q`:

```
root     = [Arity(2)][SymbolSize(3)]tag            5 bytes
ignored  = [Arity(2)][SymbolSize(3)]tag[NewVar]    6 bytes   (absolute template)
written  = root ++ ignored                        11 bytes = (tag (tag $))
```

### Why it went unnoticed

The surplus is only visible when the root is **shorter** than the template,
i.e. when the template contains a variable. When the template is fully
constant the root equals the template, the doubled path is
`template ++ template`, and the trailing copy hangs below a complete
expression — so `dump_sexpr` reads the leading part and prints the right
answer. `sink_count_constant` and `sink_count_literal` pass for exactly that
reason while still writing a malformed over-long path into the trie.

That masking is worth treating as part of the bug: these sinks have been
depositing paths with trailing garbage, which inflates the space and means any
consumer that cares about exact paths (rather than the leading complete
expression) sees something wrong.

## Affected call sites

`kernel/src/sinks.rs`, on `main` at `45bdb9a`. Two arms per sink — the
literal/`fixed` guard and the `ignored` (NewVar) guard:

| sink | `impl` at | `fixed` | `ignored` | var-ref arm (correct) |
|---|---|---|---|---|
| `CountSink` | 548 | 589 | 598 | 614 |
| `HashSink` | 627 | 678 | 691 | 710 |
| `AndSink` | 724 | 781 | 794 | 821 |
| `SumSink` | 834 | 890 | 903 | 930 |
| `FloatReductionSink` | 976 | 1032 | 1045 | 1072 |

Not affected — these already strip, and are the reference for the fix:

- `CompatSink:148` and `AddSink:170` compute `&path[.. + wz.root_prefix_path().len()..]` before writing.
- `USink:260`, `AUSink:305`, and the var-ref arm of each sink above.

Suspected but **untested**: `WASMSink:528` writes `ospan`, a full expression
built in the module's output memory, with no stripping. It is behind the `wasm`
feature and was not exercised. It should be checked before being either fixed
or dismissed.

## Fix

Cut the absolute path down to the write root at each site, matching what the
var-ref arm in the same function already does:

```rust
-  wz.move_to_path(ignored);
+  wz.move_to_path(&ignored[wz.root_prefix_path().len()..]);
```

and likewise for `fixed`.

### Check `request()` at the same time

These sinks derive the write root from the prefix of the **whole** sink
expression:

```rust
let p = &unsafe { self.e.prefix().unwrap_or_else(..).as_ref().unwrap() }[7..];
```

The root should cover only what the sink writes, which is the template. For a
constant template the whole-expression prefix runs past the template and into
the pattern and the call. In `PureSink` that combination made a valid emit
impossible and the result silently vanished (#136 follow-up); here it merely
lengthens the root and so compounds the doubling. The fix applied to
`PureSink::request()` — take the prefix of the template at offset `2 + name_len`,
falling back to one byte above a fully constant template — transfers directly.

The two changes are **coupled**: correcting `request()` without stripping the
paths turns the currently-invisible constant-template case into visibly corrupt
output (`ignored` became `ignore\xef`). Land them together, per sink.

## Suggested regression test

Model on `sink_pure_constant_template_guard` in `kernel/src/main.rs`. The key
point is that a constant template cannot catch this — the template must hold a
variable so the root is shorter than what is written:

```scheme
(foo 1) (foo 2) (foo 3)
(exec 0 (, (foo $x)) (O (count (tag $q) $ (cux $x))))
```

asserting `(tag $a)` and not `(tag (tag $a))`.
