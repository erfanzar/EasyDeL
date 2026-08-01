---
name: docstring-swarm
description: Launch a swarm of parallel agents to add and update docstrings across a codebase — module/file, class, and function/method docstrings, public and private, documenting every parameter, return value, raised error, and IN/OUT args + kwargs. Use to document or re-document a package, fix outdated/drifted docstrings, or expand `**kwargs: Unpacked[TypedDict]` into per-field docs. Never adds standalone variable/attribute string-literal docstrings and never changes code behavior.
---

# Skill: Docstring Swarm

Use this to document a codebase at scale: fan out many agents in parallel, each
owning a disjoint set of files, each adding or repairing docstrings on every
module, class, function, and method — public **and** private. The output is a
**docstring-only diff**: clear, accurate documentation and nothing else.

Two promises define the skill. **Accuracy** — every docstring describes what the
code actually does, read from the implementation, never guessed. **Safety** —
not one byte of runtime behavior changes; the only edits are docstrings.

## What Gets Documented

For the scoped target, document all of:

- **Files / modules** — a top-of-file docstring stating the module's purpose,
  what it contains, and how it fits the package.
- **Classes** (public and private) — purpose, behavior, important attributes,
  and usage notes.
- **Functions and methods** (public and private) — a summary plus the full
  IN/OUT contract.

For every function/method, the IN/OUT contract is:

- **IN** — `Args`/`Parameters`: every positional and keyword parameter, its
  type, meaning, defaults, and constraints. `*args`/`**kwargs` explained.
- **OUT** — `Returns`/`Yields`: the type and meaning of what comes back,
  including the shape of tuples/dicts and generator items.
- **Raises** — each exception the function can raise and the condition for it.
- Examples where they clarify non-obvious usage.

"Explain every single thing" is the bar: a reader should understand the unit
without reading its body.

## Hard Rules — Read Before Launching

1. **Docstring-only diffs.** Never change logic, signatures, imports, type
   annotations, decorators, or code formatting. If a function looks buggy, note
   it in your final report — do not fix it here.
2. **Never add variable / attribute docstrings.** The pattern
   ```python
   X = ...
   """this is a docstring"""   # ← FORBIDDEN, never introduce this
   ```
   is out of scope at module level, class level, and inside functions. Document
   a notable attribute inside the **enclosing class or module docstring's
   `Attributes:` section**, never as a standalone string literal after an
   assignment. Do not delete existing ones unless you are rewriting that
   docstring for accuracy.
3. **Accuracy over coverage.** Read the implementation before writing. Never
   invent parameters, return values, or behavior. If something is genuinely
   ambiguous, describe what the code does conservatively rather than asserting
   intent you can't verify.
4. **Update, don't duplicate.** Fix **outdated / drifted** docstrings: wrong or
   renamed parameter names, removed parameters, changed return types, stale
   prose that no longer matches the body. If a docstring is already correct and
   complete, leave it — minimize churn.
5. **Expand `**kwargs: Unpacked[AnyClass]`.** When kwargs are typed with PEP-692
   `Unpacked[SomeTypedDict]` (or an equivalent config/dataclass), open
   `SomeTypedDict`, read its fields, and document **each field** as a keyword
   argument (name, type, default, meaning) under the IN section, plus a line
   noting they flow through `**kwargs`. Reflect any that affect the result in
   OUT. See the worked example in `docs/reference/docstring-style.md`.
6. **Match the project's convention.** Detect the dominant existing style
   (Google / NumPy / reST-Sphinx) and follow it exactly. Do not mix styles
   within a file or convert a file's existing style.
7. **Don't break the build.** After editing a file it must still parse and (where
   cheap) import. A docstring is the first statement of a module/class/function;
   inserting one is safe only if you place and indent it correctly.

## Step 0 — Detect Convention And Scope

Before any fan-out, the orchestrator establishes shared ground:

- **Convention.** Sample existing docstrings (`rg -n '"""' <target> | head`) and
  open a few to identify Google (`Args:` / `Returns:`), NumPy (`Parameters` with
  `----` underlines), or reST (`:param:` / `:returns:`). Pick the dominant one;
  if none dominates, default to **Google style**. Record the choice — every
  agent must use it.
- **Scope.** Take the target path/package from the user. Respect `.gitignore`.
  **Skip** generated files (`*_pb2.py`, `*_pb2_grpc.py`, migrations), vendored /
  third-party trees, build artifacts, and `__pycache__`. Tests are **opt-in** —
  exclude `test_*.py` / `*_test.py` unless the user asks for them.
- Record line length / formatting norms (e.g. an 88- or 100-col limit) so
  docstrings wrap consistently with the codebase.

## Step 1 — Build The Worklist

- Enumerate the in-scope source files and get a rough count of modules,
  classes, and functions, so the run is sized and reportable.
- **Partition into batches, each file owned by exactly one agent** — never let
  two agents edit the same file (the only real source of conflicts). Batch by
  directory/module for locality; aim for batches small enough that one agent can
  read each file's implementation carefully (a handful to ~15 files, fewer for
  large files), not so many that any single agent is overloaded.

## Step 2 — Launch The Swarm

Spawn the agents in parallel, one per batch, each carrying the per-agent
contract below plus the shared convention and scope decisions.

- In this harness, prefer the **Workflow** tool: a `pipeline` over batches where
  stage one documents the batch and stage two (optional) verifies it, or a
  `parallel` fan-out over batches. Files are disjoint, so no worktree isolation
  is needed.
- Without a workflow runtime, fan out parallel **Agent** calls — one per batch,
  in a single message so they run concurrently.
- Pass each agent: its file list, the chosen docstring convention, the scope/skip
  rules, and the hard rules above.

## Step 3 — Per-Agent Contract

Hand each swarm agent these exact instructions:

```
You own this exact list of files: <files>. Edit ONLY these files.
Convention: <Google|NumPy|reST>. Line width: <N>.

For each file, in order:
1. Read the whole file first. Understand what each unit actually does.
2. Add a module docstring at the very top if missing (purpose + contents).
3. For every class, function, and method (including private, underscore-prefixed,
   nested, and overloads): add a docstring if missing, or repair it if it is
   wrong / outdated / drifted from the current signature or behavior.
   - Summary line, then a body explaining behavior.
   - IN: document every parameter (type, meaning, default, constraints),
     including *args and **kwargs.
   - For **kwargs: Unpacked[SomeClass], open SomeClass, and document each of its
     fields as a keyword argument; note they pass through **kwargs.
   - OUT: document the return / yield type and meaning (unpack tuples/dicts).
   - Raises: every exception and its trigger.
4. NEVER add a standalone string literal after a variable/attribute assignment.
   Put attribute docs in the class/module docstring's Attributes section instead.
5. Change NOTHING except docstrings — no code, signatures, imports, types, or
   reformatting. Do not touch already-correct docstrings.
6. After editing each file, confirm it still parses (python -m py_compile <file>
   or ast.parse). Fix indentation/placement if it doesn't.

Return: per file, the units documented vs updated vs left-as-is, any file that
failed to parse, and any code that looked buggy/ambiguous (describe; do not fix).
```

## Step 4 — Verify

After the swarm finishes:

- **Syntax gate.** Compile every changed file
  (`python -m py_compile $(git diff --name-only)` or an `ast.parse` sweep). A
  parse failure means a misplaced/misindented docstring — fix before reporting.
- **Diff-shape gate.** Confirm the diff is docstring-only: `git diff` should show
  only added/changed string literals in docstring position. Any change to code
  lines is a contract violation — revert it.
- **Reviewer sample.** Spawn one reviewer agent over a random sample of changed
  files to check: docstrings match the actual signature/behavior (no
  hallucinated params/returns), the convention is consistent, `Unpacked[...]`
  kwargs were expanded, and **no variable/attribute string-literal docstrings**
  were introduced. Findings feed a fix-up pass.
- **No-churn check.** Spot-check that already-correct docstrings were left alone.

## Correctness Is A Gate, Not A Step

A run is only acceptable if:

- Every changed file parses, and public modules still import.
- The diff contains no code changes — docstrings only.
- No forbidden variable/attribute string-literal docstrings exist.
- Sampled docstrings are accurate to the implementation, not invented.

## Common Mistakes

- Editing code "while in there" — reformatting, reordering imports, fixing a
  bug. Out of scope; report it instead.
- Inventing parameters or return values instead of reading the body.
- Adding `X = ...` followed by `"""..."""` variable docstrings.
- Leaving `**kwargs: Unpacked[Cfg]` as a single opaque line instead of expanding
  `Cfg`'s fields.
- Mixing Google/NumPy/reST styles, or converting a file's existing style.
- Two agents editing the same file (always partition by file).
- Mass-rewriting correct docstrings, producing a huge noisy diff.
- Misindenting an inserted docstring so the file no longer parses.

## Definition Of Done

- Scope, skip rules, and the chosen docstring convention are recorded.
- Every in-scope module, class, function, and method (public and private) has an
  accurate docstring; outdated ones were repaired.
- `Unpacked[...]` kwargs are expanded field-by-field in IN, reflected in OUT.
- No variable/attribute string-literal docstrings were introduced.
- All changed files parse; the diff is docstring-only; a reviewer sample passed.
- The report lists counts (documented / updated / unchanged), files skipped and
  why, and any buggy/ambiguous code surfaced but not modified.
