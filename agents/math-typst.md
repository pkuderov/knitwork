---
name: math-typst
description: Generate a math study guide in Typst format (theory + worked examples + practice problems), with a full LaTeX->Typst math conversion reference and Typst 0.13.1 gotchas.
purpose: Produce a compilable Typst math guide. Argument = comma-separated topics (e.g. "limits, series"). Output language follows the request (default English).
source: ~/.claude/commands/math-typst.md (translated from Russian)
---

# Math study guide (Typst) agent

Create a complete math study guide in Typst for the requested topics: **$ARGUMENTS**.

## Structure per topic

For each requested topic, create a section with these mandatory parts:
1. **Theory** — rigorous numbered definitions, theorems with proofs or sketches, explanations of the core idea.
2. **Reference tables** — all formulas needed for the topic.
3. **"How to remember" block** — mnemonics, vivid associations, key patterns.
4. **Worked examples** — at least 3 fully solved examples.
5. **Practice problems** — at least 30 problems with short answers, grouped by subtype.

## Typst syntax (critical: Typst 0.13.1)

### Math — basics
- Inline math: `$x^2 + y^2$` (no spaces next to `$`).
- Display math: `$ x^2 + y^2 $` (spaces next to `$`).
- Grouping in super/subscripts: `x^(2n)`, `a_(n+1)` — **parentheses `()`, not `{}`**.
- Sum: `sum_(n=1)^oo a_n`. Integral: `integral_a^b f(x) dif x`. Limit: `lim_(x -> 0) f(x)`.
- Infinity: `oo` (not `\infty`). Plus-minus: `plus.minus` (not `pm`).
- Fraction: `a/b` or `(a+b)/(c+d)`. Square root: `sqrt(x)`, nth root: `root(n, x)`.
- Down-right arrow: `arrow.br` (not `searrow`).
- **NO LaTeX backslash commands** — none.

### Greek letters (no backslash — they are Typst variables)
```
alpha beta gamma delta epsilon zeta eta theta
iota kappa lambda mu nu xi pi rho
sigma tau upsilon phi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega
```

### Operators and arrows
| LaTeX | Typst | LaTeX | Typst |
|---|---|---|---|
| `\to` | `->` | `\Rightarrow` | `=>` |
| `\leftarrow` | `<-` | `\Leftarrow` | `<=` |
| `\leq`,`\le` | `<=` | `\geq`,`\ge` | `>=` |
| `\neq` | `!=` | `\approx` | `approx` |
| `\cdot` | `dot` | `\times` | `times` |
| `\propto` | `prop` | `\sim` | `tilde` |
| `\ll` | `<<` | `\gg` | `>>` |
| `\implies` | `=>` | `\iff` | `<=>` |
| `\forall` | `forall` | `\exists` | `exists` |
| `\in` | `in` | `\notin` | `in.not` |
| `\subset` | `subset` | `\setminus` | `without` |
| `\odot` | `circle.small` | `\oplus` | `plus.circle` |
| `\infty` | `oo` | `\partial` | `partial` |
| `\nabla` | `nabla` | `\square` | `square` |
| `\log` | `log` | `\argmax` | `op("argmax")` |
| `\argmin` | `op("argmin")` | `\dim` | `dim` |

### Font commands
| LaTeX | Typst |
|---|---|
| `\mathbb{E}` | `bb(E)` |
| `\mathbb{R}` | `bb(R)` |
| `\mathcal{L}` | `cal(L)` |
| `\mathbf{x}` | `bold(x)` |
| `\text{word}` | `"word"` |
| `\hat{x}` | `hat(x)` |
| `\tilde{x}` | `tilde(x)` |
| `\bar{x}` | `bar(x)` |
| `\vec{x}` | `arrow(x)` |

### CRITICAL: spaces between identifiers
In Typst math, adjacent identifiers form a new variable and cause `unknown variable`. Always separate with spaces:
```
WRONG          RIGHT
gammaPhi        gamma Phi
lambdalog(x)    lambda log(x)
2gamma          2 gamma
0.9dot10        0.9 dot 10
sumalphaterm    sum alpha term
alphato0        alpha -> 0
proptoexp       prop exp
argmax          op("argmax")
```
Rule: between any two math atoms (Greek letter, function, number) — a space.

### CRITICAL: non-ASCII letters in math
Non-ASCII (e.g. Cyrillic) symbols outside quotes in math mode cause `unknown variable`. Always quote them:
```
WRONG          RIGHT
cal(H)_цель     cal(H)_"цель"
s_финал         s_"финал"
R_доп(s)        R_"доп"(s)
```

### CRITICAL: multi-letter abbreviations in math
Multi-letter Latin abbreviations (PPO, DQN, GAE, CLIP, SAC...) parse as a product of letters and may error. Always quote:
```
WRONG          RIGHT
cal(L)^PPO      cal(L)^"PPO"
cal(L)^CLIP     cal(L)^"CLIP"
J^GAE           J^"GAE"
```

### CRITICAL: semicolon in function arguments
In Typst math, `;` inside `(...)` creates a **matrix row separator**, not an argument separator:
```
WRONG (syntax error)
min(a,; b)
bb(E)_(s tilde D,; epsilon tilde N)

RIGHT
min(a, quad b)
bb(E)_(s tilde D, epsilon tilde N)
```

### CRITICAL: `*` in body text
In Typst `*text*` is bold. For a literal `*` (e.g. the A* algorithm) use another notation:
```
WRONG (opens an unclosed bold)
the A* algorithm is well known

RIGHT
the A-star algorithm is well known
the $A^*$ algorithm is well known   <- in math mode * is fine
```

### CRITICAL: duplicate named arguments
A Typst function cannot take two identical named arguments:
```
WRONG (error "duplicate argument")
line((0,0),(1,1), stroke: blue + 1pt, stroke: (dash: "dashed") + blue)

RIGHT — a single stroke:
line((0,0),(1,1), stroke: blue + 1pt)
line((0,0),(1,1), stroke: (paint: blue, thickness: 1pt, dash: "dashed"))
```

### Dashed lines in cetz
The combination `(dash: "dashed") + color + pt` is a **syntax error**. Instead use just a different color/width, or the named stroke form; there is no simple `+ dash` in cetz.

### Counters and environments
- `counter("name").step()` — just steps, no context needed.
- `#context counter("name").display()` — **must** use `context` when displaying; without it → `can only be used when context is known`.

Page-number-aware header:
```typst
header: context {
  let p = counter(page).get()
  if p != (1,) and p != (2,) [
    #text(size: 8.5pt, fill: luma(140))[Title #h(1fr) #counter(page).display()]
  ]
}
```
`counter(page).get()` returns an **array** `(n,)`; compare with `(1,)`, not `1`.

Define environments at the top of the file (theorem, definition, example, "how to remember" memo, proof, solution, practice-problem helpers) using `block(...)` with colored left borders and stepped counters. Use `#context <counter>.display()` inside labels.

### Packages
The only available local package is `@preview/cetz:0.3.4` for illustrations:
```typst
#import "@preview/cetz:0.3.4": canvas, draw
#canvas({
  import draw: *
  line((0,0),(4,0), mark: (end: ">", size: 0.2))
  circle((2,0), radius: 0.4)
  content((2,0), $s$)
})
```
cetz primitives: `line(a, b)`, `circle(center, radius: r)`, `arc(pos, start: Xdeg, stop: Ydeg, radius: r, mode: "OPEN"/"PIE"/"CLOSE")`, `bezier(...)`, `catmull(...)`, `content(pos, body)`. Line marks: `mark: (end: ">", size: 0.2)`. In `content(pos, body)`, wrap math in `$...$`.

## Mandatory verification of examples and problems

**Before writing the file**, go through every worked example and every practice problem:
1. **Compute the answer yourself** — don't trust the draft; plug in concrete numbers, expand the steps.
2. **Check consistency** — the answer in the problem list must match any separate answer section.
3. **Drop bad problems** — if an example has no clean analytic answer, replace it with a nicer one.

### Solution detail by difficulty
| Level | Signs | What to write |
|---|---|---|
| **Basic** | 1-2 arithmetic steps, plug into a formula | result + one line of computation |
| **Medium** | 2-4 steps, one theorem/formula | key intermediate steps |
| **Advanced** | several theorems, change of variables, multi-step | full step-by-step with each transition explained |
| **Hard** | non-trivial trick, convergence, case analysis | full solution + highlight the key idea (memo/italic) |

Rule: a short solution for an easy problem is the norm, not a defect. A bloated easy solution misleads; a hard problem without the key transition is useless.

## Output requirements
1. Write the full `.typ` file. Name: from the first topic, ASCII; if several topics — `math_guide.typ`.
2. Use the Write tool, into the current working directory.
3. Compile: `~/.local/bin/typst compile <name>.typ <name>.pdf`.
4. On failure, analyze errors line by line, fix with Edit, recompile. Common errors → fixes:

   | Error | Cause | Fix |
   |---|---|---|
   | `unknown variable: gammaPhi` | fused identifiers | add space: `gamma Phi` |
   | `unknown variable: цель` | non-ASCII in math without quotes | `_"цель"` |
   | `unknown variable: PPO` | abbreviation in math | `^"PPO"` |
   | `unknown variable: argmax` | not wrapped in `op()` | `op("argmax")` |
   | `unknown variable: pm` | LaTeX command | `plus.minus` |
   | `unknown variable: to` | LaTeX command | `->` |
   | `unknown variable: implies` | LaTeX command | `=>` |
   | `unknown variable: ll` | LaTeX command | `<<` |
   | `unknown variable: sim` | LaTeX command | `tilde` |
   | `unknown variable: ldots` | LaTeX command | `...` |
   | `unknown variable: searrow` | LaTeX command | `arrow.br` |
   | `unclosed delimiter` | `*` in text or unmatched `{` | find the unclosed `*`/brace |
   | `expected comma` | `;` inside `f(a,; b)` | drop `;`, use `,` |
   | `duplicate argument: stroke` | two `stroke:` in one call | keep one |
   | `can only be used when context is known` | `counter.display()` without `context` | add `context` |
   | `unexpected argument` | helper got too many blocks | merge extras into one |

5. Report to the user: the PDF path and page count.

## Document style
- Language: follow the request (default English); keep math rigorous (definition → theorem → corollary).
- No emojis. Compact answers (one line). Vivid "how to remember" blocks with concrete tricks.
- Always keep spaces in math: `gamma dot Phi(s')`, not `gammadotPhi(s')`.
