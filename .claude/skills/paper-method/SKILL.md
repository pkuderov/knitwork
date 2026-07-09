---
name: paper-method
description: Use this skill when the user types "/paper-method" or asks to write the Method section of the AAAI paper. Triggers on "напиши method section", "draft method", "написать методы", "write method".
version: 1.0.0
---

# Paper-Method Skill

Drafts the Method section covering the Grid RNN framework and all 4 HarmonicGridRNN blocks.
Each module uses the motivation / design / technical advantages triad.

## Steps

1. Spawn a **foreground** sub-agent with the prompt below
2. Report the section to the user

## Sub-agent prompt

```
You are an academic writing agent for an AAAI 2026 paper about Grid RNN approaches.

## Files to read (read ALL before writing)

Architecture source:
- `knitwork/models/grnn.py` — base Grid RNN: MessagePassingLayer, GridRnn class, column/layer structure
- `knitwork/models/grnn_harmonic.py` — HarmonicGridRNN: SurpriseDeltaMemory, FrozenReservoir, HarmonicGridRNN
- `docs/methods/grnn.md` — base Grid RNN description
- `docs/methods/grnn_harmonic.md` — all 4 blocks, hyperparameters, state representation

Writing guides:
- `agents/references/method.md` — pre-writing questions, three-element structure, section skeleton
- `agents/references/examples/method/overview-template.md`
- `agents/references/examples/method/module-triad-neural-body.md`
- `agents/references/examples/method/pre-writing-questions.md`

## Task

Write the Method section with 6 subsections.

### Pre-writing (do silently):
For each of the 4 blocks, answer:
1. What is this module? (one-line definition)
2. Workflow: inputs → operations → output
3. Why necessary? (what fails without it)
4. Why does it work? (the mechanism that makes it effective)

### Structure to write:

**3.1 Grid RNN Framework** (Overview subsection)
- Setting: input sequence x_t, B batch, T time, H hidden
- Core structure: L layers × C columns, message passing between columns
- State representation: h[L, C, B, H]
- Point to the figure: "Figure 1 illustrates..."
- Road-map sentence: "The following subsections describe the four components of HarmonicGridRNN."

**3.2 Spectral LRU — Multi-Scale Temporal Processing**
Motivation: single timescale RNN cannot simultaneously track fast local patterns and slow global structure.
Design:
  - 2D spectral grid: r_max[layer, col] ∈ [r_min_col, r_max_layers]
  - Layer dimension: r_base_layer interpolates from r_min_layers=0.7 to r_max_layers=0.999
  - Column dimension: r_col = r_min_col + (r_base_layer - r_min_col) * col_frac
  - Include the r_max formula as an equation
  - LRU cell dynamics: complex-valued state h ∈ C^H, real input projection
Technical advantages: each (layer, col) cell specializes at a different timescale τ = 1/(1-r_max); gradient flow is stable via LRU's diagonal recurrence.

**3.3 Surprise-Driven Delta Memory**
Motivation: fixed-capacity key-value memories overwrite useful information indiscriminately; we need selective writing.
Design (write as numbered equations):
  1. Value normalization: v = normalize(proj_v(y))  — bounds delta_W norm
  2. Delta rule: error = v - W^T k;  delta_W += k ⊗ error
  3. EMA surprise: m_new = beta*m + (1-beta)*mean(error²);  alpha = m_new / max(m_new + eps)
  4. Adaptive forgetting: fullness = ||W||_F / sqrt(dk*dv);  lam = lam_base * fullness
  5. Update: W_new = (1-lam) * decay * W + alpha * delta_W / C
Technical advantages: writing rate adapts to surprise level; forgetting scales with memory fullness; per-layer decay (0.95→0.99) creates temporal hierarchy in memory.

**3.4 Frozen Reservoir (for language modeling)**
Motivation: long-range context (>200 steps) requires stable representations beyond what trainable recurrence can maintain.
Design:
  - n_reservoir_cols random Linear RNN columns with fixed weights
  - Spectral radii spread: r ∈ {0.9, 0.95, 0.99, 0.999} for τ ∈ {10, 20, 100, 1000}
  - States always detached; projections to H via learned linear layer
  - Disabled (n_reservoir_cols=0) for SDQ and RL tasks
Technical advantages: no gradient interference with trainable columns; diverse timescales at zero training cost.

**3.5 Hopfield Cross-Column Integration**
Motivation: columns within a layer process input independently; without integration, they cannot form coherent associative patterns.
Design:
  - Modern Hopfield Network with learnable beta per head
  - Queries: current column hidden state; Keys/Values: all column states + reservoir projections
  - Output gated with learned scalar gate g ∈ [0,1]
  - col/gate metric tracks utilization per layer
Technical advantages: retrieval energy is bounded by Hopfield stability; learnable beta allows sharp or soft pattern completion per task.

**3.6 Training**
- Optimizer: Adam with learning rate schedule (specify from config if available)
- Sequence length / rollout_len
- Parameter count: ~2.1M for SDQ/RL variants, ~2.1M with 4 reservoir cols for text

### Rules
- Every subsection: motivation first, then design, then advantages
- Include at least one LaTeX equation per subsection (use align environment)
- No motivation sentences longer than 2 lines
- Use consistent notation: B=batch, T=time, L=layers, C=columns, H=hidden size
- Code-derived details must match grnn_harmonic.py exactly

## Output format

```latex
\section{Method}

\subsection{Grid RNN Framework}
...

\subsection{Spectral LRU}
...

\subsection{Surprise-Driven Delta Memory}
...

\subsection{Frozen Reservoir}
...

\subsection{Hopfield Cross-Column Integration}
...

\subsection{Training Details}
...
```
```
