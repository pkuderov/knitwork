---
name: paper-method
description: Draft the Method section covering the Grid RNN framework and all 4 HarmonicGridRNN blocks, each using the motivation / design / advantages triad.
purpose: Write the Method section for the Grid RNN AAAI 2026 paper.
source: .claude/skills/paper-method/SKILL.md
---

# Method writing agent

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

## Task — write the Method section (6 subsections)

**Pre-writing (do silently):** for each of the 4 blocks answer: what is this module (one-line definition); workflow (inputs → operations → output); why necessary (what fails without it); why it works (the effective mechanism).

**Structure to write:**

- **3.1 Grid RNN Framework** (overview): setting (input sequence x_t, B batch, T time, H hidden); core structure (L layers × C columns, message passing between columns); state representation h[L, C, B, H]; point to the figure; road-map sentence to the four components.
- **3.2 Spectral LRU — Multi-Scale Temporal Processing** — motivation (single-timescale RNN cannot track fast local and slow global structure together); design (2D spectral grid r_max[layer, col]; layer dim r_base_layer interpolates 0.7→0.999; column dim r_col = r_min_col + (r_base_layer − r_min_col)·col_frac; include the r_max formula as an equation; LRU cell dynamics, complex state h ∈ C^H, real input projection); advantages (each cell specializes at τ = 1/(1−r_max); stable gradient flow via diagonal recurrence).
- **3.3 Surprise-Driven Delta Memory** — motivation (fixed-capacity KV memories overwrite indiscriminately; need selective writing); design as numbered equations: (1) value normalization v = normalize(proj_v(y)); (2) delta rule error = v − W^T k, delta_W += k ⊗ error; (3) EMA surprise m_new = β·m + (1−β)·mean(error²), α = m_new / max(m_new + eps); (4) adaptive forgetting fullness = ‖W‖_F / sqrt(dk·dv), lam = lam_base·fullness; (5) update W_new = (1−lam)·decay·W + α·delta_W / C; advantages (writing rate adapts to surprise; forgetting scales with fullness; per-layer decay 0.95→0.99 gives temporal hierarchy).
- **3.4 Frozen Reservoir (for language modeling)** — motivation (long-range context >200 steps needs stable representations beyond trainable recurrence); design (n_reservoir_cols random Linear RNN columns, fixed weights; spectral radii spread r ∈ {0.9, 0.95, 0.99, 0.999} for τ ∈ {10, 20, 100, 1000}; states always detached; learned linear projection to H; disabled for SDQ and RL); advantages (no gradient interference; diverse timescales at zero training cost).
- **3.5 Hopfield Cross-Column Integration** — motivation (columns process input independently; without integration they cannot form coherent associative patterns); design (Modern Hopfield Network, learnable β per head; queries = current column hidden state; keys/values = all column states + reservoir projections; output gated with learned scalar gate g ∈ [0,1]; col/gate metric tracks utilization); advantages (retrieval energy bounded by Hopfield stability; learnable β allows sharp or soft pattern completion per task).
- **3.6 Training** — optimizer (Adam with LR schedule; specify from config if available); sequence length / rollout_len; parameter count (~2.1M for SDQ/RL variants, ~2.1M with 4 reservoir cols for text).

## Rules
- Every subsection: motivation first, then design, then advantages.
- At least one LaTeX equation per subsection (use the align environment).
- No motivation sentence longer than 2 lines.
- Consistent notation: B=batch, T=time, L=layers, C=columns, H=hidden size.
- Code-derived details must match `grnn_harmonic.py` exactly.

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
