# Multimodal SDQ — design proposal (encoder + SDQ superstructure, random modality)

> Status: **proposal / design note**, no implementation yet. It analyzes the
> current multimodal benchmark, contrasts it with SDQ, and proposes a refactor
> into two decoupled parts — a modality **encoder** and an **SDQ superstructure** —
> where the input modality is chosen **randomly** per sample/sequence.

## 1. Problem

The current multimodal benchmark (`knitwork/gens/multimodal_sum.py`,
`knitwork/models/grnn_multimodal.py`, `..._v2.py`) hardwires **two modalities
present simultaneously every step** (MNIST image + FSDD audio), fused by three
inlined `nn.Linear` projections. This entangles two concerns that we would like
to study separately:

1. **Perceptual encoding** — turning a raw modality feature into a hidden token.
2. **Associative reasoning** — storing, ignoring distractors, and querying.

We want the second concern to be the well-understood **SDQ** task
(`knitwork/gens/sdq.py`), and the first to be a swappable **encoder**, so that a
symbol can arrive as *any* modality, chosen at random. This mirrors how humans
store a concept regardless of whether it was seen or heard.

## 2. Current structure vs SDQ

| Aspect | `multimodal_sum` | `sdq` |
|---|---|---|
| Input per step | `image_feat` + `audio_feat` + `buffer_feat` tensors, both modalities always present | a single integer token id |
| Model contract | `forward(image_feat, audio_feat, buffer_feat, …)`, dims injected as `image_feat_dim`/`audio_feat_dim` | `forward(tokens)`, `input_size=n_tokens` |
| "Encoder" | three `nn.Linear` projections inlined in `_prepare_grid_input` | `nn.Embedding` inlined in `forward` |
| Reasoning/head | grid GRU + message passing, multi-column head | same grid, column-0 head |
| Task logic | running digit sum, query-gated | store / distract / query, key→value |

**Key observation.** The encoder/reasoning seam already physically exists: in
`GridRnnMultimodal._prepare_grid_input(image_feat, audio_feat, buffer_feat)` the
raw features become per-column `embedding_size` vectors, and from that point on
the grid is identical to the base `GridRnn`. The base `GridRnn` consumes an
embedded token in exactly the same shape. So the abstraction boundary is a
`[batch, embedding_size]`-per-column contract — we just need to make it explicit.

## 3. Proposed two-part architecture

### 3.1 Modality encoder (part A)

Extract the three `*_proj` linears into a standalone module dispatched by a new
`modality_id`:

```python
class ModalityEncoder(nn.Module):
    # encoders: modality_id -> (raw feature dim) -> embedding_size
    def __init__(self, *, modality_dims: dict[int, int], embedding_size: int):
        super().__init__()
        self.enc = nn.ModuleDict({
            str(m): nn.Linear(d, embedding_size) for m, d in modality_dims.items()
        })

    def forward(self, feature, modality_id):  # feature [B, Dm], modality_id [B]
        out = feature.new_zeros(feature.shape[0], self.embedding_size)
        for m, enc in self.enc.items():
            mask = modality_id == int(m)
            if mask.any():
                out[mask] = enc(feature[mask, : enc.in_features])
        return out  # [B, embedding_size]
```

- Contract: `(raw feature, modality_id) -> [B, embedding_size]`, i.e. a token
  analogue, exactly what the grid ("superstructure") already expects.
- Today each encoder is a `Linear` (seeded from `image_proj`/`audio_proj`); later
  it can become a CNN for images or a small MLP/1-D conv for audio **without
  touching the grid**.
- Different modalities have different raw dims (`image_dim` vs `audio_dim`), so
  the encoder is keyed by `modality_id` rather than padding to a common dim.

### 3.2 SDQ superstructure (part B)

Keep SDQ's store/distract/query **target logic** unchanged — `handle_store`,
`handle_distract`, `handle_query`, the key→value bookkeeping, and `sq_gaps` in
`StoreDistractQueryGenerator`. The only change: the *symbol* being stored/queried
is delivered as an encoded modality token instead of a raw integer id. The grid
itself (`MessagePassingLayer`, and `MaskedMessagePassingLayer` from
`grnn_multimodal_v2.py` for distractor-column masking) is reused as-is; the head
uses the SDQ column-0 (or multi-column) readout.

### 3.3 Random modality selection

**Generator side** (a new `MultimodalSDQGenerator` reusing `_ModalityBank` from
`multimodal_sum.py` + the SDQ phase machinery from `sdq.py`):

```python
# per store/query event, pick a modality at random
modality = self.rng.integers(0, self.n_modalities, size=n_envs)   # [B]
feature  = self._banks[modality].sample(symbol_digits)            # [B, Dm]
# emit (feature, modality_id) instead of a token id
```

Offer **two granularities** (a config flag `modality_granularity: sample|episode`):
- *per-sample*: every store/query event picks its own modality — hardest, forces
  modality-invariant storage (a symbol may be stored as image, queried as audio).
- *per-episode/sequence*: one modality per episode — easier, isolates the encoder.

**Model side.** `forward(feature, modality_id, …)` runs the `ModalityEncoder`
then the grid. Reuse SDQ's `model_forward` tuple-normalizer (`run_sdq.py`) in a
merged runner, keeping the multimodal diagnostics (`run_ablation_eval`, CKA,
sum-probe from `knitwork/exps/multimodal_sum/_viz.py`).

## 4. Design variants (pick per experiment)

| Variant | Modality choice | Encoder | Trade-off |
|---|---|---|---|
| **V1 — per-episode, per-modality encoder** (recommended first) | one modality per episode | `ModuleDict` keyed by id | easiest to train; cleanly isolates encoder quality from reasoning |
| **V2 — per-sample, per-modality encoder** | random per store/query event | `ModuleDict` keyed by id | tests true modality-invariant associative memory; matches "random input modality" most literally |
| **V3 — per-sample, shared latent encoder** | random per event | single shared encoder after per-modality adapters (Perceiver-style) | forces a modality-agnostic latent; more parameters shared, closer to "any-to-any" models |

Recommendation: implement **V1** first (validates the split with minimal risk),
then flip the granularity flag to get **V2** (the target "random modality" regime),
and keep **V3** as a stretch once the encoder abstraction is stable.

## 5. Analogs / prior art to cite

- **AV-MNIST** — degraded audio-visual digit classification; already the design of
  the feature cache (`prepare_mdsum_cache.py` deliberately lossy PCA-64 / pooled
  log-spectrogram).
- **Perceiver / Perceiver-IO** — a modality-agnostic encoder mapping arbitrary
  inputs into a shared latent; the blueprint for variant V3.
- **Any-to-any / mixture-of-modality-experts** (e.g. VATT, data2vec-style shared
  encoders) — per-modality adapters into a shared reasoning core, matching the
  encoder/superstructure split.
- **Associative-memory RNNs** (Fast Weights, Modern Hopfield) — the reasoning core
  already present in the grid; the SDQ superstructure is the benchmark that probes it.

## 6. What changes, concretely

- **New** `knitwork/gens/multimodal_sdq.py` — SDQ phase logic + `_ModalityBank`
  feature emission + random `modality_id` (granularity flag).
- **New** `knitwork/models/` encoder module (`ModalityEncoder`) + a thin model that
  composes it with the existing grid superstructure.
- **New** `knitwork/exps/multimodal_sdq/` runner reusing `run_sdq.py`'s
  `model_forward` normalizer and the `_viz.py` diagnostics.
- **Reused unchanged:** `MessagePassingLayer`/`MaskedMessagePassingLayer`,
  `StoreDistractQueryGenerator` phase handlers, `_ModalityBank`, `CKAVisualizer`.
- **Docs:** add `docs/methods/grnn_multimodal_sdq.md` alongside the new model
  (per the docs rule), and link both this note and that doc in `docs/_sidebar.md`.

## 7. Open questions

- Buffer/distractor columns: keep the pure-noise buffer columns from the current
  design, or replace distraction with SDQ's token-level distractors? (Proposal:
  use SDQ distractors delivered as random-modality features, dropping the noise
  buffer.)
- Query modality signalling: should the query indicate which modality to expect,
  or must the model be modality-agnostic at retrieval? (V2 assumes agnostic.)
- Output space: keep the digit-sum head, or switch to SDQ's key→value classification
  head. (Proposal: SDQ value classification, so the metric matches SDQ Acc/query.)
