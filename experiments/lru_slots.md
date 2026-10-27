# Slot memory experiments

Model: [grnn_lru_slots.py](../../knitwork/models/grnn_lru_slots.py), described in [GridRNN-LRU slots](../methods/grnn_lru_slots.md). One peripheral column per layer is replaced by 8 addressable key/value slots; column 0 reads first, then decides whether and where to write. Variants: `slot_alloc` (free-slot allocator), `slot_soft` (softmax address, dense control), `slot_entmax` (entmax15 sparse address). Weights: about 0.85M for Text8/SDQ (the slot column replaces an LRU column), versus 1.05M for `dynamic_control`.

## Protocols

| Task | Config | Protocol |
| --- | --- | --- |
| MQAR, easy | `exps/mqar/config/lru_slots_easy/` | train T64/K4 and T128/K8, 8 epochs; test grid: capacity (T=256, K=4..32) and forgetting (K=8, T=64..1024) |
| MQAR, bounded feedback | `lru_slots_easy/{query_read,associative}_bounded.yaml` | same data; normalized private read as a small residual, normalized inter-layer messages (`bounded_feedback`) |
| MQAR, LR control | `dynamic_control` with `--training.learning_rate=3e-4/1e-4 --training.warmup_updates=500` | short control of a smaller LR with warmup |
| Text8 | `exps/text/config/lru_slots_mid/` | mid protocol: 1e9 tokens, 90M/5M/5M split |
| SDQ | `exps/sdq/config/mid.yaml` | about 1M parameters, 1e9 tokens; controls `dynamic_control` and GRU `rnn.L2` |
| RL memory probe | `exps/mikasa/probe.py` | supervised prediction of `obs[t - delay]` on RepeatPreviousHard observations, no RL loss |

## What was observed so far

- MQAR with 8 epochs on the full segment set (T up to 256, K up to 64) did not discriminate variants: train accuracy about 0.20 (only the easiest segment is learned), validation accuracy 0.015-0.027 for `dynamic_control`, `write_gate`, `associative`, `rotations`, `two_reads`; `query_read` did not train (many skipped steps) and `associative` showed gradient norms above 1e13. This motivated the easy protocol, exact skipped-step counters and `bounded_feedback`.
- The first slot prototype wrote softly into all slots and saturated usage; the current version uses a fixed allocator, separate occupancy and age, and exact preservation at zero address weight (see `tests/test_lru_slots.py`).
- Slot results on Text8, SDQ and easy MQAR have not been obtained: the jobs were prepared (`server/enqueue_slots_jobs.py`) but the queue was stopped before they ran. Only short local smoke runs exist.
