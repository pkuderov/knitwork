#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. One invocation launches exactly ten runs.
configs=(knitwork/exps/mikasa/config/rl_shortlist_mid/[0-9][0-9]_*.yaml)
if [[ ${#configs[@]} -ne 10 ]]; then
  echo "Expected exactly ten shortlist YAML files; run from the repository root." >&2
  exit 1
fi
budget=${KNITWORK_RL_STEPS:-8388608}
device=${KNITWORK_RL_DEVICE:-cuda:0}
if [[ ! "$budget" =~ ^[0-9]+$ ]] || (( budget <= 262144 || budget % 4096 != 0 )); then
  echo "KNITWORK_RL_STEPS must be an integer multiple of 4096 greater than warmup (262144)." >&2
  exit 1
fi
for config in "${configs[@]}"; do
  run_id=${config##*/}
  run_id=${run_id%.yaml}
  uv run python -m knitwork.exps.mikasa.run "$config" \
    --device="$device" --n_steps="$budget" \
    --name="rl_mid_shortlist/${run_id}/${budget}slots/seed42"
done
