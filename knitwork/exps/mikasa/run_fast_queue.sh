#!/usr/bin/env bash
# Sequential run_fast.py queue under hard resource caps (cgroup via systemd-run).
# usage: run_fast_queue.sh <list> [log_dir]; each list line: <run_id> <overrides...>
# Rerunning a list resumes unfinished runs from their checkpoints and skips finished ones.
# Pause: pkill -TERM -f knitwork.exps.mikasa.run_fast -> checkpoint at the end of the current update,
#   the run exits with code 3 and the queue stops.
# Caps: RUN_MEM (default 3G RAM, no swap), RUN_CPU (default 200% = 2 cores), nice 10.
set -u
list=$1
log_dir=${2:-logs/run_fast}
mkdir -p "$log_dir"
cd "$(dirname "$0")/../../.."
while read -r run_id args; do
  [[ -z "$run_id" || "$run_id" == \#* ]] && continue
  echo "=== $(date +%H:%M) start $run_id"
  eval systemd-run --user --scope --quiet \
    -p MemoryMax="${RUN_MEM:-3G}" -p MemorySwapMax=0 -p CPUQuota="${RUN_CPU:-200%}" \
    env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
    uv run python -m knitwork.exps.mikasa.run_fast knitwork/exps/mikasa/config/fast/base.yaml $args --run_id="$run_id" \
    >> "$log_dir/$run_id.log" 2>&1
  rc=$?
  echo "=== $(date +%H:%M) done $run_id rc=$rc"
  if (( rc == 3 )); then echo "=== paused, rerun the same list to resume"; break; fi
done < "$list"
