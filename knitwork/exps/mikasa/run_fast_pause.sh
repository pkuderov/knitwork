#!/usr/bin/env bash
# Pause all run_fast.py runs: SIGTERM -> checkpoint at the end of the current update, exit code 3.
# Resume by rerunning the same run_fast_queue.sh list.
pids=$(pgrep -f 'python3? -m knitwork[.]exps[.]mikasa[.]run_fast( |$)' | grep -v "^$$\$")
[[ -z "$pids" ]] && { echo "no run_fast runs"; exit 0; }
echo "pausing: $pids"
kill -TERM $pids
