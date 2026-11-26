#!/usr/bin/env bash
# Third queue: starts once run_after_queue.sh has written QUEUE2_DONE.
set -u
REPO=/home/user/workspace/repo
R=$REPO/.amp/in/orb-backup/cpu_research
PY=/tmp/venv/bin/python
OUT=/home/user/research
ROLL=$OUT/p2_8b/corpus/rollouts.parquet
export NLA_ACT_CACHE=/home/user/act_cache
cd "$REPO" || exit 1

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date +%T) START $name" >> /tmp/logs/queue3.log
  "$@" > "/tmp/logs/$name.log" 2>&1
  echo "$(date +%T) $name EXIT=$?" >> /tmp/logs/queue3.log
}

while ! grep -q QUEUE2_DONE /tmp/logs/queue2.log 2>/dev/null; do sleep 10; done

step cross_patch_8b "$PY" "$R/cross_patch.py" --phase0-dir "$OUT/p0_8b_L24" --rollouts "$ROLL" --out "$OUT/cross_patch_8b.json"
step cross_patch_06b "$PY" "$R/cross_patch.py" --phase0-dir "$OUT/p0_06b_L18" --rollouts "$ROLL" --out "$OUT/cross_patch_06b.json"
echo "$(date +%T) QUEUE3_DONE" >> /tmp/logs/queue3.log
