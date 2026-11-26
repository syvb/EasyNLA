#!/usr/bin/env bash
# Fifth queue: Taboo organisms (one 8B bottom half per LoRA), after run_after_queue3.sh writes QUEUE4_DONE.
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
  echo "$(date +%T) START $name" >> /tmp/logs/queue5.log
  "$@" > "/tmp/logs/$name.log" 2>&1
  echo "$(date +%T) $name EXIT=$?" >> /tmp/logs/queue5.log
}

while ! grep -q QUEUE4_DONE /tmp/logs/queue4.log 2>/dev/null; do sleep 10; done

step taboo_transfer_robust "$PY" "$R/taboo_transfer.py" --phase0-dir "$OUT/p0_8b_L24_robust" --rollouts "$ROLL" --out "$OUT/taboo_transfer_8b_robust.json"
step taboo_transfer_stock "$PY" "$R/taboo_transfer.py" --phase0-dir "$OUT/p0_8b_L24" --rollouts "$ROLL" --out "$OUT/taboo_transfer_8b_stock.json"
echo "$(date +%T) QUEUE5_DONE" >> /tmp/logs/queue5.log
