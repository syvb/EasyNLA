#!/usr/bin/env bash
# Second queue: starts once run_after_rollouts.sh has written QUEUE_DONE.
# Both steps read the corpus activations that queue cached; behaviour_link also loads the
# top half of each 8B model (11 GB), so nothing else heavy may run beside it.
set -u
REPO=/home/user/workspace/repo
R=$REPO/.amp/in/orb-backup/cpu_research
PY=/tmp/venv/bin/python
OUT=/home/user/research
P0=$OUT/p0_8b_L24
P0R=$OUT/p0_8b_L24_robust
ROLL=$OUT/p2_8b/corpus/rollouts.parquet
export NLA_ACT_CACHE=/home/user/act_cache
cd "$REPO" || exit 1

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date +%T) START $name" >> /tmp/logs/queue2.log
  "$@" > "/tmp/logs/$name.log" 2>&1
  echo "$(date +%T) $name EXIT=$?" >> /tmp/logs/queue2.log
}

while ! grep -q QUEUE_DONE /tmp/logs/queue.log 2>/dev/null; do sleep 10; done

step alignment_variants "$PY" "$R/alignment_variants.py" --phase0-dir "$P0" --rollouts "$ROLL" --out "$OUT/alignment_variants_8b.json"
step behaviour_link_stock "$PY" "$R/behaviour_link.py" --phase0-dir "$P0" --rollouts "$ROLL" --state-cache "$OUT/behaviour_state_8b.npz" --out "$OUT/behaviour_link_8b_stock.json"
step behaviour_link_robust "$PY" "$R/behaviour_link.py" --phase0-dir "$P0R" --rollouts "$ROLL" --state-cache "$OUT/behaviour_state_8b.npz" --out "$OUT/behaviour_link_8b_robust.json"
# Cross-scale check on the plan's pilot pair. The rollouts are Qwen3-8B's, teacher-forced through 0.6B.
P06=$OUT/p0_06b_L18
step alignment_variants_06b "$PY" "$R/alignment_variants.py" --phase0-dir "$P06" --rollouts "$ROLL" --out "$OUT/alignment_variants_06b.json"
step behaviour_link_06b "$PY" "$R/behaviour_link.py" --phase0-dir "$P06" --rollouts "$ROLL" --out "$OUT/behaviour_link_06b.json"
echo "$(date +%T) QUEUE2_DONE" >> /tmp/logs/queue2.log
