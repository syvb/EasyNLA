#!/usr/bin/env bash
# Fourth queue: light analyses on cached corpus activations, after run_after_queue2.sh writes QUEUE3_DONE.
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
  echo "$(date +%T) START $name" >> /tmp/logs/queue4.log
  "$@" > "/tmp/logs/$name.log" 2>&1
  echo "$(date +%T) $name EXIT=$?" >> /tmp/logs/queue4.log
}

while ! grep -q QUEUE3_DONE /tmp/logs/queue3.log 2>/dev/null; do sleep 10; done

step corpus_behaviour_positions "$PY" "$R/corpus_behaviour_positions.py" --phase0-dir "$P0" --phase0-dir "$P0R" --rollouts "$ROLL" --out "$OUT/corpus_behaviour_positions_8b.json"
step top_direction_stock "$PY" "$R/top_direction.py" --phase0-dir "$P0" --rollouts "$ROLL" --state-cache "$OUT/behaviour_state_8b.npz" --out "$OUT/top_direction_8b_stock.json"
step top_direction_robust "$PY" "$R/top_direction.py" --phase0-dir "$P0R" --rollouts "$ROLL" --state-cache "$OUT/behaviour_state_8b.npz" --out "$OUT/top_direction_8b_robust.json"
step gate_positions_06b "$PY" "$R/gate_positions.py" --phase0-dir "$OUT/p0_06b_L18" --n-boot 200 --out "$OUT/gate_positions_06b.json"
# Whole-model 8B passes again, this time capturing the last user-content token at every layer.
step layer_sweep_user_last "$PY" "$R/layer_sweep.py" --position user_last --cache-dir /home/user/act_cache/sweep --out "$OUT/layer_sweep_8b_user_last.json"
echo "$(date +%T) QUEUE4_DONE" >> /tmp/logs/queue4.log
