#!/usr/bin/env bash
# Queue of 8B analyses that need the whole machine, run back to back once the rollouts exist.
# Each step logs to /tmp/logs/<step>.log and appends "<step> EXIT=<code>" to /tmp/logs/queue.log.
set -u
REPO=/home/user/workspace/repo
R=$REPO/.amp/in/orb-backup/cpu_research
PY=/tmp/venv/bin/python
OUT=/home/user/research
P0=$OUT/p0_8b_L24
P0R=$OUT/p0_8b_L24_robust
ROLL=$OUT/p2_8b/corpus/rollouts.parquet
MAN=$OUT/manifests/corpus_taboo_8b.jsonl
export NLA_ACT_CACHE=/home/user/act_cache
cd "$REPO" || exit 1
mkdir -p /tmp/logs "$OUT/manifests"

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date +%T) START $name" >> /tmp/logs/queue.log
  "$@" > "/tmp/logs/$name.log" 2>&1
  echo "$(date +%T) $name EXIT=$?" >> /tmp/logs/queue.log
}

while [ ! -f "$ROLL" ] || pgrep -f "phase2_build_corpus.py --step rollouts" > /dev/null; do sleep 10; done

rm -f "$MAN"
NLA_RECORD=$MAN NLA_THREADS=2 step record_corpus "$PY" "$R/corpus_geometry.py" --phase0-dir "$P0" --rollouts "$ROLL" --out /tmp/unused.json
NLA_RECORD=$MAN NLA_THREADS=2 step record_taboo "$PY" "$R/taboo_mode_check.py" --phase0-dir "$P0" --out /tmp/unused.json
step warm_corpus_taboo "$PY" "$R/warm_cache.py" "$MAN"
step taboo_mode_check "$PY" "$R/taboo_mode_check.py" --phase0-dir "$P0" --out "$OUT/taboo_mode_check_8b.json"
step corpus_geometry_stock "$PY" "$R/corpus_geometry.py" --phase0-dir "$P0" --rollouts "$ROLL" --out "$OUT/corpus_geometry_8b_stock.json"
step corpus_geometry_robust "$PY" "$R/corpus_geometry.py" --phase0-dir "$P0R" --rollouts "$ROLL" --out "$OUT/corpus_geometry_8b_robust.json"
step layer_sweep "$PY" "$R/layer_sweep.py" --cache-dir /home/user/act_cache/sweep --out "$OUT/layer_sweep_8b.json"
echo "$(date +%T) QUEUE_DONE" >> /tmp/logs/queue.log
