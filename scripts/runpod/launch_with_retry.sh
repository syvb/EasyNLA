#!/usr/bin/env bash
# Launch a head-NLA remote script on RunPod SECURE cloud, retrying across GPU types while stock is
# thin. Each attempt is one full launch_heads_quick_sft.py run; an attempt that fails with "no
# instances available" created nothing, and the next GPU type (then the next round, after 2 min)
# is tried. Any other outcome (done, failed, timed out) ends the loop with the launcher's exit code.
#
#   scripts/runpod/launch_with_retry.sh <out dir> <pod name> <launcher args...>
#
# e.g. scripts/runpod/launch_with_retry.sh /home/syvb/l24_rl heads-l24-rl \
#        --remote-script scripts/runpod/heads_l24_rl_remote.sh --secrets hf,wandb \
#        --fetch results --no-checkpoints --max-hours 8
# Run it detached (setsid nohup ... &) and find it later with: pgrep -f launch_with_retry.sh
# GPU order and secure price bands: H100 SXM ($3.99), then A100 SXM / PCIe 80 GB ($1.79; slower).
# Set GPUS="NVIDIA H100 80GB HBM3|3.5|4.5" to allow only H100s.
set -uo pipefail
OUT=$1; NAME=$2; shift 2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
GPUS=${GPUS:-"NVIDIA H100 80GB HBM3|3.5|4.5;NVIDIA A100-SXM4-80GB|1.5|2.2;NVIDIA A100 80GB PCIe|1.5|2.2"}
mkdir -p "$OUT"
cd "$REPO"
for round in $(seq 1 15); do
  IFS=';' read -ra specs <<<"$GPUS"
  for spec in "${specs[@]}"; do
    IFS='|' read -r gpu lo hi <<<"$spec"
    echo "=== round $round: $gpu ($(date -u +%H:%M:%S)) ==="
    python scripts/runpod/launch_heads_quick_sft.py "$@" --gpu-id "$gpu" \
      --min-cost-per-hr "$lo" --max-cost-per-hr "$hi" --out "$OUT" --name "$NAME" > "$OUT/attempt.log" 2>&1
    rc=$?
    cat "$OUT/attempt.log" >> "$OUT/launcher.log"
    if ! grep -q "no longer any instances available\|no instances" "$OUT/attempt.log"; then
      echo "=== launcher finished (rc $rc) on $gpu ==="
      exit "$rc"
    fi
  done
  sleep 120
done
echo "=== gave up: no secure GPU in 15 rounds ==="
exit 1
