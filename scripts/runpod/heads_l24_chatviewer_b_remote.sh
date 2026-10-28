#!/usr/bin/env bash
# Chat viewer, pod B of 2: chats train_problem, ada_lovelace. Runs heads_l24_chatviewer_remote.sh as a child process (not exec, so the
# launcher's liveness check, which looks for this script's name, keeps seeing it). Launch like the main script.
set -euo pipefail
CHATS=train_problem,ada_lovelace bash /workspace/EasyNLA/scripts/runpod/heads_l24_chatviewer_remote.sh
