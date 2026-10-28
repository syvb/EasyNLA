#!/usr/bin/env bash
# Chat viewer, pod A of 2: chats ipv4_code, lighthouse_story. Runs heads_l24_chatviewer_remote.sh as a child process (not exec, so the
# launcher's liveness check, which looks for this script's name, keeps seeing it). Launch like the main script.
set -euo pipefail
CHATS=ipv4_code,lighthouse_story bash /workspace/EasyNLA/scripts/runpod/heads_l24_chatviewer_remote.sh
