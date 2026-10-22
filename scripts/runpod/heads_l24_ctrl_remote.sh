#!/usr/bin/env bash
# Control for the matryoshka run: the same step-400 resume continued with the std objective.
# See heads_l24_mtry_remote.sh (launch the same way, naming this script). Not `exec`: the launcher's
# liveness check pgreps for THIS script's name, so this process must stay alive for the whole run.
MODE=control bash "$(dirname "$0")/heads_l24_mtry_remote.sh"
