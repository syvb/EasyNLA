#!/usr/bin/env bash
# Control for the matryoshka run: the same step-400 resume continued with the std objective.
# See heads_l24_mtry_remote.sh (launch the same way, naming this script).
MODE=control exec bash "$(dirname "$0")/heads_l24_mtry_remote.sh"
