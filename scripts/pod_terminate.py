"""Terminate the pod this script runs on (called at the end of a RunPod stage).
Kept as a file so the launcher's `bash -lc '...'` wrapper needs no nested quotes."""
import os
import sys

import runpod

pod_id = os.environ.get("RUNPOD_POD_ID")
key = os.environ.get("RUNPOD_API_KEY")
if not pod_id or not key:
    sys.exit("RUNPOD_POD_ID / RUNPOD_API_KEY not set; not terminating")
runpod.api_key = key
runpod.terminate_pod(pod_id)
print("terminated pod", pod_id)
