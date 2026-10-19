"""Launch a head-NLA job on one RunPod SECURE-cloud H100, wait, fetch results, delete the pod.

    python scripts/runpod/launch_heads_quick_sft.py --out <local results dir>          # PAID
    python scripts/runpod/launch_heads_quick_sft.py --out <dir> --dry-run              # prints the plan
    python scripts/runpod/launch_heads_quick_sft.py --out <dir> --hold \
        --remote-script scripts/runpod/heads_soft_token_remote.sh --fetch results --secrets hf

--hold keeps the pod after the remote script finishes, for interactive work over SSH (the
target is written to <out>/pod_ssh.txt), until <out>/RELEASE exists or --max-hours is up;
then the results are fetched and the pod deleted as usual.

Cost safety, in order:
  - secure cloud only: --cloud-type SECURE is always passed and there is no option for any
    other cloud type; after creation the pod record is re-read and the pod is deleted at once
    unless its cloudType is SECURE and its hourly cost is in the secure-H100 price band
    (assert_secure)
  - the pod is deleted on every exit path of this process (finally + SIGTERM handler),
    after results are pulled when the run got that far
  - a local wall-clock limit (--max-hours) ends the wait and deletes the pod
  - an on-pod watchdog deletes the pod itself after --max-hours + 0.5 h if this process
    died and never cleaned up (needs the pod's own runpodctl; checked and reported)

Secrets (--secrets: HF token, Anthropic key) go over SSH into /workspace/.secrets (mode 600),
not into the pod's environment config. The remote work is --remote-script (default
scripts/runpod/heads_quick_sft_remote.sh).
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPU_ID = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
SSH_KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"


def parse_json(text: str) -> dict | list | str:
    """First JSON value in `text` (progress lines may precede it), else the text itself."""
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return json.JSONDecoder().raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    return text


def rp(*args: str, check: bool = True) -> dict | list | str:
    out = subprocess.run(["runpodctl", *args], capture_output=True, text=True, check=False)
    if check and out.returncode != 0:
        raise RuntimeError(f"runpodctl {' '.join(args)} failed ({out.returncode}): {out.stderr or out.stdout}")
    return parse_json(out.stdout)


def walk(obj, path=""):
    """Yield (dotted path, leaf value) for every leaf of a JSON-ish object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def assert_secure(pod: dict, min_cost: float, max_cost: float) -> float:
    """Secure cloud, by two independent signals, or AssertionError (the caller deletes the pod).

    - every `cloudType` field in the record must be SECURE (nested GPU-type info carries
      `communityCloud: true` / `communityPrice` even for secure pods, so those keys are
      deliberately NOT read)
    - the pod's own top-level hourly cost must sit in [min_cost, max_cost]: an H100 SXM is
      $3.99/hr secure vs $2.69/hr community, so a community pod fails this even if the
      record had no cloudType at all
    """
    leaves = list(walk(pod))
    cloud = [(p, v) for p, v in leaves if p.split(".")[-1].lower() == "cloudtype"]
    assert all(str(v).upper() == "SECURE" for _, v in cloud), f"pod is not secure cloud: {cloud}"
    cost = pod.get("costPerHr", pod.get("adjustedCostPerHr"))
    assert isinstance(cost, (int, float)), (
        f"no top-level costPerHr in the pod record — refusing to run blind: {json.dumps(pod)[:800]}")
    assert min_cost <= float(cost) <= max_cost, (
        f"pod costs ${cost}/hr, outside the secure-H100 band [{min_cost}, {max_cost}] — wrong cloud or GPU")
    print(f"  cloudType fields: {cloud or 'none in record'}; costPerHr {cost}", flush=True)
    return float(cost)


def ssh_target(pod_id: str) -> tuple[str, int]:
    """(user@host, port) from `runpodctl ssh info`, which prints a ready ssh command."""
    info = rp("ssh", "info", pod_id)
    text = json.dumps(info) if not isinstance(info, str) else info
    m = re.search(r"ssh\s+(?:-\S+\s+\S+\s+)*?(\w+@[\w.\-]+)\s+-p\s+(\d+)", text) or \
        re.search(r"(\w+@[\w.\-]+).{0,40}?-p\s+(\d+)", text)
    if m is None:
        leaves = list(walk(info))
        ip = next((v for p, v in leaves if re.search(r"ip$", p, re.IGNORECASE) and isinstance(v, str)), None)
        # publicPort first: the record also has privatePort (always 22 inside the container)
        port = next((v for key in ("publicport", "port") for p, v in leaves
                     if p.split(".")[-1].lower() == key), None)
        assert ip and port, f"could not parse ssh info: {text[:800]}"
        return f"root@{ip}", int(port)
    return m.group(1), int(m.group(2))


class Pod:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.id: str | None = None
        self.target: tuple[str, int] | None = None

    def ssh_cmd(self, remote: str) -> list[str]:
        host, port = self.target
        return ["ssh", "-i", str(SSH_KEY), "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", host, remote]

    def ssh(self, remote: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
        return subprocess.run(self.ssh_cmd(remote), check=check, **kw)

    def create(self) -> None:
        cmd = ["pod", "create", "--name", self.args.name, "--cloud-type", "SECURE", "--gpu-id", GPU_ID,
               "--gpu-count", "1", "--image", IMAGE, "--container-disk-in-gb", "200",
               "--ports", "22/tcp", "--ssh", "--wait", "--wait-timeout", "20m"]
        print("$ runpodctl " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        out = subprocess.run(["runpodctl", *cmd], capture_output=True, text=True, check=False)
        # The pod can exist even when create "fails" (on --wait timeout the pod is kept and the
        # error carries its id): record the id BEFORE raising so the finally-delete sees it.
        m = re.search(r'"id"\s*:\s*"([A-Za-z0-9]+)"', out.stdout + out.stderr)
        if m:
            self.id = m.group(1)
        if out.returncode != 0:
            raise RuntimeError(f"pod create failed ({out.returncode}): {(out.stderr or out.stdout)[:800]}")
        pod = parse_json(out.stdout)
        assert isinstance(pod, dict), f"unexpected create output: {out.stdout[:800]}"
        self.id = pod.get("id") or self.id
        assert self.id, f"no pod id in create output: {out.stdout[:800]}"
        print(f"pod {self.id} created", flush=True)
        cost = assert_secure(rp("pod", "get", self.id), self.args.min_cost_per_hr, self.args.max_cost_per_hr)
        print(f"pod {self.id}: secure cloud confirmed, ${cost}/hr", flush=True)
        self.target = ssh_target(self.id)
        print(f"ssh {self.target[0]} -p {self.target[1]}", flush=True)

    def _ids_by_name(self) -> set[str]:
        """Every pod (running or stopped) carrying this run's unique name — catches a pod whose
        id never reached us (Ctrl-C mid-create, unparseable output)."""
        listed = rp("pod", "list", "--all", "--name", self.args.name, check=False)
        return {p["id"] for p in listed if isinstance(p, dict) and p.get("name") == self.args.name
                and p.get("id")} if isinstance(listed, list) else set()

    @staticmethod
    def _gone(pid: str) -> bool:
        out = subprocess.run(["runpodctl", "pod", "get", pid], capture_output=True, text=True, check=False)
        rec = parse_json(out.stdout + "\n" + out.stderr)   # the not-found error JSON arrives on stderr
        if isinstance(rec, dict):
            return rec.get("code") == "not_found" or str(rec.get("runtimeStatus", "")).lower() == "terminated"
        return False   # unparseable → unknown → keep trying

    def delete(self) -> None:
        ids = ({self.id} if self.id else set()) | self._ids_by_name()
        for pid in sorted(ids):
            for _ in range(6):
                out = subprocess.run(["runpodctl", "pod", "delete", pid], capture_output=True, text=True,
                                     check=False)
                time.sleep(5)
                if self._gone(pid):
                    print(f"pod {pid} deleted", flush=True)
                    break
                print(f"delete of {pid} not confirmed yet: {(out.stderr or out.stdout)[:300]}", flush=True)
                time.sleep(10)
            else:
                print(f"!!! COULD NOT CONFIRM DELETION OF POD {pid} — delete it in the RunPod console NOW",
                      flush=True)
        self.id = None


SECRET_FILES = {"hf": ("HF_TOKEN", ".hf_token"), "anthropic": ("ANTHROPIC_API_KEY", ".anthropic_key")}


def upload(pod: Pod, secrets_wanted: list[str]) -> None:
    tar_path = Path("/tmp") / f"easynla_{os.getpid()}.tgz"
    skip = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "wandb"}
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(REPO, arcname="EasyNLA", filter=lambda ti: None if Path(ti.name).name in skip else ti)
    host, port = pod.target
    subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(port), "-o", "StrictHostKeyChecking=accept-new",
                    str(tar_path), f"{host}:/workspace/easynla.tgz"], check=True)
    tar_path.unlink()
    secrets = "\n".join(f"{SECRET_FILES[k][0]}={(Path.home() / SECRET_FILES[k][1]).read_text().strip()}"
                        for k in secrets_wanted) + "\n"
    pod.ssh("umask 077 && cat > /workspace/.secrets", input=secrets, text=True)
    pod.ssh("cd /workspace && rm -rf EasyNLA && tar xzf easynla.tgz && rm easynla.tgz")


def fetch(pod: Pod, local: Path, paths: list[str]) -> None:
    try:
        _fetch(pod, local, paths)
    except Exception as e:   # never let a failed fetch mask the run status or skip the delete
        print(f"!!! fetch failed: {type(e).__name__}: {e}", flush=True)


def _fetch(pod: Pod, local: Path, paths: list[str]) -> None:
    import shutil
    local.mkdir(parents=True, exist_ok=True)
    # Room for the archive AND its extraction, checked before anything streams: run #3 fetched into
    # a 7.9 GB tmpfs scratch dir, died mid-download, and the finally-delete took the rest with the pod.
    r = pod.ssh("cd /workspace && du -sbc " + " ".join(paths) + " 2>/dev/null | tail -1 | cut -f1",
                check=False, capture_output=True, text=True)
    need = 2 * int(r.stdout.strip() or 0)
    free = shutil.disk_usage(local).free
    assert need < free, (f"{local} has {free / 1e9:.1f} GB free, the fetch needs ~{need / 1e9:.1f} GB — pass an "
                         f"--out on a bigger disk (not a tmpfs scratch dir)")
    remote = "cd /workspace && tar czf - " + " ".join(f"--ignore-failed-read {p}" for p in paths)
    with (local / "remote.tgz").open("wb") as f:
        subprocess.run(pod.ssh_cmd(remote), stdout=f, check=False)
    with tarfile.open(local / "remote.tgz") as tf:
        tf.extractall(local, filter="data")
    (local / "remote.tgz").unlink()
    print(f"results → {local}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="local directory for the pulled results")
    p.add_argument("--name", default=f"heads-quick-sft-{int(time.time())}")
    p.add_argument("--max-hours", type=float, default=3.0, help="local wall-clock limit (then delete)")
    p.add_argument("--allow-no-watchdog", action="store_true",
                   help="run even if the pod cannot delete itself (only this process would)")
    p.add_argument("--max-cost-per-hr", type=float, default=4.5)
    p.add_argument("--min-cost-per-hr", type=float, default=3.5,
                   help="a secure H100 SXM is $3.99/hr, community $2.69 — below this means not secure")
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--no-checkpoints", action="store_true", help="don't pull the LoRA checkpoints back")
    p.add_argument("--remote-script", default="scripts/runpod/heads_quick_sft_remote.sh")
    p.add_argument("--fetch", default="results,heads_q/sft", help="/workspace paths to pull back (+ ckpt "
                                                                    "unless --no-checkpoints)")
    p.add_argument("--secrets", default="hf,anthropic", help=f"comma list of {sorted(SECRET_FILES)}")
    p.add_argument("--hold", action="store_true", help="keep the pod after the script until <out>/RELEASE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    secrets_wanted = [k for k in args.secrets.split(",") if k]
    assert (REPO / args.remote_script).is_file(), f"no remote script {args.remote_script}"
    fetch_paths = [x for x in args.fetch.split(",") if x] + ([] if args.no_checkpoints else ["ckpt"])
    script_name = Path(args.remote_script).name
    for f in [SSH_KEY] + [Path.home() / SECRET_FILES[k][1] for k in secrets_wanted]:
        assert f.exists(), f"missing {f}"
    # Checked BEFORE a pod exists: a fetch that runs out of room ends in the finally-delete, which
    # takes the unfetched results with the pod (run #3: checkpoints ~2 GB, archive + extraction ~2×).
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    release = out_dir / "RELEASE"
    release.unlink(missing_ok=True)
    import shutil
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    assert free_gb >= 10, f"--out {out_dir} has only {free_gb:.1f} GB free; need ≥ 10 GB (not a tmpfs scratch dir)"
    if args.dry_run:
        print(f"would create: secure cloud, 1× {GPU_ID}, image {IMAGE}, ≤ ${args.max_cost_per_hr}/hr, "
              f"≤ {args.max_hours} h → ≤ ${args.max_cost_per_hr * (args.max_hours + 0.5):.0f} worst case")
        return

    pod = Pod(args)

    def on_term(signum, frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    t0 = time.time()
    fetched = created = False
    try:
        pod.create()
        created = True
        for attempt in range(10):                       # sshd can lag the --wait banner check
            if pod.ssh("true", check=False).returncode == 0:
                break
            time.sleep(15)
        (out_dir / "pod_ssh.txt").write_text(" ".join(shlex.quote(c) for c in pod.ssh_cmd("")[:-1]) + "\n")
        upload(pod, secrets_wanted)
        guard_s = int((args.max_hours + 0.5) * 3600)
        # Without a working watchdog only this process stands between a crash here and a pod
        # billing until the balance runs out.
        # SSH sessions don't inherit the container's environment, where RunPod puts the pod's id
        # and its pod-scoped API key — read both from PID 1.
        env = ("export $(tr '\\0' '\\n' </proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs); ")
        diag = pod.ssh(env + 'echo "pod_id=${RUNPOD_POD_ID:+set} api_key=${RUNPOD_API_KEY:+set} '
                       'runpodctl=$(command -v runpodctl || echo missing)"', check=False,
                       capture_output=True, text=True)
        print(f"  watchdog probe: {diag.stdout.strip()}", flush=True)
        # Checked on a throwaway pod of this image (runpodctl 1.14 inside): the pod-scoped key is
        # REFUSED for reads (`get pod` → Unauthorized) but `remove pod $RUNPOD_POD_ID` deletes the
        # pod. So probe the prerequisites of removal, not a read.
        probe = pod.ssh(env + 'test -n "${RUNPOD_POD_ID:-}" && test -n "${RUNPOD_API_KEY:-}" && '
                        'runpodctl remove pod --help >/dev/null 2>&1', check=False).returncode == 0
        if probe:
            pod.ssh(env + f"nohup bash -c 'sleep {guard_s}; runpodctl remove pod $RUNPOD_POD_ID || "
                    f"runpodctl pod delete $RUNPOD_POD_ID' >/workspace/watchdog.log 2>&1 </dev/null &")
            print(f"on-pod watchdog armed: self-delete after {guard_s / 3600:.1f} h", flush=True)
        elif args.allow_no_watchdog:
            print("WARNING: the pod cannot delete itself — only this process can (--allow-no-watchdog)", flush=True)
        else:
            raise SystemExit("the pod cannot delete itself (no RUNPOD_POD_ID/RUNPOD_API_KEY or no `runpodctl "
                             "remove pod` inside); refusing to run "
                             "without the watchdog — pass --allow-no-watchdog to accept that risk")
        # `;` not `&&`: with `a && b && nohup c &` bash backgrounds the whole list with ssh's stdout
        # still attached, and this call blocks until the run ends (seen on run #3).
        pod.ssh("mkdir -p /workspace/results; cd /workspace/EasyNLA; "
                f"nohup bash {shlex.quote(args.remote_script)} >/workspace/results/nohup.log 2>&1 </dev/null &")
        print("remote run started", flush=True)
        status = "timeout"
        # "[h]eads_x.sh" matches the running script but not this ssh command's own shell, whose
        # command line contains the bracketed pattern rather than the name (a bare name always
        # matched itself, so __DEAD__ could never fire)
        alive_pattern = f"[{script_name[0]}]{script_name[1:]}"
        while time.time() - t0 < args.max_hours * 3600:
            time.sleep(args.poll_seconds)
            # __DEAD__: the script is gone without writing a marker (OOM-kill, SIGKILL) — no point waiting
            r = pod.ssh("cd /workspace/results && (cat DONE 2>/dev/null && echo __DONE__); "
                        "(cat FAILED 2>/dev/null && echo __FAILED__); "
                        f"(pgrep -f {shlex.quote(alive_pattern)} >/dev/null || echo __DEAD__); "
                        "grep '^=== \\[' run.log | tail -1; tail -2 run.log",
                        check=False, capture_output=True, text=True)
            mins = (time.time() - t0) / 60
            print(f"[{mins:5.1f} min] {r.stdout.strip()[-400:]}", flush=True)
            if "__DONE__" in r.stdout:
                status = "done"
                break
            if "__FAILED__" in r.stdout:
                status = "failed"
                break
            if "__DEAD__" in r.stdout and "__DONE__" not in r.stdout:
                time.sleep(20)   # the DONE marker is the script's last act — recheck once before calling it dead
                again = pod.ssh("test -f /workspace/results/DONE && echo __DONE__", check=False,
                                capture_output=True, text=True)
                status = "done" if "__DONE__" in again.stdout else "died"
                break
        print(f"run status: {status}", flush=True)
        if args.hold and status == "done":
            fetch(pod, Path(args.out), fetch_paths)   # results are safe locally before any holding
            fetched = True
            print(f"holding the pod for interactive work: {out_dir / 'pod_ssh.txt'}; touch {release} to finish",
                  flush=True)
            while time.time() - t0 < args.max_hours * 3600 and not release.exists():
                time.sleep(20)
            print("released" if release.exists() else "max hours reached — fetching and deleting", flush=True)
        fetch(pod, Path(args.out), fetch_paths)
        fetched = True
        if status != "done":
            sys.exit(1)
    finally:
        if not fetched and created:
            # killed or crashed mid-run (signal, Ctrl-C, ssh error): pull what exists before deleting
            try:
                fetch(pod, Path(args.out), fetch_paths)
            except Exception as e:   # noqa: BLE001 — best effort; the delete below must still run
                print(f"best-effort fetch failed: {e}", flush=True)
        pod.delete()
        print(f"wall clock {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
