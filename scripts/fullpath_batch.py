#!/usr/bin/env python3
"""Run many scripts/fullpath_rollout.py episodes on the CPU, N at a time, skipping finished ones (docs/fullpath_plan.md).

Each job line of the plan file: TAG|CHECKPOINT|SCENE|SPEED|TRIAL|EXTRA ARGS. Output: OUT/TAG/<scene>_<speed>_<trial>.npz and
.log. Waits before each launch while the available RAM is below --min-free-gb (shared machine).

    fullpath_batch.py --jobs jobs.txt --out /tmp/cat-fullpath/runs --parallel 18
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/tmp/cat-energy-test-env/bin/python"


def free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parallel", type=int, default=18)
    p.add_argument("--min-free-gb", type=float, default=14.)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    env = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="offline",
               CAT_PATH_MAP="/l/users/konstantin.smirnov/cat_root=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab")
    jobs = []
    for line in a.jobs.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tag, ck, scene, speed, trial, *extra = line.split("|")
        out = a.out / tag / f"{scene}_{float(speed):g}_{trial}.npz"
        if out.exists():
            continue
        cmd = [PY, "-u", str(ROOT / "scripts/fullpath_rollout.py"), ck, "--scene", scene, "--speed", speed, "--trial", trial,
               "--out", str(out)] + (shlex.split(extra[0]) if extra else [])
        jobs.append((out, cmd))
    print(f"{len(jobs)} jobs to run", flush=True)
    running, done, failed, t0 = [], 0, 0, time.time()
    while jobs or running:
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r); done += 1
                if r[0].returncode != 0:
                    failed += 1
                    print(f"FAILED {r[1]} rc={r[0].returncode}", flush=True)
        while jobs and len(running) < a.parallel and free_gb() >= a.min_free_gb:
            out, cmd = jobs.pop(0)
            lg = out.with_suffix(".log")
            if out.exists() or (lg.exists() and time.time() - lg.stat().st_mtime < 900):
                continue                       # finished, or running under another driver
            out.parent.mkdir(parents=True, exist_ok=True)
            log = open(out.with_suffix(".log"), "w")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)), out))
            time.sleep(1.)
        if done and done % 25 == 0:
            pass
        time.sleep(3.)
        if int(time.time() - t0) % 300 < 3:
            print(f"{time.strftime('%H:%M:%S')} done {done} failed {failed} running {len(running)} left {len(jobs)} "
                  f"free {free_gb():.0f} GB", flush=True)
    print(f"finished: {done} done, {failed} failed, {time.time() - t0:.0f} s", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
