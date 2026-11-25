#!/usr/bin/env python3
"""G1 robot feasibility (docs/cos_obstacles.md): frozen GEAR-SONIC tracks COS candidate windows in the CPU MuJoCo harness.

Runs against the feature/sonic-baseline code (SonicTracker, scripts/sonic_gates.build): set PYTHONPATH to a snapshot of
that tree, e.g.  PYTHONPATH=$SONIC_TREE python scripts/cos_g1_track.py --windows W.json --library LIB --out DIR --shard i/n

Each window (event +- 1.5 s, merged per clip) is moved so its first frame sits at the flat scene's start position, the
robot is reset to that frame (joint and root velocities by finite differences) and SONIC follows the clip open-loop in time
with the harness torque noise on. Saves per-window metrics (JSON) and the tracked qpos (npz) for PARC-style correction.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Harness:
    """One flat-scene task and one SONIC tracker per process (building the task costs ~110 s); the reference function
    reads the current window from self.ref."""

    def __init__(self, sg):
        self.sg = sg
        self.task, self.scene, self.start = sg.build(sg.V7R, .3, plan=False)
        self.task.reset(scene_ids=self.scene, **self.start)
        self.ref = None
        self.tracker = None

    def reference(self):
        return self.ref()


def run_window(H, clip, noise=True):
    sg, task, scene, start = H.sg, H.task, H.scene, H.start
    sxy = start["initial_pose"][0, :2].clone() if "initial_pose" in start else torch.zeros(2)
    clip = clip.clone(); clip[:, :2] += sxy - clip[0, :2]
    last = len(clip) - 1
    jv = torch.zeros_like(clip[:, 7:]); jv[1:-1] = (clip[2:, 7:] - clip[:-2, 7:]) / .04
    q0 = clip[:1].clone(); v0 = torch.zeros(1, 35)
    v0[0, :3] = (clip[1, :3] - clip[0, :3]) / .02; v0[0, 6:] = jv[0]
    k = {"k": 0}

    def reference():
        idx = (k["k"] + torch.arange(10) * 5).clamp_max(last)
        return clip[idx][None], jv[idx][None]

    H.ref = reference
    task.reset(scene_ids=scene, initial_pose=q0, initial_velocity=v0)
    if not noise:
        task.info["rfi"].zero_()
    if H.tracker is None:
        H.tracker = sg.sonic(task, reference=H.reference)
    else:
        H.tracker.reset_history() if hasattr(H.tracker, "reset_history") else None
    tracker = H.tracker
    qs, root_err, joint_err, pel_dz = [task.data.qpos[0].clone()], [], [], []
    end, done_step = "complete", None
    for step in range(last):
        k["k"] = step
        action, target = tracker()
        out = task.step(action, position_targets=target)
        ref = clip[step + 1]; q = task.data.qpos[0]
        qs.append(q.clone())
        root_err.append(float(torch.linalg.vector_norm(q[:2] - ref[:2])))
        joint_err.append(float((q[7:] - ref[7:]).abs().mean()))
        pel_dz.append(float(q[2] - ref[2]))
        if float(ref[2]) > .6 and float(q[2]) < .5:
            end, done_step = "fell", step + 1; break
        if float(q[2] - ref[2]) < -.25:
            end, done_step = "sank", step + 1; break
        if bool(out["done"][0]):
            end, done_step = "harness_done", step + 1; break
    re = np.array(root_err); je = np.array(joint_err)
    return dict(end=end, done_step=done_step, seconds=round(len(re) * .02, 2), ref_seconds=round(last * .02, 2),
                root_err_p50=float(np.median(re)), root_err_max=float(re.max()), joint_err_p50=float(np.median(je)),
                joint_err_p90=float(np.percentile(je, 90)), pelvis_dz_min=float(np.min(pel_dz))), \
        torch.stack(qs).numpy().astype(np.float32), clip.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int); ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(os.environ["PYTHONPATH"].split(":")[0]) / "scripts"))
    import sonic_gates as sg
    torch.set_num_threads(1)
    W = json.loads(Path(a.windows).read_text())
    i, n = map(int, a.shard.split("/"))
    W = W[i::n][:a.limit]
    q = np.load(a.library + ".npz", mmap_mode="r")["qpos"]
    a.out.mkdir(parents=True, exist_ok=True)
    H = None
    for w in W:
        f = a.out / f"{w['id']}.json"
        if f.exists():
            continue
        t = time.time()
        clip = torch.tensor(np.asarray(q[w["frame0"] + w["f0"]: w["frame0"] + w["f1"]]), dtype=torch.float32)
        try:
            H = H or Harness(sg)
            m, tracked, ref = run_window(H, clip, noise=not a.no_noise)
        except Exception as e:                     # noqa: BLE001 - one bad window must not stop the shard
            m, tracked, ref = dict(end="error", error=repr(e)[:300]), None, None
        m.update(w, wall_s=round(time.time() - t, 1))
        if tracked is not None:
            np.savez_compressed(a.out / f"{w['id']}.npz", tracked=tracked, ref=ref)
        f.write_text(json.dumps(m))
        print(w["id"], m.get("end"), m.get("root_err_p50"), m.get("joint_err_p50"), m["wall_s"], flush=True)


if __name__ == "__main__":
    main()
