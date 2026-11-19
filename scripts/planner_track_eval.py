#!/usr/bin/env python3
"""Execute planner plans with the trained full-body tracker and score what the robot actually did.

For held-out situations of a planner dataset: sample plans (classifier-free guidance --guidance), keep the best of
--samples, and make two 2 s reference clips per situation, the plan and the policy's own executed motion (baseline).
The tracker (cat_mjlab/mimic.py, mjlab's G1) executes every clip in parallel from its first frame, in physics on flat
ground. The executed motion is scored against the situation's map like a plan (497-sphere cover; a touch = clearance
below -tol), with the tracking error and terminations (falls, large deviations). --probe adds the inserted column of
planner_probe.py beside the policy's path, plans with it in the map and scores against it.

    python scripts/planner_track_eval.py --data RUN/all --planner RUN/train/planner.pt --tracker LOGS/model_X.pt --out OUT
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import cat_mjlab  # noqa: E402,F401


def pick_situations(test, per_type, rng):
    chosen = []
    for b in np.unique(test["meta"][:, 1]):
        idx = np.nonzero(test["meta"][:, 1] == b)[0]
        chosen += rng.choice(idx, min(per_type, len(idx)), replace=False).tolist()
    return np.array(sorted(chosen))


def insert_columns(torch, z, frame, offset, radius, keep, device):
    """Probe columns (planner_probe.py): one per situation side that keeps start and goal clear."""
    from cat_mjlab.planner.grid import cell_centres, quantize, dequantize
    rows, cols = [], []
    for r in range(len(z["state"])):
        ex = z["chunk"][r]; k = frame - 1
        tangent = ex[min(k + 1, 19), :2] - ex[max(k - 1, 0), :2]
        if np.linalg.norm(tangent) < 1e-3:
            tangent = np.array([np.cos(ex[k, 3]), np.sin(ex[k, 3])])
        t = tangent / np.linalg.norm(tangent); n = np.array([-t[1], t[0]])
        for side in (1, -1):
            c = ex[k, :2] + side * offset * n
            if np.linalg.norm(c) - radius >= keep and np.linalg.norm(z["goal"][r][:2] - c) - radius >= keep:
                rows.append(r); cols.append(c)
    rows = np.array(rows); cols_t = torch.as_tensor(np.stack(cols), dtype=torch.float32, device=device)
    centres = cell_centres(device); edited = []
    for s in range(0, len(rows), 256):
        g = torch.as_tensor(z["grid"][rows[s:s + 256]], device=device)
        col = torch.linalg.vector_norm(centres[None, ..., :2] - cols_t[s:s + 256, None, None, None], dim=-1) - radius
        edited.append(quantize(torch.minimum(dequantize(g), col)).cpu().numpy())
    sub = {k: v[rows] for k, v in z.items()}
    sub["grid"] = np.concatenate(edited)
    return sub, cols_t.cpu().numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--planner", type=Path, required=True)
    p.add_argument("--tracker", type=Path, required=True, help="mjlab runner checkpoint (model_<iter>.pt)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--per-type", type=int, default=300)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    import torch
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.planner.data import load_split, euler_zyx
    from cat_mjlab.planner.metrics import ChunkScorer
    from cat_mjlab.planner.model import FlowPlanner, Normalizer, sample
    from cat_mjlab.runner import SCENE_BUCKETS
    from planner_train import held_out_halves
    from planner_eval import best_of
    from mimic_library import keyframes, resample, body_states
    started = time.time()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    held = load_split(args.data, "test")
    _, test_mask = held_out_halves(held["meta"], 0)
    test = {k: v[test_mask] for k, v in held.items()}
    sel = pick_situations(test, args.per_type, rng)
    z = {k: v[sel] for k, v in test.items()}
    columns = None
    if args.probe:
        z, columns = insert_columns(torch, z, 10, 0.25, 0.10, 0.35, device)
    n = len(z["state"])
    # 1. plans: best of --samples with the (possibly edited) map
    ck = torch.load(args.planner, map_location="cpu", weights_only=True)
    planner = FlowPlanner(**ck["config"]).to(device).eval(); planner.load_state_dict(ck["model"])
    norm = Normalizer(np.zeros((2, 67)), np.zeros((2, 5)), np.zeros((2, 20, 35))).to(device); norm.load_state_dict(ck["norm"])
    scorer = ChunkScorer(mujoco.MjModel.from_xml_string(assemble_training_xml()), device)
    g = torch.Generator(device=device).manual_seed(args.seed)
    plans, plan_clear, exec_clear = [], [], []
    for s in range(0, n, 32):
        b = {k: torch.as_tensor(np.ascontiguousarray(z[k][s:s + 32]), device=device) for k in ("state", "goal", "grid", "chunk")}
        x = norm.denorm("chunk", sample(planner, norm.norm("state", b["state"]), norm.norm("goal", b["goal"]), b["grid"], use_scene=True,
                                         samples=args.samples, steps=16, guidance=args.guidance, generator=g))
        sc = scorer.score(x, b["grid"], b["goal"]); best = best_of(sc)
        plans.append(x[torch.arange(len(best)), best].cpu().numpy())
        plan_clear.append(sc["min_clear"][torch.arange(len(best)), best].cpu().numpy())
        exec_clear.append(scorer.score(b["chunk"][:, None], b["grid"], b["goal"])["min_clear"][:, 0].cpu().numpy())
    plans, plan_clear, exec_clear = np.concatenate(plans), np.concatenate(plan_clear), np.concatenate(exec_clear)
    print(json.dumps(dict(situations=n, probe=bool(args.probe), plan_kinematic_free=round(float((plan_clear >= -args.tol).mean()), 3),
                          executed_free=round(float((exec_clear >= -args.tol).mean()), 3), seconds=round(time.time() - started, 1))), flush=True)
    # 2. reference clips in each situation's chunk frame: plans first, then the policies' own motion
    from mjlab.scene import Scene
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    g1 = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu").compile()
    parts = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    starts, total = [], 0
    for chunks in (plans, z["chunk"]):
        for r in range(n):
            qpos, qvel = resample(keyframes(z["state"][r], chunks[r], (0.0, 0.0, 0.0)))
            pos, quat, lin, ang = body_states(g1, qpos, qvel)
            starts.append(total); total += len(qpos)
            for k, v in zip(parts, (qpos[:, 7:], qvel[:, 6:], pos, quat, lin, ang)):
                parts[k].append(np.asarray(v, np.float32))
    clip_len = len(qpos)
    tmp = Path(tempfile.mkdtemp()) / "clips.npz"
    np.savez(tmp, fps=np.array([50]), segment_starts=np.array(starts, np.int64), **{k: np.concatenate(v) for k, v in parts.items()})
    print(json.dumps(dict(clips=len(starts), frames=total, seconds=round(time.time() - started, 1))), flush=True)
    # 3. the tracker executes every clip from its first frame
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from cat_mjlab.mimic import SegmentMotionCommand, SegmentMotionCommandCfg, mimic_env_cfg, mimic_rl_cfg
    assigned = torch.as_tensor(starts, device=device)

    class EvalCommand(SegmentMotionCommand):
        def _uniform_sampling(self, env_ids):
            self.time_steps[env_ids] = assigned[env_ids]

    @dataclasses.dataclass(kw_only=True)
    class EvalCommandCfg(SegmentMotionCommandCfg):
        def build(self, env):
            return EvalCommand(self, env)

    cfg = mimic_env_cfg(play=True, motion_file=str(tmp))
    old = cfg.commands["motion"]
    cmd_cfg = EvalCommandCfg(**{f.name: getattr(old, f.name) for f in dataclasses.fields(old)})
    cmd_cfg.sampling_mode, cmd_cfg.joint_position_range, cmd_cfg.pose_range, cmd_cfg.velocity_range = "uniform", (0.0, 0.0), {}, {}
    cfg.commands["motion"] = cmd_cfg
    cfg.scene.num_envs = len(starts)
    env = ManagerBasedRlEnv(cfg=cfg, device=str(device))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=mimic_rl_cfg().clip_actions)
    runner = MotionTrackingOnPolicyRunner(wrapped, dataclasses.asdict(mimic_rl_cfg()), device=str(device))
    runner.load(str(args.tracker), load_cfg={"actor": True}, strict=True, map_location=str(device))
    policy = runner.get_inference_policy(device=str(device))
    obs, _ = wrapped.reset()
    robot = env.scene["robot"]; origins = env.scene.env_origins
    steps = clip_len - 1
    root = torch.zeros(steps + 1, len(starts), 7, device=device); joints = torch.zeros(steps + 1, len(starts), 29, device=device)
    alive = torch.ones(len(starts), dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(steps + 1):
            root[t, :, :3] = robot.data.root_link_pos_w - origins; root[t, :, 3:] = robot.data.root_link_quat_w
            joints[t] = robot.data.joint_pos
            if t == steps:
                break
            obs, _, dones, extras = wrapped.step(policy(obs))
            alive &= ~(dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool())
    # 4. score the executed motion like a plan: 10 Hz frames in the chunk frame
    roll, pitch, yaw = euler_zyx(root[..., 3:])
    frames = torch.cat((root[..., :3], yaw[..., None], roll[..., None], pitch[..., None], joints), -1)[5::5].permute(1, 0, 2)  # [clips, 20, 35]
    grids = torch.as_tensor(np.concatenate((z["grid"], z["grid"])), device=device)
    goals = torch.as_tensor(np.concatenate((z["goal"], z["goal"])), device=device)
    out = {k: [] for k in ("min_clear", "min_hand_clear", "goal_err", "min_low")}
    for s in range(0, len(starts), 256):
        sc = scorer.score(frames[s:s + 256, None], grids[s:s + 256], goals[s:s + 256])
        for k in out:
            out[k].append(sc[k][:, 0].cpu())
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    ref = torch.as_tensor(np.concatenate((plans, z["chunk"])), device=device)
    track_err = torch.linalg.vector_norm(frames[..., :2] - ref[..., :2], dim=-1).mean(-1).cpu().numpy()
    joint_err = (frames[..., 6:] - ref[..., 6:]).pow(2).mean((-1, -2)).sqrt().cpu().numpy()
    alive = alive.cpu().numpy()
    kinds = z["meta"][:, 1]

    cut = exec_clear < -args.tol if args.probe else np.ones(n, bool)     # probes count only where the column blocks the policy

    def summary(mask):
        mask = mask & cut
        res = {}
        for name, half, kin in (("plan", slice(0, n), plan_clear), ("policy_motion", slice(n, 2 * n), exec_clear)):
            m = mask
            clear = out["min_clear"][half] >= -args.tol
            res[name] = dict(kinematic_clear=round(float((kin[m] >= -args.tol).mean()), 3),
                             executed_clear=round(float(clear[m].mean()), 3),
                             executed_clear_and_upright=round(float((clear & alive[half])[m].mean()), 3),
                             terminated=round(float((~alive[half])[m].mean()), 3),
                             pelvis_track_err_m=round(float(track_err[half][m].mean()), 3),
                             joint_rmse_rad=round(float(joint_err[half][m].mean()), 3),
                             goal_err_m=round(float(out["goal_err"][half][m].mean()), 3))
        res["situations"] = int(mask.sum())
        return res

    result = dict(all=summary(np.ones(n, bool)))
    for b in np.unique(kinds):
        result[SCENE_BUCKETS[b]] = summary(kinds == b)
    result["_info"] = dict(planner=str(args.planner), tracker=str(args.tracker), guidance=args.guidance, samples=args.samples,
                           probe=bool(args.probe), tol_m=args.tol, situations=n, seconds=round(time.time() - started, 1))
    name = "track_eval_probe" if args.probe else "track_eval"
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    np.savez_compressed(args.out / f"{name}.npz", meta=z["meta"], pose=z["pose"], state=z["state"], goal=z["goal"], plan=plans,
                        executed=z["chunk"], tracked=frames.cpu().numpy(), min_clear=out["min_clear"], alive=alive,
                        plan_clear=plan_clear, exec_clear=exec_clear, **({"column": columns} if columns is not None else {}))
    print("TRACK EVAL " + json.dumps(result["all"]), flush=True)


if __name__ == "__main__":
    main()
