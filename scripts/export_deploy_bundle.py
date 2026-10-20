#!/usr/bin/env python3
"""Export everything the real-robot side needs about one trained policy (no training code required there).

Writes <out>/:
  policy.onnx, policy_torchscript.pt   actor only: obs [B, 226] float32 -> action [B, 29] = tanh(mean), mean [B, 29]
  deploy_spec.json                     observation layout (every feature name + index), joint order, nominal pose,
                                       soft joint limits, PD gains, torque limits, action -> motor-target mapping,
                                       gait clock, command semantics, field definitions (body sites, frames, clamps,
                                       odometry delay), training noise / randomisation, full environment config,
                                       checkpoint provenance and hashes
  robot.xml                            the exact MJCF used in training (FK for the field sample points, gravity
                                       compensation, site definitions)
  reference_rollouts.npz               deterministic CPU-simulator rollouts in several scenes: per step the actor
                                       observation, the action, and the task internals it was built from (joint
                                       state, IMU, body-site positions, delayed odometry, fields, command, phase,
                                       motor targets) -- unit-test the robot-side observation builder against these
  checkpoint_full.pt is NOT copied here (see outputs/saved_checkpoints/)
Runs on the CPU (CPUSimulation, one world), so it can run next to GPU training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Actor(torch.nn.Module):
    """Deterministic actor: the mean of the policy's Gaussian, squashed by tanh (exactly what the task executes)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs):
        from cat_mjlab.learning import gaussian_parameters
        ids = torch.zeros(obs.shape[:-1], dtype=torch.long, device=obs.device)
        mean, _ = gaussian_parameters(self.model.logits(obs, ids))
        return torch.tanh(mean), mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", default="data/furniture/demo_v1")
    p.add_argument("--scenes", nargs="+", default=["course-mixed-obstacles-v1", "published-hurdle0", "published-crouch0",
                                                   "sidegap2-0162-plain-w0.47", "procedural-D7G3L9O2S2002", "flat-balance-walk-v1"])
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    from cat_mjlab.sim_cpu import CPUSimulation
    from cat_mjlab.observation_contract import CONTRACT
    from cat_ppo.furniture.control import JOINT_NAMES
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)

    learner, saved = load_policy(args.checkpoint, device="cpu", policy_id=0)
    contract = saved["contract"]
    env = copy.deepcopy(contract["environment_config"])
    deploy_env = copy.deepcopy(env)
    # Reference rollouts: no observation noise, no PD randomisation / random force injection, no pushes, no
    # training-only sampler or keyboard episodes -- clean, reproducible numbers for unit tests.
    deploy_env["push_config"]["enable"] = False
    deploy_env["noise_config"]["level"] = 0.
    deploy_env["dm_rand_config"].update(enable_pd=False, enable_rfi=False)
    for key in ("sgs", "teleop", "side_gap_curriculum"):
        deploy_env.pop(key, None)
    B = ROOT / args.bank
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device="cpu", seed=0, num_envs=1, compile_task=False,
                         nconmax=contract["nconmax"], njmax=contract["njmax"], reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=deploy_env, sim_class=CPUSimulation)

    # ---------------------------------------------------------------- actor export
    actor = Actor(learner.model).eval()
    names = list(CONTRACT["actor_features"]) + ["sdf_rate.left_hand", "sdf_rate.right_hand", "sdf_rate.left_elbow", "sdf_rate.right_elbow"]
    obs_dim = learner.config.actor_obs
    if len(names) != obs_dim:
        raise ValueError(f"feature names {len(names)} != actor input {obs_dim}")
    example = torch.zeros(1, obs_dim)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example)
    traced.save(str(args.out / "policy_torchscript.pt"))
    torch.onnx.export(actor, example, str(args.out / "policy.onnx"), input_names=["obs"], output_names=["action", "mean"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}, "mean": {0: "batch"}}, opset_version=17, dynamo=False)

    # ---------------------------------------------------------------- reference rollouts
    from cat_mjlab.sim_cpu import assemble_training_xml
    (args.out / "robot.xml").write_text(assemble_training_xml())
    index = {s["scene_id"]: i for i, s in enumerate(task.bank.manifest["scenes"])}
    record = {}
    keys = ("gf", "bf", "sdf", "gf_delay", "bf_delay", "sdf_delay", "positions", "command", "command_delay", "last_command",
            "phase", "phase_dt", "foot_height", "stop_timestep", "gait", "navi", "last_act", "motor_targets", "odom_delay", "step")
    for scene in args.scenes:
        if scene not in index:
            print("skip (not in bank):", scene); continue
        task.reset(torch.zeros(1, dtype=torch.long), torch.tensor([index[scene]]))
        for t in range(args.steps):
            rows = dict(scene=np.array([index[scene]]), t=np.array([t]),
                        obs=task.obs["state"][0].numpy().copy(),
                        qpos=sim.data.qpos[0].numpy().copy(), qvel=sim.data.qvel[0].numpy().copy(),
                        gyro_pelvis=task._sensor("gyro_pelvis", task.all_ids)[0].numpy().copy(),
                        sites_world=sim.data.site_xpos[0, task.site_ids].numpy().copy(),
                        elbow_sites_world=sim.data.site_xpos[0, task.elbow_ids].numpy().copy())
            for k in keys:
                if k in task.info:
                    rows["info_" + k] = task.info[k][0].numpy().copy()
            with torch.no_grad():
                action, mean = actor(task.obs["state"])
            rows["action"] = action[0].numpy(); rows["mean"] = mean[0].numpy()
            out = task.step(action)
            rows["done"] = np.array([bool(out["done"][0])])
            for k, v in rows.items():
                record.setdefault(k, []).append(v)
            if bool(out["done"][0]):
                break
    np.savez_compressed(args.out / "reference_rollouts.npz", **{k: np.stack(v) for k, v in record.items()},
                        feature_names=np.array(names), info_keys=np.array([k for k in keys if "info_" + k in record]))

    # ---------------------------------------------------------------- verification
    from onnx.reference import ReferenceEvaluator
    import onnx
    model = onnx.load(str(args.out / "policy.onnx")); onnx.checker.check_model(model)
    obs = np.stack(record["obs"]).astype(np.float32)
    onnx_action = ReferenceEvaluator(model).run(None, {"obs": obs})[0]
    ts_action = torch.jit.load(str(args.out / "policy_torchscript.pt"))(torch.from_numpy(obs))[0].detach().numpy()
    recorded = np.stack(record["action"])
    verification = dict(steps=len(obs), onnx_max_abs_err=float(np.abs(onnx_action - recorded).max()),
                        torchscript_max_abs_err=float(np.abs(ts_action - recorded).max()))

    # ---------------------------------------------------------------- spec
    groups = {}
    for i, n in enumerate(names):
        g = n.split(".")[0] if not n.startswith("pf.") else ".".join(n.split(".")[:2])
        groups.setdefault(g, [i, i]); groups[g][1] = i
    sites = list(task.site_names)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    spec = dict(
        policy=dict(input="obs float32 [B, %d]" % obs_dim, outputs=dict(action="tanh(mean) in (-1, 1), float32 [B, 29] -- what the task executes",
                    mean="pre-squash Gaussian mean [B, 29]"), deterministic=True, actor_hidden=list(learner.config.actor_hidden),
                    activation="see cat_mjlab/learning.py ActorCritic (exported graph is authoritative)"),
        timing=dict(control_dt=float(task.dt), physics_dt=float(sim.model.opt.timestep) if hasattr(sim, "model") else .002,
                    note="policy at 50 Hz; PD targets held between policy steps (training: PD + feedforward every physics step)"),
        joints=dict(order=list(JOINT_NAMES), nominal=task.nominal.tolist(), soft_lower=task.lower.tolist(), soft_upper=task.upper.tolist(),
                    soft_limit_factor=float(env.get("soft_joint_pos_limit_factor", .95)), kp=task.kps.tolist(), kd=task.kds.tolist(),
                    torque_limit=task.torque_limit.tolist()),
        action=dict(legs=dict(indices=list(range(12)), rule="target = clamp(previous_target + 0.5 * action, soft_lower, soft_upper)  (incremental)"),
                    upper_body=dict(indices=list(range(12, 29)), rule="requested = nominal + 0.8 * action; target = clamp(requested, previous_target -/+ 2.0 rad/s * 0.02 s); clamp to soft limits"),
                    initial_target="nominal pose", source="cat_mjlab/task_math.py motor_targets"),
        actuation=dict(torque="kp*(target - q) + kd*(0 - qdot) + feedforward, clipped to torque_limit",
                       upper_gravity_compensation=bool(env.get("upper_gravity_compensation", False)),
                       feedforward="joints 12..28 (waist + arms): negative gravitational generalised force from current body kinematics "
                                   "(cat_mjlab/upper_control.py UpperGravity, robot.xml masses); legs: none"),
        observation=dict(dim=obs_dim, feature_names=names, groups={k: dict(first=v[0], last=v[1]) for k, v in groups.items()},
                         source="cat_mjlab/task_math.py observations (actor 'state'); cat_mjlab/task.py _observe",
                         rules=dict(
                             joint_position="q - nominal (joint order above)", joint_velocity="qdot (rad/s)",
                             gyro="angular velocity of the pelvis IMU site (sensor gyro_pelvis), pelvis frame, rad/s", gravity="projected gravity: -R[2, :] with R the world rotation of the pelvis IMU site (imu_in_pelvis), i.e. the unit down vector in the pelvis frame (task._observe)",
                             last_action="previous policy action (tanh output)", motor_target="current PD targets (after the action rule)",
                             command="[move, vx, vy, yaw] DELAYED command; vx,vy rotated into the heading (navi) frame; field-driven: "
                                     "task_math.compute_cmd_from_rtf on the delayed root/limb fields, then update_phase stop logic; yaw kept only for heading-command checkpoints",
                             foot_height="gait swing height parameter (0.07 m in training)",
                             gait_phase="[cos(phase_L), cos(phase_R), sin(phase_L), sin(phase_R)]; phase += 2*pi*f*dt, f ~ U(1.3, 1.5) Hz in training (use 1.4); "
                                        "phase = 0 while stopped; anti-phase [0, pi] on restart",
                             fields="per body site: guidance gf (3), boundary bf (3, zeroed where sdf >= 0.5), signed distance (1, clamped to [-1, 0.5]); "
                                    "gf/bf rotated into the heading (navi) frame; sampled at site positions reconstructed from a HELD odometry pose "
                                    "(updated every 5 control steps = 10 Hz) and the current body-local site offsets (task_math.delay_body_pos)",
                             field_sites="11 field sites in order: " + ", ".join(sites[:11]) + "; plus 2 elbow probe sites (elbow block)",
                             heading_frame="navi: rotation with x = horizontal projection of the pelvis x axis, z up (task_math.navi_rotation)",
                             sdf_rate="time derivative of [left hand sdf, right hand sdf, left elbow sdf, right elbow sdf] (delayed), /dt, clamped [-3, 3], 0 at episode start")),
        fields=dict(generator="CAT HumanoidPF pipeline: occupancy (4 cm) -> SDF -> 3-D FMM toward the goal at 0.75 m -> obstacle-tangent blend "
                              "(cat_ppo/furniture/generalist_fields.py, upstream pf_modular); guidance stored as unit directions",
                    sites=sites, elbow_probe_sites=["left_elbow_probe", "right_elbow_probe"]),
        gait=dict(freq_range=env.get("gait_config", {}).get("freq_range", [1.3, 1.5]), foot_height_range=env.get("gait_config", {}).get("foot_height_range", [.07, .07]),
                  gait_bound=env.get("gait_config", {}).get("gait_bound", .6)),
        training_randomisation=dict(noise_config=env.get("noise_config"), dm_rand_config=env.get("dm_rand_config"), push_config=env.get("push_config")),
        provenance=dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint), git_commit=commit,
                        env_steps=None),
        verification=verification,
        environment_config=env,
    )
    (args.out / "deploy_spec.json").write_text(json.dumps(spec, indent=1, default=str) + "\n")
    hashes = {f.name: sha256(f) for f in sorted(args.out.iterdir()) if f.is_file() and f.name != "SHA256SUMS"}
    (args.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    print(json.dumps(dict(out=str(args.out), verification=verification, steps=len(obs)), indent=1))


if __name__ == "__main__":
    main()
