#!/usr/bin/env python3
"""Smoke test of tracker v1 in the real task (CPU, one world): a V-series checkpoint's environment plus the tracker,
the checkpoint's network widened with zero-weight reference inputs, a few episodes of each reference source.

    python scripts/tracker_smoke.py --checkpoint outputs/saved_checkpoints/slow_V5_u150.pt --bank data/furniture/tablegap_v1

Checks: observation widths (actor +41, critic +41), finite observations and rewards, the tracking terms and telemetry,
the command taken from the reference, and that the widened network acts exactly like the original one (the new
inputs have zero weight)."""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", default="data/furniture/tablegap_v1")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--steps", type=int, default=150, help="control steps per episode")
    args = p.parse_args(argv)
    import torch
    import cat_mjlab  # noqa: F401  (path map)
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task, _expand_actor_input, _expand_critic_input
    from cat_mjlab.learning import Learner
    from cat_mjlab.sim_cpu import CPUSimulation
    from cat_mjlab.tracking import FEATURES, SOURCES
    learner, saved = load_policy(args.checkpoint, device="cpu", policy_id=0)
    env = copy.deepcopy(saved["contract"]["environment_config"])
    if not env.get("operator_speed"):
        raise SystemExit("needs a checkpoint trained with the operator speed (V-series)")
    env["push_config"]["enable"] = False
    env["noise_config"]["level"] = 0.
    env["dm_rand_config"].update(enable_pd=False, enable_rfi=False)
    for key in ("sgs",):
        env.pop(key, None)
    env["tracker"] = dict(mix=[["field", .4], ["random", .4], ["none", .2]], sigma=.15)
    env["reward_config"]["scales"].update(track_pos=4., track_lead=-8., track_heading=1., track_height=1., track_upper=1.)
    ns = SimpleNamespace(bank_manifest=Path(args.bank + "_packed/manifest.json"), body_collision_bank=Path(args.bank + "_collision/manifest.json"),
                         body_collision_resets=Path(args.bank + "_resets/manifest.json"), device="cpu", seed=0, num_envs=1,
                         compile_task=False, nconmax=saved["contract"]["nconmax"], njmax=saved["contract"]["njmax"],
                         reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=env, sim_class=CPUSimulation)
    task.collision_every_substep = False
    old = learner.config
    actor_in, critic_in = task.observation_size, task.privileged_observation_size
    assert actor_in == old.actor_obs + FEATURES and critic_in == old.critic_obs + FEATURES, (actor_in, critic_in, old.actor_obs, old.critic_obs)
    from dataclasses import replace
    wide = Learner(replace(old, actor_obs=actor_in, critic_obs=critic_in), device="cpu")
    weights = _expand_critic_input(_expand_actor_input(learner.model.state_dict(), actor_in), critic_in)
    wide.model.load_state_dict(weights, strict=True)
    report = dict(actor_obs=actor_in, critic_obs=critic_in, episodes=[])
    gen = torch.Generator().manual_seed(1)
    for e in range(args.episodes):
        rooms = torch.nonzero(~task.bank.is_cat, as_tuple=False)[:, 0]
        pool = rooms if e % 2 == 0 and len(rooms) else torch.arange(task.bank.count)
        scene = int(pool[torch.randint(len(pool), (1,), generator=gen)])
        task.reset(None, torch.tensor([scene]))
        src = SOURCES[int(task.tracker.source[0])]
        errs, terms, cmd_dev = [], {}, []
        for t in range(args.steps):
            obs = task.obs
            assert torch.isfinite(obs["state"]).all() and torch.isfinite(obs["privileged_state"]).all()
            with torch.no_grad():
                a_wide = wide.act(obs, policy_ids=0, deterministic=True)["action"]
                a_old = learner.act(dict(state=obs["state"][:, :old.actor_obs], privileged_state=obs["privileged_state"][:, :old.critic_obs]),
                                    policy_ids=0, deterministic=True)["action"]
            assert torch.allclose(a_wide, a_old, atol=1e-5), "widened network must act like the original"
            result = task.step(a_wide)
            assert torch.isfinite(result["reward"]).all()
            for k, v in task.telemetry.items():
                if k.startswith("track_"):
                    terms.setdefault(k, []).append(float(v.float().mean()))
            if bool(task.tracker.valid[0]):
                target, _ = task.tracker.targets(torch.arange(1), .3)
                d = target - task.data.qpos[:, :2]
                c = task.info["command"][0, 1:3]
                moving = float(task.info["command"][0, 0]) > .5 and float(c.norm()) > .05
                if moving and float(torch.linalg.vector_norm(d)) > .1 and float(task.tracker.speed[0]) > .05:
                    cos = ((c @ d[0]) / (c.norm() * d[0].norm() + 1e-9)).clamp(-1., 1.)
                    cmd_dev.append(float(torch.arccos(cos)))
            if bool(result["done"][0]):
                break
        report["episodes"].append(dict(scene=task.bank.manifest["scenes"][scene]["scene_id"][:40], source=src, steps=t + 1,
                                       speed=round(float(task.tracker.speed[0]), 3),
                                       mean_track_error_m=round(sum(terms.get("track_error_m", [0])) / max(len(terms.get("track_error_m", [1])), 1), 3),
                                       command_vs_reference_rad=round(max(cmd_dev), 4) if cmd_dev else None))
    print(json.dumps(report, indent=1))
    print("TRACKER SMOKE OK")


if __name__ == "__main__":
    main()
