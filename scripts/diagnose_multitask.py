#!/usr/bin/env python3
"""Tell budget starvation apart from gradient interference.

cat_goal is falling while clutter and narrow rise, and there are three possible
causes that call for opposite fixes: the navigation task is simply getting fewer
steps, the tasks fight over the same weights, or new behaviour is overwriting old.
Guessing between them from success curves alone is how you end up rebuilding the
training setup for a problem that was a sampling ratio.

Two measurements here, both on a frozen checkpoint:

  share   what fraction of collected TRANSITIONS belongs to each task. The bank
          declares reset masses, not experience shares, and standing episodes run
          to timeout while navigation episodes end at the goal, so the two differ.

  cosine  the angle between per-task gradients of the same loss. Persistently
          negative means the tasks genuinely pull the weights apart and structural
          separation is justified. Near zero means they are merely independent and
          the problem is budget, not conflict.

Gradients are taken separately for the critic and for a policy surrogate, because
value heads are the usual place multi-task conflict actually lives: navigation and
standing have entirely different reward scales and horizons.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def task_labels(task, scene_ids):
    """cat / room / reactive per environment."""
    families = [s.get("task_kind", "room") for s in task.bank.manifest["scenes"]]
    kinds = np.asarray([0 if families[i] == "cat" else 1 for i in scene_ids.tolist()])
    if getattr(task, "reactive_objects", None) is not None:
        active = task.reactive_objects.state["active"].detach().cpu().numpy().astype(bool)
        kinds = np.where(active, 2, kinds)
    return kinds


def flat_grad(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, parameters)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "outputs/cat_fullbank_20260924/resume.pt")
    p.add_argument("--bank", type=Path,
                   default=ROOT / "data/furniture/procedural_rooms_v2_20260923/manifest.json")
    p.add_argument("--collision", type=Path,
                   default=ROOT / "data/furniture/full_collision_20260924b/manifest.json")
    p.add_argument("--resets", type=Path,
                   default=ROOT / "data/furniture/full_resets_20260924b/manifest.json")
    p.add_argument("--reactive", type=Path,
                   default=ROOT / "data/furniture/reactive_standing_v2_20260923/manifest.json")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/multitask_diagnosis_20260924.json")
    args = p.parse_args(argv)

    from cat_mjlab.runner import create_task

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = saved["contract"]
    factory = SimpleNamespace(
        num_envs=args.num_envs, device=args.device, compile_task=False,
        nconmax=contract["nconmax"], njmax=contract["njmax"], seed=0,
        bank_manifest=args.bank, body_collision_bank=args.collision,
        body_collision_resets=args.resets, reactive_bank=args.reactive,
        disable_hand_contrast=True, hand_clearance_weight=-20., arm_clearance_weight=-8.,
        tracking_root_field_weight=1., hand_clearance_target=.09,
        hand_clearance_anticipation=.20, hand_clearance_near_weight=.8,
        hand_reward_soft_floor=0., hand_raised_reset_fraction=0.,
        upper_gravity_compensation=True, require_hand_contrast=False)
    task, sim, _ = create_task(factory, environment_config=contract["environment_config"])

    from cat_mjlab.learning import Learner, LearnerConfig
    config = LearnerConfig(**contract["learner_config"])
    learner = Learner(config, device=args.device)
    # Already a single-policy PPO checkpoint: leader extraction is only for SAPG weights.
    weights = saved.get("weights") or saved.get("learner", {}).get("model") or saved["learner"]
    if "model" in weights:
        weights = weights["model"]
    learner.model.load_state_dict({k: v.to(args.device) for k, v in weights.items()}, strict=False)

    task.reset(task.all_ids)
    counts = np.zeros(3, dtype=np.int64)
    states, privileged, actions, logps, rewards = [], [], [], [], []
    labels = []
    with torch.no_grad():
        for _ in range(args.steps):
            obs = task.obs
            kinds = task_labels(task, task.scene_ids)
            acted = learner.act(obs)
            transition = task.step(acted["action"])
            counts += np.bincount(kinds, minlength=3)
            states.append(obs["state"].clone())
            privileged.append(obs["privileged_state"].clone())
            labels.append(torch.as_tensor(kinds, device=args.device))
            actions.append(acted["raw_action"].clone())
            logps.append(acted["log_prob"].clone())
            rewards.append(transition["reward"].clone())
    share = counts / max(counts.sum(), 1)
    names = ["cat", "room", "reactive"]
    print("=== ДОЛЯ ОПЫТА (переходов) ===")
    for n, c, s in zip(names, counts, share):
        print("  %-9s %9d  %6.2f%%" % (n, c, 100 * s))

    state = torch.cat(states); label = torch.cat(labels)
    action = torch.cat(actions); logp = torch.cat(logps); reward = torch.cat(rewards)
    advantage = (reward - reward.mean()) / reward.std().clamp_min(1e-6)

    privileged_state = torch.cat(privileged)
    model = learner.model
    actor_params = [q for q in model.actor.parameters() if q.requires_grad]
    critic_params = [q for q in model.value_network.parameters()] if hasattr(model, "value_network") else None
    if critic_params is None:
        critic_params = [q for n, q in model.named_parameters() if "actor" not in n and q.requires_grad]

    policy_grads, value_grads = {}, {}
    for index, name in enumerate(names):
        mask = label == index
        if int(mask.sum()) < 256:
            print(f"  {name}: слишком мало переходов ({int(mask.sum())}), пропуск")
            continue
        logits = model.logits(state[mask])
        mean, scale = logits.chunk(2, -1)
        scale = torch.nn.functional.softplus(scale) + 1e-3
        lp = (-.5 * ((action[mask] - mean) / scale) ** 2 - scale.log()).sum(-1)
        policy_grads[name] = flat_grad(-(lp * advantage[mask]).mean(), actor_params)
        predicted = model.value(privileged_state[mask]).reshape(-1)
        value_grads[name] = flat_grad(((predicted - reward[mask]) ** 2).mean(), critic_params)

    cosines = {}
    for title, grads in (("ПОЛИТИКА", policy_grads), ("КРИТИК", value_grads)):
        print()
        print(f"=== КОСИНУС МЕЖДУ ГРАДИЕНТАМИ ЗАДАЧ: {title} ===")
        keys = list(grads)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                c = float(torch.nn.functional.cosine_similarity(grads[a], grads[b], dim=0))
                cosines[f"{title}:{a}|{b}"] = c
                verdict = "КОНФЛИКТ" if c < -.1 else ("независимы" if abs(c) <= .1 else "согласованы")
                print("  %-20s %+.4f   %s" % (f"{a} vs {b}", c, verdict))

    report = dict(transitions=int(counts.sum()), counts=counts.tolist(),
                  share={n: float(s) for n, s in zip(names, share)}, cosines=cosines,
                  checkpoint=str(args.checkpoint), num_envs=args.num_envs, steps=args.steps)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nотчёт: {args.output}")


if __name__ == "__main__":
    main()
