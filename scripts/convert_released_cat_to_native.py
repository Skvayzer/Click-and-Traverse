#!/usr/bin/env python3
"""Released CAT generalist (.npz array archive) -> native 'cat-mjlab-best-v1' checkpoint for --checkpoint-native.

Only the weights come from the archive. The environment config is taken from a given native run/snapshot,
so a warm start gets our full setup (the runner widens the actor input 222 -> 226 and adds the style critic).
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default="outputs/mjlab_migration_20260919/original-cat-expanded.npz")
    p.add_argument("--environment-from", required=True, help="native checkpoint or run.json whose environment config to use")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import torch
    from cat_mjlab.learning import Learner, LearnerConfig
    from cat_mjlab.conversion import load_array_archive
    learner = Learner(LearnerConfig(algorithm="ppo"), device="cpu")
    load_array_archive(learner, args.archive, restore_optimizer=False)
    src = Path(args.environment_from)
    env = (json.loads(src.read_text())["contract"]["environment_config"] if src.suffix == ".json"
           else torch.load(src, map_location="cpu", weights_only=True, mmap=True)["contract"]["environment_config"])
    torch.save(dict(schema="cat-mjlab-best-v1", config=asdict(learner.config), model=learner.model.state_dict(), step=0,
                    contract=dict(environment_config=env), source_archive=str(args.archive),
                    note="weights: released CAT generalist; environment config: " + str(src)), args.output)
    print("wrote", args.output, "actor_obs", learner.config.actor_obs)


if __name__ == "__main__":
    main()
