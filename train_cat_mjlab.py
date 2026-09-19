#!/usr/bin/env python3
"""Train the preserved CAT whole-body task on mjlab with PPO or SAPG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "verify"):
        target = commands.add_parser(command)
        target.add_argument("--num-envs", type=int, default=36864)
        target.add_argument("--algorithm", choices=("ppo", "sapg"))
        target.add_argument("--checkpoint-npz", type=Path, required=True)
        target.add_argument("--bank-manifest", type=Path, required=True)
        target.add_argument("--body-collision-bank", type=Path, required=True)
        target.add_argument("--body-collision-resets", type=Path, required=True)
        target.add_argument("--run-dir", type=Path, required=True)
        target.add_argument("--batch-size", type=int)
        target.add_argument("--unroll-length", type=int)
        target.add_argument("--seed", type=int, default=0)
        target.add_argument("--device", default="cuda:0")
        target.add_argument("--nconmax", type=int, default=64)
        target.add_argument("--njmax", type=int, default=256)
        target.add_argument("--compile-task", action="store_true",
                            help="Compile pure Torch task kernels after constructor checks")
        target.add_argument("--max-updates", type=int, default=1 if command == "verify" else 0,
                            help="Updates this invocation; 0 is continuous (run only)")
        target.add_argument("--checkpoint-interval-updates", type=int, default=10)
        target.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                            default="disabled" if command == "verify" else "online")
        target.add_argument("--wandb-project", default="CAT-wholebody")
        target.add_argument("--wandb-entity", default="skvayzer")
        target.add_argument("--resume", action="store_true", help="Restore complete native mjlab run")
        target.add_argument("--fresh-optimizer", action="store_true",
                            help="Explicitly discard source Adam moments during initial migration")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    from cat_mjlab.runner import run
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
