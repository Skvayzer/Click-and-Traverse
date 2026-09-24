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
        target.add_argument("--lateral-corridor", action="store_true", help="Protected route corridor and whole-arm margin rewards")
        target.add_argument("--upper-gravity-compensation", action="store_true", help="Current-configuration upper gravity feedforward before torque clipping")
        target.add_argument("--upper-action-scale", action="append", metavar="JOINT=RADIANS", help="Repeat per upper joint; default .8 rad")
        target.add_argument("--protected-hand-sdf-margin", action="store_true", help="Relax positive-clearance protected hand shaping; retain contact penalty")
        target.add_argument("--num-envs", type=int, default=36864)
        target.add_argument("--algorithm", choices=("ppo", "sapg"))
        source = target.add_mutually_exclusive_group(required=True)
        source.add_argument("--from-scratch", action="store_true",
                            help="Initialize native 222/310 actor/critic and fresh optimizer, with no checkpoint")
        source.add_argument("--checkpoint-npz", type=Path)
        source.add_argument("--checkpoint-native", type=Path,
                            help="Fresh weights-only warm start from native best.pt/resume.pt; requires --fresh-optimizer")
        target.add_argument("--passage-rewards", type=Path, help="Bank-pinned per-scene/per-zone reward metadata")
        target.add_argument("--checkpoint-selection-weights", type=float, nargs=4,
                            metavar=('PROTECTED', 'FLAT_WALK', 'FLAT_PROGRESS', 'CAT'),
                            help="Nonnegative weights summing to 1; default .6 .2 .1 .1")
        target.add_argument("--flat-bonus-scale", type=float, help="Flat-only posture bonus scale; default inherited/bank (3)")
        target.add_argument("--flat-region-scale", type=float, help="Flat-only region shaping radius in metres; default inherited/bank (.15)")
        target.add_argument("--disable-hand-contrast", action="store_true", default=True,
                            help="Compatibility flag: box rewards are permanently off")
        target.add_argument("--hand-clearance-weight", type=float, help="Signed clearance weight <= 0; default -0.5")
        target.add_argument("--arm-clearance-weight", type=float, help="Signed elbow clearance weight <= 0; default -2")
        target.add_argument("--tracking-root-field-weight", type=float,
                            help="Positive velocity tracking weight across ALL tasks; default inherited (1.0)")
        target.add_argument("--heading-align-weight", type=float,
                            help="Bonus >= 0 for facing the guidance direction wherever a forward-facing body fits "
                                 "(shoulder-clearance gated, so sidling through narrow gaps is never penalised); 0 disables")
        target.add_argument("--hand-clearance-target", type=float, help="Target clearance in metres; default .04")
        target.add_argument("--hand-clearance-anticipation", type=float, help="Anticipation distance in metres; default .20")
        target.add_argument("--hand-clearance-near-weight", type=float, help="Near-contact share in [0,1]; default .8")
        target.add_argument("--hand-contrast-region-weight", type=float,
                            help="Deprecated compatibility option; box rewards are always zero")
        target.add_argument("--hand-reward-soft-floor", type=float,
                            help="Opt-in protected/transition soft floor width in reward/step, [0,1); 0 preserves hard floor")
        target.add_argument("--hand-contrast-heading-weight", type=float,
                            help="Deprecated compatibility option; box rewards are always zero")
        target.add_argument("--hand-contrast-approach-distance", type=float,
                            help="Extra incoming hand reward ramp in metres, forward-protected scenes only; default inherited (0)")
        target.add_argument("--hand-raised-reset-fraction", type=float,
                            help="Fraction of forward-protected resets seeded at certified raised poses, in [0,1]; default 0 (opt-in on each launch)")
        target.add_argument("--max-action-std", type=float, help="Optional sigma ceiling; 0 disables; default inherited/off")
        target.add_argument("--init-action-std", type=float, help="Reset the scale head on a fresh run; requires --fresh-optimizer")
        target.add_argument("--discounting", type=float, help="Override source gamma; default inherited (CAT .98)")
        target.add_argument("--hand-curriculum-success-threshold", type=float,
                            help="Hand-task clean-goal advancement rate in (0,1]; default inherited/bank value (legacy .6)")
        ladder = target.add_mutually_exclusive_group()
        ladder.add_argument("--hand-speed-curriculum", action="store_true",
                            help="Opt-in protected-zone 0/.2/.3/.45/.6 m/s ladder; excludes tolerance ladder")
        target.add_argument("--hand-speed-curriculum-window", type=int, default=256)
        target.add_argument("--hand-speed-curriculum-success-threshold", type=float, default=.8)
        target.add_argument("--hand-speed-curriculum-demote-threshold", type=float, default=.4,
                            help="Demote after two full per-role windows below this success rate")
        ladder.add_argument("--hand-tolerance-curriculum", action="store_true",
                            help="Opt-in 20/15/10/7/5 cm posture-success ladder; strict metrics remain 5 cm")
        target.add_argument("--hand-tolerance-curriculum-window", type=int, default=256,
                            help="Rolling unseeded completed episodes per protected/transition role (default 256)")
        target.add_argument("--require-hand-contrast", action="store_true",
                            help="Fail before GPU allocation unless hand and heading objectives have coverage")
        target.add_argument("--bank-manifest", type=Path, required=True)
        target.add_argument("--standing-gf-bonus", type=float,
                            help="Constant paid per guidance group when no movement is commanded. "
                                 "Default 4.0 pays +16/step for standing still, measured at ~54%% of mean reward")
        target.add_argument("--reactive-hand-guidance", action="store_true",
                            help="Point hand guidance along the outward obstacle normal in reactive scenes, "
                                 "making handsgf (the only hand-velocity term) live instead of identically zero")
        target.add_argument("--handsdf-weight", type=float,
                            help="Override the legacy handsdf scale; it out-gradients the hand-clearance term")
        target.add_argument("--experience-reactive-share", type=float,
                            help="Target share of ALL transitions for reactive standing episodes (their reset draw "
                                 "is otherwise a fixed .25 outside the solver and grew to 64%% of experience); "
                                 "group targets are scaled by (1 - share). Needs --experience-masses")
        target.add_argument("--experience-rebalance-every", type=int, default=5,
                            help="Re-solve reset masses from REALIZED episode lengths every N updates so each "
                                 "group's share of transitions holds its --experience-masses target; 0 = solve once at startup")
        target.add_argument("--experience-masses", type=float, nargs="+",
                            help="Target share of TRANSITIONS per sampling group; reset mass is solved as "
                                 "share/episode_length so short-horizon tasks are not starved. Prefer this "
                                 "over --sampling-masses, which sets reset mass directly")
        target.add_argument("--sampling-masses", type=float, nargs="+",
                            help="Override the grouped sampler's masses, in group order; renormalised. "
                                 "Default leaves one empty scene holding 0.225 while 461 rooms share 0.034")
        target.add_argument("--narrow-sampling-group", type=int,
                            help="Move every scene whose id contains -narrow- into this sampling group")
        target.add_argument("--cat-episode-length", type=int,
                            help="Override episode length for task_kind=cat scenes; default 1000 against 4000 "
                                 "for rooms, which starves CAT retention of experience")
        target.add_argument("--reactive-bank", type=Path,
                            help="Opt-in cat-reactive-standing-v1 composite manifest; adds approaching-object "
                                 "standing scenes alongside the retained bank distribution")
        target.add_argument("--body-collision-bank", type=Path, required=True)
        target.add_argument("--body-collision-resets", type=Path, required=True)
        target.add_argument("--run-dir", type=Path, required=True)
        target.add_argument("--batch-size", type=int)
        target.add_argument("--num-minibatches", type=int,
                            help="Override learner minibatch count; default inherited from checkpoint")
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
                            help="Explicitly discard source Adam moments during initial migration or native warm start")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    from cat_mjlab.runner import run
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
