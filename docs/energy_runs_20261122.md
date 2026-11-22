# Energy pricing runs, 22 November 2026

The deployed training code is `bc51c10308c91bb78bbf96b9492c1efa7e4c38e3` on `feature/energy-price`, based on `feature/plan-reflex` at `5e6079e`. The isolated cluster checkout is `~/cat_worktrees/energy_price_bc51c10`. Its bank, prior, and checkpoint links point to the existing shared inputs.

## Submitted jobs

| Job | Recipe | Partition | GPUs | CPUs | RAM | Limit | Output directory | State |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 314865 | S2H smoke, 3 updates, 12,288 worlds, seed 0, W&B offline | ws-ia | 1 | 12 | 54000M | 00:40:00 | `/l/users/konstantin.smirnov/runs/smoke_hands_S2H_314865` | Pending, requeued and held by Slurm |

Slurm reported `user_env_retrieval_failed_requeued_held` at runtime `00:00:00`. The expected stdout file is `/l/users/konstantin.smirnov/runs/logs/hands_energy_smoke_314865.out`. The file and the training status file were both absent when checked. The job remains held. The three-update smoke was expected to take about 5–15 minutes after allocation, including initialization and compilation.

The smoke blocker halted submissions before the standing, table, and 384-scene checkpoint evaluations and the two training jobs. Standing preflight also did not locate the original seed-0 S2 u180 and S1b u150 snapshots. The Nov. 13 evaluation JSON references temporary copies that are now absent, and the original runs' current status files report updates 430 and 340 respectively. The exact seed-1 and seed-2 reruns are available at their requested updates.

## Recipes and measurements

- S2H uses S2's standing price of −0.02 with the heat model, alpha 500 W, and the u150 warm start. The full budget is 180 updates, matching the S2 seed reruns.
- COMBO_NOPRICE uses COMBO1b's 500-update recipe, warm-started from `plan_reflex_COMBO1_update_0300.pt` with 10 critic-only updates and the E2 u300 anchor. It removes the explicit standing, table, leg-and-waist, global, and route energy prices, including inherited terms. It retains the shared native joint-torque regularizer.
- Existing launch recipes omit `--energy-model` and retain the historical final-sample mechanical reward. Explicit `mech`, `heat`, and `metab` charge clipped PD plus gravity feed-forward, timed over each physics substep. The route offset `--route-energy-p0` defaults to zero. The heat model uses joint torque limits with a shared alpha because motor masses were unavailable.
- Energy telemetry reports policy and applied torque for both mechanical and heat models, by legs, waist, and arms. Heat's positive net work is allocated across groups with positive net work in proportion to their contribution. Standing events cover the active approach through return to the object start. A step requires at least 5 cm horizontal travel and 2 cm lift of the same foot. The arm-first retreat measurement projects the targeted hand's torso-relative displacement onto the initial retreat direction.
