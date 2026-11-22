# Energy pricing runs, 22 November 2026

The deployed training code is `bc51c10308c91bb78bbf96b9492c1efa7e4c38e3` on `feature/energy-price`, based on `feature/plan-reflex` at `5e6079e`. The isolated cluster checkout is `~/cat_worktrees/energy_price_bc51c10`. Its bank, prior, and checkpoint links point to the existing shared inputs.

## Submitted jobs

| Job | Recipe | Partition | GPUs | CPUs | RAM | Limit | Expected runtime | After | Output directory |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 314865 | S2H smoke, 3 updates, 12,288 worlds, seed 0 | ws-ia | 1 | 12 | 54000M | 00:40:00 | 5–15 min | | `/l/users/konstantin.smirnov/runs/smoke_hands_S2H_314865` |
| 314946 | Standing u150, S1b seed 1 u150, S2 seeds 1–2 u180 | ws-ia | 1 | 8 | 48000M | 02:00:00 | 15–45 min |  | `/l/users/konstantin.smirnov/runs/energy_20261122_standing_314946` |
| 314947 | Tables T1c u640, T2 u410, E1 u500 | ws-ia | 1 | 8 | 48000M | 02:00:00 | 15–40 min |  | `/l/users/konstantin.smirnov/runs/energy_20261122_tables_314947` |
| 314948 | 384 scenes, V7r u600 | ws-ia | 1 | 8 | 48000M | 03:00:00 | about 50–65 min | 314946 | `/l/users/konstantin.smirnov/runs/energy_20261122_v7r_314948` |
| 314949 | 384 scenes, COMBO1b u500, plans on | ws-ia | 1 | 8 | 48000M | 03:00:00 | about 75–100 min | 314947 | `/l/users/konstantin.smirnov/runs/energy_20261122_combo1b_314949` |
| 314950 | S2H, 50 updates from S2 seed 1 u180, seed 0 | ws-ia | 1 | 12 | 54000M | 03:00:00 | about 70–95 min | 314948 | `/l/users/konstantin.smirnov/runs/stationary_hands_20261113_S2H` |
| 314951 | COMBO_NOPRICE, 50 updates from COMBO1b u500, seed 0 | ws-ia | 1 | 12 | 54000M | 03:00:00 | about 65–95 min | 314949 | `/l/users/konstantin.smirnov/runs/stationary_hands_20261113_COMBO_NOPRICE` |

Slurm initially requeued and held smoke job 314865 with `user_env_retrieval_failed_requeued_held`. After the shell profile fix and release, the same job completed all three updates in 15m 40s. The training metrics recorded finite policy and applied energy values for both models. The smoke log is `/l/users/konstantin.smirnov/runs/logs/hands_energy_smoke_314865.out`.

The follow-up jobs use two dependency chains with `afterany`. The runtimes are estimates after allocation and exclude queue and dependency waits. All jobs use W&B offline. Evaluation stdout goes to `/l/users/konstantin.smirnov/runs/logs/energy_<case>_<job>.out`. The eval batch script is `~/jobs/energy_eval_bc51c10_20261122.sbatch`. The 384-scene jobs use the retention list, two episodes per scene, and speeds of 0.15 and 0.3 m/s.

The submitted training jobs override the recipe defaults with 50-update weights-only fine-tunes. S2H loads `/l/users/konstantin.smirnov/checkpoints/stationary_hands_S2_seed1_u180.pt` and keeps the original u150 anchor. COMBO_NOPRICE loads `/l/users/konstantin.smirnov/checkpoints/plan_reflex_COMBO1b_u500.pt` and keeps the E2 u300 anchor. Both use fresh optimizer state with pretrained policy weights.

Standing evaluation uses the available checkpoints at the requested updates. A search of the shared runs and checkpoint archives found no seed-0 S1b u150 or S2 u180 snapshot. The Nov. 13 evaluation JSON references temporary copies that are absent. S1b seed 2's current checkpoint is u220, and its u150 file was also unavailable.

## Recipes and measurements

- S2H uses S2's standing price of −0.02 with the heat model, alpha 500 W, and the u150 warm start. The full budget is 180 updates, matching the S2 seed reruns.
- COMBO_NOPRICE uses COMBO1b's 500-update recipe, warm-started from `plan_reflex_COMBO1_update_0300.pt` with 10 critic-only updates and the E2 u300 anchor. It removes the explicit standing, table, leg-and-waist, global, and route energy prices, including inherited terms. It retains the shared native joint-torque regularizer.
- Existing launch recipes omit `--energy-model` and retain the historical final-sample mechanical reward. Explicit `mech`, `heat`, and `metab` charge clipped PD plus gravity feed-forward, timed over each physics substep. The route offset `--route-energy-p0` defaults to zero. The heat model uses joint torque limits with a shared alpha because motor masses were unavailable.
- Energy telemetry reports policy and applied torque for both mechanical and heat models, by legs, waist, and arms. Heat's positive net work is allocated across groups with positive net work in proportion to their contribution. Standing events cover the active approach through return to the object start. A step requires at least 5 cm horizontal travel and 2 cm lift of the same foot. The arm-first retreat measurement projects the targeted hand's torso-relative displacement onto the initial retreat direction.
