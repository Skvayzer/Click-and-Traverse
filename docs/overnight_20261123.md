# Overnight summary

In progress. Updated at phase boundaries. Full conclusions and uncertainty will replace this note after E0–E7.

## E0 — existing evidence on traversal refusals

Opened 2026-11-23, Dubai time. Inputs are COMBO_FIX seed 0 eval 315810 and seed 1 318295_0. Each uses 384 scenes and two episodes per scene at each speed. Rates are means ± sample s.d. over the two available training seeds. Seed 2 is evaluating in 318295_2. Terms are the logged weighted terms averaged over all transitions from eventually refused episodes and then averaged over seeds.

| Scene group | Scenes | Refused at 0.15 m/s, % | Refused at 0.30 m/s, % |
| --- | ---: | ---: | ---: |
| Side gaps | 40 | 18.12 ± 7.95 | 25.00 ± 7.07 |
| Cabinets | 160 | 3.28 ± 2.87 | 4.53 ± 2.43 |
| Furniture rooms | 40 | 0.00 ± 0.00 | 2.50 ± 1.77 |
| Generic clutter rooms | 40 | 0.00 ± 0.00 | 3.75 ± 0.00 |
| CAT courses | 104 | 17.55 ± 3.06 | 20.43 ± 1.70 |

| Speed | Stall | Global power | Track position | Track upper | Goal progress | Three hand/arm clearance terms combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | -3.09 ± 0.15 | -0.62 ± 0.10 | 2.19 ± 0.12 | 0.62 ± 0.00 | 1.49 ± 0.01 | -0.05 ± 0.03 |
| 0.3 | -2.79 ± 0.36 | -0.67 ± 0.11 | 2.00 ± 0.06 | 0.64 ± 0.03 | 0.95 ± 0.07 | -0.07 ± 0.01 |

The available logs point to side gaps and CAT courses as the main refusal groups. The stall cost is the largest recorded negative term. Global power comes next. The mean hand/arm clearance cost is much smaller, and positive tracking and goal progress remain substantial when averaged across entire refused episodes. These logs discard scene-level term totals and time ordering, and omit several posture and field terms. Attribution of stall onset requires the new frozen-policy window audit. This first read supports testing recovery and candidate selection before changing the reward.

## Run ledger

- Existing E3: `319349_0–4` on ws-ia and local COMBO_FIX seed 0 on dep-1, with 768 episodes per speed and variant, separate physical inserted contact metrics, and three paired Keynote-safe videos planned. Local `/tmp/cat-plan-blind/local-on.log`. Original code `a197178`.
- Existing E7: α250/α1000 in `318296_2–3`. Third standing seed price flip in `319350_0–1`, 150 updates warm-started from S2 seed 1 u180. Previous energy-only and contact-only measurements are in `docs/energy_runs_20261122.md`.
- The urgent furnished traversal videos were delivered before this program. They are under `/home/konstantinsmirnov/videos/cluttered/`, with handling and the scene list in the energy run log.
