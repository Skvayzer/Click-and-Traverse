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

## E0 window audit and E1 recovery variants

Both arrays use frozen COMBO_FIX checkpoints for seeds 0–2, the original 384 scene IDs and both 0.15 and 0.30 m/s commands. E0 logs the 25 weighted reward terms by scene for refused episodes whose first confirmed stall has a complete one-second pre-stall and two-second stall window. An audit run uses one episode per scene per speed. For policy comparisons we use the existing two-episode baseline and the matched two-episode E1 protocol.

| Phase | Job | Partition | Per task | Limit | Expected runtime per task | Run directories under `/l/users/konstantin.smirnov/runs` |
| --- | --- | --- | --- | --- | --- | --- |
| E0 audit | 319388 (0–2, at most 2 active) | gpu | 1 GPU, 8 CPUs, 30000 MB | 3 h | 60–95 min | `overnight_E0_baseline_seed{0,1,2}_319388` |
| E1 variants | 319389 (0–8, at most 2 active) | gpu | 1 GPU, 8 CPUs, 30000 MB | 3 h | 1–2 h | `overnight_E1_<fallback\|margin\|both>_seed{0,1,2}_319389` |

E0 waits for round 4 planner seed 2. E1 waits for E0. E1 tasks 0–2 use fallback, 3–5 use margin and 6–8 combine them. Fallback drops the plan after two seconds without a 5 cm forward gain and replans after a further 5 cm reactive gain. Margin selects the highest-scoring candidate with at least 5 cm predicted body clearance, relaxes to 3 cm if necessary, and otherwise acts reactively with a new candidate check every 0.6 seconds. Deployment commit `1bc21af` passed 34 CPU tests and a synthetic check of window timing.

The primary comparison is the paired seed-level change in refusals, clean traversal, any contact and hand contact at each speed. Refusal recovery is useful if the clean rate rises and the contact rate stays level or falls. Side gaps and CAT courses are the prespecified scene groups for diagnosis. E2 stays conditional on evidence that recovery and candidate selection leave a reward-balance problem.
