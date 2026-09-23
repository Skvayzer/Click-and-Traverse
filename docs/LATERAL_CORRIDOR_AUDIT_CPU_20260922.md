**Verdict: neither is a demonstrated standalone solution.** The corridor is a valid, less restrictive cabinet finger-protection constraint; the box is not justified by these measurements as a necessary protection target. But at an equivalent ~79 mm margin, the corridor still produces **0/12 fully compliant first-core crossings** on the existing reference-assisted walking trajectories. It is also satisfiable with unchanged nominal arms by turning sideways, so it cannot replace a forward-heading/task requirement. These data do not prove dynamic unachievability under a different controller or objective.

CPU-only reduction and forward kinematics of saved states; no simulation steps, policy inference, training, GPU initialization, bank/training changes, live checkpoint access or tmux operations. Only the new audit script, this report and a small JSON result were added. The existing working tree had many unrelated changes; they were left untouched.

**Geometry and definition.** All six finite cabinet OBBs in each actual scene.json were read. Project each box onto route-left normal n: lateral center c=(box_center-route_start)·n, support h=Σ half_size[j]|axis[j]·n|. The innermost negative-side face gives L=max(c+h); the innermost positive-side face gives U=min(c−h). Cabinets are verified parallel to the route; taking the intersection over all three pairs protects a longitudinal sweep through every module. The full Dex3 palm/finger/thumb sphere radius is **103.446173 mm**, including its existing 5 mm mesh envelope margin. With additional cabinet clearance m=78.990 mm, sphere centers must satisfy **L+r+m ≤ y ≤ U−r−m**, with **y=(hand_world−route_start)·n**. The conservative symmetric form is **|y|≤min(U,−L)−r−m**. The intervals are centered within numerical rounding. Source: scripts/audit_lateral_corridor_cpu.py:55; cat_ppo/furniture/grippers.py:125.

This is a hard sphere-center bound, with **no additional 50 mm acceptance tolerance**. Adding that tolerance would consume 50 mm of the requested clearance. If using hand coordinates relative to the root, enforce |hand_local_y + root_cross_track|≤bound, not |hand_local_y|≤bound. The latter falsely accepts **234 recorded reference-assisted steps** that fail the world-route corridor. The old box itself follows root XY; its quoted ~79 mm clearance was a centered-root guarantee, not an observed minimum throughout walking.

| Scene index | Actual gap (mm) | Bound ± (mm), 78.990 mm margin | Original box centered margin (mm) |
|---|---:|---:|---:|
| 2351 | 737.198 | 186.163 | 80.357 |
| 2353 | 737.460 | 186.294 | 80.488 |
| 2355 | 734.459 | 184.793 | 78.987 |
| 2357 | 739.658 | 187.393 | 81.587 |
| 2359 | 737.198 | 186.163 | 80.357 |
| 2361 | 737.460 | 186.294 | 80.488 |
| 2363 | 734.459 | 184.793 | 78.987 |
| 2365 | 739.658 | 187.393 | 81.587 |
| 2367 | 737.198 | 186.163 | 80.357 |
| 2369 | 737.460 | 186.294 | 80.488 |
| 2371 | 734.459 | 184.793 | 78.987 |
| 2373 | 739.658 | 187.393 | 81.587 |

Your 18.7 cm estimate is close, but the geometry gives **18.479–18.739 cm**, not a universal 18.7 cm. To preserve each individual scene’s exact original centered-box margin (78.987–81.587 mm), the symmetric bound is **184.795869 mm in all 12 scenes**. That stricter comparison passes 684/967 core endpoints (70.7%), averages 27.20% compliant full-core distance, and still yields 0/12. Thus choosing the common rounded 78.990 mm margin does not determine the verdict. Full cabinet projections, signed intervals, scene paths and hashes are in outputs/lateral_corridor_cpu/audit.json.

**Direct rescore against the box baseline.** These are the saved absolute-reference experiments in outputs/body_region_cpu/absolute_*.npz: frozen walking policy, arm IK reference intervention, certified seeded start. They are not unmodified-policy walking. First core is progress 0.50–1.85 m; reset is 0.52 m. The score is exactly the historical newly traversed compliant distance divided by 1.35 m, including its 20 mm unobserved initial sliver in the denominator. Consequently even an otherwise perfect saved crossing would score at most 98.52% distance coverage. Full-compliant crossing follows the historical criterion: reach the core end with no recorded violation after reset. This does not certify the missing initial sliver or later cores.

“Core steps” below counts recorded endpoints within the core, excluding the two endpoints just beyond the exit (967 of 969). “Initial / longest” means consecutive compliant 20 ms endpoints from reset / anywhere in the saved episode, not verified continuous-time compliance. The longest run can occur after an earlier violation. Full-core distance coverage is distinct from fraction of observed core steps: a stalled episode can accumulate many compliant steps without traversing much distance.

| Scene | Box good core steps | Corridor good core steps | Box → corridor compliant full-core distance | Box initial (steps) | Corridor initial / longest (steps) | Reached exit | Full compliant crossing |
|---|---:|---:|---:|---:|---:|---|---|
| 2351 | 42/65 (64.6%) | 52/65 (80.0%) | 10.75% → 17.09% | 18 | 16 / 36 | no | no |
| 2353 | 39/69 (56.5%) | 54/69 (78.3%) | 8.38% → 20.74% | 19 | 17 / 37 | no | no |
| 2355 | 32/84 (38.1%) | 60/84 (71.4%) | 3.16% → 16.45% | 32 | 16 / 36 | no | no |
| 2357 | 29/137 (21.2%) | 83/137 (60.6%) | 9.57% → 56.49% | 10 | 46 / 46 | yes | no |
| 2359 | 30/138 (21.7%) | 92/138 (66.7%) | 8.37% → 54.52% | 30 | 45 / 45 | yes | no |
| 2361 | 24/72 (33.3%) | 58/72 (80.6%) | 5.10% → 22.67% | 24 | 58 / 58 | no | no |
| 2363 | 29/52 (55.8%) | 45/52 (86.5%) | 7.86% → 20.58% | 14 | 45 / 45 | no | no |
| 2365 | 27/88 (30.7%) | 58/88 (65.9%) | 7.71% → 36.13% | 13 | 47 / 47 | no | no |
| 2367 | 27/52 (51.9%) | 46/52 (88.5%) | 8.17% → 21.63% | 12 | 46 / 46 | no | no |
| 2369 | 32/51 (62.7%) | 46/51 (90.2%) | 9.40% → 19.28% | 32 | 46 / 46 | no | no |
| 2371 | 28/50 (56.0%) | 45/50 (90.0%) | 8.47% → 18.66% | 14 | 45 / 45 | no | no |
| 2373 | 17/109 (15.6%) | 56/109 (51.4%) | 2.99% → 31.17% | 12 | 43 / 43 | no | no |

Pooled recorded core-step compliance rises **356/967 (36.8%) → 695/967 (71.9%)**. Mean per-scene full-core distance coverage rises **7.49% → 27.95%**. The previously quoted **7.9–10.8%** was the original four-scene subset (2351/2357/2363/2369), not all 12: its exact baseline 10.75/9.57/7.86/9.40% becomes **17.09/56.49/20.58/19.28%**. Full crossings remain **0/4 and 0/12**. Initial corridor holds are **16–58 steps (0.32–1.16 s)**; longest holds **36–58 steps (0.72–1.16 s)**. Three scenes actually have shorter initial holds than the box because route anchoring now accounts for root drift.

Only scenes 2357 and 2359 reach the core end at all. Their minimum swept hand-sphere clearances over the saved trajectories are **4.10 mm and 33.87 mm**, respectively, well below 78.99 mm. A **zero-additional-margin** corridor gives 958/967 compliant endpoints (99.1%) and 2/12 fully compliant crossings, but abandons the requested clearance buffer. The other ten trajectories end before crossing; rescoring cannot tell us what they would do after termination under a changed objective.

For strict comparison, the main rescore uses the same legacy hand sampling as the historical box metric (sites lag endpoint qpos by one 2 ms physics substep). Independent endpoint FK changes only four classifications: **691/967 (71.5%)**, 27.80% mean distance and still 0/12. All original box distance and initial-run results are reproduced by assertions. FK exactly reproduces the saved actual-hand arrays. No arm states between 20 ms endpoints are saved, so no substep safety claim is made.

**Lateral distributions.** Values below are cm. Signed y is positive to the left of the route; “worst” is max(|left|,|right|) per core endpoint. Excess is max(worst−that scene’s bound,0). These are endpoint-weighted distributions, not equally weighted episode distributions.

| Trace set / quantity | Min | P5 | P25 | Median | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference-assisted, 12 scenes / left y | 0.59 | 2.54 | 8.52 | 12.58 | 14.94 | 18.09 | 21.46 |
| Reference-assisted, 12 scenes / right y | -28.51 | -24.26 | -18.51 | -12.54 | -10.70 | -7.00 | -4.35 |
| Reference-assisted, 12 scenes / worst absolute y | 9.83 | 12.52 | 14.00 | 16.08 | 19.43 | 24.26 | 28.51 |
| Reference-assisted, 12 scenes / excess beyond bound | 0.00 | 0.00 | 0.00 | 0.00 | 0.83 | 5.55 | 9.77 |
| Reference-assisted, 12 scenes / root-relative left y | 10.39 | 11.81 | 13.28 | 15.33 | 17.11 | 19.97 | 23.34 |
| Reference-assisted, 12 scenes / root-relative right y | -20.95 | -14.92 | -12.49 | -10.86 | -8.71 | -6.46 | -2.51 |
| Unmodified policy, 4 scenes / left y | 12.42 | 18.34 | 26.19 | 27.15 | 28.15 | 29.42 | 30.86 |
| Unmodified policy, 4 scenes / right y | -28.39 | -27.11 | -26.22 | -25.51 | -24.61 | -22.46 | -12.51 |
| Unmodified policy, 4 scenes / worst absolute y | 12.51 | 22.46 | 26.81 | 27.26 | 28.16 | 29.42 | 30.86 |
| Unmodified policy, 4 scenes / excess beyond bound | 0.00 | 3.97 | 8.14 | 8.61 | 9.61 | 10.79 | 12.24 |
| Unmodified policy, 4 scenes / root-relative left y | 12.20 | 18.89 | 26.31 | 26.70 | 27.11 | 27.57 | 28.14 |
| Unmodified policy, 4 scenes / root-relative right y | -27.27 | -26.92 | -26.43 | -26.03 | -25.59 | -22.06 | -12.51 |

The reference-assisted controller is generally near the corridor: median worst offset **16.08 cm**, P95 **24.26 cm**, P95 required lateral correction **5.55 cm**, maximum **9.77 cm**. The unmodified saved policy is further away: median worst offset **27.26 cm**, median correction **8.61 cm**, P95 correction **10.79 cm**, maximum **12.24 cm**. This is roughly a **9–11 cm problem for that unmodified policy**, not 20 cm, and not reliably just 6 cm. Root-relative medians are left/right **+26.70/−26.03 cm**, so centering the root alone does not fix it.

**Unmodified walking evidence and its limits.** Existing outputs/arm_hold_diagnosis_cpu/policy.npz supplies four first episodes with original policy actions; these start in a certified protected arm pose. They are from the older frozen speedladder snapshot recorded in outputs/arm_hold_diagnosis_cpu/checkpoint.json, not a measurement of the live cat_rec policy. They make only ~9–11 cm forward progress in 4 seconds and do not cross the first core. Treat them as stalled unmodified walking attempts, not successful normal gait. No equivalent unmodified-policy trace for all 12 scenes, or live-policy lateral distribution, is established by these artifacts.

| Scene | Box good / 200 core steps | Corridor good / 200 core steps | Initial / longest corridor seconds | Compliant full-core distance | Fully compliant crossing |
|---|---:|---:|---:|---:|---|
| 2351 | 4 | 8 | 0.16 / 0.16 | 1.13% | no |
| 2357 | 4 | 8 | 0.16 / 0.16 | 1.21% | no |
| 2363 | 4 | 9 | 0.18 / 0.18 | 1.17% | no |
| 2369 | 4 | 8 | 0.16 / 0.16 | 1.21% | no |

Together, **33/800 (4.125%)** corridor-compliant endpoints versus 16/800 (2%) box-compliant endpoints. After the first second, **0/600** endpoints satisfy the corridor. Its brief initial compliance comes from the protected seeded start. This is not a criterion that the recorded unmodified policy already satisfies.

**Gameability.** Independent FK witnesses put the root on the centerline and set the full waist/arm joint vector to its saved nominal values. Forward-facing nominal hands sit at **+273.526 and −273.516 mm**, failing all 12 corridors by roughly **86–89 mm**. Turning that same nominal upper body sideways by 90° puts both hands near **−5.374 mm**, passing all 12, with whole-arm lateral clearance at least **284.228 mm**. This is a static geometric witness valid when translated down the straight corridor, not a dynamic sideways traversal demonstration. Separately, substituting nominal arms into the actual 969 walking poses gives 70 compliant endpoints; centering those same pelvis poses gives 299/969, because some observed headings/tilts narrow the route-normal arm footprint. Keeping the pelvis upright and route-facing with those saved waist poses gives 0/969.

Therefore **yes, the standalone corridor can be satisfied without changing the arms if heading is free**. It is worthless as proof that a forward-facing arm-protection behavior was learned. It is not worthless as a geometric cabinet finger-protection test: the sideways witness really clears the cabinets. If forward traversal is part of the task, retain an explicit forward-heading constraint and require forward progress/completion; do not infer either from hand clearance. Forward reach would be an additional task/posture requirement, not something established as necessary for cabinet clearance by these data. Tightening the margin arbitrarily would not address the missing heading requirement.

**Elbows and forearms.** FK transforms the existing production mesh-enclosing arm capsules, including both elbow links and all wrist/forearm links; it does not treat elbow centers as zero-radius points. We measure analytic lateral swept clearance and exact finite-cabinet capsule-to-OBB separation. The elbow/forearm proxies include their existing 3 mm geometry margin. Below, minima are over endpoint-FK hand-corridor-compliant core poses, so hand and arm measurements use a consistent pose. Units: mm.

| Scene | Compliant FK core endpoints | Elbow lateral minimum | Forearm incl. elbow/wrist lateral minimum | Elbow finite-cabinet minimum |
|---|---:|---:|---:|---:|
| 2351 | 51 | 92.55 | 92.55 | 92.55 |
| 2353 | 54 | 77.92 | 77.92 | 77.92 |
| 2355 | 58 | 66.41 | 66.41 | 66.41 |
| 2357 | 83 | 96.12 | 96.12 | 107.33 |
| 2359 | 91 | 111.36 | 111.36 | 111.36 |
| 2361 | 58 | 100.36 | 100.36 | 100.36 |
| 2363 | 45 | 71.80 | 71.80 | 71.80 |
| 2365 | 58 | 85.75 | 85.48 | 85.75 |
| 2367 | 46 | 80.70 | 80.70 | 80.70 |
| 2369 | 46 | 68.75 | 68.75 | 68.75 |
| 2371 | 45 | 84.38 | 84.38 | 84.38 |
| 2373 | 56 | 87.08 | 87.08 | 87.72 |

**No observed hand-compliant core endpoint has an elbow or forearm proxy collision.** Minimum elbow/forearm clearance is **66.41 mm**; minimum across all arm capsules is **64.01 mm**. Ten hand-compliant core endpoints have elbow/forearm clearance below 78.99 mm; eleven have some arm capsule below it. Thus the hands-only condition does **not** imply the requested whole-arm margin, even within this recorded sample. Across all recorded reference-assisted core endpoints (including hand failures), minimum forearm clearance is still positive, **34.24 mm**. The four unmodified-policy episodes have at least **136.20 mm** elbow/forearm swept clearance during their brief hand-compliant initial segment. Their hand failures are not evidence of elbow collisions.

Independent finite-cabinet hand-sphere checks give minimum **79.064 mm** on the 691 FK-compliant reference-assisted endpoints, confirming the requested hand margin. Positive enclosing-capsule clearance certifies the covered arm meshes on these saved poses; these finite samples do not prove that every arm configuration with compliant hands is safe. Keep separate elbow/forearm or full-body collision clearance. The corridor only certifies the paired cabinets, not floor clearance, self-collision or unrelated obstacles.

**Recommendation.** Do not keep the 3D box merely to express cabinet protection, but do not adopt a hands-only corridor as a solved replacement task. Choose **neither as a standalone objective**. A route-anchored corridor is the better geometric protection component, combined with required heading/progress and independent whole-arm clearance; its controllability remains unverified. The honest result at the requested margin is improved partial compliance, **still 0/12 crossings**, a measurable ~9–11 cm unmodified-policy gap, and a clear heading loophole. No training change is justified as already validated by this rescore.

**Reproduction and checks.** Run `PYTHONDONTWRITEBYTECODE=1 nice -n 15 .venv-mjlab/bin/python scripts/audit_lateral_corridor_cpu.py`. The script reads only saved traces/geometry, sets CUDA visibility empty and single-thread CPU limits, calls kinematics without mj_step, checks a >2 GiB free-space reserve, reproduces the historical box metrics, validates saved FK hand positions, checks the hand certificate against finite OBBs, and asserts no Torch CUDA initialization. Input hashes and all per-scene/per-mode distributions are retained in outputs/lateral_corridor_cpu/audit.json. No training/integration test suite was run because no production code changed. The body-relative traces were also rescored as a secondary dataset: 139/280 corridor-compliant endpoints and 0/12 crossings; these early-terminated traces do not improve the conclusion.
