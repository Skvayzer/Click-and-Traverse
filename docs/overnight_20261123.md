# Overnight summary

Provisional at 09:49 Dubai, November 24. The completed comparisons have yet to show a joint gain in clean traversal and contact.

- Original 384 scenes, three paired seeds. Reflex-on clean deltas were −0.87 and −1.78 pp at 0.15 and 0.30 m/s. The corresponding hand-contact deltas were 0.00 and −0.48 pp. All paired 95% intervals include zero.
- Plan-blind evaluation, three paired seeds. The 5 cm band reduced inserted-object hand contact at 0.30 m/s with a moving box by 1.26 pp (95% CI 0.59–1.93 pp reduction). The four-protocol adjusted interval crosses zero. Mean clean traversal fell by 0.09–0.78 pp and the four intervals include zero. Three synchronized replays are in the project energy-videos folder.
- Held-out layouts at 0.50 m/s, seed 0. Against the reflex-off tracker, reflex-on cut hand contact by 4.17 pp and raised refusals by 6.25 pp with equal total contact.
- E0's full three-seed audit found that goal-progress reward fell from 2.74 to 0.70 at 0.30 m/s around the first stall. Posture and tracking credit persisted. E2's warm-start profile passed CPU checks. The two pending E1 seed comparisons determine whether E2 launches.
- On E4 at 0.50 m/s, COMBO_FIX seed 0 gained 46.53 pp clean traversal [36.11, 56.94] and reduced contact by 54.17 pp [43.75, 64.58] versus the released BRACE tracker with the common planner. These confidence intervals resample scenes. The original 384-scene BRACE evaluation is queued.
- At 250, 500 and 1,000 W in the single-seed sensitivity test, arm-first fell from 43.68% to 38.85% to 33.06%. Failures rose from 9.46% to 11.90% to 14.19%. The 500 W run had the fewest worlds moving their feet.

## E0 — existing evidence on traversal refusals

Opened 2026-11-23, Dubai time. Inputs are COMBO_FIX seed 0 eval 315810 and seed 1 318295_0. Each uses 384 scenes and two episodes per scene at each speed. Rates are means ± sample s.d. over the two available training seeds. Seed 2 was still evaluating in 318295_2 when this first analysis was written. Terms are the logged weighted terms averaged over all transitions from eventually refused episodes and then averaged over seeds.

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

## E4 held-out planning benchmark protocol

The benchmark fixes 72 layouts, 24 each of U-shaped traps, displaced sideways-entry channels and table gaps. Geometry seed is 20261124. Table widths span 0.46–0.62 m and use the existing kinematic certificate for passing above the tabletop. Walls and collision fields share one geometry. U and sideways scenes have an independently certified root route. Their deployed guidance comes from the goal field, including for the reactive and released CAT policies. Table scenes supply the same route to all policies. At most two deterministic episodes per scene at 0.15, 0.30 and 0.50 m/s are planned, with shared starts and speeds. The main outcome is clean traversal at 0.50 m/s. The prediction is at least a five-point clean-traversal gain versus reactive on U and turn families combined without more hand contact. A gain with more hand contact or a smaller clean gain fails this criterion. Report clean, any contact, hand contact, refused, time to first clean goal and minimum palm field clearance by family and speed. Geometric clearance certificates establish a plausible route and have no guarantee of dynamic feasibility.

The E4 report ends each world at its first goal, collision or timeout. One world is allocated per episode. The evaluator records a hash of initial poses and velocities to check pairing across policies and rejects any start that needed a collision reset. The frozen V7r u600 baseline uses the same checkpoint as evaluation 314948. Released CAT uses the expanded array archive of the original policy on the common physics and operator command. The no-reflex checkpoint supplies E4's plan-only control, with neutral actor guidance, obstacle and field-rate inputs and an active planner.

## E5 ablation protocol

The ladder is reactive V7r, plan-only (COMBO_NOREFLEX with masked actor fields), plan plus field (COMBO_NOREFLEX), and plan plus field and reflex (COMBO_FIX). Each tracker rung uses the three available seed checkpoints. The difference between the two no-reflex rungs isolates removal of actor fields at inference, so it measures the frozen policy's use of those inputs and may also expose sensitivity to the changed observation distribution. Collisions and planner inputs stay intact. E3's target comparison is the paired change in inserted-object hand contact, alongside clean traversal and exposures. The available COMBO1b and COMBO_NOPRICE checkpoints differ in training duration and the price-free model continues from the priced model. Their paired-scene contrast is exploratory. A full factorial conclusion about the energy term requires identically initialized price-on and price-off fine-tunes with a common budget.

### E4 and E5 submissions

Local bank validation generated 72 unique layouts and 72 goal-column fields. The native CPU check certified 32 reset poses in each of 3,663 bank scenes. U traps have 5.1–6.0 cm of certified root-cylinder clearance along the detour, and turn-before-entry passages have 3.1–6.4 cm along the channel. The cluster independently rebuilds the exact geometry under deployment `38b62ce` in `~/cat_worktrees/energy_overnight_38b62ce/inputs/planning_v1_{packed,collision,resets,goal}`. The E4 aggregate separates time-to-goal on clean arrivals and mean episode-minimum palm clearance over all episodes.

| Job | Work | Resources on gpu | Limit | Expected runtime per task |
| --- | --- | --- | --- | --- |
| 319412 | Build and certify E4 inputs | 1 GPU, 8 CPUs, 38000 MB | 1 h 30 min | 10–25 min, CPU work |
| 319413 / 319414 | E4 V7r u600 / released CAT | 1 GPU, 8 CPUs, 38000 MB | 3 h | 35–65 min |
| 319415 / 319416 / 319417 | E4 current COMBO_FIX / COMBO_NOREFLEX / plan-only, seeds 0–2 each | 1 GPU, 8 CPUs, 38000 MB | 3 h | 45–95 min |
| 319418 / 319419 | E4 COMBO1b u500 / COMBO_NOPRICE u50 | 1 GPU, 8 CPUs, 38000 MB | 3 h | 45–95 min |

Results go to `/l/users/konstantin.smirnov/runs/overnight_E4_<model>_seed<seed>_<job>`, with files for 0.15, 0.30 and 0.50 m/s. E4 rollout jobs depend on 319412. The seed arrays also depend on completion of round 4. These current-planner results establish the reference for a winner selected after E1/E2. Baseline trace archive SHA256 is `0c3f6fa49f7d472b9ccf7b463111e01fec1106579b2fbe420b7d8a4198b58bca`.

E5 on the plan-blind benchmark continues after 319349 with the same four obstacle schedules, 768 episodes per protocol. Jobs `319436` (plan-only, seeds 0–2), `319437` (reactive V7r u600), `319438` (priced COMBO1b) and `319439` (COMBO_NOPRICE) use ws-ia, one GPU, 12 CPUs, 54000 MB and a seven-hour limit. Estimated runtime is 2–4 hours per task once admitted, with at most two ws-ia jobs running. Outputs are `/l/users/konstantin.smirnov/runs/overnight_E5_<model>_seed<seed>_<job>`.

## E6 public baseline audit

| Candidate | Verified availability and benchmark decision |
| --- | --- |
| SWAP | Its arXiv record reports a Unitree G1, and its project page still lists code as forthcoming. No public controller and weights sufficient to run its diffusion planner and tracker were found. [Paper](https://arxiv.org/abs/2610.25711), [author project page](https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/) |
| PASSAGE | The project page reports forthcoming code. The perceptive tracker and elevation-map planner weights needed for a faithful rollout are unavailable from the release links. [Paper](https://arxiv.org/abs/2609.18732), [author project page](https://galaxygeneralrobotics.github.io/PASSAGE/) |
| HALO | The repository has G1 MuJoCo and hardware deployment for its proprioceptive and depth trackers. Both trained tracker and planner weights are absent from the release, and the generator recipe requires generated training data. [Tracker](https://github.com/apple-aiml-research/ml-halo/tree/main/tracker), [generator](https://github.com/apple-aiml-research/ml-halo/tree/main/generator) |
| BeyondMimic | The author repository provides tracking training and ONNX export. The later per-motion tracker release's linked model collection returned HTTP 401 to an unauthenticated check. A general obstacle planner and a scene-conditioned tracker checkpoint are needed to reproduce traversal independently. [Author repository](https://github.com/HybridRobotics/whole_body_tracking), [checkpoint link](https://huggingface.co/KRAFTON/physical_ai_motion_tracking_experts) |
| BRACE | The published 29-joint G1 tracker has a public dynamic-batch ONNX student, robot assets and actuator metadata. The authors' recorded motion library is separately distributed. For a tracker control, supply our frozen planner's full pose and joint-velocity references to the published student. The common simulation keeps the same geometry and contact rule. [Code](https://github.com/multiplylabs/brace), [release manifest](https://github.com/multiplylabs/brace/blob/e9168b177fb6687776d549b51beb244f089fb368/manifests/artifacts.json), [weights](https://huggingface.co/sudarshanharithas/BRACE/tree/main/artifacts/student) |

The BRACE control uses zero-wrench compensation, the released PD gains and exact effort limits, with the standard CAT simulator and a 50 Hz control rate. CAT's physics step remains 2 ms and its nominal torque disturbances remain in the harness. The tracker sees its reference and proprioception. Scene perception enters through our common planner. The frozen BRACE ONNX and sidecar are checked against their release hashes. A CPU batch of four returned finite 29-joint targets, and batched rotation conversion passed its round trip. Full evaluation requires a dynamics and CUDA smoke check.

The full-scene preflight caught one colliding U-trap start under CAT's broad random-yaw reset. E4 now prescribes two paired starts at each scene's authored pose, with yaw offsets of ±0.05 rad and initial forward velocity offsets of ±0.015 m/s. The CPU collision checker verified both starts in all 72 layouts. This avoids a checkpoint-dependent jump to a fallback reset. Results retain the initial-state hash to confirm pairing on both compute targets. The upstream observation tests have two known fixture failures reproduced on the pre-overnight code. Thirty other selected planner, energy and observation tests passed.

### Updated E4 and E6 submissions

The pending E4 jobs `319413–319419` were canceled before running after the fixed-start preflight. Input build `319412` is retained. Updated deployment `ac58626` uses the certified starts:

| Job or host | Work | Resources | Limit | ETA after admission |
| --- | --- | --- | --- | --- |
| dep-1 | V7r u600, then released CAT, one at a time after plan-blind local work | RTX 6000 Ada, 4 CPU threads | 14,500 steps | 35–65 min per policy |
| 319447 / 319448 / 319449 | E4 planner / field without reflex / plan-only, three seeds each | gpu, 1 GPU, 8 CPUs, 38000 MB | 3 h | 45–95 min per task |
| 319450 / 319451 | E4 priced / price-free continuation | gpu, 1 GPU, 8 CPUs, 38000 MB | 3 h | 45–95 min per task |
| 319455 | BRACE CUDA smoke, three scenes | gpu, 1 GPU, 8 CPUs, 40000 MB | 45 min | 5–15 min |
| 319456 / 319457 | Frozen BRACE with the common planner, original 384 scenes / E4 | gpu, 1 GPU, 8 CPUs, 40000 MB | 7 h / 5 h | 2–4 h / 1–3 h |

Local E4 outputs are `~/robotics/cat-energy-local-runs/overnight_E4_{reactive,released_cat}_seed0_local_ac58626`. Cluster E4 outputs keep the same convention with new job IDs. E6 outputs go to `/l/users/konstantin.smirnov/runs/overnight_E6_BRACE_<smoke|384|E4>_<job>`. Jobs 319456 and 319457 require a passing 319455. E4 also requires 319412. The BRACE adapter ran 200 finite MuJoCo CPU transitions with a full CAT planner reference before submission. Its separated inference overlay is ONNX Runtime 1.20.1 for CUDA 11.8 and cuDNN 8.9.6.50. Deployment inputs and weights were hash-checked on the cluster.

## E7 standing evidence and remaining jobs

The existing 150-update energy-only and contact-only comparison is single-seed and both runs warmed from the same u150 snapshot. Across the whole benchmark, adding heat price reduced policy heat from 52.42 to 34.61 W and p90 foot displacement from 2.99 to 1.96 cm. The arm-first share fell from 22.93% to 19.00% across recorded threats. Among the 1,090 authored rows with a first threat in both policies, it rose by 2.94 percentage points (paired row bootstrap 95% CI +0.64 to +5.32). Energy-only logged a first threat in 1,098 of 1,334 authored threat rows, compared with 1,308 for contact-only. Exposure imbalance and the reversal across estimands leave the arm-substitution hypothesis unresolved. The lower heat and smaller foot motion are supported by this run.

Standing videos from that comparison are `energy-videos/standing-controls-danger-row-{0000,0003}.mp4` and `energy-videos/standing-controls-anticipation-row-{0001,0004}.mp4` in the project artifacts. Optimal-reference clips are `energy-videos/reference-anticipation-row-0895-arm-lean-step.mp4`, `energy-videos/reference-anticipation-row-0778-arm-lean.mp4` and `energy-videos/reference-danger-row-1752-arm-lean.mp4`.

Standing α250 task 318296_2 and α1000 task 318296_3 will be compared with the 150-update α500, arm×1 result from 318296_1. Matched arm×30 and arm×1 seed 2 tasks 319350_0–1 use the same 150-update start as seed 1 and will be used for replication. The older seed 0 arm×30 extension and short S2H control have different budgets. Their figures are reported separately from equally budgeted paired seed differences.

E1 seed 0 moved to dep-1 while the cluster tasks were still pending. Only our array elements `319389_0,319389_3,319389_6` were canceled. The remaining cluster array is `319389_[1-2,4-5,7-8]`. The three local tasks use the exact `1bc21af` code, seed-0 COMBO_FIX checkpoint, planner, selected goal fields and 384-scene sampling protocol. They run one at a time after the local E4 baselines, at about 1–2 h per task. Outputs are `~/robotics/cat-energy-local-runs/overnight_E1_{fallback,margin,both}_seed0_local_1bc21af`.

A diagnostic field integration on the 72 held-out scenes reached the goal in all three families when it followed only the point-pelvis guidance at 0.75 m. A rigid nominal G1 proxy following the multi-point projection crossed the −4 cm sampled collision threshold in 24/24 U traps and 24/24 sideways-entry scenes. The table route has a certified raised posture, and tucking may also clear some widths. These CPU probes establish a whole-body clearance challenge. The U layouts should therefore be described as geometric dead ends with a point-field detour, without assuming every reactive controller falls into a field minimum.

## Three-seed matched comparison on the original 384 scenes

Round 4 is complete for three planner seeds and three matched reflex-off seeds. Input jobs are 315810/315811 for seed 0, 318295_0/_1 for seed 1 and 318295_2/_3 for seed 2. Each value is mean ± sample s.d. across training seeds. Rates are percentages and paired differences are percentage points with 95% t intervals over the three training seeds.

| Case | Speed / family | clean | contact | refused | hands |
| --- | --- | ---: | ---: | ---: | ---: |
| COMBO_NOREFLEX | 0.15 | 61.94 ± 0.85 | 29.38 ± 1.21 | 8.51 ± 0.64 | 2.60 ± 0.26 |
| COMBO_FIX | 0.15 | 61.07 ± 1.45 | 31.16 ± 0.99 | 7.73 ± 2.08 | 2.60 ± 0.00 |
| COMBO_FIX − COMBO_NOREFLEX | 0.15 | -0.87 ± 1.17 [-3.77, +2.03] | +1.78 ± 1.63 [-2.26, +5.82] | -0.78 ± 2.61 [-7.27, +5.71] | +0.00 ± 0.26 [-0.65, +0.65] |
| COMBO_NOREFLEX | 0.3 | 63.63 ± 0.74 | 27.13 ± 0.92 | 9.11 ± 0.39 | 3.34 ± 0.66 |
| COMBO_FIX | 0.3 | 61.85 ± 1.58 | 28.12 ± 3.07 | 9.81 ± 2.26 | 2.86 ± 0.52 |
| COMBO_FIX − COMBO_NOREFLEX | 0.3 | -1.78 ± 2.10 [-7.01, +3.45] | +1.00 ± 3.94 [-8.79, +10.79] | +0.69 ± 2.57 [-5.69, +7.08] | -0.48 ± 0.66 [-2.11, +1.15] |

The original known-obstacle benchmark gives no clear gain from the 5 cm reflex. Clean traversal is slightly lower on average at both speeds, and hand contact moves by less than half a point. All four paired CIs cross zero at both speeds. This establishes the frozen baseline for E1 refusal recovery and makes the plan-blind test necessary for evaluating the reflex where obstacle information differs.

### E4 frozen baseline results

The two local baselines completed all 144 episodes per speed. Initial pose and velocity hashes match. Each paired interval resamples complete scene pairs within the indicated family. Rates are percent and deltas are percentage points. Faster times condition on the policy's clean arrivals, so they describe different subsets when clean rates differ.

| Case | Speed / family | clean | contact | refused | hands | time to clean goal (s) | episode min palm (cm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V7r | 0.15 | 58.33 | 41.67 | 0.00 | 20.14 | 20.53 | 3.09 |
| Released CAT | 0.15 | 14.58 | 85.42 | 0.00 | 27.08 | 5.70 | 4.38 |
| Released CAT − V7r | 0.15 | -43.75 [-53.47, -33.33] | +43.75 [+33.33, +53.47] | +0.00 [+0.00, +0.00] | +6.94 [+0.00, +14.58] | — | — |
| V7r | 0.15 / early-turn | 10.42 | 89.58 | 0.00 | 54.17 | 11.10 | 0.47 |
| Released CAT | 0.15 / early-turn | 0.00 | 100.00 | 0.00 | 75.00 | — | 0.69 |
| Released CAT − V7r | 0.15 / early-turn | -10.42 [-20.83, -2.08] | +10.42 [+2.08, +20.83] | +0.00 [+0.00, +0.00] | +20.83 [+2.08, +39.58] | — | — |
| V7r | 0.15 / table-gap | 72.92 | 27.08 | 0.00 | 0.00 | 28.73 | 4.42 |
| Released CAT | 0.15 / table-gap | 2.08 | 97.92 | 0.00 | 4.17 | 13.66 | 0.35 |
| Released CAT − V7r | 0.15 / table-gap | -70.83 [-83.33, -56.25] | +70.83 [+56.25, +83.33] | +0.00 [+0.00, +0.00] | +4.17 [+0.00, +10.42] | — | — |
| V7r | 0.15 / u-trap | 91.67 | 8.33 | 0.00 | 6.25 | 15.08 | 4.38 |
| Released CAT | 0.15 / u-trap | 41.67 | 58.33 | 0.00 | 2.08 | 5.30 | 12.11 |
| Released CAT − V7r | 0.15 / u-trap | -50.00 [-66.67, -33.33] | +50.00 [+33.33, +66.67] | +0.00 [+0.00, +0.00] | -4.17 [-12.50, +4.17] | — | — |
| V7r | 0.3 | 56.25 | 43.75 | 0.00 | 23.61 | 15.82 | 3.04 |
| Released CAT | 0.3 | 18.75 | 79.86 | 1.39 | 22.92 | 5.09 | 4.46 |
| Released CAT − V7r | 0.3 | -37.50 [-46.53, -28.47] | +36.11 [+27.08, +45.14] | +1.39 [+0.00, +3.47] | -0.69 [-5.56, +4.17] | — | — |
| V7r | 0.3 / early-turn | 2.08 | 97.92 | 0.00 | 70.83 | 9.78 | 0.22 |
| Released CAT | 0.3 / early-turn | 0.00 | 100.00 | 0.00 | 64.58 | — | 1.00 |
| Released CAT − V7r | 0.3 / early-turn | -2.08 [-6.25, +0.00] | +2.08 [+0.00, +6.25] | +0.00 [+0.00, +0.00] | -6.25 [-20.83, +8.33] | — | — |
| V7r | 0.3 / table-gap | 66.67 | 33.33 | 0.00 | 0.00 | 22.41 | 4.27 |
| Released CAT | 0.3 / table-gap | 0.00 | 97.92 | 2.08 | 4.17 | — | 0.20 |
| Released CAT − V7r | 0.3 / table-gap | -66.67 [-79.17, -54.17] | +64.58 [+52.08, +77.08] | +2.08 [+0.00, +6.25] | +4.17 [+0.00, +10.42] | — | — |
| V7r | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 11.56 | 4.64 |
| Released CAT | 0.3 / u-trap | 56.25 | 41.67 | 2.08 | 0.00 | 5.09 | 12.17 |
| Released CAT − V7r | 0.3 / u-trap | -43.75 [-58.33, -27.08] | +41.67 [+25.00, +58.33] | +2.08 [+0.00, +6.25] | +0.00 [+0.00, +0.00] | — | — |
| V7r | 0.5 | 52.08 | 46.53 | 1.39 | 26.39 | 12.88 | 2.91 |
| Released CAT | 0.5 | 12.50 | 87.50 | 0.00 | 24.31 | 4.92 | 5.09 |
| Released CAT − V7r | 0.5 | -39.58 [-49.31, -29.86] | +40.97 [+31.25, +50.69] | -1.39 [-3.47, +0.00] | -2.08 [-8.33, +4.17] | — | — |
| V7r | 0.5 / early-turn | 6.25 | 93.75 | 0.00 | 72.92 | 7.19 | 0.28 |
| Released CAT | 0.5 / early-turn | 0.00 | 100.00 | 0.00 | 72.92 | — | 2.70 |
| Released CAT − V7r | 0.5 / early-turn | -6.25 [-16.67, +0.00] | +6.25 [+0.00, +16.67] | +0.00 [+0.00, +0.00] | +0.00 [-16.67, +16.67] | — | — |
| V7r | 0.5 / table-gap | 66.67 | 33.33 | 0.00 | 2.08 | 17.57 | 3.96 |
| Released CAT | 0.5 / table-gap | 0.00 | 100.00 | 0.00 | 0.00 | — | 0.55 |
| Released CAT − V7r | 0.5 / table-gap | -66.67 [-79.17, -52.08] | +66.67 [+52.08, +79.17] | +0.00 [+0.00, +0.00] | -2.08 [-6.25, +0.00] | — | — |
| V7r | 0.5 / u-trap | 83.33 | 12.50 | 4.17 | 4.17 | 9.56 | 4.48 |
| Released CAT | 0.5 / u-trap | 37.50 | 62.50 | 0.00 | 0.00 | 4.92 | 12.01 |
| Released CAT − V7r | 0.5 / u-trap | -45.83 [-62.50, -29.17] | +50.00 [+33.33, +66.67] | -4.17 [-10.42, +0.00] | -4.17 [-10.42, +0.00] | — | — |

At 0.50 m/s, V7r already clears 40/48 U traps with no contact, while only 3/48 turn-before-entry cases are clean. The early-turn family is the stronger room for a planning gain on this common field-guidance protocol. The two local launches took about four minutes apiece with first-goal termination. The planner variants still need comparable rollouts.

### E7 first α sensitivity result

| Heat coefficient, same training seed | Threats | Feet moved, % of worlds | Arm-first, % | Body-first, % | Failure, % | Foot p90, cm | Pelvis p90, cm | Hand retreat p90, cm | Policy mechanical W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| α250, 318296_2 | 3945 | 5.45 | 43.68 | 0.35 | 9.46 | 1.64 | 4.97 | 16.27 | 21.38 |
| α500, 318296_1 | 4026 | 2.05 | 38.85 | 0.35 | 11.90 | 1.37 | 4.91 | 14.43 | 19.38 |

Both runs finished 150 updates from S2 seed 1 u180 and had zero falls in the benchmark. Mechanical watts are comparable across α values. Reported heat uses each checkpoint's α and scales the holding contribution accordingly, so the raw heat numbers do not isolate a behavioral change. The α1000 run 318296_3 started on gpu.

E4 seed 0 for COMBO_FIX and COMBO_NOREFLEX moved to dep-1 while tasks 319447_0 and 319448_0 were pending. Those two tasks were canceled. The other seeds remain on gpu. The local run uses `ac58626` and the certified E4 bank after the active E1 fallback run releases the GPU lock. Its outputs are `~/robotics/cat-energy-local-runs/overnight_E4_{planner,field_no_reflex}_seed0_local_ac58626`.

### E1 seed-0 stall fallback pilot

The local 384-scene run completed both speeds at 768 episodes each. Intervals resample paired scene rows and describe the uncertainty for this checkpoint. Faster fallback may trade refusal for contact. The slower command showed a clean gain and lower mean contact.

| Case | Speed / family | clean | contact | refused | hands |
| --- | --- | ---: | ---: | ---: | ---: |
| COMBO_FIX | 0.15 | 59.77 | 30.08 | 10.03 | 2.60 |
| Stall fallback | 0.15 | 63.15 | 27.60 | 9.24 | 2.73 |
| Stall fallback − COMBO_FIX | 0.15 | +3.39 [+1.04, +5.86] | -2.47 [-5.73, +0.91] | -0.78 [-3.26, +1.70] | +0.13 [-1.43, +1.69] |
| COMBO_FIX | 0.3 | 61.59 | 25.52 | 12.37 | 2.34 |
| Stall fallback | 0.3 | 61.46 | 28.91 | 9.51 | 2.60 |
| Stall fallback − COMBO_FIX | 0.3 | -0.13 [-2.73, +2.60] | +3.39 [+0.52, +6.25] | -2.86 [-4.95, -0.78] | +0.26 [-0.91, +1.43] |

At 0.30 m/s, the lower refusal rate came with 3.39 pp more contact, so fallback alone fails the prespecified clean/contact criterion for this seed at that speed. At 0.15 m/s, clean traversal gained 3.39 pp. The remaining seeds and candidate filter are needed before selection.

E0 seed 0 moved to dep-1 using the identical one-episode audit protocol and deployment `1bc21af`. Only our pending `319388_0` was canceled. The six remaining E1 tasks now depend on `319388_1` and `319388_2` individually, and Slurm confirmed the revised dependencies. The local GPU lock queues the audit behind existing local work. Its output is `~/robotics/cat-energy-local-runs/overnight_E0_baseline_seed0_local_1bc21af`.

### E3 complete four-protocol pilot, seed 0

Both runs completed 768 episodes per protocol using the same obstacle schedule and validated matching initial state. Reflex-on ran on dep-1, and reflex-off ran on ws-ia. These intervals resample paired scenes for this seed. The other two seed pairs still run.

| Case | Speed / family | clean | contact | refused | hands | inserted_hand_contact | inserted_revealed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reflex off | static_0.15 | 32.42 | 60.16 | 7.29 | 21.35 | 18.10 | 98.57 |
| Reflex on | static_0.15 | 31.12 | 57.03 | 11.72 | 19.14 | 16.80 | 98.96 |
| Reflex on − Reflex off | static_0.15 | -1.30 [-4.17, +1.43] | -3.12 [-6.64, +0.39] | +4.43 [+2.08, +6.77] | -2.21 [-5.08, +0.52] | -1.30 [-3.91, +1.17] | +0.39 [+0.00, +0.91] |
| Reflex off | static_0.3 | 34.51 | 55.47 | 9.90 | 13.02 | 10.42 | 98.70 |
| Reflex on | static_0.3 | 34.11 | 53.65 | 11.85 | 12.11 | 8.33 | 98.83 |
| Reflex on − Reflex off | static_0.3 | -0.39 [-3.26, +2.47] | -1.82 [-5.47, +1.82] | +1.95 [-0.52, +4.56] | -0.91 [-3.52, +1.69] | -2.08 [-4.30, +0.13] | +0.13 [+0.00, +0.39] |
| Reflex off | moving_0.15 | 32.55 | 59.24 | 8.07 | 12.50 | 9.11 | 98.57 |
| Reflex on | moving_0.15 | 31.38 | 56.38 | 12.11 | 12.63 | 10.29 | 98.96 |
| Reflex on − Reflex off | moving_0.15 | -1.17 [-3.91, +1.56] | -2.86 [-6.25, +0.52] | +4.04 [+1.95, +6.12] | +0.13 [-2.47, +2.73] | +1.17 [-1.17, +3.52] | +0.39 [+0.00, +0.91] |
| Reflex off | moving_0.3 | 34.51 | 55.73 | 9.64 | 13.02 | 8.46 | 98.57 |
| Reflex on | moving_0.3 | 34.24 | 54.04 | 11.72 | 10.16 | 7.29 | 98.70 |
| Reflex on − Reflex off | moving_0.3 | -0.26 [-3.12, +2.47] | -1.69 [-4.95, +1.69] | +2.08 [-0.13, +4.17] | -2.86 [-5.47, -0.26] | -1.17 [-3.39, +1.04] | +0.13 [-0.26, +0.65] |

Across all four protocols, contact fell by 1.69 to 3.12 pp and clean traversal fell by 0.26 to 1.30 pp. Refusals rose by 1.95 to 4.43 pp. The inserted object's hand contact fell in three protocols and rose in the slower moving case. All four inserted-hand-contact intervals cross zero. Exposure was above 98.5% in each condition. Plots or video examples of avoidance should retain this trade-off.

### E4 first planner seed versus V7r

Same validated initial-state hash and 144 episodes per speed. Paired 95% intervals resample whole scene rows for this one training seed. The 0.50 m/s clean change on U traps and sideways-entry layouts combined was +3.12 pp, below the +5 pp prediction for the benchmark. On all layouts the 0.50 m/s policy reduced hand contact but lost clean traversal. Table gaps accounted for 10 of its 11 refusals. Those 10 refusals spanned widths from 0.460 to 0.578 m. Early-turn hand contact fell by 20.83 pp, while total contact stayed high. This points to two separate bottlenecks for the remaining comparisons.

| Case | Speed / family | clean | contact | refused | hands | time to clean goal (s) | episode min palm (cm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V7r | 0.15 | 58.33 | 41.67 | 0.00 | 20.14 | 20.53 | 3.09 |
| COMBO_FIX | 0.15 | 54.86 | 43.06 | 2.08 | 27.08 | 30.91 | 3.58 |
| COMBO_FIX − V7r | 0.15 | -3.47 [-9.72, +3.47] | +1.39 [-5.56, +8.33] | +2.08 [+0.00, +5.56] | +6.94 [+0.69, +13.19] | — | — |
| V7r | 0.15 / early-turn | 10.42 | 89.58 | 0.00 | 54.17 | 11.10 | 0.47 |
| COMBO_FIX | 0.15 / early-turn | 10.42 | 89.58 | 0.00 | 79.17 | 18.29 | -0.02 |
| COMBO_FIX − V7r | 0.15 / early-turn | +0.00 [-8.33, +8.33] | +0.00 [-8.33, +8.33] | +0.00 [+0.00, +0.00] | +25.00 [+10.42, +39.58] | — | — |
| V7r | 0.15 / table-gap | 72.92 | 27.08 | 0.00 | 0.00 | 28.73 | 4.42 |
| COMBO_FIX | 0.15 / table-gap | 58.33 | 35.42 | 6.25 | 2.08 | 52.71 | 3.06 |
| COMBO_FIX − V7r | 0.15 / table-gap | -14.58 [-27.08, +0.00] | +8.33 [-6.25, +22.92] | +6.25 [+0.00, +16.67] | +2.08 [+0.00, +6.25] | — | — |
| V7r | 0.15 / u-trap | 91.67 | 8.33 | 0.00 | 6.25 | 15.08 | 4.38 |
| COMBO_FIX | 0.15 / u-trap | 95.83 | 4.17 | 0.00 | 0.00 | 19.01 | 7.70 |
| COMBO_FIX − V7r | 0.15 / u-trap | +4.17 [-6.25, +14.58] | -4.17 [-14.58, +6.25] | +0.00 [+0.00, +0.00] | -6.25 [-14.58, +0.00] | — | — |
| V7r | 0.3 | 56.25 | 43.75 | 0.00 | 23.61 | 15.82 | 3.04 |
| COMBO_FIX | 0.3 | 52.08 | 44.44 | 3.47 | 20.14 | 21.14 | 4.04 |
| COMBO_FIX − V7r | 0.3 | -4.17 [-10.42, +1.39] | +0.69 [-5.56, +6.94] | +3.47 [+0.69, +6.94] | -3.47 [-9.72, +2.78] | — | — |
| V7r | 0.3 / early-turn | 2.08 | 97.92 | 0.00 | 70.83 | 9.78 | 0.22 |
| COMBO_FIX | 0.3 / early-turn | 2.08 | 97.92 | 0.00 | 60.42 | 9.92 | 0.01 |
| COMBO_FIX − V7r | 0.3 / early-turn | +0.00 [-6.25, +6.25] | +0.00 [-6.25, +6.25] | +0.00 [+0.00, +0.00] | -10.42 [-27.08, +8.33] | — | — |
| V7r | 0.3 / table-gap | 66.67 | 33.33 | 0.00 | 0.00 | 22.41 | 4.27 |
| COMBO_FIX | 0.3 / table-gap | 54.17 | 35.42 | 10.42 | 0.00 | 36.87 | 3.78 |
| COMBO_FIX − V7r | 0.3 / table-gap | -12.50 [-29.17, +4.17] | +2.08 [-14.58, +18.75] | +10.42 [+2.08, +20.83] | +0.00 [+0.00, +0.00] | — | — |
| V7r | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 11.56 | 4.64 |
| COMBO_FIX | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 12.85 | 8.34 |
| COMBO_FIX − V7r | 0.3 / u-trap | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | — | — |
| V7r | 0.5 | 52.08 | 46.53 | 1.39 | 26.39 | 12.88 | 2.91 |
| COMBO_FIX | 0.5 | 46.53 | 45.83 | 7.64 | 18.06 | 15.56 | 3.76 |
| COMBO_FIX − V7r | 0.5 | -5.56 [-14.58, +2.78] | -0.69 [-8.33, +6.94] | +6.25 [+1.39, +11.11] | -8.33 [-14.58, -2.78] | — | — |
| V7r | 0.5 / early-turn | 6.25 | 93.75 | 0.00 | 72.92 | 7.19 | 0.28 |
| COMBO_FIX | 0.5 / early-turn | 4.17 | 95.83 | 0.00 | 52.08 | 9.88 | 0.22 |
| COMBO_FIX − V7r | 0.5 / early-turn | -2.08 [-6.25, +0.00] | +2.08 [+0.00, +6.25] | +0.00 [+0.00, +0.00] | -20.83 [-35.42, -6.25] | — | — |
| V7r | 0.5 / table-gap | 66.67 | 33.33 | 0.00 | 2.08 | 17.57 | 3.96 |
| COMBO_FIX | 0.5 / table-gap | 43.75 | 35.42 | 20.83 | 0.00 | 24.55 | 3.18 |
| COMBO_FIX − V7r | 0.5 / table-gap | -22.92 [-41.67, -4.17] | +2.08 [-16.67, +20.83] | +20.83 [+10.42, +31.25] | -2.08 [-6.25, +0.00] | — | — |
| V7r | 0.5 / u-trap | 83.33 | 12.50 | 4.17 | 4.17 | 9.56 | 4.48 |
| COMBO_FIX | 0.5 / u-trap | 91.67 | 6.25 | 2.08 | 2.08 | 11.53 | 7.87 |
| COMBO_FIX − V7r | 0.5 / u-trap | +8.33 [-8.33, +25.00] | -6.25 [-18.80, +6.25] | -2.08 [-8.33, +4.17] | -2.08 [-8.33, +4.17] | — | — |

### E1 seed-0 clearance filter pilot

The clearance filter reduced refusal by 2.86 pp at both speeds. At 0.30 m/s clean traversal was flat and contact rose 2.99 pp. At 0.15 m/s clean gained 1.56 pp with a scene interval spanning zero. Neither run provides a basis for choosing the filter alone under the joint clean/contact criterion.

| Case | Speed / family | clean | contact | refused | hands |
| --- | --- | ---: | ---: | ---: | ---: |
| COMBO_FIX | 0.15 | 59.77 | 30.08 | 10.03 | 2.60 |
| Clearance filter | 0.15 | 61.33 | 30.99 | 7.16 | 3.12 |
| Clearance filter − COMBO_FIX | 0.15 | +1.56 [-0.91, +4.04] | +0.91 [-2.47, +4.30] | -2.86 [-5.34, -0.39] | +0.52 [-1.04, +1.95] |
| COMBO_FIX | 0.3 | 61.59 | 25.52 | 12.37 | 2.34 |
| Clearance filter | 0.3 | 61.46 | 28.52 | 9.51 | 3.26 |
| Clearance filter − COMBO_FIX | 0.3 | -0.13 [-2.86, +2.60] | +2.99 [+0.00, +5.99] | -2.86 [-5.47, -0.39] | +0.91 [-0.52, +2.34] |

The seed-0 plan-blind videos are rendering from exact paired rollout traces on dep-1's software EGL backend. This frees the single GPU for the active audit. The first render exposed a temporary mesh lifetime bug. Commit `aff003b` compiles the model before releasing its mesh assets.

### E3 paired videos

Three original seed-0 episode pairs are in the project `energy-videos/` artifacts folder. They use synchronized recorded poses, the same camera and the authored inserted box in both panels. Blue shows the plan. Orange marks a hand within the 5 cm band used to remove that arm's tracking reward in training. Software EGL kept the local GPU available for evaluation. All clips passed checks for 1280×720 H.264 yuv420p, stereo 48 kHz silent AAC and faststart atom order.

| File | Scenario | Clip duration | Inserted hand contact |
| --- | --- | ---: | --- |
| `plan-blind-static-015.mp4` | Static at 0.15 m/s, angled 0.51 m side gap, row 370 | 15.00 s | Neither controller |
| `plan-blind-moving-015.mp4` | Moving at 0.15 m/s, angled 0.51 m side gap, row 370 | 13.80 s | Neither controller |
| `plan-blind-moving-030.mp4` | Moving at 0.30 m/s, cabinet passage, row 62 | 11.72 s | Reflex off only |

The selected traces show 20, 3 and 17 control transitions within the training reflex band, respectively. Selection favored visible reflex activity. The third clip shows a paired example of avoiding the inserted-object hand contact.

### E7 complete heat-price sensitivity

All cases used 150 updates from `stationary_hands_S2_seed1_u180.pt` with training seed 1, then the same 2,000-world standing benchmark for 800 steps. Feet moved is the share of worlds with a shadow stance violation. Response shares and displacement quantiles use danger and anticipation events. All three recorded zero falls. Use mechanical group watts for cross-α comparison because the size-weighted heat meter scales its holding term with the tested α.

| α, W | Job / seed | Threats | Feet moved, % | Arm-first, % | Body-first, % | Failure, % | Feet p50/p90/p95, cm | Pelvis p50/p90/p95, cm | Retreat p50/p90/p95, cm | Policy mechanical W, legs/waist/arms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 250 | 318296_2 / 1 | 3945 | 5.45 | 43.68 | 0.35 | 9.46 | 0.72 / 1.64 / 2.14 | 2.97 / 4.97 / 5.83 | 10.00 / 16.27 / 18.11 | 11.07 / 3.51 / 6.79 |
| 500 | 318296_1 / 1 | 4026 | 2.05 | 38.85 | 0.35 | 11.90 | 0.64 / 1.37 / 1.68 | 2.93 / 4.91 / 5.92 | 9.32 / 14.43 / 15.93 | 10.16 / 3.35 / 5.88 |
| 1000 | 318296_3 / 1 | 4038 | 11.85 | 33.06 | 0.94 | 14.19 | 0.72 / 1.92 / 2.62 | 2.86 / 4.94 / 5.82 | 8.52 / 14.66 / 16.65 | 9.70 / 3.14 / 5.51 |

For this training seed, stronger heat moved the measured arm-first share down and the failure share up. The 500 W run had the smallest feet-moved share, and body-first remained below 1% even at 1,000 W. This is a dose response at one training seed. The third arm-multiplier replication is the remaining standing comparison.

### E0 window audit, seed 0

All 384 layouts completed one nominal episode at each speed. The first complete stall window was available for all 27 refusals at 0.15 m/s and 43 of 50 at 0.30 m/s. A complete before-window was available for 26 and 40 episodes. Furniture had no refused episodes with these windows. Values are weighted reward terms per control step.

| Speed | Scene group | Window | Episodes across seeds | Stall | Tracking | Clearance | Field | Posture and motion | Energy | Goal progress |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | Side gaps | before | 6 | -0.52 | 2.58 | -0.10 | 2.07 | 5.32 | -0.63 | 2.76 |
| 0.15 | Side gaps | during | 6 | -1.64 | 3.28 | -0.13 | 2.14 | 4.29 | -0.56 | 1.49 |
| 0.15 | Cabinets | before | 5 | 0.00 | 6.01 | -0.01 | 2.27 | 5.94 | -0.64 | 3.72 |
| 0.15 | Cabinets | during | 5 | -1.56 | 5.28 | -0.03 | 2.37 | 5.96 | -0.64 | 1.38 |
| 0.15 | Furniture | before | 0 | — | — | — | — | — | — | — |
| 0.15 | Furniture | during | 0 | — | — | — | — | — | — | — |
| 0.15 | Clutter | before | 1 | 0.00 | 2.97 | -0.00 | 2.29 | 5.67 | -0.54 | 0.00 |
| 0.15 | Clutter | during | 1 | 0.00 | 1.79 | -0.00 | 2.27 | 5.83 | -0.45 | 0.00 |
| 0.15 | CAT courses | before | 14 | -0.57 | 4.07 | -0.02 | 2.17 | 4.84 | -0.65 | 3.08 |
| 0.15 | CAT courses | during | 15 | -1.61 | 3.55 | -0.06 | 2.14 | 3.53 | -0.64 | 1.34 |
| 0.3 | Side gaps | before | 10 | 0.00 | 4.53 | -0.12 | 2.41 | 5.75 | -0.64 | 2.85 |
| 0.3 | Side gaps | during | 10 | -2.65 | 3.13 | -0.08 | 2.21 | 5.20 | -0.56 | 0.67 |
| 0.3 | Cabinets | before | 8 | 0.00 | 5.29 | -0.02 | 2.07 | 5.65 | -0.73 | 3.69 |
| 0.3 | Cabinets | during | 8 | -2.09 | 3.89 | -0.05 | 2.28 | 5.87 | -0.63 | 0.69 |
| 0.3 | Furniture | before | 0 | — | — | — | — | — | — | — |
| 0.3 | Furniture | during | 0 | — | — | — | — | — | — | — |
| 0.3 | Clutter | before | 1 | 0.00 | 2.03 | -0.00 | 2.38 | 5.75 | -0.64 | 0.00 |
| 0.3 | Clutter | during | 1 | 0.00 | 3.41 | -0.00 | 2.31 | 5.86 | -0.50 | 0.00 |
| 0.3 | CAT courses | before | 21 | -0.28 | 3.61 | -0.03 | 2.23 | 5.44 | -0.77 | 2.79 |
| 0.3 | CAT courses | during | 24 | -2.17 | 3.19 | -0.05 | 2.18 | 4.90 | -0.76 | 0.78 |

The largest negative term before stalls was global power, −0.64 and −0.73 at 0.15 and 0.30 m/s. During stalls the largest negative was the stall term, −1.55 and −2.22. Goal progress dropped across side gaps, cabinets and CAT courses, while their summed tracking, posture and field terms stayed positive. The clearance cost remained small. This pattern supports testing a reward balance intervention if the frozen recovery variants keep trading refusals for contacts. It gives an association at stall onset conditional on eventual refusal. Seeds 1 and 2 are needed for the full diagnosis.

### E2 prepared conditional training protocol

The optional profile prices eight consecutive seconds without a 5 cm net route advance at a terminal cost at least as large as the configured obstacle-contact event. The moving planner must be past its first second. An unfinished moving-plan episode at its existing deadline uses the same terminal cost. When the stall clock reaches two seconds, positive upright, stand-tall, plan position, heading, height and upper-body tracking bonuses pause. Field guidance and penalties remain active. This tests the largest positive balances observed during stalls. Negative head, feet, knees, shoulders, hand and arm clearance in side gaps, cabinets, width scaffolds and table passages scales with pelvis distance. A cost integrated over a metre has the same scale as progress at the 0.30 m/s reference. The profile is disabled in frozen evaluation to preserve episode budgets across checkpoints.

If the three-seed E0 results confirm the persistent balance and none of the E1 variants raises clean traversal without increasing contact at both speeds, warm-start all three current COMBO_FIX seeds for 150 updates apiece. The existing anchor and recipe remain the controls. Predict at least a 2 pp mean gain in clean traversal and a 2 pp drop in refusal with no increase in mean contact at either speed. Evaluate the matched seeds with the original 384 scenes and the standard budget, then check the selected checkpoint family on E4. Thirty selected CPU tests passed. A MuJoCo CPU smoke produced the expected −1 terminal event and suppressed the watched positive stall bonuses.

### E4 first matched reflex ablation seed

Both planner conditions matched the initial-state hash of the two frozen baselines and completed all 144 paired episodes per speed. At 0.50 m/s, reflex activation changed overall hand contact by −4.17 pp and refusals by +6.25 pp, with equal total contact. Table-gap refusals increased from 2 to 10 of 48 episodes. At 0.30 m/s, clean traversal fell by 7.64 pp with a scene-resampling interval from −13.89 to −1.39. These intervals pertain to one training seed. Seeds 1 and 2 remain in the GPU queue.

| Case | Speed / family | clean | contact | refused | hands | time to clean goal (s) | episode min palm (cm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reflex off | 0.15 | 59.03 | 40.28 | 0.69 | 24.31 | 28.57 | 3.30 |
| Reflex on | 0.15 | 54.86 | 43.06 | 2.08 | 27.08 | 30.91 | 3.58 |
| Reflex on − Reflex off | 0.15 | -4.17 [-9.72, +1.39] | +2.78 [-4.17, +9.03] | +1.39 [-1.39, +4.86] | +2.78 [-3.47, +9.03] | — | — |
| Reflex off | 0.15 / early-turn | 6.25 | 93.75 | 0.00 | 72.92 | 17.89 | 0.02 |
| Reflex on | 0.15 / early-turn | 10.42 | 89.58 | 0.00 | 79.17 | 18.29 | -0.02 |
| Reflex on − Reflex off | 0.15 / early-turn | +4.17 [-4.17, +12.50] | -4.17 [-12.50, +4.17] | +0.00 [+0.00, +0.00] | +6.25 [-10.42, +25.00] | — | — |
| Reflex off | 0.15 / table-gap | 70.83 | 27.08 | 2.08 | 0.00 | 43.51 | 2.27 |
| Reflex on | 0.15 / table-gap | 58.33 | 35.42 | 6.25 | 2.08 | 52.71 | 3.06 |
| Reflex on − Reflex off | 0.15 / table-gap | -12.50 [-25.00, +0.00] | +8.33 [-8.33, +25.00] | +4.17 [-4.17, +14.58] | +2.08 [+0.00, +6.25] | — | — |
| Reflex off | 0.15 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 18.66 | 7.62 |
| Reflex on | 0.15 / u-trap | 95.83 | 4.17 | 0.00 | 0.00 | 19.01 | 7.70 |
| Reflex on − Reflex off | 0.15 / u-trap | -4.17 [-10.42, +0.00] | +4.17 [+0.00, +10.42] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | — | — |
| Reflex off | 0.3 | 59.72 | 40.28 | 0.00 | 22.92 | 18.47 | 3.51 |
| Reflex on | 0.3 | 52.08 | 44.44 | 3.47 | 20.14 | 21.14 | 4.04 |
| Reflex on − Reflex off | 0.3 | -7.64 [-13.89, -1.39] | +4.17 [-2.08, +11.11] | +3.47 [+0.69, +6.94] | -2.78 [-9.05, +3.47] | — | — |
| Reflex off | 0.3 / early-turn | 8.33 | 91.67 | 0.00 | 66.67 | 10.89 | 0.37 |
| Reflex on | 0.3 / early-turn | 2.08 | 97.92 | 0.00 | 60.42 | 9.92 | 0.01 |
| Reflex on − Reflex off | 0.3 / early-turn | -6.25 [-14.58, +0.00] | +6.25 [+0.00, +14.58] | +0.00 [+0.00, +0.00] | -6.25 [-25.00, +12.50] | — | — |
| Reflex off | 0.3 / table-gap | 70.83 | 29.17 | 0.00 | 2.08 | 26.90 | 2.14 |
| Reflex on | 0.3 / table-gap | 54.17 | 35.42 | 10.42 | 0.00 | 36.87 | 3.78 |
| Reflex on − Reflex off | 0.3 / table-gap | -16.67 [-35.42, +0.00] | +6.25 [-12.50, +25.00] | +10.42 [+2.08, +20.83] | -2.08 [-6.25, +0.00] | — | — |
| Reflex off | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 13.12 | 8.03 |
| Reflex on | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 12.85 | 8.34 |
| Reflex on − Reflex off | 0.3 / u-trap | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | — | — |
| Reflex off | 0.5 | 52.78 | 45.83 | 1.39 | 22.22 | 14.38 | 3.50 |
| Reflex on | 0.5 | 46.53 | 45.83 | 7.64 | 18.06 | 15.56 | 3.76 |
| Reflex on − Reflex off | 0.5 | -6.25 [-13.89, +1.39] | +0.00 [-7.64, +7.64] | +6.25 [+2.78, +10.42] | -4.17 [-12.50, +4.17] | — | — |
| Reflex off | 0.5 / early-turn | 4.17 | 95.83 | 0.00 | 66.67 | 11.57 | 0.49 |
| Reflex on | 0.5 / early-turn | 4.17 | 95.83 | 0.00 | 52.08 | 9.88 | 0.22 |
| Reflex on − Reflex off | 0.5 / early-turn | +0.00 [-8.33, +10.42] | +0.00 [-10.42, +8.33] | +0.00 [+0.00, +0.00] | -14.58 [-39.58, +8.33] | — | — |
| Reflex off | 0.5 / table-gap | 60.42 | 35.42 | 4.17 | 0.00 | 18.20 | 2.40 |
| Reflex on | 0.5 / table-gap | 43.75 | 35.42 | 20.83 | 0.00 | 24.55 | 3.18 |
| Reflex on − Reflex off | 0.5 / table-gap | -16.67 [-33.33, +0.00] | +0.00 [-16.67, +16.67] | +16.67 [+8.33, +27.08] | +0.00 [+0.00, +0.00] | — | — |
| Reflex off | 0.5 / u-trap | 93.75 | 6.25 | 0.00 | 0.00 | 12.04 | 7.60 |
| Reflex on | 0.5 / u-trap | 91.67 | 6.25 | 2.08 | 2.08 | 11.53 | 7.87 |
| Reflex on − Reflex off | 0.5 / u-trap | -2.08 [-12.50, +8.33] | +0.00 [-10.42, +10.42] | +2.08 [+0.00, +6.25] | +2.08 [+0.00, +6.25] | — | — |

At 06:02 Dubai, the pending E0 seed 1 audit also moved to dep-1. Its exact warm-trained COMBO_FIX weights were copied from 318295_0, and the matching audit output is `~/robotics/cat-energy-local-runs/overnight_E0_baseline_seed1_local_1bc21af`. Only 319388_1 was canceled. Slurm E1's six remaining tasks now wait on `319388_2`, the single remaining remote audit. The pending E4 plan-only seed 0 evaluation moved from 319449_0 to `~/robotics/cat-energy-local-runs/overnight_E4_plan_only_seed0_local_ac58626`. Seeds 1 and 2 remain on gpu. Both local runs queue behind the active E1 combined variant on the single-GPU lock.

### E1 complete three-variant pilot, seed 0

The combined fallback and clearance filter increased clean traversal by 3.26 pp at 0.15 m/s with a lower contact point estimate. At 0.30 m/s it reduced refusal by 5.47 pp and increased contacts by 3.91 pp. All three E1 variants raised the faster command's mean contact rate on seed 0. This strengthens the motivation to test the conditional reward intervention. The two remaining seed pairs determine whether any frozen variant passes the joint criterion.

| Case | Speed / family | clean | contact | refused | hands |
| --- | --- | ---: | ---: | ---: | ---: |
| COMBO_FIX | 0.15 | 59.77 | 30.08 | 10.03 | 2.60 |
| Fallback and clearance filter | 0.15 | 63.02 | 28.39 | 8.07 | 2.86 |
| Fallback and clearance filter − COMBO_FIX | 0.15 | +3.26 [+0.91, +5.86] | -1.69 [-4.95, +1.56] | -1.95 [-4.56, +0.65] | +0.26 [-1.30, +1.82] |
| COMBO_FIX | 0.3 | 61.59 | 25.52 | 12.37 | 2.34 |
| Fallback and clearance filter | 0.3 | 63.54 | 29.43 | 6.90 | 3.26 |
| Fallback and clearance filter − COMBO_FIX | 0.3 | +1.95 [-0.78, +4.56] | +3.91 [+0.78, +7.29] | -5.47 [-8.20, -2.86] | +0.91 [-0.39, +2.21] |


The reflex on/off comparison uses separately fine-tuned policies. The clearance band masks an arm's plan-tracking reward during training. During evaluation the actor remains a single network conditioned on plan and field observations. Commit `8f7a36d` clarifies the orange label on the three replays. The same selected clips were re-encoded and verified again.

### E5 frozen field-input pilot on E4

The plan-only masked actor completed all episodes with contact, yielding zero clean traversal at each command speed for seed 0. Restoring the field observations to the same frozen reflex-off weights yielded clean rates of 59.03%, 59.72% and 52.78% at 0.15, 0.30 and 0.50 m/s. The controlled intervention measures dependence on the field inputs of this trained actor. Masking them changes its observation distribution, so a trained standalone tracker remains a separate baseline. The first public BRACE GPU smoke is now running locally.

| Case | Speed / family | clean | contact | refused | hands | time to clean goal (s) | episode min palm (cm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plan-only actor | 0.15 | 0.00 | 100.00 | 0.00 | 45.14 | — | -0.24 |
| Plan and field | 0.15 | 59.03 | 40.28 | 0.69 | 24.31 | 28.57 | 3.30 |
| Plan and field − Plan-only actor | 0.15 | +59.03 [+47.92, +69.44] | -59.72 [-70.14, -48.61] | +0.69 [+0.00, +2.08] | -20.83 [-34.72, -6.94] | — | — |
| Plan-only actor | 0.15 / early-turn | 0.00 | 100.00 | 0.00 | 35.42 | — | 0.18 |
| Plan and field | 0.15 / early-turn | 6.25 | 93.75 | 0.00 | 72.92 | 17.89 | 0.02 |
| Plan and field − Plan-only actor | 0.15 / early-turn | +6.25 [+0.00, +12.55] | -6.25 [-12.55, +0.00] | +0.00 [+0.00, +0.00] | +37.50 [+18.75, +54.17] | — | — |
| Plan-only actor | 0.15 / table-gap | 0.00 | 100.00 | 0.00 | 20.83 | — | -0.74 |
| Plan and field | 0.15 / table-gap | 70.83 | 27.08 | 2.08 | 0.00 | 43.51 | 2.27 |
| Plan and field − Plan-only actor | 0.15 / table-gap | +70.83 [+54.17, +85.42] | -72.92 [-87.50, -56.25] | +2.08 [+0.00, +6.25] | -20.83 [-35.42, -8.33] | — | — |
| Plan-only actor | 0.15 / u-trap | 0.00 | 100.00 | 0.00 | 79.17 | — | -0.15 |
| Plan and field | 0.15 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 18.66 | 7.62 |
| Plan and field − Plan-only actor | 0.15 / u-trap | +100.00 [+100.00, +100.00] | -100.00 [-100.00, -100.00] | +0.00 [+0.00, +0.00] | -79.17 [-89.58, -66.67] | — | — |
| Plan-only actor | 0.3 | 0.00 | 100.00 | 0.00 | 41.67 | — | -0.25 |
| Plan and field | 0.3 | 59.72 | 40.28 | 0.00 | 22.92 | 18.47 | 3.51 |
| Plan and field − Plan-only actor | 0.3 | +59.72 [+49.31, +70.14] | -59.72 [-70.14, -49.31] | +0.00 [+0.00, +0.00] | -18.75 [-30.56, -6.94] | — | — |
| Plan-only actor | 0.3 / early-turn | 0.00 | 100.00 | 0.00 | 41.67 | — | -0.12 |
| Plan and field | 0.3 / early-turn | 8.33 | 91.67 | 0.00 | 66.67 | 10.89 | 0.37 |
| Plan and field − Plan-only actor | 0.3 / early-turn | +8.33 [+0.00, +18.75] | -8.33 [-18.75, +0.00] | +0.00 [+0.00, +0.00] | +25.00 [+8.33, +41.67] | — | — |
| Plan-only actor | 0.3 / table-gap | 0.00 | 100.00 | 0.00 | 18.75 | — | -0.75 |
| Plan and field | 0.3 / table-gap | 70.83 | 29.17 | 0.00 | 2.08 | 26.90 | 2.14 |
| Plan and field − Plan-only actor | 0.3 / table-gap | +70.83 [+56.25, +83.33] | -70.83 [-83.33, -56.25] | +0.00 [+0.00, +0.00] | -16.67 [-27.08, -6.25] | — | — |
| Plan-only actor | 0.3 / u-trap | 0.00 | 100.00 | 0.00 | 64.58 | — | 0.12 |
| Plan and field | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 13.12 | 8.03 |
| Plan and field − Plan-only actor | 0.3 / u-trap | +100.00 [+100.00, +100.00] | -100.00 [-100.00, -100.00] | +0.00 [+0.00, +0.00] | -64.58 [-77.08, -50.00] | — | — |
| Plan-only actor | 0.5 | 0.00 | 100.00 | 0.00 | 56.25 | — | 0.02 |
| Plan and field | 0.5 | 52.78 | 45.83 | 1.39 | 22.22 | 14.38 | 3.50 |
| Plan and field − Plan-only actor | 0.5 | +52.78 [+42.36, +63.19] | -54.17 [-64.58, -44.44] | +1.39 [+0.00, +3.47] | -34.03 [-43.75, -23.61] | — | — |
| Plan-only actor | 0.5 / early-turn | 0.00 | 100.00 | 0.00 | 79.17 | — | -0.02 |
| Plan and field | 0.5 / early-turn | 4.17 | 95.83 | 0.00 | 66.67 | 11.57 | 0.49 |
| Plan and field − Plan-only actor | 0.5 / early-turn | +4.17 [+0.00, +10.42] | -4.17 [-10.42, +0.00] | +0.00 [+0.00, +0.00] | -12.50 [-29.17, +4.17] | — | — |
| Plan-only actor | 0.5 / table-gap | 0.00 | 100.00 | 0.00 | 25.00 | — | -0.72 |
| Plan and field | 0.5 / table-gap | 60.42 | 35.42 | 4.17 | 0.00 | 18.20 | 2.40 |
| Plan and field − Plan-only actor | 0.5 / table-gap | +60.42 [+45.83, +75.00] | -64.58 [-77.08, -52.08] | +4.17 [+0.00, +10.42] | -25.00 [-39.58, -12.50] | — | — |
| Plan-only actor | 0.5 / u-trap | 0.00 | 100.00 | 0.00 | 64.58 | — | 0.79 |
| Plan and field | 0.5 / u-trap | 93.75 | 6.25 | 0.00 | 0.00 | 12.04 | 7.60 |
| Plan and field − Plan-only actor | 0.5 / u-trap | +93.75 [+85.42, +100.00] | -93.75 [-100.00, -85.42] | +0.00 [+0.00, +0.00] | -64.58 [-75.00, -54.17] | — | — |

At 07:39 the last audit, `319388_2`, moved from gpu to an idle ws-ia slot and started on `ws-l2-014`. It retains the 1 GPU, 8 CPU, 30,000 MB, 3 h request and the same output path. We caught an absolute manifest path before pending build 319412 started. The body collision builder now loads the project's existing relocation mapping before opening the pinned source. Recipe commit `29666df` replaces the pending build as `320478` on ws-ia with 1 GPU, 8 CPUs, 38,000 MB and a 90 min limit. It waits for 319388_2. Source and output path are unchanged. The six remaining planner ablation tasks, the two price runs and the BRACE E4 job were verified against the new dependency before our old build was canceled. Expected bank time remains 10–25 min after admission.

### E6 local CUDA smoke and held-out rollout

The original public BRACE checkpoint passed its dep-1 CUDA inference smoke from 07:42 to 07:45 Dubai. The three selected worlds ran up to 500 control steps, with finite outputs and no numerical transition. One episode reached its ordinary terminal state before the smoke cutoff. The full 72-layout E4 BRACE job started on dep-1 at 07:47 under the single-GPU lock, commit `ac58626`. Its output is `~/robotics/cat-energy-local-runs/overnight_E6_BRACE_E4_local_ac58626`. It uses the same three speeds and prescribed starts. Only our pending duplicate 319457 was canceled.

Cluster jobs 319455 and 319456 retain their IDs, 1 GPU, 8 CPUs and 40,000 MB and now use ws-ia. The cluster smoke starts after the remapping-corrected bank build 320478, then gates the original 384-scene BRACE evaluation. The cluster smoke remains a distinct check of the isolated Slurm environment.

### E0 complete three-seed temporal audit

All 2,304 audited episodes finished at the planned budgets. The tables include the first observed two-second stall from eventually refused episodes with active moving plans. A preceding full second was available for 63 of 69 qualifying episodes at 0.15 m/s and 87 of 92 at 0.30 m/s. Reward values are per-control-step contributions after weighting, averaged within each qualifying seed, with sample s.d. when at least two seeds contributed. Some very sparse rows have data from one seed only.

| Speed (m/s) | Window | Episodes | Stall | Goal progress | Position tracking | Upright | Stand tall | Global power | Hand clearance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | Before | 63 | −0.43 ± 0.06 | 3.02 ± 0.13 | 2.58 ± 0.20 | 2.53 ± 0.18 | 2.72 ± 0.18 | −0.74 ± 0.08 | −0.03 ± 0.00 |
| 0.15 | During | 69 | −1.47 ± 0.07 | 1.43 ± 0.08 | 2.46 ± 0.24 | 2.11 ± 0.23 | 2.34 ± 0.26 | −0.69 ± 0.07 | −0.04 ± 0.01 |
| 0.30 | Before | 87 | −0.16 ± 0.15 | 2.74 ± 0.31 | 2.15 ± 0.26 | 2.57 ± 0.05 | 2.81 ± 0.07 | −0.85 ± 0.15 | −0.03 ± 0.01 |
| 0.30 | During | 92 | −2.13 ± 0.23 | 0.70 ± 0.05 | 1.78 ± 0.09 | 2.32 ± 0.18 | 2.60 ± 0.13 | −0.75 ± 0.07 | −0.02 ± 0.01 |

Goal-progress credit falls at the first stall in side gaps, cabinets and CAT courses. Posture and tracking retain most of their value in these three groups. Field guidance stays near 2.1–2.4 and clearance terms stay small. CAT courses and side gaps account for 133 of 161 qualifying stalls. Furniture and clutter have at most three audit windows per row. These observations meet E2's diagnostic condition for testing the reward balance. They condition on future refusal, so they cannot identify what caused a stall or determine how a new reward changes behavior. The remaining E1 seed comparisons still gate E2.

| Speed | Scene group | Window | Episodes across seeds | Stall | Tracking | Clearance | Field | Posture and motion | Energy | Goal progress |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | Side gaps | before | 19 | -0.17 ± 0.30 | 3.64 ± 0.99 | -0.11 ± 0.04 | 2.10 ± 0.06 | 5.63 ± 0.46 | -0.73 ± 0.09 | 3.11 ± 0.33 |
| 0.15 | Side gaps | during | 19 | -1.23 ± 0.55 | 4.02 ± 0.64 | -0.10 ± 0.04 | 2.14 ± 0.03 | 4.76 ± 0.54 | -0.65 ± 0.08 | 1.56 ± 0.23 |
| 0.15 | Cabinets | before | 10 | 0.00 ± 0.00 | 5.76 ± 0.36 | -0.01 ± 0.00 | 2.21 ± 0.08 | 6.03 ± 0.12 | -0.66 ± 0.03 | 3.33 ± 0.55 |
| 0.15 | Cabinets | during | 10 | -1.38 ± 0.25 | 5.28 ± 0.00 | -0.04 ± 0.01 | 2.37 ± 0.00 | 6.04 ± 0.11 | -0.66 ± 0.03 | 1.38 ± 0.00 |
| 0.15 | Furniture | before | 0 | — | — | — | — | — | — | — |
| 0.15 | Furniture | during | 0 | — | — | — | — | — | — | — |
| 0.15 | Clutter | before | 1 | 0.00 | 2.97 | -0.00 | 2.29 | 5.67 | -0.54 | 0.00 |
| 0.15 | Clutter | during | 1 | 0.00 | 1.79 | -0.00 | 2.27 | 5.83 | -0.45 | 0.00 |
| 0.15 | CAT courses | before | 33 | -0.74 ± 0.15 | 4.26 ± 0.74 | -0.02 ± 0.01 | 2.15 ± 0.05 | 5.50 ± 0.59 | -0.76 ± 0.09 | 2.96 ± 0.23 |
| 0.15 | CAT courses | during | 39 | -1.66 ± 0.20 | 3.82 ± 0.38 | -0.06 ± 0.02 | 2.15 ± 0.02 | 4.49 ± 0.84 | -0.73 ± 0.09 | 1.41 ± 0.10 |
| 0.3 | Side gaps | before | 23 | -0.07 ± 0.12 | 4.33 ± 0.37 | -0.11 ± 0.02 | 2.38 ± 0.06 | 5.95 ± 0.18 | -0.80 ± 0.18 | 2.68 ± 0.33 |
| 0.3 | Side gaps | during | 23 | -2.21 ± 0.40 | 3.29 ± 0.32 | -0.07 ± 0.04 | 2.21 ± 0.03 | 5.50 ± 0.40 | -0.66 ± 0.09 | 0.73 ± 0.06 |
| 0.3 | Cabinets | before | 13 | 0.00 ± 0.00 | 4.49 ± 1.30 | -0.02 ± 0.00 | 1.99 ± 0.07 | 5.51 ± 0.45 | -1.07 ± 0.55 | 4.00 ± 0.40 |
| 0.3 | Cabinets | during | 13 | -1.68 ± 0.72 | 3.78 ± 0.22 | -0.06 ± 0.01 | 2.24 ± 0.09 | 5.83 ± 0.39 | -0.83 ± 0.33 | 0.67 ± 0.08 |
| 0.3 | Furniture | before | 1 | 0.00 | 1.55 | -0.00 | 2.28 | 5.83 | -0.61 | 0.00 |
| 0.3 | Furniture | during | 1 | 0.00 | 3.02 | -0.00 | 2.34 | 5.86 | -0.81 | 0.00 |
| 0.3 | Clutter | before | 3 | 0.00 ± 0.00 | 2.72 ± 0.97 | -0.00 ± 0.00 | 2.36 ± 0.02 | 5.85 ± 0.13 | -0.77 ± 0.18 | 0.00 ± 0.00 |
| 0.3 | Clutter | during | 3 | 0.00 ± 0.00 | 3.39 ± 0.03 | -0.00 ± 0.00 | 2.27 ± 0.05 | 5.85 ± 0.02 | -0.63 ± 0.18 | 0.00 ± 0.00 |
| 0.3 | CAT courses | before | 47 | -0.25 ± 0.18 | 3.54 ± 0.06 | -0.04 ± 0.02 | 2.27 ± 0.05 | 5.54 ± 0.22 | -0.85 ± 0.07 | 2.71 ± 0.08 |
| 0.3 | CAT courses | during | 52 | -2.32 ± 0.22 | 3.12 ± 0.15 | -0.04 ± 0.01 | 2.13 ± 0.06 | 4.98 ± 0.25 | -0.78 ± 0.02 | 0.75 ± 0.02 |

### E1 combined variant relocation

After E6 finished on dep-1, seeds 1 and 2 of the combined E1 variant moved to the same single-GPU lock there. Only our pending tasks 319389_7 and 319389_8 were canceled. The warm-trained checkpoints, commit 1bc21af, 768 episodes per speed and evaluator protocol are unchanged. Outputs are in `~/robotics/cat-energy-local-runs/overnight_E1_both_seed{1,2}_local_1bc21af`. The four remaining fallback and margin tasks stay on gpu. Each local evaluation expects 60–100 min after admission.

### E6 BRACE held-out results

Public BRACE with the common planner completed 144 matched episodes per speed on E4. The initial-state arrays hash matched the existing E4 controls. A CPU flat-walk check under the supplied planner and 0.30 m/s command moved forward 0.81 m in 400 steps and had 0.28 m median position tracking error. This documents basic motion and the reference-tracking error in the transferred simulator. Obstacle-scene results apply to this tracker, the common CAT planner and the CAT physics. BRACE contributes one published checkpoint. The paired 95% intervals resample scenes and COMBO_FIX uses seed 0.

| Case | Speed / family | clean | contact | refused | hands | time to clean goal (s) | episode min palm (cm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BRACE with common planner | 0.15 | 0.00 | 98.61 | 1.39 | 45.14 | — | -0.20 |
| COMBO_FIX | 0.15 | 54.86 | 43.06 | 2.08 | 27.08 | 30.91 | 3.58 |
| COMBO_FIX − BRACE with common planner | 0.15 | +54.86 [+44.44, +65.28] | -55.56 [-65.30, -45.14] | +0.69 [-2.78, +4.86] | -18.06 [-29.17, -6.25] | — | — |
| V7r | 0.15 | 58.33 | 41.67 | 0.00 | 20.14 | 20.53 | 3.09 |
| V7r − BRACE with common planner | 0.15 | +58.33 [+48.61, +68.06] | -56.94 [-66.67, -46.53] | -1.39 [-3.47, +0.00] | -25.00 [-34.72, -14.58] | — | — |
| BRACE with common planner | 0.15 / early-turn | 0.00 | 100.00 | 0.00 | 62.50 | — | -0.32 |
| COMBO_FIX | 0.15 / early-turn | 10.42 | 89.58 | 0.00 | 79.17 | 18.29 | -0.02 |
| COMBO_FIX − BRACE with common planner | 0.15 / early-turn | +10.42 [+0.00, +22.92] | -10.42 [-22.92, +0.00] | +0.00 [+0.00, +0.00] | +16.67 [-2.08, +35.42] | — | — |
| V7r | 0.15 / early-turn | 10.42 | 89.58 | 0.00 | 54.17 | 11.10 | 0.47 |
| V7r − BRACE with common planner | 0.15 / early-turn | +10.42 [+2.08, +20.83] | -10.42 [-20.83, -2.08] | +0.00 [+0.00, +0.00] | -8.33 [-27.08, +12.50] | — | — |
| BRACE with common planner | 0.15 / table-gap | 0.00 | 95.83 | 4.17 | 8.33 | — | -0.87 |
| COMBO_FIX | 0.15 / table-gap | 58.33 | 35.42 | 6.25 | 2.08 | 52.71 | 3.06 |
| COMBO_FIX − BRACE with common planner | 0.15 / table-gap | +58.33 [+41.67, +72.92] | -60.42 [-72.97, -45.83] | +2.08 [-8.33, +12.50] | -6.25 [-14.58, +2.08] | — | — |
| V7r | 0.15 / table-gap | 72.92 | 27.08 | 0.00 | 0.00 | 28.73 | 4.42 |
| V7r − BRACE with common planner | 0.15 / table-gap | +72.92 [+58.33, +85.42] | -68.75 [-83.33, -54.17] | -4.17 [-10.42, +0.00] | -8.33 [-16.67, -2.08] | — | — |
| BRACE with common planner | 0.15 / u-trap | 0.00 | 100.00 | 0.00 | 64.58 | — | 0.59 |
| COMBO_FIX | 0.15 / u-trap | 95.83 | 4.17 | 0.00 | 0.00 | 19.01 | 7.70 |
| COMBO_FIX − BRACE with common planner | 0.15 / u-trap | +95.83 [+89.58, +100.00] | -95.83 [-100.00, -89.58] | +0.00 [+0.00, +0.00] | -64.58 [-77.08, -50.00] | — | — |
| V7r | 0.15 / u-trap | 91.67 | 8.33 | 0.00 | 6.25 | 15.08 | 4.38 |
| V7r − BRACE with common planner | 0.15 / u-trap | +91.67 [+83.33, +97.92] | -91.67 [-97.92, -83.33] | +0.00 [+0.00, +0.00] | -58.33 [-72.92, -43.75] | — | — |
| BRACE with common planner | 0.3 | 0.00 | 99.31 | 0.00 | 45.83 | — | 0.07 |
| COMBO_FIX | 0.3 | 52.08 | 44.44 | 3.47 | 20.14 | 21.14 | 4.04 |
| COMBO_FIX − BRACE with common planner | 0.3 | +52.08 [+41.67, +62.50] | -54.86 [-65.28, -44.44] | +3.47 [+0.69, +6.94] | -25.69 [-36.11, -15.28] | — | — |
| V7r | 0.3 | 56.25 | 43.75 | 0.00 | 23.61 | 15.82 | 3.04 |
| V7r − BRACE with common planner | 0.3 | +56.25 [+45.83, +66.67] | -55.56 [-65.97, -45.14] | +0.00 [+0.00, +0.00] | -22.22 [-34.03, -11.11] | — | — |
| BRACE with common planner | 0.3 / early-turn | 0.00 | 97.92 | 0.00 | 62.50 | — | 0.66 |
| COMBO_FIX | 0.3 / early-turn | 2.08 | 97.92 | 0.00 | 60.42 | 9.92 | 0.01 |
| COMBO_FIX − BRACE with common planner | 0.3 / early-turn | +2.08 [+0.00, +6.25] | +0.00 [-6.25, +6.25] | +0.00 [+0.00, +0.00] | -2.08 [-20.83, +18.75] | — | — |
| V7r | 0.3 / early-turn | 2.08 | 97.92 | 0.00 | 70.83 | 9.78 | 0.22 |
| V7r − BRACE with common planner | 0.3 / early-turn | +2.08 [+0.00, +6.25] | +0.00 [-6.25, +6.25] | +0.00 [+0.00, +0.00] | +8.33 [-14.58, +31.25] | — | — |
| BRACE with common planner | 0.3 / table-gap | 0.00 | 100.00 | 0.00 | 12.50 | — | -0.21 |
| COMBO_FIX | 0.3 / table-gap | 54.17 | 35.42 | 10.42 | 0.00 | 36.87 | 3.78 |
| COMBO_FIX − BRACE with common planner | 0.3 / table-gap | +54.17 [+37.50, +68.75] | -64.58 [-77.08, -50.00] | +10.42 [+2.08, +20.83] | -12.50 [-22.92, -4.17] | — | — |
| V7r | 0.3 / table-gap | 66.67 | 33.33 | 0.00 | 0.00 | 22.41 | 4.27 |
| V7r − BRACE with common planner | 0.3 / table-gap | +66.67 [+54.17, +79.17] | -66.67 [-79.17, -54.17] | +0.00 [+0.00, +0.00] | -12.50 [-22.92, -4.17] | — | — |
| BRACE with common planner | 0.3 / u-trap | 0.00 | 100.00 | 0.00 | 62.50 | — | -0.24 |
| COMBO_FIX | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 12.85 | 8.34 |
| COMBO_FIX − BRACE with common planner | 0.3 / u-trap | +100.00 [+100.00, +100.00] | -100.00 [-100.00, -100.00] | +0.00 [+0.00, +0.00] | -62.50 [-77.08, -47.92] | — | — |
| V7r | 0.3 / u-trap | 100.00 | 0.00 | 0.00 | 0.00 | 11.56 | 4.64 |
| V7r − BRACE with common planner | 0.3 / u-trap | +100.00 [+100.00, +100.00] | -100.00 [-100.00, -100.00] | +0.00 [+0.00, +0.00] | -62.50 [-77.08, -47.92] | — | — |
| BRACE with common planner | 0.5 | 0.00 | 100.00 | 0.00 | 45.83 | — | 0.58 |
| COMBO_FIX | 0.5 | 46.53 | 45.83 | 7.64 | 18.06 | 15.56 | 3.76 |
| COMBO_FIX − BRACE with common planner | 0.5 | +46.53 [+36.11, +56.94] | -54.17 [-64.58, -43.75] | +7.64 [+3.47, +11.81] | -27.78 [-39.58, -16.67] | — | — |
| V7r | 0.5 | 52.08 | 46.53 | 1.39 | 26.39 | 12.88 | 2.91 |
| V7r − BRACE with common planner | 0.5 | +52.08 [+41.67, +62.50] | -53.47 [-63.89, -43.06] | +1.39 [+0.00, +3.47] | -19.44 [-30.56, -8.33] | — | — |
| BRACE with common planner | 0.5 / early-turn | 0.00 | 100.00 | 0.00 | 64.58 | — | 0.14 |
| COMBO_FIX | 0.5 / early-turn | 4.17 | 95.83 | 0.00 | 52.08 | 9.88 | 0.22 |
| COMBO_FIX − BRACE with common planner | 0.5 / early-turn | +4.17 [+0.00, +12.50] | -4.17 [-12.50, +0.00] | +0.00 [+0.00, +0.00] | -12.50 [-35.42, +12.50] | — | — |
| V7r | 0.5 / early-turn | 6.25 | 93.75 | 0.00 | 72.92 | 7.19 | 0.28 |
| V7r − BRACE with common planner | 0.5 / early-turn | +6.25 [+0.00, +16.67] | -6.25 [-16.67, +0.00] | +0.00 [+0.00, +0.00] | +8.33 [-12.50, +27.08] | — | — |
| BRACE with common planner | 0.5 / table-gap | 0.00 | 100.00 | 0.00 | 4.17 | — | 1.02 |
| COMBO_FIX | 0.5 / table-gap | 43.75 | 35.42 | 20.83 | 0.00 | 24.55 | 3.18 |
| COMBO_FIX − BRACE with common planner | 0.5 / table-gap | +43.75 [+27.08, +62.50] | -64.58 [-77.08, -50.00] | +20.83 [+10.42, +31.25] | -4.17 [-10.42, +0.00] | — | — |
| V7r | 0.5 / table-gap | 66.67 | 33.33 | 0.00 | 2.08 | 17.57 | 3.96 |
| V7r − BRACE with common planner | 0.5 / table-gap | +66.67 [+52.08, +79.17] | -66.67 [-79.17, -52.08] | +0.00 [+0.00, +0.00] | -2.08 [-8.33, +4.17] | — | — |
| BRACE with common planner | 0.5 / u-trap | 0.00 | 100.00 | 0.00 | 68.75 | — | 0.56 |
| COMBO_FIX | 0.5 / u-trap | 91.67 | 6.25 | 2.08 | 2.08 | 11.53 | 7.87 |
| COMBO_FIX − BRACE with common planner | 0.5 / u-trap | +91.67 [+83.33, +97.92] | -93.75 [-100.00, -85.42] | +2.08 [+0.00, +6.25] | -66.67 [-79.17, -52.08] | — | — |
| V7r | 0.5 / u-trap | 83.33 | 12.50 | 4.17 | 4.17 | 9.56 | 4.48 |
| V7r − BRACE with common planner | 0.5 / u-trap | +83.33 [+70.83, +93.75] | -87.50 [-95.83, -77.08] | +4.17 [+0.00, +10.42] | -64.58 [-79.17, -50.00] | — | — |

### Mechanical price provenance in the planner checkpoints

The six frozen checkpoint contracts confirm that both seed-0 arms explicitly price policy torque averaged over substeps, with unit arm and leg multipliers. Both arms at seeds 1 and 2 use the compatibility objective based on the final applied-torque sample. The three energy term coefficients agree within each paired seed. The primary on/off comparison remains paired. Its aggregate averages a reflex contrast over two mechanical pricing implementations, and the E3 report also shows the two repeat seeds that share the legacy objective. E2 will preserve the energy contract of each matched parent.

The held-out report measures differences in minimum palm clearance and conditional mean time to clean goal. Within a training seed it pools eligible episodes, matches scenes for resampling and retains the clean-arrival denominator for each policy. A policy with zero clean arrivals has no conditional time estimate. Rates and point estimates elsewhere retain the original episode budgets.

### Runtime update after the E4 input build

Job 320478 completed on ws-l5-006 in 10 min 9 s and certified 117,216 reset poses. Its 72-scene goal-column cache has SHA256 `07905c3fb37972c872bbb078b1104be7f7831fc12523cbd4d9fa317c92a5c`. All three planner families and both exploratory pricing jobs are ready on gpu, with the pending tasks gated by the active standing control's scheduler quota. E2 commit `f31f3b9` and all three saved warm-start paths are staged in `~/cat_worktrees/energy_overnight_e2_f31f3b9`. E2 training has not launched because the E1 gate remains open.

BRACE smoke 319455 failed before inference. A separate process occupied 24.60 GiB on its assigned 31.47 GiB GPU at ws-l5-006 when the scene bank tried to load. The scheduler canceled its dependent 319456 before it ran. Two explicit ws-ia chains now make room for all four E1 evaluations while gpu runs the third-seed standing control. They preserve the former E1, E5 and E6 protocols. The updated jobs exclude ws-l5-004, ws-l5-005 and ws-l5-006.

| Job | Purpose | Scheduling | Resources | Limit / runtime estimate | Output directory under `/l/users/konstantin.smirnov/runs` |
| --- | --- | --- | --- | --- | --- |
| 319389_2 → 319389_5 | E1 fallback, then margin, seed 2 | ws-ia, first started on ws-l2-012 | 1 GPU, 8 CPUs, 30000 MB | 3 h / 1–2 h each | `overnight_E1_{fallback,margin}_seed2_319389` |
| 319389_1 → 319389_4 | E1 fallback, then margin, seed 1 | ws-ia, after 319349_4 | 1 GPU, 8 CPUs, 30000 MB | 3 h / 1–2 h each | `overnight_E1_{fallback,margin}_seed1_319389` |
| 320528 → 320529 | BRACE smoke, then 384 scenes | ws-ia, after 319389_5 | 1 GPU, 8 CPUs, 40000 MB | 30 min / 3 h limits; 3–8 min / 1–2 h estimates | `overnight_E6_BRACE_smoke_320528`, `overnight_E6_BRACE_384_320529` |

After 319389_4, the existing E5 plan-only array 319436 now runs one task at a time, followed by 319438 then 319439. The second chain runs 319437 after the BRACE 384 job terminates. Both chains together reserve at most two concurrent ws-ia jobs. E1's combined variant remains on dep-1. Standing 319350_0 completed 150 updates and its seed-2 arm×30 eval, with 26.15% of worlds moving feet and 17.07% of threatened events classified as failures. Its matched arm×1 control 319350_1 started on gpu-02 at approximately 08:42 Dubai. A cross-seed effect awaits that control.

### E6 additional release checks

Release check on 2026-11-24. The closest navigation systems below would require their missing trained components. BRACE is the released reference-tracking control evaluated with our common planner. The other available G1 weights cover distinct command interfaces and are documented here for future matched comparisons.

| Candidate | Release evidence and scope |
| --- | --- |
| Gallant | The official repository contains the voxel-grid training code and lists evaluation scripts as forthcoming. Its complete main tree and release list contain no policy weight files. [Project](https://gallantloco.github.io/), [code](https://github.com/Sage-PhysicalAI/Gallant) |
| LP-NavOA | The authors publish an mjlab fork for limited-perception goal recovery and obstacle bypass. The evaluation command expects trained local planner and locomotion-backbone files. Its complete main tree and release list contain neither. [Project](https://shenqiqishi.github.io/LP-NavOA/), [code](https://github.com/shenqiqishi/LP-NavOA-code) |
| Moving Through Clutter | The paper reports a Unitree G1 generalist and a new benchmark. The paper and coauthor's publication page link the paper and video. No downloadable generalist weights or evaluation bundle were located in those sources. [Paper](https://arxiv.org/html/2609.21107v1), [coauthor publications](https://people.cs.gmu.edu/~xiao/) |
| REACH | The G1 crawling project lists code as forthcoming. Its height controller, transition policy and lidar planner are unavailable through the project page. [Project](https://aislab-sustech.github.io/REACH/) |
| NEXUS | A 29-joint fixed-batch ONNX tracker and inference config are available. The checkpoint requires time-aligned reference motion and processed depth. The release targets terrain-adaptive motion tracking and does not supply a sparse-goal navigation planner. [Release and interface](https://gamma-robot.github.io/NEXUS/guides/pretrained-model.html) |
| Perceptive Humanoid Parkour | Public depth-backbone and locomotion-student ONNX files include a MuJoCo path with direction commands, D435i depth and a separate G1 collision model. The released student takes a direction command for parkour. The release provides no sparse-goal indoor navigation layer. A matched comparison would need the calibrated depth pipeline and a shared way of supplying course directions. [Student release](https://github.com/amazon-far/php_parkour/releases/tag/student-assets-v1), [deployment](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md) |

The focused CPU suite now passes all 34 selected planner, blind-obstacle, energy, inverse-dynamics and refusal-training tests on the current training branch.

### E4 local transfer for the field and price comparisons

At 09:16 Dubai, pending tasks 319449_1, 319449_2, 319450 and 319451 were moved to dep-1. Only those four pending cluster jobs were canceled. They use the verified original checkpoints, the local ac58626 evaluator and the same certified 72 layouts, starts and three speeds. The single-GPU sequence starts after the existing E1 combined seed 2 evaluation, in order plan-only seed 1, plan-only seed 2, COMBO1b u500 and COMBO_NOPRICE u50. Outputs are `~/robotics/cat-energy-local-runs/overnight_E4_<model>_seed<seed>_local_ac58626`. Observed comparable local run times are 2–5 min for the masked actors and about 35–60 min for the priced and price-free planners. The four frozen E4 planner and reflex-off replications remain queued on gpu.

### E3 complete plan-blind comparison

All three paired seeds finished all four obstacle protocols. Each run scored two episodes per scene for 384 scenes, or 768 episodes per protocol. The planner's original hand path supplied the insertion schedule, which was fixed before policy rollout. All 768 guide rows per protocol had a valid placement. Mean exposure ranged from 98.35% to 98.70% across the pooled seed-condition rows. The reflex contrast compares separately warm-trained single actors with a 5 cm or 0 cm arm-tracking reward band. At seeds 1 and 2 both policies used the legacy mechanical power objective. At seed 0 both used explicit substep mechanical pricing. Across-seed values are mean ± sample s.d. The signed differences are percentage points with nominal paired seed-level 95% t intervals.

| Case | Protocol | Clean | Any contact | Refused | Any hand contact | Inserted-object hand contact | Box revealed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reflex off | Static, 0.15 m/s | 30.64 ± 1.61 | 63.15 ± 2.93 | 6.12 ± 1.32 | 20.14 ± 1.39 | 17.40 ± 0.80 | 98.70 ± 0.13 |
| Reflex on | Static, 0.15 m/s | 29.86 ± 1.09 | 62.07 ± 4.38 | 7.99 ± 3.26 | 18.84 ± 1.26 | 16.49 ± 1.26 | 98.70 ± 0.26 |
| Reflex on − Reflex off | Static, 0.15 m/s | -0.78 ± 0.60 [-2.26, +0.70] | -1.09 ± 2.02 [-6.10, +3.93] | +1.87 ± 2.26 [-3.76, +7.49] | -1.30 ± 2.29 [-6.99, +4.38] | -0.91 ± 1.79 [-5.36, +3.54] | +0.00 ± 0.34 [-0.86, +0.86] |
| Reflex off | Static, 0.30 m/s | 33.51 ± 3.18 | 56.81 ± 3.40 | 9.46 ± 0.40 | 13.54 ± 0.79 | 10.07 ± 0.60 | 98.35 ± 0.33 |
| Reflex on | Static, 0.30 m/s | 33.03 ± 1.04 | 56.64 ± 2.59 | 10.03 ± 1.63 | 12.89 ± 2.44 | 9.46 ± 1.73 | 98.61 ± 0.20 |
| Reflex on − Reflex off | Static, 0.30 m/s | -0.48 ± 2.61 [-6.95, +5.99] | -0.17 ± 3.55 [-9.00, +8.66] | +0.56 ± 1.31 [-2.69, +3.82] | -0.65 ± 1.71 [-4.89, +3.59] | -0.61 ± 2.33 [-6.41, +5.19] | +0.26 ± 0.23 [-0.30, +0.82] |
| Reflex off | Moving, 0.15 m/s | 31.55 ± 2.08 | 60.81 ± 3.53 | 7.60 ± 1.43 | 12.63 ± 0.23 | 9.55 ± 0.46 | 98.70 ± 0.13 |
| Reflex on | Moving, 0.15 m/s | 31.47 ± 0.78 | 59.64 ± 2.88 | 8.85 ± 2.83 | 12.54 ± 0.53 | 10.11 ± 0.54 | 98.70 ± 0.26 |
| Reflex on − Reflex off | Moving, 0.15 m/s | -0.09 ± 2.83 [-7.12, +6.94] | -1.17 ± 4.11 [-11.39, +9.04] | +1.26 ± 2.68 [-5.39, +7.91] | -0.09 ± 0.74 [-1.93, +1.75] | +0.56 ± 0.59 [-0.89, +2.02] | +0.00 ± 0.39 [-0.97, +0.97] |
| Reflex off | Moving, 0.30 m/s | 33.25 ± 1.24 | 57.12 ± 1.87 | 9.38 ± 0.57 | 12.59 ± 0.99 | 8.98 ± 0.79 | 98.44 ± 0.13 |
| Reflex on | Moving, 0.30 m/s | 33.16 ± 1.66 | 56.55 ± 2.20 | 10.16 ± 1.63 | 11.33 ± 1.03 | 7.73 ± 0.54 | 98.48 ± 0.20 |
| Reflex on − Reflex off | Moving, 0.30 m/s | -0.09 ± 1.96 [-4.95, +4.78] | -0.56 ± 1.95 [-5.42, +4.29] | +0.78 ± 1.19 [-2.18, +3.75] | -1.26 ± 1.56 [-5.14, +2.63] | -1.26 ± 0.27 [-1.93, -0.59] | +0.04 ± 0.15 [-0.33, +0.42] |

At 0.30 m/s with a moving box, inserted-object hand contact decreased from 8.98% to 7.73%. The effect was negative in all three seed pairs (−1.17, −1.04 and −1.56 pp) with practically unchanged reveal rate. The 95% interval adjusted across the four insertion protocols is [−2.65, +0.13] pp. The reversed slow-moving-box point estimate was +0.56 pp [−0.89, +2.02]. Mean any-contact rates decreased 0.17–1.17 pp across the four protocols, and mean refusals increased 0.56–1.87 pp. Clean traversal means decreased 0.09–0.78 pp. The paired intervals for these all-scene outcomes include zero. The localized faster-moving-box effect is promising and the data leave open whether it generalizes across protocols or improves traversal.

The three matched-image replays use the same scene geometry and obstacle schedule on both sides. They are under `energy-videos` in the project artifacts as `plan-blind-static-015.mp4`, `plan-blind-moving-015.mp4` and `plan-blind-moving-030.mp4`. All three are H.264 1280×720 yuv420p with silent stereo AAC, fast-start layout and durations of 15.00, 13.80 and 11.72 seconds respectively. The clips show selected episodes.

#### Same-price sensitivity using repeat seeds 1 and 2

This subset checks the contrast under a common energy implementation across seeds. Its intervals have one degree of freedom.

| Case | Protocol | Clean | Any contact | Refused | Any hand contact | Inserted-object hand contact | Box revealed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reflex off | Static, 0.15 m/s | 29.75 ± 0.64 | 64.65 ± 1.93 | 5.53 ± 1.20 | 19.53 ± 1.29 | 17.06 ± 0.74 | 98.76 ± 0.09 |
| Reflex on | Static, 0.15 m/s | 29.23 ± 0.09 | 64.58 ± 0.55 | 6.12 ± 0.55 | 18.68 ± 1.75 | 16.34 ± 1.75 | 98.57 ± 0.18 |
| Reflex on − Reflex off | Static, 0.15 m/s | -0.52 ± 0.55 [-5.48, +4.44] | -0.07 ± 1.38 [-12.47, +12.34] | +0.59 ± 0.64 [-5.20, +6.38] | -0.85 ± 3.04 [-28.14, +26.45] | -0.72 ± 2.49 [-23.05, +21.62] | -0.20 ± 0.09 [-1.02, +0.63] |
| Reflex off | Static, 0.30 m/s | 33.01 ± 4.33 | 57.49 ± 4.51 | 9.24 ± 0.18 | 13.80 ± 0.92 | 9.90 ± 0.74 | 98.18 ± 0.18 |
| Reflex on | Static, 0.30 m/s | 32.49 ± 0.64 | 58.14 ± 0.09 | 9.11 ± 0.55 | 13.28 ± 3.31 | 10.03 ± 2.03 | 98.50 ± 0.09 |
| Reflex on − Reflex off | Static, 0.30 m/s | -0.52 ± 3.68 [-33.61, +32.57] | +0.65 ± 4.60 [-40.71, +42.01] | -0.13 ± 0.74 [-6.75, +6.49] | -0.52 ± 2.39 [-22.03, +20.99] | +0.13 ± 2.76 [-24.69, +24.95] | +0.33 ± 0.28 [-2.16, +2.81] |
| Reflex off | Moving, 0.15 m/s | 31.05 ± 2.67 | 61.59 ± 4.60 | 7.36 ± 1.93 | 12.70 ± 0.28 | 9.77 ± 0.37 | 98.76 ± 0.09 |
| Reflex on | Moving, 0.15 m/s | 31.51 ± 1.10 | 61.26 ± 0.83 | 7.23 ± 0.28 | 12.50 ± 0.74 | 10.03 ± 0.74 | 98.57 ± 0.18 |
| Reflex on − Reflex off | Moving, 0.15 m/s | +0.46 ± 3.77 [-33.46, +34.37] | -0.33 ± 5.43 [-49.13, +48.48] | -0.13 ± 1.66 [-15.02, +14.76] | -0.20 ± 1.01 [-9.29, +8.90] | +0.26 ± 0.37 [-3.05, +3.57] | -0.20 ± 0.28 [-2.68, +2.29] |
| Reflex off | Moving, 0.30 m/s | 32.62 ± 0.83 | 57.81 ± 2.03 | 9.24 ± 0.74 | 12.37 ± 1.29 | 9.24 ± 0.92 | 98.37 ± 0.09 |
| Reflex on | Moving, 0.30 m/s | 32.62 ± 1.93 | 57.81 ± 0.37 | 9.38 ± 1.29 | 11.91 ± 0.28 | 7.94 ± 0.55 | 98.37 ± 0.09 |
| Reflex on − Reflex off | Moving, 0.30 m/s | +0.00 ± 2.76 [-24.82, +24.82] | +0.00 ± 2.39 [-21.51, +21.51] | +0.13 ± 0.55 [-4.83, +5.09] | -0.46 ± 1.01 [-9.56, +8.64] | -1.30 ± 0.37 [-4.61, +2.01] | +0.00 ± 0.18 [-1.65, +1.65] |

### E1 and E6 retries and the E5 guide-start check

Seed 2's E1 fallback run 319389_2 finished both speeds. Its margin follow-on, 319389_5, failed in Warp before physics with a CUDA primary-context error. Slurm never started dependent BRACE jobs 320528 and 320529. Margin retry 320662_5 now runs the same frozen checkpoint and exact E1 recipe on ws-l2-012, where the paired fallback finished. BRACE smoke and full evaluation are now 320663 and 320664, with the same resources and output naming as the previous pair. Their dependency starts after 320662 succeeds.

The E5 reactive job 319437 stopped at its exact guide-match check before issuing any score. Commit `b1a73a3` lets the external E5 checkpoints replay the guide's recorded scene assignments and initial physical states before the simulator creates observations. It validates metadata and requested repetitions, checks the realized states against the guide after reset and records their hash. Raised-hand starts restore the corresponding controller history from the actual recorded joint state. All four guide files have the same 768-state hash `ef07a14c4969755b2bde621308d0cb7e07a0188bc4b2d2f0f626847c943b0c86`. The 35 focused CPU tests passed. A 30-step local reactive startup smoke waits for the dep-1 single-GPU lock after the active E1 run. The completed reflex pairs and the plan-only controls retain their original reset checks.

| Job | Partition / resources | Runtime estimate / limit | Scheduling | Output under `/l/users/konstantin.smirnov/runs` |
| --- | --- | --- | --- | --- |
| 320662_5 | ws-ia, 1 GPU, 8 CPUs, 30000 MB | 1–2 h / 3 h | Running on ws-l2-012 | `overnight_E1_margin_seed2_320662` |
| 320663 → 320664 | ws-ia, 1 GPU, 8 CPUs, 40000 MB | 3–8 min / 30 min and 1–2 h / 3 h | After 320662 | `overnight_E6_BRACE_smoke_320663`, `overnight_E6_BRACE_384_320664` |
| 320668 | ws-ia, 1 GPU, 12 CPUs, 54000 MB | 2–4 h / 7 h | After 320664 | `overnight_E5_reactive_seed0_320668` |
| 320669 → 320670 | ws-ia, 1 GPU, 12 CPUs, 54000 MB | 2–4 h / 7 h each | After serial array 319436 | `overnight_E5_price_reference_seed0_320669`, `overnight_E5_no_price_seed0_320670` |

The E5 price-reference and price-free cells replace only the two pending tasks 319438 and 319439. They use the frozen COMBO1b and COMBO_NOPRICE checkpoints and all four original obstacle schedules. The replacement state replay uses the same guide tested on the original bands. The schedule preserves the two ws-ia dependency chains.
