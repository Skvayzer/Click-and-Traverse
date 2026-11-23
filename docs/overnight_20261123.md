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

Both runs finished 150 updates from S2 seed 1 u180 and had zero falls in the benchmark. Mechanical watts are comparable across α values. Reported heat uses each checkpoint's α and changes units of the holding contribution accordingly, so the raw heat numbers do not isolate a behavioral change. The α1000 run 318296_3 started on gpu.
