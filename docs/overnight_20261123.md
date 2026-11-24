# Overnight summary

Provisional at 06:00 Dubai, November 24. The completed comparisons have yet to show a joint gain in clean traversal and contact.

- Original 384 scenes, three paired seeds. Reflex-on clean deltas were −0.87 and −1.78 pp at 0.15 and 0.30 m/s. The corresponding hand-contact deltas were 0.00 and −0.48 pp. All paired 95% intervals include zero.
- Plan-blind seed 0. Reflex reduced any contact by 1.69–3.12 pp and increased refusals by 1.95–4.43 pp across the four protocols. Two more seed pairs are pending. The three synchronized replays are available in the project energy-videos folder.
- Held-out layouts at 0.50 m/s, seed 0. Against the reflex-off tracker, reflex-on cut hand contact by 4.17 pp and raised refusals by 6.25 pp with equal total contact.
- E0's first temporal audit found persistent positive posture and tracking rewards during stalls. E2's warm-start reward profile passed CPU checks and awaits the full E0/E1 gate.
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
