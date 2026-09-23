# Measured response weights — CPU, 22 September 2026

**Recommendation: provisional hand clearance −5, tracking_root_field +1. Keep arm clearance −2 for this isolated change. Do not call this a globally calibrated safe pair.** The measured poke and passage thresholds are clearance magnitudes 974.112287 and 788.813041, respectively. The proposed −20 also selects A in these measured comparisons; it is not refuted by this sample. No training or launch was performed.

**Scope / completeness:** real MuJoCo dynamics, actual production reward kernels, requested native warm-start policy, one deterministic seed (71), one front poke and one analytic narrow passage. This provides conditional finite-trace thresholds, not a global optimum or a statistical retention guarantee. Missing: full collision-bank acceptance, a multi-seed/direction/bank distribution bound, discounted whole-episode counterfactuals, and changed-policy clutter/narrow/CAT success measurements. The latter cannot be obtained from reward rescoring and were not invented.

Protocol: `scripts/measure_response_weights_cpu.py:1`; complete unrounded per-step terms: `outputs/response_weight_audit_cpu/measurement.json:1`. Frozen weights are read directly (mmap) from `outputs/cat_flat_balance_ppo_37632_20260920/resume.pt`; no copies or checkpoint/bank mutations. Config scales come from `outputs/cat_recover_23_30720_20260922/run.json`, with obsolete box rewards removed, native 222/310, noise/push/domain randomization disabled. No new avoidance flag. CPU threads=2; GPU driver was unavailable.

A is a bounded tucked-arm target intervention on the frozen walking controller; B obtains physically executed stopping/backing actions from the frozen controller using synthetic standing command/phase/guidance **only as controller inputs**, all restored before each physical step. C holds the onset arm targets. The true command stays 0.6 m/s and the true gait generator is unchanged for all three. Arms unchanged means unchanged motor targets, not perfectly rigid limb positions. A is approximately arm-led, not a perfectly constrained root path. These are feasible candidate action sequences, not claims that the unmodified policy selects them.

Poke: 6 cm sphere, initial 12 cm surface gap, 10 cm/s front approach, advected with matched C root, as in `scripts/diagnose_reactive_clearance_cpu.py:58`. Common horizon ends at C termination: 71 steps / 1.42 s. Sphere post-termination traces are excluded from all reported comparisons.

Narrow: two finite analytic boxes, 74 cm opening, 60 cm length, 140 cm height, entry 80 cm ahead of onset root. Common A/B/C horizon: 37 steps / 0.74 s (C terminates). A's root crosses the far gate at step 231 / 4.62 s. B remains upstream (maximum forward travel 0.154180 m) and eventually backs away; A and B remain unterminated through that common full-crossing horizon. Root crossing is **not** full-body passage certification. Virtual analytic fields follow the poke diagnostic; full-body collision-bank events are not measured, so body_collision_event=0 is a scope limitation, not evidence of no proxy contacts.

Measured motion check (same JSON): at poke endpoint A root moved (0.546395, 0.083929, −0.021269) m and left hand relative to root moved (−0.050265, −0.207785, −0.154392) m. C root moved (0.529590, 0.005070, −0.019417) m; matched-control root-relative front retreat is therefore 3.908467 cm for A. A has 7.885939 cm more lateral root drift than C: **this does not realize an exactly arm-only response with identical base path.** The ledger is for measured arm-target, body, and hold interventions; an exact pure-A/identical-path ledger and its threshold remain unmeasured. In the full passage A root advances 1.401672 m with −0.010258 m lateral change; B retreats 2.085705 m with +0.741748 m lateral change. These show traversal versus body refusal, not immobility.

## Every contributing term, reward per 20 ms control step

Tables are arithmetic means of actual per-step contributions over identical, pre-first-termination A/B/C horizons. Raw per-step arrays are in the JSON. All terms already include dt=0.02; collision events, if measured, would not be dt scaled. Retired box and flat bonus terms are exactly zero in this configuration. The production non-hand floor lift is shown separately. Formula: R=max(0, sum(non-hand weighted terms)×dt)−H×pressure×dt (`cat_mjlab/task.py:536`, `cat_mjlab/reward_floor.py:13`). **Do not floor the total after adding hand clearance, or floor averaged terms.**

### poke_front

| Term | A: −0.5 | B: −0.5 | C: −0.5 | A: −20 | B: −20 | C: −20 |
|---|---:|---:|---:|---:|---:|---:|
| tracking_orientation | 0.0317908404 | 0.0355372506 | 0.031706075 | 0.0317908404 | 0.0355372506 | 0.031706075 |
| tracking_root_field | 0.0155721867 | 0.00644568481 | 0.0151100029 | 0.0155721867 | 0.00644568481 | 0.0151100029 |
| body_motion | -0.00297640431 | -0.00159508448 | -0.00342218148 | -0.00297640431 | -0.00159508448 | -0.00342218148 |
| body_rotation | 0.0119941511 | 0.0138549769 | 0.0122220527 | 0.0119941511 | 0.0138549769 | 0.0122220527 |
| foot_contact | 0 | -0.0191549296 | 0 | 0 | -0.0191549296 | 0 |
| foot_clearance | -1.92949928e-05 | -0.000737627463 | -1.82202461e-05 | -1.92949928e-05 | -0.000737627463 | -1.82202461e-05 |
| foot_slip | -1.80202992e-06 | -7.26400613e-05 | -1.80565604e-06 | -1.80202992e-06 | -7.26400613e-05 | -1.80565604e-06 |
| foot_balance | -0.00377568955 | -0.00859909046 | -0.00886763205 | -0.00377568955 | -0.00859909046 | -0.00886763205 |
| straight_knee | 0 | 0 | 0 | 0 | 0 | 0 |
| foot_far | 0 | 0 | 0 | 0 | 0 | 0 |
| joint_limits | 0 | 0 | 0 | 0 | 0 | 0 |
| joint_torque | -0.00382290879 | -0.00284170815 | -0.00328256083 | -0.00382290879 | -0.00284170815 | -0.00328256083 |
| smoothness_joint | -0.000616335087 | -0.000230681784 | -0.000655113966 | -0.000616335087 | -0.000230681784 | -0.000655113966 |
| smoothness_action | -8.52732505e-05 | -8.40217843e-05 | -8.7583309e-05 | -8.52732505e-05 | -8.40217843e-05 | -8.7583309e-05 |
| headgf | 0.00174989646 | 2.79335967e-05 | 0.00109154629 | 0.00174989646 | 2.79335967e-05 | 0.00109154629 |
| feetgf | 0.0484507042 | 0.0484507042 | 0.0484507042 | 0.0484507042 | 0.0484507042 | 0.0484507042 |
| handsgf | 0.0466745534 | -0.0129849223 | 0.0313958694 | 0.0466745534 | -0.0129849223 | 0.0313958694 |
| headdf | -1.26246271e-12 | -3.84735231e-15 | -5.68406237e-13 | -1.26246271e-12 | -3.84735231e-15 | -5.68406237e-13 |
| feetdf | -6.41178991e-21 | -5.18525884e-23 | -3.43582213e-21 | -6.41178991e-21 | -5.18525884e-23 | -3.43582213e-21 |
| handsdf | -0.00241097124 | -0.000840873765 | -0.14541278 | -0.00241097124 | -0.000840873765 | -0.14541278 |
| kneesdf | -8.74968065e-15 | -2.47140243e-17 | -3.42299337e-15 | -8.74968065e-15 | -2.47140243e-17 | -3.42299337e-15 |
| shldsdf | -3.82483927e-11 | -1.90900872e-14 | -5.36719954e-12 | -3.82483927e-11 | -1.90900872e-14 | -5.36719954e-12 |
| wholebody_hand_clearance | -6.40931582e-05 | -2.41316257e-05 | -0.000964964862 | -0.00256372633 | -0.000965265028 | -0.0385985945 |
| wholebody_arm_clearance | 0 | 0 | 0 | 0 | 0 | 0 |
| wholebody_upper_target_velocity | -1.43586976e-05 | -1.22163625e-19 | -1.22163625e-19 | -1.43586976e-05 | -1.22163625e-19 | -1.22163625e-19 |
| wholebody_upper_target_acceleration | -7.48033389e-05 | -6.93488289e-05 | -6.93488289e-05 | -7.48033389e-05 | -6.93488289e-05 | -6.93488289e-05 |
| wholebody_upper_clear_posture | -4.07811308e-05 | -0.000118718702 | -1.9284095e-05 | -4.07811308e-05 | -0.000118718702 | -1.9284095e-05 |
| reward_floor_lift | 0 | 0.00755276732 | 0.0837981666 | 0 | 0.00755276732 | 0.0837981666 |
| body_collision_event | 0 | 0 | 0 | 0 | 0 | 0 |
| **Actual total** | 0.142329617 | 0.0645155384 | 0.0609729417 | 0.139829983 | 0.063574405 | 0.0233393121 |

Highest measured mean: **A_arm_candidate at −0.5; A_arm_candidate at −20.** This ranks only the three tested trajectories, not all possible responses.

### narrow

| Term | A: −0.5 | B: −0.5 | C: −0.5 | A: −20 | B: −20 | C: −20 |
|---|---:|---:|---:|---:|---:|---:|
| tracking_orientation | 0.0310732908 | 0.0351489616 | 0.0330924105 | 0.0310732908 | 0.0351489616 | 0.0330924105 |
| tracking_root_field | 0.0183227144 | 0.00902728552 | 0.0188238568 | 0.0183227144 | 0.00902728552 | 0.0188238568 |
| body_motion | -0.00323815568 | -0.00255323717 | -0.00399811315 | -0.00323815568 | -0.00255323717 | -0.00399811315 |
| body_rotation | 0.0102235346 | 0.0144058156 | 0.00784869115 | 0.0102235346 | 0.0144058156 | 0.00784869115 |
| foot_contact | 0 | -0.0151351351 | 0 | 0 | -0.0151351351 | 0 |
| foot_clearance | -6.67216005e-06 | -0.000624430207 | -1.97440803e-06 | -6.67216005e-06 | -0.000624430207 | -1.97440803e-06 |
| foot_slip | -4.88627418e-05 | -6.70705998e-05 | -8.53470955e-05 | -4.88627418e-05 | -6.70705998e-05 | -8.53470955e-05 |
| foot_balance | -0.00891018052 | -0.00472259546 | -0.00915040967 | -0.00891018052 | -0.00472259546 | -0.00915040967 |
| straight_knee | 0 | 0 | 0 | 0 | 0 | 0 |
| foot_far | 0 | 0 | 0 | 0 | 0 | 0 |
| joint_limits | 0 | 0 | -3.66250244e-05 | 0 | 0 | -3.66250244e-05 |
| joint_torque | -0.00457604507 | -0.00359643218 | -0.00407547013 | -0.00457604507 | -0.00359643218 | -0.00407547013 |
| smoothness_joint | -0.000679220195 | -0.000364715089 | -0.000875110755 | -0.000679220195 | -0.000364715089 | -0.000875110755 |
| smoothness_action | -8.81894434e-05 | -8.51604899e-05 | -9.13514846e-05 | -8.81894434e-05 | -8.51604899e-05 | -9.13514846e-05 |
| headgf | 0.000842300724 | 1.10185477e-05 | 0.00357916489 | 0.000842300724 | 1.10185477e-05 | 0.00357916489 |
| feetgf | 0.0500120351 | 0.0475676649 | 0.0522452355 | 0.0500120351 | 0.0475676649 | 0.0522452355 |
| handsgf | 0.0572602262 | 0.0295402022 | 0.045039708 | 0.0572602262 | 0.0295402022 | 0.045039708 |
| headdf | -4.60455403e-13 | -3.42029935e-15 | -2.17485409e-11 | -4.60455403e-13 | -3.42029935e-15 | -2.17485409e-11 |
| feetdf | -1.12495393e-07 | -3.87463816e-13 | -1.2941251e-07 | -1.12495393e-07 | -3.87463816e-13 | -1.2941251e-07 |
| handsdf | -2.26674034e-05 | -3.07113162e-05 | -0.0639374847 | -2.26674034e-05 | -3.07113162e-05 | -0.0639374847 |
| kneesdf | -5.15240073e-08 | -5.5911597e-13 | -1.38006618e-08 | -5.15240073e-08 | -5.5911597e-13 | -1.38006618e-08 |
| shldsdf | -3.13058787e-12 | -1.10782267e-14 | -9.68107085e-10 | -3.13058787e-12 | -1.10782267e-14 | -9.68107085e-10 |
| wholebody_hand_clearance | 0 | 0 | -0.000467301996 | 0 | 0 | -0.0186920798 |
| wholebody_arm_clearance | 0 | 0 | 0 | 0 | 0 | 0 |
| wholebody_upper_target_velocity | -2.75531765e-05 | -2.34422091e-19 | -2.34422091e-19 | -2.75531765e-05 | -2.34422091e-19 | -2.34422091e-19 |
| wholebody_upper_target_acceleration | -0.000143541542 | -0.00013307478 | -0.00013307478 | -0.000143541542 | -0.00013307478 | -0.00013307478 |
| wholebody_upper_clear_posture | -0.000118534226 | -0.000120717297 | -8.0549429e-05 | -0.000118534226 | -0.000120717297 | -8.0549429e-05 |
| reward_floor_lift | 0 | 0.000206739977 | 0.0483146956 | 0 | 0.000206739977 | 0.0483146956 |
| body_collision_event | 0 | 0 | 0 | 0 | 0 | 0 |
| **Actual total** | 0.149874316 | 0.108474409 | 0.126010806 | 0.149874316 | 0.108474409 | 0.107786028 |

Highest measured mean: **A_arm_candidate at −0.5; A_arm_candidate at −20.** This ranks only the three tested trajectories, not all possible responses.

### Full passage A versus B, 231-step common horizon

Here C is excluded because it terminated at step 37; later autoreset samples are not valid C passage evidence. This table includes A’s actual far-gate crossing and the same elapsed refusal/backing time for B.

| Term | Traverse A: −0.5 | Refuse B: −0.5 | Traverse A: −20 | Refuse B: −20 |
|---|---:|---:|---:|---:|
| tracking_orientation | 0.0337193518 | 0.0308803992 | 0.0337193518 | 0.0308803992 |
| tracking_root_field | 0.0121796384 | 0.00197617807 | 0.0121796384 | 0.00197617807 |
| body_motion | -0.00308401242 | -0.00459891922 | -0.00308401242 | -0.00459891922 |
| body_rotation | 0.00751880241 | 0.0103309094 | 0.00751880241 | 0.0103309094 |
| foot_contact | -0.00103896104 | -0.0240692641 | -0.00103896104 | -0.0240692641 |
| foot_clearance | -6.47552935e-05 | -0.000556344736 | -6.47552935e-05 | -0.000556344736 |
| foot_slip | -3.69001536e-05 | -0.0103018702 | -3.69001536e-05 | -0.0103018702 |
| foot_balance | -0.00646976932 | -0.0419855668 | -0.00646976932 | -0.0419855668 |
| straight_knee | 0 | 0 | 0 | 0 |
| foot_far | 0 | 0 | 0 | 0 |
| joint_limits | -1.23126515e-05 | -0.000116237423 | -1.23126515e-05 | -0.000116237423 |
| joint_torque | -0.00688320938 | -0.0101129068 | -0.00688320938 | -0.0101129068 |
| smoothness_joint | -0.000608366679 | -0.00104359691 | -0.000608366679 | -0.00104359691 |
| smoothness_action | -8.97940816e-05 | -9.39072133e-05 | -8.97940816e-05 | -9.39072133e-05 |
| headgf | 0.0329404502 | 6.30410142e-07 | 0.0329404502 | 6.30410142e-07 |
| feetgf | 0.0781395421 | 0.0470995634 | 0.0781395421 | 0.0470995634 |
| handsgf | 0.0473194468 | -0.0105788869 | 0.0473194468 | -0.0105788869 |
| headdf | -0.000730886793 | -6.48822775e-16 | -0.000730886793 | -6.48822775e-16 |
| feetdf | -0.00412737095 | -1.83671294e-13 | -0.00412737095 | -1.83671294e-13 |
| handsdf | -0.00743973464 | -5.33826576e-06 | -0.00743973464 | -5.33826576e-06 |
| kneesdf | -4.07644197e-05 | -1.23698065e-13 | -4.07644197e-05 | -1.23698065e-13 |
| shldsdf | -0.00875838033 | -2.07294682e-15 | -0.00875838033 | -2.07294682e-15 |
| wholebody_hand_clearance | -9.70715777e-05 | 0 | -0.00388286311 | 0 |
| wholebody_arm_clearance | -1.86044523e-07 | 0 | -1.86044523e-07 | 0 |
| wholebody_upper_target_velocity | -4.41327936e-06 | -3.75481272e-20 | -4.41327936e-06 | -3.75481272e-20 |
| wholebody_upper_target_acceleration | -2.29915024e-05 | -2.1315008e-05 | -2.29915024e-05 | -2.1315008e-05 |
| wholebody_upper_clear_posture | -6.32945866e-05 | -0.000161212818 | -6.32945866e-05 | -0.000161212818 |
| reward_floor_lift | 0.00350544093 | 0.0360616021 | 0.00350544093 | 0.0360616021 |
| body_collision_event (not instrumented) | 0 | 0 | 0 | 0 |
| **Total** | 0.175749497 | 0.0227039163 | 0.171963706 | 0.0227039163 |

## Threshold derivations

Let H=−hand_weight≥0, T=tracking weight, p=mean normalized hand pressure, and b(T)=mean of the **per-step floored** non-hand reward. For fixed measured trajectories, R_i(H,T)=b_i(T)−0.02 H p_i. A beats B iff b_A−b_B > 0.02 H(p_A−p_B). If p_A=p_B there is no hand-weight crossover. If p_A<p_B the inequality reverses. Raising T requires reapplying the floor to every sample, not adding a mean tracking term afterward.

Front poke, T=1: b_A=0.14239370973, b_B=0.0645396700262, p_A=0.00640931582205, p_B=0.00241316256901. Thus H_AB=(0.14239370973−0.0645396700262)/[0.02×(0.00640931582205−0.00241316256901)] = **974.112287** (signed hand weight **−974.112287**). B wins above this magnitude, A below it. This large number extrapolates fixed traces far outside tested training; it is not a recommended weight.

Full passage, T=1: b_A=0.175846569057, b_B=0.0227039162515, p_A=0.00970715777228, p_B=0. H_refusal=(0.175846569057−0.0227039162515)/[0.02×(0.00970715777228−0)] = **788.813041** (signed **−788.813041**). B has zero measured hand pressure and wins beyond that magnitude. Below it A wins this same-duration passage/refusal comparison.

At H=0 A already beats both alternatives in the two common encounter windows and beats B across the full traversal. Thus the measured intersection is **0≤H<788.813041 at T=1**, not an identified global usable interval. Clearance weight alone is not required to make this particular arm action attractive; failure to learn/use it cannot be diagnosed from coefficients alone.

The source warm-start learner uses gamma=0.98 (read from resume.pt learner.config.discounting). Applying gamma**step to the saved traces gives finite-horizon discounted crossover magnitudes **1085.369389** (poke) and **1487.251522** (passage). The same quotient uses discounted sums: poke non-hand A/B=5.587475076/3.396024074, pressure A/B=0.254200199/0.153246044; passage non-hand A/B=8.143017789/3.147461655, pressure A/B=0.167945908/0.000000000. Unknown continuation values are omitted; these are not infinite-horizon policy-optimal thresholds.

## Owner hypotheses and recommendation

(a) **Reject 20:1 as a general cost ratio; support arm preference only for these final, state-consistent traces.** Poke A−B tracking advantage is 0.009126501852 reward/step. Incremental target velocity+acceleration cost of A over B is 0.000019813208, a ratio of **460.63:1**, not 20:1. Velocity alone costs 0.000014358698. The A−C comparison gives a different ratio, **23.33:1**. Neither is the total ledger ratio: posture gating, handsdf, guidance, balance, torque and the floor all also change. Formulas normalize squared arm velocity/acceleration by joint weights, dt and scales (`cat_mjlab/task_math.py:138`); comparing −.05 to +1 misses those factors.

(b) **The maximum-bound arithmetic is correct, but the inference that −20 makes refusal optimal is not supported here.** Tracking max +1 and clearance max −20 are rates; at 20 ms their maxima are +0.02 and −0.40. The hand maximum requires both hands at/below zero surface gap. With one threatened hand and the other ≥20 cm, maximum is −0.20. At two-hand gaps 10 cm / 7.899 cm / 4 cm, the −20 penalties are respectively −0.020000 / −0.02928684 / −0.051200 per step, not −0.40 (`cat_mjlab/task_math.py:128`). At the measured full passage, −20 gives only -0.003882863 hand reward/step; the non-hand A−B advantage is 0.153142653. A remains preferred.

**Provisional pair (−5,+1)** gives 10× the current hand coefficient, while remaining 194.8× below the poke and 157.8× below the passage crossover. At that pair, measured poke A/B/C rewards are 0.141752778, 0.064298354, 0.052288258, and full passage A/B are 0.174875853, 0.022703916. The margins are empirical margins against these candidates only. Choosing −5 over −20 is a conservative design choice for unmeasured directions, not an experimentally identified optimum. **No tracking increase is needed in this sample.**

No single pair is proven to work across all tasks, and these results also do not prove that no such pair exists. Tracking penalizes velocity error, not world position: after a lateral detour, returning to commanded velocity can have identical tracking reward at a different position (`cat_mjlab/task_math.py:228`). Raising tracking therefore cannot guarantee path adherence. Production room commands may themselves become zero when route guidance is blocked (`cat_mjlab/task.py:239`), unlike this deliberately fixed-command audit. Global guarantees cannot be obtained from these two weights and two encounters.

**Retention risk:** unchanged frozen actions have exactly zero success-metric change under reward-only rescoring. That is not a training prediction. Raising T from 1 to 2 would add 0–0.02 pre-floor reward/step on *every* task (tracking raw in [0,1]); clipping can reduce the realized increment. There is no defensible measured conversion from that change to future clutter/narrow/CAT rates. User-supplied starting rates are .7137/.5582/.1495; without new learned-policy evidence their possible absolute changes are only trivially bounded by [−.7137,+.2863], [−.5582,+.4418], [−.1495,+.8505]. These are mathematical bounds, not forecasts. We keep T=1; clearance changes still carry unquantified retention risk. A proposed, unvalidated future acceptance tolerance of −.03 absolute would set floors .6837/.5282/.1195; this is a decision criterion, not measured risk.

## Elbows and forearms

`docs/LATERAL_CORRIDOR_AUDIT_CPU_20260922.md:102` records ten hand-compliant endpoints below 78.99 mm elbow/forearm margin: minimum 66.41 mm, all-arm minimum 64.01 mm; no elbow/forearm collision in those hand-compliant samples. **78.99 mm was the hand-envelope target, not an established whole-arm guarantee of the old box.** Hand compliance does not certify elbow/forearm clearance.

The current arm term is −2×mean(max(0,.08−d_elbow)^2), with no normalization (`cat_mjlab/task.py:497`). If one of two elbow probes has the documented 66.41 mm gap and the other is ≥80 mm, this formula gives **−0.000003693762/step**; both at that gap give −0.000007387524. This is a formula evaluation at the documented capsule gap, **not a claim that the elbow point probe measured the identical distance**. The reward samples elbow points with 5 cm radii (`cat_mjlab/task.py:272`), whereas the audit uses whole-arm proxies. Multiplying −2 by 40 gives only −0.00014775048/step for one elbow at that gap. It is not comparable to normalized hand pressure.

**Do not move arm clearance in numerical lockstep.** Keep −2 for this isolated launch recommendation, but regard whole-arm margin protection as unverified. Meaningful elbow/forearm protection needs its own calibrated, normalized clearance term and matching proxy measurements; merely scaling this point-probe squared-metre hinge does not enforce 78.99 mm, especially inside the non-hand reward floor. No arm reward or geometry was silently changed.

## Implemented telemetry, tests and launch

`cat_mjlab/response_split.py:1` defines per-environment events: onset at either hand <20 cm, closure when both ≥22 cm or at first outcome/termination. Lock threatened hand and outward normal at onset. Signed arm retreat is [(hand−root)_end−(hand−root)_start]·normal; world root displacement is ||root_end−root_start||, plus signed root retreat. This is the same translation-only subtraction as `scripts/diagnose_reactive_clearance_cpu.py:95`; root rotation is deliberately not removed. Unlike paired offline diagnostics, online events have no matched control, so normal gait swing/progress is included and these are not causal attributions.

Per-event outputs are `response_split/event_count`, `hand_relative_retreat_sum_m`, `root_world_displacement_sum_m`, `root_retreat_sum_m`, `ratio_sum`, `ratio_valid_count`, `zero_root_count`, `invalid_event_count` in transition metrics. Event_count masks individual records. Ratios are signed arm retreat / world-root displacement; root travel ≤1 mm is explicitly undefined and excluded from ratio means, not replaced with an enormous number. Training logs event-weighted means and pending-event counts under `training/response_split/`; passage acceptance logs the same sufficient statistics and ratios by existing cohorts. No unbounded event-history buffer or added observation features.

Wiring: `cat_mjlab/task.py:409` (reset state), `cat_mjlab/task.py:732` (pre-autoreset event capture), `cat_mjlab/runner.py:303` (training aggregation), `cat_mjlab/acceptance.py:125` (gate), and `cat_mjlab/acceptance.py:476` (offline world-sample adapter). State lives in the existing task info/checkpoint dictionary; legacy gate state is upgraded on load. Missing offline world samples/precomputed events cannot demonstrate an arm response; zero events are not evidence of success.

Validation: `tests/test_response_split.py:1` covers arm-only/body-only/mixed responses, undefined ratios, hysteresis, termination/reset, exact poke projection equivalence, collector logging without optimization, offline acceptance and checkpoint observer roundtrip, plus explicit tracking-weight override validation. Final focused suite: 63 passed, two unrelated synthetic runner fixtures deselected. Earlier broader checks: 66 passed; two legacy JAX parity tests could not import the absent mujoco_playground dependency. A separate full runner test attempt had two synthetic 3/4-observation fixtures rejected by the native 222/310 guard. No real optimizer updates or training run occurred. Test evidence: `outputs/response_weight_audit_cpu/tests.txt`.

Prepared command: `configs/pilots/clearance_weight_warmstart.sh:1`. Warm-start source is the requested resume.pt with a fresh optimizer, native observations, hand=−5, tracking=+1, W&B online / CAT-wholebody / skvayzer. The script was syntax/argument checked, **not executed**. This is a provisional measured-candidate recommendation, not authorization or a validated retention claim.
