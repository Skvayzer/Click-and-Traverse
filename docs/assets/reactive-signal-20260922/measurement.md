# Reactive hand signal: frozen-policy measurement, 22 September 2026

**S1 PASS (hand gradient), S2 PASS (current floor), S3 PASS (reachable contrast), S4 FAIL (tested tuck).** The exact-zero combined negative-band requirement also FAILS: native `handsdf` remains active. These are eight-seed CPU standing-probe results, not a training-success or protected-core certification.

No training, learner updates, scene-bank creation, policy/checkpoint writes, weight changes, or acceptance-gate changes. Only measurement code and text artifacts.

## Method and scope

`scripts/measure_reactive_signal.py` loads `outputs/cat_flat_balance_ppo_37632_20260920/resume.pt` read-only with mmap. It uses the checkpoint environment configuration with the already-approved launch overrides (-20 hand, -8 arm, .09 target, .20 anticipation, .8 near share, gravity compensation). The production `CATTask` analytic hook supplies observations. CPU physics is MuJoCo at 2 ms, control is 20 ms. Synthetic empty-floor standing background is generated in RAM, never stored as a bank. Observation noise, pushes and parameter randomization are disabled to isolate policy action noise.

Four conditions per seed: approach A, paired approach B+tuck, fixed-near, fixed-negative. All start from a common canonical reset; A/B phase, physical state and actor observations are asserted identical. The checkpoint Gaussian has pre-tanh upper sigma 0.15; no extra exploration noise is added. Common Gaussian draws are shared by A/B at each step, with independent seeds. A world-fixed 6 cm radius sphere targets the left hand, whose production enclosing radius is 0.10344617 m. Initial surface gaps are .35, .35, .05, .30 m. Approach speed is .10 m/s, stopping at the nominal initial-hand gap .03 m; the object does not chase the moving hand. Each trajectory runs for at most 250 steps (5 s).

Full approved sphere/capsule/box body proxies are checked against the sphere every physics substep through the normal six-region collision path. Objects are virtual terminal obstacles, as with the existing static bank, without physical impulses. No guard is enabled in this measurement. After contact/fall, analysis treats a branch as absorbing with zero remaining-horizon reward. The production elbow-point path is measured; the unvalidated capsule draft is not used. No contact after a terminal step is counted.

B overrides upper targets starting at 1.5 s (nominal anticipation entry), using joint-bounded IK for a 12 cm backward, 10 cm upward hand displacement. Waist targets remain nominal. Production slew limits still apply. This is one scripted intervention, not proof about all possible tucks. Physical clearance can come from arm and base movement; their contributions are not separately identified in this probe.

## A: implemented pure-function sweep

Positive cost convention below; reward contributions have the opposite sign. Both hands have the tabulated clearance and both elbows have matched clearance. For one threatened hand with the other safely outside the band, divide hand costs and slopes by two; likewise for one elbow. The full 71-row, 5 mm sweep is in `cpu/sweep.json`.

`C_hand = 16 clip((.09-d)/.09,0,1)^2 + 4 clip((.20-d)/.20,0,1)^2`

`C_arm = 8 max(.08-d,0)^2`; per-step costs multiply by .02.

| d (m) | Hand cost/s | Hand cost/step | Arm cost/s | Hand slope cost/s/m | Arm slope cost/s/m |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | -0.000000 |
| 0.20 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | -0.000000 |
| 0.09 | 1.21000000 | 0.02420000 | 0.00000000 | -22.000000 | -0.000000 |
| 0.05 | 5.41049383 | 0.10820988 | 0.00720000 | -188.024691 | -0.480000 |
| 0.03 | 10.00111111 | 0.20002222 | 0.02000000 | -271.037037 | -0.800000 |
| 0.00 | 20.00000000 | 0.40000000 | 0.05120000 | -395.555556 | -1.280000 |

Slopes: zero on [.20,.35]; `-200(.20-d)` on [.09,.20]; `-32(.09-d)/.09² - 8(.20-d)/.20²` on (0,.09). Arm slope is `-16(.08-d)` below .08 and zero above. Contact slopes are the approach-side limits. Outside the band there is deliberately no hand-clearance gradient.

The owner’s rounded arithmetic is verified against the implementation, not merely repeated. Ratios: 390.625 at contact; at .06641 m, hand=2.883864563/s, arm=.0014775048/s, ratio=1951.847847.

**Negative-band qualification:** `wholebody_hand_clearance` is exactly zero from .20 to .35. But the inherited native `handsdf` weight is +1, and `cat_mjlab/task_math.py:194` returns `-20 softplus((.05-d)/.02)`. At matched bilateral gaps its positive costs are .01105863/s at .20, .000074533/s at .30, and .000006118/s at .35. At .05 it adds 13.86294361/s, on top of the new 5.41049383/s. Thus the combined distance reward is not exactly zero in the negatives. This alone does not prove that permanent hunching wins after posture/effort penalties. The weight lever that removes this residual distance incentive is native `handsdf` +1 → 0, if authorized; no setting was changed.

## B: floor and conditioned slopes

Current production algebra (`cat_mjlab/task.py:549`): `clamp(sum_without_hand * dt,0,10000) + hand * dt`, followed by collision-event penalty. The hand penalty is outside the floor. The premise that clipping must erase its gradient does not match this checkout.

| Unmodified approach window | Steps | Actual base-floor clipped | Hypothetical all-inside clipped | Median pre/post partial slope, reward/step/m |
|---|---:|---:|---:|---:|
| overall | 2000 | 0.00% | 0.00% | 0.002194 / 0.002194 |
| d_lt_020 | 825 | 0.00% | 0.00% | 0.087447 / 0.087447 |
| d_lt_009 | 59 | 0.00% | 0.00% | 3.390317 / 3.390317 |
| d_lt_005 | 5 | 0.00% | 0.00% | 7.703686 / 7.703686 |

Only five unmodified approach samples are below 5 cm; this is weak coverage of the deepest band, not evidence about protected-core clipping in a bank.

| Disjoint distance bucket (m) | Steps | Median pre slope | Median post slope |
|---|---:|---:|---:|
| 0.00_0.05 | 5 | 7.70368576 | 7.70368576 |
| 0.05_0.09 | 54 | 3.09237432 | 3.09237432 |
| 0.09_0.20 | 766 | 0.08026830 | 0.08026830 |
| 0.20_0.35 | 1121 | 0.00026131 | 0.00026131 |

Slopes are autograd partials holding robot state, action and other field distances fixed, varying left-hand clearance only and re-evaluating all production reward terms. They are not policy gradients or regression slopes along correlated motion.

The fixed-near condition additionally has 9 actually clipped steps (9/228=3.95% conditioned on d<.20; 8/38=21.05% conditioned on d<.09). Their post-floor distance slopes remain positive, .212743–1.852210 reward/step/m, versus pre-floor slopes 1.227055–6.767780. Clipping removes other inside-floor distance contributions but not the explicit hand term. It is **not dead on arrival from the current floor** in this probe. No floor-width increase is indicated by these measurements.

## C: reachable policy-noise contrast

Objects are world-fixed at initial .05/.30 m gaps; achieved gaps are free to change. The following across-seed figures are all at the same 5 s time, avoiding pooled temporal drift.

| Initial gap | Clearance min / p10 / p50 / p90 / max (m) | Reward/step min / p10 / p50 / p90 / max | Clearance SD | Reward SD |
|---|---|---|---:|---:|
| near | 0.173098 / 0.214847 / 0.278131 / 0.326991 / 0.328622 | 0.343657 / 0.352595 / 0.366510 / 0.383263 / 0.384461 | 0.048744 | 0.012911 |
| negative | 0.174066 / 0.213466 / 0.285899 / 0.306826 / 0.335236 | 0.364297 / 0.367919 / 0.372965 / 0.379047 / 0.380764 | 0.045768 | 0.004977 |

Full pooled clearance/reward distributions, min/max and per-seed returns are in `cpu/report.json`. Near pooled p10/p50/p90=.193183/.264633/.346547 m (min .050675, max .453351); negative=.201293/.269905/.344679 m (min .128174, max .421646). The nominal negative enters d<.20 on 189/2000=9.45% of steps: initial gap alone does not guarantee a negative trajectory.

Using the checkpoint critic, gamma=.98, lambda=.95 and the production `compute_gae` in 32-step blocks, unnormalized first-step PPO advantage SD is .109573 (near) and .194394 (negative). Same-time clearance SDs are about 4.9 and 4.6 cm; hand-cost SDs at 5 s are .0002393 and .0002224 reward/step. Exploration is not mechanically trapped and advantages are non-degenerate. Advantage variation includes other reward terms and critic error; this does not establish a favorable hand-specific policy gradient.

## D: paired tucking counterfactual

Eight paired seeds; 4,000 paired bootstrap resamples. Finite 5 s return Δ(B−A)=-14.315403, 95% CI [-27.169901573704557, -2.6598487903829646]. Shared preterminal-prefix Δ=-1.967613, CI [-3.9604925820603967, -0.4230544526129959]. The latter removes lost future reward from unequal lifetimes, and is still negative.

Falls: A=0/8, B=0/8. Full-body contacts/terminations: A=0/8, B=3/8. B contacts at 2.42, 3.08, 4.08 s. Average hand clearance improves in 7/8 common prefixes, but the tested intervention is not reliably safe and does not pay.

All reward-term contributions below are integrated over the finite horizon; floor lift and collision penalty complete the accounting. Positive Δ favors tucking.

| Term | Mean Δ return | 95% paired CI | Mean Δ common prefix |
|---|---:|---|---:|
| feetgf | -5.41999988 | [-11.23999975, -0.91999998] | 0.00000000 |
| headgf | -2.70999994 | [-5.46224988, -0.44849999] | 0.00000000 |
| handsgf | -2.70999994 | [-5.74999987, -0.45999999] | 0.00000000 |
| tracking_orientation | -1.78225668 | [-2.98455027, -0.83563199] | -0.55564062 |
| tracking_root_field | -1.09396160 | [-1.69081321, -0.55467538] | -0.44893235 |
| foot_slip | -0.87926887 | [-1.48048113, -0.38162005] | -0.92183241 |
| wholebody_upper_target_acceleration | 0.86221651 | [0.78033641, 0.93646332] | 0.67971141 |
| body_rotation | -0.67603060 | [-1.15127575, -0.31132076] | -0.23033510 |
| collision_event | -0.37499999 | [-0.74999999, -0.12500000] | -0.37499999 |
| handsdf | 0.29792900 | [-0.57408784, 1.22577777] | 0.28747335 |
| wholebody_hand_clearance | 0.21246300 | [-0.13601503, 0.62462981] | 0.19755598 |
| joint_torque | -0.11482631 | [-0.32620002, 0.10082048] | -0.23808907 |
| wholebody_upper_target_velocity | 0.11059967 | [0.10970335, 0.11151064] | 0.08877026 |
| foot_balance | -0.08975903 | [-0.49586398, 0.33386342] | -0.34327685 |
| body_motion | 0.05927388 | [-0.06696992, 0.18195420] | -0.03763638 |
| wholebody_upper_clear_posture | -0.01063380 | [-0.01404943, -0.00677221] | -0.01128844 |
| floor_lift | 0.00852781 | [-0.00000000, 0.02558344] | 0.00852781 |
| smoothness_action | -0.00362042 | [-0.00762140, 0.00089111] | -0.00630612 |
| smoothness_joint | -0.00177856 | [-0.08833736, 0.08356058] | -0.06123764 |
| kneesdf | 0.00040647 | [-0.00082596, 0.00200799] | -0.00039262 |
| joint_limits | 0.00032114 | [-0.00066894, 0.00163236] | 0.00032114 |
| shldsdf | -0.00000551 | [-0.00000829, -0.00000300] | -0.00000574 |
| headdf | -0.00000001 | [-0.00000001, -0.00000000] | -0.00000001 |
| feetdf | 0.00000000 | [-0.00000000, 0.00000000] | 0.00000000 |
| foot_contact | 0.00000000 | [0.00000000, 0.00000000] | 0.00000000 |
| foot_clearance | 0.00000000 | [0.00000000, 0.00000000] | 0.00000000 |
| straight_knee | 0.00000000 | [0.00000000, 0.00000000] | 0.00000000 |
| foot_far | 0.00000000 | [0.00000000, 0.00000000] | 0.00000000 |
| wholebody_arm_clearance | 0.00000000 | [0.00000000, 0.00000000] | 0.00000000 |

On these fixed traces, linear replay would require hand magnitude 219.20 (instead of 20) merely to break even in mean on common prefixes. That is not a tested policy improvement, not a CI guarantee, and cannot fix contacts. Preferred lever: scene/controller design—validate an avoidance motion with 0/8 contacts and positive paired return before attributing failure to reward magnitude. No validated numerical adjustment that fixes S4 was established here.

## Reproduce / CUDA handoff

CPU measurement (already executed):
```bash
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python scripts/measure_reactive_signal.py --device cpu --mode all --seeds 8 --steps 250 --output docs/assets/reactive-signal-20260922/cpu
```

Exact same sample on the real GPU (not executed here):
```bash
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python scripts/measure_reactive_signal.py --device cuda:0 --mode all --seeds 8 --steps 250 --output docs/assets/reactive-signal-20260922/cuda
```

Larger GPU confidence sample, after parity check: replace `--seeds 8` with `--seeds 64` and output suffix with `cuda64`. The full aggregate includes every seed; detailed per-step JSONL logs the first eight to bound artifact size. CUDA uses production CATSimulation; CPU uses the existing MuJoCo bridge. CUDA timing or parity is not claimed. This is a measurement command, never a training launch.

## Verdict and limits

- S1 **PASS**: nonzero monotone hand-cost signal inside .20; separately, exact-zero combined negative-band requirement **FAILS** due native `handsdf`.
- S2 **PASS** for this standing probe: conditioned clipping measured, surviving gradients demonstrated even on clipped steps. Protected-core population clipping remains unmeasured.
- S3 **PASS** for accessible contrast: centimeter-scale variation under the checkpoint’s own action noise and non-degenerate production advantages.
- S4 **FAIL** for the tested tuck: negative paired return and 3/8 contacts. One failed intervention cannot prove that no other avoidance behavior is learnable.

The policy has an accessible, unclipped hand-reward signal, but safe beneficial avoidance is not established because the tested tuck loses return and causes contacts.

Free disk: initial inspection 2.611 GiB; sweep 2.610 GiB; final paired rollout start 2.605 GiB and end 2.605 GiB; summary/report stage 2.601318 GiB. Scripts stop below 512 MiB free; per-step text is capped at 32 MiB. Only small text reports and measurement code were written.
