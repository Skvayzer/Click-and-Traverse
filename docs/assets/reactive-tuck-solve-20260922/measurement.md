# Reactive tuck measurements — CPU, 2026-09-22

No training, launch, bank, checkpoint, reward-weight or acceptance-gate changes. Reports and measurement scripts only. Disk: 14.249 GiB initially; 14.105 GiB after direction rollouts.

## Contact attribution

Reproduction of the original 5 s probe (endpoint 0.03 m, speed 0.1 m/s): all three collision faults were **left_hip_roll_link versus analytic_sphere_0**, not hand/object, floor, static geometry or a self-collision fault. The commanded hand moved back/up, but the physical base response brought the hip into the sphere. This does not establish that avoidance is structurally unrewarding.

Clearance sign: norm(hand center − sphere center) − 0.10344617 − 0.06; positive means separated. At each failed seed’s closest measured hand approach, B actually had less clearance than A at the same time.

| Seed | Object-contact time s | Hip separation m | Hand clearance at contact m | B−A hand clearance at B closest m |
|---|---:|---:|---:|---:|
| 2 | 2.416001 | -0.0005277 | 0.186004 | -0.018858 |
| 3 | 3.072031 | -0.0007168 | 0.132911 | -0.043719 |
| 7 | 4.080069 | -0.0000598 | 0.161268 | -0.042875 |

Exact event fields/body pairs and all eight seed measurements: [contact-attribution.json](contact-attribution.json). Instrumentation: scripts/solve_reactive_tucks.py:17 and scripts/measure_reactive_signal.py collision_checker.

## Fixed-base constrained solve

**Best-found feasible clearances, not proven global maxima.** Left hand only, sphere radius 0.06 m; bucket midpoints 0.055, 0.145, 0.275 m. Each entry below is slow / fast speed (0.05 / 0.50 m/s). Joint limits intersect nominal ±0.8 rad; slew limit 2 rad/s. Full approved body proxies include forearm capsules. Fixed base and lower joints; 451 upper-involving nonadjacent self pairs. Graph neighbors within two edges are excluded.

Continuous fixed-base straight joint trajectories are certified with recursive displacement bounds (object/floor margin 1 mm, self gap >1 µm). These are commanded kinematic paths, not guarantees for the physical policy response. Floor-origin below starts at center height 0.062 m (2 mm above floor). Analytic box/capsule distances checked against production Torch formulas on 1,856 random cases: maximum absolute discrepancy 9.992e-16 m.

| Direction | Bucket midpoint m | Best clearance slow / fast m | Minimum command time slow / fast s | Available arrival time slow / fast s | Result |
|---|---:|---:|---:|---:|---|
| front | 0.055 | 0.665 / 0.665 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| front | 0.145 | 0.727 / 0.727 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| front | 0.275 | 0.837 / 0.582 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| front_left | 0.055 | 0.643 / 0.643 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| front_left | 0.145 | 0.717 / 0.717 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| front_left | 0.275 | 0.830 / 0.553 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| left | 0.055 | 0.575 / 0.575 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| left | 0.145 | 0.632 / 0.632 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| left | 0.275 | 0.729 / 0.441 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| back_left | 0.055 | 0.627 / 0.627 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| back_left | 0.145 | 0.941 / 0.941 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| back_left | 0.275 | 0.498 / 0.498 | 0.150 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| back | 0.055 | 0.670 / 0.670 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| back | 0.145 | 0.729 / 0.729 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| back | 0.275 | 0.827 / 0.578 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| back_right | 0.055 | 0.677 / 0.677 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| back_right | 0.145 | 0.737 / 0.737 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| back_right | 0.275 | 0.916 / 0.684 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| right | 0.055 | — / — | — / — | 5.900 / 0.590 | INFEASIBLE / INFEASIBLE |
| right | 0.145 | — / — | — / — | 4.100 / 0.410 | INFEASIBLE / INFEASIBLE |
| right | 0.275 | — / — | — / — | 1.500 / 0.150 | INFEASIBLE / INFEASIBLE |
| front_right | 0.055 | — / — | — / — | 5.900 / 0.590 | INFEASIBLE / INFEASIBLE |
| front_right | 0.145 | 0.830 / 0.830 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| front_right | 0.275 | 0.908 / 0.667 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| above | 0.055 | 0.570 / 0.570 | 0.400 / 0.400 | 5.900 / 0.590 | FEASIBLE / FEASIBLE |
| above | 0.145 | 0.536 / 0.536 | 0.400 / 0.400 | 4.100 / 0.410 | FEASIBLE / FEASIBLE |
| above | 0.275 | 0.567 / 0.397 | 0.400 / 0.150 | 1.500 / 0.150 | FEASIBLE / FEASIBLE |
| below | 0.055 | 0.926 / 0.926 | 0.400 / 0.400 | 6.614 / 0.661 | FEASIBLE / FEASIBLE |
| below | 0.145 | 0.972 / 0.972 | 0.400 / 0.400 | 4.814 / 0.481 | FEASIBLE / FEASIBLE |
| below | 0.275 | 0.993 / 0.888 | 0.400 / 0.221 | 2.214 / 0.221 | FEASIBLE / FEASIBLE |

Right-to-left-hand approaches already overlap the opposite hand at spawn; front-right danger crosses immutable lower-body geometry. These particular scene geometries are infeasible: change target hand/spawn/path, not weights. This is not a proof that every approach from those directions is impossible. Other directions have certified feasible configurations. Entire bucket intervals, intermediate speeds and global optimality are not certified. Source: scripts/reactive_tuck_geometry.py:18, :83, :157, :163; scripts/solve_reactive_tucks.py:43.

## Paired physical counterfactual

Eight paired seeds, frozen checkpoint, identical seeded policy innovations and world object trajectories; 250 × 20 ms = 5 s, 0.1 m/s, danger midpoint 0.055 m. No learning. Unlike the original 0.03 m endpoint / 1.5 s intervention, solved targets ramp from time zero to object arrival. Only changed upper joints are overridden; unchanged joints remain under the policy. Fractions scale the certified solved excursion, and every fractional path is separately certified. CIs are paired bootstrap 95%, 4,000 resamples. These are finite-horizon undiscounted returns, with absorbing termination.

**Contact distinction:** normal object faults are checked every 2 ms; an extra conservative self-proxy audit runs every 20 ms. Baseline already overlaps self envelopes in all eight seeds (nominal minimum proxy gap is only 0.045 mm). Added proxy overlaps are not proof of physical mesh contact, but prevent claiming a collision-free motion. This audit does not certify continuous physical self clearance between samples.

| Direction | Excursion fraction | Δ return [95% CI] | Object contacts A/B | Falls A/B | Added self-pair occurrences | Strict S4 |
|---|---:|---:|---:|---:|---:|---|
| front | 0.25 | +1.943 [+0.805, +3.491] | 0/0 | 0/0 | 1 | FAIL |
| front_left | 0.25 | +6.633 [+2.091, +11.941] | 0/0 | 0/0 | 1 | FAIL |
| left | 0.25 | -1.926 [-3.858, -0.421] | 0/0 | 0/0 | 1 | FAIL |
| back_left | 0.25 | +0.015 [-0.209, +0.221] | 0/0 | 0/0 | 1 | FAIL |
| back | 0.25 | -0.175 [-0.434, +0.064] | 0/0 | 0/0 | 5 | FAIL |
| back_right | 0.25 | -0.133 [-0.334, +0.063] | 0/0 | 0/0 | 5 | FAIL |
| below | 0.25 | +0.305 [-0.413, +0.961] | 0/0 | 0/0 | 1 | FAIL |
| front_left | 1.0 | +5.587 [+0.906, +10.893] | 0/0 | 0/0 | 1 | FAIL |
| above | 1.0 | -35.399 [-54.593, -16.441] | 1/4 | 0/4 | 9 | FAIL |
| below | 1.0 | -1.241 [-2.562, -0.012] | 0/0 | 0/0 | 5 | FAIL |
| front | 0.125 | +1.656 [+0.505, +3.182] | 0/0 | 0/0 | 2 | FAIL |
| front_left | 0.125 | +6.774 [+2.475, +11.867] | 0/0 | 0/0 | 2 | FAIL |
| front (holdout) | 0.25 | +1.240 [+0.526, +2.140] | 0/0 | 0/0 | 3 | FAIL |

Right and front-right: S4 FAIL / not runnable for the authored danger trajectories. Above 0.25: fractional path failed kinematic certification, so not simulated. Full above trajectory was simulated. All other strict FAILs mean the tested motion has not satisfied the full requirement; finite candidate search cannot establish that no paying collision-free tuck exists.

Front and front-left demonstrate a positive reward contrast without added **object** contacts. The fresh front 0.25 holdout uses seed offset 29000 instead of 19000; it also pays, but has new conservative self pairs. This distinguishes a reward success from the still-unresolved physical safety requirement. No claim of training convergence follows from this counterfactual.

### Reward deficits and collision details

Per-term means and bootstrap CIs, every paired seed, contact bodies and self-pair identities are in the linked JSON files. Term contributions include floor correction so the ledger accounts for final reward. For losses, the absolute mean deficit below is the minimum extra return needed merely to cross zero on this recorded trace; it is NOT a validated weight prescription.

- front fraction 0.25 (paired-danger-changed-verified.json): Δ=+1.943006; largest deficits: tracking_orientation -0.046525, wholebody_upper_target_acceleration -0.018837, tracking_root_field -0.011702, joint_torque -0.006894. Added proxy pairs: [{'seed': 0, 'pair': ['right_hand', 'right_hip_pitch_link']}].

- front_left fraction 0.25 (paired-danger-changed-verified.json): Δ=+6.632709; largest deficits: foot_balance -0.346024, foot_slip -0.191193, joint_torque -0.155185, body_rotation -0.122375. Added proxy pairs: [{'seed': 7, 'pair': ['torso', 'pelvis']}].

- left fraction 0.25 (paired-danger-changed-verified.json): Δ=-1.926398; largest deficits: handsdf -0.900378, foot_balance -0.392450, wholebody_hand_clearance -0.380567, joint_torque -0.131472. Added proxy pairs: [{'seed': 2, 'pair': ['torso', 'pelvis']}].

- back_left fraction 0.25 (paired-danger-changed-verified.json): Δ=+0.014633; largest deficits: foot_balance -0.092427, joint_torque -0.080991, body_rotation -0.069594, tracking_root_field -0.061250. Added proxy pairs: [{'seed': 2, 'pair': ['left_hand', 'left_hip_pitch_link']}].

- back fraction 0.25 (paired-danger-changed-verified.json): Δ=-0.175493; largest deficits: foot_balance -0.252440, body_rotation -0.094338, joint_torque -0.066819, tracking_root_field -0.056062. Added proxy pairs: [{'seed': 7, 'pair': ['right_elbow_link', 'torso']}, {'seed': 2, 'pair': ['torso', 'pelvis']}, {'seed': 3, 'pair': ['torso', 'pelvis']}, {'seed': 0, 'pair': ['torso', 'pelvis']}, {'seed': 6, 'pair': ['left_hand', 'left_hip_pitch_link']}].

- back_right fraction 0.25 (paired-danger-changed-verified.json): Δ=-0.133245; largest deficits: foot_balance -0.233331, joint_torque -0.113034, tracking_root_field -0.072763, foot_slip -0.022132. Added proxy pairs: [{'seed': 5, 'pair': ['right_elbow_link', 'torso']}, {'seed': 3, 'pair': ['torso', 'pelvis']}, {'seed': 4, 'pair': ['torso', 'pelvis']}, {'seed': 6, 'pair': ['right_elbow_link', 'torso']}, {'seed': 0, 'pair': ['right_elbow_link', 'torso']}].

- below fraction 0.25 (paired-danger-changed-verified.json): Δ=+0.305319; largest deficits: handsdf -1.209897, wholebody_hand_clearance -0.878542, tracking_orientation -0.213006, kneesdf -0.115535. Added proxy pairs: [{'seed': 1, 'pair': ['right_hand', 'right_hip_pitch_link']}].

- front_left fraction 1.0 (paired-danger-changed-max-extra.json): Δ=+5.587212; largest deficits: foot_slip -0.459132, body_rotation -0.410626, joint_torque -0.355498, wholebody_upper_target_acceleration -0.318609. Added proxy pairs: [{'seed': 7, 'pair': ['torso', 'pelvis']}].

- above fraction 1.0 (paired-danger-changed-max-extra.json): Δ=-35.398597; largest deficits: feetgf -13.120000, headgf -6.630000, handsgf -6.630000, foot_balance -3.103926. Added proxy pairs: [{'seed': 0, 'pair': ['right_hand', 'right_hip_pitch_link']}, {'seed': 1, 'pair': ['right_hand', 'right_hip_pitch_link']}, {'seed': 5, 'pair': ['torso', 'pelvis']}, {'seed': 5, 'pair': ['torso', 'right_hip_roll_link']}, {'seed': 4, 'pair': ['torso', 'right_hip_roll_link']}, {'seed': 6, 'pair': ['torso', 'pelvis']}, {'seed': 6, 'pair': ['torso', 'right_hip_roll_link']}, {'seed': 0, 'pair': ['torso', 'pelvis']}, {'seed': 0, 'pair': ['torso', 'right_hip_roll_link']}].

- below fraction 1.0 (paired-danger-changed-max-extra.json): Δ=-1.241125; largest deficits: body_rotation -0.588370, foot_balance -0.485265, foot_slip -0.464520, tracking_orientation -0.379658. Added proxy pairs: [{'seed': 1, 'pair': ['right_hand', 'right_hip_pitch_link']}, {'seed': 3, 'pair': ['torso', 'pelvis']}, {'seed': 4, 'pair': ['torso', 'pelvis']}, {'seed': 6, 'pair': ['torso', 'pelvis']}, {'seed': 0, 'pair': ['torso', 'pelvis']}].

- front fraction 0.125 (paired-danger-changed-small.json): Δ=+1.656286; largest deficits: tracking_orientation -0.040155, wholebody_upper_target_acceleration -0.026017, joint_torque -0.018314, tracking_root_field -0.009872. Added proxy pairs: [{'seed': 0, 'pair': ['right_hand', 'right_hip_pitch_link']}, {'seed': 7, 'pair': ['right_hand', 'right_hip_pitch_link']}].

- front_left fraction 0.125 (paired-danger-changed-small.json): Δ=+6.773606; largest deficits: joint_torque -0.153115, foot_slip -0.144819, foot_balance -0.122966, body_rotation -0.110719. Added proxy pairs: [{'seed': 7, 'pair': ['torso', 'pelvis']}, {'seed': 7, 'pair': ['right_elbow_link', 'torso']}].

- front fraction 0.25 (paired-danger-changed-holdout.json): Δ=+1.240127; largest deficits: joint_torque -0.086351, body_rotation -0.081576, foot_slip -0.073582, tracking_orientation -0.055921. Added proxy pairs: [{'seed': 6, 'pair': ['right_hand', 'right_hip_pitch_link']}, {'seed': 7, 'pair': ['torso', 'pelvis']}, {'seed': 7, 'pair': ['right_elbow_link', 'torso']}].

For left and below the small excursion worsens hand clearance reward itself: increasing its weight amplifies that loss. Change the physical trajectory/controller. Back/back-right small-excursion deficits are approximately 0.1755/0.1332 return, dominated by balance/effort rather than insufficient kinematic reach; that much net benefit must be recovered, but no tested weight change is established to do so. Above loses mainly field tracking and terminates early; redesign motion/base stabilization before tuning clearance. No weight changes are recommended from these traces.

## Native handsdf residual (derived from the production pure function)

Constant distance and no clipping assumed; actual recoverable native reward can be smaller because the native term is inside the base reward floor. Explicit hand term is outside that floor (cat_mjlab/task.py:549).

| Distance m | Native cost/s, both hands | Native 10 s, one hand | Native 80 s, one hand | Explicit 10 s, one hand |
|---|---:|---:|---:|---:|
| 0.2 | 0.011058630 | 0.055293148 | 0.442345180 | 0.000000 |
| 0.3 | 0.000074533 | 0.000372665 | 0.002981317 | 0.000000 |
| 0.35 | 0.000006118 | 0.000030590 | 0.000244722 | 0.000000 |
| 0.05 | 13.862943611 | 69.314718056 | 554.517744448 | 27.052469 |

At 0.20 m the residual is 0.2044% of the explicit cost at 0.05 m for the same duration/hand count. It is nonzero, but small. Zeroing native handsdf would also remove its substantial near-field penalty (13.86294/s for both hands at 0.05 m), not just the negative-band tail. Weight unchanged. Source: scripts/solve_reactive_tucks.py:275; cat_mjlab/task_math.py:194.

## Reproduce (measurement only)

Run from repository root, without uv. Existing kinematic-solutions.json is the input to paired mode. CUDA output receives a distinct filename; no training or checkpoint writes.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python scripts/solve_reactive_tucks.py --device cuda --mode paired --bucket danger --fractions .25 --audit-self --run-tag=-cuda
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python scripts/solve_reactive_tucks.py --device cuda --mode paired --directions front_left,above,below --bucket danger --fractions 1 --audit-self --run-tag=-cuda-max
```

CPU runs completed locally. GPU parity remains unmeasured here; these CUDA commands are handed off for the visible GPU. No bank or videos produced. Full physical sweep of other buckets/speeds, right-hand/bilateral targets, globally maximal configurations, mesh-level self-contact verification and a safe paying controller remain outstanding.
