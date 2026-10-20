# Deploying `generalist_finetune_v1_u596` on the real G1

Policy: three-teacher DAgger student fine-tuned with RL (fine-tune v1, update 596, 2026-10-20) -- traversal
(hurdles, crouch), narrow side gaps, rooms/tables/passages, keyboard driving. Whole body, 29 joints.

## What is saved

| Where | What |
|---|---|
| `outputs/saved_checkpoints/generalist_finetune_v1_u596.pt` | full training checkpoint (actor, critic, optimizer, task state, contract) |
| `outputs/deploy/generalist_finetune_v1_u596/policy.onnx` | actor: `obs [B,226] -> action [B,29] = tanh(mean)`, `mean [B,29]` |
| `.../policy_torchscript.pt` | same actor, TorchScript |
| `.../deploy_spec.json` | everything below in machine-readable form: all 226 feature names + indices, joint order, nominal pose, soft limits, kp/kd, torque limits, action rule, gait clock, field sites/frames/clamps, odometry delay, training randomisation, full environment config, checkpoint SHA-256 |
| `.../robot.xml` | the exact training MJCF (sites for the field points, masses for gravity compensation, FK) |
| `.../reference_rollouts.npz` | 2203 deterministic simulator steps in 6 scenes: per step the observation, action, and every input it was built from (joint state, IMU, site positions, held odometry, fields, command, phase, targets) |
| `.../SHA256SUMS` | hashes |

Regenerate: `scripts/export_deploy_bundle.py --checkpoint ... --out ...` (CPU only). The exported ONNX and
TorchScript reproduce the recorded actions to 7.5e-7.

## Interface (from deploy_spec.json; the code is authoritative)

* 50 Hz policy; PD targets held between policy steps. Joint order = Unitree G1 29-DoF motor order (legs 0-11,
  waist 12-14, left arm 15-21, right arm 22-28) -- verify on the robot variant.
* Action: legs `target = clamp(prev_target + 0.5 a)`, waist/arms `clamp(nominal + 0.8 a)` rate-limited to 2 rad/s;
  soft limits = 95% of the joint range. Initial target = nominal pose.
* Torque in training: `kp (target - q) - kd qdot + gravity feedforward` (waist + arms only), clipped. kp/kd in the
  spec (e.g. hips 100, knees 200, waist 300). On the G1: LowCmd `q = target, dq = 0, kp, kd, tau = feedforward`.
* Observation (226): projected gravity and gyro of the pelvis IMU; `q - nominal`; `qdot`; last action; current
  targets; command `[move, vx, vy, yaw]` (delayed, heading frame); foot height 0.07; gait phase cos/sin; fields at
  11 body sites + 2 elbow probes (guidance 3, boundary 3 zeroed where sdf >= 0.5, distance clamped [-1, 0.5]),
  rotated into the heading frame (horizontal pelvis x axis), sampled at site positions built from a pelvis pose held
  for 5 steps (10 Hz) plus current body-local offsets; 4 distance rates.
* Fields: CAT pipeline, 4 cm, guidance toward the goal at 0.75 m height. **Guidance vectors are unit length in
  training** (median |gf| = 1.000 at every site in the reference rollouts).
* Training randomisation: observation noise (gyro 0.2, gravity 0.05, joint pos 0.03, joint vel 1.5), kp/kd x[0.75,
  1.25], random torque noise 10% of the limit, pushes every 5-10 s. **Not randomised: action latency
  (`enable_ctrl_delay: false`), friction, mass, centre of mass.**

## Against humanoid_navigation `feature/cat-perception-preview`

That branch was built for the released CAT Generalist v1 (162 inputs, 12 leg actions, upstream feet-only MJCF).
This policy is different. Reusable as is: Mid-360 ingestion, OctoMap, map-frame volume, read-only LowState and
FK machinery, the field-computation core, the no-motion safety structure.

### Must be implemented / changed

1. **Model + embodiment**: replace the pinned checkpoint with this ONNX (226 -> 29); use `robot.xml` for FK and sites
   (head, imu_in_pelvis, imu_in_torso, feet, palms, knees, shoulders, elbow probes) -- regenerate the FK fixtures
   from it and check against `sites_world` in the reference rollouts.
2. **Observation builder** (new, the core piece): implement every rule in `deploy_spec.json`, including the command
   (`compute_cmd_from_rtf` on the delayed root/limb fields + `update_phase` stop/restart logic), the gait clock,
   the 10 Hz held pelvis pose for field sampling, heading frame, clamps, distance rates. Test offline: feed the
   recorded inputs from `reference_rollouts.npz`, require the 226-vector to match `obs` (~1e-5) for all 2203 steps.
3. **Fields match training**: unit-length guidance (not 0.6), boundary = normalised SDF gradient, same clamps;
   field cached in the map frame, recomputed ~1-2 Hz, sampled at 50 Hz at the current site positions. Decide and
   test the unknown-space policy (training never saw unknown cells; treating them as occupied is conservative but
   can cause refusals). Volume: obstacles below 10 cm are dropped as ground and above 1.4 m are not represented
   -- check against hurdle heights and beams you intend to use.
4. **Pose**: pelvis pose in the map frame at >= 10 Hz = SLAM body pose (Mid-360/IMU on the head) x calibrated
   LiDAR-to-torso transform x waist FK. Latency and drift directly move the field samples; narrow gaps need cm.
5. **Actuation** (separate review): LowCmd publisher (q, kp, kd, tau) at the motor-command rate with targets held
   for 20 ms; gravity feedforward for joints 12-28 from `robot.xml` (Pinocchio or MuJoCo); Dex3 hands held fixed
   (they are rigid in training); safe hand-over from the Unitree controller (damping -> stand at nominal -> policy),
   remote E-stop to damping, watchdogs on stale state / stale fields / pose jumps.
6. **Goal interface**: the policy walks toward a goal via the field (Click-and-Traverse). Use the existing
   `/g1_cat/goal_preview` style goal (or Nav2 goals) -> field toward that goal; hold (zero command) on arrival.

### Recommended order

1. Offline: observation builder vs `reference_rollouts.npz`; ONNX latency on the robot CPU/GPU.
2. **Sim-to-sim**: run the robot-side stack (observation builder + ONNX + PD/feedforward) against plain MuJoCo with
   `robot.xml` (not MuJoCo Warp), pose at 10 Hz with noise/latency, fields from a simulated OctoMap of a scene,
   1-2 sample delay on actions. If it does not walk there, it will not walk on the robot.
3. Shadow mode on the robot: live inputs -> observation -> action, logged, no motors; compare with sim statistics.
4. Gantry: stand (no goal), then stepping in place, reduced gains first.
5. Floor with a crane: flat walking to a goal; then one 10 cm hurdle, one beam, wide gaps; then narrower.

### Known risks of this policy on hardware

* No action-latency randomisation in training -> consider a short fine-tune with `enable_ctrl_delay` (0-2 steps)
  plus friction/mass randomisation before hardware.
* Obstacles were never physical in training (contact ended the episode): any real contact is outside its experience.
* Flat floor only; no slopes or stairs; hands rigid.
* Fields in training were perfect (no unknown cells, no LiDAR noise, static scene).
