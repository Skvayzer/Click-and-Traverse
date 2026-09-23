# Narrow-knee pilot — prepared, not launched

This is a fresh W&B lineage, optimizer, environment, sampler and update counter,
warm-starting all six policies and the critic from the stopped native snapshot.
Only the actor scale-output rows are reinitialized to sigma 0.05. No NPZ export,
checkpoint copy, field-bank rebuild or merged bank is required.

The 96-scene field/collision/reset banks remain unchanged. The bank-hash-pinned
`configs/pilots/contrastive_table_v4_narrow_knee0.json` sidecar supplies a
`sdf_reward_knee` of 0.00 for 64 certified narrow zones: 48 in narrow scenes and
16 in transition scenes. The temperature remains 0.02. The knee blends from
0.05 at zone boundaries to its metadata value using smoothstep of the existing
fade ramp. Other zones/scenes retain the exact historical DF arithmetic.
Proxy collision, field/elbow termination, reward scales and the reward floor
are unchanged.

## Command (do not execute until reviewed)

Run from the repository root. Do not create files inside the new run directory
before invocation; the runner requires a fresh empty or absent directory.

```bash
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_contrastive_sapg_mjlab_49152_20260919/resume.pt \
  --fresh-optimizer \
  --init-action-std 0.05 \
  --max-action-std 0.15 \
  --discounting 0.98 \
  --bank-manifest data/furniture/contrastive_table_v4/manifest.json \
  --body-collision-bank data/furniture/contrastive_table_v4_collision/manifest.json \
  --body-collision-resets data/furniture/contrastive_table_v4_resets/manifest.json \
  --passage-rewards configs/pilots/contrastive_table_v4_narrow_knee0.json \
  --run-dir outputs/cat_pilot_narrow_knee0_sigma015_gamma098_4224 \
  --algorithm sapg \
  --num-envs 4224 \
  --batch-size 66 \
  --unroll-length 32 \
  --max-updates 300 \
  --checkpoint-interval-updates 100 \
  --nconmax 64 \
  --njmax 256 \
  --device cuda:0 \
  --seed 0 \
  --wandb-mode offline \
  --wandb-project CAT-wholebody-pilots
```

4,096 is not divisible by six and violates SAPG's equal-policy-population
contract. The nearest compatible one-chunk setup is 4,224: 704 environments per
policy, 64 minibatches × 66 physical trajectories, augmented to 4,928
trajectories / 77 per minibatch. This preserves the six-policy architecture and
existing sampling contract. At 32 steps/update, 300 updates = 40,550,400 physical
transitions. A smaller alternative is 3,840 environments / batch size 60.

Gamma was already a learner setting; `--discounting` now exposes a per-run
source-config override. Omitting it inherits the source (.98 here). To ablate,
change the argument and use another fresh run directory, e.g. .995. No default
has changed. `--max-action-std 0` explicitly disables the ceiling; omitting it
inherits the source setting, defaulting to off for old checkpoints. The CAT
scale head uses `softplus(raw)+.001`, not literal log-sigma parameters; its
monotone raw output is capped consistently for sampling, log probabilities,
SAPG relabeling and entropy.

For context, .98 has a 50-step / 1-second effective discount horizon. A 0.65 m
module at 0.6 m/s takes approximately 54 steps; 108 steps corresponds to
0.3 m/s. The proposed command retains .98 for the initial pilot.

## Disk and VRAM estimate before launch

Read-only measurements of the stopped snapshot:

- Logical file size: 964,068,129 bytes (about 919 MiB); about 719 MiB physically
  allocated on this compressed filesystem. It is not copied or rewritten.
- Learner tensors: 15.14 MiB; simulator: 694.50 MiB; task: 209.67 MiB at 49,152
  environments. Model weights alone: 5.05 MiB.
- Scaling per-environment state to 4,224 gives approximately **93 MiB** for the
  pilot's full `resume.pt`; `best.pt` approximately **5 MiB**.

Checkpoints overwrite fixed names. There is an initial snapshot, snapshots at
100/200/300 updates, and a final save; `best.pt` overwrites whenever the selection
score improves. No checkpoint history accumulates. Atomic replacement needs
old and temporary snapshots simultaneously: approximately **191 MiB checkpoint
peak**, excluding logs. Budget **0.3–0.5 GiB incremental disk** for checkpoints,
300 offline W&B/JSONL events, and modest cache growth; backend cache growth is
not a hard bound. This is below the approximately 3.7 GiB available at review.
Torch task compilation is omitted to avoid an additional Inductor cache build.

The bank contains approximately **2.46 GiB** of immutable GPU field arrays.
The physical rollout is approximately **0.55 GiB**, plus approximately **0.65 GiB**
for SAPG augmentation. The old run logged 41.10 GiB GPU use; scaling its variable
portion suggests about 5.8 GiB before allowance for fixed overhead/caches.
Plan for **6–9 GiB VRAM**, preferably **12 GiB available**. This is an estimate:
no GPU benchmark or training was launched, and the NVIDIA driver was unavailable
in the review environment.

## Compatibility and diagnostics

Native warm start intentionally copies weights only and accepts the old missing
`max_action_std` field as the default off. Exact `--resume` still requires full
contract equality, including `source_sha256`, the reward metadata hash, gamma,
sigma ceiling and recorded initialization settings. There is no source-hash
bypass. Later resume must use the same initialization arguments except omit
`--fresh-optimizer` and add `--resume`; sigma is not reset on that path.

All earlier telemetry remains: dense leader/role compliance and costs, valid-zone
progress, leader/role collision and timeout counts, importance/ESS/clip metrics,
leg/upper sigma, and window length plus role counts. Original snapshots missing
dense history still receive empty history rather than invented zero samples.
