# Width curriculum with CAT retention — built, not launched

## Completed build

The build parameterizes the narrow module width, certifies the resulting layouts,
generates fields, composes an immutable retention mixture, builds collision indices,
and certifies reset pools. All construction and validation used CPU only. No
training process, live checkpoint, W&B run or tmux session was modified.

Final manifests:

- `data/furniture/contrastive_width_v2/manifest.json`: 96 new scenes.
- `data/furniture/cat_width_retention_v2/manifest.json`: 2,434 scenes, comprising
  the exact 2,338-scene diversity-bank record prefix plus those 96 scenes.
- `data/furniture/cat_width_retention_v2_collision/manifest.json`: collision bank.
- `data/furniture/cat_width_retention_v2_resets/manifest.json`: 32 certified
  poses per scene, 77,888 total. The original 2,338 pools retain their exact rows.

The asset revision is v2 because source-bank generation summary counts were
separated into `retention_source_provenance` before final publication. The v1
construction artifacts are superseded; they are not launch targets.

There are four matched seeds per width, each containing open, protected-forward,
narrow and transition variants. Narrow scenes contain three narrow modules;
transition scenes contain one. Widths are 0.70, 0.64, 0.58, 0.52, 0.46 and 0.40 m.
The generator also accepts a width interval, for future within-rung variation.

## Honest certificate and reward metadata

Scaffolds are explicitly labelled `width-curriculum-scaffold-v1`, with scene
`difficulty=width_curriculum`; they cannot masquerade as strict specialist
contrast cases. Every scene still passes the existing full route, transition,
35-primitive, hand-field, stance and sealed-exterior-lane checks. Each module
records whether all tested forward poses actually collide, and the achieved
sideways body and hand-field clearances. These are sampled kinematic certificates,
not dynamic guarantees or proofs that other poses are impossible.

| Width | Minimum sampled sideways body separation | Sampled hand-field minimum range | Checked forward poses all blocked? | Inline SDF knee range |
|---|---|---|---|---|
| .70 m | .18305 m | .19389–.19487 m | No | .05 |
| .64 m | .15305 m | .15388–.18387 m | No | .05 |
| .58 m | .12305 m | .13686–.15418 m | No | .05 |
| .52 m | .09305 m | .10710–.11421 m | No | .05 |
| .46 m | .06305 m | .07394–.07500 m | No | .02394–.02500 |
| .40 m | .03305 m | .03407–.06411 m | Yes | 0–.01411 |

The best tested forward body separation falls from +.12154 m at .70 m to
+.00154 m at .46 m and -.02846 m at .40 m. Thus the final rung requires a
posture change within the audited pose family; wider rungs are scaffolding.

Each narrow zone stores `certified_achievable_clearance_m` and its existing
`sdf_reward_knee` parameter. The metadata rule is
`clip(certified_hand_field_clearance - .05, 0, .05)`: legacy .05 on easy rungs,
progressively relaxed at low clearance. Voxel alignment produces the reported
per-module range. This does not certify the full reward's sign. There is no
sideways-heading override, no weight-5 config and no gait-penalty change.
Retention and non-narrow zones use the original knee exactly.

## Sampling and progression

These are **reset probabilities**, not promised fractions of environment steps;
episode durations differ.

| Population | Total reset mass |
|---|---:|
| Original/published CAT | 10% |
| Procedural CAT | 20% |
| Retention furniture | 12.5% |
| Retention ordinary clutter | 7.5% |
| Contrastive open | 2.5% |
| Contrastive protected-forward | 10% |
| Contrastive narrow | 30% |
| Contrastive transition | 7.5% |

The four `sampling_group_masses` are therefore .10/.20/.125/.575; generic
clutter includes the 50% contrastive share. A separate eight-bucket sampler
preserves the retention/contrastive split even when adaptive weights change.
Retention uses role **-1**, never contrastive open's role 0. Its historical
survival-based adaptation is preserved. Contrastive scenes adapt on first clean
outcomes; protected/transition outcomes retain their posture qualification.

Start at rung 0. Unlock the next rung after at least **64 resolved leader narrow
outcomes at the current rung and >=35% clean-goal success**. Both values are
manifest parameters, exposed by the builder through `--min-completed` and
`--success-threshold`. Masses are configurable with `--role-masses` and
`--retention-group-masses` JSON objects. Followers, retention and transition outcomes do not unlock
rungs. Earlier rungs stay eligible; failure-weighted adaptation operates only
within the fixed buckets. This is an explicit success-gated ladder: merely
adding many widths to the old adaptive sampler would overweight hard failures.

The gate was lowered from 60% to 35% before launch to reduce stalling risk.
The unused v2 manifests were patched in place, including collision/reset hash
bindings; fields, geometry, reset arrays and min_completed=64 are unchanged.
There is no assurance of immediate advancement; monitor per-rung counters.
The gate uses cumulative outcomes, so early failures can delay advancement even
after recent performance improves. At the final rung there is no further unlock.

Native mjlab supports this progression. Legacy JAX training rejects the width
manifest explicitly rather than silently using an incorrect sampler. Legacy
standalone specialist role masses are now configurable too.

`SuccessWindow` always exposes CAT, ordinary-clutter and hand-protection goal
rates when those populations have resolved outcomes. Contrastive open counts
exclude retention. Stage and per-rung leader resolved/success counters are logged.
The existing checkpoint selection score remains contrastive-role balanced;
it is not a joint retention benchmark. Retention exposure and online metrics
help monitor preservation, but do not prove it without evaluation.

## Storage and memory

Cross-root hardlinks returned EXDEV and reflinks were unsupported. The merged
manifest therefore pins the existing diversity manifest by SHA256 and verifies
its exact record prefix and field hashes. Its 13.24 GiB of fields stay in place.
The existing source bank is a required dependency; do not move/change it.
New curriculum fields are hardlinked locally between the curriculum and merged
banks. No retention field arrays were copied or regenerated.

New fields total 2.46 GiB logically (about 1.2 GiB allocated on this compressed
filesystem). Collision arrays are 74.30 MiB logically, and resets 10.70 MiB.
Allow about 2.7 GiB on an uncompressed filesystem; actual new allocation is
roughly 1.3 GiB, plus small superseded build metadata/caches. All 2,434 field
scenes occupy **15.70 GiB as resident float32 tensors** regardless of hardlinks.

At 12,672 environments, estimate **27–30 GiB VRAM**, preferably at least 32 GiB
available. This uses the live retention run's read-only logged 24.78 GiB at the
same environment/minibatch dimensions plus approximately 2.16 GiB of extra fields
and small metadata/index overhead. No GPU benchmark was run. Do not run this
concurrently on an already occupied GPU.

A runtime checkpoint is estimated at approximately 248 MiB; best weights ~5 MiB.
Atomic replacement can briefly require ~501 MiB for checkpoints. Fixed filenames
are overwritten rather than retaining checkpoint history. Reserve another
0.6–1 GiB for run checkpoints/logs/caches; cache growth is not hard-bounded.

## Prepared launch command — DO NOT execute as part of this task

After the user cleanly stops the retention run, warm-start its durable native
snapshot. Do not use strict `--resume` across a different bank. The new run has
fresh optimizer, sampler, rung state and counters, while retaining learned
per-dimension sigma with the existing .15 ceiling. No sigma reset or NPZ
conversion is required. The source run and its snapshots are not copied/changed.

```bash
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_retention_merged_20260919/resume.pt \
  --fresh-optimizer \
  --max-action-std 0.15 \
  --discounting 0.98 \
  --bank-manifest data/furniture/cat_width_retention_v2/manifest.json \
  --body-collision-bank data/furniture/cat_width_retention_v2_collision/manifest.json \
  --body-collision-resets data/furniture/cat_width_retention_v2_resets/manifest.json \
  --run-dir outputs/cat_width_curriculum_retention_v2_12672 \
  --algorithm sapg \
  --num-envs 12672 \
  --num-minibatches 16 \
  --batch-size 792 \
  --unroll-length 32 \
  --max-updates 300 \
  --checkpoint-interval-updates 100 \
  --nconmax 64 \
  --njmax 256 \
  --device cuda:0 \
  --seed 0 \
  --wandb-mode online \
  --wandb-project CAT-wholebody
```

No `--passage-rewards` flag: the per-zone knee is in canonical scene metadata.
The dimensions give 405,504 transitions/update, 121,651,200 over 300 updates,
and 64 optimizer steps/update. The count is a bounded initial run, not a promise
that all six rungs will unlock within it. W&B receives a separate run name/ID.

## Validation

CPU-only suites passed: 44 native curriculum, diagnostics, heading, pilot,
runner, sampling and collision tests; 54 legacy certificate, reward and reset
tests. The full final bank also loaded successfully on CPU: all six sampling
stages preserved the configured bucket masses, locked widths had zero mass,
retention roles/knees/headings were unchanged, all new field hardlinks matched,
and the original reset prefix was exactly preserved. All 77,888 reset poses
passed the native collision checker. The prepared command parses and its paths
and batch dimensions validate; no training invocation or GPU benchmark ran.

## Reproduction

Field generation needs the existing build dependencies including scikit-fmm.
The installed WholeBody build environment supplies these without modifying the
live native training environment. Use `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2` for every build/check invocation.

1. Run `scripts/build_width_curriculum_bank.py --output data/furniture/contrastive_width_v2
   --merged-output data/furniture/cat_width_retention_v2 --success-threshold 0.35` with a Python environment
   containing NumPy, MuJoCo, JAX and scikit-fmm. The script defaults to the existing
   2,338-scene bank and the six widths/four seeds listed above. Published manifests
   are validated and reused, never silently replaced with different settings.
2. Run `scripts/build_body_collision_bank.py --field-manifest data/furniture/cat_width_retention_v2/manifest.json
   --base-collision-manifest data/furniture/body_collision_v1_20260916/manifest.json
   --output data/furniture/cat_width_retention_v2_collision --workers 2` in the native environment.
3. Run `scripts/build_body_collision_resets.py --cpu-native
   --field-manifest data/furniture/cat_width_retention_v2/manifest.json
   --collision-bank data/furniture/cat_width_retention_v2_collision/manifest.json
   --base-reset-manifest data/furniture/body_collision_resets_v2_20260916/manifest.json
   --base-robot-xml /home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody/outputs/hand_protection_implementation_20260917/base_robot.xml
   --output data/furniture/cat_width_retention_v2_resets --batch-size 256 --candidates-per-scene 64`.
   Collision/reset outputs must be new directories when rebuilding.

The reset append verifies the original model's content identity with only mesh
path relocation, exact old field records, collision-array prefixes and every
retained pose under the final native collision checker.
