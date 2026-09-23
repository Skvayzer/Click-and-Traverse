# Raised-only v3 pilot

Created `data/furniture/cat_flat_hand_balance_v3_20260921` with separate collision and reset companions. Only the top-level runtime hand-contrast validity flags changed in scene data: 36 zones across 12 forward-protected scenes. Immutable nested source/certificate metadata remains historical provenance. Index 0 was independently selected by both hands' greater z height in every zone: 0.9398343–0.9398360 m versus tucked 0.6102141–0.6102146 m. No geometry, weights, tolerances, sampling masses, scene ordering, or reset poses changed.

Transition scenes remain byte-identical: their protected outer modules retain two alternatives, and their narrow middle modules remain inactive. This preserves the mixed-posture task and isolates the dedicated protected-scene intervention; transition scenes can still have the original trap. Narrow scenes also remain byte-identical (their hand objectives are inactive in this bank).

| Measurement | v2 | v3 |
|---|---:|---:|
| Scenes | 2375 | 2375 |
| Contrast zones | 108 | 108 |
| Hand-active zones | 60 | 60 |
| Positive-heading zones | 60 | 60 |
| Dedicated forward-protected zones | 36 | 36 |
| Those with exactly one valid region | 0 | 36 |

Existing contrast preflight passed with required hand contrast. Full file verification and collision-array loader passed; reset array SHA-256 verified. Five tests passed in `tests/test_raised_only_bank.py` and `tests/test_hand_balance_bank.py`. CPU reward evaluation at raised centers: cost 0, compliance 1; at tucked centers: cost 0.93564594, compliance 0. These are mathematical reward probes, not dynamic feasibility tests.

All 186 v2 bank files, 2391 collision files, and 2 reset files matched before/after SHA-256, size, mtime and inode snapshots. V2 manifest SHA-256 remains `646c6fb2b1c1ea0112032cfee2d24e42f091878b9eaa61a8a230b0c64093d0cc`. Copies have independent inodes. CPU-only work; no checkpoints opened or modified and no training commands issued. Direct tmux inspection was denied by the sandbox; the live metrics file nevertheless advanced from 71946 bytes/6 lines to 84052 bytes/7 lines during verification. This confirms continuing output, not a full process-health audit.

Allocated disk (`du -s -B1`): bank 433362944 bytes (413.29 MiB), collision 38880256 bytes (37.08 MiB), resets 6564864 bytes (6.26 MiB); total 478808064 bytes (456.63 MiB). Remaining disk approximately 6.8 GiB. Retention continues referencing the same existing storage root. No build temporaries remain.

The minimum-over-regions mechanism supports the basin diagnosis. A local 0.074 m optimum and 221 zero-compliance updates do not prove global impossibility. The raised acceptance region is an AABB expanded by a Euclidean 5 cm radius: axis extents 13.6 x 12.0 x 13.6 cm, with rounded edges, not a full cuboid of that size. Both hands must satisfy it simultaneously and z is absolute world height. This is not obviously impossible from size alone, but walking feasibility remains unverified; the scene explicitly has `dynamic_feasibility_validated=false`. This intervention removes the competing objective but may be insufficient. No geometry change is justified by the available evidence alone.

Launch command (prepared only, NOT executed; GPU pilot for a later user launch):

```bash
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 30720 --batch-size 768 \
  --num-minibatches 40 --unroll-length 32 \
  --checkpoint-native outputs/cat_hand_cont_30720_20260921/best.pt \
  --fresh-optimizer \
  --bank-manifest data/furniture/cat_flat_hand_balance_v3_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v3_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v3_20260921_resets/manifest.json \
  --run-dir outputs/cat_hand_raised_v3_pilot50_20260921 \
  --hand-contrast-region-weight -3 --hand-contrast-heading-weight -2 \
  --hand-contrast-approach-distance 0.40 --require-hand-contrast \
  --max-updates 50 --compile-task --device cuda:0 \
  --checkpoint-interval-updates 10 --wandb-mode disabled
```

The source best.pt is live and can change before launch; this command loads whatever version exists then. No snapshot was made.

Falsification: if `training/policy_forward_protected_hand_contrast_hand_region_compliance_fraction` stays exactly zero through all 50 updates, with positive corresponding `_sample_count`, this fix failed to break the observed zero-compliance behavior within the pilot budget. Missing exposure is inconclusive. A single positive sample falsifies the strict zero-hit symptom but is insufficient evidence of useful walking compliance; inspect sustained frequency and collision/goal outcomes. Use the protected-scene metric, since unchanged transitions retain both alternatives.
