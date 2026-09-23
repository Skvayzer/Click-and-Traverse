# Sideways-heading v3 ablation — prepared, not launched

Match the running v2 contract: 12,672 environments, batch size 792, 16
minibatches, 32-step unroll, four epochs, gamma .98, ceiling .15, seed 0.
Fresh optimizer/counters/sampler/environment; warm-start the same original
525M checkpoint, not a v2 snapshot. Preserve the loaded sigma head (no reset).
The only intentional learning change is the new passage metadata's heading
term. W&B and output directory are separate. No merged bank is used.

```bash
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_contrastive_sapg_mjlab_49152_20260919/resume.pt \
  --fresh-optimizer \
  --max-action-std 0.15 \
  --discounting 0.98 \
  --bank-manifest data/furniture/contrastive_table_v4/manifest.json \
  --body-collision-bank data/furniture/contrastive_table_v4_collision/manifest.json \
  --body-collision-resets data/furniture/contrastive_table_v4_resets/manifest.json \
  --passage-rewards configs/pilots/contrastive_table_v4_narrow_knee0_sideways.json \
  --run-dir outputs/cat_pilot_narrow_knee0_sideways_v3_12672 \
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
  --wandb-mode offline \
  --wandb-project CAT-wholebody-pilots
```

The trajectory product is 792*16=12672; both 792 and 12672 are divisible
by six. One update collects 405,504 transitions; 300 collect 121,651,200.
No source checkpoint conversion/copy is needed. Storage and GPU requirements
are those of v2 at this environment count, not the earlier 4,224-env proposal.
This command is for user review, not concurrent execution with the live pilot.

## Definition and scope

The existing `cat-passage-rewards-v1` schema accepts optional per-zone fields:
`heading_target_rad`, `heading_weight`, `heading_axis`. Specify all three
together. The supplied file sets pi/2, 1, true for exactly the same 64
certified narrow modules as the knee-only baseline (48 narrow-role + 16
transition-role). Original field/collision/reset assets and the knee-only file
are unchanged. Inline scene-zone metadata is also supported.

Let t be the route tangent, n its perpendicular and a the configured angle.
The target is d=cos(a)t+sin(a)n. For normalized horizontal pelvis and torso
headings h, axis mode costs:

    C = weight * smoothstep(zone_phase_weight) * mean(1 - abs(h dot d))

With a=pi/2 this is 1-|sin(yaw-relative-to-route)|: either shoulder leading
costs zero, forward/backward cost one. Directed mode (`heading_axis=false`)
uses 1-h dot d. Omitted overrides execute the historical forward-cost branch
exactly, including its original fade. Existing success qualification remains
unchanged. Narrow heading telemetry now measures alignment to the sideways
axis and is eligible for the existing leader-only rolling diagnostics.

The existing heading reward scale -1 and dt .02 give core penalties of -.0200
at 0 degrees, -.005858 at 45 degrees, and 0 at +/-90 degrees. The penalty
smoothly vanishes at zone edges. Existing zones start .625 m before hazards
and finish ramping .475 m before them in the probed scene, giving a precontact
learning signal. This does not prove a weight of one guarantees discovery.

## CPU numerical verification

Actual G1 IK/FK, stored fields, stopped-run scales and CATTask._rewards; no
training, GPU use, physics rollout or access to live mutable checkpoints.
The prescribed shuffle travels at .15 m/s, .714 Hz, 1 cm lift, 50% double
support and +/-2.5 cm pelvis sway. Quasistatic torques and prescribed contacts
mean this verifies reward compatibility, not dynamic executability.

| Module suffix | Sideways mean/min reward | Zero fraction | Forward mean pre/post clamp |
|---|---|---|---|
| 26b43475d00a | +.281493 / +.034920 | 0% | -3.559232 / 0 |
| 7a713c0016d4 | +.271493 / +.040817 | 0% | -3.479025 / 0 |

Sideways rewards equal the knee-only baseline. Forward-facing poses intersect
the gap: those are invalid attempted traversal samples, not episode returns.
The new term alone is -.02; it cannot restore discrimination inside an
already clamped frame. It supplies discrimination on the approach: .475 m
before the middle gap in the first scene, forward reward changes
+.136577 -> +.116577, while +90-degree reward remains +.198910 and -90-degree
reward remains +.172927. Both sideways heading costs are exactly zero;
the other terms depend on scene/body geometry. At half fade, the incremental
forward penalty is -.0100. The open-scene reference is unchanged.

The earlier five gait penalties sum to about -.0290/step for the shuffle;
it remains positive. No gait changes are included. Collision counts alone
do not prove insufficient yaw is the dominant failure cause; the newly
visible narrow heading compliance helps test that hypothesis.
