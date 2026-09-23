> **Historical record — superseded.** Native training now uses actor 222 / critic 310, no authored box features or rewards, and starts from scratch. Old expansion utilities are retired; no checkpoint trim is supported or requested. Commands and old test paths below describe the historical experiment, not the current workflow. See [current removal report](NATIVE_CLEARANCE_REMOVAL_20260922.md).

# Native mjlab hand-objective observations (2026-09-22)

The explicit hand-region objective was missing from the actor. The original
`_base_features` and `task_math.observations` include proprioception, walking
commands, gait and geometry fields, but no region bounds or hand-zone state.
`make_clutter_fields` constructs SDF/BF/GF from occupancy and the walking goal;
`CATTask._fields` replaces room guidance with route guidance, not hand targets.
`contrast_reward_terms` independently reads paired hand-region bounds. `navi`
is the world/navigation-frame rotation, not a hand command.

This is a confirmed missing-input defect, not proof that all previous training
was mathematically impossible. Fixed targets can be learned from proprioception,
and geometry can correlate with required posture. At live-log step 12,779,520,
`selection/flat_walking_compliant` was 0.3009237620, protected region cost was
0.7508263782, and seeded protected cost after 100 ms was 0.7707131623. These are
training metrics from `outputs/cat_hand_priority_30720_20260922/metrics.jsonl`,
not independent evaluation. The supplied claim about 12 experiments was not
independently audited.

## Contract and frame

`mjlab_observation_contract()` is `cat-mjlab-hand-objective-v3`:
**254 actor inputs, 342 critic inputs, 29 actions**. It appends 32 identical
features to each old tensor. All old feature indices are preserved.
`wholebody_observation_contract()` deliberately remains the legacy JAX v2
222/310 contract; no legacy JAX task claims to produce the new features.

For each of two paired region alternatives and each hand, expose centre-minus-
hand delta (3 metres) and half-extents (3 metres). Alternatives are paired across
hands, exactly like the reward; a policy may not mix left region 0 with right
region 1. This avoids silently discarding the lower/tucked alternative in v5.

Use route tangent / left normal / world up. The reward boxes are axis-aligned
in this frame, so half-extents retain their exact meaning. A body-frame box
would need its orientation too; transforming only half-extents would lose the
actual tolerance. Route-heading-minus-pelvis-heading cosine/sine relates these
coordinates to the actor. Existing gravity/proprioception describe body tilt.
Target Z is absolute world height minus hand world height, not root-relative
height, matching the reward even while crouching. No yaw, XY translation or
root-height ambiguity is introduced.

Per-hand activity and per-alternative validity are explicit. Progress is [0,1]
across the effective reward interval; fade weight is the exact existing reward
weight. For forward-protected scenes this includes the configured incoming
approach interval, bounded by the preceding zone's exit. This is reward activity,
not strict core-compliance activity. Activity remains 1 at a zero-weight zone
entry so the policy sees the target before the incoming fade increases.

Outside a hand objective, **all 32 features are zero**. Inactive hands and invalid
alternatives have zero delta/extent slots even if their packed metadata is nonzero.
Flat-balance objectives use region 0, both hands active, progress 0 and weight 1;
region 1 is invalid and zero. Current FK positions are used for both actor and
critic; these new command-like inputs are not passed through the older delayed
geometry-field sampler or random observation noise. Deployment would need the
same route/region command plus estimated hand FK; no separate arm controller is
required by this design.

The complete appended feature list, in tensor order:

```text
hand_objective.region.0.left.delta.x
hand_objective.region.0.left.delta.y
hand_objective.region.0.left.delta.z
hand_objective.region.0.left.half_extent.x
hand_objective.region.0.left.half_extent.y
hand_objective.region.0.left.half_extent.z
hand_objective.region.0.right.delta.x
hand_objective.region.0.right.delta.y
hand_objective.region.0.right.delta.z
hand_objective.region.0.right.half_extent.x
hand_objective.region.0.right.half_extent.y
hand_objective.region.0.right.half_extent.z
hand_objective.region.1.left.delta.x
hand_objective.region.1.left.delta.y
hand_objective.region.1.left.delta.z
hand_objective.region.1.left.half_extent.x
hand_objective.region.1.left.half_extent.y
hand_objective.region.1.left.half_extent.z
hand_objective.region.1.right.delta.x
hand_objective.region.1.right.delta.y
hand_objective.region.1.right.delta.z
hand_objective.region.1.right.half_extent.x
hand_objective.region.1.right.half_extent.y
hand_objective.region.1.right.half_extent.z
hand_objective.left.active
hand_objective.right.active
hand_objective.region.0.valid
hand_objective.region.1.valid
hand_objective.zone.progress
hand_objective.zone.weight
hand_objective.route.cos_yaw
hand_objective.route.sin_yaw
```

## Compatibility and checkpoints

Sizes are derived from the named contract in the native task, configuration and
learner defaults. Task allocation and assertions use those sizes. Released CAT
export/expansion now maps named features into 254/342 and zero-initializes added
input weights. CPU conversion parity tests cover that path and Adam/SAPG
conversion. Existing 222/310 native training checkpoints fail explicitly before
learner/physics allocation. There is no implicit first-layer surgery or optimizer
migration for this observation change.

The paired evaluator supports both contracts: old models explicitly consume
the unchanged 222-feature prefix, and the evaluation record labels that choice.
The native recorder uses the current contract and retains its existing strict
source/package/feature checks; recording a historical checkpoint still requires
its matching source. Generic checkpoint loading retains the serialized sizes.

Recommend `--from-scratch` for a clean causal experiment. This initializes actor,
critic, optimizer, sampling and counters without loading any checkpoint. It also
supports ordinary subsequent strict native resume. Do not resume the live run
into this code. A transfer experiment is also technically reasonable: append
zero columns to actor and critic first layers (before SAPG embedding columns),
retain old weights, use a fresh optimizer, and record the changed contract.
Locomotion is useful knowledge; the missing hand target does not imply it was
never learned. Weight surgery for failed specialist checkpoints is not implemented
or recommended for the primary scratch comparison.

## CPU validation

Result: **65 focused CPU tests passed** (37 native observation/approach/recorder/runner
checks plus 28 JAX-reference task/conversion checks). The expanded full-graph
observation test also passed for a mixed two-world flat/approach batch.

Tests in `tests/test_mjlab_hand_objective.py` cover hand-calculated rotated-route
deltas/extents, the actual immutable v5 scene metadata at a known pose, both
paired alternatives, inactive hands/zones/regions, zero-offset active targets,
approach activation/fade, flat targets, rigid-transform invariance, reward
reconstruction from features, full-graph CPU compilation, sizes/order, conversion
and evaluator compatibility, scratch initialization and early old-checkpoint
rejection. A real CPU MuJoCo task metadata intervention leaves the entire old
actor/critic prefix and qpos unchanged while changing only the new target suffix.

The native test command uses no GPU:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
  .venv-mjlab/bin/python -m pytest -q \
  tests/test_mjlab_hand_objective.py tests/test_mjlab_hand_approach.py \
  tests/test_mjlab_recorder.py tests/test_mjlab_runner.py
```

The JAX-reference task/conversion tests additionally need the existing legacy
Brax/Optax environment. No dependencies were installed; the adjacent legacy
Python was used with the native site-packages appended to locate Torch:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
  ../Click-and-Traverse-WholeBody/.venv/bin/python - <<'PYTEST'
import sys
sys.path.append('/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/.venv-mjlab/lib/python3.12/site-packages')
import pytest
raise SystemExit(pytest.main(['tests/test_mjlab_task.py',
                             'tests/test_mjlab_conversion.py', '-q']))
PYTEST
```

No production reward, termination, sampling or bank geometry was changed by this
patch. Two stale sampler test fixtures were filled with existing required bank
attributes. Broader exploratory checks also encountered unrelated existing
failures in the mocked preflight in `test_flat_balance_overrides` and the old
selection-score expectations in `test_ppo_restart`; the whole repository suite
is not claimed green.

## Launch command — not executed

Run from the repository root, only when the GPU is available. This uses the v5
bank and the requested uncapped policy. It preserves the live run's recorded
0.4 m approach distance, flat region scale 0.4, 768 x 40 rollout grouping, and
32-step unroll. The live saved config actually had `max_action_std=0.15`; the
explicit `--max-action-std 0` below follows the user's requested uncapped setting.
The explicit selection weights retain the live run's flat/retention selector.

```bash
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --from-scratch --algorithm ppo \
  --bank-manifest data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v5_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v5_20260921_resets/manifest.json \
  --run-dir outputs/cat_hand_objective_v3_scratch_30720_20260922 \
  --num-envs 30720 --batch-size 768 --num-minibatches 40 --unroll-length 32 \
  --hand-contrast-region-weight -20 --hand-contrast-heading-weight -5 \
  --hand-contrast-approach-distance 0.4 --hand-raised-reset-fraction 0.8 \
  --flat-bonus-scale 1.0 --flat-region-scale 0.4 \
  --checkpoint-selection-weights 0 0.5 0.25 0.25 \
  --max-action-std 0 --require-hand-contrast --compile-task \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
```

## Expectations and remaining uncertainty

No learning-performance prediction has been validated. With seeded resets,
nonzero instantaneous compliance alone is not evidence: it can be present at
initialization. Look for sustained compliance beyond 100 ms, unseeded protected
compliance, reduced region cost and preserved walking. A rough diagnostic budget
is 20–100 updates for a retention/cost trend, 100–300 updates for a scratch policy
to show sustained unseeded behavior; random initialization also has to relearn
locomotion. These are estimates, not promises. At this rollout size one update
contains 983,040 physical control transitions.

This fixes the missing explicit objective but may not be sufficient. The existing
reward sums weighted terms and clamps the total to [0,10000]. At weight -20,
large region penalties can put different bad poses on the same zero-reward
plateau. Tight boxes, motor-target slew limits, exploration and dynamic balance
also remain. No one of these is established here as the next root cause. If
post-100-ms/unseeded measures remain pinned after roughly 100 updates, inspect
reward clipping and realized arm motion before merely increasing weights.
Heading overrides in other banks still have their existing observation semantics;
this patch exposes hand regions and route orientation, not arbitrary yaw commands.

Extra raw rollout storage for 32 additional actor and critic values, each stored
as current and next float32 observations at 30,720 x 32, is 503,316,480 bytes
(0.469 GiB), before temporary buffers. GPU capacity/compilation/performance were
not tested. The live tmux session, checkpoints and immutable banks were untouched.

## Repository-wide size audit

Searched exact 222/310 values and all substring occurrences. Native allocations,
assertions, learner/config defaults, released expansion/export, recorder and
paired evaluation were updated. Historical JAX v2 is intentionally a separate
contract. Remaining exact Python occurrences are classified below; historical
documentation/assets are preserved, and values such as 2226 scenes, 131072 field
rows, joint limit 1.97222 and drawing coordinates are not observation widths.

Current compatibility rejection and test fixtures:

- `cat_mjlab/runner.py`
- `tests/test_mjlab_hand_objective.py`

Legacy JAX v2 task, policy, conversion fixtures or historical report generators:

- `scripts/validate_cat_diversity_runtime.py`
- `scripts/validate_cat_gpu_capacity.py`
- `scripts/verify_contrastive_training.py`
- `scripts/verify_sapg_training.py`
- `scripts/visualize_collision_proxy_proposal.py`
- `tests/test_cat_body_collision_integration.py`
- `tests/test_cat_contrast_navigation.py`
- `tests/test_cat_host_migration.py`
- `tests/test_cat_room_navigation_integration.py`
- `tests/test_cat_wholebody.py`
- `tests/test_coherent_arm_distribution.py`
- `tests/test_compact_cat_learning.py`
- `tests/test_contrastive_rewards.py`
- `tests/test_fixed_leg_noise_reference.py`
- `tests/test_replay_sapg_update.py`
- `tests/test_sapg_networks.py`
- `tests/test_sapg_numerics_migration.py`
- `tests/test_stabilized_launch.py`
- `tests/test_wholebody_checkpoint_distribution.py`
- `tests/test_wholebody_distribution.py`
- `train_cat_wholebody.py`

Drawing coordinates (not observation sizes):

- `scripts/build_cat_setup_report.py`
- `scripts/render_clutter_rollouts.py`
- `scripts/visualize_compact_cat_points.py`
