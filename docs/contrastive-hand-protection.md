# Contrastive hand-protection specialist

**Current native contract:** actor **222** / critic **310**. Authored box features and rewards are removed; see [removal and from-scratch launch](NATIVE_CLEARANCE_REMOVAL_20260922.md).

A [table-height extension](table-edge-hand-protection.md) adds 32 matched table scenes to this 64-scene cabinet bank, preserving the original scenes and training objective weights.

The previous specialist could reach goals by turning sideways or lowering its body without moving its arms appreciably. Its clearance penalties discourage contact, but do not prefer forward-facing travel or a particular safe hand posture. The neutral-arm penalty could also return immediately after raised hands became clear.

This change adds explicit route-conditioned posture objectives and matched geometry. It is opt-in for a new specialist bank. The existing running experiment is not modified or restarted by this implementation.

## Scene design

Sixteen seed groups contain four matched variants each, for 64 training scenes. Group members share cabinet height, orientation, placement jitter, and route; passage widths change. Each room has three obstacle modules and open turn bays between them.

| Role | Passage width | Intended solution |
| --- | --- | --- |
| Open | 1.50–1.52 m | Walk forward with ordinary arms |
| Forward protected | 0.729–0.741 m | Walk forward with raised or tucked hands |
| Narrow | 0.400–0.410 m | Turn sideways; no forward-heading penalty |
| Transition | protected / narrow / protected | Change arm posture and heading as geometry changes |

Cabinets are 1.24–1.28 m high, preventing a simple crouch underneath. Crosswalls close the outer bypass lanes. The original table/chair banks remain available; this bank deliberately isolates the posture learning problem and contains no original-environment or ordinary-clutter population. Its goal metric must not be compared directly with the mixed generalist's success rate.

Each seed stays within a single dataset split. The builder also supports separate validation/test banks with different seeds. Episode resets reserve 25% probability for each role; failure-based adaptation changes relative scene weights only within a role. All roles are present immediately, without the old easy/medium/hard unlocking curriculum.

## Rewards and observations

The actor still receives **222 values**, the critic **310**, and the policy outputs **29 joint actions**. Scene role, hand target regions and route-phase labels are reward metadata; they are not additional policy observations. Existing signed-distance observations, native rewards and approved 35 body collision primitives remain active.

In a forward-permitted zone, with route tangent t and projected pelvis/torso forward axes f:

`c_heading = phase * mean(1 - dot(normalize(f), t))`

There is no heading cost in a narrow zone. Both pelvis and torso must face the route, preventing a torso-only twist from satisfying the objective.

For each allowed paired left/right hand region, compute the distance from the hand-sphere center to its axis-aligned target box in route coordinates. The hand cost is:

`c_hand = phase * min_over_allowed_pairs(mean_over_active_hands(distance^2 / (distance^2 + (0.15 m)^2)))`

The smooth bounded cost continues to improve as a distant hand moves toward its target; it does not saturate at a hard distance cutoff. The two region alternatives correspond to raised and tucked postures. A pair is chosen jointly for both hands. Reaching a region gives zero cost; moving farther inside does not earn additional reward. XY coordinates are relative to the current root in the route tangent/normal frame. Z is absolute world height. Turning or crouching therefore cannot rotate or lower the targets with the robot.

Both costs have scale **−1 before multiplication by the existing 0.02 s control interval**. Native hand signed-distance cost remains unchanged and is included in the static posture reward-budget checks. At the three protected module centers of seed 20260919, the sum of native body-keypoint SDF costs, added hand/elbow pressure, arm-neutral cost and new objectives is about −0.24 to −0.27 for raised forward poses versus −1.79 to −1.80 for nominal sideways poses before the time-step factor. Tucking is permitted but can retain a larger native proximity cost. These static comparisons do not predict dynamic return; the weights are initial settings, not a guarantee of successful learning.

Neutral-arm regularization is disabled throughout the hand-protection approach, hazard and departure zone, including after the hands become clear. Waist stabilization, arm motion costs and action limits remain. The posture costs fade at zone boundaries; unshaped turn bays allow rotation before entering narrow modules.

## Navigation, collision and success

The original 0.23 m navigation cylinder remains unchanged for legacy scenes. Certified narrow/transition scenes use a 0.15 m navigation-only radius. Loading them requires a matching enabled body-collision bank and certified clear reset pool; the smaller navigation cylinder does not replace robot-body collision checks.

The generator checks all 35 approved proxies against canonical geometry along the route, arm transitions and turns, plus sampled floor-referenced crouching and stepping configurations. Hand target regions are checked against analytic geometry and the actual conservative 4 cm SDF. This is sampled kinematic validation, not a dynamic traversability proof, exhaustive posture search, or proof that every inverse-kinematic arm solution is safe.

Training keeps the clean-goal first-outcome metric and adds only three qualified rates:

- `success/forward_protected_success_rate`
- `success/narrow_passage_success_rate`
- `success/posture_transition_success_rate`

Forward-protected and transition successes require a clean goal, visiting every required zone, and correct heading/hand posture for at least 90% of each zone's fully active steps. Heading tolerance is 15 degrees for both torso and pelvis. Each active hand must be within 5 cm of one common allowed paired region. Narrow success requires a clean goal and permits sideways walking. Denominators include failures and timeouts; an episode is resolved once at its first clean goal or failure. A learner update with no resolved episodes omits its rate instead of plotting zero. Counts accompany rates, and no evaluation rollouts are added.

A dedicated four-chart W&B saved view can be created after starting a new run with `scripts/configure_training_success_workspace.py --contrastive --saved-only --run-id RUN_ID --backup BACKUP.json --apply`. It shows overall clean goals and the three rates above.

## Reproduce and launch

From the repository with its existing environment installed:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/build_contrastive_hand_bank.py \
  --output data/furniture/contrastive_hand_v2 --seed 20260919 --groups 16
.venv/bin/python scripts/build_body_collision_bank.py \
  --field-manifest data/furniture/contrastive_hand_v2/manifest.json \
  --output data/furniture/contrastive_hand_v2_collision
JAX_PLATFORMS=cpu .venv/bin/python scripts/build_body_collision_resets.py \
  --field-manifest data/furniture/contrastive_hand_v2/manifest.json \
  --collision-bank data/furniture/contrastive_hand_v2_collision/manifest.json \
  --output data/furniture/contrastive_hand_v2_resets --seed 20260919
```

The following is a **new run**, not an exact resume of an experiment using another scene bank/reward. It uses the released CAT checkpoint initialization and existing SAPG implementation, without evaluation, rollback or retention. PPO remains available by changing `--algorithm`. The environment count below matches the previous SAPG run; GPU memory must be measured for the larger bank before adopting that count on a server.

```bash
.venv/bin/python train_cat_wholebody.py run \
  --algorithm sapg --finetuning cat_train_only \
  --num-envs 36864 --batch-size 576 \
  --bank-manifest data/furniture/contrastive_hand_v2/manifest.json \
  --body-collision-bank data/furniture/contrastive_hand_v2_collision/manifest.json \
  --body-collision-resets data/furniture/contrastive_hand_v2_resets/manifest.json \
  --run-dir outputs/cat_contrastive_hand_sapg_v2 --wandb-mode online
```

LR, PPO clipping, entropy/SAPG configuration, checkpoint migration and exploration behavior are unchanged by this patch. The launcher automatically enables the new objective only when the explicit contrastive bank marker is present. Ordinary generalist manifests still require their complete original scene collection.

## Visual examples

[Four scene roles](assets/contrastive-passages-20260919/contrastive-scenes.png) and [nominal versus protected hand poses](assets/contrastive-passages-20260919/protected-postures.png) are actual MuJoCo renders with the G1 Dex3 grippers. They show illustrative prescribed static poses, **not newly learned behavior**.

## Local validation

The final sealed bank contains 64 scenes, 1.641 GiB of field arrays, and 2,048 verified clear reset poses. The worst sampled body separation is 3.33 cm and hand SDF surface clearance is 3.47 cm across the static route/transition/stance checks. Open-passage hand field clearance exceeds 0.30 m.

The complete bank was loaded through the training wrapper with explicit field arguments. Actual CPU JIT reset and step with two environments produced finite observations and rewards and preserved the 222/310/29 contract. This checks implementation compatibility, not policy performance or GPU capacity. See [runtime verification](assets/contrastive-passages-20260919/runtime-verification.json) and [static reward budget](assets/contrastive-passages-20260919/static-reward-budget.json).

The final targeted regression suite passed 154 tests covering scene certificates, rewards, navigation, sampling, outcome accounting, logging, launch configuration and legacy specialist behavior. The earlier physical-collision integration suite also passed.
