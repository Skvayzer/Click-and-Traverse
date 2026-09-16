# Full-body collision and corrected navigation training

This change enables the approved **35 body collision shapes** in the MJX
training task: 11 boxes, 22 capsules and two enclosing hand spheres. Feet use
separate flat sole and upper-foot boxes; articulated ankles have their own
shapes. These are simulator checks, with **no added observations**: actor 222,
critic 310, actions 29.

## Collision and learning behavior

Each shape is transformed by its owning robot body's current pose. Spheres and
capsules use exact volume intersection with obstacle boxes; robot boxes use all
15 separating axes against obstacle boxes. A conservative spatial index selects
candidates, with no dropped or truncated neighbors. Narrow-phase candidates are
processed eight at a time to bound temporary GPU memory.

Checks run at every 2 ms physics pose, plus the final integrated pose of each
20 ms control interval. Any collision during the interval is latched. It ends
the episode without the original 50-control-step grace period, denies goal
credit and subtracts **1.0 reward once**, after native reward clipping and without
timestep multiplication. The autoreset wrapper preserves that terminal reward
in the actual PPO transition. A collision at the episode horizon is a true
termination, not a timeout/truncation.

The floor and CAT's existing self-contact physics remain unchanged. These
obstacle checks do not add contact impulses or alter the PD controller.
Original field observations and clearance rewards continue to provide dense
guidance; full-body overlap additionally supplies training failure and reward.
Checks are at discrete physics poses, **not continuous collision detection**.
The enclosing primitives are conservative approximations of the robot mesh;
they can reject motion through empty space inside an enclosing shape.

## Scene geometry and navigation

The corrected field bank includes all **2,338 scenes**: 37 original configuration
slots (36 verified released arrays and one explicitly reconstructed missing
slot), 27 additional published scenes, 2,226 procedural CAT scenes, 24 furniture
rooms and 24 generic clutter rooms. This is the available scene collection,
not a reproduction of the paper's unreleased indoor scans. The existing reset sampling masses remain
20% released/published CAT, 40% procedural CAT, 25% furniture and 15% generic
clutter, with adaptation within groups.

All 48 rooms use their actual canonical oriented obstacle boxes, including
thin, rotated chair legs. Original/procedural CAT geometry uses a lossless union
of occupied 4 cm voxel cells, merged into cuboids. Occupancy is checked against
the verified training SDF sign. Cells are centered at CAT's runtime/render
coordinates `origin + index * dx`, retaining its historical half-cell difference
from the generator convention. This occupied-cell solid can be more conservative
at edges than the smoothed marching-cubes visualization.

The shared collision bank contains **187,825 obstacle boxes**, 2,535,238 spatial
cells and 9,458,866 candidate entries. Its measured maximum candidate list is 64;
shared arrays total **69,471,104 bytes (66.25 MiB)**. This is the static bank size,
not the full collision computation's peak memory.

The room navigation corrections in `d7423b9` are included: ordered route
waypoints, visibility checks, conservative obstacle rasterization, a swept
root-cylinder guard and fresh room/reset history. CAT scenes retain their
original guidance fields. Full task state and history now reset coherently on
CAT-to-CAT transitions as well when full-body collision training is enabled.

Native randomized reset poses are retained when collision-free. An initially
overlapping pose is replaced by a random member of a separately validated pool
of 32 clear poses for the **same scene**, drawn from the same reset distribution.
Every scene remains included. Pool generation and runtime use the same geometry
and intersections; all stored poses are rechecked before publication.

The published pool has **74,816 verified poses**. All 48 room scenes accepted
every sampled candidate. Candidate rejection was 35.27% for configured original
scenes, 18.19% for additional published scenes and 39.27% for procedural scenes.
These are offline reset-geometry measurements, not rollout success rates. Remote
mesh coverage was also checked against the actual compiled robot: all 49 unique
mesh placements are enclosed, with at least the approved 3 mm body margin.

## Fine-tuning and monitoring

The restart uses the **original released CAT generalist checkpoint and fresh
Adam state**. None of the previously fine-tuned weights are reused. The existing
stabilized settings remain: learning rate 3e-5, PPO clip 0.1, soft reference KL
coefficient 0.05 on original leg actions in CAT scenes, bounded upper-action
standard deviation 0.02–0.10 initialized at 0.05, and physical upper-target
regularization. The shared policy remains trainable.

One online W&B run aggregates all scene types. Fixed-scene deterministic and
stochastic validation uses the **same new collision checks** at baseline and
during training. This monitor uses 16 fixed scenes from the training bank,
16 seeds each, in both action modes (512 episodes); it is not held-out
generalization evaluation. Therefore success/reward values from older collision rules are
not directly comparable. Main metrics:

- `validation/cat_goal_success_rate` versus this run's baseline: retained CAT skills.
- `validation/clutter_goal_success_rate`: new traversal skill.
- `validation/body_collision_rate` and `training/body_collision_rate`: full-body overlaps.
- `training/body_collision_{feet,legs,trunk,head,arms,hands}_rate`: affected regions.
- `training/reset_pose_replacement_rate`: fraction of completed episodes whose initial random pose required replacement.
- `training/fall_rate`, `training/upper_std_max`, and `health/nonfinite`: stability.

Retain one best eligible model and one atomically overwritten full resume
snapshot. Production training has no configured step limit. Restart identity,
capacity measurements and initial runtime verification are recorded below.

## Verification and restart

**123 relevant tests passed across focused runs**, covering primitive volume
geometry, conservative broadphase, terminal reward delivery through the actual
PPO/reset stack, an intermediate-substep collision with a clear final pose,
collision versus timeout precedence, compact observations and reset history.
The chunked candidate tests include collisions only in the final candidate of
9- and 17-entry lists and masked/padded tails. Existing CAT compatibility and
room navigation regressions passed.

The [complete PPO capacity check](assets/cat-body-collision-20260916/capacity-16384.json)
passed two updates at **16,384 environments**, batch size 256 and 524,288
transitions per update. Observed peak device occupancy was **37,989 MiB**, with
**10,644 MiB free**. JAX peak live allocation was 29,246,582,272 bytes. One warm
update took 21.97 s, or **23,865 transitions/s**, including host callback overhead.
Production checkpoint I/O and fixed-scene validation add elapsed time. This
checks capacity and finite execution, not convergence or long-run stability.

The capacity check used the isolated working source recorded in its report.
The committed production Python source differs only in the whole-body module's
descriptive docstring; executable code is the same. Production uses frozen
commit `bb141fd1cede3f043fad05db514909febe0c680f`.

The [continuous launch](assets/cat-body-collision-20260916/launch.json) started on
**ws008090 at 2026-09-16 16:17:11 UTC**, PID **267506**, with a fresh optimizer and
the original released CAT actor/critic. Warm-start parity errors are zero for
the mapped original actor outputs and critic. Single W&B run:
[cc931c02](https://wandb.ai/skvayzer/CAT-wholebody/runs/cc931c02).

Run directory: `outputs/cat_collision_navigation_20260916`. Stop deliberately
with `touch outputs/cat_collision_navigation_20260916/STOP` from the remote
repository. The previous stopped checkpoints are preserved.

The [original-checkpoint baseline](assets/cat-body-collision-20260916/validation-baseline.json)
completed before learning. Deterministic goal success under the new rules was
32.81% on the selected CAT scenes and 67.19% on the selected clutter scenes.
These small training-bank subsets use their own start/goal distribution and
must not be interpreted as paper reproduction or generalization results.
They are the within-run retention reference, not fine-tuning results.

[Production startup verification](assets/cat-body-collision-20260916/startup-verification.json)
at **16:30:43 UTC** confirmed **three PPO updates / 1,572,864 transitions**, a
full resume snapshot at the same step, exactly one learner and the same step
visible in online W&B. All logged numbers were finite; `health/nonfinite=0`.
Upper-action standard deviation remained 0.0489–0.0517, within its configured
bounds. A warm update reported 23,668 transitions/s. The GPU then reported
38,005 MiB occupied, 10,628 MiB free and 100% utilization. Training remains
continuous with no stop marker. This verifies successful startup and logging;
it does not yet establish improvement or preservation of original skills.
