# Reactive avoidance: perception passed; training admission blocked

This is **incomplete enabling work**, not the requested training bank. No training,
checkpoint loading, bank generation, or reward changes occurred. Existing banks
and checkpoints were not written. The owner-required no-contact guarantee is
not certified. There is no valid launch command for this incomplete environment.

The task explicitly requires stopping rather than building on an unaffordable
analytic path. Affordability at the requested GPU scale cannot be established
in this execution environment: `nvidia-smi` fails to communicate with the driver,
there are no `/dev/nvidia*` devices, and Torch 2.7.1+cu118 reports
`torch.cuda.is_available() == False`. The CPU eager implementation is expensive.
This is not evidence that a fused implementation on the owner's GPU is impossible.

## Implemented and verified

- [analytic_objects.py:13](../cat_mjlab/analytic_objects.py#L13): exact signed
  point-to-sphere, rotated-box, and capped-cylinder distances and normals. The
  cylinder also represents an 8 mm radius rod without a grid.
- [analytic_objects.py:58](../cat_mjlab/analytic_objects.py#L58): minimum static
  versus analytic surface clearance, nearest normal, tangential projection of
  guidance, static tie preference, and invalid-object masking.
- [task.py:270](../cat_mjlab/task.py#L270): opt-in analytic hand/body field merge
  after static command construction; hand radius subtracted exactly once.
- [task.py:292](../cat_mjlab/task.py#L292): elbow sphere merge reaches the existing
  elbow observations and arm-clearance reward input. **Whole forearm capsule
  queries are not implemented.** This must not be called complete arm coverage.
- [analytic_objects.py:102](../cat_mjlab/analytic_objects.py#L102): approach,
  hold, then retreat, with no endpoint overshoot. Per-environment geometry is
  saved/restored with task state; time comes from simulation state. No hidden
  object clock or random stream. This does not perform scene/reset admission.

Dynamic perception uses actual world query positions and the simulation clock
for both actor and critic, independently of held odometry used by the old static
field. There is no added dynamic-sensor latency. It retains the task's existing
pre-final-integration derived-pose convention. A substep safety implementation
still needs a separate, current-pose/full-trajectory contract. Normalization and
the standing guidance move gate are unchanged, so standing guidance remains
zero; distance and obstacle normal carry the approaching stimulus.

The [demonstration and measurement script](../scripts/verify_analytic_objects.py)
uses a real `CATTask` and MuJoCo-derived G1 pose, an unchanged static `bank.sample`,
and the actual `_fields -> _observe -> 222-element actor` path. It takes **zero
physics steps**, loads no policy, and moves only the analytic obstacle/time.

The actor's `pf.hands.df.0.distance`, index **178**, reads **0.350, 0.250, 0.150,
0.050, 0.050, 0.150, 0.350 m** at **0, 1, 2, 3, 4, 5, 7 s**. Maximum robot qpos
change is **0**. The actual hand-envelope radius is **0.10344617 m**. Normal is
`[-1, 0, 0]` for this front approach. Raw evidence is in
[perception-and-cost.json:2](assets/reactive-analytic-20260922/perception-and-cost.json#L2).

## Measured costs, not GPU estimates

30,720 environments, two CPU threads, Torch eager, 3 warmups and 10 timed
repetitions. One measured unit is four hybrid query passes: 11 body points twice
and 2 elbows twice. It includes object-center evaluation, distances, normals,
selection, surface subtraction and guidance projection. It excludes static grid
lookup, physics, full forearm capsules, continuous collision checks and any
runtime safety guard. Timing is on a shared host, not an isolated benchmark host.

| Objects per environment | Median ms / query tick | Persistent state MiB | Peak temporary tensors MiB |
|---|---:|---:|---:|
| 1 | 555.49 | 2.61 | 80.68 |
| 2 | 1205.21 | 5.21 | 160.02 |
| 4 | 2190.48 | 10.43 | 311.02 |
| 8 | 4555.38 | 20.86 | 613.01 |

Persistent bytes are calculated from actual allocated tensor storage sizes.
Temporary allocation high-water marks are measured using Torch profiler memory
events, tracking allocations and frees. They are not total RSS or VRAM claims.
The eager implementation computes all shape branches before selection. Fusion,
shape grouping, querying only the active scene subset, and avoiding duplicate
actor/critic dynamic queries remain optimization opportunities.

**Maximum affordable GPU object count: unknown, not measured.** No positive
object count in this CPU implementation meets a 20 ms wall-clock query budget.
That illustrative budget is one simulated control period, not a claim that
training must run in real time. The measurements alone do not establish an
acceptable training slowdown. No GPU speedup factor is inferred.

## Tests and admission status

The first test run reported **22 passed, 5 failed, 2 errors**. All seven unsuccessful
tests were blocked at imports of absent legacy-reference dependencies:
`ml_collections`, `jaxlie`, and `mujoco_playground`. No dependency installation was
attempted on the tight shared pool. The eight new test cases passed, covering
analytic normals, rotated boxes, thin rods, guidance, masks/ties, trajectory
clamping, real observation response, and exact absent-versus-masked task parity
over four transitions including reset. Native runtime and replay tests also passed.
This is not a complete flat/clutter/CAT/narrow-bank regression certification.
After the clock/replay checks, the dependency-available suite was rerun:
**22 passed, 7 deselected**, in **15.99 s**.

Admission is **BLOCKED**:

- perception demonstration: PASS;
- measured GPU affordability: BLOCKED (GPU inaccessible);
- whole forearm capsule path: NOT IMPLEMENTED;
- continuous full-body no-contact certificate: NOT IMPLEMENTED;
- standing/walking/turning scene bank and 75% retention mix: NOT BUILT;
- bank preflight: NOT RUN / NOT PASSED;
- corrected below and bucket renders: NOT BUILT;
- launch command: NOT ISSUED, because no admitted bank exists.

A static start-pose certificate cannot establish "no contact in any episode"
under arbitrary policy actions: the robot can move into the obstacle. A runtime
guard or an explicit bound on reachable robot motion is necessary. A question
about guard semantics was sent; it has not been answered. Merely disabling
MuJoCo contacts or terminating on overlap would not satisfy the requirement.

## Disk

Initial filesystem availability: **5,205,131,264 bytes (4.848 GiB)**. Before
demonstration/benchmark: **5,202,116,608 bytes (4.845 GiB)**. After benchmark:
**5,202,903,040 bytes (4.846 GiB)**. Shared-pool fluctuations are not attributed
to this task. The script checks a **1 GiB free-space stop threshold**, writes
one small JSON, and creates no grids, checkpoints, or profiler trace files.

Reproduce the isolated gate, without training:

```bash
.venv-mjlab/bin/python scripts/verify_analytic_objects.py
```

GPU exposure is needed to measure and optimize the intended deployment path.
Only after that gate is accepted should standing scenes be built, followed by
full-body continuous certification, guarded runtime validation, bank preflight,
and renders. Keep the named warm-start checkpoint untouched until a runnable,
certified environment exists.
