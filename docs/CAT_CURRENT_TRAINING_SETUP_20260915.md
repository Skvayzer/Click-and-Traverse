# CAT whole-body traversal: current dep-0 setup

[Illustrated PDF report](CAT_CURRENT_TRAINING_SETUP_20260915.pdf) · [Single W&B run](https://wandb.ai/skvayzer/CAT-wholebody/runs/1d39c55c)

This describes the corrected continuous run, as observed **15 September 2026 at 14:47:33 UTC / 18:47:33 Dubai**. It does not describe the earlier overnight implementation. Source, fields and run state were inspected without changing or interrupting training.

## What is running

One trainable CAT actor controls **29 G1 body joints: 12 leg, 3 waist and 14 arm/wrist**. It starts from the released final CAT generalist, including its critic. CAT's original task, reward and mixed-scene native PPO pipeline are retained, with explicit whole-body/clutter extensions and a smaller batch for the RTX 5090. There is no GRAIL component in this experiment.

The actor is a feed-forward MLP, `406 → 512 → 256 → 128 → 64 → 58`. Its final outputs are 29 Gaussian means and 29 raw scales; actions use `tanh(sample)`, with `sigma = softplus(raw_scale) + 0.001`. The separate training critic is `494 → 1024 → 512 → 256 → 128 → 1`. Hidden activations are Swish/SiLU; observation normalization is disabled. There is no recurrent state or stacked 15-frame input in this execution path.

Actor input consists of 131 proprioceptive/control values, 77 original field features and 198 added hand/arm features. Eleven original body sites each sample three guidance components, three boundary components and one signed distance. Twenty-two added probes each contribute three clearances, a boundary vector, age, unknown flag and uncertainty. The critic uses 208 current/noiseless base values, 88 privileged values and 198 true probe features.

The original checkpoint contract is 162/250 actor/critic inputs and 12 actions. Named mapping preserves retained input and output weights; new input rows initialize to zero, new action means to zero and new standard deviations to 0.05. Mapped retained actor/value output parity error is zero at initialization; changed hands and upper-body actuation mean physical trajectories need not match. **The added upper-body action heads do not contain pretrained arm skills.** All parameters are trainable; Adam initializes once for fine-tuning.

## Control, fields and furniture

Control is 50 Hz; MJX physics is 500 Hz, with ten 2 ms physics steps per policy step. Legs retain CAT's incremental target update, `q_previous + 0.5 * action`. The 17 upper-body actions use `q_nominal + 0.8 * action`, limited to 2 rad/s target change (0.04 rad per control step), followed by soft joint limits. PD torques, torque clipping and CAT disturbances remain.

The robot uses **Unitree Dex3-1 three-finger hands**, retaining native inertias. Seven finger joints per hand are fixed in the configured standing pose; wrists are controlled. The three-finger hardware choice is distinct from the number of finger joints. There are no finger actions in this run.

The actor sees samples from a known static map. Root odometry is held for five control steps; articulation and field queries still update every control step. No camera, depth reconstruction, SLAM or active odometry-noise injection is used here. CAT proprioceptive noise remains.

The bank contains **39 fixed scenes**: 36 byte-verified released scenes, one reconstructed missing original slot, one dense furniture room and one generic-clutter room. Scene IDs and reset state vary during training; room layouts are not regenerated each reset. The missing `D8G2L3O2S13` was reconstructed from the original generator configuration because its public bytes are unavailable.

Both new rooms measure 9.00 × 9.69 × 2.4 m, with six approximately 68–78 cm bottlenecks. The furniture room has 9 tables, 36 chairs and 298 primitive boxes representing tops, seats, backs, legs and other obstructions. Generic clutter uses 45 objects / 91 boxes: crates, low blocks, partitions and shelves. The 39.85 m geometric reference route is a construction check for root clearance, not an actor input or learned trajectory. It does not prove dynamic whole-body traversability.

Occupancy at 4 cm resolution feeds original CAT signed-distance, boundary-gradient and progressive 3D fast-marching guidance generation. New grids are `230 × 248 × 60`; original fields retain their native arrays and interpolation conventions. Ragged storage contains about 324 MiB of field-array files. **Training retains CAT's flat-floor/feet contact physics: furniture is represented by fields, not physical obstacle-contact forces.** Physical furniture evaluation is a separate tool.

## Hand protection and limitations

Each hand has an envelope derived from all mesh vertices, including the thumb, with 5 mm padding. Eight corner probes plus a 3 cm palm sphere give 18 hand probes across both hands. Four additional 5 cm spheres cover forearms/elbows. Mesh containment is checked for the configured fixed finger pose.

For each hand probe, `d = min(d_now, d_0.2s, d_0.4s)`. Future positions extrapolate current velocity; they do not simulate future actor commands. The hand cost is:

```text
deficit = max(0.12 - d, 0)
closing_speed = max(-velocity · boundary_direction, 0)
urgency = clip(closing_speed / max(d_now, 0.02), 0, 10)
hand_cost = mean_over_18_probes(deficit² * (1 + urgency))
arm_cost = mean_over_4_probes(max(0.08 - d_now, 0)²)
```

Scales are −5 for hand cost and −2 for arm cost. They augment the full 22-term CAT reward before its 0.02 s scaling and nonnegative clipping. CAT's outer sampling wrapper zeros learner rewards at terminal transitions.

The policy can learn to raise, tuck, slow down or reposition arms if that improves return. There is no explicit reward for human-like raising and **no human motion dataset, imitation loss or motion prior in this run**. Point probes are not a swept-volume guarantee; the box is not a hard action filter. Scene-aligned human demonstrations remain a proposed follow-up, not an implemented component.

## PPO, resets and one unified run

| Setting | Active run |
|---|---:|
| Parallel environments | 8,192 |
| Unroll length / chunks | 32 / 4 |
| Transitions per update | 1,048,576 |
| Trajectories / transitions per minibatch | 512 / 16,384 |
| Minibatches / optimization passes | 64 / 4 |
| Adam steps per update | 256 |
| Learning rate / entropy coefficient | 0.0003 / 0.003 |
| Discount / GAE lambda | 0.98 / 0.95 |
| PPO clip / max gradient norm | 0.2 / 1 |
| Advantage normalization / observation normalization | enabled / disabled |

Released CAT uses 65,536 environments and 2,048 trajectories per minibatch, for 4,194,304 transitions per update. The smaller batch is an explicit resource change. The active settings override the more conservative CLI defaults. The final generalist already completed distillation, so this is direct PPO fine-tuning with no teacher/DAgger stage.

Original scenes retain CAT resets and 1,000-step / 20 s episodes. New rooms reset near their entrance with ±8 cm XY jitter and CAT's yaw, joint and velocity randomization; their time limit is 4,000 steps / 80 s. Falls and invalid states terminate immediately. Selected self-contact, original body-field penetration and added probe penetration use the original **0.0 m threshold and 50-step collision grace**. Timeouts are truncations, with CAT's native Brax GAE truncation handling retained: timeout deltas/advantages are masked, rather than directly bootstrapping from a final pre-reset observation. A collision at the time limit remains a termination.

Reaching the goal does not itself end the episode. The new rooms use their actual goal in reward logic; the original scenes retain CAT's crossing condition. The adaptive sampler uses per-scene timeout/completion EMAs with decay 0.95, and normalized weights `max(1 - timeout_fraction, 0.001)`. This measures timeout survival, not goal-reaching success.

The same learner, optimizer, RNG, environment and sampler persist across updates. There are no scene stages, optimizer resets or best-model rollbacks. Aggregate metrics and per-scene diagnostics share W&B run **`1d39c55c`** and a single global-step axis. Training has no finite budget or automatic evaluation.

One native best checkpoint is selected by a maximum single-update aggregate training reward proxy, not held-out furniture success. The score comes from the rollout before the saved post-optimization parameters. The live learner never rolls back to this model. One overwritten `resume.msgpack` (about 1.03 GiB) holds complete current recovery state; the native best checkpoint is about 6 MB. Recovery state is saved atomically before training and after each update. Exact resume rejects changed source, field or configuration identities.

## Measured state and what it means

The snapshot contains **1,086 complete updates / 1,138,753,536 logged transitions** and 1,555,703 completed episodes. PID 11525 remains on attempt 1 with zero logged nonfinite events. The status file was read one update before the last metric: 1,137,704,960 saved versus 1,138,753,536 observed. That is expected collection timing, not a restart.

| Metric | Updates 21–70 | Updates 1037–1086 |
|---|---:|---:|
| Mean reward per transition | 0.3322 | 0.3270 |
| Logged episode duration | 12.30 s | 14.85 s |
| All-scene early termination | 44.30% | 39.10% |
| Original CAT slots | 37.54% | 35.10% |
| Furniture | 99.96% | 97.42% |
| Generic clutter | 99.51% | 53.54% |

Generic clutter survival has improved substantially. Furniture still fails early in almost all completed episodes. The hand proximity penalty magnitude per logged episode step is 26.38% higher, which is worse. Longer episodes and lower critic loss do not establish safer hands or goal completion. The early comparison is already fine-tuned, not a matched pretrained baseline. The scene mix adapts throughout training.

Episode-weighted failure rates were reconstructed from the 1,000-update rolling logs, with independent integer-count, cumulative-count and scene-partition checks. Duration/reward summaries remain the logged trailing-1,000-completed-episode statistics, averaged over each comparison window. See [evidence and reproduction](assets/setup-report-20260915/EVIDENCE.md) and [summary JSON](assets/setup-report-20260915/progress-summary.json).

NVIDIA reports 30,425 MiB / 29.71 GiB allocated on the 32,607 MiB RTX 5090. JAX peak live allocation is 22.64 GiB, within its 28.26 GiB allocator limit. These overlap and must not be added together. Execution samples reach 99–100% utilization, with gaps for host checkpoint writes; the snapshot itself caught 0% at a boundary. Recent epoch throughput is about 87,600 transitions/s, excluding checkpoint writes.

## Exact source, setup and operator paths

| Component | Identity |
|---|---|
| Fork / branch | `Skvayzer/Click-and-Traverse` / `feature/whole-body-furniture-traversal` |
| Frozen training commit | `40b4dd2682dd4e5d75a6b853c24333da99379cf3` |
| Python training source SHA256 | `14292904cb7065229cbba32f90ca43ff1084260e5053239f28f54e7d7617ef2d` |
| Original CAT commit | `866ba392f1c1e84b92ad75fa66550f26e8af8e48` |
| Released configuration SHA256 | `0e38cac262a3c1c95bacdf859193340056fef53b8d2f8414293789cee5d45d2d` |
| Pretrained model revision | `46ce4b57ba0639168d51741b661ff62f7ce6f045` |
| Native checkpoint | `logs_v1/generalist_v1/checkpoints/005033164800` |
| Public field dataset revision | `db8c202fa724bc4d3dba2cb9fead1267f15fd811` |
| Manifest file SHA256 | `bbcccb9a11ebee99ad14f5330de71a6a9461aed0e5c2115774b8c3b4ce19c633` |
| Manifest canonical-content SHA256 | `cb49fb194e12ce7861d1fc90778d8acb176e5143714efa168afbd096840d48e5` |

The two manifest hashes describe different serializations of the same bank; they are labeled separately to avoid treating them as conflicting identities.

Software: Python 3.12.9, JAX 0.4.38, Brax 0.12.3, Flax 0.10.4, MuJoCo/MJX 3.3.1, Playground 0.0.4, Optax 0.2.5, Orbax 0.11.5 and W&B 0.24.2. The CUDA environment lock is [requirements/locks/linux-cuda12-py312.txt](../requirements/locks/linux-cuda12-py312.txt). The installed environment occupies about 7.2 GB. Source checkout and field/model preparation are described in [WHOLE_BODY.md](../WHOLE_BODY.md).

Remote root: `/home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody` on `konstantin.smirnov@dep-0`. The running command uses the frozen source and its shared `.venv`:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
outputs/sources/cat_wholebody_generalist_20260915/.venv/bin/python -u \
  outputs/sources/cat_wholebody_generalist_20260915/train_cat_wholebody.py run \
  --bank-manifest data/furniture/cat_generalist/manifest.json \
  --run-dir outputs/cat_wholebody_generalist_20260915 \
  --profile single_gpu_32gb --num-envs 8192 --batch-size 512 --seed 0 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
```

This records the current launch; it is not a request to start a duplicate process. Start time is 10:37:15 UTC, 15 September. Console: `output_logs/cat_wholebody_generalist_20260915.log`. Full launch metadata: `output_logs/cat_wholebody_generalist_20260915.launch.json`. Recovery and best paths are under the run directory. [Cooperative stop and exact resume](CORRECTED_RUN_20260915.md#operator-paths-and-stopping).

## Validation and report reproduction

Earlier validation passed 243 full regression checks and 72 later focused recovery checks. It includes original CAT reset/step comparison, warm-start parity, original field hashes and interpolation, adaptive sampler parity, mixed-scene reset/step/fall/timeout integration, plus actual GPU compilation and training at the active profile. These checks validate implementation and integration. Controlled goal/hand-clearance evaluation and physical-furniture validation remain outstanding.

The report uses actual stored room geometry, field arrays, native Dex3 STL/XML and measured training metrics. The diagrams do not present an invented learned trajectory. Builders and lightweight evidence live in [assets/setup-report-20260915](assets/setup-report-20260915/).

With `reportlab`, `pymupdf`, `matplotlib` and `numpy` available in a separate reporting environment:

```bash
python scripts/build_cat_setup_report.py
```

This regenerates the ten-page PDF from the supplied plot PDFs and JSON, validates required claims and text bounds, and writes page previews. It never contacts dep-0 or changes a training run.
