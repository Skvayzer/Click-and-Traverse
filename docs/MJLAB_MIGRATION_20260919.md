# Whole-body CAT migration to mjlab — 19 September 2026

## Scope and preservation

The new `cat_mjlab` backend uses **mjlab 1.6.0 / MuJoCo Warp 3.11.0** for GPU physics and native PyTorch for the task, PPO and SAPG. It does not wrap a JAX task or substitute mjlab's example humanoid task. Original JAX/MJX entry points remain intact.

The dep-1 training process stopped cooperatively at **660,602,880 physical environment steps**. Its full 4,712,175,013-byte runtime checkpoint is preserved using a hard link outside the original run directory:

```
/home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody/outputs/mjlab_migration_20260919/preserved_sapg/resume.msgpack
SHA256: 728d793c5b3bec063efb692e48e66280c0a00c5de69a8ac4a5727061d8e1001b
```

The old run's `STOP` marker remains. Migration verification does not start an overnight run, create a W&B run, or evaluate policy performance.

## What is preserved

| Component | Ported behavior |
| --- | --- |
| Robot | Same G1 MJCF, fixed Dex3 grippers, 29 actuators and joint ordering |
| Observations | 222 actor and 310 critic values, identical feature order, noise, held odometry and delayed fields |
| Actions | Same incremental leg targets; upper-body nominal-relative targets, scaling, rate limits and joint limits |
| Timing | 2 ms physics, 20 ms control; original pre-final-integration observation timing |
| Scene representation | Shared SDF/BF/GF arrays and original interpolation convention; certified ordered routes |
| Collision | Same 35 approved spheres/capsules/boxes, including box feet; every physics substep plus final pose; penalty and termination; no extra policy inputs |
| Reward | Native terms, clipping and time scaling; existing upper-body stabilization and hand-clearance terms |
| Resets | Certified poses, clean history, per-scene horizons, original initial horizon offsets and native contact grace |
| Learners | Both PPO and SAPG; original loss, GAE and distribution conventions; SAPG overflow fix |
| Logging | One W&B run per native run, separately named `success/*` training outcomes, no periodic evaluation or retention |
| Checkpoints | One best policy and one atomically overwritten complete native resume |

Floor contact still uses the physics engine. Clutter contact still uses the approved volume-intersection penalty/termination system, rather than introducing new furniture contact dynamics as part of the backend change.

The source SAPG run uses six policies with a 16-dimensional learned policy embedding; learning rate `3e-4`, PPO clip `0.2`, entropy coefficient `0.003`, discount `0.98`, GAE lambda `0.95`, gradient norm limit `1`, 64 minibatches and four optimization passes. The embedding receives both actor and critic gradients. The log-domain follower-to-leader importance calculation preserves the recent overflow fix.

## Checkpoint conversion and restart semantics

`scripts/export_mjlab_checkpoint.py` runs in the old environment and produces a small NumPy archive. `cat_mjlab/conversion.py` loads it without Flax or JAX. It transposes dense kernels, preserves the actor, critic, all six policy embeddings, and Adam first/second moments and update count. A released original CAT checkpoint is also supported through the existing explicit 162/250/12 to 222/310/29 adapter.

An MJX physics state cannot be restored as a Warp solver state. The first Warp run therefore starts fresh episodes with the converted learned state. This is a **backend migration**, not a claim of identical trajectories or uninterrupted RNG streams. A separate W&B identity records its parent checkpoint. Subsequent `--resume` restores the complete native Torch/Warp task, physics, optimizer, random generators and success window, with strict setup checks; GPU physics is not promised to be bitwise deterministic.

For the exact same field-bank hash, the archive also preserves the simulator-independent sampling state: **curriculum stage 2**, unlocked levels, adaptive scene EMA and probabilities, and curriculum counters. Completed counts were `[187947, 60308, 53269]`, with clean goals `[123170, 42316, 29335]`. Fresh episodes are sampled from that preserved distribution. Different banks explicitly start their own sampler; scene IDs are never mapped implicitly. The new training success window starts fresh rather than mixing outcomes across physics backends.

The old specialist was training on **24 scenes** in `cat_hand_specialist_v1_20260918`. The newer **96-scene contrastive bank** is a separate selectable bank: 64 contrastive passages plus 32 table-height variants. Selecting it also enables its existing heading/hand-region rewards. It must not be described as the identical scene distribution of the stopped run. Generalist procedural/clutter banks remain selectable too.

## dep-1 installation

The new checkout is `/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab`, on `feature/mjlab-migration`. The isolated environment is `/home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody/.venv-mjlab`; the previous `.venv` is unchanged.

The RTX 6000 Ada machine has driver 550.163.01. PyTorch 2.7.1 and torchvision 0.22.1 use their matching CUDA 11.8 wheels; Warp uses its CUDA 12 wheel. Both work with the existing driver. No driver or system CUDA replacement is needed. `requirements-mjlab-lock-cu118.txt` records the complete environment; `requirements-mjlab.txt` records the direct dependencies.

To reproduce in a separate environment:

```bash
uv venv --python 3.12 .venv-mjlab
uv pip install --python .venv-mjlab/bin/python \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv-mjlab/bin/python -r requirements-mjlab-lock-cu118.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
```

Use `JAX_PLATFORMS=cpu`: a few existing scene metadata/validation helpers import JAX, but no JAX GPU physics or learning is used. `--compile-task` optionally fuses the pure Torch field, navigation, reward, action and collision kernels. Compilation does not change the formulas; first-use compilation time is excluded from warmed throughput reports.

## Validation and operation

`scripts/check_mjlab_physics.py` checks real Warp/native MuJoCo kinematics, a matching physics step, final-pose collision kinematics, reset isolation, finite dynamics and allocation capacity. `train_cat_mjlab.py verify` performs a bounded real rollout and optimizer update with W&B disabled. `scripts/benchmark_mjlab.py` measures the full task and learner without publishing checkpoints or W&B runs.

All Warp contact/constraint overflow flags are latched before autoreset. An overflow aborts before the optimizer update and prevents publishing a corrupt resume checkpoint. Contact buffers are never silently truncated by the training loop.

The numerical tests cover field interpolation, approved collision geometry, action targets, observations, rewards, delayed commands, reset histories, horizons, GAE, losses and gradients, parameter/Adam conversion, and crash-safe logging/resume handling. Hardware validation results and measured capacity are recorded below.

A continuous run uses `train_cat_mjlab.py run`; its default `--max-updates 0` means no artificial training-length limit. Choose the source bank explicitly, use a new run directory for migration, and pass `--resume` only for a native mjlab run with an unchanged contract. Creating `STOP` in its run directory requests a cooperative save and stop.

## Measured checks on the RTX 6000 Ada

- **52 numerical/CPU tests passed.**
- Actual GPU physics: native MuJoCo vs Warp one-step maximum `qpos` error `2.01e-8`; exact isolation of continuing-world observations across reset; finite dynamics and zero capacity overflows.
- Actual GPU SAPG update on the preserved specialist bank; compiled update on the new 96-scene bank; then a complete native save/resume and another update. W&B disabled throughout.
- **36,864 environments**, original specialist bank, restored stage-2 sampling, original 576 × 64 × 32 batch: **55,037 physical environment steps/s**, consisting of 15.741 s rollout plus 5.693 s optimization. Device memory after the update: **28.80 GiB used / 18.69 GiB free**, including Warp and other processes. Maximum observed constraint count was 81/256, with no overflow.
- Full **2,338-scene procedural/clutter bank**, original released checkpoint expanded for PPO, **18,432 environments**: complete rollout and optimizer update passed, **27.36 GiB** device memory, **11,865 steps/s**, no capacity overflows. This uses a different bank, algorithm and batch; its speed is not a comparison with the specialist baseline. [Full diversity measurement](assets/mjlab-migration-20260919/diversity-18432.json).
- The stopped MJX run's last 100 updates averaged **39,340 steps/s** (29.986 s/update) for the same scene bank and batch. The short Warp probe is about 40% faster, but begins from fresh physical episodes; this is indicative throughput, not an overnight stability guarantee or a policy-performance comparison. First-use compilation is excluded from warmed throughput.

See [physics measurements](assets/mjlab-migration-20260919/physics.json) and [complete specialist benchmark](assets/mjlab-migration-20260919/source-36864.json). Larger scene banks consume additional shared field memory; the specialist measurement must not be applied blindly to a generalist scene bank.

## Prepared checkpoints

Inside the new checkout, under `outputs/mjlab_migration_20260919/`:

- `sapg-660602880-with-sampling.npz`: model, critic, all embeddings, Adam and same-bank sampler/curriculum. SHA256 `ac3b4550de39ac5d42ac9295794124a5ef94bec1a7975a6c63ca58e94840c39b`.
- `original-cat-expanded.npz`: original released generalist, expanded using the unchanged feature/action adapter, for fresh PPO or SAPG fine-tuning. SHA256 `0c3f6fa49f7d472b9ccf7b463111e01fec1106579b2fbe420b7d8a4198b58bca`.

The first 64 parameter/optimizer arrays in the sampler-preserving archive were checked to be identical to the earlier learner-only conversion. The full original runtime checkpoint remains untouched. Bounded verification checkpoints are engineering artifacts, not new recommended training checkpoints.

## Prepared continuous SAPG command (not started)

This selects the exact previous specialist bank and restores its sampler. To deliberately train the new contrastive scenes, replace the three bank paths with `data/furniture/contrastive_table_v4/manifest.json`, `data/furniture/contrastive_table_v4_collision/manifest.json`, and `data/furniture/contrastive_table_v4_resets/manifest.json`, and use a different run directory.

```bash
cd ~/robotics/Click-and-Traverse-Mjlab
cat_source=../Click-and-Traverse-WholeBody
JAX_PLATFORMS=cpu MUJOCO_GL=egl TORCHINDUCTOR_COMPILE_THREADS=4 \
  .venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm sapg --num-envs 36864 --batch-size 576 --unroll-length 32 \
  --compile-task --max-updates 0 \
  --checkpoint-npz outputs/mjlab_migration_20260919/sapg-660602880-with-sampling.npz \
  --bank-manifest "$cat_source/data/furniture/cat_hand_specialist_v1_20260918/fields/manifest.json" \
  --body-collision-bank "$cat_source/data/furniture/cat_hand_specialist_v1_20260918/collision/manifest.json" \
  --body-collision-resets "$cat_source/data/furniture/cat_hand_specialist_v1_20260918/resets/manifest.json" \
  --run-dir outputs/sapg_mjlab_migrated \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
```

For PPO, select `--algorithm ppo` with `original-cat-expanded.npz` (fresh optimizer) or export a compatible PPO runtime. A conditioned SAPG runtime is deliberately rejected as a PPO checkpoint rather than silently dropping its follower/embedding state.

The singleton reset case uses the identical eager kernels to avoid a confirmed Torch 2.7/Triton shape-specialization compiler defect. Batches larger than one use the compiled path. Other compilation errors propagate; there is no broad silent fallback that could hide an incorrect kernel.

## Recording from native mjlab checkpoints

`best.pt` and `resume.pt` use a new format; historical JAX checkpoint loaders cannot read them directly. `scripts/record_mjlab_rollout.py` loads either native format, validates package/source/scene identities, selects a policy explicitly (leader 0 by default), and records one requested scene with a frame bound. It preserves the terminal physical pose before autoreset. It creates `model.xml`, `trajectory.npz`, `scene.json`, and metadata identifying the backend and checkpoint.

The recorder has four synthetic/loader tests. No policy-quality recording or evaluation was run during migration. Rooms can use the existing `scripts/render_clutter_rollouts.py`; both rooms and CAT occupancy scenes can use the new generic replay command:

```bash
.venv-mjlab/bin/python scripts/record_mjlab_rollout.py render \
  --input-dir RECORDING_DIRECTORY --output recording.mp4
```

Rendering replays saved poses and does not run policy inference or additional physics. The recorder's `--help` lists the required checkpoint, scene bank, exact scene ID and frame-bound arguments.
