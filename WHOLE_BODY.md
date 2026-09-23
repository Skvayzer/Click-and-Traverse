# CAT whole-body traversal and hand protection

**Current native contract:** actor **222** / critic **310**. Authored box features and rewards are removed; see [removal and from-scratch launch](docs/NATIVE_CLEARANCE_REMOVAL_20260922.md).

**2026-09-16 update:** [full-body primitive collision training and restart](docs/CAT_BODY_COLLISION_TRAINING_20260916.md) adds the approved 35 shapes, physics-substep collision termination and an explicit terminal penalty, together with the corrected room routes. The contract remains [222 actor inputs / 310 critic inputs / 29 actions](docs/CAT_COMPACT_SPHERE_SETUP_20260916.md). The implementation report records capacity and current launch status; [earlier stopped-run evidence](docs/ROOM_NAVIGATION_FIX_20260916.md) is preserved.

The setup, run settings and commands below describe the **September 15 historical configuration**, including its retired 406-input contract; use the dated September 16 reports above for the current state.

This branch extends the released Click-and-Traverse generalist with 29 body actions, fixed Unitree Dex3-1 hands, hand/arm clearance features and dense clutter. It is CAT-only. The September 15 correction restores CAT's task and mixed-scene learner. Earlier overnight runs used a materially different task and remain historical experimental artifacts.

![Dense demonstration scene: nine tables and 36 chairs](docs/assets/dense-room.png)

Corrected learned traversal success has not yet been demonstrated. See [correction and validation](docs/CAT_SETUP_CORRECTION.md), [field provenance](docs/CAT_FIELD_BANK.md), and [visuals](docs/VISUALS.md).

Corrected continuous training restarted on dep-0 on September 15 with 8,192 environments and [one W&B run](https://wandb.ai/skvayzer/CAT-wholebody/runs/1d39c55c). [Run settings, capacity evidence and stop command](docs/CORRECTED_RUN_20260915.md).

[Illustrated current-setup PDF](docs/CAT_CURRENT_TRAINING_SETUP_20260915.pdf) and [technical companion](docs/CAT_CURRENT_TRAINING_SETUP_20260915.md): architecture, control/PPO pipeline, actual furniture layouts, hand protection, source identities and progress at **1.139 billion transitions (14:47 UTC, September 15)**. Generic clutter survival improved; furniture early termination remains 97.42%, and safe hand raising is not yet demonstrated.

[Progress analysis at 566 million transitions](docs/TRAINING_PROGRESS_20260915_1243.md): survival improved, especially in generic clutter, while furniture early termination remains high and the hand-clearance objective needs attention. No controlled traversal-success improvement has been established.

## Preserved CAT behavior

The reference is upstream commit `866ba392f1c1e84b92ad75fa66550f26e8af8e48`, the checksum-pinned [released configuration](configs/cat_generalist_released.json), and public native generalist checkpoint `005033164800` at revision `46ce4b57ba0639168d51741b661ff62f7ce6f045`. The final generalist already completed distillation; direct PPO fine-tuning restores actor and critic and initializes Adam once.

The whole-body task inherits CAT's reset and step implementation: complete reward, 20 ms control/2 ms physics, raw-field commands, stopping gait, five-step odometry hold, proprioceptive noise, randomized starts, PD/torque disturbances and pushes. Collision termination uses the released **0.0 m threshold and 50-control-step grace**. Falls remain immediate. CAT's adaptive reset wrapper resamples scene IDs; its survival statistic measures timeout survival, not goal-reaching success.

Training uses CAT's flat floor/feet contact model. Obstacles are potential fields, as in CAT. Physical furniture and strict physical-contact evaluation remain separate tools.

## Explicit extensions

- Actions cover 12 leg, 3 waist and 14 arm/wrist joints. Leg targets retain CAT's incremental semantics and 0.5 scale. Upper actions produce nominal offsets with 0.8 scale and 2 rad/s slew limit.
- Dex3-1 three-finger hands retain official inertias. Seven finger joints per hand are fixed; wrists remain controlled. Mesh-derived safety envelopes include thumbs with 5 mm padding.
- Actor/critic observations expand from 162/250 to 406/494. Named weight mappings preserve released inputs/outputs; added inputs have zero initial weights, new action means start at zero and standard deviation at 0.05. The original interpolation convention is retained.
- Twenty-two hand/arm probes add clearance, approximate predicted clearance and surface directions. Hand and arm penalties (initial scales −5 and −2) augment the **full CAT reward**, inside its existing dt scaling and nonnegative clipping. Added probe penetration uses CAT's same collision grace.
- Dense rooms use their entrance with ±8 cm XY jitter and CAT's yaw/joint/velocity randomization. Original scenes retain their reset distribution. New rooms have an 80 s limit because routes are longer; original scenes retain 20 s. Timeouts are truncations. Only new rooms replace CAT's hardcoded crossing plane with their actual goal.

These are research settings; geometry checks do not establish dynamic feasibility or hardware injury thresholds.

## Scene bank and one learner

One process, one live PPO learner and one W&B run contain all scenes. Scene IDs change per environment at reset. There are no stage subprocesses, optimizer resets or best-model rollbacks. CAT's adaptive sampling uses alpha 1 and EMA decay 0.95, initialized uniformly over scene slots. Pooled episode/loss plots and per-scene diagnostics share one continuous step axis.

The prepared bank has **36 byte-verified released fields, one explicitly reconstructed missing configured scene, and two dense FMM rooms**. The release config names `D8G2L3O2S13`, but the public dataset lacks it. Reconstruction uses the original generator parameters/pipeline and is not claimed to reproduce its unknown original bytes. [Details and source pins](docs/CAT_FIELD_BANK.md).

Furniture has nine tables and 36 chairs with separate tops, seats, backs and legs. Generic clutter uses blocks, partitions and shelves. Both use 4 cm occupancy and CAT's original progressive 3D fast marching, SDF and boundary-gradient functions. The former route-lookahead training field is retired. Ragged storage preserves original coordinates, boundaries and samples without padding every field to room size.

| PPO setting | Released CAT | Active dep-0 run |
|---|---:|---:|
| Learning rate / entropy / discount | 0.0003 / 0.003 / 0.98 | same |
| Unroll / minibatches / passes | 32 / 64 / 4 | same |
| Clip / GAE / max gradient norm | 0.2 / 0.95 / 1 | same |
| Observation normalization | disabled | same |
| Actor hidden layers | 512,256,128,64 | same |
| Critic hidden layers | 1024,512,256,128 | same |
| Parallel environments | 65,536 | 8,192 |
| Trajectories per minibatch | 2,048 | 512 |
| Transitions per update | 4,194,304 | 1,048,576 |
| Rollout chunks per update | 2 | 4 |

The active run explicitly reduces optimizer batch size fourfold: expanded observations make the original full rollout too large for this 32 GB GPU. Changing `--num-envs` changes parallelism/chunk count, **not** minibatch size or transitions per update. The conservative CLI defaults remain 2,048 environments and 256 trajectories per minibatch; the active run overrides both after the GPU capacity check and sets the allocator fraction to 0.90.

## Setup and commands

Existing Mac and dep-0 environments use Python 3.12.9, JAX 0.4.38, Brax 0.12.3, MuJoCo/MJX 3.3.1 and Playground 0.0.4. Requirements are in `requirements/furniture-cpu.txt` and `requirements/furniture-cuda12.txt`; full platform snapshots are under `requirements/locks/`. Original robot assets are required in `data/assets/unitree_g1` and already exist on both machines. Dex3 additions are tracked.

```bash
git checkout feature/whole-body-furniture-traversal
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python prepare_cat_generalist.py --download --reconstruct-missing-original \
  --output data/furniture/cat_generalist
.venv/bin/python train_cat_wholebody.py plan
.venv/bin/python train_cat_wholebody.py validate
```

`plan` prints settings. `validate` checks fields, observation shapes and native weight mapping without training. Prepared fields and pinned weights are reused.

To launch another run with the checked dep-0 settings, use a new run directory and an available GPU:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
.venv/bin/python train_cat_wholebody.py run \
  --num-envs 8192 --batch-size 512 \
  --run-dir outputs/cat_wholebody_generalist --wandb-mode online
```

There is no finite training budget or automatic evaluation. Stop at a completed PPO update:

```bash
.venv/bin/python train_cat_wholebody.py stop --run-dir outputs/cat_wholebody_generalist
```

Storage is **one best model** selected by aggregate training reward proxy, plus **one overwritten `resume.msgpack`** containing current parameters, Adam, RNG, normalizer, environment and sampler. Selection does not certify held-out success. The best model never replaces the live learner. Full recovery state is saved before the first update as well.

After deliberately removing that run's `STOP` file, `run --resume` restores its complete state and same online W&B ID. Resume refuses incompatible task/source/field/resource contracts. Nonfinite metrics are reported and stop the process. The retired `train_furniture.py` and sequential `cat_ppo.furniture.continuous run` CLIs refuse new training. Historical recordings and evaluation tools remain available for their original checkpoint contracts.
