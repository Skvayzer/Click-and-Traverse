# Hand-protection specialist with SAPG — 18 September 2026

**Status: sustained training launched on `tl-server-0` as Slurm job 680**, using
36,864 environments and batch size 576. CPU bank job 676 and both production-size
capacity checks passed. This establishes initialization and complete-update
capacity; learned hand-protection skill still needs training and later videos.

- Run: `outputs/cat_hand_specialist_sapg_680`.
- Pinned source: `outputs/sources/cat_sapg_680_2d2f76acc205`.
- [W&B run f017f302](https://wandb.ai/skvayzer/CAT-wholebody/runs/f017f302).
- [Hand-protection success view](https://wandb.ai/skvayzer/CAT-wholebody?nw=2f7040a5654),
  filtered to this run with one expanded success chart. Its API representation
  was read back and verified; existing workspaces were preserved.
- Fresh original checkpoint initialization; no evaluation, retention or rollback.
- Seven-day scheduler allocation, no learner-step cap. Existing dep-1 training
  was not modified.

The production run completed its first optimizer update at **1,179,648 physical
transitions**. Local metrics were finite, W&B reported `state=running` at that
same step with `health/nonfinite=0`, and both the best policy and full resume
snapshot were saved. A diagnostic inside its Slurm allocation measured
33,358 MiB used and 15,152 MiB free. Initial hand success was zero after only
32 control steps per environment; this is startup evidence, not a skill result.

Implementation commit: `2d2f76acc20536e07979497ef589bbe0e192d646`.
Validation: 171 focused tests passed; 18 relevant tests passed again after the
metadata correction. The rerun is not an additional set of 18 unique tests.
Capacity jobs 677 and 679 used detached source
`outputs/sources/cat_hand_capacity_2d2f76a`.

| Capacity job | Environments | Physical transitions checked | Warm update | NVIDIA used / free |
|---|---:|---:|---:|---:|
| 677 | 24,576 | 1,572,864 | 34.04 s | 33,358 / 15,152 MiB |
| 679 | 36,864 | 2,359,296 | 52.96 s | 33,358 / 15,152 MiB |

Each check completed two native-size updates with finite metrics and parameters,
preserved runtime embeddings and a verified folded leader export. JAX live peak
rose from 17.72 GB to 26.36 GB; the allocator retained the same 34.36 GB pool,
which explains the identical NVIDIA memory readings. These are completed-update
device snapshots, not a promise of long-run peak usage. Reports are in
`outputs/hand_specialist_sapg_20260918/capacity_{24576,36864}/gpu_report.json`.
The sustained run starts afresh and does not reuse the checker's trained weights.

## Task-only scene collection

`scripts/build_hand_specialist_bank.py` extracts all **24 existing hand-protection
layouts** from the complete 2,362-scene collection: 12 hand-height table-edge
aisles and 12 staggered shelf passages. Each kind has four seeds at each of
easy, medium and hard difficulty. This specialist samples no original/generated
CAT scenes or ordinary clutter rooms; the complete source bank remains intact.

Geometry, distance/guidance fields, checked routes and certified reset poses
retain their existing values. The builder copies selected scene assets, compacts
collision-array addresses, remaps scene indices, and verifies source hashes and
scene correspondence. It publishes separate `fields/`, `collision/` and
`resets/` manifests plus provenance in a fresh output directory. It does not
regenerate scenes. Selected fields occupy approximately 325 MB on disk; the
capacity report records actual device-array sizes separately.

The completed bank is `data/furniture/cat_hand_specialist_v1_20260918`.
Its published manifest SHA-256 values are:

| Manifest | SHA-256 |
|---|---|
| `fields/manifest.json` | `2da26dc8ca5de0dd321612583e1cb20f4a1f316b9e0e40d58e63b6701274d553` |
| `collision/manifest.json` | `b942fd451d20e99f01c006cd0d196271e37a2e89a5b78fb264477d0fc0e481e3` |
| `resets/manifest.json` | `0431add14e7e3301039439974a3f12c0a74150dbb70fc66459d9311a071ebbcc` |

At episode reset, **50% probability goes to table aisles and 50% to shelf
passages**. These are sampling probabilities, not promised transition fractions.
Existing adaptive weights select among eligible scenes within each kind.
Training starts with eight easy scenes. The shared curriculum unlocks medium,
then hard after at least **64 resolved attempts and 60% clean-goal success at
the current level**, pooled across both scene kinds and all policy groups.
Previously unlocked levels remain eligible, yielding 16 and then 24 scenes.

An attempt resolves once at first clean goal arrival, first fault, or timeout
before arrival; simultaneous goal and fault is failure. Goal arrival does not
reset physics. Later collisions still penalize and terminate the physical
episode without erasing its already counted navigation outcome. Existing
4,000-step room horizons, hand/elbow checks, approved body collision primitives,
route following and hand-clearance/arm-motion rewards remain enabled.

These layouts certify that raised or tucked poses can provide clearance. They
do not force a unique behavior: sideways alternatives exist. Learned raising or
tucking still requires rollout evidence. Task-only training also does not
establish retention of the pretrained policy's broader traversal abilities.

## Initialization and learner

Start from the **original released CAT generalist actor and critic**, model
revision `46ce4b57ba0639168d51741b661ff62f7ce6f045`, with fresh Adam state.
`--finetuning cat_train_only` rejects a fine-tuned best-model warm start. The
existing whole-body expansion preserves original leg outputs, adds arm/torso
control, and initializes added action scales to `0.05`.

SAPG uses six equal environment groups: leader 0 and followers 1–5. One shared
actor and one shared critic use an actor-owned table of six learned
16-dimensional embeddings. Added embedding-input weights start at zero, so
conditioning initially preserves every policy's output. The robot interface
stays **222 actor observations, 310 critic observations and 29 actions**;
embedding coordinates are internal network inputs. All actor/critic parameters
and state-dependent independent Gaussian action scales are trainable.

| Setting | Value |
|---|---|
| Adam learning rate / gradient-norm limit | `3e-4` constant / `1.0` |
| PPO clipping / entropy coefficient | `0.2` / `0.003` for all six policies |
| Discount / GAE lambda / reward scaling | `0.98` / `0.95` / `1.0` |
| Rollout length / minibatches / optimizer passes | `32` / `64` / `4` |
| Observation normalization | Disabled |
| Advantage normalization | Once over the complete augmented rollout |
| Target-preparation chunk size | 64 trajectories |
| Evaluation, retention, reference KL, rollback | Disabled |

Every update includes all six on-policy blocks and one randomly chosen follower
block copied for leader optimization. Targets remain frozen over all optimizer
passes; copied samples do not increase physical steps or success counts. The
importance-corrected actor objective, one-step off-policy critic target and
unweighted `0.25 * MSE` value loss are documented in
[SAPG_TRAINING_20260918.md](SAPG_TRAINING_20260918.md).

One W&B run records actual training outcomes pooled across policies. For this
hand-only bank, overall and hand-protection success rates describe the same
population; original-scene and ordinary-clutter counts are zero and their rates
are omitted. Best-model selection uses **leader mean training transition
reward**, not a separate success evaluation. Storage retains one folded leader
best model and one overwritten full SAPG `resume.msgpack`.

## Reproduce the bank and capacity check

On `tl-server-0`, use a clean checkout of the reviewed implementation commit.
The builder, GPU capacity check and training all run through Slurm, including
substantial CPU work. To reproduce the build, select a fresh output directory;
to reuse the completed bank above, skip the build submission.

```bash
cd /data1/users/konstantin.smirnov/Click-and-Traverse-WholeBody
export CAT_REPO_ROOT="$PWD"
export CAT_SOURCE_COMMIT="$(git rev-parse HEAD)"
export CAT_SPECIALIST_BANK="$CAT_REPO_ROOT/data/furniture/cat_hand_specialist_v1_20260918"
sbatch --wait --export=ALL <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=cat-hand-specialist-build
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$CAT_REPO_ROOT"
test "$(git rev-parse HEAD)" = "$CAT_SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
.venv/bin/python scripts/build_hand_specialist_bank.py \
  --field-manifest data/furniture/cat_hand_protection_v1_20260917/manifest.json \
  --collision-bank data/furniture/body_collision_hand_v1_20260917/manifest.json \
  --reset-manifest data/furniture/body_collision_hand_resets_v1_20260917/manifest.json \
  --output "$CAT_SPECIALIST_BANK"
SLURM

export CAT_FIELD_BANK="$CAT_SPECIALIST_BANK/fields/manifest.json"
export CAT_COLLISION_BANK="$CAT_SPECIALIST_BANK/collision/manifest.json"
export CAT_RESET_BANK="$CAT_SPECIALIST_BANK/resets/manifest.json"
export CAT_NUM_ENVS=36864
export CAT_BATCH_SIZE=576
```

The first candidate has 4,096 environments per policy and **786,432 physical
transitions / 917,504 optimizer samples per update**, before four-pass reuse.
The larger `36864 / 576` candidate has 6,144 environments per policy and
1,179,648 physical / 1,376,256 augmented samples. Both passed the bounded capacity
check; sustained training uses the larger pair.

```bash
sbatch --export=ALL <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=cat-hand-sapg-capacity
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$CAT_REPO_ROOT"
test "$(git rev-parse HEAD)" = "$CAT_SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
.venv/bin/python scripts/verify_sapg_training.py \
  --production-batch --updates 2 \
  --num-envs "$CAT_NUM_ENVS" --batch-size "$CAT_BATCH_SIZE" \
  --bank-manifest "$CAT_FIELD_BANK" \
  --body-collision-bank "$CAT_COLLISION_BANK" \
  --body-collision-resets "$CAT_RESET_BANK" \
  --min-free-vram-mib 3072 \
  --report "$CAT_REPO_ROOT/outputs/hand_sapg_capacity_${SLURM_JOB_ID}/report.json"
SLURM
```

The checker performs two complete updates and exits. It verifies finite metrics
and parameters, physical-step accounting, embeddings in runtime state and
folded leader export. Its report includes scene identities, source/warm-start
hashes, SPS, JAX live/peak memory and NVIDIA used/free/total VRAM on the GPU
resolved from its own process ID. It leaves Slurm's GPU selection intact.
The 3,072 MiB threshold is a capacity screen, not an overnight stability
guarantee. The checker creates no W&B run or evaluation episodes; its disposable
updates are not the warm start for sustained training.

## Reproduce sustained training

After selecting a resource pair from a successful capacity report, retain the
same three specialist manifest variables and exact reviewed source commit:

```bash
export CAT_RUN_PREFIX=cat_hand_specialist_sapg
bash -n scripts/slurm/train_cat_sapg.sbatch
sbatch --time=7-00:00:00 --export=ALL scripts/slurm/train_cat_sapg.sbatch
```

The recipe creates a detached source worktree and a fresh
`outputs/cat_hand_specialist_sapg_JOBID` run. It requests one RTX 6000 Ada and
starts again from original pretrained weights. There is no learner-step cap;
manual cooperative stop or the seven-day Slurm allocation limit ends the job.
The explicit submission override replaces the recipe's two-day default.
To stop this specific active run cooperatively, create
`outputs/cat_hand_specialist_sapg_680/STOP`, or send its batch shell USR1 with
`scancel --signal=USR1 --batch 680`. The learner completes its update and saves
its current runtime state. No improvement in learned behavior is claimed from
the capacity checks.
