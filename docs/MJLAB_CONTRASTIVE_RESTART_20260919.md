# Contrastive SAPG restart from the released generalist

The approved restart starts from the **original released CAT generalist**, not the stopped SAPG specialist or a migration verification checkpoint. The expanded archive is `outputs/mjlab_migration_20260919/original-cat-expanded.npz`, SHA256 `0c3f6fa49f7d472b9ccf7b463111e01fec1106579b2fbe420b7d8a4198b58bca`. Its metadata records hashes of the release files in `data/furniture/native_generalist_v1`.

The original actor and critic are expanded from 162/250 observations and 12 actions to 222/310 observations and 29 actions. Added observation weights start at zero; inherited leg outputs are preserved; new upper-body action means start at zero with standard deviation 0.05. Six new 16-dimensional SAPG embeddings have zero initial input weights, so all policies initially reproduce the same expanded generalist. Adam and training step counters start fresh. The older specialist checkpoint remains preserved.

## Approved bookkeeping corrections

1. The four headline `success/*` charts count **leader policy 0 only**, using first resolved outcomes and a 100-update rolling window. Missing outcomes are omitted, not reported as zero. Follower and virtual relabeled transitions cannot inflate these rates.
2. Scene sampling still uses every policy's physical outcomes, but protected and transition scenes require the same posture qualification as their success metrics. A sideways clean arrival cannot make an unsolved protected scene look easy. Open and narrow scenes retain clean-goal success; legacy banks retain their existing adaptation.
3. `best.pt` uses the mean of four role success rates: open clean goal, forward protected with qualified posture, narrow clean goal, and qualified transition. Selection waits until all four roles have resolved leader episodes. The roles have equal weight regardless of their observed episode frequencies. This is a training-history selection proxy: the window contains past rollouts and published weights are post-update, without evaluation of that exact snapshot.

The W&B saved view contains exactly four charts: `success/goal_success_rate`, `success/forward_protected_success_rate`, `success/narrow_passage_success_rate`, and `success/posture_transition_success_rate`. Overall goal success is deliberately a clean-navigation measure; the protected and transition charts additionally require posture. Historical charts and runs are preserved.

## Task and learner

- 96 contrastive scenes: 64 cabinet passages and 32 table-height variants. Open/protected/narrow/transition roles each receive 25% reset mass. The initial table share is 25%; within-role adaptive sampling can change it.
- Forward pelvis/torso alignment where forward travel is certified feasible; sideways travel remains allowed in narrow zones.
- Paired hand regions guide raising or tucking in cabinet passages; table passages use raised regions. These regions are reward metadata, not extra observations or rigid constraints. Neutral-arm regularization is disabled throughout active protection zones.
- Goal qualification requires every required zone visited, with heading and hand conditions each satisfied on at least 90% of the zone's fully active steps.
- Fixed Dex3 grippers, 29 joint actions, 35 approved body collision primitives, conservative obstacle geometry, and certified ordered route guidance remain enabled.
- SAPG uses one leader and five followers, shared actor/critic and embeddings, and the existing log-domain importance-ratio overflow fix.
- Learning rate `3e-4`, PPO clip `0.2`, entropy `0.003`, discount `0.98`, GAE lambda `0.95`, gradient norm limit `1`, 64 minibatches and four passes. Rollout length is 32.
- This is a hand-protection specialist bank, without the broader procedural/generalist or ordinary random-clutter collection. Training success here does not establish retention of those other skills.
- Continuous training, one W&B run, no evaluation jobs, no retention or rollback, no motion prior. One best model and one atomically replaced full resume checkpoint.

## Capacity and launch

The full 96-scene task and a complete SAPG update were measured on dep-1's RTX 6000 Ada at **49,152 environments**, with batch size 768 and 8,192 environments per policy. The initial capacity probe used 42.709 GiB of 47.492 GiB available device memory, leaving 4.784 GiB. It processed 45,553 physical control transitions/second, with no contact/constraint overflow. This short engineering probe does not establish overnight stability or policy quality. The remaining headroom is reserved for reset variation, compilation and snapshot creation; contact capacities are not reduced to fit extra worlds.

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  .venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm sapg --num-envs 49152 --batch-size 768 --unroll-length 32 \
  --compile-task --max-updates 0 \
  --checkpoint-npz outputs/mjlab_migration_20260919/original-cat-expanded.npz \
  --bank-manifest data/furniture/contrastive_table_v4/manifest.json \
  --body-collision-bank data/furniture/contrastive_table_v4_collision/manifest.json \
  --body-collision-resets data/furniture/contrastive_table_v4_resets/manifest.json \
  --run-dir outputs/cat_contrastive_sapg_mjlab_49152_20260919 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
```

Each full update collects 1,572,864 physical transitions. Runtime snapshots overwrite `resume.pt` every ten updates and on a cooperative stop. Create `STOP` inside the run directory to stop cooperatively. Native `--resume` requires an identical saved training contract.
