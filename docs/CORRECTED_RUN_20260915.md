# Corrected CAT generalist run — 15 September 2026

Continuous training was restarted on dep-0 (`WS012743`) at **10:37:15 UTC**, after the user authorized restart and a final CAT comparison and GPU compiler capacity check passed. The learner starts from the released final CAT generalist, not the earlier overnight experiment.

- W&B: [cat_wholebody_generalist_20260915](https://wandb.ai/skvayzer/CAT-wholebody/runs/1d39c55c), one experiment ID `1d39c55c`.
- Training source: `40b4dd2682dd4e5d75a6b853c24333da99379cf3`; Python source hash `14292904cb7065229cbba32f90ca43ff1084260e5053239f28f54e7d7617ef2d`.
- Original CAT reference: `866ba392f1c1e84b92ad75fa66550f26e8af8e48` and checksum-pinned `configs/cat_generalist_released.json`.
- Native initialization: `Axian12138/Click-and-Traverse`, revision `46ce4b57ba0639168d51741b661ff62f7ce6f045`, `logs_v1/generalist_v1/checkpoints/005033164800`. Actor and critic restored; mapped actor/value output parity error is zero. Adam initializes once for fine-tuning.
- Initial learner PID: `11525`. Source is frozen in a detached worktree under `outputs/sources/cat_wholebody_generalist_20260915`; the environment and assets are shared with the main checkout.

## CAT fidelity and deliberate differences

The final code review found no remaining unintended deviation in the CAT task or native PPO procedure. The full reward, original reset distribution, noise/disturbances, gait/odometry behavior, potential-field training physics, adaptive scene sampling, networks and non-resource PPO coefficients remain. Collision termination retains the **0.0 m threshold and 50-step grace**; falls terminate immediately. The released final generalist has already completed distillation, so this run fine-tunes it directly with PPO.

The planned extensions are 29 body actions, fixed Dex3-1 fingers, thumb-covering envelopes, hand/arm probes and penalties, and dense clutter. Original scenes retain 20-second episodes. Larger rooms use their entrance and goal and an 80-second timeout. See [full setup](../WHOLE_BODY.md).

Two other differences are explicit: the optimizer batch is reduced for this GPU, and one scene missing from the public release is reconstructed. The bank contains 36 byte-verified original scenes, that reconstructed original slot, and two added dense rooms: **39 scene slots in one learner**. Its on-disk manifest SHA256 is `bbcccb9a11ebee99ad14f5330de71a6a9461aed0e5c2115774b8c3b4ce19c633`. [Field provenance](CAT_FIELD_BANK.md).

| Resource setting | This run | Released CAT |
|---|---:|---:|
| Parallel environments | 8,192 | 65,536 |
| Trajectories per minibatch | 512 | 2,048 |
| Transitions per PPO update | 1,048,576 | 4,194,304 |
| Transitions per minibatch | 16,384 | 65,536 |
| Rollout chunks per update | 4 | 2 |
| Unroll / minibatches / passes | 32 / 64 / 4 | 32 / 64 / 4 |

No stage resets, optimizer resets, best-weight rollbacks, automatic evaluation, or finite training budget are configured. Aggregate plots and scene diagnostics use the same W&B history. There is one selected best model plus one atomically overwritten full recovery file.

## GPU validation

Before launch, the actual native PPO epoch compiled with 8,192 environments and a 512-trajectory minibatch. The diagnostic executed zero PPO updates, opened no W&B run and wrote no checkpoint. Its compiler buffer estimate was **21.08 GiB** against a **28.26 GiB allocator limit**, leaving about **7.18 GiB** of estimated buffer headroom. Generated-code accounting and initialization allocations are recorded separately; this estimate alone does not establish runtime stability.

[Compiler capacity measurements](assets/cat-corrected-capacity-20260915.json).

The actual training process uses `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` on the RTX 5090 (32,607 MiB reported total).

At **10:40:57 UTC**, the same process had completed **three PPO updates / 3,145,728 transitions**, with the full resume state saved at that step and a selected native best checkpoint present. All recorded losses were finite. The second and third updates reported about **88,000 transitions/s**, excluding checkpoint-writing time; the first includes compilation.

Actual NVIDIA memory usage was **30,425 MiB (29.71 GiB)** with **100% utilization** in the execution sample, leaving 2,182 MiB of reported total capacity. JAX reported **24,309,893,632 bytes (22.64 GiB)** of peak live allocations and a 28.26 GiB retained allocator pool. NVIDIA usage includes that pool and CUDA overhead; it is not all simultaneously live tensors.

The W&B server independently showed exactly one new run in this project, state `running`, and successive optimizer updates on its shared `global_step` axis. [Startup runtime and W&B evidence](assets/cat-corrected-runtime-20260915.json). These observations establish successful initial updates and checkpoint writes, not overnight stability or learned traversal success. The learner remains running without a scheduled cutoff.

## Operator paths and stopping

All paths below are relative to `/home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody` on dep-0:

- Run: `outputs/cat_wholebody_generalist_20260915`
- Console: `output_logs/cat_wholebody_generalist_20260915.log`
- Exact argv/environment/PID/start time: `output_logs/cat_wholebody_generalist_20260915.launch.json`
- Full recovery: `outputs/cat_wholebody_generalist_20260915/resume.msgpack`
- Selected model: `outputs/cat_wholebody_generalist_20260915/checkpoints/best`

Manual cooperative stop from the Mac:

```bash
ssh konstantin.smirnov@dep-0 'cd /home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody && .venv/bin/python train_cat_wholebody.py stop --run-dir outputs/cat_wholebody_generalist_20260915'
```

This requests stopping after a complete PPO update and recovery write. An exact resume uses the original frozen source, identical arguments plus `--resume`, and the same W&B ID, after deliberately removing `STOP`. Training is intended to continue until manually stopped; process or hardware errors can still terminate it.
