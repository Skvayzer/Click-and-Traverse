# Larger continuous CAT run — September 15, 2026

This run stopped at 06:15 Dubai time on September 15 after a GPU OOM in the
first dense generic scene (2,048 environments). It completed 12 stages and
102,301,696 cumulative transitions including its imported offset. The retained
checkpoint was verified intact. See [the recovery notes](RECOVERED_RUN_20260915.md)
for the subsequent restart; the commands and measurements below describe this
historical run.

The user requested a restart with more parallel environments, targeting about
30 GB of GPU memory. The initial smaller run was cooperatively stopped and its
selected generic-clutter checkpoint was verified before importing it into this
run. No reset to the public starting weights occurred.

- Launch: September 15, 01:17:40 Dubai time (September 14, 21:17:40 UTC).
- Host: dep-0 / WS012743; RTX 5090, 32,607 MiB VRAM.
- Source: immutable worktree at `12a4ad8f1b03fc319db03d9d0dab6b8278f549a3`.
- Run: `/home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_large`.
- Log: the same absolute path with `.log` appended.
- W&B group: `continuous_cat_dex3_20260915_large` in `skvayzer/CAT-furniture`.
- First restarted stage: [r000000-p0-generic](https://wandb.ai/skvayzer/CAT-furniture/runs/f4qcqtdm).
- Supervisor PID at launch: `786591`; initial worker: `786603`.
- Imported selection: generic stage step `589824`, after `1048576` preceding CAT-forward transitions.
- Initial cumulative W&B offset: `1638400`.

| Scene family | Parallel environments |
| --- | ---: |
| Typical original CAT and simple generic clutter | 8,192 |
| Simple furniture | 4,096 |
| Random original CAT | 2,048 |
| Dense generic clutter | 2,048 |
| Dense furniture | 1,024 |

The new run replays the interrupted generic scene from its selected actor,
critic and normalizer, with a fresh optimizer and the larger batch. There are
16-step rollouts, four minibatches and four PPO passes. Scene cadence increases
to 8,388,608 transitions, preserving 64 rollout batches per light scene and
amortizing compilation over more work. Candidates are considered every 524,288
transitions. There is no global time, step or stage limit.

The launcher sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.93` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. The fraction is an allocator limit;
unused VRAM is not preallocated to make utilization appear higher. Completed
epochs log available `training/gpu_0/bytes_in_use`, `peak_bytes_in_use`,
`bytes_reserved` and related allocator statistics to JSONL and W&B. `nvidia-smi`
also includes the allocator pool, GPU context and display allocations, so it
need not equal live tensor bytes.

The first restarted 8,192-environment batch reached **30,826 MiB total VRAM**
(about 30.1 GiB) at **100% GPU utilization**. Samples at 21:20:31 and 21:20:53 UTC
drew 498.91 W and 511.46 W respectively. By the latter sample, rollout/optimizer
progress had advanced to 262,144 new transitions. This is actual larger-batch
training with dynamic allocation, rather than an increase in preallocated space.

The first completed epoch at 524,288 new transitions reported 10,460,624,640
live allocator bytes (9.74 GiB), a peak of 18,874,545,152 bytes (17.58 GiB), and
a retained pool of 31,343,487,232 bytes (29.19 GiB). Therefore the 30.1 GiB
device-level number includes reusable allocator capacity and context memory;
it is not 30.1 GiB of simultaneously live tensors. W&B's remote API confirmed
the uploaded epoch loss and cumulative `global_step=2162688`. The first epoch's
3,016.83 transitions/second includes compilation and is not steady throughput.
The second completed epoch, at 1,048,576 new transitions, measured **11,295.32
transitions/second** with the same live-memory peak and pool size. Subsequent
rollout counters continued advancing; no scheduled stop or test cutoff was set.

Dense family counts are capacity estimates pending their actual PPO execution.
Native MJX state sizes vary from about 0.293 MiB per environment on light scenes
to 9.44 MiB in dense furniture. A separate compile-only profile of one batched
physics step helped distinguish physical buffers from PPO memory; it performed
no simulation steps or learning updates. It is not a full dense PPO peak test.

Restart, old-configuration compatibility, source integrity, offsets, ownership,
failure retention and subsequent handoff checks passed 32 focused orchestration
tests. Two CPU PPO stop-control checks also passed. No performance evaluation
was launched. The smaller source run remains preserved; the new run retires
its owned imported copy after a verified successful stage handoff.

To request a saved stop from the Mac:

```bash
ssh -J tl-server-0 dep-0 'touch /home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_large/STOP'
```

The active trainer finishes its current checkpoint epoch, saves/exports the
selected model and exits. Closing SSH or the Mac terminal does not stop it.
