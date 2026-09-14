# Continuous CAT training on dep-0 — September 15, 2026

Launched at 01:02:47 Dubai time (September 14, 21:02:47 UTC) on dep-0's RTX
5090. This is the requested ongoing training run, with no global step, scene or
time cap. It stops on an explicit manual request or an error. The run initializes
from the pinned public CAT generalist checkpoint and uses the fixed Dex3 hands
and whole-body policy described in [the training plan](TRAINING_READINESS.md).

- Source: immutable worktree at commit `02a01c6b2331c206703f13097106c2c1fe2b20df`.
- Run directory: `/home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_live`.
- Log: the same absolute run path with `.log` appended.
- W&B project: [CAT-furniture](https://wandb.ai/skvayzer/CAT-furniture).
- W&B group: `continuous_cat_dex3_20260915`; each scene creates a run with a cumulative `global_step` axis.
- First scene: [r000000-p0-cat-forward](https://wandb.ai/skvayzer/CAT-furniture/runs/hkxmcix9).
- Detached supervisor PID at launch: `757162`; the worker PID changes between scenes.

The batches contain 1,024 environments for typical CAT and simple clutter, 512
for random CAT, and 256 for dense clutter. The sequence uses 50% original CAT,
25% generic clutter and 25% furniture by transition count, switching scenes every
1,048,576 transitions and considering a checkpoint every 65,536 transitions.
Only the selected generation is retained, with the predecessor retired after a
verified successful handoff. Selection uses within-scene training reward; these
are not held-out performance measurements.

After first-step compilation, repeated samples at 21:05:56–21:06:17 UTC showed
99% GPU utilization, 416–438 W and approximately 5,059 MiB allocated. PPO
reported about 10,300 transitions/second on the original CAT forward scene.
W&B's remote API independently confirmed uploaded optimizer metrics and
`global_step=573440`. These observations establish actual updates and online
logging; they do not establish successful traversal. At that early snapshot,
the completed-episode success window was still zero. Compilation and scene
handoffs reduce utilization between stretches of GPU computation; dense-scene
throughput and memory must be assessed separately.

The first forward scene required roughly 142 seconds of main compilation,
besides startup/reset, versus about 102 seconds of steady compute per million
transitions. Thus 99% is active-compute utilization, not average utilization over
scene changes. The persistent JAX cache contains reset executables; the current
host telemetry callbacks prevent caching the main training epoch, and changing
scene constants/shapes further limits reuse. Moving telemetry outside the
compiled epoch and reusing compatible compiled scene functions are future
throughput improvements. Filling unused VRAM alone is not a reason to enlarge a
batch that already saturates the GPU.

At approximately 21:07:32 UTC the first scene completed all 1,048,576
transitions, exported its selected checkpoint and automatically handed off to
[r000000-p0-generic](https://wandb.ai/skvayzer/CAT-furniture/runs/vupl9qrt).
The next worker restored the preceding checkpoint and connected to W&B with
`global_step_offset=1048576`. The supervisor remained active throughout.

The process survives closing the Mac terminal or dropping SSH. To request a
saved stop, run this on the Mac (the jump host avoids the unreliable direct
connection):

```bash
ssh -J tl-server-0 dep-0 'touch /home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_live/STOP'
```

The trainer completes its current checkpoint epoch, exports the selected model
and exits. No stop request was issued during launch verification. The run's
`continuous_state.json`, `runner.json`, stage `metrics.jsonl` and `wandb.json`
provide the current state and active scene URL.
