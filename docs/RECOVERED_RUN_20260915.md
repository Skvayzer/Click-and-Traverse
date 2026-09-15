# CAT restart after dense-scene OOM — September 15, 2026

The previous run stopped at 06:15 Dubai time when its first dense generic
scene requested another 4.08 GiB and exhausted GPU memory. Light-scene GPU
utilization and memory measurements had not established dense-scene capacity.
The retained `r000001-p0-cat-forward` step-8,388,608 checkpoint was verified
intact, including all 12 native files. It is the recovery source; cumulative
training progress is 102,301,696 transitions. The failed dense scene produced
no saved training progress and is replayed from that checkpoint.

The restart preserves the proven larger light-scene batches and reduces dense
batches. It uses 8,192 environments for typical CAT/simple generic, 4,096 for
simple furniture, 2,048 for random CAT, 1,024 for dense generic and 256 for dense
furniture. The old dense counts were 2,048 and 1,024 respectively. Actual full
PPO peaks on completed stages were 15.80–21.24 GiB for light scenes, 18.28 GiB for
simple furniture and 20.06 GiB for random CAT. A standalone physics-step memory
estimate was insufficient for predicting the failed full PPO allocation.

The configured counts are upper limits. Before launching each actual scene, an
empirical full-PPO memory envelope, `num_envs * (1 + 0.25 * primitive_boxes) MiB`,
caps its batch to a 25 GiB planning budget by halving. This envelope exceeds the
five observed completed-stage peaks by approximately 13–16%; it is not a formal
memory guarantee. In particular, randomized CAT layouts with 53–79 primitives
use 1,024 environments instead of retaining the 2,048 count that worked for
42 primitives. The exact decision is recorded with the scene's geometry.
See [the capacity evidence](assets/dense-capacity-restart-20260915.json).

The allocator fraction is 0.90, with preallocation disabled. This is a memory
limit, not a request to fill unused VRAM. Dense counts must be verified from
actual PPO execution; allocator live peaks, retained pool and device-level
memory remain separate metrics. No configuration can guarantee immunity from
every hardware, runtime or numerical failure.

The scene cadence stays at 8,388,608 transitions with no global time or step
cap. Checkpoint candidates are considered every 131,072 transitions (64 epochs
per scene), retaining one selected model. This shortens the interval between
recovery points and manual-stop checks without accumulating checkpoints.

Only positively identified GPU OOMs before observed training progress can
trigger automatic recovery: up to two halvings per scene, with a floor of 128
environments. Failed attempt logs and the starting checkpoint are preserved.
Other exceptions or failures after progress halt for inspection. W&B failures
are explicitly marked unsuccessful; the previous code's unconditional
`finish()` could misleadingly label a failed stage as Finished.

The recovery launched at 10:26:41 Dubai time (06:26:41 UTC), using pinned
source commit `ea299cce7abb0e6e58b221e2908de2304eaa1f27`. The supervisor PID
is `1772487` and the first trainer PID is `1772502`. The first stage uses 1,024
environments and the same 91-primitive dense generic scene that failed before.
Its actor and critic outputs exactly matched the imported checkpoint in the
warm-start parity check. The 92 focused local tests passed before launch.

The run directory is
`/home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_recovered`.
Its sibling `.log` file contains the trainer output. The run's restart anchor
records source checkpoint hashes, curriculum position and cumulative offset.
The original failed run remains preserved. W&B keeps the existing
`continuous_cat_dex3_20260915_large` group so stage records from the restart
remain with the previous training lineage on the cumulative `global_step` axis.
The recovered dense stage is [W&B run 269cr8uc](https://wandb.ai/skvayzer/CAT-furniture/runs/269cr8uc).

By 10:37 Dubai time, three complete checkpoint epochs had finished at
131,072, 262,144 and 393,216 new transitions, with further training advancing
past 475,136. No OOM retry occurred. All logged numeric metrics were finite.
The first epoch included compilation; the next two achieved 1,068.80 and
1,067.98 transitions/second. Peak live allocator usage stayed at
22,080,870,912 bytes (20.56 GiB), below its 30,332,406,185-byte (28.25 GiB)
limit. The retained allocator pool was 28.25 GiB, and a device-level sample
showed 29,863 MiB used, 2,279 MiB free, 96% GPU utilization and 497 W.
A later sample at 10:37:41 Dubai time showed the same memory use with 100% GPU
utilization and 542.54 W, while new training transitions had reached 507,904.
Retained pool memory can be reused; it is not additional live tensor usage.
These results verify actual PPO execution on the scene that previously failed,
not learned traversal performance or every future scene's memory requirement.

The first selected checkpoint at 131,072 transitions had exactly one retained
candidate and all 13 payload files matched their recorded sizes and hashes.
The original source and owned imported copy still matched all 12 source files.
W&B's API confirmed the recovered stage was running in the existing group and
receiving advancing cumulative steps (102,793,216 at the final API check).
Training was left running. See [the runtime samples](assets/recovered-runtime-20260915.json).

Request a saved stop from the Mac using the current run path:

```bash
ssh -J tl-server-0 dep-0 'touch /home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_recovered/STOP'
```

The process finishes its current checkpoint epoch, exports the selected model
and exits. Closing the terminal or disconnecting SSH does not stop training.
