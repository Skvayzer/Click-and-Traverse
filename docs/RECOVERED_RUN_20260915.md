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
PPO peaks on completed stages were 17.58 GiB for light scenes, 18.28 GiB for
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

The planned run directory is
`/home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_recovered`.
Its sibling `.log` file contains the trainer output. The run's restart anchor
records source checkpoint hashes, curriculum position and cumulative offset.
The original failed run remains preserved. W&B keeps the existing
`continuous_cat_dex3_20260915_large` group so stage records from the restart
remain with the previous training lineage on the cumulative `global_step` axis.

Request a saved stop from the Mac using the current run path:

```bash
ssh -J tl-server-0 dep-0 'touch /home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_recovered/STOP'
```

The process finishes its current checkpoint epoch, exports the selected model
and exits. Closing the terminal or disconnecting SSH does not stop training.
