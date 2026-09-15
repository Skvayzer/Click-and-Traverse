# Compact CAT restart on ws008090 — 2026-09-16

The new continuous production run was launched on **ws008090 at 02:44:36 Asia/Dubai on 2026-09-16** (2026-09-15 22:44:36 UTC), with **16,384 parallel environments and batch size 256**. The initial process is PID **64320**. Its single online W&B run is [30a1f02d](https://wandb.ai/skvayzer/CAT-wholebody/runs/30a1f02d). Startup verification passed, and the learner was left running continuously.

The previous 406-input run was stopped cooperatively at 2,739,929,088 completed transitions. Its checkpoint is preserved for historical evaluation and is not an initialization source for this experiment.

## Initialization and fine-tuning

The only initialization source is the original released CAT generalist: `logs_v1/generalist_v1/checkpoints/005033164800`, pinned to model revision `46ce4b57ba0639168d51741b661ff62f7ce6f045`. Each cached checkpoint file is checked against the committed size/SHA-256 manifest. Actor, critic and existing observation statistics are mapped by feature/action name; Adam starts fresh. Disposable GPU checks do not write checkpoints or initialize W&B.

The actor has **222 inputs**, the critic **310**, and the policy controls **29 body joints**. Added input weights start at zero. The original 12 leg outputs are preserved; the 17 new outputs start with zero mean and standard deviation 0.05. Forward parity at the mapping boundary was exactly zero for the preserved actor outputs and critic on the validation samples. Changed hand-sphere measurements intentionally mean that physical trajectories need not match the original policy.

Fine-tuning uses learning rate **3e-5**, PPO clip **0.1**, and entropy coefficient **0.003**. A frozen copy of the initial adapted CAT actor supplies a **0.05 × KL(reference || current)** penalty on the original 12 leg actions, on CAT-scene transitions only. Room transitions and the new action outputs receive no direct reference penalty. Both policies consume the same compact observations. The reference is persisted with the full learner state and never re-anchored to a later trained model. These settings reduce policy drift; they do not establish retention of hurdle or narrow-passage skills without evaluation.

## Environment collection

All **2,338 scenes** are loaded: 37 original checkpoint slots, 27 published typical layouts, 2,226 native procedural CAT scenes, 24 independently randomized furniture rooms, and 24 generic clutter rooms. Reset group probabilities are respectively 20% for original/published, 40% for generated CAT, 25% furniture and 15% generic clutter. These are probabilities at reset, not proportions of transitions.

The bank occupies **13.239 GiB**. It retains CAT's field-based obstacle detection and flat-floor/feet-contact physics; furniture is not rigid obstacle collision geometry in the training simulator. See [environment coverage and limitations](CAT_PROCEDURAL_DIVERSITY_20260916.md) and [the compact observation/hand contract](CAT_COMPACT_SPHERE_SETUP_20260916.md).

For ws008090, distinguish the embedded canonical manifest fingerprint from the hash of the complete JSON file:

- Embedded fingerprint: `7fb43ee27324057f5a026eab05d837d380e208a45637a1f6e584208b1fa10967`.
- Complete manifest file SHA-256, used by learner resume identity: `dc97e6e19c7767c5d5b567f5abdde0a4fb8a85873893be22adff7863e1b8f6c0`.

## GPU capacity measurements

The GPU is an NVIDIA RTX 6000 Ada Generation with 49,140 MiB reported total memory. Checks execute the native PPO learner against the entire bank, with released-checkpoint initialization, the frozen-reference loss, and production training-metric computations. W&B and checkpoint writing are disabled during these disposable checks. Initial updates include JIT compilation and cannot establish steady training speed.

| Parallel environments | Batch size | Transitions/update | Peak device memory | Minimum free memory | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4,096 | 256 | 524,288 | 37,562 MiB | 11,071 MiB | One full update passed; all metrics finite |
| 8,192 | 256 | 524,288 | 37,562 MiB | 11,071 MiB | Two full updates passed; warm update 56,199 transitions/s |
| 16,384 | 256 | 524,288 | 37,562 MiB | 11,071 MiB | Two full updates passed; warm update 54,959 transitions/s; selected |
| 32,768 | 512 | 1,048,576 | 37,536 MiB before failure | 11,097 MiB before failure | Failed allocation before a completed update |

Peak live JAX allocations were 21.15 GiB at 4,096 environments, 23.01 GiB at 8,192 and **26.98 GiB at 16,384**. Their larger device-memory footprints include the retained allocator pool. See the [4,096 report](assets/cat-diversity-20260916/gpu-capacity-4096-256-ws008090.json), [8,192 report](assets/cat-diversity-20260916/gpu-capacity-8192-256-ws008090.json), and [16,384 report](assets/cat-diversity-20260916/gpu-capacity-16384-256-ws008090.json).

The 8,192 and 16,384 profiles had the same PPO batch geometry: 524,288 transitions and 256 optimizer steps per update. Their measured warm updates took approximately 9.35 and 9.56 seconds including callback overhead, respectively. This is only one warm observation per profile, not a statistically established speed ranking. The 2.2% difference is small; 16,384 was selected to honor the preference for more parallel environments with comparable throughput and more than 10 GiB physical reserve. These checks exclude production checkpoint writes and online W&B transmission.

All 18 GPU utilization samples during the 8,192-environment warm update were 98–100%. Its compute workload already saturates this GPU. The [32,768-environment failure report](assets/cat-diversity-20260916/gpu-capacity-32768-512-ws008090.json) records a failed 13,000,002,312-byte temporary allocation, with 30,649,344,768 bytes live and a 46,914,801,040-byte allocator limit. Aggregate bytes alone therefore do not predict successful allocation: pool layout and the available growth allowance matter. The failed disposable check created neither a W&B experiment nor a checkpoint.

This does not establish that 32,768 environments are impossible under every allocator configuration. Preallocating the same 92% pool could reduce fragmentation, as described in the [JAX memory documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html), but that variant is untested. Production retains the allocator settings used in the accepted capacity checks instead of relying on this extrapolation.

**24,576 environments are plausible but untested.** This would add 8,192 environments and require batch size 384 to satisfy the native batch geometry, yielding 786,432 transitions/update. Approximately 10.8 GiB free on the current profile does not establish that its larger contiguous allocations would succeed. The running overnight configuration remains 16,384 environments; the 24K estimate is not a validated capacity or speed claim.

## Source, storage and operation

The production checkout is frozen at `71f9a1cee1ef86847b5ad18baeb53a341efbd666`, under `outputs/sources/cat_compact_diversity_gentle_20260916`. The source digest over the launcher and `cat_ppo` Python files is `2d93221bdcf3688d023a9f243de9d821fe558dd67cdd4bf3c58f593756d971d6`. Later capacity-script or report commits do not change this training source.

The run directory is `/home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody/outputs/cat_compact_diversity_gentle_20260916`. Production uses one online W&B ID for every environment family, one selected best model, and one atomically overwritten `resume.msgpack`. Best-model selection uses training reward as a proxy, not held-out evaluation. A replacement temporarily needs space for both old and new files. The run has no timestep stop limit; a `STOP` file requests a clean stop after the current complete PPO update.

The [launch record](assets/cat-diversity-20260916/production-launch-ws008090.json) contains the exact detached command, environment, source digest, bank digest, capacity-report digest and initial PID. The initial launch has no `--resume` flag. The process has its own session and does not depend on an open SSH connection or this Codex turn.

The W&B API independently confirmed the live configuration, original released model revision, fresh optimizer and zero feature-mapping parity errors. It also confirmed that this is the only project run created in the production launch window. The initial full recovery snapshot was written successfully before the first optimizer update.

At **02:51:36 Asia/Dubai**, the local [startup verification](assets/cat-diversity-20260916/production-startup-verification-ws008090.json) recorded 2,097,152 durably saved transitions and 2,621,440 observed/logged transitions while the next snapshot was being written. All recorded metrics were finite, with zero nonfinite-health events. Independent W&B API checks confirmed actual uploaded optimizer histories and exactly one running experiment. This establishes successful startup and logging, not learned traversal or retained-skill quality.

Production warm PPO updates measured **52,619–54,659 transitions/s**, excluding checkpoint and logging work between updates. Device memory remained **37,562 MiB used**, with **11,071 MiB free**. A 60-second post-startup sample averaged **65.4% GPU utilization including pauses**, while compute phases reached 100%. Thus this is not a claim of continuous 100% utilization: synchronous recovery/checkpoint and logging work adds overhead. Increasing environment count alone does not remove those pauses. The running overnight job was left unchanged after verification.

The full resume file was approximately **2.70 GB**, exactly one best generation was retained, and free filesystem capacity remained **22.48 GB** at capture time. The initial compile is included only in the first update's timing; it should not be used to estimate steady training speed.

To stop cleanly:

```bash
ssh konstantinsmirnov@ws008090 'touch /home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody/outputs/cat_compact_diversity_gentle_20260916/STOP'
```

Host readiness checks found 23.57 GB free before launch and verified frozen imports, virtualenv/data symlinks, and robot assets. Full scene diagnostics can emit 7,014 scalar keys, approximately 0.28–0.38 MB JSON plus 0.30 MB console text per update. At one update per 10 seconds, 12 hours produces roughly 1.2–1.7 GB JSON and 1.3 GB console text, before W&B history overhead. `WANDB_CONSOLE=off` avoids a second console copy while retaining scalar logging. Logs are outside the initially empty run directory.
