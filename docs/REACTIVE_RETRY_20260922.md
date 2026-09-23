# Reactive standing retry: stopped by declining disk headroom

The eight-object observation-query cost gate is cleared by the user-provided
external RTX 6000 Ada measurements. The earlier hypothetical 20 ms budget is
withdrawn as an acceptance criterion. GPU invisibility inside the sandbox is
not evidence of host GPU absence.

`cat_mjlab/runner.py` measures update elapsed time across collection and learning.
The last training log line gives 101.05985484 seconds / 32 steps per environment
= 3.15812046 seconds per batched control step, amortized over learning. The
benchmark already contains four observation passes: true/actor body and arm
queries. Do not multiply those timings by four again. No rollout/learning split
was logged. No training was run to obtain one.

See `assets/reactive-analytic-20260922/gpu-cost-user-evidence.json` for attributed
GPU numbers and arithmetic. Eight objects extrapolate to 26.110875 ms per step,
0.835548 seconds per update (0.826786%), and 787923763 peak incremental bytes
at 37,632 environments, plus persistent state. Linear scaling is an estimate.
The completed runtime would additionally require full-body guard/collision at
ten physics substeps and continuous arm capsules; those are not in this timing.
Maximum benchmarked affordable observation count: eight. A larger maximum and
complete environment overhead are unmeasured. GPU access remains unavailable
through this sandbox; no independently measured GPU result is claimed.

## Disk stop and preserved draft

Free disk declined from 2,804,416,512 bytes (2.612 GiB) to 967,442,432
(0.901 GiB), 852,885,504 (0.794 GiB), then 717,619,200 (0.668 GiB).
This work wrote only small source/patch/report files. The shared-pool loss
exceeds 2 GiB and is external to these artifacts. Artifact generation stopped.

`cat_mjlab/reactive.py` is an UNVALIDATED draft: object-only clearance-limited
substep motion, continuous capsule queries with an explicit numerical bound,
conservative whole-body proxy covers, and contact attribution. The capsule
solver repeatedly evaluates closed-form primitive SDFs; it is not a fully
closed-form segment-cylinder solver. Swept robot contact currently bounds
linear sphere-center chords, not rotational arcs; this must be resolved before
certifying arbitrary robot motion. No safety or production-readiness claim.

`assets/reactive-analytic-20260922/standing-integration-draft.patch` preserves
the untested reset, command, collision, telemetry, checkpoint and CLI integration.
It is NOT applied. Only this turn's integration changes were reversed; existing
workspace edits were preserved. Draft syntax compiled before isolation, but
no behavior tests or scene certification were run in this retry.

Standing bank, scene certification, preflight, corrected-below/bucket renders,
completed-path GPU benchmark, and usable launch command remain undelivered.
Walking/turning was dropped first. No training was launched. Existing banks,
checkpoints, reward weights, and acceptance gate were not modified.
