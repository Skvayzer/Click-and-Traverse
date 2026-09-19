# SAPG learner move to dep-1 — 19 September 2026

The user requested moving the running hand-protection specialist from
`tl-server-0` to `dep-1`, preserving the recent SAPG numerical repair.

## Stopped source and exact continuation

Source Slurm job **689** stopped cleanly with exit `0:0`, after saving
**451,805,184 physical transitions**. Its STOP marker remains in place.
The complete learner checkpoint is **4,712,088,489 bytes**, SHA-256
`77dcdc2985b619c507c6776f8165f09c5ccf1de51cb90547ac7d5c201eadfabd`.
The original source run and its best checkpoint remain on the server.

Runtime source remains pinned to
`351e336a079a68c93b39a40c8087c8da1fed56e4`, including the log-domain SAPG
surrogate repair in `4d36846`. Its source fingerprint is
`bd762e7cea773d042bd4aeaf4443cf6f75be7b0b53f801bc715e6cac19b1d301`.
The continuation uses the same [W&B run f017f302](https://wandb.ai/skvayzer/CAT-wholebody/runs/f017f302),
36,864 environments, six policy groups, original optimizer settings, 24-scene
hand-specialist bank and curriculum. No original-checkpoint reinitialization,
evaluation, retention rollback or observation/action change is part of this move.

## Destination preparation

The destination is `konstantinsmirnov@dep-1`, system hostname `ws008090`.
Its existing dirty checkout and stopped PPO run are preserved. Training code is
in a separate detached worktree at
`outputs/sources/cat_sapg_numerics_351e336a079a` under
`/home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody`.
Keep that worktree's tracked robot assets intact; specialist-bank paths are
absolute, so no original RandObs collection or released native checkpoint is
needed for full-state continuation.

Python 3.12.9, JAX/jaxlib 0.4.38, Flax 0.10.4, Optax 0.2.5, MuJoCo/MJX 3.3.1,
Brax 0.12.3, Orbax 0.11.5, W&B 0.24.2 and the CUDA Python packages match the
source environment. The destination has an idle RTX 6000 Ada and approximately
78 GiB available system RAM. Source job maximum RSS was approximately 35 GiB.
No environment rebuild or driver change is required. The existing destination
driver is 550.163.01; the source driver is 580.178.04. Destination PPO already
used the identical installed JAX/CUDA stack successfully.

Initial destination free disk space was **14,223,540,224 bytes**. Preserve space
for both the full runtime and its next atomic replacement. Compressed transfer
parts are temporary and can be removed after verification and extraction.
The older PPO checkpoint is retained.

## Transfer and identity checks

Direct office-LAN connections between the hosts timed out in both directions.
To respect the instruction against large Tailscale transfers, the stopped run
and 338 MB specialist bank are compressed and staged in a **private** GitHub
repository, `Skvayzer/cat-training-transfer`, as a draft release
`cat-sapg-dep1-20260919`. Each compressed part has a SHA-256 in
`transfer-manifest.json`; the complete compressed stream has SHA-256
`ceaab1f740981e5b4d9b15ad674abdac0c45202742fcbf523cc1adfe84ed34f4`.
The two parts total **1,887,809,834 bytes**. Checkpoint/data traffic travels via
authenticated GitHub HTTPS; SSH over Tailscale is used only for control and
small metadata. Original W&B local cache files are excluded; online history,
run identity, local metrics and all learner/checkpoint files are retained.

`scripts/migrate_cat_host.py` changes only the four operational filesystem
paths (`bank_manifest`, `body_collision_bank`, `body_collision_resets`, `stop`)
and the corresponding launch/status specification hashes. It appends an audited
`host_migrations` entry and preserves historical warm-start/code provenance.
The complete `resume.msgpack` is never rewritten; its checksum must match the
stopped source before and after migration. The copied STOP remains until the
migration finishes. Existing strict source, scene-bank and runtime contract
checks remain enabled.

`scripts/resume_cat_sapg.py` validates the detached fixed source and reconstructs
the exact saved training arguments before invoking the ordinary launcher with
`--resume`. It provides `--print-command` for preflight validation and sets no
GPU, optimization or episode parameters. **26 focused tests passed** for host
migration and this portable resume entry.

## Completed transfer and destination launch

On 19 September at **12:25:27 Dubai time**, the transfer pipeline launched the
continuous learner on `dep-1` in tmux session **`cat-sapg-hand-20260919`**.
Its initial PID is **631122**. The complete source checkpoint passed SHA-256
verification after extraction and remained byte-identical throughout the host
metadata migration. All three scene/collision/reset manifest hashes match the
source. W&B resumed `f017f302`, with only the four operational paths and the
audited `host_migrations` field updated; every other remote configuration field
was checked unchanged.

The destination's temporary compressed parts were removed after verification.
Free space after transfer was **9,897,377,792 bytes**, leaving room for the next
atomic full-runtime replacement. The stopped PPO run and its best/full
checkpoints were retained. Source Slurm job 689 remains completed, its run is
stopped, and its STOP marker remains in place.

Destination control/log locations, relative to the project directory:

- Run: `outputs/cat_hand_specialist_sapg_680`.
- Training log: `outputs/transfer_dep1_20260919/training-dep1.log`.
- Transfer verification: `outputs/transfer_dep1_20260919/transfer-verified.json`.
- Host migration journal:
  `outputs/cat_hand_specialist_sapg_680/host-migrations/tl-server-0-to-dep-1-451805184/migration.json`.
- Reviewed migration/resume tools extracted from commit
  `a4020c561ff0fd7435040147fdd4448cbab45918`:
  `outputs/migration_tools_a4020c561ff0/scripts`.

Training uses the destination's existing `.venv`, 16 OpenMP/MKL threads,
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and memory fraction `0.90`. `GLI_PATH`
is unset, so canonical robot assets come from the pinned checkout. There is no
training-step cap or automatic evaluation/recovery job.

The destination completed and durably saved updates at **452,984,832** and
**454,164,480 transitions**. Both reported finite losses and
`health/nonfinite=0`; the live W&B API independently confirmed the latter step
in the original run. The second update reported approximately **41,775 physical
transitions/second** during its timed learner work. NVIDIA reported **33,892
MiB** used. The source remains stopped. The full checkpoint, rather than the
older selected best leader, is the state being continued.

See [the setup and performance audit](SAPG_RESUME_AUDIT_20260919.md) for the
comparison of source/destination assets, state restoration and metric meaning.
