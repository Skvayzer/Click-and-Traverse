# Dependency repair, 2026-09-22

Rebuilt v3 faithfully from v2, including collision and reset companions. All 2,579 restored files match SHA-256 values in the pre-deletion snapshot `outputs/raised_region_v4_20260921/source_before.json` (see `historical-match.json:2`). The restored field manifest hash is `f9978fa010792a9c00975800c4b2c7ca1dd7f49c318c435c70b310d385860802`, exactly the existing v5/v6 ancestor pin. No successor manifests or validators were changed.

## Dependency diagnosis

`cat_ppo/furniture/balanced_region_bank.py:26` checks the v3 manifest hash, then lines 38–43 read and hash v3 protected scene JSON and compare the successor with the specified geometry transformation. Thus this is executable integrity evidence, not optional provenance. `cat_ppo/furniture/protected_heavy_bank.py:13` validates v6 through v5 and requires the mixture-only transformation. The inherited v3 pin is also retained in v6.

Runtime fields resolve locally for contrast/flat scenes and to the existing retention root otherwise (`cat_ppo/furniture/generalist_fields.py:413`). v5/v6 do not fetch their runtime arrays from v3; their validation does require v3 protected scene metadata. The restored complete v3 bank also supports its historical consumers. Its own validity transformation is verified against v2 (`cat_ppo/furniture/raised_only_bank.py:32`). Option (a) preserves the exact original contract and avoids either inventing new ancestry or weakening verification.

## Verification

`validation.json:19` and `validation.json:36` record successful full file verification (`verify_files=True`), collision loader verification, reset manifest/payload hash verification, and contrast preflight for v5 and v6. Both have 2,375 scenes, 108 zones, 60 hand-active zones, 60 active hand-objective zones, and 60 positive-heading zones. Roles are 12 narrow, 12 forward-protected, 12 transition. v3 passed the same checks (`validation.json:2`).

CPU command: `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-mjlab/bin/python docs/assets/dependency-repair-20260922/validate_repair.py`.

The exact command arguments in `configs/pilots/hand_reward_floor_v6_50.sh` were extracted and passed to the real `train_cat_mjlab.parser()`; its specified field, collision and reset banks were verified above. `bash -n` also passed. See `validate_repair.py:22` and `validation.json:53`. No shell launch, runner invocation, simulation construction, environment step, optimizer update, GPU use, or W&B startup occurred. GPU initialization, IK certification and later training startup remain UNVERIFIED; this is the requested argument/bank/preflight dry validation, not proof that all subsequent runtime initialization will succeed.

`tests/test_raised_only_bank.py`: 2 passed in 18.58 seconds, including rejection of unrelated manifest changes. Builder `--help` passed. `scripts/build_raised_only_bank.py:34` now snapshots a log only if explicitly supplied via `--live-log`; its default has no run-directory dependency. The entire builder was not rerun because the restored target exists and its overwrite guard is intentional.

## Complete deleted-name audit

`deleted-path-hits.txt` contains every emitted match with original file and line, grouped by all 11 requested directory names; `grep-counts.json` contains counts. Search used `rg --hidden --no-ignore -n -F`, including ignored data and outputs. Exclusions: Git object/administrative storage, virtual environments, Python bytecode caches, and this audit's own generated directory (to avoid recursive self-matches). Binary files use rg's default binary detection. These are substring counts, so base-bank searches also count collision/reset suffixes.

| Name | Matching lines |
|---|---:|
| cat_flat_hand_balance_v3_20260921 | 72 |
| cat_flat_hand_balance_v3_20260921_collision | 20 |
| cat_flat_hand_balance_v3_20260921_resets | 19 |
| cat_flat_hand_balance_v4_20260921 | 25 |
| cat_flat_hand_balance_v4_20260921_collision | 3 |
| cat_flat_hand_balance_v4_20260921_resets | 3 |
| cat_hand_cont_30720_20260921 | 6 |
| cat_hand_approach_r3_h2_30720_20260921 | 6 |
| cat_hand_raised_v3_pilot50_20260921 | 3 |
| cat_hand_interim_30720_20260921 | 2 |
| cat_hand_interim2_30720_20260921 | 2 |

It would be false to claim no references remain. The following other historical workflows still have missing inputs:

- `configs/pilots/hand_raised_seeded_20260921.sh:8` requires the deleted continuation run's `best.pt`. Cannot faithfully reconstruct trained weights from the available evidence; left unchanged and not run.
- `tests/test_feasible_region_bank.py:11` and `outputs/raised_region_v4_20260921/measure.py:11` load deleted v4; `outputs/raised_region_v4_20260921/pilot.sh:8` also requires v4 and its collision/reset companions. These remain absent. v4 is not in the validated v5/v6 dependency chain.
- `scripts/build_balanced_region_bank.py:35` includes v4 in its protected snapshot audit; `snapshot()` at line 21 uses rglob, so absent directories produce empty snapshots. This is a lost historical audit input, not a v5/v6 runtime dependency.
- `docs/HAND_RAISED_SEEDING_20260921.md:45` documents an evaluation command requiring the deleted v3 pilot's run.json. That historical command cannot run as written.
- `configs/pilots/hand_approach_20260921.sh:12` names the deleted approach run as an output destination, not an input. Interim/interim2 matches are historical launch logs only. Remaining matches are preserved historical records, reconstruction outputs, or restored v3 dependencies; see the complete inventory.

## Disk

Measured pool used bytes: 963,529,539,584 before; 963,533,864,960 after (+4,325,376). Available bytes: 5,498,339,328 before; 5,493,227,520 after. Shared-pool changes are not attributable solely to this operation. See `disk.json:6`.

Restoration added 80 independent JSON files: 19,859,155 logical bytes, 4,284,416 allocated bytes reported by st_blocks. The 2,499 binary files were verified to share inodes with v2, consuming no duplicated binary payload. Directory metadata and small repair reports/scripts add overhead. Thus the repair consumes real space, approximately a few MiB, not zero and not another full bank. No files were deleted. Restored binary files must continue to be treated as immutable because they are hardlinked.

Reconstruction code is retained in `restore_v3.py`; it checks the expected manifest pin before creating any bank and refuses existing targets.
