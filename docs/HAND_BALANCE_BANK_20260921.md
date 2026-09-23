# Certified hand-objective balance successor

Built on CPU; no training/evaluation launched and no checkpoint or live run files written.

Bank: `data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json`, with matching
`cat_flat_hand_balance_v2_20260921_collision/manifest.json` and
`cat_flat_hand_balance_v2_20260921_resets/manifest.json`.
All manifests load and validate. Independent measured results are in
[the audit](assets/hand-balance-bank-20260921/audit.json).

| Measurement | Original bank | Successor |
|---|---:|---:|
| Scenes | 2351 | 2375 |
| Open / protected / narrow / transition | 0 / 0 / 12 / 0 | 0 / 12 / 12 / 12 |
| Contrast zones | 36 | 108 |
| Zones with hand_active | 0 | 60 |
| Zones with hand_active and valid target regions | 0 | 60 |
| Zones with positive forward/heading weight | 0 | 60 |

Protected is stored as `forward_protected`. Every original scene record is unchanged;
24 certified width-bank scenes at rungs 0–2 are appended. Narrow zones still have
zero hand activity, no valid target regions, and zero heading weight. No open scenes
were added because the requested missing hand-region objective is present in protected
and transition scenes; retained CAT/clutter/flat tasks already supply substantial ordinary exposure.

Fixed reset masses: CAT 54% (18% original, 36% procedural), ordinary clutter 9%
(5.625% furniture, 3.375% generic), narrow 4.5%, flat 22.5%, protected 7%, transition 3%.
Each old category retains 90% of its previous share. The new 10% allocation concentrates
on protected posture practice, with less transition exposure because it combines objectives.
These are design choices, not measured performance claims. Flat reward settings and source
scene semantics are preserved. Tests exercise the actual runtime probability method with
nonuniform adaptive weights and confirm these category masses.

Across all three directories: 1,119,024,280 logical bytes (1.042173 GiB),
478,411,776 allocated file bytes (0.445556 GiB), including shared hardlinks.
Files with no external hardlinks occupy 33,242,112 bytes (0.030959 GiB).
Existing retained fields remain referenced; this is not a standalone copy of that dataset.
Allocated bytes do not include filesystem metadata and are not a measurement of shared-pool
free-space delta. Final observed free pool space: 9,513,336,832 bytes.

The reset array is [2375,32,36]. All 76,000 poses exactly match the pinned source rows
for identical scenes; all per-scene geometry hashes agree. No pose was transferred to a
different scene or invented. Existing certifications are preserved, not replaced by a claim
of new dynamic certification. Source passages explicitly mark dynamic feasibility unvalidated.

Implementation references (repository-relative file:line):

- `cat_ppo/furniture/hand_balance_bank.py:5`: explicit schema, masses, pinned composition validation and sampler.
- `cat_ppo/furniture/balance_bank.py:16` and `:54`: versioned validation/sampling dispatch preserving v1 semantics.
- `scripts/build_flat_balance_bank.py:89`: legacy coverage reporting and explicit `--hand-objectives` builder selection.
- `scripts/build_hand_balance_bank.py:23`: CPU builder, source pins, scene-specific geometry/reset composition and build preflight.
- `cat_mjlab/scene_bank.py:83`: derive contrast roles from all records instead of the fixed v1 array.
- `cat_ppo/furniture/contrast_preflight.py:8`: hash-check actual scene files, emit counts, reject missing hand or heading coverage.
- `cat_mjlab/runner.py:379`: automatic startup preflight before simulation import/construction.
- `train_cat_mjlab.py:29`: explicit `--require-hand-contrast` also rejects banks with no contrast scenes.
- `tests/test_hand_balance_bank.py:20`: zero-coverage rejection, pre-simulation startup failure, runtime masses, mutation rejection.
- `scripts/audit_hand_balance_bank.py:16`: independent scene/reset/geometry audit.
- `configs/pilots/flat_hand_balance_v2_20260921.sh:7`: exact successor command, prepared only.

Guard demonstrations in [preflight.json](assets/hand-balance-bank-20260921/preflight.json):
old bank exits 1 with `Hand-contrast training requested but active hand/heading coverage is zero`;
new bank exits 0. Automatic inference also rejects the old bank without the explicit flag.
Seven CPU tests passed, with the unrelated compiled test deselected;
see [tests.txt](assets/hand-balance-bank-20260921/tests.txt). Shell syntax and diff whitespace checks passed.

Reproduce the independent audit:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv-mjlab/bin/python scripts/audit_hand_balance_bank.py
```

Exact successor launch, after GPU availability:

```bash
bash configs/pilots/flat_hand_balance_v2_20260921.sh
```

The script warm-starts the current PPO run's `best.pt` with a fresh optimizer, retains flat
bonus 10 and region scale 0.40, and uses 30,720 environments to leave more room for the added
fields. Successor GPU memory fit and learning performance are unverified; it was not launched.
The checkpoint path resolves to the best checkpoint present at launch time.

Safety verification: all 2,435 files in the old bank's three directories have identical
SHA256, size and modification time compared with the pre-build snapshot. Hardlink counts
can change because identical immutable files are shared. The live training metrics file grew
from 2,491,332 to 2,501,929 bytes during the work. No stop/kill/tmux mutation or GPU workload
was issued. Host processes are hidden and tmux socket access is denied in this sandbox;
paired-evaluation process health is therefore unverified. Existing evaluation artifacts were
only read. Temporary audit/build/test files were removed after preserving the small reports.

No disagreement with the root-cause diagnosis. The scope remains structural objective
coverage and certified scene composition; nonzero training gradients during actual rollouts
and improved hand behavior require subsequent evaluation.
