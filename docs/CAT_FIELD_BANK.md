# CAT generalist field bank

The corrected learner mixes all 37 scene slots from the released generalist
configuration with two additional dense clutter scenes in one process. Each
environment samples a scene at reset. CAT's adaptive sampling statistics and
initial equal scene weights apply across the complete bank; the two added
scenes initially receive 2/39 of resets in total.

## Original assets and the missing scene

The public dataset contains **36 of the 37 configured scenes**. Their `sdf.npy`,
`bf.npy`, and `gf.npy` files are downloaded without rewriting their bytes and
verified against the dataset's SHA256 LFS object identities.

The remaining configured scene, `D8G2L3O2S13`, is absent from the public dataset.
It was also unavailable in the accessible workstation files. Preparation
requires an explicit `--reconstruct-missing-original` option to fill that slot
using the original random obstacle generator with difficulty 0.8, seed 13,
2 ground obstacles, 3 obstacles on each lateral side, and 2 overhead obstacles,
followed by CAT's original field pipeline.

This reconstruction **does not recover the missing original bytes**. As a
cross-check, regenerating a known released scene, `D8G2L3O2S37`, with the current
upstream generator produced different arrays. The manifest records 36 verified
originals and one reconstruction separately; that exception remains visible in
the training configuration. Recovering the missing released asset later should
produce a new bank with new provenance, rather than overwrite an experiment's
bank.

Source pins:

| Source | Immutable revision / fingerprint |
| --- | --- |
| CAT code | `866ba392f1c1e84b92ad75fa66550f26e8af8e48` |
| Hugging Face dataset `Axian12138/Click-and-Traverse` | `db8c202fa724bc4d3dba2cb9fead1267f15fd811` |
| Dataset paths | `assets_v0/RandObs/<scene>/{sdf,bf,gf}.npy` |
| Released model repository revision | `46ce4b57ba0639168d51741b661ff62f7ce6f045` |
| Released configuration SHA256 | `0e38cac262a3c1c95bacdf859193340056fef53b8d2f8414293789cee5d45d2d` |

The original field arrays have shape `75 × 50 × 38`, resolution 0.04 m, and
runtime origin `[-0.5, -1.0, 0.0]`. Original runtime sampling is preserved,
including its corner ordering, clipping at grid boundaries, and half-cell
coordinate convention. These details affect the pretrained policy's inputs.

## Added dense scenes

The prepared seed `20260915` adds:

- A 9.0 × 9.69 m furniture room with 9 tables, 36 chairs, 298 primitive boxes,
  and six narrow passages.
- The same route topology filled with 45 generic objects: 12 crates,
  11 low blocks, 11 partitions, and 11 shelves.

Both fields use 0.04 m resolution and have shape `230 × 248 × 60`. Boxes are
rasterized at voxel centers, then passed through the existing upstream
`make_sdf`, `grad3`, and `make_guidance_field_progressive` functions. Guidance is
three-dimensional fast marching toward the selected goal. The generator's
route is used only to check the room geometry; it does not supply guidance to
the policy. The obstacle SDF excludes the floor.

These additional rooms use their admitted start location with ±0.08 m XY
randomization. CAT's original scene reset distribution remains unchanged.
Yaw, body state, gains, pushes, and other task randomization remain controlled
by the CAT task. The smaller XY range is explicit because applying CAT's ±1 m
XY range to a cluttered room entrance can place the robot in a wall or table.

The room checks establish a connected free-space voxel path and a geometric
route for a small root cylinder. They do not prove full-body clearance at every
randomized reset or successful dynamic traversal. Training retains CAT's
field-based obstacle representation; the separate physical evaluation scene
contains rigid furniture geometry.

## Storage and preparation

The complete bank contains 339,299,376 bytes of field files (about 324 MiB).
Ragged storage concatenates the original and room grids without padding all 37
small fields to room size. Per-scene offsets, dimensions, origins and clipping
bounds select the correct samples. Tests compare its JIT-compiled interpolation
bit-for-bit with CAT on interior positions and clipped boundaries.

```bash
.venv/bin/python prepare_cat_generalist.py \
  --released-config configs/cat_generalist_released.json \
  --output data/furniture/cat_generalist \
  --download --reconstruct-missing-original

.venv/bin/python prepare_cat_generalist.py \
  --validate data/furniture/cat_generalist/manifest.json
```

Preparation verifies existing files and refuses to replace an existing bank
with a different configuration. The manifest uses relative paths so the whole
directory can be copied to another machine. `bank_config(manifest_path)` returns
the `pf_config` settings, including `bank_manifest`, consumed by the whole-body
generalist environment. Neither teachers nor specialist checkpoints are needed
to load these fields for PPO fine-tuning of the released final generalist.
