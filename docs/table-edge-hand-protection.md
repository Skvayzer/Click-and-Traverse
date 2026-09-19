# Table-height hand-protection passages

This extension adds **32 table scenes** (16 matched open/protected pairs) to the existing 64 tall-obstacle contrastive scenes. The combined bank has **96 scenes**. The existing tall scenes and their field arrays are retained exactly; no running experiment is changed by building the new bank.

Each room has six actual tables arranged as three pairs. Each table has a 6 cm tabletop, four 5 cm legs, and a support apron. Table heights vary between **70 and 72 cm**. Protected gaps vary between **43.0 and 44.4 cm**; matched open gaps remain **150–152 cm**. The seed changes height, gap, layout rotation and translation. Turn spaces remain between pairs. Outer baffles connect the table ends to the perimeter so the commanded central passage cannot be bypassed around their outer edges.

Unlike the tall narrow passages, these low passages admit the checked forward-facing body posture. Ordinary hand spheres intersect the table edges. Raising the hands puts their **lowest sphere surface above the tabletop**, while the spheres partly overhang the tabletop horizontally. Thus the clearance comes from lifting above the surface rather than merely pulling both hands inside the gap. The reference raised hand-sphere bottom is about 83.6 cm high, giving approximately **11.6–13.6 cm** vertical clearance. This is a static pose measurement, not achieved walking performance.

The open/protected table pair changes only gap width. The tall-obstacle bank still contains genuinely narrow passages where sideways walking is appropriate. The two geometry families therefore teach different posture choices instead of making every narrow passage require the same response.

## Training integration

The existing route-heading and hand-region objectives are reused with the same weights. Only the **raised-above-table target pair** is valid in protected table zones; the old low tucked target is disabled there. Targets retain absolute world height, so crouching does not lower the target with the robot. The nominal-arm penalty remains relaxed throughout the approach and hazard. Open scenes keep the ordinary neutral-arm incentive.

No observations, policy architecture, action dimensions, optimizer settings, or reward coefficients change: **222 actor inputs / 310 critic inputs / 29 actions**. Existing body collision checks remain active. The initial reset probability is 25% per behavior role. Within open and forward-protected roles, half the scenes are tables and half are tall obstacles; table scenes therefore initially receive 25% of resets overall. Adaptive weighting remains within each role. Success metrics are unchanged; table protected successes contribute to the existing forward-protected metric.

All scenes are checked using the same approved 35 collision shapes and conservative 4 cm SDF, along sampled routes, arm transitions, and floor-referenced crouching/stepping poses. The full swept raised-hand target regions must also clear both the SDF and tabletop height. The table-band cross-section check states its actual height explicitly: tables remain open underneath, so it is not claimed to seal every floor-level route. Ordered navigation uses projected table geometry, and runtime body checks remain mandatory. These certificates are not exhaustive motion-planning or dynamic-feasibility proofs.

## Build and inspect

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/build_table_contrastive_bank.py \
  --base-manifest data/furniture/contrastive_hand_v2/manifest.json \
  --output data/furniture/contrastive_table_v4
.venv/bin/python scripts/build_body_collision_bank.py \
  --field-manifest data/furniture/contrastive_table_v4/manifest.json \
  --output data/furniture/contrastive_table_v4_collision
JAX_PLATFORMS=cpu .venv/bin/python scripts/build_body_collision_resets.py \
  --field-manifest data/furniture/contrastive_table_v4/manifest.json \
  --collision-bank data/furniture/contrastive_table_v4_collision/manifest.json \
  --base-reset-manifest data/furniture/contrastive_hand_v2_resets/manifest.json \
  --output data/furniture/contrastive_table_v4_resets --seed 20260919
JAX_PLATFORMS=cpu MUJOCO_GL=glfw .venv/bin/python scripts/render_table_edge_passages.py
```

Use these three new manifests in the existing [training launch recipe](contrastive-hand-protection.md#reproduce-and-launch). Changing the scene bank requires a new run configuration, not an exact-state resume of the old experiment. GPU memory for the larger bank has not been measured.

[Scene examples](assets/table-edge-passages-20260919/table-edge-scenes.png) and [normal versus raised hands](assets/table-edge-passages-20260919/table-edge-hand-comparison.png) are actual MuJoCo renders of illustrative prescribed poses, not newly learned rollouts.

## Verification

83 targeted tests passed. The final bank contains 2.461 GiB of field arrays and 3,072 certified clear reset poses; all original 64 scene records and fields are unchanged. Actual CPU JIT reset/step through the training wrapper explicitly exercised table scene slots 64 and 65 with finite observations/rewards and unchanged dimensions. See [bank verification](assets/table-edge-passages-20260919/bank-verification.json) and [runtime verification](assets/table-edge-passages-20260919/runtime-verification.json). No training was started or restarted.
