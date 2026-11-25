# Improve-overnight videos (2026-11-25)

COMBO_FIX seed 0 (`stationary_hands_20261113_COMBO_FIX`, u150) at 0.15 m/s, one episode per scene, recorded with CPU MuJoCo
(`scripts/record_plan_reflex.py --device cpu`, contact checked once per control step, so outcomes can differ from the GPU
benchmark) and rendered with software EGL. `no-reference-*`: the same network with no plan input (`--mix none=1`, the I4
variant, best on the 384 scenes at 0.15 m/s for seed 0). `planner-reflex-*`: the deployed plan-on mode. Scenes: one per
obstacle type among the 384 evaluation scenes where the no-reference mode was clean in both GPU episodes (fixed random
choice, `scripts/improve_local/video_scenes.json`); a second side-gap style had no such scene and was replaced by the
published side-step course. 1280×720 H.264 yuv420p, faststart, silent AAC track.

| # | Obstacle type | Scene | No reference | Planner–reflex |
| ---: | --- | --- | --- | --- |
| 0 | side gap, plain | `sidegap2-0326-plain-w0.51` | clean, 6.4 s | clean, 8.5 s |
| 1 | cabinet, narrow | `contrast-train-20260939-r0-w0.713733-narrow-7f7bbce976f1` | clean, 30.6 s | clean, 48.0 s |
| 2 | cabinet, forward protected | `contrast-train-20260961-r0-w0.589923-forward_protected-ac6cb02ce61e` | clean, 29.4 s | clean, 51.2 s |
| 3 | furniture room | `random-furniture-pilot-train-20261276-0d12d2fa4b90` | clean, 14.4 s | clean, 28.4 s |
| 4 | clutter room | `random-generic_clutter-dense-train-20261145-0ff90da39d8e` | clean, 38.6 s | clean, 59.6 s |
| 5 | CAT D8 course | `D8G0L1O0S3` | clean, 5.7 s | clean, 9.1 s |
| 6 | procedural CAT course | `procedural-D6G0L7O1S2001` | clean, 17.2 s | clean, 13.7 s |
| 7 | published hurdle | `published-hurdle0` | clean, 6.2 s | clean, 7.9 s |
| 8 | published side hurdle | `published-side-hurdle3` | contact, 2.2 s | contact, 3.5 s |
| 9 | published side step | `published-side4` | clean, 8.3 s | clean, 10.5 s |
Planner–reflex clips are capped at 40 s (cabinets, clutter room).
