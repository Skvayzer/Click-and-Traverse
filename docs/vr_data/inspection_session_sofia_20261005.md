# VR hand-protection dataset: first inspection (`session_sofia_20261005-2059`, inspected 2026-11-13)

Source: Google Drive `session_sofia_20261005-2059.tar.gz` (257 MB) and https://github.com/Gamershmidt/custom-teleop-room.
Working copy and analysis scripts on dep-1: /tmp/claude-1004/-home-konstantinsmirnov/b9355b37-92cb-4f44-ae4b-ebe6675dc864/scratchpad/vr_data (analyze.py, analyze_pinch.py, stats.json).

**Verdict:** clean, well documented, full scene geometry -- but a tiny pilot: 24 takes, 5.3 min, one operator, one scene
family. Enough for a first human-vs-G1 comparison in narrow passages; far too little for a motion prior.

## Contents
- 24 takes = 4 scenes x 2 segments x 3 takes; 318 s total (4.2-40.7 s, median 8.9 s); 23,216 samples, ~49-79 Hz
  (81 gaps > 0.1 s, max 0.91 s). Per take: motion.npz, meta.json, pov.mp4 (headset view, 640x480 ~13.7 fps).
- Tracked: head + wrists (6-DoF 4x4; wrists = handheld controllers + fixed offset; untracked rows are all ZEROS, not
  NaN), PICO 24-joint body skeleton (ankle trackers drive legs; pelvis/spine/shoulders/elbows estimated). No fingers.
- Per-sample G1-proxy clearances (Dex3 envelope, forearm, torso), nearest object, tracked/glitch flags.
- Frames: XR frame (OpenXR y-up, metres, human scale); meta.T_scene_from_xr maps to scene frame (z-up, G1 scale),
  similarity alpha = 1.32/1.80 = 0.733. body_joints_xr is in ALPHABETICAL order (meta.body_joint_names), not SMPL.

## Scenes
- Geometry saved with each take (oriented boxes with categories, start/goal/route, a G1 certificate labelling every
  5 cm of route as free / arms / side, e.g. {kind: side, gap 0.435 m}, {kind: arms, gap 0.603 m}).
- One family only: `narrow-junk-train-0000{16,17,20,23}` (generator mtc-narrow-chaotic-v1, schema
  cat-furniture-scene-v1). Same schema/intent as our side-gap and forward-protected passages, but these scene ids are not
  in our training bank. No table-edge, shelf/aisle or furniture-room scenes; the one tabletop is > 1 m off the route.

## Quality
- NaN only in untracked rows; rotations orthonormal; body tracking 100%. All 24 takes reach the goal.
- 20/24 flagged safe; 4 have 20-28% glitch samples on one hand (one with a 9 mm hand penetration). Raw controller jumps
  up to 144 m/s; after a glitch filter three 7-8 m/s jumps remain.
- Plausible: head p5-p95 1.56-1.65 m (1.80 m operator), ankles 7-17 cm. Slow: 0.11-0.65 m/s; long takes 35-66% standing.

## Behaviour (G1 scale, medians)
| route label | shoulder yaw vs route | wrist spread across route | Dex3 clearance | wrist above shoulder |
|---|---|---|---|---|
| free | 19 deg | 0.19 m | 0.35 m | 10% |
| arms | 18 deg | 0.21 m | 0.23 m | 10% |
| side | 41 deg | 0.10 m | 0.24 m | 18% |
- `side` pinches (21 passes, 0.43-0.47 m): turned >= 45 deg only 7 times; 14 near-frontal (5 with hands together/raised).
- `arms` pinches (15 passes, 0.60-0.64 m): always frontal, hands never raised, spread like free walking -- no distinct tuck.
- Min hand clearance at pinches: median 0.135 m, lowest 0.039 m. No wrist within 10 cm of a tabletop.

## Use for the G1 / gaps
- Usable now: scene-frame keypoints (head, wrists, ankles, PICO elbows/pelvis) to compare with robot rollouts in the
  same scenes (needs these 4 scenes added to our bank). For the AMP prior: needs whole-body IK to 29 joints first.
- Scaling: uniform alpha leaves eyes ~8 cm / pelvis ~7 cm below the G1's; scaled forearm 0.23-0.24 m vs G1 0.184 m;
  shoulders 0.29 m apart vs 0.20 m.
1. Coverage: 1 person, 5.3 min, 1 family, no tables (the capture README plans ~1,047 takes over 14 families).
2. "safe" is not G1-feasible: the live check has no upper arms / legs / margin; two-thirds of side pinches were passed
   frontally where the G1 must go sideways. Retargeted motion must be re-checked with the full G1 collision model.
3. Arms weakly observed: controllers (grip pose, no fingers), inferred elbows/shoulders; mask zero rows and gaps.
