# Counterfactual obstacle synthesis (COS) — log and pre-registrations

Branch `feature/cos-obstacles` (from `feature/fullpath-plan` cd40fee). Proposal:
`~/.local/share/openresearch/files/humanoid-protection/amass-obstacles/counterfactual-obstacle-synthesis.md`.
Log copy after each gate: `~/.local/share/openresearch/files/humanoid-protection/amass-obstacles/cos_obstacles.md`.
Large outputs: cluster `/l/users/konstantin.smirnov/cos/`.

## Summary

(written at the end of the night)

---

## Data

`library_v1.{npz,json}` of the CAT style-prior work (`Click-and-Traverse-Mjlab/data/motion_library`): 3,622 clips, 864,160
frames at 50 Hz (4.8 h), MuJoCo qpos of the G1 training model. Sources: 938 obstacle-selected AMASS clips (CMU, KIT,
BMLmovi, SFU) retargeted with GMR; a 2,484-chunk PHUMA subset (G1-native, everyday categories); 200 OmniRetarget box-carry
clips. All heights below are of the approved 35-primitive collision proxy surface (`cat_mjlab/cos/features.py`).

Calibration on the 122,154 frames the library tags `walk`: head/trunk top (arms excluded) median **1.297 m**; lowest
point of the hand spheres median 0.493 m; lateral arm half-width (torso frame) 5th percentile 0.241 m.

## G0 — event audit (pre-registered 2026-11-26 01:06, before the first count)

**Hypothesis.** The library contains enough deviation events to seed COS for most families: duck, sideways, detour and
arms-raised have ≥ 20 events each; crawl and high step are rare (the proposal's coverage risk).

**Detectors** (`cat_mjlab/cos/events.py`, 50 Hz; speed = pelvis xy speed smoothed 0.5 s; path direction = pelvis
displacement over ±0.5 s; "crawlish" = pelvis < 0.50 m or a hand < 0.12 m):

| family | frame test | min duration |
|---|---|---|
| duck | head/trunk top < 1.297 − 0.15 m, speed > 0.15 m/s, not crawlish | 0.3 s |
| crawl | pelvis < 0.50 m and (a hand or a knee joint < 0.12 m), speed > 0.08 m/s | 0.5 s |
| high step | one sole lifts > 0.05 m and peaks at 0.20–0.60 m while the other sole is < 0.10 m at the apex, lands ≥ 0.25 m further along the path, body moving | one lift |
| sideways | torso yaw vs path direction in 50–130°, speed > 0.20 m/s, not crawlish | 0.5 s |
| arms raised | highest hand-sphere bottom > max(0.72, 0.493 + 0.20) m, moving, not crawlish | 0.5 s |
| arms tucked | lateral arm half-width < 0.241 m, moving, not crawlish | 0.5 s |
| detour | 3 s window, chord ≥ 1.2 m, one-sided max deviation off the chord ≥ 0.30 m, heading change < 45°, path/chord < 1.5 | merged windows |

**Metric.** Events, distinct takes (PHUMA chunks of one sequence and OmniRetarget augmentations = one take) and seconds,
per family and per source.

**Pass rule.** A family is *trainable* if it has ≥ 20 events from ≥ 10 distinct takes. G0 passes if ≥ 4 of the 7 families
are trainable. For each family below the bar, name a source (LAFAN1 obstacles/ground takes, a bigger PHUMA pull).
Visual spot check: render 3 example clips of each family (kinematic) and report detector false positives found.

**Amendment 01:15 (after the first spot check, before any G1/G2 use).** A contact sheet of 4 random raw events per family
showed about half were wrong-reason motions (OmniRetarget box pick-ups as "duck", weapon lunges, kicks, dancing, push
recovery, kneeling down in place). Added a **clean tier** (`events.clean`): drop OmniRetarget (box manipulation), clip
names matching fight/kick/box/dance/jump/weapon/push/pick/sit/... and CMU subjects that are dance, breakdance, acrobatics or
getting up (5, 49, 55, 60, 61, 85, 87–90, 93, 94, 131, 140); require ≥ 0.5 m pelvis travel over the event ± 0.5 s (crawl
0.3 m); duck mean speed ≥ 0.25 m/s; sideways torso yaw change < 45° within the event; high step: pelvis ≥ 0.20 m forward
during the lift, apex ≤ 0.50 m, no running (< 1.6 m/s), foot under/ahead of the pelvis at the apex, no kick-out-and-back
(> 0.10 m overshoot). The pass rule is applied to the clean tier.

### G0 result (01:25) — PASS (7 of 7 families trainable on the clean tier)

| family | raw events / takes | **clean events / takes** | clean seconds | clean by source (events/takes) |
|---|---|---|---|---|
| duck | 544 / 332 | **106 / 83** | 102 | CMU 52/36, KIT 16/12, BMLmovi 15/13, PHUMA 23/22 |
| crawl | 326 / 217 | **125 / 99** | 491 | BMLmovi 92/70, CMU 16/15, KIT 13/12, SFU 4/2 |
| high step | 111 / 52 | **24 / 13** | 15 | CMU 8/6, PHUMA 16/7 |
| sideways | 1134 / 652 | **368 / 225** | 720 | BMLmovi 173/73, PHUMA 126/110, CMU 53/33, KIT 13/8 |
| arms raised | 1191 / 712 | **444 / 345** | 765 | PHUMA 351/280, CMU 84/61 |
| arms tucked | 230 / 173 | **122 / 100** | 129 | PHUMA 116/94 |
| detour | 24 / 23 | **21 / 20** | 65 | PHUMA 12/12, KIT 5/5, CMU 4/3 |

Spot check of the clean tier (4–5 random events per family, rendered): sideways (BMLmovi side-stepping) and crawl
(BMLmovi/KIT crawl takes) look genuine; duck still includes some deep lunges and getting-up-from-crawl phases (geometry
is still a real height drop, the *reason* is not always ducking); high step is mostly PHUMA knee-high marching and CMU 143
step-overs — the clean filter removed running strides and breakdance. **Thin families:** high step (24 events, 13 takes) and
detour (21). Source proposal: LAFAN1 `obstacles1–6` takes (step-overs, ducking, detours; public BVH, GMR has a LAFAN1
loader) and a wider PHUMA pull; not fetched yet (G2 runs first on CPU).
Artifacts: `amass-obstacles/g0/` (audit log, events JSON, spot-check sheets).
