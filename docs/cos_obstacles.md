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

## G1 — robot feasibility with SONIC (pre-registered 2026-11-26 01:35, after a 4-window timing test only)

**Setup.** `scripts/cos_windows.py`: every clean G0 event ± 1.5 s, merged per clip (≤ 12 s, ≥ 3 s) → **740 windows**
(57.7 min; family counts with overlap: arms raised 368, sideways 245, arms tucked 109, crawl 102, duck 85, detour 21, high
step 15). `scripts/cos_g1_track.py` runs frozen GEAR-SONIC (default release, ONNX, CPU) in the CPU MuJoCo harness of
`feature/sonic-baseline` 84a4467 (`sonic_gates.build`: flat scene, harness torque noise on, SONIC PD gains) — the robot
is reset to the window's first frame (finite-difference velocities) and SONIC follows the clip open-loop in time. SONIC
does not see the global position, so root drift is expected and does not count against the clip.

**Metric** per window, from the saved tracked/reference trajectories: end state (complete / fell: pelvis < 0.5 m while the
reference pelvis > 0.6 m / sank: pelvis > 0.25 m below the reference / harness termination); joint error (mean |Δq| per
frame, 90th percentile); local body error (proxy primitive centres in the pelvis-heading frame, mean over primitives,
90th percentile); heading drift at the end; root drift per second.

**Tracked within bounds** := complete AND joint-error p90 ≤ 0.15 rad AND local-body p90 ≤ 0.10 m AND |heading drift| ≤ 30°.

**Pass rule.** Reported per family; G1 passes for a family if ≥ 50% of its windows are tracked within bounds. Crawl is
reported separately: the harness's termination rules are left unchanged; if crawl windows end by termination or "sank",
the needed change (floor contact allowed for knees/hands/forearms) is named, not made.
Tracked trajectories replace the kinematic clips in G2 where tracked (PARC-style correction), as a second G2 variant.

## G2 — COS synthesis and validation (pre-registered 2026-11-26 01:50, after a 40-window smoke test)

**Method** (`cat_mjlab/cos/{default_motion,synth}.py`, `scripts/cos_g2_synth.py`). Per window M (740 windows; kinematic
clips first, SONIC-tracked rollouts as a second variant once G1 is done): chord frame (start pelvis at the origin, goal
on +x; windows with chord < 0.5 m are skipped). **Default motion D(M):** a stride cycle of steady straight upright human
walking (124 cycles cut from the library, 0.41–1.63 m/s; the 5 cycles nearest M's mean speed, one drawn), distance-indexed
on the station s(t) = running max of M's pelvis progress along the chord, so D passes every station at the same time
as M, upright, facing +x, with natural arm swing, on the straight chord. **Families:** duck → beam or bar on two posts
(bottom between M's top + m and min(M's top + m + slack, D's top)); crawl → low ceiling on 4 posts or tunnel with side
walls; high step → hurdle under the swing-foot apex (height ≤ M's lowest point over it − m); sideways → wall pair with
each face m … m + slack outside M's lateral extent over a 0.15–1.2 m stretch; arms tucked → the same narrow gap; arms raised
→ counter or table (top + 4 legs) beside the path on the raised hand's side, top below the raised hand − m, inner edge
m … m + slack outside M's body below the top; detour → box or column on the chord, face m … m + slack outside M on the
bulge's far side. Rejection sampling (≤ 60 candidates per event, keep ≤ 3) on every 2nd frame (25 Hz):
M clear of all boxes ≥ m, main part within m + slack of M, D penetrates the main part by > 3 cm.
Then 2–5 distractors (box, column, wall segment; clear of M and D by ≥ m, ≥ 0.1 m from the main obstacle) and two room
walls beyond both motions. m = 0.05 m, slack = 0.20 m.

**Pass rule.** Independent verification of every accepted scene at full rate (50 Hz frames plus interpolated midpoints,
100 Hz) against all boxes: M clearance ≥ m AND D touches the main obstacle (separation < 0). **G2 passes if ≥ 95% of
accepted scenes verify.** Reported per family: acceptance rate (events with ≥ 1 accepted scene / events tried),
candidate acceptance, verification rate, also with the 3 cm penetration re-checked at 100 Hz. Visual check: a contact
sheet of 24 random accepted scenes (the proposal's blind 50-scene check needs a second person; flagged, not claimed).

**Amendments before the full run (01:55).** (1) Minimum default penetration 3 cm in the search (grazing defaults are
weak counterfactuals); verification still reports the pre-registered < 0 rule and the 3 cm rule. (2) After the first full
run (high step 3/24 events accepted) the hurdle sampler was changed to place the hurdle where the floor clearance profile
under the motion (lowest body point over all frames per 2 cm x-bin) is highest within ± 0.35 m of the swing-foot apex.
Numbers below are from the second run; the first run gave 1,126/1,134 verified (99.3%).

### G2 result (kinematic clips, 02:05) — PASS: 1,112 of 1,118 accepted scenes verify (99.5%)

| family | events tried | events with ≥ 1 scene | candidate acceptance | scenes | verified at 100 Hz | D hits with |
|---|---|---|---|---|---|---|
| duck | 95 | 58 (61%) | 172 / 2,801 | 172 | 171 (99.4%) | head 157, trunk 14 |
| crawl | 61 | 26 (43%) | 75 / 2,663 | 75 | 75 (100%) | trunk 69, head 5 |
| high step | 24 | 8 (33%) | 24 / 1,008 | 24 | 24 (100%) | feet 23 |
| sideways | 196 | 118 (60%) | 330 / 6,842 | 330 | 328 (99.4%) | hands 310 |
| arms tucked | 119 | 65 (55%) | 184 / 4,608 | 184 | 183 (99.5%) | hands 162 |
| arms raised | 403 | 105 (26%) | 279 / 20,528 | 279 | 278 (99.6%) | hands 252 |
| detour | 21 | 18 (86%) | 54 / 324 | 54 | 53 (98.1%) | hands 40 |

- 145 of 740 windows were skipped for chord < 0.5 m (mostly slow crawls and in-place sidesteps), so crawl tried only 61
  of its 125 clean events.
- 1,112/1,118 also pass with the 3 cm penetration rule at 100 Hz. Median real-motion clearance 9.2 cm, median default
  penetration 7.3 cm. All 6 failures are real-motion clearances of 4.2–4.9 cm (between search frames) — fixable by
  searching at 50 Hz.
- 302 distinct clips; PHUMA 645 scenes, CMU 285, KIT 90, BMLmovi 83, SFU 15. Obstacle kinds: gap 330, narrow gap 184,
  counter 173, beam 118, table 106, bar 54, box 50, ceiling 39, tunnel 36, hurdle 24, column 4.
- **Weakness seen in the numbers:** for sideways, arms-tucked and detour scenes the default motion hits the obstacle with
  its *hands* (arm swing), not its shoulders/trunk — the wall gaps are narrower than the arm swing but often wider than
  the shoulders, so they explain "keep arms in" as much as "turn sideways". A stricter sideways rule (D's trunk must
  hit) is the obvious next step.
- Visual check (contact sheet of 24 random scenes, 8 videos): geometry is right in all of them; scenes are cluttered by
  2–5 distractors. The blind 50-scene plausibility check needs a second person — not done.

Videos (1280×720 H.264 yuv420p, faststart, poster JPEG): `amass-obstacles/videos/` 01 duck→beam (CMU 127), 02 duck→bar
(CMU 108), 03 crawl→tunnel (CMU 107), 04 high step→hurdle (CMU 143), 05 sideways→gap (PHUMA), 06 arms tucked→narrow gap
(PHUMA), 07 arms raised→counter (PHUMA), 08 detour→box (PHUMA). Contact sheet `amass-obstacles/g2/g2_contact_sheet.png`.

**G2 amendment 02:15 (requested by the user, before any use of the scenes).** A "turn sideways" scene keeps its label
only if the default motion's trunk, head or shoulder links (proxy regions trunk/head + shoulder pitch/roll capsules)
penetrate the explaining obstacle; otherwise it is relabelled **arms_in** (the gap is narrower than the arm swing only).
Result on the regenerated scenes (deterministic RNG from here on): 62 sideways kept, 271 relabelled arms_in;
1,111/1,117 scenes verify (99.5%).

### G1 result (03:00) — PASS for 5 of 7 families; duck and crawl FAIL

Amendment: the full harness (22 GB RAM and ~110 s build per process; ~20 s per window) was too slow for the data scale,
so G1 ran on `cat_mjlab/cos/lite_harness.py`: the bare training model, the same physics options, SONIC PD gains, the
actuator torque limits, per-episode torque noise rfi = U(0.5, 1.5)·0.1·limit, no latency, no episode termination.
**Validation on the 148 windows tracked both ways: the tracked flag agrees on 145 (98%)**; tracked 109 (full) vs 112
(lite), median joint p90 0.111 vs 0.108 rad, median local-body p90 3.9 vs 4.0 cm. 740 windows take 148 s on 12 processes.

| family (windows) | tracked within bounds | complete | sank | fell |
|---|---|---|---|---|
| duck (85) | **14%** | 55% | 23 | 15 |
| crawl (102) | **0%** | 0% | 101 | 1 |
| high step (15) | 60% | 87% | 2 | 0 |
| sideways (245) | 75% | 89% | 19 | 8 |
| arms raised (368) | 66% | 90% | 32 | 6 |
| arms tucked (109) | 89% | 97% | 2 | 1 |
| detour (21) | 90% | 100% | 0 | 0 |
| all (740) | 62% (458) | | | |

Crawl: the lite harness never terminates on floor contact (knees, hands and forearms may touch the floor), and the robot
still sinks > 25 cm below the reference within the first second — frozen SONIC does not track crawling; neither do deep
crouches (duck: 14%). Needed: a crawl/deep-crouch-capable tracker (fine-tuned SONIC or a dedicated tracker trained on
these clips), not a contact-rule change. Tracked flags go into the data manifest.

## Scenes v1 — full-data preparation (scope change 02:30; no planner or policy training)

**Sources.** (1) All SMPL-X sequences of the four raw AMASS subsets on dep-1 (CMU, KIT, BMLmovi, SFU; 7.9 GB archives
copied to Lustre), retargeted with GMR exactly as in September (`scripts/cos_retarget_amass.py`, 8,123 sequences in
two 12-CPU jobs). (2) If the user's download job lands: ember-lab AMASS→G1 (all of AMASS; the retargeting is fleaven's,
CC-BY-4.0 on the repo, AMASS non-commercial research terms underneath), full PHUMA (Apache-2.0), Unitree's LAFAN1 G1
retargeting (LAFAN1 CC BY-NC-ND 4.0). Fallback: the September PHUMA subset (2,484 chunks) already on Lustre.
Every source goes through `scripts/cos_library.py` (50 Hz, joint-limit check) → `cos_process.py events` (G0 clean tier)
→ `cos_feasibility.py` (SONIC, lite harness, tracked flag per window) → `cos_process.py synth`.

**Multi-fit synthesis** (`cat_mjlab/cos/kinds.py`). Per clean event up to K = 6 accepted fits (≤ 150 candidates),
cycling through several obstacle kinds per family: duck → beam / bar / shelf cantilevered from a side wall / low
ceiling (passage entrance); crawl → ceiling on posts / tunnel / table; high step → hurdle / low box; sideways →
gap / doorway (lintel) / cabinet-and-wall gap; arms tucked → narrow gap / narrow doorway; arms raised → counter /
table / counters on both sides; detour → box or column / wall stub from the side / table. Same acceptance as G2
(M clear ≥ 5 cm, explaining part within 25 cm of M, D penetrates > 3 cm). A fit is a near-duplicate of a kept one if
it has the same kind and its main part's centre and half-sizes are all within (10, 10, 5, 8, 8, 5) cm. Windows with
≥ 2 fitted events also get up to 3 multi-obstacle scenes (one random fit per event). Each scene gets 2–5
distractors and room walls and is verified at 100 Hz (G2 rule); sideways scenes without a trunk/shoulder hit are
labelled arms_in.

**Quality metrics** (`scripts/cos_report.py`), per family and source: yield (clips → clean events → events with ≥ 1
fit → scenes → verified; verified scenes per mocap hour); explanation strength = default penetration of the
explaining part; tightness = the real motion's minimum clearance to all boxes; clearance to the explaining part;
realism = dimension ranges of real objects (beam/shelf/ceiling-entrance bottom 0.8–1.6 m; crawl ceiling/table
0.35–0.95 m; hurdle/box top 0.08–0.6 m; gap width 0.30–1.2 m; counter top 0.6–1.0 m); diversity = kinds per family
and fits per event; G1 trackability = SONIC tracked flag of the window, and whether the tracked rollout itself clears
every box (tracked_clear ≥ 0).
