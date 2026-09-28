---
title: CAT Whole-Body — Project State and Direction
date: 2026-09-28
tags: [robotics, humanoid, unitree-g1, mjlab, ppo, cat, hand-protection, motion-prior]
status: living document
authors: Konstantin Smirnov (project), Claude (write-up)
---

# CAT Whole-Body — Project State and Direction

> [!abstract] One paragraph
> We extend **Click-and-Traverse (CAT / HumanoidPF)** — a goal-conditioned whole-body navigation policy for the Unitree G1 — into a policy that (1) **protects its hands and fingers** (fixed Dex3 grippers) by moving them away from obstacles and approaching objects, standing or walking; (2) **traverses narrow passages** (0.40–0.74 m) and table-edge passages; and (3) **keeps CAT's original navigation ability**. Training is PPO on 40,960 parallel MuJoCo Warp environments (mjlab), warm-started from the released CAT generalist, on a procedurally generated scene bank whose obstacle geometry is delivered to the policy as **potential fields** (guidance direction, boundary direction, signed distance) sampled at 13 body points. This page records the architecture, the RL setup as of 2026-09-28, measured results, open problems, and the direction suggested by *Moving Through Clutter* (MTC, arXiv 2609.21107).

**Companion fact sheet (auto-extracted from the repository):** [[_facts_20260928]]

---

## 1. Goals and priorities

| # | goal | status 2026-09-28 |
|---|---|---|
| 1 | Hands/fingers move away from obstacles and approaching objects, standing or moving | Mechanism in place and measured; behaviour present but not yet human-like (no tuck/raise) |
| 2 | Narrow-passage traversal (sidling through 0.40 m gaps) | 30–35 % zone success; was 53 % on the warm-start checkpoint before a reward change that is now reverted |
| 3 | Preserve CAT navigation | ~12–13 % goal success in training windows on CAT scenes (stable, not eroding); released CAT measured higher on a deterministic paired eval → still open |

---

## 2. System architecture

```
                 scene bank (2,887 scenes, packed fields)          reactive object bank (2,000 rows)
                 ┌───────────────┐  gf/bf/sdf int16/f32           ┌──────────────────────┐
                 │ CAT scenes    │  sampled at 13 body points     │ shapes, sizes, speed │
                 │ rooms/clutter │──────────────┐                 │ bucket: danger/antic/│
                 │ passages      │              ▼                 │ negative             │
                 │ table edges   │      ┌─────────────────┐       └──────────┬───────────┘
                 └───────────────┘      │  CATTask (mjlab)│◄─────────────────┘ analytic objects
                                        │  MuJoCo Warp    │   merged into the hand fields
                                        │  40,960 envs    │
                                        └────────┬────────┘
                              obs 226 (actor) / 310 (critic)   rewards (≈30 terms)
                                                 ▼
                                        ┌─────────────────┐
                                        │ PPO (asymmetric)│ actor MLP 226→[512,256,128,64]→29
                                        │ fresh optimiser │ critic MLP 310→[1024,512,256,128]→1
                                        └────────┬────────┘
                                                 ▼
                                  checkpoints, wandb, per-scene-type metrics
```

### 2.1 Robot and simulation
- **Unitree G1**, 29 actuated joints (12 legs, 3 waist, 2×7 arms), **fixed Dex3 grippers** (finger joints welded). PD position control at 50 Hz control / higher-rate physics substeps.
- **mjlab / MuJoCo Warp** on one RTX 6000 Ada (48 GB). 40,960 envs at ~120–145 s per PPO update (32-step unroll → 1.3 M env-steps per update).
- The training model is assembled at runtime (`cat_mjlab/model.py`): base MJCF + Dex3 hands + query spheres. Physical contact is limited to explicit pairs (feet–floor, feet–feet/shins, **and since 2026-09-24 the approved hand sphere (r = 0.1034 m) against both thigh and both shin capsules**, so fingers can no longer pass through the legs).
- Environment geometry does **not** produce MuJoCo contacts: obstacles live in precomputed fields and a voxel body-collision bank; collisions are detected geometrically and end episodes.

### 2.2 Perception: potential fields instead of maps
For every scene the bank stores three volumetric fields on a grid (dx = 4 cm): **gf** guidance direction (where each body part should move), **bf** boundary direction (to the nearest obstacle), **sdf** signed distance. Directions are octahedrally packed to 2×int16 and decoded only for the 8 gather corners (this halved VRAM and is what allows 40,960 envs). The policy samples them at 11 body sites (head, pelvis, torso, 2 feet, 2 hands, 2 knees, 2 shoulders) plus 2 elbows.

**Actor observation (226):** proprioception (gyro, gravity, joint pos/vel, last action, PD targets), the delayed command, gait phase, the *delayed* field samples, elbow fields, and since 2026-09-24 the **rate of change of the hand/elbow distance samples** (4 dims, added as a function-preserving expansion with zero-initialised weights). **Critic (310):** true (undelayed) fields plus privileged keypoint positions/velocities.

> [!note] Why the rate features
> CAT's actor saw one delayed frame of the fields; only the critic saw keypoint velocities. An actor cannot tell an object closing at 0.5 m/s from one parked 20 cm away without a rate signal.

### 2.3 Scene bank (`data/furniture/table_edges_v1_packed`)
Hash-pinned, extensible bank; the chain is *CAT flat/hand-balance bank → narrow widths → procedural rooms → table edges*, each extension a pinned addition fragment. Reporting **buckets** (what is failing) are decoupled from **sampling groups** (what is drawn):

| bucket | count | notes |
|---|---|---|
| procedural CAT | 2,226 | CAT's generator, 1000-step episodes |
| original / published CAT | 37 + 27 | authored, incl. crouch/hurdle/side modules |
| clutter rooms (dense/pilot/legacy) | ~290 | knee-height boxes, 4000-step episodes |
| furniture rooms | ~104 | |
| passages: narrow / forward-protected / transition / open | 62 / 62 / 62 / 50 | contrastive cabinet corridors, 0.40–0.74 m |
| **table edges** (new) | 32 | 16 matched open / protected pairs, 70–72 cm tables |
| flat standing scene | 1 | background for reactive episodes |

Reactive bank: 2,000 rows of object parameters (sphere/box/cylinder/rod; speed, hold, clearance), buckets *danger* (aimed at the hand) / *anticipation* / *negative* (passes by). Since 2026-09-24 rows supply **parameters only**: episodes reset from the ordinary pose and the object is aimed at the **live** hand.

### 2.4 Experience distribution (adaptive balancer)
Reset masses are re-solved every 5 updates from **realised** episode lengths so each sampling group's share of *transitions* tracks its target (`--experience-masses`), and the reactive standing coin is solved jointly (`--experience-reactive-share 0.08`). Rationale: share = reset mass × episode length; rooms end in ~450 steps not 4000, CAT in ~500, standing episodes never early — a static solve drifted CAT from 66 % to 30 % of samples on 2026-09-24.

---

## 3. RL setup

### 3.1 Algorithm
Asymmetric-critic PPO, 40 minibatches of 1,024 trajectories × 32 steps, GAE with truncation masking, reward floor `clamp(Σ r·dt, 0, 10000)` (so negative terms cannot drive the sum below zero — every new term is designed as a **bounded bonus or bounded cost** for this reason). Warm start from a native checkpoint with a fresh optimiser; observation widening for the rate features is function-preserving.

### 3.2 Reward (active terms, grouped) — exact weights in [[_facts_20260928]]
- **Navigation (CAT-native):** field following for feet/head/hands (`*gf`), distance shaping (`*df`), root-velocity tracking, orientation (roll/pitch), gait/foot contact/clearance/slip/balance, joint limits/torque/smoothness.
- **Hand protection:** `wholebody_hand_clearance` (target 9 cm, anticipation 20 cm) and `handsdf` near-field, `arm_clearance` for elbows; **terminations** on hand/body collision.
- **Posture (2026-09-24):** `upright` bonus (torso pitch within ~10°), `stand_tall` bonus (head near 1.20 m), `torso_rate` cost — all multiplied by *not-crouch-required*, where crouch is required when the head guidance field points down or something is within 20 cm of the head. This closes the loophole where `tracking_orientation` ignored pitch below a 1.1 m head.
- **Fingers (2026-09-24):** `self_clearance` — hand envelope spheres vs own thigh/shin capsules, quadratic ramp from 4 cm; plus the physical contact pairs and a `hand_self_contact` episode flag.
- **Heading (2026-09-24):** `heading_align` — bonus for facing the guidance direction, gated on whether a forward-facing body would fit (probes at ±0.16 m, here and one stride ahead); off in gaps under ~0.42 m so sidling is never penalised.
- **Standing (2026-09-25):** `stand_still` (root speed/yaw rate while commanded to stand), standing bonus paid only under 0.05 m/s, `spot_hold` cost on distance from the standing spot.
- **Reactive events (opt-in):** ±w per finished approach (handled / unhandled threatening object).

### 3.3 Standing-object (reactive) task — redesigned 2026-09-24
Episode = 800 steps on the flat scene, a **stream of approaches**: object re-armed after each retreat with a 0.5–2 s pause, parameters re-drawn per approach, approach ray chosen among 8 candidates to be clearest of the body (so it meets the hand, not the thigh). **Event success** = root stayed on its spot, no contact, targeted hand retreated ≥ 5 cm along the approach axis relative to the root (or did not flinch for a *negative* object). A **walking variant** (30 %) keeps the route command and aims at the predicted hand position.

> [!bug] Root cause found 2026-09-25
> `update_phase` rewrites `move=1` at zero speed and the stance correction only ran under a speed curriculum, so standing episodes had been commanding "walk at zero velocity" throughout; the standing bonus never paid. Fixed: standing reactive envs get a real stance.

### 3.4 Metrics (wandb, per update)
`scene/<type>/{success_rate, completion_rate, episodes_ended, fall_rate, hand_self_contact_rate, mean_episode_length, torso_pitch_abs}` for 17 buckets; `progress/p1_*` (event success, left-spot rate, stance speed), `p2_narrow_*`, `p3_cat_navigation_success`; `balance/*_experience_share` vs `_target`; `reactive/*`; `posture/*`; `self_clearance/*`; `reward_term/*`, `reward_share/*` for every term; `health/*`.

### 3.5 Tooling
- `scripts/record_mjlab_rollout.py` — deterministic/stochastic single-scene recordings on CPU while training holds the GPU (`--reactive-row`, `--allow-source-mismatch`, `--allow-bank-mismatch`); `render_clutter_rollouts.py` draws the moving objects.
- `scripts/measure_self_clearance.py` — ground-truth hand/finger vs leg distances and posture from recordings (MuJoCo geom distance).
- Bank tooling: `generate_table_edges.py`, `pack_scene_bank.py`, `build_body_collision_bank.py`, `build_body_collision_resets.py`.
- Launch config: `configs/pilots/bundle_20260924.sh` (single source of truth for the current run).

---

## 4. Measured state (runs of 2026-09-24/25)

| metric (windowed) | warm-start ckpt | pilot 1 (u380) | pilot 3 (u111, 2026-09-28) |
|---|---|---|---|
| narrow passage zone | 52.8 % | 34.9 % (+6.6/100 upd) | 36.4 % (replay-clean 47.8 %) |
| transition | 65.0 % | 52.0 % | 45.5 % |
| protected passage | 68.9 % | 59.9 % | 53.3 % |
| clutter rooms | 57–78 % | 79.7 % | 78.4 % |
| table edges (per-update, n≈17) | — | 31 % | 20 % |
| CAT navigation | 12.3 % | 12.1 % | 11.3 % (CAT starved to 19 % of samples, see §4.1) |
| any goal | 27.4 % | 19.8 % | 28.3 % |
| reactive: threatening object handled | — | 83–84 % | 84–85 % (negative-bucket calm 18 %) |
| reactive: root left spot | — | 79 % | **29 %**; stance speed 0.113 → 0.109 m/s |
| torso pitch, clutter rooms | 30–50° | 30–50° | **2°** (weight ×3); CAT scenes 7°, table 7° |
| finger–leg penetration (steps) | 8.5 % | 0.1 % | ~0.1 % |
| seconds / update | 158 | 122 | 120 |

Recordings (11 scenes, same seeds) live in `outputs/videos_20260925/videos/`.

### 4.1 Balancer defect found in pilot 3 (fixed 2026-09-28, pilot 4)
The group length estimate was the mean length of episodes that *ended* in the window. For long-episode groups those are the early failures: rooms were estimated at ~430 steps while their episodes ran 1,400–1,800, so rooms drew 47 % of samples against a 20 % target and CAT fell to 19 % against 51 %. Replaced by **Little's law** (env-steps spent in the group ÷ episodes ended), which counts in-flight episodes, with a ×2/÷2 clamp per re-solve. Pilot 4 restarted from pilot 3's checkpoint with this estimator; `balance/*_experience_share` vs `_target` is the check.

> [!info] Run ledger
> pilot 1 `cat_bundle_20260924` (u380, 09-24/25) → pilot 2 `cat_bundle2_20260925` (u17, hand objective reverted, stance fixes) → pilot 3 `cat_bundle3_20260925` (u111, posture ×3; **stopped by request at 17:24 on 09-25**, GPU idle until 09-28) → pilot 4 `cat_bundle4_20260928` (launched 14:07 on 09-28 with the Little's-law balancer).

> [!warning] Lessons that cost days
> - A metric from a 24-pair eval was repeated as fact for days; the 3,200-pair eval reversed it. Check the file before quoting a memory.
> - `pgrep -f` matches the shell that runs it. Use PIDs.
> - Two overrides of the same quantity one line apart (reactive horizon 800 vs manifest 4000) silently picked the wrong one for a whole run.
> - The balancer's first re-solve used survivorship-biased lengths and flooded the batch with standing episodes for 150 updates.

---

## 5. Open problems

1. **Human-likeness of hand avoidance.** Clearance and outward-normal velocity terms make the hand *leave*, not *tuck* or *raise*. The posture prior is gated off exactly when avoiding.
2. **Narrow gaps at 30–35 %** (53 % before the −60 clearance weight; reverted to −20 + `handsdf` on 2026-09-25 — confirm recovery).
3. **Rooms hunch** (30–50° while stepping over clutter) — the CAT high-step gait; posture bonuses tripled in pilot 3.
4. **CAT retention** unresolved against the released generalist (paired deterministic eval pending on the current checkpoint).
5. **Reactive stance**: creeping under the stillness gate → fixed in pilot 2/3, verify `reactive/stance_speed_mean_m_s`.

---

## 6. Direction: what MTC changes for us

*Moving Through Clutter* (Wang et al., 2026): VR demonstrations in procedurally generated clutter with **embodiment scaling** (scene and keypoints scaled by h_robot/h_operator), a **scene-aware trajectory-level retargeter** (SAMR: Laplacian posture + alignment + smoothness + log-sum-exp margin, hard capsule-vs-OBB clearance constraints, 80-frame windows), RL **tracking teachers** distilled by DAgger into a voxel-observing student with **perceptual hallucination**. Results vs CAT: side-walk 97.7 % vs 49.8 %, crawl 81 % vs 0 %, but duck 84 % vs 99 % and step-over 92.5 % vs 100 %.

**What we adopt, and what we keep:**

| MTC idea | use in our project | status |
|---|---|---|
| Human motion as the *what* of posture (tuck, raise, side-step) | Upper-body **imitation/posture-library term** active under clearance pressure; later **AMP-style upper-body style reward** | proposed; datasets being surveyed |
| Embodiment-scaled VR capture in *our* procedural rooms | Own capture of protective/tucking motions (nothing public contains them) | future; hardware: headset + 3 trackers |
| Scene-aware retargeting with hard clearance | Turn CIRCLE/TRUMANS/AMASS clips into collision-free G1 references inside our scenes | depends on tool release (OmniRetarget/GMR as fallback) |
| Perceptual hallucination (jitter obstacles around a fixed behaviour) | Augmentation for our field observations | cheap, planned |
| Teacher–student per skill | Only for skills a human must show (side-step, tuck); the RL objective stays the outer loop | optional |
| Pure imitation instead of task RL | **Not adopted**: no reactive objects, no hand-safety objective, weaker than CAT on ducking/stepping | — |

### 6.1 Motion-prior data — what is actually available (survey verified 2026-09-28)

| dataset | what it is | usable for | access / licence |
|---|---|---|---|
| **MTC** (2609.21107) | 32 h VR clutter traversal, G1 refs | sidle, duck, crawl, step-over | **not released** (paper promises; no URL, no repo) |
| **PASSAGE** (2609.18732) | 100 h VR + IMU suit, 1,500 corridors | same | **not released** ("code coming soon") |
| **PHUMA** | ~70 h physics-corrected G1 locomotion (`root_trans/root_ori/dof_pos(29)`, 30 fps) | whole-body / arm style prior; crouch & step "vertical" class (pelvis > 0.6 m, no crawl) | HF `DAVIAN-Robotics/PHUMA`, Apache-2.0, 3.4 GB |
| **OmniRetarget dataset** | 4.0 h G1 qpos (36/43-D): 623 unique box-carry clips ×aug (3.0 h), 29 climb clips, 22 chair-carry/climb | **carry posture** (arms in front) prior; terrain | HF, MIT |
| Unitree **LAFAN1-G1** | 41 CSVs walk/run/dance/fight/fall (the LAFAN1 *obstacles* theme was not retargeted) | general style only | CC BY-NC-ND (research only) |
| **AMASS** (curated clips) | CMU 107_11, 108_21–26 duck/stoop/crawl-under; 127_29–38 run-and-duck; 141_32/33, 143_40 walk sideways; KIT "walking around obstacle objects" (60) and "around a moving obstacle" (158); SFU vault/crawl; BMLmovi crawl; BABEL labels duck/crouch/crawl/sidestep | per-behaviour references (tens of short clips) | MPI non-commercial; retarget with GMR |
| **CIRCLE** (Stanford) | 10 h, 7,228 clips of *reaching* in one furnished HSSD apartment, SMPL-X 120 fps + scene glb; median root path 0.3–0.6 m (not locomotion); ~1 h bend/crouch/kneel reaches, ~1 h reaching onto/under tables; physics off (8–11 cm cumulative interpenetration); no finger data | reach-low/high **posture library**, scene-aware retargeting possible | S3, no registration, CC BY-NC 4.0 |
| **TRUMANS** | 15 h, 100 scenes (71 released), SMPL-X + scene/object meshes + occupancy grids; only 10 action labels; unlabelled walking between furniture (CHIP obtained G1 gap-sidestepping after RL post-training on it) | implicit furniture navigation | gated Google Form, non-commercial |
| **Nymeria / NymeriaPlus** | 300 h egocentric daily life (XSens 17-IMU → SMPL-X in Plus); no room meshes | ~22 h furniture-heavy scenarios, must be keyword-mined | gated, CC BY-NC 4.0, ~80 TB total (filterable) |

**Tools:** GMR (MIT; SMPL-X/BVH/FBX/PICO-live/video → G1 29-DoF, 35–70 fps CPU, no scene handling — 31 % collision-free on clutter per MTC), OmniRetarget code in `amazon-far/holosoma` (Apache-2.0; per-frame SQP with SDF collision-pair hard constraints, object URDFs, box terrains) and `project-instinct/omniretargeting` (MIT; arbitrary terrain meshes), PHUMA/PhySINK (Apache-2.0). **No public SAMR-style space-time capsule-vs-OBB retargeter exists.** Nothing public contains a protective hand retreat or a hand raised past a table edge.

**How much data a prior needs (evidence):** AMP-style style rewards: 10–434 s (AMP), 42.6 s on a full-size humanoid (HumanMimic), 4.5 s (Escontrela); per-clip tracking teachers: one clip per behaviour (DeepMimic 1–3 s, BeyondMimic 25–40 clips of minutes, 2.5 h total); latent-skill priors: 15–40 min (PARC 14 min, T-GMP 30 min, ASE/CALM ~30 min); scene-conditioned generative planners: 25–100 h (PASSAGE: 6 h 48.1 % → 24 h 57.8 % → 48 h 64.3 % → 100 h 68.9 %). VR capture: PICO 4 Ultra full-body tracking (MTC v1: 348 trajectories, 2.3 h, ~24 s each, GMR), PASSAGE: Noitom PN Link suit + headset, colliding takes rejected via haptics, ~18.6 s per sequence.

Sequencing: (1) posture-library term with authored targets + comfort costs (no data) → (2) replace targets with human references from released datasets (OmniRetarget carry motions, CIRCLE reaching in clutter, retargeted) → (3) own embodiment-scaled VR capture for protective retreat and gap tucking → (4) AMP upper-body style reward on the union.

---

## 7. Runbook

```bash
# stop / start
CHECKPOINT=outputs/<run>/resume.pt bash configs/pilots/bundle_20260924.sh
# record the current checkpoint on a scene (CPU, while training runs)
.venv-mjlab/bin/python scripts/record_mjlab_rollout.py --allow-source-mismatch --checkpoint <ckpt> \
  --bank-manifest data/furniture/table_edges_v1_packed/manifest.json \
  --body-collision-bank data/furniture/table_edges_v1_collision/manifest.json \
  --body-collision-resets data/furniture/table_edges_v1_resets/manifest.json \
  --scene-id <scene_id> --frames 4000 --device cpu --output-dir outputs/rec/<name>
# reactive episode: add --reactive-bank data/furniture/reactive_standing_v2_20260923/manifest.json --reactive-row N
# render:  scripts/render_clutter_rollouts.py --input-dir outputs/rec/<name> --output <mp4>
# ground truth self-clearance / posture: scripts/measure_self_clearance.py outputs/rec/<name>
```

Transfer to laptop: `scp -r konstantinsmirnov@dep-1:/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/outputs/videos_20260925/videos ~/Downloads/`

---

## 8. Timeline (condensed)
- 2026-09-16…23 — CAT reproduction in mjlab; collision banks; hand-clearance objective; contrastive passages; reactive standing bank v2; packed fields; extensible banks.
- 2026-09-24 — Distribution diagnosis (one scene at 37.8 % of experience; then standing at 64 %); adaptive balancer; heading bonus; posture terms; physical hand–leg pairs and self-clearance; reactive redesign; table-edge family; 19-scene baseline recordings; pilot 1.
- 2026-09-25 — Pilot 1 analysis (balancer transient, stance bug, hand-weight regression); pilots 2 and 3; per-scene posture metrics.
- 2026-09-28 — MTC review; motion-prior data survey; this page.
