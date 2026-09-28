---
title: Motion-Prior Pipeline — task RL + human motion prior
date: 2026-09-28
tags: [robotics, humanoid, unitree-g1, amp, motion-prior, vr-capture, gallant, dagger, proposal]
status: proposal (nothing implemented or trained yet)
related: [[CAT-WholeBody-Project-State]]
---

# Motion-Prior Pipeline — task RL + human motion prior

> [!abstract] Idea
> Keep the reward-trained policy as the definition of **what** to do (reach goals, avoid collisions, protect hands, stand straight). Add a **learned, context-conditioned motion prior** that defines **how** to move, learned from a small human-motion library (minutes per skill, not hours). Deploy through a **LiDAR voxel student** (Gallant-style) distilled from the field-based teacher.

> [!warning] Scope
> **Crawling is theoretical.** It is kept in the design so the prior and scene families support it later, but it will **not** be tried on the real robot: ground contact can break the fixed Dex3 fingers. Stage 3 (crawl) is simulation-only and optional.

---

## 0. Starting point
- Field-based PPO policy (CAT warm start), 40,960 envs, ~30 reward terms: navigation, hand clearance, self-clearance, posture, heading, standing, reactive events. See [[CAT-WholeBody-Project-State]].
- Good at *what*; weak at *how*: unnatural arm avoidance, no tucking or raising, awkward stepping gaits.

---

## 1. Data

### A — general human movement (default style everywhere)
| source | content | format | licence |
|---|---|---|---|
| PHUMA | ~70 h physically cleaned locomotion: walk, turn, stop, crouch, step, stand | G1 29-DoF, 30 fps | Apache-2.0 |
| OmniRetarget dataset | 4 h, mostly box carrying (arms held in front) | G1 qpos 36/43-D, 30 fps | MIT |

A few hundred clips are enough; not all 70 h.

### B — obstacle-specific human movement (retargeted AMASS / CIRCLE)
| behaviour | clips |
|---|---|
| duck / stoop / crawl under | CMU 107_11, 108_21–26, 127_29–38 |
| walk sideways | CMU 141_32/33, 143_40 |
| around obstacles | KIT: ~60 static, ~158 moving obstacle |
| crawling (sim-only) | BMLmovi (~90 subjects), CMU, SFU |
| reach low/high near furniture | CIRCLE (~1 h onto/under tables, scene mesh included) |
| search | BABEL labels: duck, crouch, crawl, sidestep |

Retarget with **GMR** (SMPL-X/BVH → G1). Clips with a scene: OmniRetarget code in `amazon-far/holosoma` (collision-aware). Licence non-commercial → fine for research, replace before any product.

### C — own targeted VR capture (what no dataset contains)
- Rig: **PICO 4 Ultra** with full-body tracking; recorded inside **our procedural scenes**; **embodiment scaling** (scene × G1 height / operator height) so motion is robot-feasible (MTC recipe).
- Shot list (~10–20 min usable each):
  1. sidle through 0.40–0.74 m gaps, arms tucked;
  2. walk past 0.65–0.80 m table edges, hands lifted;
  3. virtual object approaching the hand from 8 directions × 3 heights × 3 speeds, hand pulled back toward the body, body still;
  4. squeezing past a single side obstacle;
  5. (sim-only) stand → crawl → stand transitions.
- Total ≈ **1–1.5 h usable**, ≈ 3–6 h in the headset (colliding takes discarded).

### D — the robot's own rollouts
Negative examples for the discriminator (below).

### Processing (all sources)
1. Retarget to G1.
2. Physics check: replay with a quick tracker in MuJoCo; drop clips with joint-limit breaks, self-penetration, foot skating (PHUMA filter).
3. Label each clip: **skill** ∈ {walk, carry, sidle, duck, step-over, reach-low, protect, crawl} and **context vector** *c*: overhead clearance, lateral gap width, floor-obstacle height, near-hand object (distance, closing speed). Scene-coupled clips (CIRCLE, own capture) → from the scene; generic clips → from the skill label.
4. Pack: per-frame whole-body state + *c*, and a list of valid start states per skill.

---

## 2. How the motion prior works (AMP, from first principles)

**Why:** a safety reward like "hand ≥ 9 cm from the wall" is satisfied by *any* arm motion, including ugly ones. "Move like a human" cannot be written by hand — it is **learned**.

1. **Discriminator D** — small MLP (e.g. 1024-512). Input: a motion snippet = body state at *t* and *t+1*: joint angles and velocities, root height/orientation/velocity in the robot frame, hand and foot positions relative to the pelvis.
2. **Training D** — supervised classification alongside PPO: human snippets (A–C) → "human"; current robot rollouts (D) → "robot". Least-squares loss + gradient penalty (keeps D smooth and its reward usable).
3. **Style reward** — r_style = max(0, 1 − 0.25·(D(snippet) − 1)²), bounded in [0, 1].
4. **Policy reward** — r = r_task + w · r_style. PPO does what it always does; it now earns more by doing the task *and* moving in ways D cannot tell from human.
5. **Loop** — as the robot looks more human, D looks harder (GAN; the policy is the generator), until robot snippets are indistinguishable from human ones *in the states the task requires*.

**What AMP does not do:** replay a clip (no reference, no time index); decide *when* to use a skill (task + scene decide); need demos of *this* room (only the *kind* of motion must exist in the data) → minutes suffice.

### Context conditioning (T-GMP / CALM idea)
One plain D would call walking "human" everywhere and push the robot upright in a crawl tunnel or arms-swinging in a narrow gap. So D judges **motion given the situation**: D(snippet, *c*), with *c* = overhead clearance (head SDF/field), lateral gap (heading probes), floor-obstacle height ahead, near-hand object flag + distance + closing speed (rate features).
- Positives: human snippet + the context it was recorded in. Negatives: robot snippet + its actual context.
- **Simpler equivalent (recommended first):** one discriminator per skill group — {walk/carry}, {sidle}, {duck/step-over}, {protect}, {crawl, sim-only} — each trained on its own clips; the style reward is a context-weighted mix (e.g. crawl weight rises as overhead clearance < 0.8 m). Easier to debug.

---

## 3. Training pipeline

| stage | what | output |
|---|---|---|
| 0 | current field-based PPO policy is the base, always warm-started | — |
| 1 | build the motion library (data A + B), label, pack, viewer | ~1–2 h labelled G1 motion |
| 2 | whole-body context-conditioned style reward in open/cluttered rooms and passages | natural gait, arm swing, carry posture, sidle, duck |
| 3 | *(sim-only, optional)* crawl family: tunnels 0.45–0.75 m, start states from crawl frames, context-dependent rules | crawling in simulation |
| 4 | add own VR capture (C) as positives for their skill groups; retrain D | human tucking, raising, protective retreat |
| 5 | *(optional)* tracking teacher + DAgger imitation loss for stubborn skills | demonstration-quality execution where style alone fails |
| 6 | Gallant-style LiDAR voxel student via DAgger | deployable policy |

### Stage 2 details
- Style weight ramp 0 → ~0.5 over ~50 updates; frozen/reduced if task metrics drop.
- **Separate critics** for task/safety vs style, combined only in the advantage (CWI trick) so a large style reward cannot swamp small but crucial safety terms.
- Checks: per-scene success does not drop; posture/self-clearance improve; per-skill style score (mean D output).

### Stage 3 details (theory only)
- Crawl scenes where crawling is the only route.
- **Reference-state initialisation:** part of the crawl-scene episodes start from crawl clip frames (hands-and-knees, mid-crawl, going down), the rest standing so the transition is learned.
- Rules relaxed only in crawl context: "fall" = uncontrolled fall (torso angular velocity / impact), not low head; knees, shins, forearms may touch the floor; grippers must stay off the floor (forearm-and-knee crawl).
- **Not transferred to hardware** (finger risk).

### Stage 5 details
For a skill that stays rare or poor: train a BeyondMimic-style tracker (PPO, reward = "match this clip"); in those scenes add a small loss pulling our policy's action toward the tracker's action in the same state (DAgger used as an auxiliary loss). Task reward and style stay primary.

---

## 4. How hand tucking and raising are learned

Three forces, all necessary:

1. **Task and safety rewards say *that* the hands must be protected** — hand clearance, self-clearance, hand-violation termination, reactive event success. Alone they give *some* avoidance, often ugly (elbow out, arm twisted, swung wide).
2. **The style prior picks *which* avoidance is natural for the context:**
   - **tight gap** (lateral gap ≈ 0.45 m) → sidle/protect group has humans with elbows in, hands at the chest → tucking earns style reward;
   - **table edge** (obstacle at ~0.7 m near the hands) → clips of hands lifted over the edge → raising earns style reward;
   - **approaching object** (near-hand flag, closing speed) → clips of the hand pulled back toward the body → retreat toward the body, not radially outward.
   Reward says "don't get hurt"; prior says "and do it the way people do". Neither alone gives tucking.
3. **Start states and observation make it learnable:**
   - some gap and table episodes start from tucked/raised clip frames, so tucking is experienced early rather than discovered;
   - the context features must also be in the **policy's** observation (gap width, table height near hands, closing speed) so it can know *when* to tuck.

Before the own capture exists: authored tuck/raise poses (FK-checked joint configurations) serve as crude positives for the protect group; replaced by the captured clips in stage 4.

---

## 5. Perception: fields for training, Gallant for deployment

**Now:** precomputed potential fields (guidance direction, boundary direction, SDF) sampled at body points. Fast, noise-free, and carry *where to go* — but they do not exist on the real robot (would need a live map and field computation; impossible for moving objects).

**Gallant** (CVPR 2026): LiDAR point cloud → voxel grid (e.g. 32×32×40 occupied/free) → **z-grouped 2D CNN** (height layers as channels). One policy on a G1 handles ground, lateral and overhead obstacles from raw LiDAR. MTC reused it for its student.

**Recommendation:** keep fields for the **teacher** (all of sections 2–4); distil into a **Gallant-style voxel student** for deployment:
- inputs: proprioception + goal + voxel grid from a simulated torso LiDAR (MuJoCo Warp ray-casting against scenes and analytic moving objects);
- encoder: z-grouped 2D CNN;
- training: **DAgger** — the student drives, the field-based teacher labels every visited state with its action, supervised regression;
- **perceptual hallucination:** jitter obstacles slightly while the teacher's behaviour stays valid, so the student reads geometry instead of memorising rooms;
- sensor realism: noise, dropout, latency, limited field of view;
- **moving objects:** LiDAR ~10 Hz and voxels carry no velocity → stack 2–3 voxel frames (or a frame difference) so approach speed is visible.

**To check:** Gallant's code release (repo `InternRobotics/Gallant` exists; paper said commands "released soon"). The encoder is simple to reimplement if needed.

---

## 6. Timeline and decisions

| stage | effort |
|---|---|
| 1 motion library | ~1 week data work |
| 2 style reward + separate critics | ~1 week code, then a pilot |
| 3 crawl (sim-only) | ~1 week, optional |
| 4 own VR capture | capture days + retargeting |
| 5 tracking teacher | optional |
| 6 voxel student | ~2 weeks; required for hardware |

**Open decisions**
- [ ] Stage 2 with public data now, or wait for own capture?
- [ ] Voxel student now (hardware soon) or after the prior work?
- [x] Crawling: simulation-only for now (finger risk).

---

## 7. References
- AMP — Peng et al. 2021, https://arxiv.org/abs/2104.02180
- SMP (score-matching priors) — https://arxiv.org/abs/2512.03028
- T-GMP (terrain-conditioned priors) — https://arxiv.org/abs/2606.06944
- Context-Aware Motion Priors — https://arxiv.org/abs/2608.03234
- Selective AMP — https://arxiv.org/abs/2604.19102
- CWI (part-wise imitation, multi-critic) — https://arxiv.org/abs/2606.27676
- StyleLoco (adversarial distillation) — https://arxiv.org/abs/2503.15082
- Hybrid Motion Priors (G1, codebook) — https://arxiv.org/pdf/2607.24083
- BeyondMimic — https://arxiv.org/abs/2508.08241
- MTC — https://arxiv.org/abs/2609.21107 ; PASSAGE — https://arxiv.org/abs/2609.18732
- Gallant — https://arxiv.org/abs/2511.14625 , https://github.com/InternRobotics/Gallant
- Data: PHUMA https://davian-robotics.github.io/PHUMA ; OmniRetarget https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset ; CIRCLE https://tml.stanford.edu/circle_dataset/ ; GMR https://github.com/YanjieZe/GMR ; holosoma https://github.com/amazon-far/holosoma
