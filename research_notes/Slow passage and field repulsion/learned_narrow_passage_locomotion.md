# Learned narrow-passage locomotion: obstacle-structure perception, slow passage, and the HumanoidPF / Click-and-Traverse family

Scope note. Papers were read in full or by section from arXiv sources converted into the shared `papers/` folder: 2601.16035 (read from its LaTeX source and PDF), 2401.17583, 2511.14625, 2403.00187, 2309.05665, 2209.12827, 2306.14874, 2505.19214, 2609.21107, 2609.18732, 2610.27688, 2609.09158, and 2610.25711 (read from its LaTeX source). The CAT release code was read from the local fork's first code commit (`b3bf06a`, "v0: init runable code base"). I assume this commit mirrors the official repository but did not check it against GitHub HEAD. WebSearch and the Semantic Scholar API were unavailable (session budget used up; HTTP 429), so discovery went through citation lists (CAT, Gallant, PASSAGE, MTC) and the repo's earlier research notes. "Inference" marks my reasoning, not a source's claim.

## Q1. What does the Click-and-Traverse / HumanoidPF paper (and its release) say about speed, how the velocity command is built from the fields, narrow passages, and limitations?

### Takeaway
The CAT paper never states a commanded or achieved speed and never discusses momentum or slow motion. It poses the task as reaching a goal position (success means reaching within 0.1 m of the goal in 5 s), and its only stated limitations are the lack of contact-rich interaction and open-ended generalization. The released code turns this into velocity-command tracking. The command is the pelvis guidance-field vector, scaled, with a one-sweep barrier-normal correction, and anything below 0.2 m/s becomes "stop". In the released field generator the guidance speed is a constant 0.6 m/s everywhere except within 0.3 m of the goal, so the field contains no clearance-dependent slowdown in narrow gaps.

### Cited Findings
**Paper: formulation and fields.** Xue, Liang, Zhang, Zeng, Liu, Lian, Wang, Liu, Shi & Li 2026, "Collision-Free Humanoid Traversal in Cluttered Indoor Scenes", arXiv 2601.16035.
- The task is stated as: "Given a target position g ∈ R³, and a set of indoor obstacles … the humanoid needs to move to g without any collision". The real-world system lets the user "specify a desired goal by simply clicking on a grid map". The paper describes no velocity-command interface. — [CAT, arXiv 2601.16035](https://arxiv.org/abs/2601.16035)
- HumanoidPF is a classical APF. The attractive part is U_att = η‖x−g‖_geo, a geodesic 3D distance that "inherently accounts for obstacle geometry". The repulsive part is Khatib's U_rep = ½ξ(1/d − 1/d0)² for d ≤ d0. The field is F = −∇(U_att + U_rep), queried at body-part locations. — [CAT](https://arxiv.org/abs/2601.16035)
- The authors name the symmetric-gap failure themselves. When "the robot faces an obstacle directly ahead … the potential fields on the left and right sides of the body direct it toward opposite paths. In a symmetrical configuration, these vectors cancel each other out, leading to a multi-modal dilemma where the robot becomes trapped in a local minimum or exhibits oscillatory behavior." — [CAT](https://arxiv.org/abs/2601.16035)
- Their fix is priority weighting. The root gets w0 = 1 and other parts 0.5. A collision-urgency weight w1 = λ·max(−∇d·v_k, 0.5)·exp(−d) depends on signed distance and on the body part's velocity toward the obstacle. The final field is F_H = w0·w1·F/‖F‖, a unit direction scaled by priority and urgency, which "selectively amplifie[s]" minute asymmetries to resolve the dilemma. — [CAT](https://arxiv.org/abs/2601.16035)
- The observation is F_H sampled at K = 13 body parts. The paper argues this lets the policy "reason about traversal decisions rather than implicit inference from raw visual data". It also claims the field acts "like a low-pass perceptual filter", giving a "surprisingly negligible sim-to-real gap". — [CAT](https://arxiv.org/abs/2601.16035)
- The reward R_Field is a von Mises–Fisher log-likelihood of each body part's unit motion direction v̂_k = v_k/‖v_k‖ around μ = F_H/‖F_H‖, with concentration κ = κ_max‖F_H‖. It rewards direction, not speed. — [CAT](https://arxiv.org/abs/2601.16035)

**Paper: training, evaluation and stated limitations.**
- Training scenes are 5 m × 5 m crops of 3D-FRONT plus procedurally generated boxes that "may … be placed in close proximity to form narrow traversal passages". Goals are "randomly sampled on a circle with a radius of 2 m around the start". There are 139 + 216 specialist scenes, each trained with 32,768 environments and 5,000,000 episodes, then distilled by DAgger. Training adds sensor noise and force perturbations. — [CAT](https://arxiv.org/abs/2601.16035)
- Success means "the robot reaches within 0.1 meters of the target location in 5 seconds without colliding with any obstacle". The paper reports no speeds. The project page also has nothing on speed, momentum, failure cases or limitations. — [CAT](https://arxiv.org/abs/2601.16035); [CAT project page](https://axian12138.github.io/CAT/)
- Simulation side-pass success rates (Table II): CAT 97.3%; "Ours w/o OBS_Field" (multi-layer elevation map instead of HumanoidPF) 95.2%; "Ours w/o R_Field" (plain collision penalty) 28.0%; ASTraversal 55.2%; Humanoid Parkour 41.3%. On Side-Hurdle-Crouch the figures are 86.6 / 53.7 / 0.0 / 0.5 / 0.4%. — [CAT](https://arxiv.org/abs/2601.16035)
- On the real robot, the same specialists were distilled into three students. Side-pass success was 5/5 for HumanoidPF, 2/5 for voxel grids and 1/5 for elevation maps; across four scenes HumanoidPF scored 4–5/5. The voxel student "often exhibits instability because it is sensitive to fine-grained geometric details … and to noise in task-irrelevant regions". — [CAT](https://arxiv.org/abs/2601.16035)
- The real system uses Fast-LIO2 plus OctoMap, rebuilding the map and field at 10 Hz. — [CAT](https://arxiv.org/abs/2601.16035)
- Stated limitations, in full: "Our current framework does not yet exploit contact-rich interactions, such as leaning on obstacles or stepping onto support surfaces, and generalization to entirely unseen, highly unstructured, and extremely cluttered scenes remains an open challenge." Narrow passages, momentum, slow motion and static fields are not mentioned. — [CAT](https://arxiv.org/abs/2601.16035)

**Release code: how fields become a velocity command** ([official repo](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse); read from local commit `b3bf06a`: `cat_ppo/envs/g1/env_cat.py`, `env_cat_pri.py`, `procedual_obstacle_generation/pf_modular.py`).
- Three fields are precomputed on a voxel grid: `sdf.npy`, `bf.npy` (boundary normal) and `gf.npy` (guidance). They are sampled at head, pelvis, torso, feet, hands, knees and shoulders. — [CAT code](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse)
- `compute_cmd_from_rtf` works as follows:
  - Take v = 0.7 × (pelvis gf)_xy in `env_cat.py`, or 0.6 × in the privileged `env_cat_pri.py`.
  - For each of the head, feet and hands, with unit boundary normal b̂_k, compute L_k = b̂_kᵀ gf_k. Where b̂_kᵀv < L_k, add the correction (L_k − b̂_kᵀv)·b̂_k.
  - Average the corrections over the M points and add the mean to v.
  - In `env_cat.py`, scale the command by 0.75.
  - If the command norm is below 0.2, replace it with a stop command.

  The command is recomputed every control step. A delayed, noisy copy is refreshed every 5 steps from noisy odometry. — [CAT code](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse)
- Field generation (`make_guidance_field_progressive`):
  - The direction is −∇T of a scikit-fmm fast-marching distance to a goal seed of radius 0.12 m.
  - Within r_proj = 5 voxels (0.2 m at the 0.04 m default) of obstacles, the normal component is smoothly removed, making the guidance "more tangential near the edge". Inside obstacles it is set to the outward normal.
  - The direction is normalized and multiplied by a scalar speed: speed = v_max (0.6 m/s) when the planar goal distance is above 0.3 m, and v_max·(dist/0.3)³ otherwise. The comment reads "far: vmax, near goal decays to 0". No speed term depends on clearance or gap width.

  — [CAT code](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse)
- In the initial release's default env configs, field reward entries (`headgf`, `handsgf`, `feetgf`, `*df`) exist but are scaled 0.0, and `tracking_lin_vel` = 1.0 against this field-derived command. I could not find a vMF R_Field by name in that commit. — [CAT code](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse)
- The < 0.2 m/s stop rule matches the legged_gym convention, which zeroes resampled xy commands with norm ≤ 0.2: `self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)`. — [legged_gym `legged_robot.py`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)

### Inferences
- **The release trains essentially one commanded speed.** With gf stored at v_max = 0.6 m/s, the nominal command away from the goal is about 0.6 × 0.7 × 0.75 ≈ 0.32 m/s in `env_cat.py` and about 0.36 m/s in `env_cat_pri.py`. Barrier corrections change this somewhat, and the stop band removes 0–0.2 m/s. So slow non-zero commands (0.1–0.2 m/s), and slow commands specifically inside gaps, are probably rare or absent in training. This assumes the shipped `gf.npy` files used the default v_max. A different scaling in the user's fork (their 0.525 m/s maximum suggests unit-norm gf × 0.7 × 0.75) would not change the conclusion if gf magnitude is constant away from the goal.
- **The field gives no "slow down here" signal.** Speed is constant up to 0.3 m from the goal, and near-wall guidance only rotates tangentially. Any slowing in gaps must come from the policy itself, which the paper never asks for or measures.
- **The paper's own symptom list matches the user's.** Symmetric cancellation, local minima and oscillation are what the paper describes. Its priority weighting acts only through the observation and reward magnitudes, so whether it resolves the dilemma at low speed is untested. Note that w1 has a floor of 0.5 on the approach-velocity factor, so urgency does not fall below that floor however slowly the robot approaches.
- **The 5 s criterion may imply speed above the release command.** If evaluation goals were about 2 m away like training goals, success in 5 s needs ≥ 0.4 m/s average, faster than the release's nominal ~0.32–0.36 m/s command. The evaluation goal distances are not stated, so this is uncertain.
- **The field reward, not the field observation, drives side passes in CAT's ablation.** Replacing the HumanoidPF observation with elevation maps barely hurt side-pass success (97.3 → 95.2%). Removing the field reward collapsed it (97.3 → 28.0%).

### Gaps
- The paper reports no measured speeds, speed profiles, momentum or behaviour under different commanded speeds. No supplementary material with speeds was found on the project page.
- I did not verify whether later official commits changed v_max, the 0.2 m/s stop rule or the reward scales, or where the paper's vMF reward is implemented. Only the first code commit was inspected.
- The real-world success table gives no speeds and no gap widths for the "Side-Pass" scenes.

## Q2. Which learned locomotion works use ray distances, SDF samples or potential fields as observations or rewards, what do they report about speed near obstacles (rushing, ignoring slow commands, hesitation, oscillation), and how did they make robots slow down or stop safely?

### Takeaway
Sparse distance perception (ABS's 11 rays, Miki's spherical scans, Omni-Perception's sectorized LiDAR distances) works for collision avoidance. No paper reports a policy that follows slow commands through gaps because of such perception. The rushing mechanisms the literature documents are mostly about the objective:
- discounting plus per-step penalties produce "wait, then run fast at the last moment" (Rudin 2022);
- time-integrated penalties make sprinting through obstacles optimal (Robot Parkour Learning);
- agile goal-reaching policies run fast into tight spaces and collide (ABS).

Documented fixes for slowing down are:
- a velocity-aware learned safety value with a separate velocity-tracking recovery policy that slows in tight spaces (ABS);
- time-to-go conditioning, where more time gives "a slow and safe gait" (Rudin 2022, ANYmal Parkour);
- capped speed targets (PASSAGE);
- penalties scaled by velocity, i.e. per distance (Robot Parkour Learning);
- explicit width-based speed scaling in a modular command layer (ATT).

Classical APF analysis predicts the user's symptom directly. In a gap the summed repulsion "points straight away from the opening", and entry depends on the goal force's magnitude.

### Cited Findings
**Classical potential-field limits.** Koren & Borenstein 1991, ICRA.
- They list four inherent problems: "1. Trap situations due to local minima … 2. No passage between closely spaced obstacles. 3. Oscillations in the presence of obstacles. 4. Oscillations in narrow passages." — [Koren & Borenstein 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- On a door frame: "The sum of all repulsive forces … points straight away from the opening between the two obstacles. Depending on the relative magnitude of the target-directed force Ft, the robot will either approach the opening further, or it will turn away." — [Koren & Borenstein 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- In narrow corridors, "the robot experiences repulsive forces simultaneously from opposite sides". A wall perturbation "excites the robot into unstable oscillations and eventually a collision". The stability limit ties corridor width to robot width, speed V, delay τ and sampling time T. The condition "is harder to be satisfied for larger V, τ, and T; for example, if a heavy vehicle moves at high speed, it is more likely that oscillations occur". They add that some researchers test "at slow speeds which conceal the disadvantages of the PFMs". — [Koren & Borenstein 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)

**ABS: ray-distance observations.** He, Zhang, Xiao, He, Liu & Shi 2024, "Agile But Safe", RSS 2024, arXiv 2401.17583.
- Exteroception is "the distances that 11 rays travel from the robot to obstacles, similar to sparse LiDAR readings". The rays are cast horizontally from the base, evenly spaced over [−π/4, π/4], and enter the observation as log values. A ResNet-18 network predicts them from depth images. — [ABS](https://arxiv.org/abs/2401.17583)
- The agile policy is goal-reaching. Its r_agile term rewards forward speed (v_x/4.5 m/s) or standing at the goal ("the robot has to either run fast or stay at the goal"). A −20·r_stall term "penalize[s] the waiting behaviors", and undesired collisions cost −100. Position rewards apply only in the last T_r seconds, "free from explicit motion constraints such as target velocities". — [ABS](https://arxiv.org/abs/2401.17583)
- Rushing is reported: "the π^Agile baseline runs fast but collides near the second tight space." Reward tuning and PPO-Lagrangian "trade off within a boundary". The Lagrangian baseline "runs much slower" everywhere. — [ABS](https://arxiv.org/abs/2401.17583)
- The slow-down fix:
  - A policy-conditioned reach-avoid value V̂ is learned from base twist, goal and rays. "As the robot's velocity changes, the landscape of RA values changes accordingly."
  - When V̂ ≥ −0.05, a twist command is searched to minimize future goal distance subject to V̂ < threshold (gradient descent, about 5 steps).
  - A separate recovery policy with no exteroception tracks the twist command, trained with v_x^c ~ U(−1.5, 1.5) m/s and initial speeds up to 5.5 m/s.
  - Result: ABS "runs fast in the open spaces, and slows down in the tight spaces for safety thanks to the shielding of RA values and the recovery policy".

  — [ABS](https://arxiv.org/abs/2401.17583)
- Numbers from the LAG-shielding table: ABS 79.1% success, 5.7% collisions, mean peak speed 3.48 m/s, mean speed 2.08 m/s. LAG reaches 1.41 m/s; LAG + RA shielding 1.22 m/s with 2.8% collisions. — [ABS](https://arxiv.org/abs/2401.17583)
- Hesitation evidence: without "illusion" randomization, which overwrites ray distances with U(d_goal + 0.3, true distance), "the robot will sometimes tremble near a wall which it has never seen in simulation". — [ABS](https://arxiv.org/abs/2401.17583)
- Failure: "when the obstacles are too dense and form a local minimum, our policy can easily fail, which is also reflected by the high timeout rates". — [ABS](https://arxiv.org/abs/2401.17583)

**Rudin, Hoeller, Bjelonic & Hutter 2022**, "Advanced Skills by Learning Locomotion and Local Navigation End-to-End", IROS 2022, arXiv 2209.12827.
- Velocity tracking restricts behaviour: "the locomotion policy must be allowed to slow down or even stop in dangerous situations. These cases would result in higher velocity tracking costs making the desired behaviors harder to achieve." — [Rudin 2022](https://arxiv.org/abs/2209.12827)
- Rushing from discounting: with per-step penalties "larger when the robot is walking", "it is beneficial to push the negative penalties as far into the future as possible. Therefore, the policy learns to wait until the last moment when it suddenly runs fast towards the target." The fix is r_stall = −1 if speed < 0.1 m/s while more than 0.5 m from the target. If T_r is too short, "the policy learns to jump or lean towards the target at the last moment". — [Rudin 2022](https://arxiv.org/abs/2209.12827)
- A continuous position-tracking reward produces a policy "trained to run as fast as possible", which they did not deploy because it is "prone to aggressive motions". — [Rudin 2022](https://arxiv.org/abs/2209.12827)
- Speed can be commanded through time-to-go: "When the policy believes it has a lot of time to reach the target, it will adopt a slow and safe gait. As that time is reduced, it will try to close the gap faster and faster." — [Rudin 2022](https://arxiv.org/abs/2209.12827)

**Robot Parkour Learning.** Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn & Zhao 2023, CoRL 2023, arXiv 2309.05665.
- The obstacle-penetration penalty is multiplied by forward velocity v_x "to prevent the robot from exploiting the penetration reward by sprinting through the obstacles to avoid high cumulative penalties over time". In other words, a per-time penalty made sprinting through the obstacle optimal, and scaling by v_x turns it into a per-distance penalty. — [RPL](https://arxiv.org/abs/2309.05665)

**ANYmal Parkour.** Hoeller, Rudin, Sako & Hutter 2024, Science Robotics, arXiv 2306.14874.
- Low-level skills and the navigation module use the time-budgeted position formulation, which "allows the policy to explore the terrain to find safer paths and take its time where needed". A "Don't wait" term penalizes ‖v_b‖ < 0.2 m/s by −1. The navigation policy must handle cases where "the robot arrives at high speeds on a narrow obstacle. It often has to quickly decelerate the robot and then turn on the spot". — [ANYmal Parkour](https://arxiv.org/abs/2306.14874)

**Miki, Lee, Wellhausen & Hutter 2024**, "Learning to walk in confined spaces using 3D representation", ICRA 2024, arXiv 2403.00187.
- The high-level teacher uses spherical ray scans; the authors "used the sparse ray tracing pattern, because using more complicated data structure, such as depth images or voxel grids, could slow down the reinforcement learning process". Its task is to "follow a desired velocity target". It adds a base-distance reward exp(−α(d_max − min(d, d_max))) to keep clearance, plus a large collision penalty. — [Miki 2024](https://arxiv.org/abs/2403.00187)

**Omni-Perception.** Wang et al. 2025, arXiv 2505.19214. Raw LiDAR point clouds feed a velocity-tracking policy.
- This is an APF built into the reward. The target velocity is v_cmd + V_avoid, where V_avoid sums, over 36 sectors, components "pointing away from" obstacles nearer than about 1 m, with magnitude exp(−d·α_avoid). A "distance maximization" reward pushes "towards open areas". — [Omni-Perception](https://arxiv.org/abs/2505.19214)
- Narrow-passage failure: the robot treated grass as an obstacle, "because it was close, it quickly moved to the left". Then, "because the entire passage was narrow, the robot was forced to enter the grass … perceived as a dangerous area, which led to mission failure". — [Omni-Perception](https://arxiv.org/abs/2505.19214)

**Gallant.** Ben et al. 2025, arXiv 2511.14625.
- An APF-like shaping reward aligns velocity with a(p, g) = Σ w_j·u_r,j + κ Σ w_j·γ_j·t_j: a repulsion term plus a tangential left/right term with κ = 0.8, with weights w_j that grow sharply below about 0.2 m. It is applied to Forest and Door terrain. A "No movement" termination ends episodes if the robot has not moved 1 m within 4 s. — [Gallant](https://arxiv.org/abs/2511.14625)

**PASSAGE.** Ma et al. 2026, arXiv 2609.18732.
- HumanoidPF guidance is precomputed from the scene SDF and destination, "sample[d] at 11 anchors", and used as a planner loss L_PF. Removing it drops success from 98.7% to 84.8% and contact-free success from 70.3% to 20.1%. — [PASSAGE](https://arxiv.org/abs/2609.18732)
- The RL post-training reward includes r_speed = exp[−(vᵀĝ − v*)²/(0.2 m/s)²] with v* = min(0.6 m/s, d/1 s). Arrival "requires remaining within 0.35 m of the destination below 0.3 m/s for 0.2 s". — [PASSAGE](https://arxiv.org/abs/2609.18732)

**CAT itself.** The paper names oscillation and local-minimum trapping from cancelling left and right fields as a risk of applying APF to multi-body humanoids, and addresses it only through weighting. — [CAT](https://arxiv.org/abs/2601.16035)

### Inferences
- **Two documented ways to learn "run through"** are (a) discounted or per-time penalties near obstacles (Rudin 2022; RPL) and (b) speed-agnostic or speed-rewarding objectives (ABS r_agile; Rudin's continuous tracking). Both fit a G1 that bursts through gaps at 0.66–0.97 m/s whatever the command. Any per-step proximity, collision or field-misalignment cost accrued while inside a 0.44–0.48 m gap pays the policy to minimize time spent there, unless it is scaled by speed or by distance travelled as in RPL. I have not checked the user's reward terms; this is a mechanism to test, not a finding.
- **Perception richness and slowing look separable in the literature.** The only explicit "slows down in tight spaces" result (ABS) uses the same sparse 11-ray perception as the agile policy that rushes. The slowing comes from a velocity-conditioned safety value plus a velocity-tracking fallback.
- **The APF force-balance story fits the user's hypothesis** of repulsion winning over attraction at the gap mouth. Koren & Borenstein's analysis also predicts that higher speed worsens oscillation in narrow corridors. A policy that has learned to "punch through" may be exploiting momentum to escape a region where its learned value of slow approach is low. That is my inference; no learned-locomotion paper reports it.
- **Ready-made fixes for the field-to-command rule:** clearance-dependent speed limits (ATT, Q3), speed caps of the PASSAGE form v* = min(v_max, d/τ), and tangential tie-breaking with hysteresis (Gallant's tangential term; ATT's side latch).

### Gaps
- No paper found reports a velocity-commanded legged policy that ignores slow commands specifically near obstacles. The closest are Rudin's general statement that velocity tracking penalizes slowing and ATT's note that generic priors degrade under sustained lateral commands (Q3).
- I could not read the full ASTraversal paper (Chen et al., RA-L 2025, multi-layer elevation maps; cited by CAT) or "Humanoid Parkour Learning" for speed findings. Liao et al. 2023 (IROS, model-based "Walking in narrow spaces") was not read.
- No source quantifies how often per-step proximity costs cause sprint-through in humanoids. The RPL statement is qualitative, about quadrupeds.

## Q3. Which works trained legged robots or humanoids to pass narrow gaps or confined spaces, what perception did they use, did they command or achieve slow speeds, and do they report momentum-dependent passage?

### Takeaway
Since 2023, narrow-gap and confined-space traversal has been learned with many perception types:
- depth images and memory (Robot Parkour Learning tilt through 0.28 m slits, at a 0.5 m/s target);
- neural 3D reconstruction (ANYmal Parkour crouch);
- voxel grids (Miki 2024; Gallant's Door terrain; MTC side-stepping);
- LiDAR point-cloud encoders (SWAP walk-sideways; Omni-Perception);
- three-layer elevation maps (PASSAGE squeeze-through);
- RGB (TANGO).

Commanded speeds, where reported, are moderate: 0.5 m/s tilt target, 0.6 m/s speed cap, 0.2–0.8 m/s AMP sidestep commands. Few papers report actual speeds through gaps. None reports momentum-dependent passage or tests slow-command following inside gaps.

The 2026 humanoid works that do well at sideways passage either imitate human motion with dense structural perception (MTC, PASSAGE) or schedule posture and speed explicitly from a width estimate (ATT, TANGO). In two independent benchmarks, the released CAT generalist does markedly worse on side-stepping and contact-free passage.

### Cited Findings
**Quadrupeds.**
- Robot Parkour Learning (A1/Go1):
  - The policy squeezes "through thin slits of 0.28m by tilting (less than the robot width)".
  - Specialist teachers use privileged sparse geometry: "the distance from the robot's current position to the obstacle in front of the robot, the height of the obstacle, the width of the obstacle, and a 4-dimensional one-hot category".
  - The student is DAgger-distilled onto 48×64 depth at 10 Hz with a GRU.
  - The target forward speed is "around 1 m/s" for skills overall and 0.5 m/s for tilting (supplementary table).
  - The blind student "cannot complete any … tilting trials". An MLP without memory reaches "much lower performance on crawling and tilting".

  — [RPL](https://arxiv.org/abs/2309.05665)
- ANYmal Parkour (ANYmal D):
  - Perception is a multi-resolution neural 3D reconstruction from voxelized point clouds, with 12.5 cm coarse and 6.25 cm fine voxels and auto-regressive memory. The navigation module consumes its latent tensor.
  - Speeds reach up to 2 m/s; crouch passages go down to 0.4 m height.
  - Crouching success "drops the quickest when the passage becomes narrower than the height of the robot".
  - The memory lets the network remember the table top while crouching under it.

  — [ANYmal Parkour](https://arxiv.org/abs/2306.14874)
- Miki 2024 (ANYmal C/D, confined spaces):
  - Structure is hierarchical. A low-level policy tracks 6D commands (v_x, v_y, ω_z, roll, pitch, height). The high-level policy outputs velocity residuals plus posture.
  - The teacher sees spherical ray scans. The student sees a 32×32×32 occupancy voxel grid at 0.08 m, encoded by 3D convolutions into a GRU belief encoder. It "closely matched the performance of the teacher policy".
  - In the field, "the operator provided commands for the robot's x, y, and yaw velocities and the robot autonomously determined its own body height, roll, and pitch".
  - No speeds are reported. The simulation success criterion is 6 m within 30 s.
  - Limitation: "advanced maneuvers like using body contacts in tight spaces".

  — [Miki 2024](https://arxiv.org/abs/2403.00187)
- ABS (Go1): runs at high speed through cluttered rooms and "a dim and narrow corridor". The only ABS collision on that testbed was "due to the 'undetected' objects by the ray-prediction network as the corridor is quite dim". — [ABS](https://arxiv.org/abs/2401.17583)

**Humanoids (Unitree G1 unless noted).**
- Gallant:
  - Two Hesai JT128 LiDARs feed a robot-centric 32×32×40 binary voxel grid at 5 cm (1.6 × 1.6 × 2 m), processed by OctoMap at 10 Hz and a z-as-channels 2D CNN.
  - "Door presents narrow gaps demanding precise lateral clearance". On hardware, "it … plans lateral motions to pass through narrow doorways".
  - It uses goal-reaching with a 10 s horizon and goals on the perimeter of an 8 m × 8 m block.
  - No speeds are reported. The main stated bottleneck is LiDAR latency of more than 100 ms.

  — [Gallant](https://arxiv.org/abs/2511.14625)
- MTC (Wang, Xu, Kosukhin, Yeung, Lu & Xiao 2026):
  - VR human demonstrations (1185 trajectories, 327 environments) are retargeted with "strictly enforcing robot-scene clearance". "Where the robot cannot physically match human speed, extra frames are added via time dilation."
  - The student sees "a 32 × 32 × 40 binary voxel grid from a simulated torso LiDAR, encoded via a 2D CNN following Gallant".
  - On MTC-Challenge the MTC generalist scores 97.7% on side-stepping ("Sidewalk") against 49.8% for the CAT generalist. Corridor is 70.2% against 42.2%, and crawl 81.0% against 0.0%.
  - It is deployed on a G1 with a FAST-LIO2 voxel grid.

  — [MTC](https://arxiv.org/abs/2609.21107)
- PASSAGE (Ma et al. 2026):
  - About 100 h of scene-aligned human motion over 1,500 scenes trains a flow-matching planner at 6.25 Hz and a perceptive tracker at 50 Hz, working from "a torso-centered three-layer elevation map" extracted from a ROG-Map occupancy grid built online from LiDAR.
  - Behaviours include "turning sideways through narrow openings".
  - Squeeze-through on hardware: 10 trials, 10/10 and 9/10 (reported columns).
  - Against the released CAT generalist zero-shot: 98.7% vs 70.3% success and 70.3% vs 14.0% contact-free success. The authors call this "a system-level transfer comparison".
  - They state CAT's "reported controller actuates only 12 leg joints".

  — [PASSAGE](https://arxiv.org/abs/2609.18732)
- SWAP ("Test-Time Steerable Humanoid Whole-Body Collision Avoidance", 2026):
  - A Mid-360 LiDAR point cloud at 10 Hz feeds a PointNet++/GRU encoder (32-D, supervised by point-cloud and obstacle-configuration reconstruction). A diffusion planner at 20 Hz produces whole-body trajectory chunks, tracked by a 50 Hz RL controller.
  - "Walk sideways — navigating through a narrow lateral gap between two close obstacles": 88.4% success and 2.2% collisions in simulation.
  - Test-time steering targets objectives including "base speed" and improves targeted rewards by 34% on average.
  - Real failures are "primarily due to LiDAR occlusion from extreme body poses and sim-to-real tracking mismatch".

  — [SWAP](https://arxiv.org/abs/2610.25711)
- ATT (Ge 2026, "Adaptive Traversal Tubes"):
  - LiDAR localization feeds A* and Hybrid A* paths and a path-parameterized "effective passage width" field w(s). This drives a frontal-to-sidestep heading schedule, with the side chosen once by a minimum-yaw rule and held by a hysteresis latch.
  - Progress speed "is further scaled down when ρ is large (narrow tube ⇒ move more carefully)".
  - A sidestep-enhanced AMP policy is trained with v* ~ U[0.2, 0.8] m/s, because "generic priors are biased toward frontal walking. They degrade when IVH asks for sustained lateral speed v_y".
  - Quantitative results are "intentionally omitted" pending trials.

  — [ATT](https://arxiv.org/abs/2610.27688)
- TANGO (2026, RGB whole-body VLA):
  - Its data-generation planner "detects narrow passages and inserts 90-degree heading changes, inducing sideways walking".
  - It shows real "side-stepping through a narrow pathway".
  - Limitations: low-level tracker capability, and "vision input … relies solely on RGB images".

  — [TANGO](https://arxiv.org/abs/2609.09158)
- CAT: "detouring through narrow passages" and "squeezing through tight indoor spaces" with HumanoidPF at 13 points. No speeds are reported (see Q1). — [CAT](https://arxiv.org/abs/2601.16035)

### Inferences
- **No learned-locomotion paper I read describes passage that works only with momentum,** or failure when commanded slowly. Speed inside gaps is rarely measured. RPL's sprint-exploit note and ANYmal Parkour's "arrives at high speeds on a narrow obstacle … quickly decelerate" are the closest statements.
- **Strong 2026 humanoid side-passing results do not come from RL reward shaping alone.** They come from either reference motions (MTC, PASSAGE), which supply side-step kinematics and human-paced, time-dilated speed by construction, or explicit width-to-posture-and-speed scheduling (ATT, TANGO). The sideways burst the user sees is consistent with an RL policy that found side-on passage but never learned a slow side-step gait under a slow command. ATT reports exactly this weakness for generic priors under sustained lateral commands.
- **Memory and look-ahead matter in slits.** RPL's no-memory MLP collapses on tilting, and ANYmal Parkour relies on remembering occluded structure. A single-frame field at 13 body points carries no memory of gap geometry already passed and no look-ahead beyond the field's influence radius. A slow policy may therefore lack the information to commit to the gap (inference).

### Gaps
- No paper reports speed profiles through lateral gaps of about 0.44–0.48 m for a G1 or similar, and none compares passage success across commanded speeds.
- MTC's real-robot per-domain numbers did not survive conversion of its table; only the simulation columns are reliable here.
- ATT has no quantitative results yet. TANGO reports navigation metrics (NE/SR/SPL/CR), not gap-passage speeds.
- "Traversing Narrow Paths" (Huang et al. 2025, arXiv 2508.20661), cited by CAT, appears to be about narrow footholds (beams), not lateral gaps. Not read.

## Q4. Is there evidence that a policy perceiving obstacle structure (local map, voxels, depth) follows slow commands through gaps better than one with sparse body-point distances, and are there direct comparisons of sparse point-sampled distance features with dense structural representations?

### Takeaway
No study tests slow-command following through gaps as a function of perception type, so the user's hypothesis is neither confirmed nor refuted by the literature. The direct comparisons that exist point in mixed directions:
- CAT's own ablations favour the 13-point HumanoidPF over elevation maps and voxels, especially on the real robot (side-pass 5/5 vs 1/5 and 2/5). But all students were distilled from the same field-trained specialists.
- Gallant shows that a 2.5D height map fails on doors where a voxel grid succeeds.
- Two 2026 system-level benchmarks show voxel- or elevation-map-based imitation systems far outperforming the CAT generalist on side-stepping and contact-free traversal. These comparisons are confounded by training data, references and action space.

Separately, sparse rays sufficed as privileged teacher input in Miki 2024 and RPL. Where slowing in tight spaces was achieved (ABS), it came from a velocity-aware safety module, not from denser perception.

### Cited Findings
- **CAT simulation ablation:** HumanoidPF observations vs multi-layer elevation maps, same field reward. Side-Pass 97.3 vs 95.2%, Side-Hurdle 95.4 vs 60.4%, Side-Hurdle-Crouch 86.6 vs 53.7%, Hurdle-Crouch 93.9 vs 77.8%. — [CAT](https://arxiv.org/abs/2601.16035)
- **CAT real-world comparison:** same specialists distilled into each representation. Side-Pass: HumanoidPF 5/5, voxel grids 2/5, elevation maps 1/5. Crouch-Pass: 4/5, 1/5, 3/5. The voxel policy is "sensitive to fine-grained geometric details … and to noise in task-irrelevant regions". — [CAT](https://arxiv.org/abs/2601.16035)
- **Gallant, voxel vs height map:**
  - In simulation, "Only-Height-Map fails on terrains such as Ceiling". On hardware, "the HeightMap baseline fails on overheading (e.g., Ceiling) and lateral (e.g., Door) obstacles due to its limited 2.5D representation".
  - With voxels, "Ceiling and Door … become the easiest".
  - Resolution: "the 10 cm grid … coarse quantization impairs fine contact- and clearance-sensitive interactions", while 2.5 cm reduces field of view.
  - The critic gets the height map as privileged input; the actor gets voxels only.

  — [Gallant](https://arxiv.org/abs/2511.14625)
- **Miki 2024:** a sparse ray-scan teacher was distilled into a voxel student that "closely matched the performance of the teacher". The sparse pattern was chosen because dense structures "could slow down the reinforcement learning process". — [Miki 2024](https://arxiv.org/abs/2403.00187)
- **RPL:** teachers with five sparse privileged scalars (distance, height, width, one-hot type) learned tilting through 0.28 m slits. The depth-plus-GRU student reproduced it, and blind or memory-less students failed or degraded. — [RPL](https://arxiv.org/abs/2309.05665)
- **ABS:** argues the low-dimensional ray representation makes the agile policy and RA network "easier to train" and "highly interpretable". Failures came from dense local minima and undetected objects in a dim corridor, not from representation sparsity as such. — [ABS](https://arxiv.org/abs/2401.17583)
- **MTC vs CAT on MTC-Challenge** (simulation): Sidewalk 97.7% (voxel-grid MTC generalist) vs 49.8% (CAT); Corridor 70.2% vs 42.2%. This is a benchmark from the MTC authors, with demonstration-based training and CAT used as released. — [MTC](https://arxiv.org/abs/2609.21107)
- **PASSAGE vs CAT:** 98.7% vs 70.3% success and 70.3% vs 14.0% contact-free success, "a system-level transfer comparison" on PASSAGE-distribution scenes encountered zero-shot by CAT. Within PASSAGE, removing tracker-side local geometric perception cut contact-free success from 70.3% to 18.4%. Adding HumanoidPF as a planner prior helped (contact-free 20.1% without it, 70.3% with it). — [PASSAGE](https://arxiv.org/abs/2609.18732)
- **ANYmal Parkour:** a learned 3D reconstruction remembers overhangs the elevation-map baseline mis-maps: "the baseline method is not designed to handle such scenarios with overhangs". — [ANYmal Parkour](https://arxiv.org/abs/2306.14874)

### Inferences
- **What each piece of evidence supports.** The literature shows that dense structure (voxels, multi-layer maps, point-cloud encoders) supports reliable sideways, contact-free gap passage in 2026 systems. CAT's ablations show that sparse field samples can match or beat them under CAT's own training and sim-to-real conditions. Neither line measures speed compliance. The claim "richer structure gives slower, more precise passage" is therefore unproven. The best-supported levers for slow passage are objective and command design (Q2), the training command distribution (Q1: the release trains roughly one speed plus stop), and explicit width-aware speed scheduling (ATT).
- **Body-point sampling does have a specific information deficit.** It conveys local clearance and directions at the body, but not upcoming gap width or the distance to the far side of the gap. A policy cannot tell whether slowing down will leave it wedged. Width-aware representations (ATT's w(s), voxel grids, a gap-width feature) supply this directly. A cheap test is to add a few sparse look-ahead features (gap width ahead along the guidance path, the SDF minimum along the path) to the existing 13 points and measure speed compliance at 0.1–0.3 m/s commands. This is a proposal; no source tested it.
- **CAT's real-world win shows the trade-off:** sparse, smoothed fields transfer better. Moving to voxels may trade the user's speed problem for the sim-to-real noise CAT reports. A hybrid (field observations plus a coarse local occupancy or width channel, or a field-only actor with a structure-aware critic as in Gallant) is consistent with the evidence (inference).
- **The cross-paper comparisons cannot isolate perception.** MTC and PASSAGE beat CAT in their own benchmarks but change training data, reference motions and controller scope at the same time.

### Gaps
- No paper evaluates slow commands (≤ 0.3 m/s) through lateral gaps for any perception type, or varies only perception while measuring speed compliance near obstacles.
- No study compares body-point-sampled SDF or field features against dense maps with all else fixed (same teacher, reward and command distribution) on gap-passage speed. CAT's distillation ablation is the closest, but it measures success, not speed.
- WebSearch and Semantic Scholar were unavailable, so post-2025 works outside the citation graphs used (CAT, Gallant, MTC, PASSAGE references and the repo's notes) may have been missed. In particular, humanoid LiDAR navigation papers on velocity tracking near obstacles were not searched systematically.
