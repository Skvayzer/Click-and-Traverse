# Safety filters, MPC/planning and hybrid learned+safety methods for slow, close passage of legged/humanoid robots through narrow gaps

*Method note: WebSearch was unavailable (session search budget exhausted), so sources are papers I identified, verified by their arXiv abstract pages and read in full or in their relevant sections. Full-text Markdown for most of them is in `papers/<arXiv-id>/` in this folder: 2212.14199, 2109.12313, 2007.11718, 2505.11494, 2510.14959, 2010.09819, 2104.14895, 1911.09146, 2205.05787, 2103.13987, 2309.05837, 2011.00032, 2203.02638, 2405.00846, 2302.07469, 2309.06647, 2304.08586, 1610.00700, 2602.17822. For 2109.09047 and 1412.1082 only the LaTeX source/PDF is there, because Markdown conversion failed. Where a fact comes from another paper's bibliography or summary, that is stated. Lines marked "Inference" are mine, not the sources'.*

## Q1. CBF / safety-filter methods on legged robots and humanoids near obstacles: narrow passages, deadlock, "repulsion-dominated" stopping, and the fixes

### Takeaway
CBF filters limit the robot's **approach velocity** in proportion to the remaining distance (∇h·v ≥ −αh). They do not add a repulsive force, so a slow command is the least modified one. The deadlocks reported in the literature have three causes, none of which is low speed: (a) over-inflated robot/obstacle shapes or margins that disconnect the free space at a gap; (b) undesired equilibria where the goal pull is exactly opposed by the barrier gradient; (c) one-step (myopic) optimisation. The fixes are tight shapes (polytopes) with small cm-level margins, MPC horizons, detect-and-circulate/rotate phases, explicit CBF-over-goal priority with slack, and clipping the goal-tracking pull. Additive potential fields, by contrast, are documented to fail between closely spaced obstacles and to oscillate in narrow corridors. That behaviour matches the user's "repelled" description.

### Cited Findings
**APF versus CBF, and the classic narrow-passage failure**
- Koren & Borenstein (ICRA 1991) found that potential-field methods "often fail to find trajectories between closely spaced obstacles; they also can produce oscillatory behavior in narrow corridors". This is quoted from Fox, Burgard & Thrun's summary — [Fox et al. 1997, IEEE RAM](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf). The original PDF returned HTTP 403 to me; the reference is in bibliographies in the papers folder.
- In a single-integrator example using Khatib's APF (K_att = K_rep = 1), the field works at influence radius ρ0 = 1 and 0.25, gets stuck in a local minimum at ρ0 = 0.5, and oscillates at ρ0 = 0.1, "due to the interaction between the attractive and repulsive forces". The equivalent CBF-QP completes the task "for all values of α ... with no oscillations" — [Singletary et al. 2020, arXiv 2010.09819](https://arxiv.org/abs/2010.09819).
- Singletary et al. prove that an APF's repulsive potential can be turned into a CBF, with the attractive potential used as the *desired velocity* instead of being summed. The CBF then intervenes only when −∇hᵀ∇U_att + α(h) < 0: "safety is imposed only when necessary rather than always (via the addition of potentials)". The APF-derived CBF "gets closer to the obstacles with fewer oscillations" — [Singletary et al., arXiv 2010.09819](https://arxiv.org/abs/2010.09819).
- In quadrotor tests with 1 m and 0.7 m wide "doorways", APF oscillations occurred only where the drone failed to reach the goal (the wall and the 0.7 m door). On hardware, "APFs fail to reach the goal in the case of the narrow door, while the CBFs succeeds" — [Singletary et al., arXiv 2010.09819](https://arxiv.org/abs/2010.09819).
- DWA authors: "By solely maximizing the target heading the robot quickly would get stopped by the first obstacle that blocks its way, unable to move around it". Combining heading, clearance and velocity terms lets the robot circumvent obstacles "as fast as it can" while still progressing — [Fox et al. 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf).

**Closed-form CBF filters used on legged robots (what the filter actually does)**
- Model-free safety-critical control. A safe velocity is designed on a reduced-order (single-integrator) model and tracked by the walking controller. The obstacle CBF is n_oᵀ q̇_s ≥ −α(d − r), with closed form q̇_s = q̇_d + max{−n_oᵀ q̇_d − α(d − r), 0}·n_o. The full system is safe if tracking is faster than the approach rate (λ > α), and "for smaller α the system may become more conservative, evolving farther from the boundary of the safe set". Hardware: a Unitree A1 under an ID-QP walking controller tracked the safe velocity "by forward- and side-stepping" (α = 0.2 s⁻¹, K_P = 0.1 s⁻¹), or by walking forward and turning with a unicycle model — [Molnar et al., RA-L 2022, arXiv 2109.09047](https://arxiv.org/abs/2109.09047).
- The exponential CBF for relative-degree-2 constraints (h_e = ḣ + α1(h) ≥ 0) is used at both the MPC and the torque layer on ANYmal, for stepping stones rather than walls. Footholds were shrunk by 5 cm for state-estimation error — [Grandia et al., ICRA 2021, arXiv 2011.00032](https://arxiv.org/abs/2011.00032).
- Discrete-time CBF for bipedal navigation (simulation, 3D biped). The DCBF acts stride-to-stride and gives the minimum per-step heading change. The CLF (path following) is relaxed with a slack "so that V can grow when the CLF and CBF conditions are conflicting (when the obstacles are on the desired path, our primary goal is to avoid the obstacles rather than to follow the desired path)". Feasibility of the optimisation is assumed — [Agrawal & Sreenath, RSS 2017](http://www.roboticsproceedings.org/rss13/p73.pdf).

**Deadlock in a narrow passage on a legged robot: cause and fix**
- *Walking in Narrow Spaces*: a Unitree A1 (0.3 m wide, 0.6 m long; modelled as a 0.32 × 0.6 m rectangle) runs centroidal NMPC with exponential DCBF duality constraints between polytopes, plus WBC. With spheres (Euclidean-distance DCBF) instead, "the robot can get stuck more often and adopt a very slow speed because of the over-approximation using spheres ... the robot gets into a deadlock in narrow spaces". Results:
  - Random map, 48 trials: 8.4 s vs 13.5 s travel time; 20.8% vs 50% collision or stuck.
  - Long narrow corridor: the sphere baseline "failed in all 16 trials because the robot gets stuck to entering the corridor"; the polytope method failed in 8.3%, partly because the global path ignored the robot's configuration.
  - [Liao et al., IROS 2023, arXiv 2212.14199](https://arxiv.org/abs/2212.14199)
- The same paper adds a margin β in d̃ = max{d − β, 0} "to prevent the robot from adopting a conservative strategy, e.g., taking a large detour or stopping in front of the obstacle at the earlier stage of the NMPC". Hardware values: α = 0.03 m minimum distance, β = 0.06 m, decay γ = 0.98 — [Liao et al., arXiv 2212.14199](https://arxiv.org/abs/2212.14199).
- "When a tight-fitting obstacle avoidance motion is expected, the above over-approximations of the shape of the robot can lead to deadlock maneuvers". Also: "When one-step control input is optimized [Agrawal 2017], it could lead to a deadlock situation such that the robot is safe but unable to track the reference command. A nonlinear model predictive control formulation can overcome these problems". Further points from the same paper:
  - "Reversing motion for deadlock avoidance" is a short-horizon motion primitive.
  - A larger terminal-cost weight "improves deadlock avoidance, but can also increase velocity of the robot, leading to unsafe motion".
  - γ near 1 "converges to ∂S slowly but can easily become infeasible"; γ near 0 tracks faithfully but can approach quickly and become unsafe.
  - [Thirugnanam, Zeng, Sreenath, ICRA 2022, arXiv 2109.12313](https://arxiv.org/abs/2109.12313)
- Distance constraints in MPC "will not confine the robot's movement unless the robot is relatively close to the obstacles", so they need long horizons; DCBF constraints in MPC act earlier with short horizons — [Zeng, Zhang, Sreenath, ACC 2021, arXiv 2007.11718](https://arxiv.org/abs/2007.11718). Li et al. use DCBFs for the same reason — [arXiv 2205.05787](https://arxiv.org/abs/2205.05787).

**Theory of CBF-QP deadlock (undesired equilibria) and resolution schemes**
- Reis, Aguiar & Tabuada (IEEE Control Systems Letters 5(2):731–736) show that "even for a single integrator dynamics with a circular obstacle, the [CLF-CBF-QP] could induce non-origin equilibrium points that are locally stable". Their fix adds an extra CBF constraint, but its feasibility "is only assumed". This is as summarised by [Tan & Dimarogonas, arXiv 2104.14895](https://arxiv.org/abs/2104.14895).
- Tan & Dimarogonas: undesired equilibria lie either in the interior or on the boundary of the safe set. A boundary equilibrium where the goal (CLF) gradient is parallel to the barrier gradient (∇V = k∇h, k > 0) remains an equilibrium for every QP weight p (Prop. 3). Their modified QP removes all interior undesired equilibria and boundary equilibria with L_g h = 0, but not every boundary equilibrium — [arXiv 2104.14895](https://arxiv.org/abs/2104.14895).
- Grover, Liu & Sycara: CBF deadlock is "an equilibrium of system dynamics [that] causes robots to come to a standstill before reaching their goals". It is "characterized by a force-equilibrium". The deadlock set is non-empty, bounded, measure-zero, and lies on the safety boundary. Random perturbations therefore help, but "there is no guarantee robots will not fall back in deadlock". Their resolution:
  1. Detect deadlock when ‖u*‖, ‖v‖ and ‖p − p_d‖ fall below thresholds.
  2. Rotate the robots around each other while holding the safety distance (a circulation-type manoeuvre).
  3. Hand back to the nominal controller.
  - Earlier work added perturbation terms (Wang et al. 2017; Rodríguez-Seda et al. 2016) or used coordination graphs for narrow corridors (Li 2005).
  - [Grover et al., arXiv 1911.09146](https://arxiv.org/abs/1911.09146)
- Composing several CBFs with a smooth min (log-sum-exp) *under-approximates* the intersection by up to ln N/κ, so the composed safe set lies inside the true one and approaches it as κ → ∞ — [Molnar & Ames 2023, arXiv 2309.06647](https://arxiv.org/abs/2309.06647). SHIELD likewise smooths the SDF barrier, h = λ(1 − e^{−γ·sdf}), "to reduce chattering oscillation that can occur when the closest obstacle switches" — [Yang et al. 2025, arXiv 2505.11494](https://arxiv.org/abs/2505.11494).
- On safety filters in general: a robot can stay safe "by never moving; of course, some hard-to-please customers may return the robot on the basis that it doesn't do anything". "Constraining a robotic system to remain in an overly restrictive safe set can render it impractically inefficient at its intended function, or even unable to fulfill it". Switch-type least-restrictive filters cause "control chatter and erratic motion ... near the safe set boundary", which motivates smoother intervention and filter-aware task policies — [Hsu, Hu, Fisac, arXiv 2309.05837](https://arxiv.org/abs/2309.05837).
- In MPC with a soft barrier, ANYmal "clip[s] the position tracking error ... to a maximum threshold of 15 cm" to handle "the occasional conflict between the collision avoidance and the position tracking objectives". This prevents "a sudden pull towards the commanded trajectory once the robot negotiates the blocking obstacle" — [Gaertner et al., ICRA 2021, arXiv 2103.13987](https://arxiv.org/abs/2103.13987).

### Inferences
- **Inference — additive repulsion and CBF projection scale oppositely with speed.**
  - In an additive APF, the repulsive term does not depend on the commanded speed. If the attractive or guidance term is scaled by an operator-chosen speed, lowering the speed shifts the balance toward repulsion and creates an equilibrium (local minimum) in front of a gap. This is exactly the user's "repulsion outweighs attraction at slow speed" hypothesis, and it fits the Koren & Borenstein and Singletary findings.
  - In a CBF filter, the correction is max{−n·v_d − αh, 0}: it removes only the *excess approach velocity*. A slow nominal command is therefore modified least, or not at all.
  - So the observed "can't pass slowly, bursts through fast" behaviour looks APF-like (distance-based repulsion), not CBF-like. The policy's learned obstacle response probably depends on distance rather than on approach rate.
- **Inference — likely reason for bursting.** If the RL reward penalises proximity per timestep, the integrated penalty for a gap of length L scales as L/v. A faster pass is then cheaper whatever the commanded speed, which matches 0.66–0.97 m/s bursts at 0.1–0.525 m/s commands. CBF-RL's reward penalises only violations of ∇h·v ≥ −αh and the deviation from the filtered action, so it does not reward short exposure (see Q3). This mechanism is not reported in any source I read; it should be checked in the user's reward terms.
- **Inference — disconnected free space.** If an effective learned margin plus the modelled body extent exceeds the 0.44–0.48 m gap, a CBF/MPC controller would deadlock (as Liao et al. and Thirugnanam et al. report for over-approximated shapes). An RL policy cannot stop forever, so it may instead treat passage as an unavoidable violation and minimise its duration. The model-based fixes are tight body geometry and cm-level static margins (Liao: 3 cm and 6 cm), plus speed-dependent inflation only (Q4).
- **Inference — a symmetric gap is not, by itself, a CBF deadlock.** Inside a gap the barrier gradients of both walls are mostly lateral, so they constrain lateral velocity, not forward velocity. The goal-opposed boundary equilibria (Tan & Dimarogonas Prop. 3; Grover) arise at corners or off-centre approaches. That is why horizon (MPC), circulation/rotation phases and centring terms are the usual fixes.

### Gaps
- I found no CBF paper on legged or humanoid robots that explicitly sweeps commanded speed and reports deadlock as a function of speed. All the deadlock reports above are geometric or myopic.
- Circulation- or tangential-field CBF variants (beyond Grover's rotation phase) and modulation-matrix dynamical-system obstacle avoidance could not be verified here, because the search budget was exhausted. They are a lead to check.
- Gaertner et al.'s numeric safety-margin values were lost in conversion and the LaTeX source had no matching text, so they are not reported.

## Q2. MPC and whole-body planning for humanoids or legged robots through narrow or confined gaps (side-stepping, torso rotation), with speeds and perception

### Takeaway
Model-based legged systems pass narrow gaps by **slowing, re-orienting (yaw up to 90°, side-stepping), then speeding up inside the passage**. Shape-aware constraints (polytopes, sphere decompositions, SDFs) are the enabling factor. Reported speeds are low: about 0.5 m/s desired for the A1 NMPC, and ABS characterises MPC-based legged collision avoidance as below 0.5 m/s. For humanoids, the classic narrow-space planners (Kanehiro et al. 2004; Kumagai et al. 2018) change locomotion style to a lateral (sidestep) posture. I could verify them only through bibliographies and secondary descriptions, without speeds. A 2026 G1 preprint schedules sidestep yaw from corridor width and slows down in narrow segments, but has no quantitative results yet.

### Cited Findings
- **A1, NMPC + exponential DCBF on polytopes.** Desired linear velocity 0.5 m/s; NMPC horizon 1.0 s (δt ≈ 0.015 s); 0.5 s trotting gait cycle; four closest obstacles within a 1 m box; polytopes of up to 15 vertices. Perception: two onboard depth cameras plus a tracking camera, Octomap at 0.025 m, Euclidean clustering into polytopes, and a Dijkstra global planner that ignores the robot's shape. Behaviour: "in the straight corridor trial [0.5 m minimum clearance], the robot can slow down and turn its heading 90 degrees to fit itself into the corridor. After the robot enters the corridor, it speeds up to the desired speed". In L- and V-shaped corridors it "decelerates at the corners to turn its body slowly". No collisions in hardware trials — [Liao et al. 2023, arXiv 2212.14199](https://arxiv.org/abs/2212.14199).
- **ANYmal, collision-free MPC.** A soft squared-hinge barrier on SDF distances of a sphere decomposition (6 overlapping body spheres plus 2 for the handle and lidar cage), with an ESDF from Voxblox. Hardware showed "sidestepping, aligning to walls, and backward movement in order to execute a commanded turn". In "artificially narrowed down office floors" for over five minutes there was one minor collision, with a slim, poorly mapped obstacle. With fast dynamic obstacles "the robot trots in place till the obstacles have passed", "although continuously commanded to go ahead" — [Gaertner et al., ICRA 2021, arXiv 2103.13987](https://arxiv.org/abs/2103.13987).
- **Speed comparisons by ABS.** MPC-based legged collision avoidance (citing Gaertner 2021, Chiu 2022, Teng 2021, Liao 2023) suffers "slow movements (velocity <0.5 m/s)". Classical collision-free planning leads "to slow and statically stable gaits". Existing navigation works "typically exhibit limited agility (velocity <1 m/s) to ensure safety" — [He et al., RSS 2024, arXiv 2401.17583](https://arxiv.org/abs/2401.17583).
- **Humanoid narrow-space planning (bibliographic only).**
  - Kanehiro, Hirukawa, Kaneko, Kajita, Fujiwara, Harada, Yokoi, "Locomotion Planning of Humanoid Robots to Pass Through Narrow Spaces" (ICRA 2004).
  - Kumagai, Morisawa, Nakaoka, Kanehiro, "Efficient Locomotion Planning for a Humanoid Robot with Whole-Body Collision Avoidance Guided by Footsteps and Centroidal Sway Motion" (Humanoids 2018).
  - Ge (2026) describes them as showing that a humanoid can reshape "its effective passage width by changing body yaw and locomotion style — walking face-forward in open space, then rotating into a lateral (sidestep) posture when the corridor tightens". He also notes that contact-level footstep and whole-body planners "can traverse tight spaces" but focus on footholds and balance rather than on "an explicit, tunable mapping from corridor ... width to body yaw".
  - Bib entries are in [Ge 2026, arXiv 2610.27688](https://arxiv.org/abs/2610.27688) and [Liao et al. 2023](https://arxiv.org/abs/2212.14199).
- **ATT/IVH on Unitree G1 (navigation layer above a learned policy).**
  - A sidestep ratio ρ comes from tube width w via a cubic smoothstep between body depth w_dp (full sidestep) and shoulder width w_sh (frontal). Desired yaw is θ = ψ ± (π/2)ρ. The left/right side is chosen by a minimum-yaw rule and latched with hysteresis "to avoid left/right chatter".
  - The progress speed "is further scaled down when ρ is large (narrow tube ⇒ move more carefully)", and lateral path-centring gain may be raised during transitions. Output is (v_x, v_y, ω_z) to a sidestep-enhanced AMP policy or the default G1 controller. Perception: Mid-360 LiDAR with A*/Hybrid A*.
  - Training reference motions: frontal 0.3–1.0 m/s; stage-B speeds U[0.2, 0.8] m/s.
  - The evaluation is "planned"; the paper states "No quantitative success rates or timing statistics are claimed".
  - [Ge 2026, arXiv 2610.27688](https://arxiv.org/abs/2610.27688)
- **Humanoid whole-body planning in clutter (simulation).** The Randomized Possibility Graph uses *sufficient* conditions (a bounding box of all walking motions) and *necessary* conditions (a minimal collision geometry). RPG edges "that squeeze between obstacles are Indeterminate" and are handed to a full whole-body multi-modal planner. Scenarios involve ducking, stepping over and stepping across gaps in doorways; no speeds are given — [Grey, Ames, Liu, arXiv 1610.00700](https://arxiv.org/abs/1610.00700).
- **Cassie with an RL policy plus NMPC-DCBF (simulation).** The robot reaches "full speed (around 1 m/s) when there are no obstacles while quickly pulling back when obstacles are in range", and crouches under an arch "with a relative high speed". The task takes about 25 s versus 50 s for a prior HZD-based stack that "results in a conservative planned speed", with "reduced deadlocks" — [Li et al., RSS 2022, arXiv 2205.05787](https://arxiv.org/abs/2205.05787).
- **Footstep-level DCBF for bipeds.** Obstacles are avoided by limiting per-step heading change (simulation) — [Agrawal & Sreenath 2017](http://www.roboticsproceedings.org/rss13/p73.pdf).
- **Tight-maze NMPC-DCBF with oriented polytopic robots** (triangle, rectangle, L-shape; simulation): robot orientation is a decision variable, so the robot rotates to fit — [Thirugnanam et al., arXiv 2109.12313](https://arxiv.org/abs/2109.12313).

### Inferences
- **Inference — the model-based pattern is the opposite of the observed G1 behaviour.** Model-based legged stacks slow down at the entrance, re-orient, then resume the commanded speed inside. The G1 policy instead accelerates through the gap whatever the command. A navigation-layer posture and speed schedule (like ATT/IVH, or Liao's NMPC behaviour) puts the turn-sideways decision *before* the gap, at low speed, rather than leaving it to emergent RL behaviour.
- **Inference — what the 13-point field offers.** Representing the body by a few tight primitives (Liao: polytopes; Gaertner: 6+2 spheres) is what turned "stuck" into "passes" in those papers. The user's 13 body points carry per-point signed distances, which is already a multi-primitive representation. The question is whether the policy's implicit margin per point is larger than the gap allows sideways.

### Gaps
- I could not verify any humanoid MPC or whole-body planner that reports passing gaps **narrower than shoulder width** with measured speeds and onboard perception. Kanehiro 2004 and Kumagai 2018 are known only from bibliographic entries and Ge's description.
- No verified humanoid source on arm-tucking or torso-rotation constraints in MPC for passages.
- Gaertner et al.'s speeds and margins are not recoverable from the converted text.

## Q3. Hybrid learned-locomotion + safety layer (filter, shield, reach-avoid value, recovery policy): behaviour at low commanded speed and whether the robot keeps moving through tight spaces

### Takeaway
On humanoids, the closest prior art is **SHIELD** and **CBF-RL**, both from Ames' group and both on the Unitree G1. Both act at the *velocity-command* level with a single-integrator barrier: SHIELD at runtime, with a stochastic DTCBF on top of an RL velocity-tracking policy; CBF-RL inside training (filter plus barrier reward), deployed with no filter. Neither tests sub-shoulder-width gaps; both model the robot as a disc or cylinder. **ABS** shows the typical cost of shielding: much fewer collisions but many more "timeouts" (getting stuck), with dense-obstacle local minima as the stated failure mode. It slows down in tight spaces by design. No hybrid paper I read reports a "burst regardless of commanded speed" behaviour; the filters only remove approach velocity.

### Cited Findings
- **SHIELD (Unitree G1).**
  - Architecture: a "safety layer that sits 'above' the nominal learning-based controller". A PPO LSTM velocity-tracking policy is modelled as a single integrator with a CVAE-learned residual. Commands are filtered at 100 Hz with a stochastic DTCBF, E[h(x_{k+1})] ≥ αh(x_k), with α chosen online from a risk level P.
  - Hardware setup: obstacles are cylinders of radius 0.3 m, with "a safety margin of 0.38 m from the center of mass". A* gives a path, and the nominal velocity points to the next node. Perception: Livox Mid-360 LiDAR with Euclidean clustering and a T265 for localisation.
  - Result: "The nominal DTCBF filter, being unaware of the dynamics residual, either performs too aggressively colliding with the obstacle, or performs to conservatively taking almost twice the time to arrive at the goal". SHIELD can "adjust conservativeness online to only enforce maximum safety conditions when needed".
  - The simulation used a constant commanded velocity of 0.5 m/s.
  - [Yang, Werner, Cosner et al. 2025, arXiv 2505.11494](https://arxiv.org/abs/2505.11494)
- **CBF-RL (Unitree G1).**
  - A closed-form CBF-QP filter during training: v_safe = v_policy + max(b − aᵀv_policy, 0)·a/‖a‖², with a = ∇h and b = −αh.
  - A reward r_cbf = min(aᵀv_policy − b, 0) + exp(−‖v_policy − v_safe‖²/σ²) − 1, which "penalizes actions whenever the safety filter is activated".
  - G1 obstacle task: h = ‖p_o‖ − R_r − R_o; the policy learns to "modulate its own velocity to avoid the obstacle, even when the velocity command prompts it to [collide]". Hardware ran with no runtime filter; a ZED 2 point cloud is clustered into cylinders.
  - The paper notes that runtime filters unknown to the policy "can lead to conservative behaviors", and that real-world state-estimation noise "hinders the deployment of runtime explicit filters".
  - [Yang, Werner, de Sa et al. 2025, arXiv 2510.14959](https://arxiv.org/abs/2510.14959)
- **ABS (Unitree Go1).**
  - Structure: an agile *goal-reaching* policy (rewarded by r_agile ∝ forward speed: "the robot has to either run fast or stay at the goal") plus a twist-tracking recovery policy. The switch happens when the learned reach-avoid value satisfies V̂ ≥ −0.05. The recovery twist solves argmin future goal distance subject to V̂ < threshold, via about 5 gradient steps. Recovery commands span v_x ∈ [−1.5, 1.5], v_y ∈ [−0.3, 0.3] m/s and ω ∈ [−3, 3] rad/s.
  - Simulation, 8 obstacles: ABS-a/n/c collision rates were 4.4/5.7/2.9% vs 26.1/21.7/15.5% for the agile policy alone. **Timeouts rose to 16.7/15.2/11.3% vs 0.6/1.0/1.3%.**
  - Failure mode: "when the obstacles are too dense and form a local minimum, our policy can easily fail, which is also reflected by the high timeout rates".
  - Behaviour: ABS "runs fast in the open spaces, and slows down in the tight spaces for safety". Peak 3.1 m/s; 2.5 m/s peak and 1.5 m/s average in tight space with dynamic obstacles.
  - Without softening the failure label, the learned RA value "has local minima in front of the obstacles".
  - [He et al., RSS 2024, arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- **Cassie: RL policy identified as a low-order linear model, then NMPC-DCBF.** The linear-model fit holds below about 0.6 Hz input frequency, and the reactive planner output is low-pass filtered at 0.5 Hz. Cassie and obstacles are over-approximated by cylinders plus a safety buffer for model mismatch. Results are in Q2 (≈1 m/s free, "pulling back" near obstacles, "reduced deadlocks"); simulation only — [Li et al., RSS 2022, arXiv 2205.05787](https://arxiv.org/abs/2205.05787).
- **Model-free CBF on a velocity interface (A1).** The guarantee requires the walking controller to track faster than the CBF lets the boundary be approached (λ > α). In practice "one can pick a small enough α ... by gradually increasing α from 0" — [Molnar et al., arXiv 2109.09047](https://arxiv.org/abs/2109.09047).
- **Critique of reduced-order filters on legged robots.** Existing filters "rely on local, reduced-order models. These filters tend to overly restrict agility and can still fail when perturbed". The gameplay filter (full-order, 36-D quadruped) targets falls under tugging and terrain, not obstacles — [Nguyen, Hsu, Yu et al., CoRL 2024, arXiv 2405.00846](https://arxiv.org/abs/2405.00846).
- **Learner/recovery switching for safe RL** (for falls on narrow-footing tasks, not obstacles). Naive switching "may cause frequent switch[ing]" — [Yang et al., IROS 2022, arXiv 2203.02638](https://arxiv.org/abs/2203.02638).
- A stochastic DTCBF was evaluated on a *simulated* quadruped walking a narrow path (safe region); no hardware gap tests — [Cosner et al., RSS 2023, arXiv 2302.07469](https://arxiv.org/abs/2302.07469).

### Inferences
- **Inference — what SHIELD and CBF-RL would do at the user's commanded speeds.**
  - Both bound only the approach rate, so at 0.1–0.5 m/s commands they would leave forward motion along a gap almost unmodified.
  - Their disc model would not fit a 0.44–0.48 m gap: SHIELD's 0.38 m CoM margin alone implies a ≥0.76 m clear width. Adopting either for gaps requires a body-shape barrier (polytope, or per-point SDFs like the user's 13 points) and margins on the order of centimetres.
- **Inference — CBF-RL is the most direct template for retraining.**
  - It replaces a proximity or "repulsion" reward with an approach-velocity barrier reward plus a training-time filter on the commanded planar velocity, on the same robot.
  - Its penalty is zero whenever ∇h·v ≥ −αh, so slow or tangential motion near a wall is not penalised. A per-timestep distance penalty is, and rewards fast exposure.
- **Inference — expect more stalling from any shield.** Adding a shield or recovery layer above the current policy would likely trade collisions for more stalls (ABS: timeouts ×10 or more) unless the geometry and margin problem (Q1) is fixed. A horizon or circulation mechanism is also needed to escape local minima.

### Gaps
- None of the hybrid papers reports results at *low commanded speeds in sub-shoulder-width gaps*, or a speed sweep. SHIELD and CBF-RL test isolated cylinders or pedestrians; ABS is goal-reaching and does not track operator speeds in the agile mode.
- No source found on reach-avoid or shielded RL for humanoids specifically in narrow passages.

## Q4. Speed-dependent clearance: should margins shrink at low speed and grow at high speed?

### Takeaway
Yes. Every principled scheme I found makes the required clearance (or the admissible speed) a function of speed through **reaction distance plus braking distance**:
- DWA's admissible velocities;
- the SSM protective-separation logic of ISO/TS 15066, now ISO 10218-2:2025 Annex L;
- MPC-CBF "stop within the horizon" tuning;
- CBF/velocity-damper approach-speed bounds;
- ABS's velocity-conditioned reach-avoid values;
- time-to-collision-based repulsion in human crowds.

At low speed the speed-dependent part vanishes and only a static term (sensing, estimation and tracking error) remains. So slow motion is what *permits* small clearances. The user's policy, which can only pass narrow gaps fast, behaves the opposite way.

### Cited Findings
- **DWA.** A velocity is admissible "if the robot is able to stop before it reaches the closest obstacle on the corresponding curvature": V_a = {(v, ω) | v ≤ √(2·dist(v, ω)·v̇_b) ∧ ω ≤ √(2·dist(v, ω)·ω̇_b)}. The example uses v̇_b = 50 cm/s² and ω̇_b = 60 deg/s². The method avoided collisions "with speeds of up to 95 cm/sec". With the full objective, "the fastest trajectory leading through the door area gets the highest evaluation" — [Fox, Burgard, Thrun 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf).
- **SSM sizing rule.** Protective-field radius R ≥ v·t_r + v_h·(t_r + v/a_max) + v²/(2a_max) + C, where v is robot speed, t_r reaction time (scanner, PLC, brake), v_h = 1.6 m/s worst-case human speed (from ISO 13855), a_max maximum deceleration, and C measurement tolerance plus intrusion. Robot speed is scaled by a linear ramp s(d) that reaches zero at the protective field — [Bajrami et al. 2026, arXiv 2611.07621](https://arxiv.org/html/2611.07621). The equation was extracted from the arXiv HTML; check the typography before quoting it.
- **Standards status.** ISO/TS 15066 §5.5.4 SSM "formulas for calculating the protective separation distance, which accounted for system reaction times and operator speeds, have been relocated to the new normative Annex L" of ISO 10218-2:2025 — [Hartmann et al. 2026, arXiv 2602.17822](https://arxiv.org/abs/2602.17822). I did not access the ISO text itself.
- **MPC-DCBF tuning.** "For safety, the reference tracking velocity should be such that the robot can come to a halt within the prediction horizon with the input as the maximum deceleration". The obstacle-consideration radius "is calculated using the reference tracking velocity, prediction horizon and the maximum deceleration" — [Thirugnanam et al., arXiv 2109.12313](https://arxiv.org/abs/2109.12313).
- **CBF approach-rate bounds.**
  - Molnar et al.: n_oᵀq̇ ≥ −α(d − r), i.e. approach speed ≤ α × clearance — [arXiv 2109.09047](https://arxiv.org/abs/2109.09047).
  - Exponential CBF h_e = ḣ + α1(h) — [Grandia et al., arXiv 2011.00032](https://arxiv.org/abs/2011.00032).
  - DTCBF h(x_{k+1}) ≥ αh(x_k): "the value of h(x_k) can only decay as fast as the geometric sequence α^k h(x_0)" — [Cosner et al., arXiv 2302.07469](https://arxiv.org/abs/2302.07469).
- **ABS.** The learned reach-avoid value is conditioned on the base twist: "As the robot's velocity changes, the landscape of RA values changes accordingly". The resulting system "slows down in the tight spaces" — [He et al. 2024, arXiv 2401.17583](https://arxiv.org/abs/2401.17583).
- **Human crowds.** Interaction energy follows time-to-collision τ, not separation: E(τ) = (k/τ²)·e^{−τ/τ0}. The pair distribution plotted against distance r "shows very different behavior ... for pedestrian pairs with different rate of approach", while against τ the curves "collapse onto each other". Fitted τ0 is about 1.4 s (dense "Bottleneck" data, mean speed 0.55 m/s) and 2.4 s (sparse "Outdoor", 0.86 m/s). Simulated agents reproduce "arching around narrow passages" and "clogging and 'zipping' patterns at bottlenecks" — [Karamouzas, Skinner, Guy, Phys. Rev. Lett. 2014, arXiv 1412.1082](https://arxiv.org/abs/1412.1082).
- **Static margins used in legged hardware for unmodelled error.**
  - A1 NMPC: α = 0.03 m (plus β = 0.06 m to avoid early stopping) — [Liao et al.](https://arxiv.org/abs/2212.14199).
  - ANYmal footholds: 5 cm for state estimation — [Grandia et al.](https://arxiv.org/abs/2011.00032).
  - Cassie: "a safety buffer" for model mismatch — [Li et al.](https://arxiv.org/abs/2205.05787).
  - G1: 0.38 m disc — [SHIELD](https://arxiv.org/abs/2505.11494).

### Inferences
- **Inference — the margin decomposes into a static and a speed-dependent part.** Apply the SSM/DWA structure to static obstacles (v_h = 0): margin(v) ≈ C + v·t_r + v²/(2a_max). With illustrative values I chose (t_r = 0.15 s, a_max = 1 m/s²; not from a source):
  - At 0.1 m/s the speed-dependent part is ≈ 1.5 + 0.5 cm, so only C matters.
  - At 0.9 m/s it is ≈ 13.5 + 40 cm, which should *forbid* a 0.44–0.48 m gap entirely.
  - The policy's behaviour (refusing the gap slowly, accepting it fast with momentum) is therefore the reverse of the speed-and-separation principle.
- **Inference — a speed-aware field term.** A speed-aware obstacle term would weight repulsion by time-to-contact or by approach velocity, not by distance alone. Options are a CBF-RL-style barrier on ∇h·v, a TTC weighting as in Karamouzas et al., or v-dependent inflation of the per-point signed distances. This weakens "repulsion" exactly when the commanded speed is low and the motion is tangential, which is the slow side-step regime needed in 0.44–0.48 m gaps.
- **Inference — evaluation metrics.** The literature's evaluation would track minimum clearance against speed at the gap, timeouts or stalls, and time-in-gap, separately per commanded speed. ABS and Liao report timeouts and stuck counts; SHIELD reports time to goal.

### Gaps
- I found no legged or humanoid paper that explicitly implements a *speed-scheduled* geometric margin (e.g. d_safe(v) = C + v·t_r + v²/2a) inside a CBF or MPC for narrow gaps. The speed dependence in the papers above is implicit (via α, the horizon, or the RA value).
- I did not access the primary ISO/TS 15066 or ISO 10218-2:2025 text, which is paywalled. The formula comes from the Bajrami et al. 2026 arXiv paper.
- Krogh's 1984 "generalized potential field", a velocity-dependent APF, and later velocity-dependent APFs could not be verified without search.
