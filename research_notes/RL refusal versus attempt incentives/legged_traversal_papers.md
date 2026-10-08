# How published legged/humanoid obstacle-traversal RL works structure rewards, terminations, time limits and curricula to prevent stalling/refusal

Scope note: the primary sources were read directly: arXiv PDFs, converted to text and searched for reward tables, terminations and curricula, plus the official GitHub code for legged_gym, rsl_rl, ZiwenZhuang/parkour and chengxuxin/extreme-parkour. Years are first arXiv versions, with the venue where known. In "Code" bullets, "HEAD" means the default branch as fetched on 2026-10-08.

Papers covered:
- Robot Parkour Learning (RPL, Zhuang+ 2023, CoRL'23)
- Extreme Parkour (Cheng+ 2023, ICRA'24)
- ANYmal Parkour (Hoeller+ 2023, Sci. Robotics 2024)
- Advanced Skills / final-position task (Rudin+ 2022, IROS'22)
- Learning to Walk in Minutes / legged_gym (Rudin+ 2021, CoRL'21)
- Humanoid Parkour Learning (Zhuang+ 2024, CoRL'24)
- Learning Agile Locomotion on Risky Terrains (Zhang+ 2023, ICRA'24)
- Agile But Safe (ABS, He/Zhang+ 2024, RSS'24)
- Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots (Lee+ 2024, Sci. Robotics)
- Learning to walk in confined spaces using 3D representation (Miki+ 2024, ICRA'24)
- CaT: Constraints as Terminations (Chane-Sane+ 2024)
- SoloParkour (2024, CoRL'24)
- Parkour in the Wild (Rudin+ 2025)
- Learning Humanoid Locomotion over Challenging Terrain (Radosavovic+ 2024)
- Collision-Free Humanoid Traversal in Cluttered Indoor Scenes (HumanoidPF / Click-and-Traverse, Xue+ 2026)
- PASSAGE (2026-09)
- Moving Through Clutter / MTC (2026-09)

## Q1. Reward structures: progress/velocity, goal bonus, alive bonus, penalties (with magnitudes)

### Takeaway
Two families exist:
- **Dense velocity tracking with a positive alive bonus.** Examples are RPL, Humanoid Parkour, Extreme Parkour and legged_gym. These keep obstacle contact as a small per-step penalty, not an episode-ending event.
- **Sparse "reach the target by time T" rewards.** Examples are Rudin 2022, ANYmal Parkour, Risky Terrains, ABS and Parkour in the Wild. Every work in this family pairs the sparse goal reward with an explicit anti-waiting term ("stall", "don't wait") plus a direction-of-motion shaping term, because without them the policy waits.

### Cited Findings
**Robot Parkour Learning (Zhuang et al., 2023)**
- The skill reward is r_skill = r_forward + r_energy + r_alive. The terms are:
  - r_forward = −α1·|v_x − v_x^target| − α2·|v_y|² + α3·e^{−|ω_yaw|}
  - r_energy = −α4·Σ_j |τ_j·q̇_j|²
  - r_alive = 2 per step
  - Target speed is about 1 m/s.
  - [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- Per-skill scales (paper Tables 3–7):

  | Skill | α1 | α2 | α3 | α4 (energy) | α5 = α6 (penetration) |
  |---|---|---|---|---|---|
  | Climb | 1 | 1 | 0.1 | 2e-6 | 1e-2 |
  | Leap | 1 | 1 | 0.05 | 2e-6 | 4e-3 |
  | Crawl | 1 | 1 | 0.05 | 2e-5 | 6e-2 |
  | Tilt | 1 | 1 | 0.05 | 1e-5 | 3e-3 |
  | Run | 1 | 1 | 0.05 | 1e-5 | 0 |

  Velocity commands are run 0.8, climb 1.2, leap 1.5, crawl 0.8 and tilt 0.5 m/s. PPO uses γ = 0.99, 4096 envs and 24 steps per env per batch. [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- Penetration reward: r_penetrate = −Σ_p (α5·1[p] + α6·d(p))·v_x, where p ranges over collision points sampled on the robot body (denser at hips and shoulders).
  - The paper says it is multiplied by forward velocity v_x "to prevent the robot from exploiting the penetration reward by sprinting through the obstacles to avoid high cumulative penalties over time." [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
  - In the released code the multiplier is instead the speed of each sample point plus 1e-3, not base v_x. [parkour repo legged_robot_field.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot_field.py)
- The released leap config (`a1_leap_config.py`) uses these scales:
  - tracking: world_vel_l2norm −1 (‖cmd − world vel‖), tracking_ang_vel 0.05
  - alive +2
  - penetrate_depth −1e-2, penetrate_volume −1e-2
  - collision −0.5
  - exceed_dof_pos_limits −0.4, exceed_torque_limits_l1norm −0.8, torques −2e-5
  - lin_pos_y −0.1, yaw_abs −0.1
  - Command lin_vel_x is in [1.0, 1.5] m/s.
  - [parkour repo a1_leap_config.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/a1/a1_leap_config.py)
- Net effect: the per-step reward is strongly positive (+2 alive) while the robot is upright. Obstacle collision is a small penalty (−0.5 per step), not a termination.

**Extreme Parkour (Cheng et al., 2023)**
- Tracking reward: r_tracking = min(⟨v, d̂_w⟩, v_cmd), with d̂_w = (p − x)/‖p − x‖ toward the next waypoint p.
  - Velocity is measured in the world frame to prevent "the unintended behavior of turning around the obstacle."
  - Edge clearance penalty: r_clearance = −Σ c_i·M[p_i], where M marks feet within 5 cm of an edge.
  - [Extreme Parkour arXiv 2309.14341](https://arxiv.org/abs/2309.14341)
- Released code: `tracking_goal_vel` is min(⟨v_world, d̂⟩, cmd)/cmd, so it is normalized to at most 1. Scales are:
  - tracking_goal_vel 1.5, tracking_yaw 0.5
  - collision −10 (per penalized body with contact force > 0.1 N)
  - lin_vel_z −1, ang_vel_xy −0.05, orientation −1
  - dof_acc −2.5e-7, action_rate −0.1, delta_torques −1e-7, torques −1e-5
  - hip_pos −0.5, dof_error −0.04, feet_stumble −1
  - feet_edge −1, only active for terrain level > 3
  - `only_positive_rewards = True`
  - There is no alive term.
  - [extreme-parkour legged_robot_config.py](https://github.com/chengxuxin/extreme-parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot_config.py); [extreme-parkour legged_robot.py](https://github.com/chengxuxin/extreme-parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot.py)

**legged_gym / Learning to Walk in Minutes (Rudin et al., 2021)**
- Default scales:
  - tracking_lin_vel 1.0 and tracking_ang_vel 0.5, both exp(−err²/0.25)
  - lin_vel_z −2, ang_vel_xy −0.05, torques −1e-5, dof_acc −2.5e-7
  - feet_air_time +1, collision −1, action_rate −0.01
  - termination −0.0, i.e. no terminal penalty
  - `only_positive_rewards = True`, commented "negative total rewards are clipped at zero (avoids early termination problems)"
  - [legged_gym legged_robot_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- Contacts with knees, shanks, or between the feet and vertical surfaces count as collisions and are penalized. Contacts with the base "are considered crashes and lead to resets." [Learning to Walk in Minutes arXiv 2109.11978](https://arxiv.org/abs/2109.11978)

**Rudin et al., 2022, "Advanced Skills…" (final-position task)**
- Task reward: r_task = (1/T_r)·1/(1 + ‖x_b − x_b*‖²), applied only when t > T − T_r.
  - T = 6 s and T_r = 1 s, so the reward is given only in the last second.
  - Targets are sampled 1–5 m away.
  - [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- Penalties are applied throughout: −c1‖q̈‖² − c2‖τ‖² − c3·N_collisions − c4‖a − a_prev‖² − c5·Σ‖ẍ_feet‖. Exact c values are not given in the paper. [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- Exploration bias reward: r_bias = ẋ_b·(x* − x_b)/(‖ẋ_b‖‖x* − x_b‖), the cosine between base velocity and the direction to the target.
  - It is "automatically removed … once r_task reaches 50% of its maximum value."
  - [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- Stall penalty: r_stall = −1 if ‖ẋ_b‖ < 0.1 m/s and ‖x_b − x*‖ > 0.5 m, else 0. [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)

**ANYmal Parkour (Hoeller et al., 2023/2024)**
- Locomotion-skill rewards (Table S2):

  | Term | Expression | Weight |
  |---|---|---|
  | Position tracking | 1_{t*<1}(1 − 0.5‖r_xy − r*_xy‖) | 10 |
  | Heading tracking | 1_{t*<1}(1 − 0.5‖ψ − ψ*‖) | 5 |
  | Joint velocity | | −0.001 |
  | Torque | | −1e-5 |
  | Joint velocity limit | | −1 |
  | Torque limit | | −0.2 |
  | Base acceleration | | −0.001 |
  | Feet acceleration | | −0.002 |
  | Action rate | | −0.01 |
  | Feet contact force | Σ max(‖F‖ − 700, 0)² | −1e-5 |
  | **Don't wait** | 1(‖v_b‖ < 0.2) | **−1** |
  | **Move in direction** | cos⟨v_b, r* − r⟩ | **+1** |
  | Stand at target | | −0.5 |
  | Collision | 1(knee/shank collision) | −1 |
  | Stumble | 1(‖F_xy‖ > 2‖F_z‖) | −1 |
  | **Termination** | 1(base collision) + 1(F > 1500 N) | **−200** |

  [ANYmal Parkour arXiv 2306.14874](https://arxiv.org/abs/2306.14874)
- Navigation (high-level) rewards:
  - Position tracking 1_{t*=0}(40·S_N − ‖r − r*_G‖), weight 0.15, given only on the last time step.
  - Termination 1_{α<π/2} + 1_{F>2500}, weight −0.5.
  - [ANYmal Parkour arXiv 2306.14874](https://arxiv.org/abs/2306.14874)
- The climbing-up skill deliberately reduces the base and knee contact penalty weights so the robot may use them. The paper says this "leads to the natural progression where the policy first learns to climb using its knees and then starts using its feet instead." [ANYmal Parkour arXiv 2306.14874](https://arxiv.org/abs/2306.14874)

**Learning Agile Locomotion on Risky Terrains (Zhang, Rudin, Hoeller, Hutter, 2023)**
- Reward table (Table IV):
  - Position tracking 10·δ_T(2)·1/(1 + ‖p − p*‖²) (25·δ_T(4) on the two hardest specialist terrains)
  - Heading tracking 5·δ_p(2)·δ_T(4)
  - **Termination penalty −200** ("const for early termination")
  - Collision −1 (thighs/shanks)
  - Joint velocity −0.001, joint-velocity limit −1, base acceleration −0.001, feet acceleration −0.0005
  - Action rate −0.01, torque −1e-5, torque limit −0.2, contact force −2.5e-5
  - **Don't wait 1(‖v‖ < 0.2) with weight −(1 − δ_p(1))**, i.e. it is not applied within 1 m of the target
  - **Move in direction cos⟨v, p* − p⟩ with weight 1 for the first 150 iterations only**
  - Stand still −δ_p(0.25)·δ_ψ(0.5)·δ_T(1)
  - Curiosity (RND) intrinsic reward on top
  - Episode lengths are 5–10 s depending on task.
  - [Risky Terrains arXiv 2311.10484](https://arxiv.org/abs/2311.10484)

**Agile But Safe (ABS, 2024)**
- r_penalty = −100·1(undesired collision): base, thighs, calves and horizontal foot collisions. [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- r_task = 60·r_pos_soft + 60·r_pos_tight + 30·r_heading − 10·r_stand + 10·r_agile − 20·r_stall. The tracking terms take the form 1(t > T − T_r)/T_r · 1/(1 + (err/σ)²) with these settings:
  - Soft position: σ = 2 m, T_r = 2 s.
  - Tight position: σ = 0.5 m, T_r = 1 s.
  - Heading: σ = 1 rad, T_r = 2 s.
  - [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- r_agile = max(ReLU(v_x/v_max)·1(correct direction), 1(d_goal < σ_tight)), with v_max = 4.5 m/s. "To maximize this term, the robot has to either run fast or stay at the goal." [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- r_stall = 1 "if the robot stays static when d_goal > σ_soft and the robot is not in the 'correct direction'… This term penalizes the robot for time waste." Episode length is U(7, 9) s and goals are 1.5–7.5 m away. [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)

**Humanoid Parkour Learning (Zhuang, Yao, Zhao, 2024)**
- Reward terms (Table 4):

  | Term | Weight |
  |---|---|
  | Linear velocity tracking exp(−‖v − v_cmd‖/0.25) | 1.0 |
  | Angular velocity tracking | 1.5 |
  | Orientation | −2 |
  | Energy | −2.5e-7 |
  | DoF velocity | −1e-4 |
  | DoF acceleration | −2e-6 |
  | Weighted torques | −1e-7 |
  | Contact forces | −3e-4 |
  | **Collision 1(‖F_i‖ > 0.1)** | **−10** |
  | Action rate | −6e-3 |
  | Arm DoF error | −0.3 |
  | Waist DoF error | −0.1 |
  | Hip-yaw DoF error | −0.1 |
  | Feet away | +0.4 |

  PPO uses γ = 0.99, lr 3e-5, 4096 envs and 24 steps. [Humanoid Parkour arXiv 2406.10759](https://arxiv.org/abs/2406.10759)
- Virtual (penetrable) obstacles are added on leap/jump terrains with r_penetrate = α·Σ_p d(p)·‖v(p)‖, α = −5e-3, to stop edge exploitation. Stairs get a footstep reward r_step = 6·(−ln‖d_x‖). [Humanoid Parkour arXiv 2406.10759](https://arxiv.org/abs/2406.10759)

**Wheeled-legged navigation (Lee et al., 2024)**
- High-level reward:
  - Sparse goal reward 1.0 when within 0.75 m of waypoint 1.
  - Dense progress reward clip(v·ê_wp, 0, 0.5)/0.5, annealed away during training.
  - "Exploration bonus" = −n_buf^i for each buffered past position within 1.0 m. This is effectively a penalty on revisiting or loitering at the same place.
  - Near-goal stability reward.
  - **"Not terminating was densely rewarded: r_h,surv := 1.0."**
  - [Lee 2024 arXiv 2405.01792](https://arxiv.org/abs/2405.01792)
- The low-level controller penalizes all non-wheel body contacts with r_bc = −|I_c,body \ I_c,wheel|. [Lee 2024 arXiv 2405.01792](https://arxiv.org/abs/2405.01792)

**Other works**
- Confined spaces (Miki et al., 2024): the high-level policy uses velocity tracking, a base-distance reward exp(−α(d_max − min(d, d_max))) that keeps clearance from obstacles, and "a large negative reward when the body hits obstacles". [Miki 2024 arXiv 2403.00187](https://arxiv.org/abs/2403.00187)
- Radosavovic et al. (2024, humanoid on challenging terrain): uses an alive reward r_a := 1 that "rewards longer episode lengths". [Radosavovic 2024 arXiv 2410.03654](https://arxiv.org/abs/2410.03654)
- Click-and-Traverse / HumanoidPF (Xue et al., 2026):
  - The paper describes the core field reward R_Field = Σ_k [log C_d(κ_k) + κ_k·μ_kᵀ·v̂_k], a von Mises–Fisher log-likelihood aligning each body part's motion direction with the potential-field direction.
  - The ablation "w/o R_Field" replaces it with "a basic collision-penalty reward commonly used in collision-avoidance tasks [Humanoid Parkour]".
  - The paper text does not list alive, goal or termination weights.
  - [CAT/HumanoidPF arXiv 2601.16035](https://arxiv.org/abs/2601.16035)
- Parkour in the Wild (Rudin et al., 2025): the RL fine-tuning reward table reuses the ANYmal Parkour terms, including "Don't wait 1(‖v_b‖ < 0.2)", collision and termination. Termination there is 1_{α>135°} + 1_{q̇>q̇lim}. [Parkour in the Wild arXiv 2505.11164](https://arxiv.org/abs/2505.11164)

### Inferences
- In every dense-velocity work the robot is not paid for standing still. Forward-velocity tracking is the dominant positive term and is ≈0 or negative at zero velocity. Where an alive bonus exists (RPL +2, Radosavovic +1, Lee +1), collisions do not terminate the episode, so the alive bonus does not compete with attempting the obstacle.
- In every sparse-goal (ETH) work, an explicit standing-still penalty is present:
  - Rudin 2022 stall −1
  - ANYmal Parkour don't wait −1
  - Risky Terrains −1 except near target
  - ABS −20·r_stall
  - Parkour in the Wild "Don't wait"

  This is the strongest pattern across the literature. It appears to be considered necessary once the main reward is sparse.
- Collision penalty magnitudes relative to the per-step positive reward:
  - RPL code: −0.5 vs +2 alive
  - Extreme Parkour: −10 per body vs ≤1.5 tracking, clipped by only_positive_rewards
  - Humanoid Parkour: −10 vs ~2.5 max tracking
  - ABS: −100 per step

  So large collision penalties are common, but they are per-step penalties, not terminations (except ABS, see Q2 gap).

### Gaps
- Rudin 2022 does not publish c1–c5 penalty constants or the stall weight beyond "−1" in the paper.
- The Humanoid Parkour Learning paper does not list its alive or termination terms.
- The CAT/HumanoidPF paper text does not list its full reward table (alive, goal and success weights). Those must come from the CAT code base, which was not examined here.

## Q2. Termination conditions and whether timeouts are bootstrapped

### Takeaway
Across these works, obstacle contact by feet, legs or limbs is almost never a hard termination. Terminations are falls (roll/pitch/height), base/torso crashes, or extreme impact forces. The legged_gym/rsl_rl lineage (Rudin 2021; RPL, Extreme Parkour and Humanoid Parkour code) bootstraps the value on time-outs. Extreme Parkour even labels goal completion as a time-out. Rudin 2022 explicitly removed timeout bootstrapping when it made time-to-go an observation, and found this more stable.

### Cited Findings
**legged_gym / rsl_rl**
- Termination is base contact force > 1 N (termination_contact_indices) or episode_length > max (20 s), with the comment "no terminal reward for time-outs". [legged_gym legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- The PPO time-out bootstrap is `rewards += γ · V(s) · time_outs`. [rsl_rl v1.0.2 ppo.py](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/algorithms/ppo.py)
- The 2021 paper motivates it: "we bootstrap the target of the critic with its own prediction" on time-outs. Bootstrapping "greatly reduces the critic loss and improves the total reward by approximately 10% to 20%". [Learning to Walk in Minutes arXiv 2109.11978](https://arxiv.org/abs/2109.11978)

**Robot Parkour Learning code**
- Termination terms are roll, pitch, z_low and z_high, with optional out_of_track. Base thresholds are roll 0.8 rad, pitch 1.6 rad, z_low 0.15 m and z_high 1.5 m.
- Thresholds can be obstacle-conditioned: e.g. tilt roll 1.5, and leap pitch 1.5 / roll 0.4 in the leap config.
- **There is no collision/contact termination.**
- `timeout_at_finished` marks finishing the track as a *time-out*, which is bootstrapped.
- The rsl_rl fork keeps the same `time_outs` bootstrap.
- Sources: [parkour legged_robot_field.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot_field.py); [parkour a1_field_config.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/a1/a1_field_config.py); [parkour rsl_rl ppo.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/rsl_rl/rsl_rl/algorithms/ppo.py)

**Extreme Parkour code**
- `check_termination` resets on roll > 1.5, pitch > 1.5, height < −0.25 m, or timeout (20 s).
- `reach_goal_cutoff` (all 8 waypoints reached) is OR-ed into `time_out_buf`, so success is bootstrapped like a timeout.
- **Collisions do not terminate.** They give −10 per body.
- Sources: [extreme-parkour legged_robot.py](https://github.com/chengxuxin/extreme-parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot.py); [extreme-parkour rsl_rl ppo.py](https://github.com/chengxuxin/extreme-parkour/blob/HEAD/rsl_rl/rsl_rl/algorithms/ppo.py)

**Rudin 2022**
- "the value function bootstrapping was introduced to mimic an infinite horizon problem, which is necessary in the case of velocity tracking since there is no termination condition… However, in our case, the time until task termination is clearly defined and provided as input to both the actor and the critic. We, therefore, remove the bootstrapping and obtain more stable training." [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- The episode was also shortened from 20 s to 6 s, and the batch was raised to 4096 × 48 steps. The task reward is only collected "if the robot does not crash until then". [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)

**ANYmal Parkour**
- Locomotion skills terminate on base collision or foot force > 1500 N, with a **−200** penalty.
- The climbing-down skill adds "a termination condition on high impact forces on the feet… essential to get a transferable motion".
- The navigation episode "is also terminated if the robot falls or the contact forces are too high". Its goal reward is given only on the last step, and remaining time is an observation.
- [ANYmal Parkour arXiv 2306.14874](https://arxiv.org/abs/2306.14874)

**Other works**
- Risky Terrains: a constant −200 "for early termination". Thigh/shank collisions are a −1 per-step penalty, not a termination. [Risky Terrains arXiv 2311.10484](https://arxiv.org/abs/2311.10484)
- ABS: undesired collisions (base, thighs, calves, horizontal foot) are a −100 per-step penalty. The text does not state whether they also terminate. Goal-reaching episodes last 7–9 s and the policy observes time left T − t. [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- CaT (Constraints as Terminations, 2024):
  - Constraint violations become *stochastic* terminations δ = max_i p_i^max·clip(c_i⁺/c_i^max, 0, 1).
  - **Hard** constraints (base/knee contact, foot contact forces) use p^max = 1.
  - **Soft** constraints use p^max increased from 0.05 to 0.25 over training. Rationale: "the RL algorithm might violate [them] during exploration and learn to recover from"; high p^max "might lead to overly conservative exploration".
  - The method **assumes positive rewards r ≥ 0**, so that termination is always a loss.
  - [CaT arXiv 2403.18765](https://arxiv.org/abs/2403.18765)
- SoloParkour extends CaT off-policy, using p^max to soften constraints early in training. [SoloParkour arXiv 2409.13678](https://arxiv.org/abs/2409.13678)
- PASSAGE (humanoid G1 cluttered traversal, 2026) explicitly separates true terminations from time-limit truncations in GAE: "true terminals suppress bootstrapping, whereas truncations retain the next-state bootstrap but stop advantage recursion" (γ = 0.98, λ = 0.95). Evaluation episodes are ≤ 60 s. Falls are low root height or excessive tilt, and contact is a separate metric. [PASSAGE arXiv 2609.18732](https://arxiv.org/abs/2609.18732)
- Click-and-Traverse evaluation success requires "reach[ing] within 0.1 m of the target location in 5 seconds without colliding with any obstacle". The paper text does not state the training termination or bootstrap rules. [CAT/HumanoidPF arXiv 2601.16035](https://arxiv.org/abs/2601.16035)

### Inferences
- The combination in our setup (contact terminates, timeout bootstrapped) is atypical:
  - Our contact is a hard termination; other works make it a penalty or a stochastic/soft termination.
  - Our timeouts are bootstrapped, so they are effectively free.

  With a bootstrapped timeout and a positive or neutral per-step reward, standing in front of the obstacle is valued like an infinite-horizon "alive" state. Attempting risks a terminal cutoff, so refusing can be the value-maximizing policy.
- Two published ways out:
  - Do **not** bootstrap at timeouts and give time-to-go as an observation (Rudin 2022, ANYmal Parkour, ABS, Risky Terrains). A finite deadline then makes waiting costly when combined with an end-of-episode goal reward.
  - Keep bootstrapping but make collisions non-terminal or soft-terminal (legged_gym, RPL, Extreme Parkour, CaT soft constraints).
- Extreme Parkour's choice to label goal completion as a timeout means success yields bootstrapped future value. This is analogous to treating success as non-terminal continuation, so reaching the goal is never worse than lingering.

### Gaps
- Whether ABS terminates on collision: the paper text found here only specifies the −100 penalty.
- Humanoid Parkour Learning termination rules: not stated in the paper, and no code was examined.
- Whether the HumanoidPF/CAT training code terminates on contact and bootstraps timeouts: not stated in the paper. The local CAT code base would need to be checked.

## Q3. Curricula: obstacle difficulty, promotion/demotion rules, and the soft-obstacle/penetration curriculum

### Takeaway
Nearly all works use the legged_gym "game-inspired" per-robot terrain-level curriculum. The demotion rule is distance-based: a robot that does not move far enough is sent to an easier level. Stalling therefore directly removes the robot from hard obstacles, which keeps the batch full of attempts it can solve. RPL adds the best-known exploration fix: obstacles are initially **penetrable** (soft dynamics) with a penetration penalty and a penetration-based difficulty curriculum, then are hardened for fine-tuning.

### Cited Findings
**legged_gym / Rudin 2021**
- Promote if distance > terrain_length/2. Demote if distance < 0.5·‖cmd‖·T_episode. Robots solving the max level are sent to a random level "to increase the diversity and avoid catastrophic forgetting".
- Defaults: 10 rows (levels) × 20 columns (types), start at max_init_terrain_level = 5, episode 20 s.
- Sources: [Learning to Walk in Minutes arXiv 2109.11978](https://arxiv.org/abs/2109.11978); [legged_gym legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)

**Robot Parkour Learning (paper)**
- Two-stage RL.
  - Stage 1 (soft dynamics): "we set the obstacles to be penetrable so the robot can violate the physical dynamics in the simulation by directly go through the obstacles without get stuck near the obstacles as a result of local minima of RL training with the realistic dynamics." The reward is r_skill + r_penetrate.
  - Stage 2: "fine-tune … on the realistic hard dynamics constraints … using only the general skill reward r_skill", with obstacle properties randomly sampled.
  - Timing: about 12 h soft and 6 h hard on one RTX 3090.
  - [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- Penetration curriculum:
  - Per robot, after each reset: "If the penetration reward is over a threshold, we increase the difficulty score s… by one unit (0.05); if lower, then we decrease it."
  - s starts at 0 with a maximum of 1. The obstacle parameter is (1 − s)·l_easy + s·l_hard.
  - Training ranges: climb height 0.2→0.45 m, leap gap 0.2→0.8 m, crawl clearance 0.32→0.22 m, tilt width 0.32→0.28 m.
  - [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- Ablation:
  - "Both RND and Oracles w/o Soft Dyn cannot make any learning progress on climbing and leaping."
  - Oracles without soft dynamics did learn crawl and tilt, the clearance-type obstacles.
  - Soft-dynamics oracles reach about 95% average success.
  - [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)

**Robot Parkour Learning (released code)**
- Move-up requires `moved` (distance > 1.5 × block length) **and** normalized penetration volume/depth below the "harder" thresholds.
- Move-down if *not moved* or penetration exceeds the "easier" thresholds. Penetration sums are normalized by the obstacle depth actually passed.
- Leap thresholds: volume harder 9000 / easier 10000; depth harder 300 / easier 5000.
- `virtual_terrain=True` toggles penetrable obstacles, and there is a separate optional `no_moveup_when_fall`.
- Leap fine-tuning resumes from the virtually trained checkpoint with entropy_coef 0.0 and clip_min_std 0.2.
- Sources: [parkour legged_robot_field.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot_field.py); [parkour a1_leap_config.py](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/a1/a1_leap_config.py)
- Tracks are connected end to end, so a robot that finishes one obstacle continues to the next, harder one. The map is 800 tracks in a 20 × 40 grid with a linear difficulty curriculum. [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)

**Extreme Parkour**
- Paper: promote if the robot traverses more than half the length; demote if it travels less than half the expected distance v_cmd·T. [Extreme Parkour arXiv 2309.14341](https://arxiv.org/abs/2309.14341)
- Code: promote if dist > 0.8·cmd·T, demote if dist < 0.4·cmd·T, with T = 20 s. max_init_terrain_level is 5. [extreme-parkour legged_robot.py](https://github.com/chengxuxin/extreme-parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot.py)

**Other works**
- Humanoid Parkour Learning:
  - 10 rows of difficulty × 40 columns over 10 obstacle types, with parameter (1 − i/10)·l_easy + i/10·l_hard.
  - Promote if the robot "safely finishes the task and moves at least 3/4 distance"; demote "if the robot fails at a distance smaller than 1/2".
  - Virtual obstacles plus a penetration penalty are kept on leap/jump terrains.
  - [Humanoid Parkour arXiv 2406.10759](https://arxiv.org/abs/2406.10759)
- ANYmal Parkour:
  - Per-skill curricula: gap size up to 1 m; box height up to 1 m for climbing up and down; passage height down to 0.4 m for crouching.
  - Skills train on 80% obstacle and 20% rough terrain.
  - Navigation: "we first place the global targets close to the robots' starting positions and then move them further away on the terrain as the reward increases."
  - [ANYmal Parkour arXiv 2306.14874](https://arxiv.org/abs/2306.14874)
- Risky Terrains: the standard promote/demote/random-at-top rule suffered from "difficulty gaps between levels". The fix was to relax demotion: "the robot only gets demoted when the policy cannot outperform random actions in the remaining distance to the target… the robot can keep interacting with the same level until it overcomes it." [Risky Terrains arXiv 2311.10484](https://arxiv.org/abs/2311.10484)
- ABS:
  - Promote if final d_goal < σ_tight (0.5 m); demote if d_goal > σ_soft (2 m); random level when promoted past the top.
  - Higher levels add more obstacles (0–8 cylinders of radius 0.4 m), rougher terrain (0–7 cm) and stronger ERFI-50 torque perturbation.
  - [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- CaT soft-constraint schedule: p^max goes from 0.05 to 0.25 for soft constraints, i.e. a curriculum on how strictly violations end the episode. [CaT arXiv 2403.18765](https://arxiv.org/abs/2403.18765)
- Click-and-Traverse:
  - "a layout-agnostic difficulty factor to control obstacle complexity, such as the number and size of boxes" with "progressively increasing scene difficulty".
  - Specialists are trained per scene: 32,768 envs and 5M episodes, on 139 3D-FRONT crops plus 216 procedural scenes. Scenes "empirically found to be non-traversable are manually filtered out".
  - Goals are sampled on a 2 m circle.
  - [CAT/HumanoidPF arXiv 2601.16035](https://arxiv.org/abs/2601.16035)
- Lee 2024: training goals come from paths on a navigation graph (Dijkstra) over Wave Function Collapse-generated worlds, so goals are guaranteed feasible. Look-ahead distance is sampled in [5, 20] m. [Lee 2024 arXiv 2405.01792](https://arxiv.org/abs/2405.01792)

### Inferences
- Distance-based demotion makes stalling self-limiting at the population level. A refusing robot is demoted to an easier obstacle it does attempt, until the policy succeeds there. If our curriculum promotes or holds on "no collision" or survival rather than on progress, refusers stay on hard levels, or worse, get promoted, which reinforces refusal.
- RPL's soft-dynamics stage targets exactly the "stand in front of the hurdle" local minimum. While obstacles are penetrable, contact cannot end or block the attempt. The penetration penalty, plus a curriculum that hardens geometry when penetration is low, gradually reintroduces the constraint. Its ablation suggests climbing and leaping (hurdle-like) needed it, while crawl and tilt (clearance-type, similar to narrow gaps) could be learned without it.

### Gaps
- No paper reported a quantitative comparison of soft vs hard dynamics for humanoids specifically. Humanoid Parkour uses virtual obstacles only as a safety margin on leap/jump, not as an exploration stage, per the paper text.
- Exact CAT curriculum promotion and demotion rules are not given in the paper.

## Q4. Explicit mentions of stalling, refusing or conservative behavior, and how they were fixed

### Takeaway
Waiting or refusing is a widely acknowledged failure mode. The documented fixes are:
- explicit stall / "don't wait" penalties (ETH line, ABS);
- direction-of-motion bias rewards (often annealed);
- world-frame progress tracking (Extreme Parkour);
- penetrable obstacles early (RPL);
- softened or stochastic constraint terminations (CaT);
- feasible-goal sampling (Lee 2024);
- relaxed or distance-based curricula (Risky Terrains, ANYmal Parkour);
- revisit penalties (Lee 2024).

### Cited Findings
- Rudin 2022, on why discounting plus end-of-episode reward causes waiting: "it is beneficial to push the negative penalties as far into the future as possible. Therefore, the policy learns to wait until the last moment when it suddenly runs fast towards the target… we add a small penalty waiting while being far away from the target", i.e. r_stall = −1. They also add r_bias, removed once r_task reaches 50%. [Rudin 2022 arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- ANYmal Parkour, navigation: "Without this step [curriculum moving targets farther], the robot struggles to discover the correct behaviors and gets stuck in front of larger obstacles." Proposed future fix: pre-training with expert demonstrations or brute-force search. [ANYmal Parkour arXiv 2306.14874](https://arxiv.org/abs/2306.14874)
- Robot Parkour Learning names the hard-dynamics local minimum as getting "stuck near the obstacles as a result of local minima of RL training". Soft dynamics fixes it. [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- RPL also reports emergent re-trying: after a failed climb attempt, "the robot will push itself away from the obstacle to ensure adequate run-up space for subsequent attempts". This emerges because failure is not terminal. [RPL arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- Extreme Parkour ablation (NoInner, base-frame instead of world-frame inner-product tracking): the robot learns "to walk around the obstacle instead of getting over it", and on steps "all it learns is a colliding and retrying behavior". [Extreme Parkour arXiv 2309.14341](https://arxiv.org/abs/2309.14341)
- ABS: the stall term "penalizes the robot for time waste", and r_agile is designed so "the robot has to either run fast or stay at the goal". ABS also notes velocity-tracking planners "may have to be conservative". [ABS arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- CaT: "CaT (Tracking Rewards) often refuses to walk over the stairs sideways." The variant that formulates the task itself as constraints/terminations ("Tracking Constraints") "is more prone to explore unsafe behaviors to fulfill the task constraints, resulting in better success rates at the expense of more constraint violations." CaT also notes that high termination probability "might lead to overly conservative exploration". [CaT arXiv 2403.18765](https://arxiv.org/abs/2403.18765)
- Lee 2024: training on arbitrary goals made "the policy become overly conservative with distant goals. This is due to the high occurrence of infeasible goals, leading to increased failures during training." The fix was graph-based path sampling so goals are feasible, a dense progress reward annealed to sparse, and a position-buffer revisit penalty. [Lee 2024 arXiv 2405.01792](https://arxiv.org/abs/2405.01792)
- Risky Terrains: robots "get stuck on high-difficulty" terrains and the network "forgets the samples on low-difficulty terrains". Symmetry augmentation was the most critical fix, then relaxed demotion and curiosity rewards. [Risky Terrains arXiv 2311.10484](https://arxiv.org/abs/2311.10484)
- Parkour in the Wild: standard RL from scratch "manages to solve a sub-set of the terrains while remaining completely stuck on others". Their fix is multi-expert distillation followed by RL fine-tuning, which keeps "Don't wait" in the reward. [Parkour in the Wild arXiv 2505.11164](https://arxiv.org/abs/2505.11164)
- SoloParkour excludes trajectories where the robot "gets endlessly stuck on a high obstacle" from its constraint statistics. This acknowledges stuck behavior even in the final policy. [SoloParkour arXiv 2409.13678](https://arxiv.org/abs/2409.13678)
- Humanoid clutter works that avoid the issue with demonstrations:
  - MTC tracks collision-free reference trajectories retargeted from VR human demonstrations, which reaches a 70.2% collision-free rate on its benchmark. [MTC arXiv 2609.21107](https://arxiv.org/abs/2609.21107)
  - PASSAGE uses a planner–tracker trained on 100 h of scene-aligned human motion. [PASSAGE arXiv 2609.18732](https://arxiv.org/abs/2609.18732)
  - In both, the reference motion supplies forward progress, so refusal is not a reward-optimal option for the tracker.

### Inferences
Candidate fixes for the CAT G1 refusal problem, ordered by how directly the literature supports them:
1. **Make obstacle contact non-terminal**, as a penalty, or soft/stochastic à la CaT with a small p^max early. Keep hard termination only for falls and torso crashes.
2. **Add a stall / "don't wait" penalty**, e.g. −1·1(‖v‖ < 0.1–0.2 m/s ∧ d_goal > 0.5–1 m). Also add a direction-of-motion cosine reward, possibly annealed.
3. **Remove timeout bootstrapping if time-to-go is observed**, or otherwise make timeouts without reaching the goal cost something. The finite-horizon formulation of Rudin 2022, ANYmal Parkour, ABS and Risky Terrains is the published pattern.
4. **Make demotion progress-based**, so stallers are demoted rather than held or promoted.
5. **Use an RPL-style soft-dynamics pre-stage**: penetrable hurdles or gaps with a velocity-weighted penetration penalty and a penetration-threshold curriculum, then harden them.
6. **Ensure goals are feasible** (Lee 2024), and use revisit penalties against loitering.

Further, CaT's analysis implies a sign condition. Termination only deters violations when the remaining expected reward is positive. If per-step rewards are mostly negative (penalties dominate), termination becomes attractive. If they are positive (alive), termination is costly and refusal is attractive whenever timeouts are free. Either way, the balance must be designed explicitly.

### Gaps
- No paper found here studies the exact interaction of "contact = terminate" with "timeout = bootstrapped" for humanoids, nor quantifies refusal rates.
- The Click-and-Traverse paper does not mention refusal or stalling. Whether its released configuration terminates on collision must be checked in the code.
