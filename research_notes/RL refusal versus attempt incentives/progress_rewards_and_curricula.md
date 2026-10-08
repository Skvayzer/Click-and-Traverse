# Progress rewards, stall penalties, and curricula against "stalling" local optima in goal-reaching RL

Scope: (a) progress/goal reward shaping that makes stalling costly without new exploits; (b) curricula and exploration methods for risky maneuvers that rarely succeed at first. Context: G1 humanoid, PPO+GAE, ~+15/step positive reward, guidance-field velocity alignment, contact = termination, 20 s timeouts bootstrapped, the policy stands still in front of obstacles, and narrow-gap success is 0-15%.

Source texts were read in full (PDF to text) for: Ng 1999, Grzes 2017, Savva 2019, Wijmans 2020, Pardo 2018, Rudin 2021 (legged_gym), Rudin 2022 (Advanced Skills), Hoeller 2023/24 (ANYmal Parkour), Zhuang 2023 (Robot Parkour Learning), Cheng 2023 (Extreme Parkour), Zhuang 2024 (Humanoid Parkour), Florensa 2017 (Reverse Curriculum), Florensa 2018 (Goal GAN), Salimans & Chen 2018, and Peng 2021 (AMP).

---

## Q1. Potential-based reward shaping (PBRS) with a geodesic distance-to-goal potential: invariance conditions, discounting, episode ends, and navigation practice

### Takeaway
PBRS (F = γΦ(s') − Φ(s)) is the only additive shaping that is guaranteed to leave optimal policies unchanged. That guarantee cuts both ways: PBRS cannot make stalling suboptimal if stalling is already optimal under the base reward. It only speeds up learning or reshapes exploration. In episodic tasks with early termination, invariance needs Φ(terminal) = 0, which creates a visible reward on the terminal step. Navigation practice (Habitat) uses the undiscounted geodesic-progress form d_{t−1} − d_t, together with a success bonus and a small time penalty.

### Cited Findings
- **Definition and theorem (Ng, Harada, Russell, ICML 1999).** A shaping function F is "potential-based" if there is a real-valued Φ: S → R such that F(s,a,s') = γΦ(s') − Φ(s) for all s, a, s'. Being potential-based is "a necessary and sufficient condition for it to guarantee consistency with the optimal policy". Sufficiency: "every optimal policy in M' will also be an optimal policy in M (and vice versa)". Necessity: if F is not potential-based, then "there exist (proper) transition functions T and a reward function R ... such that no optimal policy in M' is optimal in M." — [Ng et al. 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **Assumptions in Ng 1999.** In the undiscounted case (γ = 1) the paper assumes a distinguished absorbing state s0 ("the MDP 'stops' after a transition into s0, with no further rewards thereafter") and that all policies are proper, meaning they reach s0 with probability 1. For infinite state spaces Φ should be bounded so that F is bounded. — [Ng et al. 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **Motivating failure of non-potential progress rewards (Ng 1999).** In Randløv and Alstrøm's bicycle task the agent was given positive reward for progress toward the goal, and it "learned to ride in tiny circles near the start state because no penalty was incurred for riding away from the goal." In general, any state cycle with net positive shaping reward can be exploited. — [Ng et al. 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **Best potential and distance-based potentials (Ng 1999).** Corollary 2 suggests Φ(s) = V*_M(s), with Φ(s0) = 0 in the undiscounted case. In a 10x10 grid with −1 per step and 80% action success, they used a crude V* estimate Φ0(s) = −manhattan(s, goal)/0.8 (expected steps to the goal; the minus sign is lost in the PDF text extraction but follows from the −1/step reward). Both Φ0 and 0.5·Φ0 "significantly helped speed up learning", and the gains were "even more dramatic" on a 50x50 grid. Shaping can help "even if Φ is far from V*". — [Ng et al. 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **Episodic and terminal-state problem (Grześ, AAMAS 2017).** With PBRS the shaped return of a trajectory is U(s̄) + γ^N Φ(s_N) − Φ(s0). The γ^N Φ(s_N) term "depends on actions (because the terminal states depend on actions executed earlier in the trajectory), and, as a result, this term can modify the policy." Their counterexample has two terminal states g1 (r = 0) and g2 (r = 100). The shaped rewards become γ·1000 vs 100 + γ·10, so the policy is altered for γ > 10/99. Fix: "require Φ(s_N) = 0 whenever the reinforcement learning trajectory is terminated at state s_N". This applies "only when s_N is a terminal state for the trajectory"; the same state keeps its potential when it is non-terminal elsewhere. The paper notes that "multiple terminal states arise naturally in tasks where the trajectories are stopped after a fixed number of steps." — [Grześ 2017](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf)
- **Habitat PointNav reward (Savva et al., ICCV 2019).** "r_t = s + d_{t−1} − d_t + λ if goal is reached; d_{t−1} − d_t + λ otherwise", where d_t is the geodesic distance to the goal, s is the success reward and λ is a time penalty "to encourage efficiency". "In our experiments s is set to 10 and λ is set to −0.01." Geodesic distances come from a navmesh/grid at 0.01 m resolution. Episodes have a geodesic shortest-path distance between 1 m and 30 m. — [Savva et al. 2019](https://arxiv.org/abs/1904.01201)
- **DD-PPO PointNav reward (Wijmans et al., ICLR 2020).** "The agent receives terminal reward r_T = 2.5 SPL, and shaped reward r_t(a_t, s_t) = −Δgeo_dist − 0.01". Here the success bonus is scaled by path efficiency (SPL), and success means stopping within 0.2 m of the target. Training used γ = 0.99 and GAE τ = 0.95. The authors did not normalize advantages because they found "this leads to instabilities". The result was near-perfect navigation "within 3% of" a shortest-path oracle. — [Wijmans et al. 2020](https://arxiv.org/abs/1911.00357)
- **Habitat observation on easy episodes acting as a curriculum.** Gibson-trained agents beat Matterport3D-trained agents even on Matterport3D because "Gibson agents are trained on 'easier' episodes and encounter positive reward more easily during random exploration, thus bootstrapping learning." — [Savva et al. 2019](https://arxiv.org/abs/1904.01201)
- **Exploit of un-anchored velocity or progress terms in parkour.** Extreme Parkour tracks velocity toward the next waypoint in the world frame, r_tracking = min(⟨v, d̂_w⟩, v_cmd) with d̂_w = (p − x)/‖p − x‖. A world frame is used rather than the base frame "to prevent the robot from exploiting the reward and learning the unintended behavior of turning around the obstacle." In an ablation with base-frame velocity tracking (NoInner), the robot walked around hurdles. On steps it learned only "a colliding and retrying behavior". — [Cheng et al. 2023 (ICRA 2024)](https://arxiv.org/abs/2309.14341)

### Inferences
- **PBRS cannot remove the stall optimum.** Because PBRS preserves optimal policies (Ng 1999), a geodesic-progress potential cannot by itself make standing still suboptimal if standing still is optimal under the base reward. That is likely the case here: +15/step alive-like reward, contact termination, and bootstrapped timeouts. To make stalling costly you must change the base reward (stall penalty, success bonus, smaller alive term, or time penalty). PBRS then helps credit assignment toward the goal.
- **Sign of Φ under γ < 1 (derived from the formula, not stated in a source).** With Φ = −k·d_geo, a stationary agent gets F = (1 − γ)·k·d > 0 every step, which is a small reward for standing still far from the goal. With Φ = k·(D_max − d_geo) ≥ 0, a stationary agent gets −(1 − γ)·k·(D_max − d) < 0. For γ = 0.99, k = 1, d = 2 m, either term is about ±0.02/step, negligible next to +15/step. Prefer the positive-offset form, or use the γ = 1 telescoping form d_{t−1} − d_t as Habitat does.
- **Termination interaction.** Under the invariance-preserving rule Φ(s_terminal) = 0 (Grześ), the shaping sum over any episode telescopes to −Φ(s0). Total shaping is then the same whether the agent reaches the goal, crashes, or stalls, so it adds no incentive to attempt. The common practical choice is not to zero Φ at contact termination, i.e. pay raw d_{t−1} − d_t until death. Then the agent keeps credit for progress made before crashing, which mildly favors attempting over standing still. This is not invariant, but it is arguably the bias you want. Choose deliberately and keep the timeout step as a normal bootstrapped transition.
- **Geodesic distance from a fast-marching field is the right Φ for traps.** It is the analogue of Habitat's geodesic distance. Euclidean distance creates local optima at walls and gap entrances. The Ng 1999 bicycle example and the Extreme Parkour "turning around the obstacle" exploit both show that non-potential velocity-alignment terms (like the current guidance-field alignment) can be farmed. Velocity alignment rewards motion, not net progress, so it admits cycles such as oscillating back and forth at a gap mouth.

### Gaps
- No source found that quantitatively compares PBRS geodesic progress against velocity-alignment rewards for legged or humanoid obstacle traversal.
- Did not retrieve Wiewiora 2003 (PBRS equivalent to Q-value initialization) or Devlin & Kudenko 2012 (dynamic PBRS). Grześ cites the existence of non-stationary-potential results but this report did not verify them.

---

## Q2. Step/time penalties, stall penalties, and success bonuses: magnitudes, effects on risk-taking, and failure modes (including early-termination exploits)

### Takeaway
The sign of the per-step reward decides whether termination is attractive or something to avoid. Large positive per-step rewards make any action that risks termination expensive, and so encourage "safe" stalling. Large negative per-step rewards encourage suicide. Robotics navigation work handles this with:
- time-limited sparse goal rewards plus an explicit stall penalty (−1 per step when slow and far from the goal);
- large termination penalties (for example −200) set against a goal reward;
- in legged_gym, clipping the total reward at ≥ 0 before adding termination.

Small time penalties (−0.01 against a success bonus of 2.5-10) are standard in Habitat navigation.

### Cited Findings
- **Sign and termination.** "when using negative rewards, termination is a good thing for the agent. When using positive rewards, it is bad and will need to be avoided." The author also describes a hovering failure with distance-based rewards (for example e^{−d}): the "value of almost reaching our goal is ... much higher than the value of actually reaching the goal". Remedies: goal reward larger than the near-goal value stream, truncation (self-loop) instead of termination, or delta-progress shaping instead of absolute-distance reward. (Researcher blog, 2025; secondary source.) — [Voelcker 2025](https://cvoelcker.de/blog/2025/reward-functions/)
- **legged_gym default.** `only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)`. In the code the summed reward is clipped `torch.clip(self.rew_buf, min=0.)`, and the termination reward is added "after clipping". Default `termination = -0.0`, `episode_length_s = 20`, `tracking_sigma = 0.25` (tracking reward = exp(−error²/σ)). — [legged_gym config](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py); [Rudin et al., CoRL 2021](https://arxiv.org/abs/2109.11978)
- **Timeout bootstrapping (Rudin 2021).** "Resets based on failure or reaching a goal are not a problem because the critic can predict them. However, a reset based on a time out can not be predicted (we do not provide episode time in the observations)." They bootstrap the critic on time-outs. — [Rudin et al. 2021](https://arxiv.org/abs/2109.11978)
- **Time limits (Pardo et al., ICML 2018).** For time-limited tasks the TD target should be y = r at real terminations and r + γ·v̂(s', T − t − 1) otherwise. A time-unaware agent faces state aliasing. They propose (1) time-awareness (remaining time as input) for genuinely time-limited tasks and (2) partial-episode bootstrapping at timeouts when the time limit is only a training artifact. — [Pardo et al. 2018](https://arxiv.org/abs/1712.00378)
- **Rudin et al. 2022 (IROS), "Advanced Skills by Learning Locomotion and Local Navigation End-to-End". Most directly relevant source.**
  - **Task reward.** r_task = (1/T_r)·1/(1 + ‖x_b − x*_b‖²) if t > T − T_r, else 0. T = 6 s, T_r = 1 s, targets 1-5 m away. "T_r needs to be long enough to force the policy to stop at the target in a stable configuration. Otherwise, the policy learns to jump or lean towards the target at the last moment".
  - **Exploration (bias) reward.** r_bias = (ẋ_b · (x*_b − x_b)) / (‖ẋ_b‖·‖x*_b − x_b‖), i.e. the cosine of velocity toward the target. It is "removed once r_task reaches 50% of it's maximum value" so it "doesn't constrain the final solution".
  - **Stalling penalty.** r_stall = −1 if ‖ẋ_b‖ < 0.1 m/s and ‖x_b − x*_b‖ > 0.5 m, else 0. Rationale: with γ < 1 and per-step penalties, "it is beneficial to push the negative penalties as far into the future as possible. Therefore, the policy learns to wait until the last moment when it suddenly runs fast towards the target."
  - **Stability.** Time-until-end is given to actor and critic, timeout bootstrapping is removed ("we therefore remove the bootstrapping and obtain more stable training"), episodes are shortened from 20 s to 6 s, and the batch is doubled (4096 robots × 48 steps). The sparse terminal reward made the critic harder to learn: "a random seed change can cause a failed training run."
  - **Results versus velocity tracking (95% success thresholds).** Gap 1.2 m vs 0.15 m, pit 0.95 m vs 0.1 m, obstacles 0.85 m vs 0.35 m, stairs 0.4 m vs 0.22 m. Continuous position tracking "completely fails to navigate obstacles with a sufficient reliability". Terrain curriculum: success means finishing within 0.5 m of the target.
  - Source: [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- **ANYmal Parkour (Hoeller et al., arXiv 2023 / Science Robotics 2024), locomotion reward table.**

  | Term | Definition | Weight |
  |---|---|---|
  | Position tracking | 1_{t*<1}(1 − 0.5‖r_xy − r*_xy‖) | 10 |
  | Heading tracking | 1_{t*<1}(1 − 0.5‖ψ − ψ*‖) | 5 |
  | "Don't wait" | 1(‖v_b‖ < 0.2) | −1 |
  | "Move in direction" | cos⟨v_b, r* − r⟩ | 1 |
  | Stand at target | | −0.5 |
  | Knee/shank collision | | −1 |
  | Stumble | | −1 |
  | Termination | base collision or foot force > 1500 N | −200 |

  Small regularizers (e.g. action rate −0.01) are also included. Navigation: "the distance-to-goal penalty is only activated on the last time-step," because "This sparse formulation allows the policy to explore the terrain to find safer paths and take its time where needed." Goal-distance curriculum: "we first place the global targets close to the robots' starting positions and then move them further away on the terrain as the reward increases." — [Hoeller et al. 2023](https://arxiv.org/abs/2306.14874)
- **Robot Parkour Learning (Zhuang et al., CoRL 2023).** r_skill = r_forward + r_energy + r_alive, where r_forward = −α1|v_x − v_x^target| − α2|v_y|² + α3·e^{−|ω_yaw|}, r_energy = −α4 Σ_j |τ_j q̇_j|², r_alive = 2 per step, and target speed ≈ 1 m/s. This shows a positive alive bonus combined with a forward-velocity error penalty, which makes standing still cost |0 − 1| per step. — [Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)
- **Habitat magnitudes.** Success +10 (Savva), or +2.5·SPL (Wijmans), against a time penalty of −0.01/step and progress of about one unit per metre. — [Savva et al. 2019](https://arxiv.org/abs/1904.01201); [Wijmans et al. 2020](https://arxiv.org/abs/1911.00357)

### Inferences
- **Diagnosis of the current setup.** With about +15/step mostly unconditional, contact termination, and bootstrapped timeouts, the critic sees "standing still" as an effectively infinite stream worth about 15/(1 − γ). With γ = 0.99 that is roughly 1500. An attempt with success probability p loses about (1 − p) of that stream on contact. Unless success (reaching the goal) pays more than the forgone stream, refusal is the rational optimum, not just a local one. This is the "positive rewards make termination bad" regime (Voelcker) made worse by timeout bootstrapping (Rudin 2021; Pardo 2018). If reaching the goal also ends the episode, the success bonus must exceed the alive stream from the goal state, or the agent should be allowed to keep collecting reward while standing at the goal (truncation or self-loop).
- **Fixes with published precedent.**
  1. Add a stall penalty gated on being far from the goal and slow, about −1/step (Rudin 2022; Hoeller 2023 "Don't wait"). Make it comparable to the per-step positive terms; with +15/step, −1 may be too weak. Rescale so the positive terms are smaller, or make the alive-like terms conditional on progress.
  2. Replace dense velocity alignment with a time-limited terminal position reward plus time-to-go observation. Remove timeout bootstrapping for the goal task (Rudin 2022).
  3. If termination stays, set a termination penalty and a goal reward on the same scale (ANYmal: +10/+5 tracking over the last second vs −200 termination). Clip non-termination reward ≥ 0 so negative shaping never makes suicide attractive (legged_gym).
- **Alternatively, remove the stall optimum structurally.** Reduce the alive-like positive terms so the per-step base is about 0, and make most of the return come from geodesic progress and success (Habitat style). Watch for the opposite failure: per-step-negative regimes encourage early termination via contact. Guard this with a termination penalty at least as large as the reachable remaining negative return.
- **Time-aware critics.** The "wait then sprint" pathology (Rudin 2022) and time-unaware aliasing (Pardo 2018) suggest feeding remaining episode time to the critic, or at least the actor and critic, if timeouts are kept and the goal task is time-limited.

### Gaps
- No source gives an explicit study of how the ratio of alive bonus to termination penalty to success bonus affects risk-taking in legged robots. The guidance above combines several papers' reward tables.
- The Voelcker post is a secondary, non-peer-reviewed source. The sign/termination point is standard RL lore, but I found no peer-reviewed paper stating it as a theorem in this session.

---

## Q3. Start-state and goal curricula: reverse curriculum, Goal GAN, demo-state resets, adaptive terrain curricula; effectiveness for rare-success skills

### Takeaway
Concentrating training on starts or goals of intermediate difficulty (success between about 10% and 90%) is the most consistently reported fix for rare-success tasks:
- reverse curriculum starting from the goal (Florensa 2017);
- demonstration-state resets moved backward when about 20% succeed (Salimans & Chen 2018);
- per-robot promote/demote terrain levels (Rudin 2021; parkour papers).

Curricula that vary only difficulty but always start from the far start state are weaker. Moving the start backward from success states is what converts exponential exploration into roughly quadratic exploration.

### Cited Findings
- **Reverse Curriculum Generation (Florensa et al., CoRL 2017).**
  - **Assumptions.** (1) the agent can be reset to any start state; (2) at least one goal state s_g is given; (3) the random-action Markov chain has "a minimum degree of reversibility".
  - **Good starts.** S⁰_i = {s0 : R_min < R(π_i, s0) < R_max} with R_min = 0.1 and R_max = 0.9, interpreted as bounds on success probability.
  - **New starts.** Generated by "Brownian motion" rollouts from good starts: horizon T_B = 50, actions ~ N(0, I), M = 10000 candidates subsampled to N_new = 200, plus N_old = 100 replayed old starts. TRPO, γ = 0.998, batch 50,000.
  - **Results.** With uniform start sampling, success was "around 10% for the ring task and 2% for the key task". This meant reliably reaching the goal only from very nearby positions, and "None of the learned policies trained on the original ρ0 learn to reach the goal from more distant start states". Their method reached the goal "from a wide range of far away start states".
  - **Ablations.** Brownian sampling from all starts (no filtering by R_min/R_max) is "still better than not modifying the start state distribution but has a slower learning rate". Asymmetric self-play performed "very poorly" because the start-proposing "Alice" gets stuck in local optima.
  - **Limitation.** Starts grow uniformly outward from a single goal.
  - **Contact exploitation.** The final manipulation policies "learned to exploit the contacts instead of avoiding them".
  - Source: [Florensa et al. 2017](https://arxiv.org/abs/1707.05300)
- **Goal GAN (Florensa, Held, Geng, Abbeel, ICML 2018).** Trains on Goals Of Intermediate Difficulty, GOID_i := {g : R_min ≤ R_g(π_i) ≤ R_max}, with 0.1 and 0.9. "very robust to these hyperparameters (any value of R_min ∈ (0, 0.25) and R_max ∈ (0.75, 1) would yield basically the same result)". A GAN generates new goals in GOID; it is trained with LSGAN plus negative examples. — [Florensa et al. 2018](https://arxiv.org/abs/1705.06366)
- **Learning Montezuma's Revenge from a Single Demonstration (Salimans & Chen, 2018).** Episodes reset to demo states. The reset point τ moves backward "if at least 20% of the rollout workers achieve returns comparable to the provided demonstration" (ρ = 20%). PPO with 1024 workers, about 50 billion frames. Score 74,500, exceeding the demo's 71,500. Their toy analysis says standard RL "scales exponentially in the number of states between rewards. Our method reduces this to quadratic scaling."
  - **Caveats.** The agent "is generally unable to reach the exact state from later on in a demonstration when it starts from an earlier state", so it must generalize between similar states. This worked for Montezuma but "much less well" for Gravitar and Pitfall.
  - **Entropy tuning.** Success "required careful tuning of the coefficient of the entropy bonus used in PPO": too random means too many mistakes; too deterministic means "the agent stops learning because it does not explore alternative actions."
  - Source: [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)
- **Game-inspired terrain curriculum (Rudin et al., CoRL 2021).** Each robot has a terrain type and level. Walking past the terrain border promotes it; moving "less than half of the distance required by its target velocity" demotes it. "Robots solving the highest level are looped back to a randomly selected level to increase the diversity and avoid catastrophic forgetting." In code: move_up = distance > env_length/2; move_down = distance < ‖cmd‖·T·0.5. Defaults: 10 rows (levels), max_init_terrain_level = 5. — [Rudin et al. 2021](https://arxiv.org/abs/2109.11978); [legged_gym config](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- **Robot Parkour Learning (Zhuang et al., CoRL 2023): per-robot difficulty score.** s ∈ [0, 1] moves ±0.05 per reset according to whether the episode-averaged penetration reward is above a threshold. Obstacle size = (1 − s)·l_easy + s·l_hard. For "Tilt" (squeezing through slits narrower than the robot by tilting) the path width goes from 0.32 m to 0.28 m in training. — [Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)
- **Humanoid Parkour Learning (Zhuang, Yao, Zhao, CoRL 2024).** Three stages: (1) a walking policy for forward, sideways and turning commands on fractal-noise terrain; (2) a parkour policy with an auto-curriculum, promoting when the robot "safely finishes the task and moves at least 3/4 distance of the sub-terrain" and demoting when it "fails at a distance smaller than 1/2"; (3) DAgger distillation to depth. — [Zhuang et al. 2024](https://arxiv.org/abs/2406.10759)
- **ANYmal Parkour goal-distance curriculum.** Targets start close and move "further away on the terrain as the reward increases". — [Hoeller et al. 2023](https://arxiv.org/abs/2306.14874)

### Inferences
- **Current width curriculum versus reverse curriculum.** The width curriculum plus "some starts already sideways near the gap" is a partial reverse curriculum. Florensa's results suggest the missing ingredients are (1) adaptive selection of starts at intermediate success (10-90%) instead of a fixed fraction, and (2) growing starts backward from successful in-gap or post-gap states. Brownian rollouts would not work for a humanoid; use states harvested from successful rollouts instead, Salimans-style with ρ ≈ 20%.
- **Concrete reverse-curriculum recipe for the narrow gap.** Seed with states mid-gap and just past the gap (sideways), and estimate per-start success. Keep starts with success in [0.1, 0.9], spawn new starts slightly earlier along successful trajectories, replay old starts (Florensa's N_old ≈ 1/3), and keep a share of nominal starts. Per-start success estimates come cheaply from thousands of parallel envs, as in Rudin's per-robot curriculum.
- **Possible cause of the plateau.** Florensa found unfiltered Brownian expansion slower than filtered good starts. Starts that are always solved (>90%) or never solved (<10%) waste samples. A plateau at 0-15% suggests most nominal-start samples sit in the "never solved" band.
- **Curriculum does not fix the reward optimum.** A start-state curriculum fixes exploration but not the incentive problem from Q2. If refusal is optimal under the reward, a policy that learns the gap from near-gap starts may still refuse from far starts. Fix the incentive first, then curriculum.

### Gaps
- No humanoid-specific reverse-curriculum or demo-reset result for narrow-gap traversal was found.
- Did not retrieve Go-Explore (Ecoffet et al. 2019/2021) or PLR/ACCEL-type automatic curricula, so no numbers are reported for them.

---

## Q4. Exploration methods for risky maneuvers: soft dynamics instead of termination, entropy, motion priors, privileged teacher-student

### Takeaway
The strongest published evidence for learning rare, risky obstacle maneuvers is replacing hard collision termination with penetrable ("soft") obstacles plus a velocity-scaled penetration penalty and a curriculum, then fine-tuning with hard dynamics. Without it, PPO learned 0% on climb and leap. Other levers:
- privileged-observation teachers distilled with DAgger;
- motion priors (AMP) that make agile maneuvers reachable;
- a carefully tuned PPO entropy coefficient;
- temporary exploration bias rewards that are switched off later (Rudin 2022).

### Cited Findings
- **Soft dynamics constraints (Robot Parkour Learning, Zhuang et al., CoRL 2023).**
  - **Mechanism.** Obstacles are made penetrable so the robot can "directly go through the obstacles without get stuck near the obstacles as a result of local minima of RL training with the realistic dynamics". The penalty is r_penetrate = −Σ_p (α5·1[p] + α6·d(p))·v_x, computed at collision points sampled on the body (denser at hips and shoulders).
  - **Why v_x scaling.** It is there "to prevent the robot from exploiting the penetration reward by sprinting through the obstacles to avoid high cumulative penalties over time".
  - **Stages.** Pre-train with soft dynamics plus a curriculum, then fine-tune with hard dynamics using only r_skill.
  - **Results (privileged oracles, success %).**

    | Training | Climb | Leap | Crawl | Tilt |
    |---|---|---|---|---|
    | Without soft dynamics | 0 | 0 | 93 | 86 |
    | With soft dynamics | 95 | 82 | 100 | 100 |

  - **RND comparison.** "RND struggles to learn meaningful behaviors with scenarios that require fine-grained maneuvers such as crawling through a thin slit". Both RND and oracles without soft dynamics "cannot make any learning progress on climbing and leaping."
  - Source: [Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)
- **Humanoid Parkour (Zhuang et al., CoRL 2024).** Uses virtual obstacles with r_penetrate = α·Σ(d(p)·‖v(p)‖), α = −5×10⁻³, to keep the policy away from edge-exploiting behavior. It also uses a footstep reward r_step = 6·(−ln‖d_x‖) on stairs. Reward terms involve no motion reference. — [Zhuang et al. 2024](https://arxiv.org/abs/2406.10759)
- **Temporary exploration bias (Rudin 2022).** The cosine-to-target velocity reward is removed once the task reward reaches 50% of its maximum. — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- **Entropy.** Salimans & Chen needed "careful tuning of the coefficient of the entropy bonus used in PPO" to balance mistakes against stagnation. — [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)
- **AMP (Peng et al., ACM TOG / SIGGRAPH 2021).**
  - **Reward.** r_t = w^G r^G_t + w^S r^S_t with w^G = w^S = 0.5 for all tasks. r^S comes from a discriminator trained on unstructured motion clips.
  - **Obstacle task.** A humanoid traversed an obstacle course with gaps, steps and overhead obstacles "that the character must duck under". Given locomotion and rolling clips, "The character learns to leap over obstacles such as gaps... it transitions into a rolling behavior in order to pass underneath the obstacles." The task return for Run + Leap + Roll was relatively low (0.27 ± 0.10 normalized).
  - **Initialization and termination.** Reference state initialization (starting episodes from random dataset states) and early termination on non-foot ground contact.
  - Source: [Peng et al. 2021](https://arxiv.org/abs/2104.02180)
- **Privileged teacher, then student.** Robot Parkour Learning trains specialized privileged oracles per skill and distills them with DAgger into one vision policy; the parkour policy reaches 86/80/100/73% on climb/leap/crawl/tilt. Humanoid Parkour and Extreme Parkour also train with privileged terrain info (scandots, waypoint headings) and then distill. — [Zhuang et al. 2023](https://arxiv.org/abs/2309.05665); [Zhuang et al. 2024](https://arxiv.org/abs/2406.10759); [Cheng et al. 2023](https://arxiv.org/abs/2309.14341)
- **Symmetry augmentation.** In ANYmal Parkour the position-tracking policy developed asymmetric motions, for example climbing only forwards and turning around otherwise. They fixed this with symmetric data augmentation across front-back and left-right mirroring. — [Hoeller et al. 2023](https://arxiv.org/abs/2306.14874)

### Inferences
- **Soft obstacles are the closest analogue to this problem.** Contact termination is what makes attempts at the narrow gap so expensive. Robot Parkour's "Tilt" skill (squeezing through slits narrower than the body by tilting) is nearly the same maneuver as turning sideways in a narrow gap, and soft dynamics raised oracle success from 86% to 100%; for climb and leap, from 0% to 95%/82%. Suggested recipe: make gap walls, hurdles and beams penetrable during pre-training, with a velocity-scaled penetration-depth penalty instead of termination and a curriculum on the threshold; then fine-tune with hard contacts.
- **Motion priors for sideways stepping.** Since the G1 setup already has AMASS motion data (sidestep/sidle clips exist in the scratch area), an AMP-style style reward or reference-state initialization from sidestepping clips could make the sideways maneuver reachable by random exploration. AMP's own obstacle-course results show that maneuvers like leaping and rolling can emerge when matching clips exist, though task returns stayed modest.
- **Temporary bias rewards.** Any exploration bonus, such as alignment with the guidance field, should be annealed or switched off by a success threshold (Rudin 2022's 50% rule). Otherwise it becomes part of the optimum and can be farmed.

### Gaps
- No direct comparison found of action-noise schedules (std annealing) against entropy bonuses for rare-success legged maneuvers.
- No paper found that applies soft dynamics specifically to humanoid narrow-gap sidestepping. The extrapolation from quadruped "Tilt" is an inference.
- Teacher-student results such as Lee et al. 2020 (quadruped in the wild) and RMA were not retrieved here beyond the RMA baseline row in Zhuang 2023, where RMA scored 74% on Tilt (no other skills reported).
