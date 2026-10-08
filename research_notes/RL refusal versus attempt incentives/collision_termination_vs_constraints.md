# Collision handling in legged-robot RL: termination vs reward penalty vs constrained RL, and how each affects attempting vs refusing a risky maneuver

Naming note: "CaT" below means Chane-Sane et al.'s *Constraints as Terminations* (2024). It is not the project's own "CAT" (Click-and-Traverse) acronym.

Context assumed: Unitree G1, PPO+GAE, about 40k MuJoCo Warp envs, hurdles and narrow gaps, with a hard no-touch requirement that is strictest for the hands. Hard termination on any obstacle contact led to stalling. A non-terminal penalty of 3 per step (against roughly +15 per step of positive reward) also led to stalling, with lower success.

## Q1. CaT (Chane-Sane et al., IROS 2024): how constraint violations become stochastic terminations, how it interacts with PPO, and results

### Takeaway
CaT turns each constraint violation into a termination probability δ = max_i p_i^max · clip(c_i^+/c_i^max, 0, 1). It multiplies rewards by (1−δ) and feeds δ into PPO/GAE as a soft "done". Hard constraints such as knee/base collisions use p_max = 1. Soft constraints ramp p_max from 0.05 to 0.25. CaT beat the N-P3O constrained-RL baseline and naive binary termination (ET-MDP), and it learned parkour on Solo-12. Two caveats matter for us. CaT needs positive rewards, and a 2026 follow-up shows it can collapse into "hovering just outside the goal" when constraints activate exactly where the task must be completed.

### Cited Findings
- **Termination probability (Eq. 6):** `δ = max_{i∈I} p_i^max · clip(c_i^+ / c_i^max, 0, 1)`, with `c_i^+ = max(0, c_i(s,a))`. `c_i^max` is an exponential moving average of the maximum violation seen in the batch: `c_i^max ← τ^c·c_i^max + (1−τ^c)·max_batch c_i^+(s,a)`. — [CaT arXiv 2403.18765](https://arxiv.org/html/2403.18765)
- **Objective (Eq. 4):** `max_π E[ Σ_t ( Π_{t'=0}^{t} γ(1−δ(s_t',a_t')) ) r(s_t,a_t) ]`, which amounts to a state-action-dependent discount. The authors say terminations "are not environment resets, but merely future reward terminations from a policy learning perspective." — [CaT](https://arxiv.org/html/2403.18765)
- **PPO integration (Algorithm 1):** `rewards ← rewards × (1−δ)` and `dones ← δ`, followed by standard PPO/GAE. No extra critic is needed. The method requires positive rewards. — [CaT](https://arxiv.org/html/2403.18765)
- **Why stochastic rather than binary:** "A naive approach is to terminate the future rewards if any constraint is violated ... might be overly conservative with respect to the constraints and impair exploration and learning." Also: "Allowing δ to take values in ]0,1[ enables the agent to learn to recover from constraint violations" and "allows some exploration inside the region of constraint violation." — [CaT](https://arxiv.org/html/2403.18765)
- **Hard vs soft constraints:**
  - Hard constraints (p_max = 1.0) are knee/base collisions ("prohibit collisions to the knee and the base of the robot") and the foot contact-force limit.
  - Soft constraints have "p_i^max increases from 0.05 to 0.25 throughout the course of training". They cover torques, joint velocity and acceleration, action rate, base orientation, air time and similar quantities.
  - "lower values of p_i^max will allow the learning agent to discover higher reward regions of the behavior space."
  - Source: [CaT](https://arxiv.org/html/2403.18765)
- **Terrain-dependent relaxation:** style constraints are active only on flat ground and set to 0 elsewhere, "providing room for the RL algorithm to adapt the learned behavior on more challenging terrains." — [CaT](https://arxiv.org/html/2403.18765)
- **Results, flat terrain in simulation (Table II):**
  - ET-MDP (binary termination on any violation) "entirely fails to learn locomotion policies".
  - N-P3O reached 593.2±49.5 tracking reward with 8%±1% torque violation.
  - CaT reached 682.9±5.8 with 0.5%±0.3%.
  - Source: [CaT](https://arxiv.org/html/2403.18765)
- **Results, real-robot parkour (Table III):** CaT (tracking rewards) had 79.4% success and 0.5% violation. CaT (tracking constraints) had 90.6% success and 1.7% violation. The "style always active" ablation had 34.4% success and 3.7% violation. — [CaT](https://arxiv.org/html/2403.18765)
- **Code:** the official Isaac Lab / Isaac Gym implementation (CleanRL or RL-Games) defines constraints as `ConstraintTerm(func=constraints.contact, max_p=1.0, ...)` for hard constraints and `ConstraintTerm(func=constraints.joint_torque, max_p=0.25, ...)` for soft ones, through a `ConstraintsManager`. — [Gepetto/constraints-as-terminations](https://github.com/Gepetto/constraints-as-terminations)
- **SoloParkour (Chane-Sane et al., 2024):**
  - Extends CaT to off-policy, visual parkour with the constraint `P_{(s,a)∼ρ^π_γ}[c_i(s,a)>0] ≤ ε_i`.
  - Treats knee/base collision as a binary-indicator hard constraint.
  - Soft constraints "allow the RL agent to discover agile locomotion during the early stage of training while enforcing more the constraints later on."
  - Real Solo-12 success: 40 cm climb 85%, 35 cm gap leap 70%, 20 cm crawl 100%.
  - Source: [SoloParkour arXiv 2409.13678](https://arxiv.org/html/2409.13678v1)
- **Failure mode: "feasibility collapse" (SCoCaT, Arora et al., Univ. Luxembourg, 2026):**
  - When safety constraints switch on exactly where the agent must enter to succeed (spacecraft docking), CaT agents "hover" just outside the goal: "the survival-weighted objective suppresses return precisely where the agent must enter to succeed; states just outside the corridor offer near-maximal reward at near-zero termination probability."
  - Their Proposition 1 shows hovering is a strict local optimum. In the dense-reward case the goal value is discounted by κ, which drops from 1.0 to 0.023 as violation severity goes from 0 to 0.3.
  - Results:
    - Floating platform: CaT task completion 0.4% (99.4% compliance), SCoCaT 100% (98.8%).
    - CubeSat: CaT 0% completion, SCoCaT 61.3% (another summary passage gives 79.4% "declared success", so the figures are inconsistent).
    - Hardware docking: CaT 0/4, SCoCaT 3/4.
  - Source: [SCoCaT arXiv 2609.06061](https://arxiv.org/html/2609.06061)
- **SCoCaT fix:**
  - Adds a second critic V_s for a per-step success indicator, bootstrapped only on *environment* terminations rather than CaT's δ: `A_r = GAE(r(1−δ), V_r, δ)`, `A_s = GAE(I_succ, V_s, d^env)`, `A_tot = standardize(A_r + A_s)`.
  - Ablation: adding success as a plain reward bonus instead raised violations sharply (docking velocity 0.1% → 3.7%, line of sight 6.6% → 82.6%).
  - Source: [SCoCaT](https://arxiv.org/html/2609.06061)
- **Stochastic Decision Horizons (Milosevic et al., 2026):** formalizes survival-based constrained RL. Violations geometrically shorten the decision horizon, "reducing current reward credit and future bootstrapping". This is equivalent, across fixed Lagrange multipliers, up to a constant reward offset that "implicitly incentivizes survival". — [arXiv 2602.04599](https://arxiv.org/abs/2602.04599)

### Inferences
- At a hard-constraint state CaT is plain termination (δ = 1 when the violation reaches c^max). For contact, which is a binary indicator, clip(c^+/c^max) is 1 on any contact. **CaT with p_max = 1 on a binary contact constraint is therefore identical to our current hard termination.** The softness that helps exploration only appears if (a) contact is measured as a graded quantity (penetration depth, force, or a proximity margin) or (b) p_max starts below 1 and is annealed upward.
- SCoCaT's "hovering" is the same phenomenon as our "stalling in front of obstacles". The geometry of hurdles and gaps puts the risk of termination exactly on the path to further reward. The theory implies that stalling is a rational local optimum whenever per-step reward is available while standing still. It does not imply a bug.

### Gaps
- CaT does not report an ablation of p_max for *collision* constraints, nor success rates under a hard-vs-soft collision constraint. All its collision constraints were hard.
- I did not verify CaT's exact τ^c value or the p_max schedule shape (linear or otherwise) from code.
- No public CaT results on a humanoid hand-contact constraint were found. The search engine said the method had been tested on a Unitree H1, but I could not confirm the source.

## Q2. Constrained-MDP methods in legged robotics (PPO-Lagrangian, P3O/N-P3O, IPO, CRPO, CPO), Safety Gym findings, and how they compare to termination

### Takeaway
For legged locomotion, the first-order CMDP methods (N-P3O, PPO-Lagrangian, IPO with adaptive thresholds) give good constraint satisfaction with less reward tuning than penalties. In Safety Gym, however, Lagrangian agents kept costs low partly by giving up much of the return (normalized return 0.24 vs PPO's 1.0). That is the "refusal" side of the trade-off. CaT reports beating N-P3O. No study found compares CMDP methods to termination on contact-free obstacle passing specifically.

### Cited Findings
- **Safety Gym (Ray, Achiam, Amodei, 2019), argument against fixed penalties:** "If the designer selects a penalty that is too small, the agent will learn unsafe behavior, and if the penalty is too severe, the agent may fail to learn anything." Also, "There is no invertible map between 'desired safety specification' and 'correct trade-off parameter'". A fixed trade-off also "does not account for a requirement to satisfy safety requirements throughout training." — [Safety Gym paper](https://cdn.openai.com/safexp-short.pdf)
- **Safety Gym results (Table 1, normalized to PPO = 1.0, SG18 slate, cost limit d = 25 per 1000-step episode):**

  | Method | Return | Violation | Cost rate |
  |---|---|---|---|
  | PPO-Lagrangian | 0.24 | 0.026 | 0.245 |
  | TRPO-Lagrangian | 0.331 | 0.018 | 0.265 |
  | CPO | 0.784 | 0.593 | 0.646 |

  - "Lagrangian methods more-or-less reliably enforce constraints"
  - CPO's "approximation errors ... prevent it from fully satisfying constraints"
  - "Constrained RL algorithms attain lower levels of return"
  - Constrained methods "struggle to learn safe locomotion policies" on the Doggo robot.
  - Source: [Safety Gym](https://cdn.openai.com/safexp-short.pdf); also [OpenAI blog](https://openai.com/index/safety-gym/)
- **PID Lagrangian (Stooke et al., ICML 2020):** the standard multiplier update "amounts to integral control on the constraint", which causes oscillation and overshoot, so constraints are violated during learning. The proposed update is `λ ← (K_P·Δ + K_I·I + K_D·∂)_+`. On Safety Gym it "expanded the Pareto frontier ... into a new region of high rewards at relatively low cost." — [arXiv 2007.03964](https://arxiv.org/html/2007.03964v1)
- **Lee et al. 2023, *Evaluation of Constrained RL Algorithms for Legged Locomotion* (ETH RSL, ANYmal wheeled-legged):**
  - Five first-order methods were compared with ε_i = 0 for all costs:
    - P3O: `L^CLIP_R − Σ κ_i max(0, L^VIOL_Ci)`
    - PPO-Lagrangian: `λ'_i = λ_i + α(J_Ci − ε_i)`
    - IPO, a log barrier: `φ(x) = log(−x)/k`
    - CRPO, which switches between reward and cost steps
    - FOCOPS
  - "Three algorithms could achieve high reward and less than a single constraint violation on average: P3O, N-P3O and PPO-Lagrangian. The N-P3O achieved the lowest constraint violation." N-IPO had the highest reward but was sensitive to its threshold.
  - N-P3O was chosen because it had the fewest parameters, needing only a single κ.
  - On hardware the N-P3O policy "slows down before stepping down to reduce impact, while normal PPO policy gains speed". The PPO baseline violated more at the front-wheel collision when stepping up.
  - Their PPO baseline penalized body contact as −(number of non-wheel contacts).
  - Source: [arXiv 2309.15430](https://arxiv.org/pdf/2309.15430)
- **Lee et al., practical point:** with PPO penalties, "the impact of each coefficient is non-intuitive, often demanding numerous trial-and-errors". With separate cost critics "this effort is removed by design". The overhead was 0.07 s per iteration against 0.74 s of simulation. — [arXiv 2309.15430](https://arxiv.org/pdf/2309.15430)
- **Kim et al. 2024, *Not Only Rewards But Also Constraints* (IEEE T-RO):**
  - Formulation: CMDP `π* = argmax J(π) s.t. J_Ck(π) ≤ d_k`, solved with a modified IPO.
  - It uses *probabilistic* constraints (indicator costs, e.g. body contact and joint limits) and *average* constraints. It has 11 constraints in total, with a multi-head cost critic.
  - Adaptive threshold to avoid initial infeasibility: `d_k^i = max(d_k, J_Ck(π_i) + α·d_k)`. The feasible region starts wide and tightens toward d_k.
  - It needs 3 reward terms, against 11 to 14 in prior work.
  - Source: [arXiv 2308.12517](https://arxiv.org/html/2308.12517)
- **ET-MDP (Sun et al., 2021):** defines an early-terminated MDP with the same optimal value as the CMDP, motivated by people stopping "immediately when encountering danger rather than learning to behave safely in danger". Solved with an off-policy context-model algorithm, it showed better asymptotic performance than direct CMDP solvers. — [arXiv 2107.04200](https://arxiv.org/abs/2107.04200)
- **CaT vs N-P3O:** CaT had higher reward (682.9 vs 593.2) and lower torque violation (0.5% vs 8%) on flat-terrain Solo-12. Binary termination (ET-MDP style) failed to learn at all. — [CaT](https://arxiv.org/html/2403.18765)

### Inferences
- Lagrangian methods behave like a penalty whose size adapts. If the cost limit is near zero (no touches) and the task requires going near obstacles, λ grows until the cheapest feasible policy (not approaching) wins. Safety Gym's 76% return loss for PPO-Lagrangian is consistent with that conservatism. Constrained RL alone will not cure stalling unless the task reward makes progress clearly dominant.
- Kim et al.'s adaptive threshold is effectively a violation-budget curriculum (a relaxed budget early that tightens later). It is the CMDP analogue of CaT's p_max ramp.

### Gaps
- Lee et al. 2023 and Kim et al. 2024 contain no direct experiment of termination vs CMDP for obstacle contact, and no attempt-rate metric.
- Achiam et al. 2017 (CPO) was not fetched directly. Its characterization here comes from Safety Gym and Lee et al.

## Q3. Known issues: termination makes the agent conservative, penalties get "bought", penalty magnitude relative to per-step reward, and terminal penalty vs termination

### Takeaway
Termination is an implicit penalty equal to the forfeited future value. With positive per-step rewards that is huge (about r/(1−γ)), so any risky maneuver is strongly discouraged unless passing it unlocks much more value than standing still. With negative rewards, termination instead becomes attractive (the "suicidal agent"). Small fixed penalties get "bought" (Safety Gym), and large ones stop learning. Penetration-type penalties work best when coupled to the motion that causes them.

### Cited Findings
- **Termination with non-positive rewards creates false optima:** "If rewards are not strictly non-negative ... a good strategy may be to terminate episodes as early as possible". The mitigation is a termination penalty or an alive bonus. — Hämäläinen et al. 2019, *Visualizing Movement Control Optimization Landscapes* ([arXiv 1909.07869](https://arxiv.org/pdf/1909.07869), via search snippet; the full PDF was too large to fetch)
- **legged_gym's handling:**
  - `only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)`. The termination reward is added *after* clipping: `rew = self._reward_termination() * scale`, where `_reward_termination = reset_buf * ~time_out_buf`.
  - The default scales are `termination = -0.0` and `collision = -1.` (count of `penalize_contacts_on` bodies with force > 0.1 N). Termination happens on contact force > 1 N on `terminate_after_contacts_on` bodies.
  - Sources: [legged_robot_config.py](https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py), [legged_robot.py](https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py)
- **IsaacLab:** a separate terminal penalty is available as `is_terminated` ("Penalize terminated episodes that don't correspond to episodic timeouts"). `is_terminated_term` targets specific termination terms, and `is_alive` is the alive bonus. — [IsaacLab mdp/rewards.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/rewards.py)
- **Penalty size:** too small means unsafe behavior is learned (the penalty is bought). Too large means "the agent may fail to learn anything". — [Safety Gym](https://cdn.openai.com/safexp-short.pdf)
- **CaT positive-reward requirement:** CaT's (1−δ) scaling only acts as a deterrent if rewards are positive. — [CaT](https://arxiv.org/html/2403.18765)
- **Survival offset:** survival/termination-based constrained RL equals Lagrangian RL up to a constant reward offset that "implicitly incentivizes survival". — [Milosevic et al. 2026](https://arxiv.org/abs/2602.04599)
- **Goal-proximal constraints:** termination-based constraints produce hover/refuse optima when constraints activate near the goal. Adding a task bonus to the same reward stream restores attempts but buys violations (line-of-sight violation 6.6% → 82.6%). A separately bootstrapped success critic restores attempts at a small compliance cost. — [SCoCaT](https://arxiv.org/html/2609.06061)
- **Coupling the penalty to motion:** Robot Parkour Learning multiplies penetration terms by forward velocity, "to prevent the robot from exploiting the penetration reward by sprinting through the obstacles". — [arXiv 2309.05665](https://arxiv.org/html/2309.05665)

### Inferences (applied to our numbers)
- **Hard termination:** with about +15 per step of positive reward, an assumed γ ≈ 0.99, and long episodes, the value lost by terminating is roughly 15/(1−γ) ≈ 1500 (horizon-limited). An attempt that succeeds with probability p must gain more than (1−p)·V_stall in extra value. Early in training p is near 0, so stalling dominates (SCoCaT Proposition 1).
- **Non-terminal penalty of 3 per step against +15 per step:** each contact step is still net +12. The penalty is fully "affordable", yet the agent still stalled.
  - This strongly suggests the stall is not caused by the contact penalty. More likely, stalling collects most of the +15 per step (alive, posture or tracking-style terms that do not require crossing the obstacle) while crossing adds little extra or extra risk.
  - The lever is then the reward structure: make per-step reward conditional on progress, use a time-bounded goal reward, or penalize standing still. Tuning the size of the collision cost will not fix it.
  - Success fell further plausibly because, without termination, envs stuck at the obstacle occupy the whole episode and never reset to new attempts.
- **Terminal penalty vs termination:** with positive rewards, an extra terminal penalty only makes refusal stronger. With partly negative rewards, the terminal penalty is needed to prevent "suicide".

### Gaps
- No paper found that measures "attempt rate" vs "violation rate" for collision-terminating vs penalized locomotion directly, apart from SCoCaT (docking) and the Robot Parkour Learning ablation.
- The verbatim Hämäläinen 2019 text was not retrieved; the quote comes from the search snippet.

## Q4. Soft terminations, termination curricula (relaxed to strict), and violation budgets for contact

### Takeaway
The strongest evidence for getting the agent to *attempt* hard obstacles comes from relaxing physics or constraints first and tightening them later:
- Robot Parkour Learning's soft-dynamics pretraining (penetrable obstacles plus a penetration penalty) took climb and leap success from 0% to 95% and 82%.
- CaT/SoloParkour ramp soft-constraint p_max from 0.05 to 0.25.
- Kim et al. use an adaptive constraint threshold.

### Cited Findings
- **Robot Parkour Learning (Zhuang et al., CoRL 2023), soft dynamics constraints:**
  - In pre-training "obstacles are penetrable so the robot can violate the physical dynamics in the simulation by directly go through the obstacles".
  - The penalty is `r_penetrate = −Σ_p (α_5·1[p] + α_6·d(p))·v_x`, using collision points on the body.
  - Fine-tuning then uses hard (realistic) dynamics with only the skill reward. Episodes do not terminate on penetration.
  - Ablation, simulation success with vs without soft dynamics: climb 95% vs 0%, leap 82% vs 0%, crawl 100% vs 93%, tilt 100% vs 86%. "Both RND and Oracles w/o Soft Dyn cannot make any learning progress on climbing and leaping, the two most difficult parkour skills."
  - Source: [arXiv 2309.05665](https://arxiv.org/html/2309.05665)
- **Humanoid Parkour Learning (Zhuang et al., 2024):**
  - Uses "virtual obstacles" around terrain edges with `r_penetrate = α·Σ_p d(p)·||v(p)||`, α = −5×10⁻³, to stop edge exploitation.
  - Auto-curriculum: a robot is promoted after it "safely finishes the task and moves at least 3/4 distance" and demoted if it fails under 1/2.
  - There is no collision-handling ablation.
  - Source: [arXiv 2406.10759](https://arxiv.org/html/2406.10759v2)
- **CaT soft constraints:** p_max rises from 0.05 to 0.25 over training. Hard constraints, including collisions, are fixed at p_max = 1. — [CaT](https://arxiv.org/html/2403.18765); SoloParkour gives the same rationale ("discover agile locomotion during the early stage") — [SoloParkour](https://arxiv.org/html/2409.13678v1)
- **Kim et al. adaptive budget:** `d_k^i = max(d_k, J_Ck(π_i) + α·d_k)`, which relaxes the budget for probabilistic contact constraints early and tightens it to d_k. — [arXiv 2308.12517](https://arxiv.org/html/2308.12517)
- **SCoCaT:** keeps the constraints strict but decouples the "attempt" incentive through an environment-terminated success critic. — [SCoCaT](https://arxiv.org/html/2609.06061)

### Inferences (design options for G1 hurdles and gaps, ranked by evidence)
1. **Soft-dynamics or penetration pretraining:** the strongest ablation evidence (0% → 95%).
   - Make obstacles non-colliding (or contact-disabled) early and penalize penetration depth × velocity.
   - Then switch to real collisions and add termination for hand contact.
   - In MuJoCo Warp this could be contype/conaffinity masking plus geometric penetration queries.
2. **Graded contact constraint plus a CaT-style p_max curriculum:**
   - Use penetration depth, force or a clearance margin rather than a binary indicator, so δ is fractional.
   - Ramp body-contact p_max from about 0.05 to 1.0, and keep hand contact closer to hard (or ramp it faster).
   - CaT itself never relaxed collision constraints, so this is an extrapolation.
3. **Decouple the attempt incentive (SCoCaT):** add an obstacle-passed or progress critic bootstrapped on environment terminations only, so the possibility of a contact termination does not wipe out the value of crossing.
4. **Fix the stall reward:** make positive per-step reward conditional on progress, or add a goal-reach bonus with a time limit. This is needed whichever collision scheme is used, because both of our experiments stalled.
5. **CMDP (N-P3O or PPO-Lagrangian with PID, adaptive threshold) for the hand-contact constraint:** better-principled, but Safety Gym shows large return sacrifice under tight limits.

### Gaps
- No found paper uses a "termination curriculum" for *contact* specifically (from a fractional to a deterministic termination probability) with reported attempt and success rates. CaT ramps only soft, non-collision constraints.
- No found humanoid paper requires that hands never touch obstacles and reports how it was enforced.
- No found direct comparison of soft dynamics vs CaT vs Lagrangian on the same obstacle task.
