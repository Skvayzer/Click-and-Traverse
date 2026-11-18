# RL incentives that make policies rush, hesitate or pass on momentum near obstacles, and the fixes for slow-command compliance

Scope note: this complements `reports/RL refusal versus attempt incentives.md` (refusal, stall income, timeout bootstrapping, soft contact, reverse curricula). It covers speed instead: haste from discounting and terminal bonuses, time-based costs, exposure minimisation under contact termination, momentum and lunge exploits, hesitation patterns, and slow-command compliance near obstacles. Full texts of the arXiv papers cited below were read from the shared folder `research_notes/Slow passage and field repulsion/papers/<id>/`. Discovery was limited: the WebSearch budget for this session was already used up, so sources were found from known arXiv IDs, one arXiv lookup query, and direct fetches (GitHub code, the Europe PMC API). "Inference" marks my own reasoning or arithmetic. Everything else is stated in the cited source.

User setup used in the arithmetic: PPO, γ = 0.98 per 20 ms step (effective horizon 1/(1−γ) = 50 steps = 1 s), goal bonus 2000 at a success line just past each obstacle, per-second clearance and distance costs, cosine field alignment over about 13 body points, termination on any contact, and a per-step reward clip at 0.

---

## Q1. Does discounting plus a terminal goal reward create a documented time-to-goal bias, and how do PBRS and capped progress rewards keep the reward speed-neutral?

### Takeaway
Yes. Several sources state the mechanism outright:
- CADRL defines the value as γ^(t_g·v_pref), so value is literally time-to-goal.
- ETH found that a continuous position cost makes the bang-bang, "as fast as possible" solution optimal and "needlessly aggressive".
- Pardo et al. show that agents facing a deadline take riskier actions as it approaches.
- Motor-neuroscience work shows that steeper reward discounting produces faster movements.

The published speed-neutral designs avoid paying for early arrival altogether:
- **Final-time reward.** The task reward is paid only in the last T_r seconds of a fixed, time-aware horizon, and the time budget then acts as the speed command. The ETH papers show a policy given more time "will adopt a slow and safe gait".
- **Dense progress capped at the commanded speed.** Extreme Parkour uses min(⟨v, d̂⟩, v_cmd).

PBRS preserves whatever haste the base reward already contains; it cannot remove it.

### Cited Findings

**Direct statements that discounted or continuous goal objectives produce haste**
- CADRL (Chen, Liu, Everett, How; ICRA 2017) sets the objective as minimising expected time to goal, argmin E[t_g]. The value is V*(s) = Σ_t γ^(t·v_pref) R, and value-network targets are y = γ^(t_g·v_pref). Rewards are +1 at the goal, −0.25 for collision, and −0.1 − d_min/2 when d_min < 0.2 m. The paper says v_pref "is used here as a normalization factor for numerical reasons, because otherwise the value function of a slow moving agent could be very small." The action space is bounded by the preferred speed (‖v‖ < v_pref) — [Chen et al. 2017, arXiv 1609.07845](https://arxiv.org/abs/1609.07845)
- CADRL reports that the time-to-goal objective produced aggressive behaviour: "Without this modification, an agent would frequently travel straight toward its goal, expecting the other agent to yield". The fix subtracts 0.1 from the training value when the agent's extra time t_e = t_g − d_g/v_pref is small while the other agent's is large — [Chen et al. 2017](https://arxiv.org/abs/1609.07845)
- CrowdNav/SARL (Chen, Liu, Kreiss, Alahi; ICRA 2019) reuses the γ^(Δt·v_pref) discount "for numerical reasons". It also uses a per-step discomfort term (−0.1 + d_t/2 when d_t < 0.2 m, as printed), where d_t is the minimum separation over the last step — [Chen et al. 2019, arXiv 1809.08835](https://arxiv.org/abs/1809.08835)
- Rudin, Hoeller, Bjelonic, Hutter (IROS 2022), on penalising the distance to the target at every step: "the optimal solution would be to reach the goal as fast as possible, resulting in a behavior close to the 'bang-bang' solution of optimal control problems … needlessly aggressive and dangerous motions." Their continuous-position-tracking baseline (task reward every step, T_r = T) "completely fails to navigate obstacles with a sufficient reliability". It was not deployed because it "is prone to aggressive motions which could potentially damage the robot" — [Rudin et al. 2022, arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- Rudin 2022 also explains the reverse timing artifact under discounting. The policy gets a positive task reward at the end and penalties throughout that "are larger when the robot is walking", so "it is beneficial to push the negative penalties as far into the future as possible. Therefore, the policy learns to wait until the last moment when it suddenly runs fast towards the target." Their fix is r_stall = −1 when speed < 0.1 m/s and distance > 0.5 m. They add that "the real-world energy consumption should be optimized without a discount factor" — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- **Momentum and lunge exploit at a reward instant.** "T_r needs to be long enough to force the policy to stop at the target in a stable configuration. Otherwise, the policy learns to jump or lean towards the target at the last moment, which would result in a crash on the real robot." They used T = 6 s, with the task reward over the final T_r = 1 s — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- Pardo, Tavakoli, Levdik, Kormushev (ICML 2018) note that for time-limited tasks γ = 1 can be used because the return is bounded, "however, in practice it is still common to keep γ smaller than 1 in order to give more priority to short-term rewards". Such an agent "could for example learn to take more risky actions leading to higher expected returns as approaching the time limit." In their Queue of Cars task, a "safe" action advances 50% of the time, while a "dangerous" overtake advances 80% of the time with a 10% chance of a collision that terminates the episode. A time-aware PPO agent changes its dangerous-action probability with the remaining time, while standard PPO learns seed-dependent suboptimal mixtures. In Hopper the time-aware agent "learns to efficiently jump forward and fall towards the end of its time … performing a 'photo finish'", which is a momentum exploit at the horizon — [Pardo et al. 2018, arXiv 1712.00378](https://arxiv.org/abs/1712.00378)
- Pardo's Two-Goal Gridworld: the time-unaware agent "always tries to go for the closest goal even if the remaining time is not sufficient", while the time-aware agent "learns to stay in place when there is not enough time" — [Pardo et al. 2018](https://arxiv.org/abs/1712.00378)
- Zhang, Rudin, Hoeller, Hutter ("Learning Agile Locomotion on Risky Terrains"; ANYmal, navigation formulation): "To solve the task in the given episode length … first slowly aligning itself towards the center and then making a speed burst to quickly traverse the terrain on balance beams." The figure shows the burst at 3–5 s, and peak speeds were ≥ 2.5 m/s (2.7 m/s on stones). The fix was a longer time budget: "We slightly extend the episode length to slow down the aggressive motions on 'Stones-2Rows' and 'Balance-Beams'" — [Zhang et al. 2023, arXiv 2311.10484](https://arxiv.org/abs/2311.10484)
- **Agile But Safe (ABS) is an intentional haste design.** Its goal-tracking terms have the form (1/(1+‖err/σ‖²))·1(t > T−T_r)/T_r, "by doing so, the robot only needs to reach the goal before T−T_r to maximize the tracking rewards". Speed is then added explicitly:
  - r_agile = max{ReLU(v_x/v_max)·1(correct direction), 1(d_goal < σ_tight)}, with v_max = 4.5 m/s. "To maximize this term, the robot has to either run fast or stay at the goal."
  - −20·r_stall "penalizes the robot for time waste".
  - Episode length is U(7, 9) s.

  The unshielded agile policy averaged 2.04–2.55 m/s on successful runs, with 15.5–26.1% collisions in simulation — [He et al. 2024 (RSS), arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- Zhang & Ross (ICML 2021): "the discounted objective biases the optimal policy to choose actions that lead to high near-term performance rather than to high long-term performance" — [Zhang & Ross 2021, arXiv 2106.07329](https://arxiv.org/abs/2106.07329)

**Cross-disciplinary theory: discounting and reward rate set movement speed**
- Shadmehr, Orban de Xivry, Xu-Wilson, Shih (J Neurosci 2010; 30(31):10507–16): "each voluntary movement carries a cost because its duration delays acquisition of reward", and "We show that delay of reward elevates the cost of saccades, reducing velocities." "The hyperbolic cost of this delay not only accounts for kinematics of saccades in adults, it also accounts for the faster saccades of children, who temporally discount reward more steeply." Steeper discounting therefore means faster movement — [Shadmehr et al. 2010, PMID 20685993](https://pubmed.ncbi.nlm.nih.gov/20685993/) (abstract text via [Europe PMC API](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:20685993&resultType=core&format=json))
- Niv, Daw, Joel, Dayan (Psychopharmacology 2007; 191(3):507–20), an average-reward RL account: "Optimal control balances the costs of acting quickly against the benefits of getting reward earlier and thereby chooses a best response latency … the long-run average rate of reward plays a key role as an opportunity cost and mediates motivational influences on rates and vigor of responding." — [Niv et al. 2007, PMID 17031711](https://pubmed.ncbi.nlm.nih.gov/17031711/) (abstract via [Europe PMC API](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:17031711&resultType=core&format=json))

**Speed-neutral or speed-capped reward designs**
- Potential-based shaping F(s,a,s′) = γΦ(s′) − Φ(s) is the form that guarantees optimal-policy invariance. It therefore inherits, and cannot remove, any haste already present in the base reward — [Ng, Harada, Russell 1999 (ICML)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Extreme Parkour (Cheng, Shi, Agarwal, Pathak; ICRA 2024) uses r_tracking = min(⟨v, d̂_w⟩, v_cmd), where v is the world-frame velocity, d̂_w is the unit direction to the next waypoint and v_cmd is "the desired speed". The commanded speed is a policy input. The world frame is used "to prevent the robot from exploiting the reward and learning the unintended behavior of turning around the obstacle". The curriculum demotes a robot that travels "less than half the expected distance v_cmd·T". The base-frame velocity-tracking ablation (NoInner) walked around hurdles, and on steps "all it learns is a colliding and retrying behavior" — [Cheng et al. 2023, arXiv 2309.14341](https://arxiv.org/abs/2309.14341)
- Robot Parkour Learning (Zhuang et al.; CoRL 2023) uses the forward term −α1·|v_x − v_target| with "the target speed for all skills [set] to around 1 m/s", so overspeed is penalised symmetrically, plus an alive bonus of +2 — [Zhuang et al. 2023, arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- **Time budget as the speed command.** Fixing a target 3 m ahead and a constant "time left" input gives pseudo-velocity control: "When the policy believes it has a lot of time to reach the target, it will adopt a slow and safe gait. As that time is reduced, it will try to close the gap faster and faster." On hardware, joystick magnitude "is used to scale the remaining time given to the policy". On energy, the velocity-tracking policy "performs worse at slow speed", and the continuous-position policy "is trained to run as fast as possible" — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- The final-time formulation outperformed velocity tracking on obstacles. Maximum difficulty at 95% success was:

  | Obstacle | Velocity tracking | Continuous position | Final position |
  |---|---|---|---|
  | Gap width | 0.15 m | 0.65 m | 1.2 m |
  | Obstacle length | 0.35 m | — | 0.85 m |
  | Pit depth | 0.1 m | 0.65 m | 0.95 m |

  — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)

**Rewards in the two closest humanoid systems**
- **Click-and-Traverse (the paper this project builds on).**
  - The field reward is Σ_k [log C_d(κ_k) + κ_k μ_kᵀ v̂_k] with v̂_k = v_k/‖v_k‖, so it scores direction only.
  - A collision-urgency weight scales with approach velocity: max(−∇d·v_k, 0.5)·exp(−d).
  - A trial succeeds if the robot "reaches within 0.1 meters of the target location in 5 seconds without colliding".
  - There is no speed command.

  — [Xue et al. 2026, arXiv 2601.16035](https://arxiv.org/abs/2601.16035) (LaTeX source `sec/3_method.tex`, `sec/4_exp.tex`)
- **Gallant (G1 humanoid local navigation).** Goal-reaching uses a fixed 10 s horizon with T_left in the observation and rewards follow Rudin 2022. The directional reward is printed as r = (a·v)/‖a·v‖₂, where a combines APF-like repulsion and tangential terms. As printed, that is a sign-like, speed-independent ±1. The robot is terminated if it "fails to move at least 1 m away from its initial position within 4 seconds", or if any external force on the torso, hip or knee exceeds 100 N — [Ben et al. 2025, arXiv 2511.14625](https://arxiv.org/abs/2511.14625)

### Inferences
- **The user's bonus arithmetic checks out (my computation).** B = 2000, γ = 0.98, dt = 0.02 s gives V(arrive at 0.5 s) = 1207 and V(3.3 s) = 71, a difference of 1136. The marginal value of arriving one second earlier is about:

  | γ | Effective horizon | 1 s earlier, arriving at 1 s | at 2 s | at 3.3 s |
  |---|---|---|---|---|
  | 0.98 | 1 s | 1272 | 463 | 125 |
  | 0.99 | 2 s | ~790 | ~480 | ~250 |
  | 0.995 | 4 s | ~440 | ~345 | ~250 |
  | 0.998 | 10 s | ~190 | ~170 | ~150 |
  | 0.999 | 20 s | ~98 | ~93 | ~87 |

  Raising γ shrinks the per-second haste incentive about 13-fold at 1 s (0.98 → 0.999) but never removes it. Only an undiscounted (γ = 1), time-aware, non-repeating episode makes a terminal bonus exactly time-neutral, which Pardo notes is admissible for time-limited tasks.
- **Average reward is not a cure by itself (inference from Niv et al. 2007).** If episodes reset and goals repeat, the average-reward objective maximises goals per second, and the average reward rate ρ acts as an opportunity cost of ρ per second of delay. That is also a haste incentive, linear instead of exponential.
- **CADRL's γ^(t·v_pref) implies a speed-dependent per-second discount** of γ_base^(v_pref). For a slow-commanded agent this is closer to 1, so it reduces impatience per second for slow commands. It still rewards arriving earlier than the commanded speed implies, because v_pref is fixed while t shrinks. CADRL also bounds speed by v_pref in the action space, which a whole-body joint-space policy cannot do directly. CADRL's stated motive was numerical only, so applying this to haste is my extrapolation.
- **Exact PBRS versus undiscounted progress (my algebra).** With Φ = −d_geo:
  - The exact PBRS term is F = d_t − γd_{t+1}. A stationary robot is paid (1−γ)·d per step, which is 0.04 per step (2 per second) at d = 2 m and γ = 0.98. Exact PBRS therefore leans mildly toward lingering far away, but cancels in the optimum.
  - The common undiscounted progress d_{t−1} − d_t equals PBRS minus (1−γ)·d_t, which is a hidden per-step time cost proportional to remaining distance. That cost mildly favours haste.
  - Neither shaping term removes the haste created by the discounted 2000 bonus.
- **Cosine alignment pays more for fast motion when gait sway exists (illustrative arithmetic, not a measured sway value).** With forward speed v and orthogonal sway speed s, cos = v/√(v²+s²). For an assumed s = 0.1 m/s, the cosine is 0.71 at 0.1 m/s, 0.98 at 0.5 m/s and 0.99 at 0.9 m/s. Unit-vector rewards (C&T's v̂, Gallant's sign-like term, Rudin's r_bias, Xu et al.'s exploration term) therefore carry a speed incentive. A projection reward min(⟨v, d̂⟩, v_cmd)/v_cmd (Extreme Parkour form) is unaffected by orthogonal sway and gives no gain above v_cmd.
- **The success line just past each obstacle is the "short T_r" situation Rudin describes.** It pays crossing an instant line, not stable arrival, so lunging and momentum passage are rewarded. The literature fix is a reward window long enough to force a stable stop or hold (Rudin's T_r; ABS's r_postight and r_stand at the goal).

### Gaps
- I found no robotics paper that measures traversal speed against γ, or sweeps γ to show haste. The evidence is mechanistic statements (Rudin, CADRL, Pardo) plus the motor-neuroscience result.
- No paper evaluates CADRL-style speed-normalised discounting as a haste fix. Average-reward PPO results (ATRPO) are on tasks that reward speed itself, not on goal reaching.

---

## Q2. Termination on contact and per-time costs: rushing to minimise exposure versus refusing to approach. What do works say about termination as an implicit reward, and which fixes work?

### Takeaway
- **Per-time penalties near obstacles cause sprinting.** Robot Parkour Learning documents it directly: robots exploited a per-step penetration penalty "by sprinting through the obstacles". The fix was to multiply the penalty by forward velocity, turning a per-second cost into a per-metre cost.
- **Termination acts like an extra discount.** It is formally a transition discount of 0 (White 2017), and CaT's soft version multiplies future rewards by Π γ(1−δ). Contact risk therefore makes the agent more impatient exactly where risk is present.
- **What reduced collisions without slowing open-space motion** was a learned reach-avoid shield with a twist-tracking recovery policy (ABS). Lagrangian penalties and reward reweighting "only trade off agility and safety", and Lagrangian methods oscillate (PID damping helps).

### Cited Findings
- **Sprinting exploit and its fix.** Robot Parkour Learning uses r_penetrate = −Σ_p (α5·1[p] + α6·d(p))·v_x. "We multiply both the penetration volume and the penetration depth with the forward base velocity v_x to prevent the robot from exploiting the penetration reward by sprinting through the obstacles to avoid high cumulative penalties over time." A per-robot curriculum keyed on the penetration reward hardens obstacles — [Zhuang et al. 2023, arXiv 2309.05665](https://arxiv.org/abs/2309.05665)
- **Continuous distance-to-goal costs give bang-bang behaviour.** See Q1 — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- **Positive per-step clearance reward plus a collision penalty.** Miki et al. use a high-level velocity-following policy rewarded for base distance, r_base = exp(−α(d_max − min(d, d_max))), with "a large negative reward when the body hits obstacles" — [Miki et al. 2024 (ICRA), arXiv 2403.00187](https://arxiv.org/abs/2403.00187)
- **Termination is a discount.** White (ICML 2017) defines the return with a transition-based discount, G_t = Σ_i (Π_j γ(s_{t+j}, a_{t+j}, s_{t+1+j})) R_{t+1+i}. At the end of the episode "γ(s,a,s′) = 0, making the product of these discounts zero" — [White 2017, arXiv 1609.01995](https://arxiv.org/abs/1609.01995) (LaTeX source `paper.tex`)
- **CaT (Constraints as Terminations).**
  - It maximises E[Σ_t (Π_{t′≤t} γ(1−δ(s_{t′}, a_{t′}))) r_t]: "we down-scale all the future rewards based on the magnitude of the constraint violations … through the discount factor". In code, rewards are multiplied by (1−δ) and δ is passed as dones.
  - Hard constraints use p_max = 1. Soft constraints ramp p_max from 0.05 to 0.25.
  - ET-MDP, which terminates on any violation, "entirely fails to learn".
  - CaT can also pose velocity tracking itself as a constraint, ‖v_des − v‖ − ε_track with ε_track = 0.2 m/s.
  - On the real parkour course, Tracking Constraints averaged 90.6% success with 1.7% violations, against 79.4% and 0.5% for Tracking Rewards. On sideways stairs it reached 85.0% against 42.5%.

  — [Chane-Sane et al. 2024, arXiv 2403.18765](https://arxiv.org/abs/2403.18765)
- **Deadline against termination risk.** Pardo's Queue of Cars: a time-aware agent increases risky actions, those carrying a 10% termination chance, as the deadline approaches — [Pardo et al. 2018](https://arxiv.org/abs/1712.00378)
- **ABS: shield plus recovery as the speed-safety fix.**
  - Agile-policy collisions are penalised at −100·1(undesired collision).
  - A policy-conditioned reach-avoid (RA) value with γ_RA = 0.999999 predicts failure.
  - When V̂ ≥ threshold, a twist command is searched (a Lagrangian on the constraint converges "within 5 steps") and a recovery policy tracks it.
  - "ABS runs fast in the open spaces, and slows down in the tight spaces for safety thanks to the shielding of RA values and the recovery policy."

  Simulation results across the paper's three test settings (speeds are means over successful runs only):

  | Method | Success | Collisions | Mean speed |
  |---|---|---|---|
  | ABS | 78.9–85.8% | 2.9–5.7% | 1.87–2.15 m/s |
  | Agile policy alone | 73.3–83.2% | 15.5–26.1% | 2.04–2.55 m/s |
  | PPO-Lagrangian | 49.1–82.5% | 7.4–10.9% | 1.12–1.69 m/s |

  The authors state that "using the Lagrangian-based RL or adjusting reward weights only trades off agility and safety", and that with Lagrangian methods "the constraint is often enforced before convergence, hindering exploration and lowering returns" — [He et al. 2024, arXiv 2401.17583](https://arxiv.org/abs/2401.17583)
- **Lagrangian oscillation.** "Their learning dynamics exhibit oscillations and overshoot which … leads to constraint-violating behavior during agent training". The classic multiplier oscillates with a 90° phase shift, and PID control on λ "damps oscillations" — [Stooke, Achiam, Abbeel 2020 (ICML), arXiv 2007.03964](https://arxiv.org/abs/2007.03964). Safety-Gymnasium likewise finds "Lagrangian-based methods tend to display more oscillation" than projection-based methods, with CPO and PPO-Lag oscillating around the cost limit — [Ji et al. 2023 (NeurIPS D&B), arXiv 2310.12567](https://arxiv.org/abs/2310.12567)
- **Contact used as support when it is cheap.** In Xu et al.'s confined-3D-space study the collision penalty is per count (c3·N_c), not terminal. The better hierarchical policy "is more willing to touch obstacles and still achieves a higher success rate than its more conservative counterpart", with average collision counts of 3.8/32.3/43.6 against 7.9/16.2/29.8 for end-to-end — [Xu, Raj, Xiao, Stone 2024, arXiv 2403.03848](https://arxiv.org/abs/2403.03848)
- **Per-step reward clip at zero, as in legged_gym.** The code is `if self.cfg.rewards.only_positive_rewards: self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)` — [legged_gym legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- **Constant reward offsets change learning in discounted RL.** Value estimates carry a state-independent r(π)/(1−γ) term. Mean-centering removes it, "if a problem's rewards are shifted by a constant, then standard methods perform much worse, whereas methods with reward centering are unaffected", and the benefit "increases further as the discount factor approaches one" — [Naik, Wan, Tomar, Sutton 2024 (RLC), arXiv 2405.09999](https://arxiv.org/abs/2405.09999)

**Hesitation and oscillation patterns reported in front of obstacles**
- ABS: "without the illusion, the robot will sometimes tremble near a wall which it has never seen in simulation". The "illusion" randomisation overwrites far ray distances with random values in U(d_goal + 0.3, ray distance) — [He et al. 2024](https://arxiv.org/abs/2401.17583)
- Extreme Parkour (NoInner): a "colliding and retrying behavior" on steps — [Cheng et al. 2023](https://arxiv.org/abs/2309.14341)
- Slow alignment followed by a speed burst on beams — [Zhang et al. 2023](https://arxiv.org/abs/2311.10484)
- Waiting followed by a sprint, and leaning or jumping at the last moment — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- Hopper's "photo finish" — [Pardo et al. 2018](https://arxiv.org/abs/1712.00378)

### Inferences
- **The hazard argument (inference built on White 2017 and CaT).** A per-step contact-termination probability p while inside a gap multiplies future value by (1−p) per step, which is equivalent to an effective discount of γ(1−p) inside the gap. If p does not rise with speed, survival (1−p)^n is maximised by minimising n, the number of steps spent in the gap, so the policy rushes. If p rises with speed, an interior optimum exists. The observed 0.66–0.97 m/s bursts are consistent with the policy judging that fast passage lowers total contact risk, perhaps through momentum-stabilised sideways stepping. This is a hypothesis to test, not a finding.
- **Rushing and refusing share one logic.** Both minimise exposure to the termination hazard: rush when passage is feasible, never enter when it is not. They are two exits of the same incentive. The prior report covers the refusal exit.
- **Per-second clearance costs scale as c·L/v over a gap of length L.** That makes 0.1 m/s nine times more expensive than 0.9 m/s. Robot Parkour Learning's velocity scaling makes the cost c·L, independent of speed. Caveat (inference): per-metre costs make standing still in the gap free, so they need a stall or time-budget term. RPL multiplies by v_x, so a sign convention for backward motion is needed.
- **Interaction with the clip at zero (inference).** Near walls, clearance costs can exceed the positive per-step income, and the clip then flattens the penalty to zero. Distinctions between "slightly close" and "very close" vanish, and the only remaining wall signal is termination plus the discounted bonus. Both favour haste.

### Gaps
- No source measures traversal speed under hard contact termination against soft or graded contact on the same task.
- No source reports a humanoid narrow-gap speed distribution.
- The SCoCaT "hovering" analysis cited in the prior report remains unverified.

---

## Q3. Which locomotion works track an arbitrary, possibly very slow, commanded speed while avoiding obstacles or traversing confined terrain, and how did they verify low-speed compliance?

### Takeaway
Very few obstacle-aware legged policies take an operator speed, and I found none that verifies compliance at ≤ 0.2 m/s near obstacles. The designs that do expose speed fall into three groups:
- **Hierarchical.** A navigation policy outputs velocity commands under a user-set speed limit with an overspeed penalty (Jin et al. 2024), or a velocity-target-following high level drives a command-tracking low level (Miki et al. 2024; ABS recovery).
- **Time-budget conditioned.** Rudin 2022; Zhang 2023.
- **Point-agent navigation.** v_pref bounds the action (CADRL).

Two common defaults conflict directly with a 0.1 m/s command:
- legged_gym zeroes commands with norm ≤ 0.2 m/s.
- Goal-reaching recipes add anti-stall terms at < 0.1 m/s, or terminate for less than 1 m in 4 s.

### Cited Findings
- **legged_gym default command sampling** sets small commands to zero: `self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)`. Tracking reward is exp(−‖v_cmd − v‖²/tracking_sigma) — [legged_gym legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- **Speed-limit-conditioned navigation (quadruped, hierarchical).** Jin, Zhang, Frey, Rudin, Mattamala, Cadena, Hutter (ICRA 2024). The policy observes "the velocity limit v̄ allowing users to condition the robot velocity" along with time left T−t. Regularisation penalises "the overspeeding above the limit" and "the velocity commands being too small when the target is distant". A per-timestep penalty applies "unless it is close to the target". The navigation policy outputs velocity commands to a pre-trained locomotion policy — [Jin et al. 2024, arXiv 2310.03581](https://arxiv.org/abs/2310.03581). I could not find, in the text, the sampled range of v̄ or any measured compliance with it.
- **Confined spaces, hierarchical.** Miki, Lee, Wellhausen, Hutter (ICRA 2024). The high-level policy's "task is to follow a desired velocity target by commanding the low-level policy" (6D: v_x, v_y, yaw rate, roll, pitch, body height), with velocity tracking, clearance (r_base) and collision rewards. Evaluation counted success as reaching a goal 6 m away "within 30 seconds". Low-level command tracking was shown as plots; no slow-speed metric was reported — [Miki et al. 2024, arXiv 2403.00187](https://arxiv.org/abs/2403.00187)
- **Velocity tracking against goal reaching in confined 3D spaces.** Xu et al. compared three approaches:
  - End-to-end goal reaching: +1 at the goal and terminate, a stall penalty of −1 per step "when the robot's linear velocity is smaller then 0.1 m/s", and a cosine exploration reward.
  - Hierarchical goal reaching toward planner-computed local goals.
  - "Parameterized motor skills": velocity and pose tracking trained on "the real command distribution of the task", not random commands.

  | Approach | Easy | Medium | Hard | Steps to converge |
  |---|---|---|---|---|
  | End-to-end goal reaching | 29.4% | 7.0% | 5.1% | 600 M |
  | Hierarchical goal reaching | 96.8% | 51.5% | 39.4% | 40 M |
  | Velocity-tracking skills | 65.7% | 4.0% | 2.4% | 80 M |

  — [Xu et al. 2024, arXiv 2403.03848](https://arxiv.org/abs/2403.03848)
- **ABS.** Agile-policy speed is not commanded. The recovery policy tracks twist commands with r_linvel = exp(−‖v − v_c‖²/σ²), σ = 0.5 m/s, so that it can "track a given twist command as fast as possible". It is trained on 2 s episodes starting from v_x ~ U(−0.5, 5.5) m/s and promoted when tracking error < 0.7σ. ABS notes that prior RL navigation planners bounded velocity commands to 1 m/s for safety, and that existing controllers were "conservative (< 1.0 m/s)" — [He et al. 2024](https://arxiv.org/abs/2401.17583)
- **Commanded speed as input with capped reward** (min(⟨v, d̂⟩, v_cmd)), with a curriculum tied to v_cmd·T — [Cheng et al. 2023](https://arxiv.org/abs/2309.14341)
- **Time budget as speed control**, with the qualitative statement "slow and safe gait" when given more time — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827). A longer episode length was used to slow aggressive motions — [Zhang et al. 2023](https://arxiv.org/abs/2311.10484)
- **Walk These Ways (CoRL 2022).** Behaviour parameters (gait timing, frequency, footswing height, body height and others) condition a velocity-tracking policy. Velocity commands follow the "grid adaptive curriculum strategy" of Margolis et al. Training was "only on flat ground without any randomization of terrain geometry", and obstacles were handled by a human pilot tuning behaviour. "Enforcing low frequency (2Hz) makes high speeds less attainable. The gait-free policy offers slightly better performance at the lowest and highest speeds." — [Margolis & Agrawal 2022, arXiv 2212.03238](https://arxiv.org/abs/2212.03238)
- **Motion-prior compliance on flat ground (AMP).**

  | Commanded speed | 0.4 m/s | 0.8 m/s | 1.2 m/s | 1.6 m/s |
  |---|---|---|---|---|
  | AMP, measured | 0.36 | 0.77 | 1.11 | 1.52 |
  | Complex style reward, measured | 0.41 | 0.88 | 1.28 | 1.67 |
  | No style reward, measured | 0.42 | 0.82 | 1.22 | 1.61 |
  | AMP cost of transport | 1.07 | 0.93 | 1.02 | 1.12 |
  | Complex style cost of transport | 1.54 | 1.37 | 1.40 | 1.41 |

  The no-style policy "learns to move by vibrating its legs at high speeds with large torques". Commands were sampled from −1 to 2 m/s — [Escontrela et al. 2022 (IROS), arXiv 2203.15103](https://arxiv.org/abs/2203.15103) (LaTeX source)
- **Point-agent speed conditioning.** v_pref is in the state and the action is bounded by ‖v‖ < v_pref — [Chen et al. 2017](https://arxiv.org/abs/1609.07845); [Chen et al. 2019](https://arxiv.org/abs/1809.08835)
- **Explicit velocity constraint benchmark.** Safety-Gymnasium velocity tasks use cost(s,a) = [v(s,a) > v_limit] for Ant, HalfCheetah, Humanoid and others. "All SafeRL algorithms achieve average episodic costs within the cost_limit for velocity tasks" — [Ji et al. 2023](https://arxiv.org/abs/2310.12567)
- **The humanoid baselines nearest this project carry no speed command and set a minimum speed implicitly.**
  - Click-and-Traverse requires success within 5 s — [Xue et al. 2026](https://arxiv.org/abs/2601.16035)
  - Gallant on the G1 uses a 10 s horizon and terminates for less than 1 m in 4 s, an implied ≥ 0.25 m/s — [Ben et al. 2025](https://arxiv.org/abs/2511.14625)

### Inferences
- **Command sampling.** If the project's command sampler follows legged_gym (unitree_rl_gym for the G1 derives from it; not verified here), commands of 0 < ‖v‖ ≤ 0.2 m/s are never trained. A 0.1 m/s operator command would then be out of distribution, or behave like "stand". This is worth checking in the mjlab sampler and the field-to-command mapping.
- **Absolute anti-stall terms are incompatible with slow commands.** These include r_stall at 0.1 m/s (Rudin, Xu), ABS's stall term, "too small commands" penalties, and Gallant's 1 m in 4 s. Any such term needs to be expressed relative to v_cmd, for example speed < 0.5·v_cmd while far from the goal. CaT's ε_track = 0.2 m/s is larger than the 0.1 m/s command, so it needs a relative tolerance too.
- **The velocity-tracking reward's pull is small at this scale (my arithmetic).** With the legged_gym kernel and its default tracking_sigma of 0.25 (the config value is from memory and was not re-fetched in this session), running 0.9 m/s under a 0.1 m/s command still earns exp(−0.64/0.25) ≈ 0.08 of the tracking reward per step. A one-off 2000 bonus discounted at 0.98 dominates any tracking term of order 1 per step. Tracking only enforces compliance if it is not out-scaled by the goal bonus.
- **A compliance check that matches the literature's metrics** (ABS reports v̄ and v_peak on success; Jin et al. report time cost) would record v̄ and the 95th-percentile speed inside the gap against v_cmd, for each commanded speed, with contact and success rates. This is my suggestion, not a published protocol.

### Gaps
- No paper found trains a humanoid or legged whole-body policy with an operator-chosen slow speed (≤ 0.2 m/s) through narrow gaps and verifies compliance there. Jin et al.'s v̄ conditioning is the closest, but its range and compliance were not reported in the text.
- The search was constrained (no WebSearch), so humanoid speed-conditioned clutter work from 2025–2026 may exist unseen.

---

## Q4. Which fixes are reported to work?

### Takeaway
The evidence is strongest for a group of changes:
- Paying the task reward at a fixed time, or holding it, instead of at a discounted arrival instant.
- Conditioning on time-to-go with a budget that sets speed.
- Converting per-second proximity costs to per-metre costs.
- Capping dense progress at v_cmd.
- Using a shield or a command-tracking layer to slow down in tight spaces.

Higher γ, average reward and reward centering are supported as general improvements and shrink the haste term, but none is shown to fix haste alone. Explicit speed constraints are easy to satisfy on benchmarks, but Lagrangian methods oscillate. Motion priors improve slow-speed gait quality on flat ground.

### Cited Findings
- **Final-time reward with time-to-go, no timeout bootstrap and short episodes.** Rudin 2022 used T = 6 s, a reward over the last 1 s, time left given to actor and critic, and removed bootstrapping "and obtain more stable training". It solved 1.2 m gaps against 0.15 m for velocity tracking. The time budget modulates speed ("slow and safe gait"), but a stall penalty was required against wait-then-sprint — [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)
- **Longer time budget to slow aggressive motions** — [Zhang et al. 2023](https://arxiv.org/abs/2311.10484)
- **Distance-scaled (per-metre) proximity cost** to stop "sprinting through the obstacles" — [Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)
- **Dense progress capped at commanded speed**, in the world frame toward the next waypoint, which also removed the walk-around exploit — [Cheng et al. 2023](https://arxiv.org/abs/2309.14341)
- **Average reward and discount choice.**
  - ATRPO outperformed TRPO on Humanoid by 50.1% at its best γ (1000-step evaluation) and by 913% (10,000-step evaluation).
  - The best γ for TRPO differed by task: 0.99 for HalfCheetah and Ant, 0.95 for Humanoid.
  - Larger γ "can also increase the variance of the gradient estimator … and degrade generalization", and discounted algorithms "become unstable as γ → 1".

  — [Zhang & Ross 2021](https://arxiv.org/abs/2106.07329). Reward centering makes high γ easier to estimate and makes methods robust to constant offsets — [Naik et al. 2024](https://arxiv.org/abs/2405.09999)
- **Explicit speed constraints.**
  - Velocity-cost CMDP tasks are met by all tested safe-RL algorithms — [Ji et al. 2023](https://arxiv.org/abs/2310.12567)
  - Overspeed above a user limit as a penalty — [Jin et al. 2024](https://arxiv.org/abs/2310.03581)
  - Velocity tracking as a CaT constraint with ε = 0.2 m/s, which explored more and succeeded more on obstacles (90.6% against 79.4%) — [Chane-Sane et al. 2024](https://arxiv.org/abs/2403.18765)
  - PID multiplier updates to damp Lagrangian oscillation — [Stooke et al. 2020](https://arxiv.org/abs/2007.03964)
- **Shield and recovery tracking.** Collisions fell from 15.5–26.1% to 2.9–5.7% while keeping about 1.9–2.2 m/s average, by slowing only in tight spaces — [He et al. 2024](https://arxiv.org/abs/2401.17583)
- **Hierarchy.** Local goals from a planner with a goal-reaching low level gave 96.8/51.5/39.4% success against 2.4–65.7% for pure velocity-tracking skills — [Xu et al. 2024](https://arxiv.org/abs/2403.03848)
- **Speed-aware curricula.**
  - Grid adaptive curriculum over the command space — [Margolis & Agrawal 2022](https://arxiv.org/abs/2212.03238)
  - Promotion and demotion against v_cmd·T — [Cheng et al. 2023](https://arxiv.org/abs/2309.14341)
- **Imitation and motion priors.** AMP tracked a 0.4 m/s command at 0.36 m/s with cost of transport 1.07 against 1.54 for the complex hand-designed style reward. It reduced "aggressive and overly-energetic behaviors", on flat ground only — [Escontrela et al. 2022](https://arxiv.org/abs/2203.15103)
- **Remove the anti-cooperative haste of a time-to-goal objective** with an explicit penalty on finishing much faster than reference (CADRL's extra-time term) — [Chen et al. 2017](https://arxiv.org/abs/1609.07845)

### Inferences
These map the user's incentives (a)–(e) to the literature fixes. All are my synthesis.
- **(a) Discounted goal bonus.** Replace it with one of these, or combine them:
  - A hold-based or final-window reward whose deadline is set by the commanded speed, T_budget ≈ L_path/v_cmd + slack, with time-to-go in actor and critic and no early-arrival premium. Rudin's "slow and safe gait with more time" is the supporting evidence, and his wait-then-sprint result is the risk.
  - A small bonus plus dense world-frame progress capped at v_cmd.
  - If a terminal bonus stays, raise γ toward 0.995–0.999 with reward centering. That cuts the per-second haste premium from about 1270 to about 100 at 1 s, but does not remove it.

  An overspeed penalty above (1+ε)·v_cmd (Jin et al.) closes the remaining gap.
- **(b) Per-second clearance and distance costs.** Make them per metre (×‖v‖·dt, as in RPL), or bound them by the time budget. Pair this with a relative stall term so that per-metre costs do not make standing in the gap free.
- **(c) Cosine field alignment.** Replace v̂ with a projection capped at v_cmd, min(⟨v_k, μ_k⟩, v_cmd)/v_cmd, computed on smoothed or centre-of-mass velocity so that orthogonal gait sway does not depress slow-speed alignment.
- **(d) Contact termination.** Use graded contact, or CaT-style δ with ramped p_max, or a reach-avoid shield with a command-tracking fallback (ABS). These remove the "minimise steps in the gap" pressure that hard termination creates.
- **(e) Clip at zero.** Re-centre rewards (Naik et al.) rather than clip, so that near-wall cost gradients survive. This changes the termination balance analysed in the prior report and must be re-checked against refusal.
- **Command sampling.** Ensure commands are trained across 0.05–0.6 m/s inside gaps (no 0.2 m/s dead zone). Use a speed-aware curriculum and a sideways-gait motion prior (AMP) for slow sidestepping.
- Several of these changes interact with the refusal incentives in the prior report. A reward that stops paying for haste can make stalling attractive again unless stall income is also removed.

### Gaps
- No study compares these fixes head to head on slow-speed compliance in narrow passages.
- No humanoid evidence was found that a speed-scaled time budget reproduces Rudin's "slow and safe gait" at 0.1 m/s.
- The magnitude of the haste reduction from reward centering with γ → 1 under PPO in robotics is untested in the sources read.
