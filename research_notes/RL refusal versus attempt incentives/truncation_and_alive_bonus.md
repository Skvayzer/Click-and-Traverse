# Truncation vs termination, and alive bonuses, as drivers of "stall safely instead of attempting" behaviour

Scope: theory and established practice for (a) time-limit truncation vs termination in PPO/GAE and (b) positive per-step ("alive"/survival) rewards interacting with failure terminations, applied to a G1 obstacle-traversal policy that stands in front of hurdles/gaps until the 20 s timeout. Collisions terminate with no bootstrap; the timeout is bootstrapped; ~15/step of mostly positive per-step reward.

Local code facts used below (no URL; read directly in this session):
- `cat_mjlab/learning.py:156` `compute_gae` is the Brax formulation: `mask = 1 - truncation`; `delta = (r + γ(1-termination)·V_next - V)·mask`; the accumulator is multiplied by `mask`. So the truncated transition is dropped from the loss, and the step before it bootstraps from V(last state). Functionally this is partial-episode bootstrapping (PEB) for every timeout, successful or not.
- `cat_mjlab/task.py:1409` `truncated = timeout & ~terminated`. Every timeout, including "never reached the goal", is a truncation. The `goal_hold_seconds` path (task.py ~1401-1408) also ends successful room episodes as truncations, explicitly "so arriving early is not taxed by losing the future standing reward".
- The default discount in `compute_gae` is 0.98. The actual training config value was not checked.

---

## Q1. What do Pardo et al. (ICML 2018) recommend, and when should a timeout be terminal vs bootstrapped?

### Takeaway
Pardo et al. distinguish two cases. (i) If the time limit is part of the task, the timeout is a real environmental terminal, and the agent must observe the remaining time. (ii) If the limit only exists to reset/diversify training on an indefinite task, bootstrap at the timeout (PEB). Bootstrapping a timeout tells the learner "the episode would have continued with the same value". For a goal-reaching task that silently converts "ran out of time without succeeding" into "worth as much as continuing to live", so stalling becomes free.

### Cited Findings
- Paper: Fabio Pardo, Arash Tavakoli, Vitaly Levdik, Petar Kormushev, "Time Limits in Reinforcement Learning", ICML 2018, PMLR 80:4042-4051. The abstract says mishandled time limits cause "state aliasing and invalidation of experience replay, leading to suboptimal policies and training instability." — [arXiv 1712.00378](https://arxiv.org/abs/1712.00378)
- Case (i), fixed-period tasks: "the terminations due to time limits are in fact part of the environment, and thus a notion of the remaining time should be included as part of the agent's input to avoid violation of the Markov property" (time-awareness, TA). — [arXiv 1712.00378](https://arxiv.org/abs/1712.00378)
- Case (ii), indefinite tasks: "we argue to bootstrap at states where termination is due to time limits, or more generally any other causes than the environmental ones. We refer to this approach as partial-episode bootstrapping (PEB)." — [Pardo et al. 2018, §1](https://arxiv.org/pdf/1712.00378)
- "A common mistake, however, is to then consider the terminations due to such time limits as environmental ones" (§3, about indefinite tasks). The one-step TD target under PEB is y = r if s' is an environmental terminal, else r + γ v̂π(s'), including at timeout states. — [Pardo et al. 2018, §3](https://arxiv.org/pdf/1712.00378)
- Two-Goal Gridworld (T=3, −1 per move, goals worth 50 and 20) shows three regimes (Fig. 1):
  - The standard, time-unaware agent that treats timeouts as terminal "always tries to go for the closest goal even if the remaining time is not sufficient".
  - The time-aware agent "learns to stay in place when there is not enough time to reach a goal".
  - The PEB agent "maximizes its return over an indefinite horizon, it learns to go for the most rewarding goal."
  - The aliasing derivation for the time-unaware agent is v(s) = 2/3(−1 + γ max v(s')) + 1/3(−1): values "leak" to states that are out of reach.
  — [Pardo et al. 2018, §2.2, Fig. 1](https://arxiv.org/pdf/1712.00378)
- "Last Moment problem": a time-unaware agent cannot learn to "stay in place for T − 1 steps and then jump". On Hopper-v1 (T=300) the time-aware PPO "learns to jump forward just before the time limit in order to maximize its forward distance" (a "photo finish"). Time-unaware PPO "does not learn this behavior and its training is highly destabilized when the discount factor is large." — [Pardo et al. 2018, §2.1, Fig. 5](https://arxiv.org/pdf/1712.00378)
- In the time-aware experiments the remaining time was added as a scalar normalized to [−1, 1]. Monte Carlo methods avoid the value leakage but still cannot solve the Last Moment problem without time-awareness. — [Pardo et al. 2018, §2](https://arxiv.org/pdf/1712.00378)
- PEB result: on Hopper-v1, "even if trained with episodes of only 200 steps, the agent with partial-episode bootstrapping manages to learn to hop for at least 10^6 time steps (two hours)." — [Pardo et al. 2018, §1](https://arxiv.org/pdf/1712.00378)
- Legged-robot practice that matches case (i): Rudin et al. 2022 (ETH/NVIDIA, "Advanced Skills by Learning Locomotion and Local Navigation End-to-End", IROS 2022) give "the remaining time to reach that location" as part of the command. They say bootstrapping "was introduced to mimic an infinite horizon problem, which is necessary in the case of velocity tracking since there is no termination condition for this task. However, in our case, the time until task termination is clearly defined and provided as input to both the actor and the critic. We, therefore, remove the bootstrapping and obtain more stable training." — [arXiv 2209.12827](https://arxiv.org/abs/2209.12827)

### Inferences
- The traversal task ("reach the goal past the obstacles within 20 s") is a case (i) task in Pardo's terms. Success or failure is defined relative to the deadline. The 20 s limit is not a training convenience on an indefinite task, so by Pardo's criterion a timeout without success is an environmental terminal.
- Bootstrapping a failed timeout (what `compute_gae` does now) gives the critic the target r + γV(s_T). With dense positive rewards, V(s_T) of a robot standing upright ≈ r_stand/(1−γ). At r≈15 and γ=0.98 that is ≈750, roughly 50 steps' worth. So the timeout carries no information that the task was failed. Attempting and colliding, on the other hand, gets a hard 0-bootstrap.
- Back-of-envelope: stalling scores ≈ "live forever at r_stand". Attempting scores ≈ (1−p_crash)·(live forever at r_attempt, plus success extras) + p_crash·(0 after the crash). Unless the reward gap between progressing and standing exceeds p_crash·r_stand/(1−γ), refusal is the optimal policy under the current MDP. This is a correctly optimized but mis-specified objective, not a PPO failure.
- Pardo's time-aware Gridworld shows that when the task is genuinely finite-horizon and success is unreachable, "stay in place" can be optimal. The cure is to make the objective say that failing to arrive is bad, not just to add time to the observations.
- A failed timeout should be bootstrapped only if the intended task is "behave well indefinitely" (velocity tracking, standing). Treat it as terminal, ideally with time-to-go in the observations, when the task is defined by reaching something before the deadline.

### Gaps
- Pardo et al. do not directly analyse "bootstrapping a timeout in a task whose success is defined by the deadline". The stall incentive above is inferred from their framework plus the reward structure, not a reported experiment.
- The training config's discount and GAE λ were not verified. The 750 figure uses the function default γ=0.98.

---

## Q2. How do Gymnasium, SB3, CleanRL, rsl_rl, legged_gym and IsaacLab handle time-outs in GAE?

### Takeaway
All major robot-RL stacks default to treating time-outs as truncation with value bootstrapping, the right choice for infinite-horizon locomotion such as velocity tracking. That default is inherited, often unnoticed, by goal-reaching tasks. IsaacLab exposes an explicit `is_finite_horizon` switch to turn it off. CleanRL's reference PPO does not bootstrap on truncation.

### Cited Findings
- **Gymnasium** defines termination as "the episode ending after reaching a terminal state that is defined as part of the environment definition" (task success or failure). Truncation is "the episode ending after an externally defined condition (that is outside the scope of the Markov Decision Process)", e.g. time limits in infinite-horizon environments. The correct target is `vf_target = rew + gamma * (1 - terminated) * vf_next_state`. For time limits inherent to a finite-horizon environment, "a representation of the remaining time must be present in the agent's observation" to keep the Markov property. — [Gymnasium "Handling Time Limits"](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- **rsl_rl** (`PPO.process_env_step`, current main) adds a bootstrap to the reward on time-outs:
  ```python
  # Bootstrapping on time outs
  if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
          self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)
  ```
  The `dones` flag still cuts the GAE trace, so the bootstrap is folded into the reward. It uses `transition.values`, the value of the current state s_t, as a stand-in for V(s_{t+1}), because the true next observation is already post-reset. — [rsl_rl ppo.py](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py)
- **legged_gym**: `self.reset_buf = torch.any(contact_forces[termination bodies] > 1., dim=1)`; `self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs`; `reset_buf |= time_out_buf`; `extras["time_outs"] = self.time_out_buf`. The optional termination penalty is `_reward_termination = self.reset_buf * ~self.time_out_buf`, which fires only for non-timeout resets. — [legged_gym legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- **IsaacLab**: `ManagerBasedRLEnvCfg.is_finite_horizon: bool = False`. The docstring says:
  - "Finite horizon: no penalty or bootstrapping value is required by the agent for running out of time."
  - "Infinite horizon: the agent needs to bootstrap the value of the state at the end of the episode. This is done by sending a time-limit (or truncated) done signal."
  - The rsl_rl wrapper does `if not self.unwrapped.cfg.is_finite_horizon: extras["time_outs"] = truncated`.
  - Built-in reward terms: `is_alive` returns `(~terminated).float()`; `is_terminated` is described as "Penalize terminated episodes that don't correspond to episodic timeouts."

  — [IsaacLab manager_based_rl_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py), [IsaacLab rsl_rl vecenv_wrapper.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py), [IsaacLab mdp/rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/rewards.py)
- **Stable-Baselines3** on-policy rollout: "Handle timeout by bootstrapping with value function, see GitHub issue #633". If `done` and `infos[idx]["TimeLimit.truncated"]`, it computes `terminal_value = policy.predict_values(terminal_obs)` and does `rewards[idx] += self.gamma * terminal_value`. Unlike rsl_rl, this uses the true terminal observation. — [SB3 on_policy_algorithm.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py)
- **CleanRL** `ppo_continuous_action.py`: `next_done = np.logical_or(terminations, truncations)`, then `nextnonterminal = 1.0 - next_done`. Truncations are treated exactly like terminations, with no bootstrap. — [CleanRL ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
- **Rudin et al. 2022** turned bootstrapping off for their time-conditioned goal-reaching task, with time-to-go in the actor and critic inputs, and report more stable training. They also lengthened the rollout (48 steps per env, ~2M samples per batch) and shortened episodes from 20 s to 6 s. — [arXiv 2209.12827](https://arxiv.org/abs/2209.12827)

### Inferences
- The legged_gym, IsaacLab and rsl_rl default (bootstrap all time-outs) was designed for velocity tracking, where there is no goal and the episode limit is pure truncation. A goal-reaching or traversal task built on these stacks inherits a case (ii) treatment for a case (i) task unless someone flips it. IsaacLab's `is_finite_horizon=True` is the official knob, and it disables bootstrapping for all time-outs.
- A finer option, which none of these libraries offer directly but which is easy to implement: emit `terminated=True` for timeout-without-success, and keep `truncated` only for time-outs where the robot is already at or holding the goal (as `goal_hold_seconds` does).
- rsl_rl's V(s_t)-for-V(s_{t+1}) approximation is harmless for standing, but it reinforces the same effect: a robot standing still has s_t ≈ s_{t+1}, so the bootstrap is exactly "keep standing".

### Gaps
- The mjlab upstream (not this repo's fork) default handling of time-outs was not checked.
- No source was found that benchmarks PEB vs terminal treatment specifically on goal-reaching locomotion beyond Rudin et al.'s qualitative stability statement.

---

## Q3. How do positive alive bonuses interact with termination, and what does the literature say about reward offsets?

### Takeaway
In episodic tasks a constant reward offset is not policy-invariant. A positive per-step reward makes any termination an implicit penalty of about r/(1−γ), which drives survival-seeking. A negative per-step reward makes termination attractive, which drives early-ending behaviour. This is textbook (Sutton & Barto Ex. 3.16), was formalised as "reward bias" for imitation learning (DAC, 2018), is exploited deliberately (Hopper `healthy_reward`, DeepMimic early termination, CaT), and also appears in offline RL as a "survival instinct" (2023). Our ~15/step positive reward makes collision termination a ~r/(1−γ) penalty that dwarfs any progress reward.

### Cited Findings
- **Sutton & Barto (2nd ed., 2018), Exercise 3.16**: "adding a constant c to all the rewards in an episodic task, such as maze running. Would this have any effect, or would it leave the task unchanged as in the continuing task above?" The standard answer: in continuing tasks the constant adds c/(1−γ) to every value and leaves the policy unchanged. In episodic tasks it changes the task, because with positive c the agent prefers never to end the episode. — [Sutton & Barto book](http://incompleteideas.net/book/RLbook2020.pdf); answer as summarised in [Scott Jeen's exercise solutions](https://enjeeneer.io/sutton_and_barto/rl_exercises.pdf) (community solutions, not the authors)
- **Kostrikov et al., "Discriminator-Actor-Critic" (arXiv Sep 2018; ICLR 2019)**:
  - "a strictly positive reward function prevents the agent from solving tasks in a minimal number of steps and a strictly negative reward function is not able to emulate a survival bonus."
  - On r = −log(1−D) (always positive): "The agent is now incentivized to move in loops or take small actions ... The agent keeps collecting positive rewards without actually attempting to solve the task", and "since the reward function is strictly positive it implicitly provides a survival bonus ... it always rewards the policy for avoiding absorbing states."
  - On log D (always negative): it "penalizes every step and leads to collapsing in environments with a survival bonus". With AIRL's early negative bias, "it is common for learned agents to finish an episode earlier (to avoid additional negative penalty)".
  - Their fix: learn an explicit reward for absorbing states instead of implicitly assigning 0.

  — [arXiv 1809.02925](https://arxiv.org/abs/1809.02925)
- **Gymnasium MuJoCo Hopper**: reward = `healthy_reward + forward_reward - ctrl_cost`. healthy_reward is a fixed value (default 1) given "every timestep that the Hopper is healthy". With `terminate_when_unhealthy` (default) the episode terminates when z < 0.7, the angle is outside [−0.2, 0.2], or the state is out of range. Default episode length is 1000 steps. In v5 the healthy reward is only given when healthy. — [Gymnasium Hopper docs](https://gymnasium.farama.org/environments/mujoco/hopper/)
- **DeepMimic (Peng et al., SIGGRAPH 2018)**: the episode terminates when the torso/head touches the ground. "Once early termination has been triggered, the character is left with zero reward for the remainder of the episode. This instantiation of early termination provides another means of shaping the reward function to discourage undesirable behaviors." ET also curates data. ET and reference state initialization were identified as "critical" for dynamic skills. — [arXiv 1804.02717](https://arxiv.org/abs/1804.02717)
- **CaT: Constraints as Terminations (Chane-Sane et al., arXiv Mar 2024)**:
  - Explicitly "assume positive rewards r ⩾ 0" so that terminating "the future rewards" acts as a penalty.
  - Replaces hard terminations with stochastic ones, δ_t ∈ [0,1], so "the sum of all future rewards at t and after time are re-scaled by (1 − δt)". This lets the agent explore inside the constraint-violation region and learn to recover.
  - Notes naive hard termination on any violation "does not readily scale to complex systems such as quadruped robots with dozens of constraints."

  — [arXiv 2403.18765](https://arxiv.org/abs/2403.18765)
- **Reward shifting (Sun et al.)**: "reward shifting in the form of the linear transformation is equivalent to changing the initialization of the Q-function in function approximation". A positive shift gives conservative (pessimistic) value estimation, useful offline. A negative shift gives optimistic estimates that encourage exploration and improve online sample efficiency. arXiv Sep 2022, titled "Optimistic Curiosity Exploration and Conservative Exploitation with Linear Reward Shaping". The NeurIPS 2022 version is "Exploit Reward Shifting in Value-Based Deep-RL"; the venue is from my knowledge, not verified this session. — [arXiv 2209.07288](https://arxiv.org/abs/2209.07288)
- **Survival Instinct in Offline RL (Li, Misra, Kolobov, Cheng; arXiv Jun 2023, NeurIPS 2023)**: offline RL produces "well-performing and safe policies even when trained with 'wrong' reward labels, such as those that are zero everywhere or are negatives of the true rewards". The cause: "pessimism endows the agent with a 'survival instinct', i.e., an incentive to stay within the data support in the long term". — [arXiv 2306.03286](https://arxiv.org/abs/2306.03286)

### Inferences
- In our setup the collision termination functions as a penalty of size ≈ E[Σ γ^k r_alive] ≈ r/(1−γ). It is implicit and very large, and it scales with the total positive reward, not with any deliberate "collision cost". Raising upright/orientation weights silently raises the crash penalty and so the refusal incentive. This is DAC's "survival bias" in a hand-designed reward.
- Sun et al.'s reward-shifting result is about value-initialization effects in value-based deep RL without terminations. It does not override the episodic-task fact above. In tasks with terminations the sign and offset of per-step reward change the optimal policy itself, not just exploration (Sutton & Barto; DAC).
- Simply shifting rewards negative (e.g. subtracting 15/step) flips the problem. Termination becomes attractive, and the robot would prefer to crash early (DAC's AIRL/log D observation). Collision would then need an explicit terminal penalty at least as large as the remaining negative stream.
- CaT's stochastic termination is a principled way to keep "termination as penalty" while bounding its size per violation. A soft collision termination (δ<1 for light contact) would cut the cliff that makes any obstacle interaction catastrophic. This is inference; CaT did not study refusal.

### Gaps
- "Is Bang-Bang Control All You Need?" (Seyde et al., NeurIPS 2021) was suggested. I did not find it addressing reward offsets or termination in a way relevant here, and did not fetch it; it is not cited.
- I found no paper that quantitatively measures alive-bonus magnitude vs refusal rate in locomotion. The relationship is analytical, not empirically benchmarked in a source I found.

---

## Q4. Recommended ways to make "failing to reach the goal within the time limit" cost something, with pros and cons

### Takeaway
The best-sourced recipe comes from Rudin et al. 2022, which is close to our setting: legged robot, obstacle navigation, PPO in Isaac Gym. Put time-to-go in the actor and critic, stop bootstrapping at the deadline, pay the task reward only for being at the goal near the deadline, and add an explicit stall penalty. More generally: (1) a failed timeout should be terminal (Pardo case i), ideally with a penalty or without the alive stream; (2) the alive stream should not dominate progress; (3) the critic needs time-to-go to value the deadline.

### Cited Findings
- **Time-to-go in observation** (Pardo TA; Gymnasium; Rudin 2022):
  - Required for Markovian values when the deadline is part of the task.
  - It enabled "photo finish" behaviour in Hopper.
  - In Rudin 2022, remaining time is part of the command and is given to both actor and critic.
  - Con (Pardo §2.2): with time-awareness alone, the optimal policy may rationally "stay in place when there is not enough time to reach a goal".

  — [Pardo 2018](https://arxiv.org/pdf/1712.00378); [Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/); [Rudin 2022](https://arxiv.org/abs/2209.12827)
- **Treat the timeout as terminal (no bootstrap)**: Rudin et al. removed bootstrapping because the "time until task termination is clearly defined and provided as input", and got more stable training. IsaacLab's `is_finite_horizon=True` does the same for all time-outs. — [Rudin 2022](https://arxiv.org/abs/2209.12827); [IsaacLab cfg](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py)
- **Sparse terminal task reward, time-windowed**: r_task = (1/T_r)·1/(1+‖x*_b − x_b‖²) if t > T − T_r, else 0. Here T = 6 s and T_r = 1 s, success means within 0.5 m at episode end, and the reward is received "only if the robot does not crash until then".
  - Pro: penalties no longer trade off against the task. "As long as the robot reaches the target in time, it can freely minimize the different quantities."
  - Con: "predicting the value function seems to be more difficult with our temporally sparse task reward", which makes training "prone to instabilities, where a random seed change can cause a failed training run". Mitigated with bigger batches, shorter episodes, and removing bootstrapping.

  — [Rudin 2022](https://arxiv.org/abs/2209.12827)
- **Stall penalty** (Rudin 2022): r_stall = −1 if ‖ẋ_b‖ < 0.1 m/s and ‖x_b − x*_b‖ > 0.5 m, else 0. Their stated reason: with only negative per-step penalties and a positive terminal reward, discounting makes it "beneficial to push the negative penalties as far into the future as possible. Therefore, the policy learns to wait until the last moment when it suddenly runs fast towards the target." The stall penalty counterbalances this. — [Rudin 2022](https://arxiv.org/abs/2209.12827)
- **Exploration/direction bias reward, annealed**: r_bias = ẋ_b·(x*_b − x_b)/(‖ẋ_b‖‖x*_b − x_b‖). It is "removed once r_task reaches 50 % of it's maximum value", so it "doesn't constrain the final solution". — [Rudin 2022](https://arxiv.org/abs/2209.12827)
- **Explicit reward for absorbing/terminal states** (DAC): instead of an implicit 0 for terminal states, learn or assign the absorbing-state reward. This generalises to setting a deliberate terminal value for "timed out without success" vs "collided" vs "succeeded". — [arXiv 1809.02925](https://arxiv.org/abs/1809.02925)
- **Termination penalty only on non-timeout resets** is the legged_gym/IsaacLab convention (`reset_buf * ~time_out_buf`, `is_terminated`). Note that it penalises crashes, not failing to arrive, so it increases rather than decreases refusal when used alone. — [legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py); [IsaacLab rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/rewards.py)

### Inferences
Options for our setup, with pros and cons; all are syntheses, not tested here:

1. **Failed timeout becomes `terminated=True`** (keep truncation only for success or goal-hold).
   - Pro: a minimal code change at `task.py:1409`, and it matches Pardo case (i). The stall branch then ends with 0 future value instead of ≈ r/(1−γ).
   - Con: without time-to-go the critic cannot know when that cliff is coming, so values alias across time, as in Pardo's Gridworld. Add normalised time-to-go to the critic at minimum; the actor too, if a deadline-dependent policy is acceptable. Also, with γ=0.98 (horizon ≈ 50 steps ≈ 1 s at 50 Hz, unverified), a terminal at 20 s is invisible from the early part of the episode. That argues for a penalty plus a stall term, or a larger γ, rather than relying on the cliff alone.
2. **Terminal penalty on failed timeout** of size ≥ r_stand/(1−γ), or proportional to remaining distance.
   - Pro: explicit and tunable.
   - Con: the same horizon issue. Also, a distance-proportional penalty may reward crashing forward rather than stalling unless collision carries an equal or larger penalty.
3. **Stall penalty / progress gating** (Rudin's r_stall; or gate upright/orientation/alive terms on progress toward the waypoint).
   - Pro: dense, so it works within the γ horizon, and it targets the observed behaviour directly.
   - Con: can be gamed by jittering or creeping, so use thresholds as in Rudin (0.1 m/s, 0.5 m). It also needs a guidance target that is genuinely traversable.
4. **Reduce the alive-stream share**: scale down constant-ish positive terms, or make them potential-like, so that r_attempt − r_stand is large relative to p_crash·r_stand/(1−γ).
   - Pro: shrinks the implicit crash penalty.
   - Con: shifting negative makes crashing attractive (DAC; Sutton & Barto), so pair it with an explicit collision penalty.
5. **Soften collision termination** (CaT-style stochastic δ, or terminate only on hard/fall contact and penalise light contact).
   - Pro: removes the cliff that makes any attempt near an obstacle maximally risky, and lets the policy learn recovery.
   - Con: may allow scraping through obstacles unless the per-contact penalty is meaningful.
6. **Success bonus at the goal** (terminal reward or Rudin-style time-windowed task reward). Pro: gives attempting a positive payoff. With the current truncation-on-success, the success branch already gets the bootstrap, so a bonus has to beat "stand forever" in value terms. It works best combined with 1–4.

### Gaps
- No source evaluates options 1–6 head-to-head on humanoid traversal. Rudin 2022 is the only primary source found that ablates time-dependent task reward and documents a stall fix in legged navigation.
- Potential-based shaping (Ng, Harada, Russell, ICML 1999) is the standard way to add progress rewards without changing the optimum. I did not fetch it this session, so it is not cited as a finding.

---

## Q5. "Reward hacking by stalling or survival" in locomotion and navigation, and how authors fixed it

### Takeaway
Stalling or survival hacking is documented in two forms. (a) "Wait until the last moment" under discounting with negative per-step penalties plus a terminal task reward (Rudin 2022), fixed with a stall penalty. (b) "Collect positive per-step reward without attempting the task" under strictly positive rewards plus implicit zero-valued terminals (DAC 2018), fixed by explicitly valuing absorbing states. Our refusal behaviour is form (b), aggravated by bootstrapping failed time-outs.

### Cited Findings
- Rudin et al. 2022 (IROS 2022; ANYmal; PPO, Isaac Gym) observed "the policy learns to wait until the last moment when it suddenly runs fast towards the target". They fixed it with r_stall = −1 when slow (<0.1 m/s) and far (>0.5 m) from the target. — [arXiv 2209.12827](https://arxiv.org/abs/2209.12827)
- Kostrikov et al. 2018: with a strictly positive reward, the "agent is now incentivized to move in loops or take small actions ... keeps collecting positive rewards without actually attempting to solve the task". The fix is explicit absorbing-state rewards. — [arXiv 1809.02925](https://arxiv.org/abs/1809.02925)
- Pardo et al. 2018: a time-aware agent rationally learns to "stay in place when there is not enough time to reach a goal". Stalling can therefore be optimal under a correctly specified finite-horizon objective if reaching the goal is not worth enough. — [arXiv 1712.00378](https://arxiv.org/pdf/1712.00378)
- Li et al. 2023: pessimism plus termination and data coverage yields a "survival instinct" that dominates the actual reward labels in offline RL. This is a related but distinct mechanism (offline). — [arXiv 2306.03286](https://arxiv.org/abs/2306.03286)

### Inferences
- The pattern in our G1 policy (approach, stop at the obstacle, stand upright until 20 s) is consistent with form (b). Upright, stand-tall and orientation terms are earned equally well while standing. Collision termination costs ≈ r/(1−γ). The failed timeout is bootstrapped, so refusal costs nothing at the deadline.
- The literature-consistent fix is to attack all three levers together rather than one:
  - the value of the timeout: make failure terminal and add time-to-go;
  - the value of standing: gate or shrink alive-like terms, or add a stall penalty;
  - the cliff of collision: soften or bound it.
- Fixing only one lever leaves the others to sustain refusal. A crash penalty alone makes it worse.

### Gaps
- I found no humanoid-specific (G1/H1) paper documenting obstacle refusal under alive bonuses. Quadruped parkour papers (e.g. ANYmal Parkour 2023, Robot Parkour Learning 2023, Extreme Parkour 2023) were not examined for stall fixes in this session.
