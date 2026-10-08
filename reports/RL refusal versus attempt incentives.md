# Make standing still cost more than trying

The G1 refuses because, as the MDP is currently written, refusing is the right answer. PPO is optimizing it correctly. About 15 per step of reward comes from upright, orientation and guidance-alignment terms that the robot earns just as well standing in front of a hurdle as it does crossing one. A failed 20 s timeout is bootstrapped as if the episode went on forever. Any contact cuts that stream off. So crossing has almost no upside and real downside. In the literature this is the "positive per-step reward plus implicit zero-value terminal" trap, the same mechanism as the survival bias in DAC ([Kostrikov et al.](https://arxiv.org/abs/1809.02925)) and the textbook result that a constant reward offset changes the optimal policy in episodic tasks ([Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)).

There is no single fix in the literature. Published traversal systems that avoid refusal all change three things together:

- **What a failed deadline is worth.** Make it terminal, give the critic time-to-go, and shorten episodes ([Pardo et al. 2018](https://arxiv.org/abs/1712.00378); [Rudin et al. 2022](https://arxiv.org/abs/2209.12827)).
- **What standing still earns.** Add an explicit "don't wait" or stall penalty and stop paying posture terms to a robot that is stalled ([Hoeller et al. 2023](https://arxiv.org/abs/2306.14874)).
- **How hard the collision cliff is.** Make body contact non-terminal or graded, and use penetrable obstacles early ([Zhuang et al. 2023](https://arxiv.org/abs/2309.05665); [Chane-Sane et al. 2024](https://arxiv.org/abs/2403.18765)).

Your experiment (b) failed because it touched only the third lever, and only partly: hand contact stayed terminal. That leaves curricula and exploration aids. They help only after the incentive is fixed, which is consistent with your width curriculum plateauing.

## Refusal is the optimum of the current MDP

Two local code facts drive the problem. In `cat_mjlab/task.py:1409`, `truncated = timeout & ~terminated`, so every timeout counts as a truncation, including "stood at the hurdle for 20 s". The Brax-style `compute_gae` in `cat_mjlab/learning.py:156` then bootstraps the last step from V(s_T). For a robot standing upright, V(s_T) ≈ r_stand/(1−γ). With r ≈ 15 and the function's default γ = 0.98 (the training config value was not checked), that is about **750**. The deadline therefore carries no signal that the task failed. A contact termination, by contrast, gets a hard zero bootstrap. This is the effective comparison:

| Branch from the obstacle face | Approximate value |
|---|---|
| Stall to the timeout | 15 per step to the deadline, then a bootstrap of about 750: the full stream |
| Attempt and touch (terminal) | Reward up to the contact, then 0 |
| Attempt and cross cleanly | About 15 per step, much the same as stalling, plus a small alignment gain |

A clean crossing pays almost exactly what stalling pays, so attempting is worthwhile only when the success-only extras exceed p_crash·r_stand/(1−γ). Early in training p_crash is close to 1, so that condition fails.

Pardo et al. define the correct treatment. When a time limit is part of the task, hitting it is a real terminal and the agent must observe the remaining time. Bootstrapping, which they call partial-episode bootstrapping, is reserved for limits that exist only to reset indefinite tasks ([Pardo et al. 2018](https://arxiv.org/pdf/1712.00378)). "Reach the goal past the obstacles" is a deadline task. The bootstrap default came from velocity tracking, where Rudin et al. found it "greatly reduces the critic loss and improves the total reward by approximately 10% to 20%" ([Rudin et al. 2021](https://arxiv.org/abs/2109.11978)). That justification is real, but it applies to tasks with no goal. In their later goal-reaching work the same group wrote that bootstrapping "was introduced to mimic an infinite horizon problem … However, in our case, the time until task termination is clearly defined and provided as input to both the actor and the critic. We, therefore, remove the bootstrapping and obtain more stable training" ([Rudin et al. 2022](https://arxiv.org/abs/2209.12827)). IsaacLab exposes the same choice as `is_finite_horizon` ([IsaacLab](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py)).

The positive stream also turns termination into a large implicit penalty, roughly r/(1−γ). Its size is tied to your posture weights, not to any deliberate collision cost. Raising the upright weights therefore also raises the crash penalty, and with it the incentive to refuse. DAC describes the same thing: a strictly positive reward "implicitly provides a survival bonus … the agent keeps collecting positive rewards without actually attempting to solve the task" ([Kostrikov et al.](https://arxiv.org/abs/1809.02925)). Shifting the offset negative has the mirror-image failure, where the agent prefers to end episodes early ([Kostrikov et al.](https://arxiv.org/abs/1809.02925); [Hämäläinen et al. 2019](https://arxiv.org/pdf/1909.07869)). legged_gym guards against that with `only_positive_rewards`, which clips the total reward at zero before a termination penalty is added ([legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)). The base reward has to be designed so that stalling, crashing and crossing are ranked on purpose. Rescaling it does not achieve that.

### Your four experiments, read through this lens

**(a) Hard contact termination.** Refusal is predicted from the analysis above.

**(b) Contact as a 3-per-step penalty, hands still terminal.** Refusal rose because this changed the cost of attempting without creating any reward for crossing. Contact cost 3 per step, so a robot in contact earned about 12 per step while a stalled robot earned about 15. A robot that was through the obstacle earned about 15, the same as stalling. Being near or in contact therefore became strictly worse than standing back, with nothing on the far side to compensate. Three further effects likely made it worse:

- Hand contact stayed terminal. In a sideways gap passage the hands are the most likely point of contact, so the cliff stayed exactly where the attempt happens. This is our inference.
- Without body-contact resets, stuck episodes run their full 20 s, which means fewer fresh attempts per sample. This is our inference.
- Penalties of fixed size get "bought" or freeze learning depending on their magnitude, and "there is no invertible map between 'desired safety specification' and 'correct trade-off parameter'" ([Ray et al., Safety Gym](https://cdn.openai.com/safexp-short.pdf)).

**(c) Width curriculum with sideways starts.** This improved exploration but left the optimum unchanged, so it plateaued. More on this in the curriculum section below.

**(d) Sideways bonus not farmed by standing.** This shows the problem is not that one bonus is exploitable. Most of the stall income comes from terms that are legitimately earned while upright.

One term deserves a code check. In the Click-and-Traverse paper the field reward is Σ_k[log C_d(κ_k) + κ_k μ_kᵀ v̂_k], a von Mises–Fisher log-likelihood over unit motion directions ([Xue et al. 2026](https://arxiv.org/abs/2601.16035)). If your implementation follows it:

- the log C_d(κ) normalizer is paid whether or not the robot moves;
- v̂ is a unit vector, so tiny jitter in the right direction can collect the full alignment term.

Both are inferences from the formula, not measurements. They are worth confirming because they would put "guidance alignment" in the stall income column. Extreme Parkour hit a related exploit: base-frame velocity tracking taught the robot to walk around hurdles, and on steps "all it learns is a colliding and retrying behavior". World-frame inner-product tracking toward the next waypoint fixed it ([Cheng et al. 2023](https://arxiv.org/abs/2309.14341)).

## Every successful traversal paper prices waiting explicitly

The published systems fall into two families, and neither pays a robot for standing still in front of an obstacle.

**Dense velocity tracking: Robot Parkour Learning, Extreme Parkour, Humanoid Parkour, legged_gym.** The main positive term is forward-velocity tracking, which is zero or negative at zero speed. Robot Parkour Learning's r_forward = −α1|v_x − v_target| makes standing cost about |0 − 1| per step even with an alive bonus of +2 ([Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)). Where an alive bonus exists (Robot Parkour Learning +2, [Radosavovic et al. 2024](https://arxiv.org/abs/2410.03654) +1, [Lee et al. 2024](https://arxiv.org/abs/2405.01792) +1), obstacle contact does not terminate the episode. The alive bonus therefore never competes with attempting. Robot Parkour Learning's code terminates only on roll, pitch and height, and charges collision −0.5 per step against the +2 alive bonus ([parkour repo](https://github.com/ZiwenZhuang/parkour/blob/HEAD/legged_gym/legged_gym/envs/a1/a1_leap_config.py)). Extreme Parkour charges −10 per colliding body and has no alive term ([extreme-parkour](https://github.com/chengxuxin/extreme-parkour/blob/HEAD/legged_gym/legged_gym/envs/base/legged_robot_config.py)).

**Sparse "be at the target by time T": the ETH line and ABS.** Every paper in this family pairs the sparse goal reward with an explicit anti-waiting term:

- Rudin 2022: r_stall = −1 when speed is below 0.1 m/s and the robot is more than 0.5 m from the target, plus a cosine-to-target bias that is removed once the task reward reaches 50% of its maximum ([Rudin et al. 2022](https://arxiv.org/abs/2209.12827)).
- ANYmal Parkour: "Don't wait" 1(‖v_b‖ < 0.2) at −1, "Move in direction" at +1, position tracking only in the last second (weight 10), and base-collision termination at **−200** ([Hoeller et al. 2023](https://arxiv.org/abs/2306.14874)).
- Risky Terrains: "Don't wait" disabled within 1 m of the target, and the direction reward applied only for the first 150 iterations ([Zhang et al. 2023](https://arxiv.org/abs/2311.10484)).
- ABS: −20·r_stall, and an r_agile term designed so "the robot has to either run fast or stay at the goal" ([He et al. 2024](https://arxiv.org/abs/2401.17583)).

Rudin et al. explain why the stall term is needed even without an alive bonus. Under discounting the policy learns "to wait until the last moment when it suddenly runs fast towards the target" ([Rudin et al. 2022](https://arxiv.org/abs/2209.12827)). Their finite-horizon goal reward also cleared obstacles far larger than velocity tracking did at 95% success: gaps of 1.2 m vs 0.15 m, and obstacles of 0.85 m vs 0.35 m. The cost was a harder critic, where "a random seed change can cause a failed training run". They mitigated it with time-to-go in both actor and critic, no timeout bootstrap, episodes shortened from 20 s to 6 s, and a doubled batch. Your 20 s episode is long compared with the 5 s success criterion in the Click-and-Traverse evaluation itself ([Xue et al. 2026](https://arxiv.org/abs/2601.16035)).

Adding progress shaping will not fix this on its own. Potential-based shaping F = γΦ(s′) − Φ(s) is the only additive shaping guaranteed to preserve optimal policies ([Ng et al. 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)). If refusal is optimal under the base reward, a geodesic potential leaves it optimal and only speeds up learning. Pure PBRS also telescopes to the same total for stalling, crashing and succeeding once terminal potentials are set to zero ([Grześ 2017](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf)). The practical form, Habitat-style undiscounted d_{t−1} − d_t on a geodesic field, plus a success bonus and a small time penalty ([Savva et al. 2019](https://arxiv.org/abs/1904.01201); [Wijmans et al. 2020](https://arxiv.org/abs/1911.00357)), deliberately breaks invariance. A robot keeps credit for progress made before a crash, which mildly favors attempting. That is the bias you want, but it works only alongside a base reward that stops paying for stalling. Ng's bicycle example, an agent paid for progress that "learned to ride in tiny circles", is a warning against velocity-alignment terms that are not potential-based ([Ng et al. 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)).

## Collision handling: soften the cliff, but keep the hand constraint for the end

Across these works, obstacle contact by limbs is almost never a hard termination. Terminations are reserved for falls, base or torso crashes, and extreme impact forces ([Hoeller et al. 2023](https://arxiv.org/abs/2306.14874); [Rudin et al. 2021](https://arxiv.org/abs/2109.11978)). A contact-terminates, timeout-bootstraps combination like yours did not appear in any surveyed paper.

### Constraints-as-Terminations helps only with graded contact or a schedule

CaT (Constraints as Terminations) is the principled middle ground. It maps each violation to a termination probability δ = max_i p_i^max·clip(c_i⁺/c_i^max, 0, 1), multiplies rewards by (1−δ), and passes δ to GAE as a soft done ([Chane-Sane et al. 2024](https://arxiv.org/abs/2403.18765)). It beat N-P3O on flat terrain: 682.9 vs 593.2 reward, with 0.5% vs 8% torque violation. Binary termination on every violation "entirely fails to learn". Its parkour policies reached 79–91% real-robot success.

Two caveats matter for you:

- **CaT's collision constraints were all hard (p_max = 1).** On a binary contact indicator, CaT with p_max = 1 is exactly your current hard termination. The exploration benefit appears only if contact is graded (penetration depth, force, or a clearance margin) or if p_max is ramped. CaT ramped p_max from 0.05 to 0.25 only for soft constraints, so applying a ramp to contact is an extrapolation.
- **CaT itself saw refusal-like behavior.** "CaT (Tracking Rewards) often refuses to walk over the stairs sideways." The variant that encodes the task itself as constraints explored more and succeeded more, at the cost of more violations ([Chane-Sane et al. 2024](https://arxiv.org/abs/2403.18765)).

### Constrained-RL methods show the same conservatism

Lagrangian methods do not escape this. In Safety Gym, PPO-Lagrangian satisfied the cost limit but kept only **0.24 of PPO's return** ([Ray et al.](https://cdn.openai.com/safexp-short.pdf)). That is the refusal trade-off seen at benchmark scale. ETH's comparison found N-P3O, P3O and PPO-Lagrangian all workable on legged locomotion, and their main advantage over hand-tuned penalties was less coefficient tuning ([Lee et al. 2023](https://arxiv.org/pdf/2309.15430)). A constrained-RL method is a reasonable way to enforce the hand constraint after the attempt incentive exists. It does not create that incentive.

### A recent analysis formalizes the hovering failure (unverified)

A 2026 preprint gives a formal account of the failure you see. **SCoCaT** (arXiv 2609.06061; not independently verified) proves that when constraints activate exactly where the agent must pass to succeed, "hovering" just outside is a strict local optimum of CaT's survival-weighted objective ([Arora et al. 2026](https://arxiv.org/html/2609.06061)). It reports CaT completing 0.4% of docking tasks against 100% for its fix. The fix is a separate success critic bootstrapped only on environment terminations, A_tot = standardize(A_r + A_s). Adding success as a plain reward bonus instead sharply increased violations: line-of-sight violations went from 6.6% to 82.6%. Some of its figures are internally inconsistent (61.3% vs 79.4% on one task). Treat it as a supporting theory for your stall, and its decoupled critic as an optional experiment. A related 2026 preprint shows that survival-based constrained RL equals Lagrangian RL up to a constant offset that "implicitly incentivizes survival" ([Milosevic et al. 2026](https://arxiv.org/abs/2602.04599)). It is also recent and secondary.

### Your hand requirement

The hard hand requirement is not standard in the literature. No surveyed humanoid paper enforces "hands never touch" or reports how it would. The defensible structure separates the training signal from the evaluation criterion:

- Body contact becomes a graded, velocity-scaled penetration or contact penalty.
- Hands get a dense clearance-margin penalty that starts before contact, so the gradient arrives before the cliff.
- Hand contact during exploration is soft: graded δ with p_max ramped toward 1, or a large finite penalty.
- Hard hand termination is restored only in a final fine-tuning stage.
- Strict success (no contact at all) stays the only metric reported.

Robot Parkour Learning's velocity scaling exists "to prevent the robot from exploiting the penetration reward by sprinting through the obstacles" ([Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)). Use the same scaling here.

## Curricula and soft obstacles fix exploration, not incentives

The strongest ablation in this literature is Robot Parkour Learning's soft-dynamics stage. Obstacles are made penetrable so the robot does not get "stuck near the obstacles as a result of local minima", with a penetration penalty and a per-robot difficulty score that hardens geometry when penetration is low. Hard dynamics are restored for fine-tuning ([Zhuang et al. 2023](https://arxiv.org/abs/2309.05665)):

| Skill | Success without soft dynamics | Success with soft dynamics |
|---|---|---|
| Climb | 0% | 95% |
| Leap | 0% | 82% |
| Crawl | 93% | 100% |
| Tilt | 86% | 100% |

Neither RND curiosity nor privileged oracles without soft dynamics made any progress on climbing and leaping. The split matters for you. Hurdles resemble climb and leap, where soft dynamics was decisive. Narrow side gaps resemble tilt, which is squeezing through slits narrower than the body and was learnable without it. Penetrable hurdles should therefore be the higher-value change. For gaps, start-state design matters more.

Your width curriculum with sideways starts is a partial reverse curriculum. The literature identifies what it is missing. Reverse curriculum generation keeps only starts with success probability between 0.1 and 0.9 and grows new starts backward from successful states ([Florensa et al. 2017](https://arxiv.org/abs/1707.05300)). Agents trained from uniform starts reached only about 10% and 2% success and never learned from distant starts. Unfiltered expansion was "still better than not modifying the start state distribution but has a slower learning rate". Goal GAN reports these 0.1–0.9 bounds are robust ([Florensa et al. 2018](https://arxiv.org/abs/1705.06366)). Salimans & Chen move the reset point backward once 20% of workers match the demonstration, and argue this turns exponential exploration into quadratic ([Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)).

A plateau at 0–15% success fits a batch dominated by starts that never succeed. The concrete change:

- Harvest in-gap and post-gap states from successful rollouts.
- Estimate per-start success, which thousands of parallel environments make cheap.
- Keep starts in the 10–90% band and replay older ones.
- Keep a share of nominal starts.

Demotion rules matter just as much. legged_gym-style curricula demote robots that do not cover enough distance ([Rudin et al. 2021](https://arxiv.org/abs/2109.11978)), and Humanoid Parkour promotes only when the robot "safely finishes the task and moves at least 3/4 distance" ([Zhuang et al. 2024](https://arxiv.org/abs/2406.10759)). Under these rules a robot that stalls is moved off hard levels automatically. A curriculum that holds or promotes on survival or "no contact" does the reverse.

Several published works also report getting stuck when the goal itself is too hard to discover:

- ANYmal Parkour without its goal-distance curriculum "gets stuck in front of larger obstacles" ([Hoeller et al. 2023](https://arxiv.org/abs/2306.14874)).
- Lee et al. found infeasible goals made the policy "overly conservative". They fixed it with feasible graph-sampled goals and a penalty for revisiting buffered positions ([Lee et al. 2024](https://arxiv.org/abs/2405.01792)).

Other aids are optional. AMP-style motion priors could make sidestepping reachable, since your scratch area contains AMASS sidestep clips ([Peng et al. 2021](https://arxiv.org/abs/2104.02180)). Recent G1 clutter systems such as PASSAGE and MTC sidestep the problem by tracking human reference motion, which supplies forward progress by construction ([PASSAGE](https://arxiv.org/abs/2609.18732); [MTC](https://arxiv.org/abs/2609.21107)). Both are 2026 preprints and were not independently verified. Every curriculum paper above assumes an incentive structure that already makes crossing worth more than standing. Applied to the current reward, a reverse curriculum would teach the gap from close starts and still leave refusal from far starts.

## Prioritized recommendations for the G1 setup

The table orders changes by expected impact per unit of effort and by how firmly the literature supports them. Make changes 1–3 together, because each one alone leaves the other levers sustaining refusal. "Established" means peer-reviewed and widely reproduced. "Extrapolation" means our inference beyond what the cited work tested. "Recent/unverified" means it rests mainly on the 2026 preprints.

| # | Change | Concrete setting | Evidence |
|---|---|---|---|
| 0 | Measure the incentive before changing it | Log attempt rate (crossed the obstacle's front plane), strict success, and the critic's V at the obstacle face for stalled vs approaching states | Diagnostic |
| 1 | Stop paying the robot for stalling | Gate the upright/orientation/alignment terms: multiply them by a not-stalled factor, or rescale them so total standing reward is about 1–2 per step. Add a stall penalty of the same order as the remaining standing income when speed < 0.1–0.2 m/s and d_goal > 0.5 m. Use net progress over a window, or a revisit-buffer penalty, so creeping and jitter don't pass | Established ([Rudin 2022](https://arxiv.org/abs/2209.12827), [Hoeller 2023](https://arxiv.org/abs/2306.14874), [ABS](https://arxiv.org/abs/2401.17583), [Lee 2024](https://arxiv.org/abs/2405.01792)); sizing against 15 per step is extrapolation |
| 2 | Make a failed deadline terminal | At `task.py:1409`, set `terminated=True` for timeouts that did not reach the goal; keep `goal_hold` successes as truncation. Add normalized time-to-go to the critic, and to the actor too. Shorten training episodes toward 6–10 s. With γ around 0.98 a 20 s cliff is invisible from early steps, which is why #1 must ship alongside | Established ([Pardo 2018](https://arxiv.org/abs/1712.00378), [Rudin 2022](https://arxiv.org/abs/2209.12827)) |
| 3 | Pay for crossing | World-frame geodesic progress d_{t−1}−d_t on the fast-marching field, not zeroed at crashes; a success bonus at goal or hold; a cosine-to-waypoint bias that is annealed off at 50% task success. Check the vMF normalizer and unit-vector issue in the alignment term | Established ([Ng 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), [Savva 2019](https://arxiv.org/abs/1904.01201), [Cheng 2023](https://arxiv.org/abs/2309.14341)) |
| 4 | Keep the reward signs safe | Clip the per-step total at ≥ 0 before adding terminal terms. Size the contact termination penalty so V(stall to deadline) is at or slightly below V(crash) during exploration, as in the ETH balance of −200 termination against a −1 per step "don't wait" over 5–10 s, then tighten it | Established ([legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py), [Zhang 2023](https://arxiv.org/abs/2311.10484)); the balance rule is extrapolation |
| 5 | Soften contact, staged | Body: graded, velocity-scaled penetration penalty with no termination. Hands: a clearance-margin penalty plus graded δ with p_max ramped from about 0.1 to 1. A final fine-tune with hard hand termination. Report strict success only | Established ([Zhuang 2023](https://arxiv.org/abs/2309.05665), [CaT](https://arxiv.org/abs/2403.18765)); the ramp on contact is extrapolation |
| 6 | Penetrable-obstacle pre-stage for hurdles | MuJoCo contype/conaffinity masking, a penetration penalty, a per-robot difficulty score of ±0.05 driven by penetration, then a hard-dynamics fine-tune | Established, 0→95% on climb ([Zhuang 2023](https://arxiv.org/abs/2309.05665)); humanoid transfer untested |
| 7 | Replace fixed sideways starts with an adaptive reverse curriculum | Starts harvested from successful rollouts, kept in the 10–90% success band, about 1/3 replay; progress-based promotion and demotion | Established ([Florensa 2017](https://arxiv.org/abs/1707.05300), [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381), [Rudin 2021](https://arxiv.org/abs/2109.11978)) |
| 8 | Optional: decouple the attempt incentive from contact termination | A separate success critic bootstrapped on environment terminations only | Recent/unverified ([SCoCaT](https://arxiv.org/html/2609.06061)) |
| 9 | Optional: hand constraint via constrained RL after #1–#7 work | N-P3O or PID-Lagrangian on hand-contact probability with an adaptive budget | Established in locomotion ([Lee 2023](https://arxiv.org/pdf/2309.15430), [Kim 2024](https://arxiv.org/html/2308.12517)); expect a return cost ([Safety Gym](https://cdn.openai.com/safexp-short.pdf)) |

Several gaps remain open. No surveyed paper measures refusal rate against alive-bonus size, compares soft dynamics, CaT and Lagrangian methods on the same obstacle task, or applies any of them to a humanoid with a strict no-hand-contact rule. The training γ, GAE λ and control rate in this repo were not checked, and the 750 figure depends on them. Whether the vMF normalizer is actually paid while stalled also needs a code read.

## Conclusion

The refusal problem is a specification error, not an exploration failure. Experiment (b) is the clearest evidence: once collision was cheaper but crossing was still worth nothing extra, refusals went up rather than down. The literature's answer is to give the deadline a cost and to price standing still against the same scale as the posture rewards, so that the gradient favors attempting before any exploration aid is applied. Soft obstacles and reverse curricula then become accelerators instead of attempts to fight the optimum. A practical consequence follows: refusal rate and the critic's value at the obstacle face should be tracked as first-class training metrics. If V(stall) is still at least V(approach) after a change, no curriculum or contact schedule will produce consistent attempts.

The hand requirement is best treated as the final layer, not the first. A cliff that is present from the start teaches the robot to avoid the region where it has to learn. Staging it, from graded clearance to probabilistic termination to hard termination with strict-success evaluation, is an extrapolation from CaT and Robot Parkour Learning rather than a published recipe. It is also the only route the evidence leaves for meeting a zero-contact requirement without bringing refusal back.
