# Classical artificial-potential-field (APF) pathologies in narrow passages, their fixes, and how they map onto HumanoidPF

Scope and how this was researched. These notes cover the classical (mostly 1984–2009) APF and reactive-navigation literature on repulsion versus attraction near narrow passages, and the fixes proposed for it. The session's WebSearch budget was already used up, so every source below was fetched directly and read in full or in the relevant sections: Koren & Borenstein 1991; Borenstein & Koren 1991 (VFH); Ulrich & Borenstein 1998 (VFH+); Khatib 1986; Krogh & Thorpe 1986; Rimon & Koditschek 1992; Barraquand, Langlois & Latombe 1992; Connolly & Grupen (harmonic functions); Fox, Burgard & Thrun 1997 (DWA); Brock & Khatib 1999 (global DWA); Kanehiro et al. 2008 (velocity damper on HRP-2); Sethian's fast-marching navigation note; J. V. Gómez's FM2 PhD thesis; LaValle's *Planning Algorithms* ch. 8; Choset's ch. 4 lecture slides; and one arXiv paper (Huber, Billard & Slotine 2022, arXiv:2205.04928, built into `papers/2205.04928/2205.04928.md`). The one allowed arXiv search returned only irrelevant physics papers. Ge & Cui (2000, 2002), Tilove (1990), Kim & Khosla (1992), vortex/circulatory-field papers, Minguez's ND and the Choset/Latombe textbooks could not be opened; they are listed under Gaps.

Project facts used for the mapping. These come from the repo, not from the literature:
- **Command rule.** `compute_cmd_from_rtf` (`cat_mjlab/task_math.py` line 69; same rule in `cat_ppo/envs/g1/env_cat.py` ~line 1394) works like this:
  - It starts from `v = 0.7 * g_root`.
  - At each body point i (head, two feet, two hands in the env_cat version) it computes `L_i = b̂_i·g_i`, where b̂ is the boundary normal and g the guidance at that point.
  - Wherever `L_i > b̂_i·v`, it adds `(L_i − b̂_i·v) b̂_i`. These corrections are **averaged over all points**: `v_new = v + mean_i(delta_i)`.
  - The command is `0.75·v_new`, and it becomes a stop if its norm is below 0.2 m/s.
  - `_operator_command` (`cat_mjlab/task.py` ~line 711) keeps only the direction of this projected command, applies the operator speed, and stops below a `cutout`.
- **Reward target.** `safe_velocity_target` (`task_math.py` line 323) removes the inward component of the target velocity point by point. Its weight ramps from 0 at 0.60 m to 1 at 0.25 m clearance. Its docstring says: "driven straight at a table the target becomes stop, at an angle it slides along the edge, along a gap it keeps going."
- **Rewards.**
  - Guidance-alignment reward (`_re_gf0`): `5·cos(v, g)`, gated by `sigmoid(40·(0.3 − sdf))`, plus a fixed bonus of 4 after crossing.
  - SDF penalty (`_re_sdf`): `−20·softplus((0.05 − sdf)/0.02)`.
  - Both are in `env_cat.py` lines 1166–1240. Both are per-step, so position-only terms are paid per unit of time.

---

## Q1. Which works documented APF failures in narrow passages and between closely spaced obstacles, and what exactly did they observe?

### Takeaway
Koren & Borenstein (1991) list four inherent APF problems, measured on a real 125 kg robot at 0.8 m/s:
1. traps due to local minima;
2. **no passage between closely spaced obstacles**;
3. oscillation in the presence of obstacles;
4. **oscillation in narrow passages**.

Problem 2 is a static force balance: the summed wall repulsion points away from the opening. Whether the robot enters depends only on how strong the target force is compared with it. Problems 3–4 are dynamic instabilities that get **worse with higher speed**, latency and robot width. The classical literature therefore treats speed as the enemy of stable passage, never as the thing that enables it. A failure that appears only at slow speed matches problem 2 (a static barrier or stall), not problems 3–4.

### Cited Findings

#### Koren & Borenstein 1991, "Potential Field Methods and Their Inherent Limitations for Mobile Robot Navigation", ICRA 1991, pp. 1398–1404
Links: [DOI 10.1109/ROBOT.1991.131810](https://doi.org/10.1109/ROBOT.1991.131810); copy read: [CMU mirror](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf).
- **Method (VFF).** A 33×33 window of 10 cm histogram-grid cells moves with the robot.
  - Each occupied cell pushes the robot with `F_ij = F_cr·W^n·C_ij / d^n(i,j)`, along the unit vector from the cell to the robot. Here W is the robot width and n = 2 is used.
  - The target pulls with a force `F_ct` of **constant magnitude**.
  - The resultant is `R = F_r + F_t`. Its direction sets the steering-rate command `Ω = k(δ − θ)`. — [Koren & Borenstein 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **The four problems.** Quote: "we identified the following 4 significant problems that are inherent to PFMs and independent of the particular implementation: 1. Trap situations due to local minima (cyclic behavior). 2. No passage between closely spaced obstacles. 3. Oscillations in the presence of obstacles. 4. Oscillations in narrow passages." — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **No passage (Sec. 5.2).** Quote: "Fig. 4 shows a mobile robot at an attempt to pass among two closely spaced obstacles (e.g., passing through a door frame). With PFMs, the repulsive forces from obstacle 1 and 2 are combined into the two lumped repulsive forces F′r1 and F′r2 … The sum of all repulsive forces (Fr=F′r1+F′r2) points straight away from the opening between the two obstacles. Depending on the relative magnitude of the target-directed force Ft, the robot will either approach the opening further, or it will turn away." Fig. 4 caption: "Under PFM control, the robot does not pass among densely spaced obstacles." No speed term appears in this explanation. — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Robot–environment model.**
  - The steering motor is a first-order lag: `τ·ω̇ + ω = Ω`.
  - Next to a long wall, the lateral repulsion is `F_r = F_rx·W^n/d^n`, with `f = F_rx/F_ct`.
  - Under a sampling delay T the characteristic equation is `τs³ + s² + k(1 − NVT)s + kNV = 0`, where `N = n·W^n·f·cos²δ/(d^{n+1}·sin α)`.
  - The Routh stability condition is `1 − NV(τ+T) > 0` (Eq. 20). Quote: "The condition is harder to be satisfied for larger V, τ, and T; for example, if a heavy vehicle moves at high speed, it is more likely that oscillations occur." The tunable parameter is f, the repulsion/attraction ratio. — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Wall following (Sec. 5.3).**
  - Stability needs `d^{n+1} ≥ nV(τ+T)W^n f / sin α` (Eq. 21).
  - At the steady-state stand-off, where lateral forces balance (`F_r = F_ct cos α`), this becomes `f ≥ cos α·[nV(τ+T)/(W tan α)]^n` (Eq. 24).
  - Experiment parameters: f_av = 0.8, T = 65 ms, τ = 300 ms, V = 0.8 m/s, n = 2, W = 0.8 m. A discontinuity in a wall at α = 20° drove CARMEL into "oscillatory and unstable motion". — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Narrow corridors (Sec. 5.4).** Quote: "A similar yet more severe problem with PFMs occurs when the robot travels in narrow corridors, in which the robot experiences repulsive forces simultaneously from opposite sides."
  - Stability criterion: `L^{n+1} > 2nW^nV(T+τ)f` (Eq. 25). L is the distance from the centre line to each wall.
  - Fig. 6 (f_av = 0.8, n = 2, T = 0.065 s, τ = 0.3 s, V = 0.8 m/s): with L = 1.2 m each side the path stays stable after a wall protrusion. With L = 0.8 m each side, "the sudden change excites the robot into unstable oscillations and eventually a collision." — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Calibration of f.** It was set at the robot's maximum speed: "such that the robot would just barely avoid the obstacle when approaching it head-on, at maximum speed", which gave f_av = 0.8. Platform: CARMEL, Vmax = 0.8 m/s, 120°/s steering, ~125 kg, a ring of 24 sonars of 0.8 m diameter. Obstacles were inserted into the grid by hand for repeatability. — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Conclusion.** Quote: "Two of these drawbacks are related to the possibility of oscillations which become apparent only when the PFM is implemented in a high-speed real-time system … Other researchers work with actual mobile robots, but at slow speeds which conceal the disadvantages of the PFMs." The authors "abandoned potential field methods altogether" for VFH. On traps, heuristic recovery "likely result[s] in a non-optimal path", so they invoke a global path planner when a trap is detected. — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)

#### Borenstein & Koren 1991, "The Vector Field Histogram – Fast Obstacle Avoidance for Mobile Robots", IEEE T-RA 7(3):278–288
Links: [DOI 10.1109/70.88137](https://doi.org/10.1109/70.88137); [copy read](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf).
- **VFF field numbers.** Quote: "Typically, it traversed an obstacle course at an average speed of 0.5m/sec, provided the obstacles were placed at least 1.8m apart (the robot diameter is 0.8m). With less clearance between two obstacles (e.g., a doorway), some problems were encountered. Sometimes, the robot would not pass through a doorway, because the repulsive forces from both sides of the doorway resulted in a force that pushed the robot away." — [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)
- **Grid discretization.** Moving one 10 cm cell changes a 1/d² force by 42% (k/0.6² → k/0.5²). The resultant "may fluctuate by up to 180°" when the target force lies between successive repulsions, which is most likely while travelling alongside an obstacle. The low-pass filter needed to smooth this "introduces a delay that adversely affects the robot's steering response". — [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)
- **Corridors.** Quote: "If, however, the robot strays slightly to either side of the center-line, it experiences a strong virtual repulsive force from the closer wall. This force usually 'pushes' the robot across the center-line, and the process repeats with the other wall. Under certain conditions, this process results in oscillatory and unstable motion." — [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)
- **CARMEL under VFH.**
  - Vmax = 0.78 m/s. Theoretical minimum passage width `W_min = 0.8 + 2×0.27 = 1.34 m`, since sonars cannot see closer than 0.27 m.
  - It ran a course of thin poles spaced about 1.4 m apart, at 0.6–0.7 m/s on average. Speed "was only reduced when an obstacle was approached head-on". — [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)

#### Khatib 1986, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots", IJRR 5(1):90–98
Links: [DOI 10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106); [copy read](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf).
- Quote: "In a cluttered environment, local minima can occur in the resultant potential field. This can lead to a stable positioning of the robot before reaching its goal. While local procedures can be designed to exit from such configurations, limitations for complex tasks will remain. This is because the approach has a local perspective of the robot environment." — [Khatib 1986](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf)

#### Later restatements
- **DWA paper.** It summarises Borenstein & Koren: potential-field methods "often fail to find trajectories between closely spaced obstacles; they also can produce oscillatory behavior in narrow corridors." — [Fox, Burgard & Thrun 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)
- **DWA heading weight.** A low heading weight "may detain the robot from reaching target points behind narrow openings if sensors with low angular resolution are used … because the robot turns away from the opening before its sensors are able to detect it." — [Fox et al. 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)
- **Brock & Khatib 1999.** "While artificial potential field approaches are computationally efficient, the robot can get stuck in local minima before reaching the goal position." The plain DWA is "susceptible to local minima" too. — [Brock & Khatib 1999](https://khatib.stanford.edu/publications/pdfs/Brock_1999_ICRA.pdf)
- **LaValle 2006.** For `φ = φ_a + φ_r`: "it is difficult in general to ensure that the potential function will not contain multiple local minima." — [LaValle, Planning Algorithms ch. 8](http://lavalle.pl/planning/ch8.pdf)
- **Rimon & Koditschek 1992.**
  - Local minima are "the most important" problem, and "this phenomenon persists even with obstacles having simple shape".
  - They also name a second, "seemingly contradictory requirement": obstacles should exert an arbitrarily large repulsive potential, yet actuator bounds limit the allowable repulsion, and "docking, parts mating … all require navigation at or along the boundary." — [Rimon & Koditschek 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf) ([DOI](https://doi.org/10.1109/70.163777))
- **Huber, Billard & Slotine 2022.**
  - The VFH successor family "has the shortcoming of the control velocity being zero in narrow pathways, for example, in front of (narrow) doorways."
  - When normals from several walls are summed in a concave corner, "the algorithm is not aware that the resulting normal is obtained from multiple distinct walls"; the summed direction "points opposite the corner and has a large magnitude." — [Huber et al. 2022, arXiv:2205.04928](https://arxiv.org/abs/2205.04928)

### Inferences
- **Two failure modes.** The classical literature separates two mechanisms that both show up in "narrow gaps":
  - (a) a **static** repulsion-versus-attraction balance at the entrance (K&B §5.2). It contains no speed term.
  - (b) **dynamic** oscillation inside corridors (K&B §5.3–5.4). The critical half-width `L_crit = [2nW^nV(T+τ)f]^{1/(n+1)}` grows with speed (as V^{1/3} for n = 2), with delay (T+τ), with the repulsion ratio f and with width W.

  Mode (b) predicts that **slower** passage is **more** stable. A failure that occurs at slow speed and disappears with momentum cannot be K&B mode (b). It fits mode (a): a static barrier or equilibrium in front of the gap.
- **Check of K&B's numbers (my arithmetic).** With Fig. 6 parameters and W = 0.8 m (from the Fig. 5 parameter box), `2nW^nV(T+τ)f = 4·0.64·0.8·0.365·0.8 ≈ 0.60 m³`, so L_crit ≈ 0.84 m. That fits the observation that L = 0.8 m was unstable and L = 1.2 m stable.
- **Illustrative derivation (mine, not from any source).**
  - Model the two gap edges as point obstacles at ±D/2 and use K&B's 1/d² law. On the approach axis the lateral parts cancel and the longitudinal parts add as `2x/(x² + D²/4)^{3/2}`.
  - This peaks at `x = D/(2√2) ≈ 0.35·D` in front of the gap plane, with magnitude ∝ 1/D².
  - So the "barrier" grows sharply as the gap narrows. It depends on position only, not speed. A quasi-static robot crosses only if the attraction exceeds this peak.
  - With real extended walls, many cells contribute and the barrier is larger.
- **Width ratios in the classical experiments.**
  - VFF ran fine with gaps ≥ 1.8 m for a 0.8 m robot (2.25×) and failed at doorways.
  - VFH's theoretical minimum was 1.34 m (1.68×), and it ran poles 1.4 m apart.
  - Huber et al. passed doorways "marginally wider than the (simulated) robot's diameter" only by abandoning additive repulsion (see Q3).

  A 0.44–0.48 m gap for a G1 whose frontal span, with arms, is likely at or above the gap width is beyond what any classical additive-APF experiment handled. It is a narrow passage in **configuration space**: the robot must change orientation. This is my inference; I did not measure the G1's width.
- **Mapping to HumanoidPF.**
  - The command rule is not an additive 1/dⁿ field, so K&B's exact formulas do not transfer.
  - The structural analogue of §5.2 does exist. As the robot nears a doorway in a wall, the body points facing the wall flanks have normals pointing back at the robot. Their projection corrections all point backwards and **add up** rather than cancel. Only the lateral parts cancel, exactly as in K&B Fig. 4.
  - A forward-facing robot wider than the gap therefore meets a backward "lumped" correction at the entrance. Sideways turning removes the flank-facing points. This is my inference from the repo rule.

### Gaps
- I did not find a classical paper that measured the "no passage" effect against approach speed. K&B's analysis of §5.2 is qualitative, gives no gap-width threshold, and has no speed term.
- Tilove 1990 (GM report GMR-6650; ICRA 1990), cited by K&B for local minima, was not accessible.
- The Siegwart/Nourbakhsh, Choset et al. 2005, Latombe 1991 and Springer Handbook (Minguez/Lamiraux/Laumond) treatments of these failures were not accessible here.

---

## Q2. Is there classical work where repulsion depends on velocity or approach speed, and what does it imply for passing a gap slowly or fast? Does any work report that passage needs speed or momentum?

### Takeaway
Yes, there are velocity-dependent formulations:
- Krogh's "generalized potential" (Krogh 1984; Krogh & Thorpe 1986) is the inverse of the reserve time left to stop before an obstacle.
- The Faverjon–Tournassoud **velocity damper** bounds only the approach rate ḋ.
- Khatib sizes ρ₀ by the operating speed and the deceleration ability.
- DWA's safety margin grows linearly with speed.
- Rimon & Koditschek bound the robot's kinetic energy against the potential height.

All of them make repulsion **vanish or shrink at low approach speed**. Every system I examined slows down at doors: RHINO goes to about 20 cm/s at 80–110 cm doors, and VFH lowers speed for doorways. **I found no classical source in which a robot passes a narrow passage only with momentum, or fails because it is slow.**

The user's observation instead matches what a **position-only** repulsion does to a dynamic system. A slow, quasi-static robot stalls at the equilibrium where repulsion balances attraction. A robot with kinetic energy can coast over the barrier, which is the reverse reading of Rimon & Koditschek's energy bound.

### Cited Findings
- **Khatib 1986: ρ₀ chosen by speed.** FIRAS potential: `U_O(x) = ½η(1/ρ − 1/ρ₀)²` if ρ ≤ ρ₀, else 0. Force on a point subjected to the potential: `F = η(1/ρ − 1/ρ₀)(1/ρ²)∂ρ/∂x`. Quote: "The selection of the distance ρ₀ will depend on the end effector operating speed V_max and on its deceleration ability." — [Khatib 1986](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf)
- **Khatib 1986: velocity limit.**
  - Attraction: `F*_m = −k_p(x − x_d) − k_v ẋ`, rewritten as `ẋ_d = (k_p/k_v)(x_d − x)`, `F*_m = −k_v(ẋ − ν ẋ_d)` with `ν = min(1, V_max/√(ẋ_dᵀẋ_d))`.
  - Effect: the end effector moves "at that speed in a straight line, except during the acceleration and deceleration segments or when it is inside the repulsive potential field regions of influence." — [Khatib 1986](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf)
- **Krogh & Thorpe 1986, generalized potential.**
  - The system is a point with acceleration control: `ẍ = u`, `‖u‖ ≤ α`, `‖v‖ ≤ κ`.
  - The generalized potential depends on position and velocity: `P(x,v) = P_g(x,v) + P_o(x,v)`.
  - Quote: P_o "is defined as the inverse of the reserve avoidance time, τ_M − τ_m, where the minimum avoidance time τ_m is the minimum time in which the velocity toward the obstacle can be brought to zero using maximum deceleration, and the maximum avoidance time τ_M is the maximum time in which the velocity toward the obstacle can be brought to zero under constant deceleration without hitting the obstacle. This gives `P_o(x,v) = v_o(v_o² − 2αx_o)^{-1}`, where −v_o is the velocity toward the obstacle and x_o is the distance to the obstacle barrier … P_o(x,v) grows to infinity as the capability to avoid crossing the obstacle barrier goes to zero."
  - P_o is computed "only for visible obstacles in the direction of the current system velocity v".
  - Krogh & Thorpe state that potentials "which are both position and velocity dependent" eliminate "the possibility of the stalling at local minima in the potential field".
  - Critical points from the Path Relaxation global planner serve as subgoals. — [Krogh & Thorpe, ICRA 1986](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/krogh_bruce_1986_1/krogh_bruce_1986_1.pdf) (original: B. H. Krogh, "A Generalized Potential Field Approach to Obstacle Avoidance Control", Robotics International Robotics Research Conf., Bethlehem PA, Aug. 1984; not accessed)
- **Velocity damper (Faverjon & Tournassoud 1987), as restated by Kanehiro et al. 2008.**
  - When the closest distance is below an influence distance d_i: `ḋ ≥ −ξ (d − d_s)/(d_i − d_s)`, with security distance d_s < d_i and "ξ … a positive coefficient for adjusting convergence speed".
  - It is imposed as a linear constraint `(q̇ | J(q,p₁)ᵀn) ≥ −ξ(d − d_s)/(d_i − d_s)` in `min_q̇ ‖J_τ q̇ − τ̇‖²`. This guarantees `d(t) ≥ d_s + (d(0) − d_s)e^{−ξt/(d_i − d_s)} > d_s`.
  - Kanehiro et al. note that ḋ must already satisfy the inequality on entering the zone, otherwise it is constrained discontinuously, "Therefore, the value of ξ must be tuned according to applications."
  - Parameters used: d_i = 0.4 m, d_s = 0.2 m, ξ = 0.5 m/s in a planar example; d_i = 0.05 m, d_s = 0.03 m, ξ = 0.5 m/s for HRP-2 whole-body motion. — [Kanehiro, Lamiraux, Kanoun, Yoshida & Laumond, RSS 2008](http://www.roboticsproceedings.org/rss04/p20.pdf)
- **DWA (Fox, Burgard & Thrun 1997).**
  - Admissible velocities are those that can stop before the nearest obstacle on the arc: `V_a = {(v,ω) | v ≤ √(2·dist(v,ω)·v̇_b) ∧ ω ≤ √(2·dist(v,ω)·ω̇_b)}`.
  - Objective: `G(v,ω) = σ(α·heading + β·dist + γ·vel)`. The example uses weights 2.0 / 0.2 / 0.2; the experiments use 0.8 / 0.1 / 0.1. — [Fox et al. 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)
  - Speed-dependent side clearance: "a safety margin around the robot, which grows linearly with the robot's translational velocity. Thus the robot will travel with high speed through corridors and will decelerate when driving through narrow doors."
  - RHINO numbers: maximum about 95 cm/s; "when moving through doors, RHINO typically decelerates to approximately 20 cm per second in the vicinity of the door"; doors were about 80–110 cm. Through a passage under 80 cm wide it slowed to below 20 cm/s yet averaged 65 cm/s.
  - Door example: at 75 cm/s "the robot moves too fast to perform the sharp turn to the right necessary to drive through the open door"; at 40 cm/s the dynamic window "includes curvatures leading through the door". Accelerations: 50 cm/s², 60°/s². — [Fox et al. 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)
- **VFH speed law and doorway policy.**
  - `V′ = V_max(1 − h″_c/h_m)` with `h″_c = min(h′_c, h_m)`, where h′_c is the smoothed obstacle density in the direction of travel.
  - Then `V = V′(1 − Ω/Ω_max) + V_min`, with Ω_max = 120°/s and V_min = 4 cm/s.
  - For doorways: "If the global plan calls for passing through a narrow passage (e.g., a doorway), the threshold is temporarily raised while the travel speed is lowered." — [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)
- **K&B 1991.** The oscillation conditions (Eqs. 20, 21, 24, 25) all tighten as V(τ+T) grows. f was calibrated so the robot "would just barely avoid the obstacle when approaching it head-on, at maximum speed." — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Rimon & Koditschek 1992: energy and speed bound for second-order (mechanical) potential control.**
  - Controller: `τ = −∇V(p) + d(p,ṗ)`, where d is dissipative.
  - Quote: "trajectories of a dissipative system with admissible potential energy that start in F with suitable initial velocity remain away from the obstacles … a potential-field controller obtained from an admissible function is bounded and steers the robot away from the obstacles if some initial speed limit is imposed on the robot. In fact, this speed limit must be provided as a function of initial position: large but bounded speeds are tolerable from initial configurations away from the obstacles; initial configurations exactly on the obstacles' boundary must have zero velocity."
  - Construction: with U admissible and maximal height 1, choose `c ≥ K(q₀, q̇₀)/(1 − U(q₀))` (K = kinetic energy) and use V = cU. — [Rimon & Koditschek 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf)
- **Huber et al. 2022 ("tail effect").** Modulation-based avoidance "is active even when the nominal dynamics v^N are already moving away from an obstacle … This tail effect can be reduced by modifying the wake eigenvalues." — [Huber et al. 2022](https://arxiv.org/abs/2205.04928)

### Inferences
- **What velocity-dependent repulsion means for slow and parallel passage.**
  - Krogh's P_o is zero for v_o ≥ 0 (moving parallel or away), and tends to 0 as the approach speed → 0.
  - The velocity damper constrains only ḋ, so tangential motion is unconstrained.
  - Neither can create a static stall equilibrium; Krogh & Thorpe make exactly this claim about stalling. Both leave a slow sideways shuffle along the gap axis completely unpenalised.
  - Position-only repulsion (FIRAS, VFF, or an SDF penalty) acts with equal strength on a robot at rest. It **can** hold a slow robot in front of a gap.
- **Why momentum can help (energy view; my reading of Rimon & Koditschek's bound, which they state only for safety).**
  - In a dissipative mechanical system, total energy K + V does not increase.
  - A quasi-static, velocity-commanded robot cannot climb a potential rise. A robot with kinetic energy K can coast through a region where the effective potential rises by up to K.
  - So "passes only with momentum" is the signature of a **potential barrier or saddle on the approach path**, the K&B §5.2 balance, combined with inertia.
  - The same energy argument is why classical work bounds speed for safety: kinetic energy is also what carries a robot into obstacles.
- **Lag carry-through.**
  - In K&B's model every stability condition contains V(τ+T): the distance travelled during the response lag.
  - A fast robot moves further into the gap before the repulsion reacts. With 0.66–0.97 m/s bursts and policy/gait latencies of roughly 0.1–0.3 s (my rough guess, not measured), that is several to tens of centimetres. That is likely larger than the per-side clearance a sideways G1 has in a 0.44–0.48 m gap (I did not measure the body depth).
- **An RL-specific mechanism with no classical counterpart (inference from the repo reward definitions).**
  - Classical APF controllers are memoryless, so the time spent near a wall costs nothing.
  - In this project the SDF penalty and the proximity-gated terms are paid **per control step**. The penalty for crossing a gap therefore scales as ∫c(x)dt = ∫c(x)/v ds, which falls as 1/v. Passing a gap at 0.9 m/s instead of 0.1 m/s cuts the accumulated proximity penalty about ninefold.
  - The FM2/DWA/VFH convention, slow near obstacles, is optimal when cost is **time** under a clearance-limited speed. A per-time proximity penalty with no clearance-limited speed makes the **burst** optimal. This could explain why the policy ignores the commanded 0.1–0.525 m/s inside gaps, though other reward terms such as the crossing bonus may also contribute.
- **Mapping the command rule onto the velocity damper.**
  - The rule's half-space test `b̂·v ≥ L = b̂·g` is a velocity damper whose bound is set by the guidance's normal component instead of `−ξ(d − d_s)/(d_i − d_s)`.
  - Inside the 20 cm band, where g is projected tangentially, L ≈ 0. That is a damper with ξ = 0: **no approach at all**, rather than an approach rate shrinking toward d_s.
  - `safe_velocity_target` likewise removes 100% of the inward component at ≤ 0.25 m.
  - A classical damper with ξ > 0 would let a slow robot keep creeping toward a flank down to d_s. The hard "zero approach" band does not, so slow, careful entry is not representable as a target.
- **Speed-scaled influence radius.** Khatib's ρ₀(V_max, deceleration) and DWA's speed-proportional margin imply that the influence region should **shrink** at low speed. HumanoidPF's projection band (20 cm), the reward window (sdf < 0.3 m) and the target ramp (0.25–0.60 m) do not depend on speed. A slow robot is therefore "repelled" from as far away as a fast one, although it needs much less stopping distance.

### Gaps
- **Ge & Cui 2002**, "Dynamic Motion Planning for Mobile Robots Using Potential Field Method", *Autonomous Robots* 13:207–222 ([DOI 10.1023/A:1020564024509](https://doi.org/10.1023/A:1020564024509); title and venue confirmed via the OpenAlex and Semantic Scholar records). This is the best-known velocity-dependent APF: a repulsive potential depending on relative velocity, active only while approaching, with a stopping-distance term. It could not be opened (Springer authentication; NUS repository HTTP 403). Its equations should be checked before being cited.
- Krogh's 1984 original paper and Krogh 1985, "Guaranteed steering control", were not accessed. The stalling claim above is quoted from Krogh & Thorpe 1986, which cites them.
- No source found, in either direction, on legged or humanoid robots crossing gaps with momentum. Robot-learning literature (e.g., ABS, arXiv:2401.17583, already in the shared papers folder) is outside this classical scope.

---

## Q3. What fixes were proposed, and how do they affect slow traversal of narrow gaps?

### Takeaway
The classical fixes fall into six families:
1. **Navigation functions** (Rimon–Koditschek, harmonic, NF1/NF2, FMM/FM2), which remove local minima for a point.
2. **Representation changes** that choose a direction instead of summing forces (VFH/VFH+), centring the robot in narrow openings.
3. **Constraint and velocity-space methods** (velocity damper, DWA, modulation of a nominal velocity), which bound or rotate the approach component instead of adding repulsion.
4. **Hybrids with global planners** (subgoals, NF1 inside DWA, trap detection and replanning).
5. **Parameter scheduling**: ρ₀ set by speed, thresholds and speed lowered at doorways, margins set by speed.
6. **Escape heuristics**: random motions, rotate-away modes.

Every family that handles narrow openings explicitly does it by **centring and slowing down**, not by adding forward drive. The only method that proves no stalling outside a margin from sampled data (Huber et al.) does it by **capping the summed obstacle influence** and **amplifying tangential motion** while damping the normal component.

### Cited Findings

#### Navigation functions (Rimon & Koditschek 1992)
- **Definition.** A navigation function is "1) smooth on F (at least a C⁽²⁾ function); 2) polar at q_d, i.e., has a unique minimum at q_d on the path-connected component of F containing q_d; 3) admissible on F, i.e., uniformly maximal on the boundary of F; 4) a Morse function." — [Rimon & Koditschek 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf)
- **Sphere-world formula.** `φ_κ(q) = ‖q − q_d‖² / [‖q − q_d‖^{2κ} + β(q)]^{1/κ}`, where β is the product of the obstacle functions. It "is a navigation function, so long as the parameter κ exceeds a certain function of the geometric data". "All the undesired local minima disappear as the parameter κ increases"; the result has "a saddle point near each internal obstacle". Saddles cannot be avoided: a sphere world's field "must have at least as many saddles as there are internal obstacles". — [Rimon & Koditschek 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf)
- **Lecture-slide gloss.** "By increasing κ, we can make the gradient at any direction point toward the goal"; "One problem: the value of O(q) may be very large", fixed with a switch σ_λ(x) = x/(λ + x) and sharpening η_κ(x) = x^{1/κ}. — [Choset ch. 4 slides](https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf)
- **Practical tuning.** In footnote 2, the authors say they found "little sensitivity" to the geometric bounds in practice, and in "all of our simulation studies, the parameters have been simply set 'very high'." — [Rimon & Koditschek 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf)
- **Main practical obstacle.** Navigation functions need a "topological model" of the free configuration space, which is the "chief practical obstruction". LaValle adds that C-space obstacles are represented only implicitly, "which makes it difficult to correctly apply the method to complicated high-dimensional problems." — [R&K 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf); [LaValle ch. 8](http://lavalle.pl/planning/ch8.pdf)

#### Harmonic potentials (Connolly, Burns & Weiss 1990; Connolly & Grupen; Kim & Khosla 1992)
- **Dirichlet boundaries.** Quote: "obstacles are raised to a constant high potential, while goal regions are kept at a low potential … the gradient of φ is aligned with the surface normals of the obstacles … it will tend to repel the robot away from obstacles." — [Connolly & Grupen, UMass report 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/connolly_harmonic_functions.pdf)
- **Neumann boundaries.** n·∇φ = 0 makes "∇φ … tangential to obstacle surfaces. Here, the paths generated will graze any obstacles they encounter". It also "can result in a tendency for a robot to stay close to obstacle surfaces." — [Connolly & Grupen](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/connolly_harmonic_functions.pdf)
- **Flatness of Dirichlet solutions.** Quote: "the Dirichlet solution tends to have precision problems. Flat regions can develop resulting in very small (but of necessity nonzero) gradients". — [Connolly & Grupen](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/connolly_harmonic_functions.pdf)
- **Blending.**
  - `φ = k·φ_D + (1 − k)·φ_N`, with k ∈ [0,1], mixes the "repel" and "graze" behaviours. The result "is harmonic, has no local minima and, provided that k ∈ [0,1], guarantees collision-free paths", because the boundary gradient keeps an outward-normal component. Keeping k < 1 "avoids shallow gradients".
  - Their 2-link arm example weights the Dirichlet part at 0.98.
  - Where repulsion helps: repulsion "is desirable in fragile environments", while grazing suits exploration where contact is acceptable. — [Connolly & Grupen](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/connolly_harmonic_functions.pdf); [Connolly & Grupen, "On the Applications of Harmonic Functions to Robotics"](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/connolly_harmonic_application.pdf)
- **LaValle's summary.** Neumann boundary conditions force "the velocity vectors to be tangent to the obstacle boundary". Combined boundary conditions give "a harmonic navigation function … that avoids obstacles by moving parallel to their boundaries and eventually landing in the goal". The costs are an explicit boundary and expensive numerics. — [LaValle ch. 8](http://lavalle.pl/planning/ch8.pdf)

#### Grid navigation functions NF1/NF2 and global DWA
- **NF2.** "One problem that typically arises in mobile robotics is that optimal motion plans bring robots too close to obstacles … shortest Euclidean paths … must be allowed to touch obstacle vertices." NF2 sends the robot to the maximum-clearance skeleton, along it, then to the goal. Limitation: it "does not … maximize the nearest approach to an obstacle, if there are multiple routes." — [LaValle ch. 8 §8.2](http://lavalle.pl/planning/ch8.pdf)
- **Global DWA (Brock & Khatib 1999).**
  - It combines DWA with NF1, which "makes the global dynamic window approach immune to local minima, since NF1 is a local minima-free potential".
  - But "NF1 has the disadvantage of producing trajectories that graze obstacles", and because NF1's gradient comes in multiples of 45°, it gives "unnatural behavior along passages that are not grid-aligned."
  - NF1 is recomputed in a goal-aligned window at each command. — [Brock & Khatib, ICRA 1999](https://khatib.stanford.edu/publications/pdfs/Brock_1999_ICRA.pdf)

#### Fast marching and Fast Marching Square (FM2)
- **FMM has no spurious minima.** Quote: "no local minima can appear. However, saddle points are possible but they are not a problem as gradient descent" handles them. — [J. V. Gómez, PhD thesis (UC3M), via the FM2 library repo](https://github.com/jvgomez/fastmarching)
- **Plain FMM drawbacks.** Quote: "1) robot dimensions are not explicitly taken into account, and 2) kinematic constraints are not taken into account. 1) is partially solved in practice by obstacle dilation … commonly by using the maximum radius of the robot projection on the ground or by including a third dimension to the configuration space in order to represent the yaw (heading angle) of the robot." — [Gómez thesis](https://github.com/jvgomez/fastmarching)
- **FM2.**
  - First wave: "all the obstacles [are] wave sources". The resulting arrival-time map is a slowness or velocity map in which speed grows with distance from obstacles. "A robot whose speed at each point is given by the T_i value will never collide, as T_i → 0 when approaching the obstacles."
  - Second wave: from the goal, through that velocity map. The path is "the fastest path (in time) assuming the robot moves at the maximum allowed speed at every point". Plain FM2 paths keep "as far as possible from obstacles", similar to Voronoi paths.
  - Saturated FM2: the map is scaled to a maximum speed reached at a "safe distance", so the speed rises from 0 at obstacles to the maximum at that distance. — [Gómez thesis ch. 5](https://github.com/jvgomez/fastmarching)
  - FM2 is attributed to Garrido, Moreno, Abderrahim & Blanco 2009, "FM2: A Real-time Sensor-based Feedback Controller for Mobile Robots". The thesis lists the venue as IEEE Trans. Automatic Control 24(1), which looks inconsistent; I believe the paper is in the *Int. J. Robotics and Automation*, but that is unverified.
- **FM2 caveat.** The library README warns: "Gradient Descent for FM2* could fail if very narrow passages are in the way of the path." — [jvgomez/fastmarching README](https://github.com/jvgomez/fastmarching)
- **FMM in configuration space (Sethian; Kimmel & Sethian 1996).** For a ladder or piano, "you are no longer a point, but now have to negotiate the ladder through the narrow paths". The eikonal equation `|∇T| = f(x, y, θ)` is solved in (x, y, θ), and "the constraints of the walls are manifested by modifying configuration space as a function of the angle of the stick". — [Sethian, "Robotic Navigation with Constraints"](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated2/sethian_fast_march_nav.pdf)

#### Direction selection instead of force summation (VFH / VFH+)
- **VFH valley selection.**
  - Polar histogram sectors of α = 5°. A valley is "wide" if more than s_max = 18 consecutive sectors are below the threshold. Steering is θ = (k_n + k_f)/2, which yields a stable stand-off d_s along walls.
  - Narrow valleys: "the direction of travel is again chosen as θ = (k_n + k_f)/2 and the robot maintains a course centered between obstacles." The ability to pass doorways "results from the ability to identify a narrow valley and to choose a centered path through that valley".
  - Threshold trade-off: too high, and the robot may "approach the obstacle too closely (especially when traveling at high speed)"; too low, and "the robot will not pass through narrow passages". — [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)
- **VFH+ changes.**
  - Obstacle cells are enlarged by `r_{r+s} = r_r + d_s` (robot radius plus minimum distance), so that "the robot can be treated as a point-like vehicle".
  - Threshold hysteresis (τ_low, τ_high) cures the case where a narrow opening flickers open and closed, causing "indecisive behavior, during which the mobile robot can get very close to an obstacle".
  - Trajectories are masked by turning radius.
  - Cost: `g(c) = μ1·Δ(c,k_t) + μ2·Δ(c, θ_i/α) + μ3·Δ(c, k_{n,i−1})`, with μ1 > μ2 + μ3 for goal-orientedness and μ = 5, 2, 2 recommended. A narrow opening (≤ s_max = 16 sectors) gives a single candidate, its centre `c_n = (k_r + k_l)/2`.
  - Results: "safe travel at speeds up to 1 m/s without noticeable oscillations" on the GuideCane. — [Ulrich & Borenstein, ICRA 1998](http://www.cs.cmu.edu/~iwan/papers/vfh+.pdf) ([DOI](https://doi.org/10.1109/ROBOT.1998.677362))
- **VFH limitation.** The VFH family's control velocity drops to zero "in front of (narrow) doorways." — [Huber et al. 2022](https://arxiv.org/abs/2205.04928)

#### Velocity-space and constraint methods
- **DWA.** It bounds velocity by stopping distance and lowers speed through a speed-proportional margin (numbers in Q2). Local minima are handled with a "rotate away mode": "if no admissible trajectory allows the robot to translate … the robot rotates away from the obstacle until it is able to translate again." — [Fox et al. 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)
- **Velocity damper in a QP (Faverjon–Tournassoud, Kanehiro et al.).** Each body–obstacle closest-point pair is a hard linear constraint on q̇. The QP otherwise tracks the task velocity. In their example, an "L"-shaped object passes through a torus (a hole) without violating d_s. — [Kanehiro et al. 2008](http://www.roboticsproceedings.org/rss04/p20.pdf)
- **Modulation of a nominal velocity (Huber, Billard & Slotine 2022, arXiv:2205.04928).**
  - Velocity is `ξ̇ = M(ξ)·v^N` with `M = E·D·Eᵀ`. The eigenvalue along the reference direction (toward the obstacle) is λ_r ≤ 1, and can reflect the velocity when approaching. The tangent eigenvalue is λ_e ∈ [1, 2], so tangential speed is **amplified** near obstacles.
  - For sampled surface points, the summed weight is capped: `w_sum = min(Σ_i ŵ_i, w_norm)`. With `w_norm = D_gap·δ/(2D_scal)`, "the modulated DS does not vanish further than D_gap away from any obstacle".
  - Theorem 2: trajectories stay in free space "and will not stop outside of the gap distance".
  - Results: the robot "passes narrow doorways successfully … Even though the gap is marginally wider than the (simulated) robot's diameter, no local minimum is observed in front of the doorway." The agent "only stopped when directly driving towards obstacles". — [Huber et al. 2022](https://arxiv.org/abs/2205.04928)
  - Caveat: in a 4-obstacle convergence test, sampled data alone reached the attractor in **45%** of 100 runs, against **83%** when analytic obstacle descriptions were added. Average time was 24.6 s versus 29.7 s. — [Huber et al. 2022](https://arxiv.org/abs/2205.04928)

#### Hybrids with global planners and escape strategies
- **Khatib 1986 two-level scheme.** "At high level control, generating a global strategy for the manipulator's path in terms of intermediate goals (rather than finding an accurate collision-free path)"; the low level executes with real-time sensing. — [Khatib 1986](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf)
- **Krogh & Thorpe 1986.** Path Relaxation computes "critical points" along a global path, and the generalized potential field drives the vehicle between them as subgoals. — [Krogh & Thorpe 1986](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/krogh_bruce_1986_1/krogh_bruce_1986_1.pdf)
- **Trap handling in VFF and VFH.** K&B use a global path planner on trap detection because heuristic recovery gives non-optimal paths. VFH flags a trap when the search diverts by 180° (36 sectors), then slows (possibly to a halt) and uses global replanning via-points. — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf); [VFH 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf)
- **Barraquand, Langlois & Latombe 1992.** Local minima of a numerical C-potential are escaped by best-first search, "random motion" (Brownian motions), valley-guided and constrained-motion techniques. In best-first search, "When a local minimum is reached, the algorithm naturally fills up the local minimum well until a saddle point is reached." — [Barraquand et al. 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/b/barraquand_langlois_latombe_potential.pdf) ([DOI](https://doi.org/10.1109/21.148426))

#### Tuning of influence radius, gains and thresholds
- Khatib sets ρ₀ by V_max and deceleration [(Khatib 1986)](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf).
- K&B name f as the only stability knob, calibrated at maximum speed [(K&B 1991)](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf).
- VFH uses s_max to set the stand-off distance and an adaptive threshold plus lower speed at doorways [(VFH 1991)](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf).
- DWA's heading weight trades free circumvention against not reaching "target points behind narrow openings" [(Fox et al. 1997)](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf).
- The velocity damper uses d_i, d_s and ξ, with "ξ must be tuned according to applications" [(Kanehiro et al. 2008)](http://www.roboticsproceedings.org/rss04/p20.pdf).

### Inferences
- **Where HumanoidPF sits.** Its guidance (−∇T from FMM, projected tangentially within 20 cm) belongs to the **NF1/FMM and Neumann-"graze"** family.
  - For a point it is free of local minima, and like NF1 it grazes obstacles; hence the tangential projection.
  - The "repulsion" in the system does not come from the guidance. It comes from the **normal channel plus the projection rule plus the SDF penalty**, which is functionally Dirichlet-like.
  - Connolly & Grupen show the two can be fused into **one** minimum-free field, `kD + (1 − k)N`. HumanoidPF instead leaves the arbitration between "graze" and "repel" to the policy and the rule, and that is exactly where an equilibrium in front of the gap can form.
- **Fixes most likely to cure slow-passage failure (all inference).**
  1. **Encode the body in the field.** Use FMM over (x, y, yaw) (Sethian/Kimmel), or per-point FMM with obstacles dilated by each link's radius (VFH+ `r_r + d_s`; Gómez's dilation). Then the guidance itself says "turn sideways, then go". Today the policy must discover the rotation while the point-guidance points straight through a gap the forward-facing body cannot fit.
  2. **Replace position-only repulsion with velocity-damper terms** in both the command rule and the reward. Constrain or penalize the approach rate ḋ relative to `−ξ(d − d_s)/(d_i − d_s)`, not proximity per se. Slow, parallel progress then costs nothing, and the hard ξ = 0 band becomes a graded allowance.
  3. **Speed-shaped targets as in FM2, saturated FM2, DWA or VFH.** Prescribe lower speed where clearance is small, and pay progress per distance (or reward time-to-goal under a clearance-limited speed) rather than charging proximity per second. This removes the incentive, under a per-step penalty, to minimise time spent in the gap (Q2).
  4. **Cap the summed obstacle influence** (Huber's `w_norm`) or use a min-plus-ε·max arbitration (Barraquand; see Q4). Several body points facing the flanks then cannot jointly cancel forward progress outside a defined margin.
  5. **Commitment and hysteresis** (VFH+ τ_low/τ_high; μ2 and μ3 terms) to stop flickering between "fit/doesn't fit" at the entrance.
- **Fixes that do not address the issue.**
  - Raising κ, or switching to another point-navigation function, does not change the fact that the body's orientation is the binding constraint.
  - Escape heuristics such as random motion or rotate-away suit traps, not a gap that is passable only after reorientation. Rotate-away is, in effect, what the policy learned by turning sideways.
- **The 20 cm band against the gap width.** In a 0.44–0.48 m gap the wall distance is 0.22–0.24 m at the centre. So nearly the whole gap lies within the 20 cm tangential band, and the whole gap lies within `safe_velocity_target`'s full-weight zone (≤ 0.25 m). Inside the gap the field reduces to "move parallel to the walls or not at all", much as in VFH's narrow valley, but without VFH's explicit "is this valley wide enough for me" test. This is a geometric observation from the stated parameters.

### Gaps
- Not accessed, so no exact conditions or equations are given for them:
  - **Vortex and rotational fields**: De Luca & Oriolo 1994. Brock & Khatib mention "circulatory fields [15]", most likely Singh, Stephanou & Wen 1996.
  - **Virtual-obstacle local-minimum escape**: e.g., Chengqing et al., ICRA 2000.
  - **Superquadric potentials** (Volpe & Khosla 1990) and **panel-method harmonic potentials** (Kim & Khosla, IEEE T-RA 8(3), 1992, [DOI](https://doi.org/10.1109/70.143352)).
  - **Nearness Diagram**: Minguez & Montano, T-RA 2004, [DOI](https://doi.org/10.1109/TRA.2003.820849), which explicitly targets "troublesome" narrow-gap scenarios.
- **Ge & Cui 2000** ("New potential functions for mobile robot path planning", IEEE T-RA 16(5):615–620, [DOI](https://doi.org/10.1109/70.880813)) was not accessible. It defines the "goals non-reachable with obstacles nearby" (GNRON) problem and fixes it by scaling repulsion with the distance to the goal. That is the closest classical analogue of "the goal lies just behind an obstacle", and its exact form and conditions should be verified.
- No source quantifies the rule "κ must grow as obstacles approach each other". Rimon & Koditschek say only that κ must exceed "a certain function of the geometric data (which is given in closed form in [25])"; their ref. 25 was not read.
- No classical study evaluates any of these fixes on a robot that must **reorient** to fit a gap (a legged, humanoid or non-circular body). The VFH+ dilation trick explicitly assumes a robot "whose shape can be approximated by a" circle.

---

## Q4. For an extended body sampled at several points, what does the classical literature say about passing narrow passages and about repulsion acting on several control points at once?

### Takeaway
- **Points subjected to the potential.** Classical whole-body APF, starting with Khatib's "points subjected to the potential" (PSPs), sums each point's repulsion through its Jacobian transpose. It protects links by tracking each link's **closest point** to the obstacle rather than fixed points.
- **Competing control points.** Barraquand, Langlois & Latombe (1992) showed that combining per-point fields that are each free of local minima **does not** give a whole-body field free of them. The control points "are competing among themselves", and simple summation makes conflicts "frequent, producing numerous local minima".
- **Closest-point switching.** Kanehiro et al. (2008) showed that switching the active closest point between body features causes oscillation and even collision.

HumanoidPF samples fixed body points, averages their projections, and its guidance comes from a point (workspace) FMM. That is exactly the classical setting in which a wide body meets a whole-body local minimum in front of a gap, escapable only by reorientation. This supports the user's hypothesis that sparse body sampling and no explicit obstacle structure contribute.

### Cited Findings
- **Khatib 1986, multi-point control.**
  - "Any point of the robot can be subjected to the artificial potential field." Joint forces for a PSP are `Γ = J_psp(q)ᵀ Λ(x) F*_psp`. Several obstacles superpose (`F*_psp = Σ_i F*_(O_i,psp)`), and several points add: `Γ_obstacles = Σ_j J_psp_jᵀ(q) Λ(x) F*_psp_j`.
  - Quote: "Specifying an adequate number of PSPs enables the protection of all of the manipulator's parts."
  - Links: "Link collision avoidance is achieved by continuously controlling the link's closest point to the obstacle. At most, n PSPs then have to be considered." Obstacles are modelled as analytic primitives (n-ellipsoids and n-cylinders that approximate boxes, cylinders and cones).
  - The earlier f(x)-based potential "is difficult to use for asymmetric obstacles where the separation between an obstacle's surface and equipotential surfaces can vary widely". — [Khatib 1986](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf)
- **Choset ch. 4 slides.** "Pick several points on the manipulator; Compute attractive and repulsive potentials for each; Transform these into the configuration space and add" (via Jᵀ). "Pick enough points to 'pin down' robot (2 in plane)" and "More points please". Using the revolute joints as control points "could lead to problems in some cases". — [Choset slides](https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf)
- **Barraquand, Langlois & Latombe 1992, control points and arbitration.**
  - Each control point p_i gets a workspace potential ("W-potential") V_{p_i}. The C-potential is `U(q) = G(V_{p_1}(X(p_1,q)), …, V_{p_s}(X(p_s,q)))`, where G is the "arbitration function".
  - Quote: "the resulting C-potential may still have spurious local minima. One reason for that is that the information contained in the W-potentials does not completely characterize the connectivity of the free space; indeed, the W-potentials are computed for a finite (usually small) number of control points. The other reason is that the combination of several local-minima-free W-potentials does not guarantee that the C-potential is without local minima. Indeed, the effect of the C-potential is to concurrently attract the various control points, which are related by fixed or variable kinematic constraints, toward their respective goal positions. Thus, the control points are competing among themselves to attain these positions, and this competition may create undesired minima." — [Barraquand et al. 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/b/barraquand_langlois_latombe_potential.pdf)
  - Arbitration options:
    - `G = Σ y_i`: "does not favor one control point over the others. However, precisely for that reason, conflicts among the points tend to be frequent, producing numerous local minima."
    - `G = min y_i`: favours the best-placed point, but the potential goes flat once one point arrives.
    - `G = min y_i + ε·max y_i`, with ε = 0.1 in their experiments.
    - `G = max y_i` "tends to increase the number of competitions … However, it can be a good choice for robots with many DOF's"; it "increases the number of local minima, but reduces their volumes."
  - W-potential variants: the simple one (L1 wavefront from the goal, i.e. NF1) "induces paths that typically graze obstacles". The "improved W-potential" follows the workspace skeleton (NF2-like) "in order to both reduce the risk of creating local minima in the C-potential function and enlarge the maneuvering space of the robot". — [Barraquand et al. 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/b/barraquand_langlois_latombe_potential.pdf)
- **Kanehiro et al. 2008, closest-point switching.**
  - Example: a 0.2 × 0.8 m rectangle is commanded downward against the floor under a velocity damper (d_i = 0.4 m, d_s = 0.2 m, ξ = 0.5 m/s). It rotates because one corner is constrained. Then "the closest points start to oscillate between C1 and C2. The object rotates clockwise … when C1 is constrained and does counterclockwise when C2 is constrained. As a result, the minimum distance becomes smaller than d_s and the rectangle eventually collides with the floor".
  - Fix: track several continuously moving point pairs per facet and edge, so constraints appear and vanish continuously.
  - HRP-2 "pick up an object under the table" used d_i = 0.05 m, d_s = 0.03 m, ξ = 0.5 m/s, at about 100 ms per QP step. They suggest a "thin influence zone or simplified shapes" to cut the pair count. — [Kanehiro et al., RSS 2008](http://www.roboticsproceedings.org/rss04/p20.pdf)
- **Body size in FMM and VFH+.**
  - FMM: robot dimension is handled by obstacle dilation by the maximum projected radius, or by adding yaw as a third configuration dimension [(Gómez thesis)](https://github.com/jvgomez/fastmarching). Kimmel & Sethian solve `|∇T| = f(x,y,θ)` for an extended stick [(Sethian)](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated2/sethian_fast_march_nav.pdf).
  - VFH+ enlarges obstacles by `r_r + d_s` so the robot is "point-like" [(Ulrich & Borenstein 1998)](http://www.cs.cmu.edu/~iwan/papers/vfh+.pdf).
- **K&B 1991.** Robot width enters the VFF force law (∝ W^n) and every stability bound (Eqs. 21, 24, 25). — [K&B 1991](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf)
- **Huber et al. 2022, sampling and structure.**
  - Discrete sampling "might miss the closest point of an obstacle, i.e., an edge". They add a margin `R_marg/R_robo = sin(δ/2)/tan(φ_min/2) + (1 − cos(δ/2))`, with φ_min = 45°.
  - Sampled-only perception roughly halved convergence: 45% against 83% with analytic obstacle shapes.
  - In concave corners the summed normal from multiple walls is large and points out of the corner. — [Huber et al. 2022](https://arxiv.org/abs/2205.04928)
- **Rimon & Koditschek 1992.** Navigation functions are defined on the robot's configuration space (an n-DOF robot in a generalized sphere world). Correctness rests on knowing that space's geometry and topology. — [R&K 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf)

### Inferences
- **HumanoidPF is Barraquand's setting.** Every body point follows its own workspace guidance field (−∇T), which is free of local minima for that point. The policy has to turn these into a whole-body motion.
  - In front of a gap narrower than the forward-facing body, the per-point guidances "compete". Points facing the flanks must go sideways and around, while the root goes straight.
  - The whole-body "C-potential" implied by the per-point fields can have a minimum in front of the gap. In Barraquand's terms, that minimum is left only through a change of the kinematic configuration: turning sideways.
  - This is a structural argument; I have not verified it in simulation. It supports the hypothesis that the robot "does not know the obstacle's structure". The fields encode the full geometry, but only as seen from 13 points, and the guidance is point-based, not C-space.
- **The averaging rule is non-classical.** `compute_cmd_from_rtf` averages per-point projection corrections, `v + mean_i(delta_i)`.
  - Khatib and Choset **sum** point contributions through Jᵀ. Barraquand **arbitrates** with min, max or sum. Kanehiro imposes **each** point's constraint as a hard QP inequality.
  - A mean over M = 5 points both **under-protects** and **over-couples**:
    - A single point moving into a wall has only 1/5 of its inward component removed.
    - Several points facing the same flank add up. Two hands facing the flanks of a doorway remove about 2/5 of the forward command plus their L terms, and enough of them triggers the stop cutout.
  - This is the multi-point form of K&B §5.2. It never depends on speed, because the projection uses the guidance magnitude (0.7·g_root), not the operator speed.
- **Fixed sample points miss the true closest point.** Khatib controls each link's closest point, and Kanehiro tracks closest-point pairs. HumanoidPF reads the fields at fixed points (head, hands, feet, etc.), so a shoulder, elbow or forearm between samples can be nearer the gap edge than any sample.
  - Huber's margin formula is the classical way to bound that miss, by inflating by the worst-case gap between samples.
  - The repo already subtracts `hand_radii` from hand SDFs (`cat_mjlab/task.py`, `_fields`). Similar inflation for unsampled links, or "floating" closest-point samples, would be the classical analogue.
- **Switching active points causes jitter, and a burst shortens it.** As the robot rotates sideways in the gap, the nearest sample point switches (hand → shoulder-adjacent point → hip/foot). Each switch steps the constraint direction and magnitude, which is the discontinuity Kanehiro showed can cause oscillation and even collision. A fast passage spends less time in this switching regime, which may be one more reason the learned policy prefers to burst.
- **Classical remedies for the multi-point problem, ranked by how directly they address it (inference):**
  1. **Compute guidance in configuration space**: (x, y, yaw), or per-link FMM with dilation. The "competition" disappears because one field accounts for the whole body.
  2. **Use improved, skeleton-based W-potentials** (NF2 or FM2-style centring) for the body points. This is what Barraquand recommends to reduce whole-body minima and "enlarge the maneuvering space".
  3. **Change the arbitration**: min + ε·max (ε ≈ 0.1), or a capped sum as in Huber's `w_norm`, instead of an unweighted mean.
  4. **Track closest points continuously** (Khatib's per-link closest point; Kanehiro's pairs) instead of fixed samples, with explicit inflation for the sampling gap.

### Gaps
- Choset et al. 2005, *Principles of Robot Motion* §4.6, which includes the "floating repulsive control point" construction, and Latombe 1991, *Robot Motion Planning* ch. 7 (control points in numerical potential fields), were not accessible. Only the lecture slides and Barraquand et al. 1992 were read.
- I did not access any classical humanoid-specific APF study of narrow passages: e.g., Stasse et al., ICRA 2008, HRP-2 self-collision; Sentis & Khatib whole-body obstacle avoidance; Brock & Khatib's elastic strips, IJRR 2002. Kanehiro et al. 2008 is the only humanoid whole-body source read.
- The claim that HumanoidPF's whole-body field has a minimum in front of the gap is a structural inference. It has not been checked by evaluating the per-point fields or the command rule over (x, y, yaw) for the actual G1 point set and gap geometries. That would be a cheap, decisive offline test: sweep yaw and approach distance and map the projected command's forward component.
