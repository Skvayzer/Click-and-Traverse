# Fixed Unitree three-finger hands for CAT traversal

These are the Unitree three-finger hand meshes from MuJoCo Menagerie's
`unitree_g1/g1_with_hands.xml`, derived from Unitree's published
`g1_29dof_with_hand_rev_1_0.xml` at commit
`c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65`. This is the Dex3-style three-finger
hand, with seven physical finger joints per hand, rather than a five-finger
BrainCo hand or the CAT model's old static rubber hand.

`source-hands.xml` contains the two unmodified wrist/hand subtrees from that
source. `provenance.json` records source and asset SHA-256 hashes. The source
BSD-3-Clause license is retained in `LICENSE`.

`cat_ppo/furniture/grippers.py` fixes the finger transforms at the source stand
pose: thumb joint 1 is +1.0472 rad on the left and -1.0472 rad on the right;
all other finger joints are at zero. This clamps the source stand keyframe's
slightly out-of-limit +/-1.05 rad thumb value to its published joint limit.
Finger articulation is not a learned action in this traversal task. The 29
body joints, actuator order and CAT checkpoint feature mapping are retained.

The source wrist/palm inertial replaces CAT's assumed 0.8 kg distal wrist/hand
inertial; each fixed finger body's source mass, COM and inertia are preserved.
Each complete distal wrist/palm/finger assembly weighs 0.7811226 kg. Protection
boxes and visual meshes add no mass. This is a source-model simulation
assumption, not hardware calibration.

Protection boxes are computed from every palm and finger STL vertex after the
fixed transforms, with **5 mm on every side**, including the thumb. Left and
right bounds differ by reflection. The physics boxes, clearance-probe corners
and visual envelope all use these same bounds. Bounds and all source hashes
are recorded in the observation contract. Changing a finger pose requires
recomputing these bounds; this box does not claim to enclose arbitrary grasp
configurations.

`geometry-contract.json` is a persisted snapshot of the same derived bounds
and provenance. Regenerate it by JSON-serializing `geometry_contract()` after
an intentional asset or fixed-pose update; a regression check prevents stale
snapshots from silently drifting from the model.

`validate_hand_envelopes(model, data)` independently checks compiled MuJoCo
vertices after `mj_forward`, including arbitrary shoulder and wrist poses.
The regression tests exercise nominal, raised, tucked and rotated wrists, and
real contacts in the thumb region missed by the previous generic box.
