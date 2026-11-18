"""Whole-body motion planner (v1, 2026-11-18): scene-conditioned flow matching over short whole-body chunks.

Modules:
    fk       batched forward kinematics in torch (MuJoCo's algorithm for the free root + hinge joints) and the
             conservative sphere cover of the 35-part collision body
    grid     the robot-centred scene grid: correct trilinear sampling of a scene bank's signed distance field,
             quantization, and sampling of a stored local grid
    data     window extraction from policy rollouts and the on-disk dataset format
    model    the conditional flow-matching planner
    metrics  kinematic evaluation of planned chunks (clearance, goal error, foot artefacts, diversity)
"""
