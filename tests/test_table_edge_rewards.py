"""Table-specific posture incentives using the actual G1 and obstacle field.

These compare static posture costs at equal motion. They are not full-return
comparisons or evidence that a trained policy already raises its hands.
"""
import math

import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from cat_ppo.envs.g1 import constants
from cat_ppo.envs.g1.env_cat import G1CatEnv
from cat_ppo.furniture.contrastive_passages import _hands, _poses
from cat_ppo.furniture.contrastive_rewards import eval_context, pack_hand_contrast, reward_terms
from cat_ppo.furniture.generalist_fields import sample_ragged_field
from cat_ppo.furniture.hand_passages import _robot, _sdf, arm_pose
from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair
from cat_ppo.furniture.wholebody_stability import hand_clearance_pressure, upper_stability_terms


@pytest.fixture(scope="module")
def protected_table():
    return generate_table_edge_pair(20260919, certify=False)[1]


def _context(scene, module):
    yaw = scene["generator"]["passage_yaw_rad"]
    return eval_context(pack_hand_contrast([scene]), 0,
                        module["center_local_x_m"] + 3.5,
                        jp.array([math.cos(yaw), math.sin(yaw)]))


def test_table_target_requires_entire_hand_above_table_and_does_not_accept_tucking(protected_table):
    scene = protected_table
    height = scene["hand_contrast"]["tabletop_top_z_m"]
    radii = _hands("nominal")[1]
    for zone in scene["hand_contrast"]["zones"]:
        assert zone["hand_active"] == [1, 1]
        assert zone["region_valid"] == [True, False]
        lower = np.array(zone["hand_regions_min"])[0]
        assert np.all(lower[:, 2] - radii > height + .06)

    # Test posture shaping in the scene's actual route frame. The unselected
    # tucked region remains present for fixed shapes but must not win the min.
    module = scene["hand_contrast"]["modules"][0]
    context = _context(scene, module)
    tangent = np.asarray(context["route_tangent"])
    rotation = np.array([[tangent[0], -tangent[1]], [tangent[1], tangent[0]]])
    for mode in ("raised", "tucked", "nominal"):
        local = _hands(mode, .95)[0].copy()
        local[:, :2] = local[:, :2] @ rotation.T
        terms, telemetry = reward_terms(context, jp.asarray(local), jp.zeros(2), tangent, tangent)
        if mode == "raised":
            assert float(terms["wholebody_hand_contrast_region"]) == pytest.approx(0.)
            assert telemetry["hand_contrast_hand_good"] == 1
        else:
            assert float(terms["wholebody_hand_contrast_region"]) > .7
            assert telemetry["hand_contrast_hand_good"] == 0


def _static_posture_scores(scene, module):
    model, _ = _robot()
    data = mujoco.MjData(model)
    sdf, origin = _sdf(scene)
    radii = jp.asarray(_hands("nominal")[1])
    nominal = jp.asarray(arm_pose()[19:])
    yaw = scene["generator"]["passage_yaw_rad"]
    context = _context(scene, module)

    def sample(names):
        positions = np.array([data.site(name).xpos for name in names])
        return sample_ragged_field(sdf.reshape(-1, 1), positions, origin=origin,
                                   dx=.04, shape=np.array(sdf.shape), offset=0)

    results = {}
    for label, mode, turn in (("raised", "raised", 0.), ("tucked", "tucked", 0.),
                              ("nominal", "nominal", 0.), ("sideways", "nominal", math.pi / 2)):
        qpos = arm_pose(mode, .95)
        qpos[:2] = _poses(scene, [module["center_local_x_m"]])[0, :2]
        heading = yaw + turn
        qpos[3:7] = [math.cos(heading / 2), 0., 0., math.sin(heading / 2)]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        hands = sample(constants.HAND_SITES).reshape(2) - radii
        elbows = sample(["left_elbow_probe", "right_elbow_probe"]).reshape(2)
        # Preserve the released feet SDF weight of two, all other keypoints one.
        native = sum(weight * float(G1CatEnv._re_sdf(None, sample(names))) for names, weight in
                     ((["head"], 1.), (constants.FEET_SITES, 2.), (constants.KNEE_SITES, 1.),
                      (constants.SHOULDER_SITES, 1.)))
        native += float(G1CatEnv._re_sdf(None, hands))
        hand_positions = jp.asarray([data.site(name).xpos for name in constants.HAND_SITES])
        facing = jp.array([math.cos(heading), math.sin(heading)])
        contrast, _ = reward_terms(context, hand_positions, jp.asarray(qpos[:2]), facing, facing)
        target = jp.asarray(qpos[19:])
        stability, _ = upper_stability_terms(target, target, target, nominal, hands, elbows,
            dt=.02, hand_protection=True, contrast_arm_active=context["hand_active"])
        pressure, _ = hand_clearance_pressure(hands)
        elbow_cost = jp.mean(jp.maximum(.08 - elbows, 0.) ** 2)
        score = (native - sum(float(v) for v in contrast.values()) - .5 * float(pressure)
                 - 2. * float(elbow_cost) - .05 * float(stability["wholebody_upper_clear_posture"]))
        results[label] = dict(score=score, hands=np.asarray(hands),
                             contrast=float(contrast["wholebody_hand_contrast_region"]))
    return results


def test_raised_forward_beats_sideways_tucked_and_nominal_in_static_native_reward_budget(protected_table):
    for module in protected_table["hand_contrast"]["modules"]:
        values = _static_posture_scores(protected_table, module)
        raised = values["raised"]
        assert raised["hands"].min() > .10
        for label in ("nominal", "tucked", "sideways"):
            assert raised["score"] > values[label]["score"] + .5, (module, values)


def test_lowering_the_body_cannot_move_target_below_table_or_deactivate_hazard(protected_table):
    scene = protected_table
    context = _context(scene, scene["hand_contrast"]["modules"][0])
    tangent = np.asarray(context["route_tangent"])
    rotation = np.array([[tangent[0], -tangent[1]], [tangent[1], tangent[0]]])
    # This isolates target dependence, not a valid crouching trajectory. Actual
    # bent-leg, floor-referenced stance safety is checked by the scene certificate.
    raised = _hands("raised", .95)[0].copy()
    raised[:, :2] = raised[:, :2] @ rotation.T
    costs = []
    for lowering in (0., .10, .25):
        hands = raised.copy()
        hands[:, 2] -= lowering
        terms, telemetry = reward_terms(context, jp.asarray(hands), jp.zeros(2), tangent, tangent)
        costs.append(float(terms["wholebody_hand_contrast_region"]))
        assert telemetry["hand_contrast_hand_active"] == 1
        if lowering >= .10:
            assert telemetry["hand_contrast_hand_good"] == 0
    assert costs[0] == pytest.approx(0.)
    assert costs[0] < costs[1] < costs[2] < 1.
