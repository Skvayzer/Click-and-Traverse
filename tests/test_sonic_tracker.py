"""SONIC adapter tables against the official deployment constants (gear_sonic_deploy policy_parameters.hpp)."""
import math

import torch

from cat_mjlab import sonic_tracker as s

# policy_parameters.hpp, verbatim
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22,
                      24, 26, 28]
MUJOCO_TO_ISAACLAB = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20,
                      27, 21, 28]


def test_joint_maps_match_deployment():
    names = list(s.MUJOCO_JOINTS)
    mj_to_il = [names.index(joint) for joint in s.ISAACLAB_JOINTS]
    assert mj_to_il == MUJOCO_TO_ISAACLAB
    assert torch.tensor(mj_to_il).argsort().tolist() == ISAACLAB_TO_MUJOCO


def test_gains_and_scales():
    w = 10 * 2 * 3.1415926535
    assert math.isclose(s.KPS[0], 0.025101925 * w * w, rel_tol=1e-6)            # hip pitch, 7520_22
    assert math.isclose(s.KPS[4], 2 * 0.003609725 * w * w, rel_tol=1e-6)        # ankle pitch, 2 x 5020
    assert math.isclose(s.KDS[13], 2 * 4 * 0.003609725 * w, rel_tol=1e-6)      # waist roll, 2 x 5020
    assert math.isclose(s.ACTION_SCALE[4], .25 * 25. / (0.003609725 * w * w), rel_tol=1e-6)
    assert math.isclose(s.ACTION_SCALE[20], .25 * 5. / (0.00425 * w * w), rel_tol=1e-6)   # wrist pitch, 4010
    assert s.DEFAULT_ANGLES[3] == 0.669 and s.DEFAULT_ANGLES[23] == -0.2


def test_anchor_orientation_identity():
    q = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    rel = s.quat_mul(s.quat_conj(q), q)
    six = s.quat_matrix(rel)[..., :2].reshape(5, -1)
    assert torch.allclose(six, torch.tensor([1., 0., 0., 1., 0., 0.]).expand(5, 6), atol=1e-6)
