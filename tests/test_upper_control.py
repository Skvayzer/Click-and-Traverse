import numpy as np
import pytest
import torch
import mujoco
from types import SimpleNamespace
from cat_mjlab.upper_control import UpperGravity, action_scales
from cat_mjlab.task_math import pd_torque, protected_hand_sdf_reward, sdf_reward
from cat_mjlab.model import assemble_training_xml


def test_gravity_matches_stationary_inverse_dynamics():
    model=mujoco.MjModel.from_xml_string(assemble_training_xml())
    data=mujoco.MjData(model);gravity=UpperGravity(model,'cpu');rng=np.random.default_rng(4)
    for _ in range(20):
        data.qpos[7:]=rng.uniform(model.jnt_range[1:,0],model.jnt_range[1:,1])
        q=rng.normal(size=4);data.qpos[3:7]=q/np.linalg.norm(q)
        mujoco.mj_forward(model,data)
        actual=gravity(SimpleNamespace(xpos=torch.tensor(data.xpos[None]).float(),xmat=torch.tensor(data.xmat[None]).float()))
        np.testing.assert_allclose(actual[0],data.qfrc_bias[18:],atol=1e-5,rtol=1e-5)


def test_feedforward_precedes_saturation_and_is_not_gain_scaled():
    zeros=torch.zeros((2,29));ones=torch.ones(29)
    torque=pd_torque(zeros,zeros,zeros,ones,ones,torch.tensor([.75,1.25]),torch.ones(2),zeros,zeros,ones,feedforward=ones*2)
    torch.testing.assert_close(torque,torch.ones_like(zeros))


def test_scales_and_protected_clearance_preserve_contact():
    scale=action_scales({'upper_action_scales':{'left_shoulder_pitch_joint':1.}})
    assert scale[3]==1 and np.count_nonzero(scale!=.8)==1
    with pytest.raises(ValueError):action_scales({'upper_action_scales':{'bad':1.}})
    sdf=torch.tensor([[[-.01],[0.]],[[.05],[.1]]])
    c=dict(hand_active=torch.ones((2,2),dtype=torch.bool),role=torch.ones(2,dtype=torch.long),phase_weight=torch.ones(2))
    old=sdf_reward(sdf);new=protected_hand_sdf_reward(sdf,c)
    assert new[0]==old[0] and new[1]>old[1]
    c['role'].zero_();torch.testing.assert_close(protected_hand_sdf_reward(sdf,c),old)
