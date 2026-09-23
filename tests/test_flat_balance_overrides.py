import pytest
import torch
from cat_mjlab.balance import reward_overrides,posture_terms
from cat_ppo.furniture.balance_bank import SETTINGS
from cat_mjlab.learning import task_share_diagnostics


def test_override_defaults_inheritance_and_rejection():
    assert reward_overrides(SETTINGS)=={'bonus_scale':3.,'region_scale':.15}
    assert reward_overrides(SETTINGS,{'bonus_scale':10.,'region_scale':.4})=={'bonus_scale':10.,'region_scale':.4}
    assert reward_overrides(SETTINGS,{'bonus_scale':3.},bonus_scale=10.,region_scale=.4)=={'bonus_scale':10.,'region_scale':.4}
    for value in [0.,-1.,float('nan'),float('inf'),True]:
        with pytest.raises(ValueError):reward_overrides(SETTINGS,bonus_scale=value)
        with pytest.raises(ValueError):reward_overrides(SETTINGS,region_scale=value)
    with pytest.raises(ValueError):reward_overrides(None,bonus_scale=10.)
    assert reward_overrides(None) is None
    assert SETTINGS['bonus_scale']==3. and SETTINGS['region_scale']==.15


def test_new_magnitude_isolation_and_compile():
    center=torch.tensor([[.429,.125,.870],[.429,-.125,.870]])
    hands=center[None].repeat(3,1,1);half=torch.tensor([.018,.010,.018])
    args=(torch.tensor([True,False,True]),hands,torch.zeros(3,2),torch.tensor([[1.,0.]]).repeat(3,1),torch.tensor([[.6,0,0],[.6,0,0],[0.,0,0]]),center-half,center+half)
    expected,_=posture_terms(*args,bonus_scale=10.,region_scale=.4)
    torch.testing.assert_close(expected*.02,torch.tensor([.2,0.,0.]))
    f=torch.compile(posture_terms,backend='inductor',fullgraph=True)
    actual,_=f(*args,bonus_scale=10.,region_scale=.4);torch.testing.assert_close(actual,expected)


def test_share_diagnostics_detect_reweighting():
    data=dict(task_bucket=torch.tensor([[0,1,4,5]]),raw_advantage=torch.tensor([[1.,1.,1.,10.]]),
              advantage=torch.tensor([[-.577,-.577,-.577,1.732]]),reward=torch.tensor([[1.,1.,1.,10.]]),initial_value_error=torch.tensor([[1.,1.,1.,10.]]))
    r=task_share_diagnostics(data);p='diagnostics/task_share/'
    assert r[p+'flat/sample_fraction']==.25
    assert r[p+'flat/absolute_reward_share']==pytest.approx(10/13)
    assert r[p+'flat/initial_value_error_squared_share']==pytest.approx(100/103)
    assert r[p+'flat/normalized_advantage_abs_share']>.49
    assert r[p+'retention/sample_count']==2


def test_cli_overrides():
    from train_cat_mjlab import parser
    args=parser().parse_args(['run','--checkpoint-native','x','--bank-manifest','b','--body-collision-bank','c','--body-collision-resets','r','--run-dir','o','--flat-bonus-scale','10','--flat-region-scale','.4'])
    assert (args.flat_bonus_scale,args.flat_region_scale)==(10.,.4)

def test_sapg_labels_follow_relabeling_and_diagnostics_do_not_change_targets():
    from cat_mjlab.learning import Learner,LearnerConfig,prepare_sapg_rollout
    torch.manual_seed(4)
    cfg=LearnerConfig(actor_obs=3,critic_obs=4,action_size=2,actor_hidden=(5,),critic_hidden=(5,),num_policies=3,embedding_dim=2,num_minibatches=2,num_updates_per_batch=1)
    learner=Learner(cfg,device='cpu')
    data={k:torch.randn(6,4,w) for k,w in [('state',3),('next_state',3),('privileged_state',4),('next_privileged_state',4)]}
    data['policy_id']=torch.arange(3).repeat_interleave(2)[:,None].expand(6,4)
    acted=learner.act(data['state'],data['privileged_state'],data['policy_id'])
    data.update(raw_action=acted['raw_action'],log_prob=acted['log_prob'],reward=torch.randn(6,4),discount=torch.ones(6,4),truncation=torch.zeros(6,4))
    before=prepare_sapg_rollout(learner.model,data,cfg,follower_id=1)
    data['task_bucket']=torch.tensor([[0,1,4,5]]).repeat(6,1)
    after=prepare_sapg_rollout(learner.model,data,cfg,follower_id=1)
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    torch.testing.assert_close(after['task_bucket'][-2:],data['task_bucket'][2:4])
    metrics=learner.update(data)
    assert 'diagnostics/task_share/flat/normalized_advantage_abs_share' in metrics
    assert 'diagnostics/task_updates/retention/ppo_ratio_clip_fraction' in metrics


def test_resolved_environment_config_records_overrides(monkeypatch):
    from types import SimpleNamespace
    import cat_mjlab.sim as sim
    import cat_mjlab.scene_bank as banks
    import cat_mjlab.collision as collision
    import cat_mjlab.task as task
    from cat_mjlab.runner import create_task
    monkeypatch.setattr('cat_ppo.furniture.contrast_preflight.contrast_preflight',lambda *a,**k:None)
    monkeypatch.setattr(sim,'CATSimulation',lambda *a,**k:SimpleNamespace(model=None))
    monkeypatch.setattr(banks,'SceneBank',lambda *a,**k:SimpleNamespace(balance_settings=SETTINGS,has_contrast=True))
    monkeypatch.setattr(collision,'CollisionChecker',lambda *a,**k:None)
    monkeypatch.setattr(task,'CATTask',lambda *a,**k:SimpleNamespace())
    args=SimpleNamespace(num_envs=1,device='cpu',nconmax=64,njmax=256,bank_manifest='b',body_collision_bank='c',body_collision_resets='r',seed=0,compile_task=False,flat_bonus_scale=10.,flat_region_scale=.4)
    _,_,config=create_task(args)
    assert config['flat_balance_reward']=={'bonus_scale':10.,'region_scale':.4}
    # This returned environment config is what runner.run stores in contract.
    args.flat_bonus_scale=args.flat_region_scale=None
    _,_,inherited=create_task(args,environment_config=config)
    assert inherited['flat_balance_reward']==config['flat_balance_reward']
