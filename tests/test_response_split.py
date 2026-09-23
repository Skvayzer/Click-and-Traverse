"""Response decomposition, event lifecycle, task/runner/gate wiring; CPU only."""
import torch
import pytest
from cat_mjlab.response_split import initial, advance, summaries, FIELDS
from cat_mjlab.acceptance import PassageGate, TrainingAcceptance
from test_mjlab_acceptance import geometry


def sample(state, root, arm, clearance=.1, done=False):
    root=torch.tensor(root,dtype=torch.float32).reshape(-1,3)
    hands=root[:,None]+torch.tensor(arm,dtype=torch.float32).reshape(-1,2,3)
    normals=torch.tensor([1.,0.,0.]).expand_as(hands)
    return advance(state,hands,root,torch.full((len(root),2),clearance),normals,
                   torch.full((len(root),),done))


def test_arm_body_mixed_and_zero_displacement_events():
    s=initial(torch.zeros(3,3))
    sample(s,[[0,0,0]]*3,[[[0,0,0]]*2]*3)
    r=sample(s,[[0,0,0],[.2,0,0],[.1,0,0]],
             [[[.2,0,0]]*2,[[0,0,0]]*2,[[.2,0,0]]*2],.23)
    torch.testing.assert_close(r['hand_relative_retreat_sum_m'],torch.tensor([.2,0,.2]))
    torch.testing.assert_close(r['root_world_displacement_sum_m'],torch.tensor([0,.2,.1]))
    torch.testing.assert_close(r['ratio_sum'],torch.tensor([0,0,2.]))
    assert r['zero_root_count'].tolist()==[1,0,0]
    assert r['event_count'].sum()==3
    assert not s['active'].any()
    assert summaries({k:float(v.sum()) for k,v in r.items()})['hand_to_root_ratio_mean']==pytest.approx(1.)


def test_hysteresis_terminal_event_and_no_double_count():
    s=initial(torch.zeros(1,3));sample(s,[[0,0,0]],[[[0,0,0]]*2])
    r=sample(s,[[.1,0,0]],[[[.1,0,0]]*2],.21)
    assert r['event_count'].item()==0
    r=sample(s,[[.2,0,0]],[[[.1,0,0]]*2],.21,True)
    assert r['event_count'].item()==1
    assert sample(s,[[0,0,0]],[[[0,0,0]]*2],.3)['event_count'].item()==0


def test_acceptance_records_each_event_and_training_exports_ratios():
    g=PassageGate(geometry(),torch.zeros(1,2),20.)
    s=initial(torch.zeros(1,3));sample(s,[[0,0,0]],[[[0,0,0]]*2])
    r=sample(s,[[.1,0,0]],[[[.2,0,0]]*2],.23)
    z=torch.zeros(1,dtype=torch.bool)
    g.advance(torch.tensor([[.1,0.]]),1.,clean_goal=z,done=z,hand_collision=z,
              body_collision=z,fall=z,response_split=r)
    assert g.counts()['response_split_ratio_sum'].item()==pytest.approx(2.)
    observer=TrainingAcceptance();observer.gate=g;observer.masks={'':torch.ones(1,dtype=torch.bool)}
    observer.previous_counts={k:torch.zeros_like(v) for k,v in g.counts().items()}
    observer._accumulate()
    assert observer.metrics()['training/passage_acceptance_response_split_hand_to_root_ratio_mean']==pytest.approx(2.)
    restored=TrainingAcceptance();restored.load_state_dict(observer.state_dict())
    assert restored.metrics()==observer.metrics()


def test_real_task_emits_pre_autoreset_event_and_preserves_contract():
    from test_mjlab_task import _CPUSimulation,_tiny_bank
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    cfg=wholebody_config();cfg['randomize_initial_episode_steps']=False
    bank=_tiny_bank(1);bank.episode_lengths[:]=2
    def field(name,p,ids):
        if name=='sdf':return torch.full((*p.shape[:-1],1),.15)
        v=torch.zeros_like(p);v[:,:,0]=1. if name=='bf' else .6
        return v
    bank.sample=field
    task=CATTask(_CPUSimulation(1),bank,cfg,seed=4)
    first=task.step(torch.zeros(1,29));second=task.step(torch.zeros(1,29))
    assert first['metrics']['response_split/event_count'].item()==0
    assert second['metrics']['response_split/event_count'].item()==1
    assert task.info['response_split_active'].all()  # New near-obstacle reset starts a fresh event.
    torch.testing.assert_close(task.info['response_split_root'],task.data.qpos[:,:3])
    assert task.obs['state'].shape==(1,222)
    assert task.obs['privileged_state'].shape==(1,310)
    for key in FIELDS:assert 'response_split/'+key in second['metrics']


def test_runner_logs_event_weighted_means_without_training():
    from test_mjlab_runner import FakeTask,tiny_config
    from cat_mjlab.learning import Learner
    from cat_mjlab.runner import collect_rollout
    class Events(FakeTask):
        def step(self,action):
            result=super().step(action)
            for key in FIELDS:
                result['metrics']['response_split/'+key]=torch.full((self.num_envs,),
                    {'event_count':1.,'ratio_valid_count':1.,'ratio_sum':2.,
                     'hand_relative_retreat_sum_m':.2,'root_world_displacement_sum_m':.1}.get(key,0.))
            return result
    _, info=collect_rollout(Events(),Learner(tiny_config(),device='cpu'),unroll_length=2,
                           trajectories=6,policy_ids=torch.arange(3).repeat_interleave(2))
    assert info['metrics']['training/response_split/event_count']==12
    assert info['metrics']['training/response_split/hand_to_root_ratio_mean']==2


def test_translation_relative_definition_matches_poke_projection():
    hand=torch.tensor([.3,.5,.9]);root=torch.tensor([.1,.1,.7])
    control_hand=torch.tensor([.1,.4,.9]);control_root=torch.tensor([0.,.1,.7])
    approach=torch.tensor([0.,1.,0.])
    expected=-((hand-root)-(control_hand-control_root))@approach
    s=initial(control_root[None])
    normals=(-approach).expand(1,2,3)
    advance(s,control_hand.expand(1,2,3),control_root[None],torch.full((1,2),.1),normals,torch.tensor([False]))
    r=advance(s,hand.expand(1,2,3),root[None],torch.full((1,2),.23),normals,torch.tensor([False]))
    assert r['hand_relative_retreat_sum_m'].item()==pytest.approx(expected.item())


def test_offline_acceptance_computes_response_from_world_samples():
    from cat_mjlab.acceptance import evaluate_assigned_trials
    room=dict(route=[[0.,0.],[10.,0.]],hand_contrast=dict(
        zones=[dict(start_m=1.,end_m=2.)],modules=[dict(width_m=1.)]))
    assignment=dict(trial_id='poke',room=room,start_xy=[0.,0.],deadline_s=5.,cohort='upstream',
                    reset_body_collision=False,reset_hand_collision=False,initial_hand_clearance=[.1,.3])
    def rollout(row):
        for j in (0,1):
            yield dict(time_s=j+1.,xy=[.1*j,0.],root_xyz=[.1*j,0.,.8],
                       hand_xyz=[[.3*j,0.,1.],[.3*j,0.,1.]],hand_normals=[[1.,0.,0.]]*2,
                       hand_clearance=[.1,.3] if j==0 else [.23,.3],
                       clean_goal=False,done=bool(j),hand_collision=False,body_collision=False,fall=False,other=False)
    result=evaluate_assigned_trials([assignment],rollout,clearance_primary=True)
    assert result['trials']['poke']['response_split_ratio_sum']==pytest.approx(2.)
    assert result['trials']['poke']['response_split_event_count']==1


def test_tracking_override_is_explicit_validated_and_inherited():
    from cat_mjlab.config import wholebody_config
    from train_cat_mjlab import parser
    args=parser().parse_args(['run','--from-scratch','--bank-manifest','b',
        '--body-collision-bank','c','--body-collision-resets','r','--run-dir','o',
        '--tracking-root-field-weight','2'])
    c=wholebody_config(tracking_root_field_weight=args.tracking_root_field_weight)
    assert c['reward_config']['scales']['tracking_root_field']==2
    assert wholebody_config(c)['reward_config']['scales']['tracking_root_field']==2
    for value in (0,-1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):wholebody_config(tracking_root_field_weight=value)


def test_legacy_clearance_observer_upgrade_preserves_columns_by_name():
    from copy import deepcopy
    from cat_mjlab.acceptance import ClearancePassageGate
    g=ClearancePassageGate(geometry(),torch.zeros(1,2),20.,initial_clearance=torch.tensor([[.1,.3]]))
    observer=TrainingAcceptance();observer.gate=g;observer.masks={'':torch.ones(1,dtype=torch.bool)}
    observer.previous_counts={k:torch.zeros_like(v) for k,v in g.counts().items()}
    observer._accumulate()
    state=deepcopy(observer.state_dict());state['gate'].pop('response_counts')
    keep=[j for j,k in enumerate(state['count_names']) if not k.startswith('response_split_')]
    state['totals']=state['totals'][:,keep]
    state['count_names']=[state['count_names'][j] for j in keep]
    state['previous_counts']={k:v for k,v in state['previous_counts'].items() if not k.startswith('response_split_')}
    restored=TrainingAcceptance();restored.load_state_dict(state)
    before=restored.metrics()
    restored._accumulate()
    assert restored.metrics()==before
    assert before['training/passage_acceptance_clearance_trial_minimum_mean_m']==pytest.approx(.1)
    assert before['training/passage_acceptance_response_split_event_count']==0
