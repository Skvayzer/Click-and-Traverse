"""Opt-in protected-zone gait curriculum, with bounded standing and hysteresis.

Local first-zone trials are training diagnostics, not full-passage acceptance.
All state is tensor-valued so CATTask checkpoints preserve in-flight trials.
"""
import math
import torch

RUNGS = (0., .2, .3, .45, .6)
ACQUIRE_SECONDS = 2.5
HOLD_SECONDS = 2.
RELEASE_SECONDS = 2.
RELEASE_PROGRESS = .1
COMPLIANCE = .9
PROGRESS_FRACTION = .8


def configure(config, *, enabled=False, window=256, threshold=.8, demote_threshold=.4):
    if type(window) is not int or window < 1:
        raise ValueError('Speed curriculum window must be a positive integer')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (threshold, demote_threshold)) or not 0 <= demote_threshold < threshold <= 1:
        raise ValueError('Speed thresholds require 0 <= demote < advance <= 1')
    config.pop('hand_speed_curriculum', None)
    if enabled:
        if not config.get('wholebody_hand_contrast'):
            raise ValueError('Speed curriculum requires hand contrast')
        if config.get('hand_tolerance_curriculum'):
            raise ValueError('Run speed and tolerance curricula separately for attribution')
        if config.get('hand_contrast_metric_tolerance', .05) != .05:
            raise ValueError('Speed curriculum requires strict 0.05 m tolerance')
        config['hand_speed_curriculum'] = dict(rungs=list(RUNGS), window=window,
            threshold=threshold, demote_threshold=demote_threshold)
    return config


def initialize(settings, n, device):
    def zeros(*shape, dtype=torch.long):return torch.zeros(shape, dtype=dtype, device=device)
    return dict(stage=zeros(), generation=zeros(), history=zeros(2,settings['window'],dtype=torch.bool),
        count=zeros(2), bad_windows=zeros(2), reviewed=zeros(2),
        completed=zeros(5,2), successes=zeros(5,2), compliance_sum=zeros(5,2,dtype=torch.float),
        rates=zeros(5,2,dtype=torch.float), promotions=zeros(), demotions=zeros(),
        episode_rung=zeros(n), episode_generation=zeros(n), started=zeros(n,dtype=torch.bool),
        finished=zeros(n,dtype=torch.bool), ticks=zeros(n), good=zeros(n), zone=zeros(n),
        origin=zeros(n,dtype=torch.float), release_origin=zeros(n,dtype=torch.float),
        previous_progress=zeros(n,dtype=torch.float), drift=zeros(n,dtype=torch.float),
        compliant_progress=zeros(n,dtype=torch.float),
        seeded_completed=zeros(5), seeded_successes=zeros(5))


def reset(state, ids, progress):
    for key in ('started','finished','ticks','good','zone','drift','compliant_progress'):
        state[key][ids]=0
    state['episode_rung'][ids]=state['stage']
    state['episode_generation'][ids]=state['generation']
    for key in ('origin','release_origin','previous_progress'):state[key][ids]=progress


def active_mask(context):
    # Actual declared hand zones only, not incoming approach shaping or a
    # scene-wide override. Transition narrow modules have hand_active=False.
    return ((context['role']==1)|(context['role']==3)) & context['enabled'] \
        & context['hand_active'].any(-1) & context['region_valid'].any(-1) & (context['phase_weight']>0)


def speed_limit(state, context, ids, dt):
    rung=state['episode_rung'][ids]
    speed=torch.tensor(RUNGS,device=rung.device)[rung]
    # Stage zero approaches at .2, allows acquisition, measures a two-second hold,
    # then releases at .2 for the remainder of this episode, even on failure.
    hold=(rung==0)&context['core_active']&~state['finished'][ids]&(state['ticks'][ids]<round((ACQUIRE_SECONDS+HOLD_SECONDS)/dt))
    speed=torch.where(rung==0,torch.where(hold,0.,.2),speed)
    active=active_mask(context)
    return active, speed, active&hold


def restart_phase(phase, previous_command, command, eligible):
    # The standing phase has both feet at zero. Restore anti-phase stepping
    # on release instead of accidentally commanding a synchronized hop.
    restart=eligible&(previous_command[:,0]==0)&(command[:,0]>.5)
    stagger=phase.new_tensor([0.,math.pi]).expand_as(phase)
    return torch.where(restart[:,None],stagger,phase)


def standing_phase(command, command_delay, stop, phase, gait, hold):
    """A real stance: legacy phase updater otherwise rewrites move=1 at zero speed."""
    command=torch.where(hold[:,None],0.,command)
    command_delay=torch.where(hold[:,None],0.,command_delay)
    stop=torch.where(hold,0,stop)
    phase=torch.where(hold[:,None],0.,phase)
    gait=torch.where(hold[:,None],1.,gait)  # both feet in stance
    return command,command_delay,stop,phase,gait


def record(state, settings, *, roles, rungs, generations, eligible, success, compliance, seeded):
    # All episodes are frozen at reset, including generation: old episodes may
    # not re-enter a rung's window after demotion and later re-promotion.
    current=eligible&(rungs==state['stage'])&(generations==state['generation'])
    stage=int(state['stage']);w=settings['window']
    seed=current&seeded
    state['seeded_completed'][stage]+=seed.long().sum()
    state['seeded_successes'][stage]+=(seed&success).long().sum()
    sustained=bool((state['bad_windows']>=2).any())
    for col, role in enumerate((1,3)):
        mask=current&~seeded&(roles==role)
        values=success[mask];n=values.numel()
        if not n:continue
        old=int(state['count'][col])
        state['completed'][stage,col]+=n
        state['successes'][stage,col]+=values.long().sum()
        state['compliance_sum'][stage,col]+=compliance[mask].sum()
        # Process windows in order, so large simultaneous batches cannot hide
        # two consecutive failing windows behind one final tail.
        offset=0
        while offset<n:
            count=int(state['count'][col]);chunk=min(n-offset,w-count%w)
            slots=(torch.arange(chunk,device=values.device)+count)%w
            state['history'][col,slots]=values[offset:offset+chunk]
            state['count'][col]+=chunk;offset+=chunk
            if int(state['count'][col])%w==0:
                rate=state['history'][col].float().mean()
                state['bad_windows'][col]=torch.where(rate<settings['demote_threshold'],state['bad_windows'][col]+1,0)
                state['reviewed'][col]=state['count'][col]
                sustained |= bool(state['bad_windows'][col]>=2)
        state['rates'][stage,col]=state['history'][col].float().sum()/min(old+n,w)
    demote=stage>0 and sustained
    promote=(not demote and stage<len(RUNGS)-1 and bool((state['count']>=w).all())
             and bool((state['rates'][stage]>=settings['threshold']).all()))
    if demote or promote:
        state['stage']+=1 if promote else -1
        state['generation']+=1
        state['promotions' if promote else 'demotions']+=1
        for key in ('history','count','bad_windows','reviewed'):state[key].zero_()


def observe(state, settings, *, context, previous_progress, progress, strict_good,
            heading_good, failure, resolved, leader, seeded, dt):
    """One bounded first-core-zone trial per reset; no-entry outcomes fail.

    Zero rung: 2.5 s acquisition allowance, then 2 s strict hold (<=5 cm
    accumulated motion), then >=10 cm net
    release progress within 2 s. Positive rungs: 2 s strict compliance and
    >=80% commanded net progress, <=120% (plus 2 cm discretization allowance).
    Leaving the starting core or failing physically before observation ends
    fails closed. Success is local; it does not certify the entire passage.
    """
    active=active_mask(context)&context['core_active']
    new=active&~state['started']&~state['finished']
    state['started']|=new
    state['origin']=torch.where(new,previous_progress,state['origin'])
    state['zone']=torch.where(new,context['zone_index'],state['zone'])
    running=state['started']&~state['finished']
    in_zone=active&(context['zone_index']==state['zone'])
    before=state['ticks'].clone();hold_ticks=round(HOLD_SECONDS/dt)
    rung=state['episode_rung'];zero=rung==0
    acquire=torch.where(zero,round(ACQUIRE_SECONDS/dt),0)
    target_ticks=acquire+hold_ticks
    scoring=running&(before>=acquire)&(before<target_ticks)
    state['ticks']+=running.long()
    state['good']+=(scoring&in_zone&strict_good&heading_good&~failure).long()
    state['drift']+=torch.where(scoring&zero,(progress-previous_progress).abs(),0.)
    state['compliant_progress']+=torch.where(scoring&in_zone&strict_good&heading_good&~failure,
        progress-previous_progress,0.)
    end_hold=running&zero&(state['ticks']==target_ticks)
    state['release_origin']=torch.where(end_hold,progress,state['release_origin'])
    release=(progress-state['release_origin'])>=RELEASE_PROGRESS
    timed=state['ticks']>=target_ticks+round(RELEASE_SECONDS/dt)
    finish=running & (failure|resolved|~in_zone|torch.where(zero,(state['ticks']>target_ticks)&release|timed,state['ticks']>=target_ticks))
    role=(context['role']==1)|(context['role']==3)
    no_entry=resolved&role&~state['started']&~state['finished']
    finish|=no_entry
    compliance=state['good'].float()/hold_ticks
    speed=torch.tensor(RUNGS,device=rung.device)[rung]
    distance=progress-state['origin'];expected=speed*HOLD_SECONDS
    motion=torch.where(zero,(state['drift']<=.05)&release,
        (distance>=PROGRESS_FRACTION*expected)&(state['compliant_progress']>=PROGRESS_FRACTION*expected)
        &(distance<=1.2*expected+.02))
    success=finish&state['started']&in_zone&~failure&(state['ticks']>=target_ticks)&(compliance>=COMPLIANCE)&motion
    record(state,settings,roles=context['role'],rungs=rung,generations=state['episode_generation'],
        eligible=finish&leader,success=success,compliance=compliance,seeded=seeded)
    state['finished']|=finish
    state['previous_progress'].copy_(progress)


def metrics(state):
    stage=int(state['stage'])
    result={'hand_speed/rung':stage,'hand_speed/command_m_s':RUNGS[stage],
            'hand_speed/promotions':int(state['promotions']),'hand_speed/demotions':int(state['demotions'])}
    for col,role in enumerate(('protected','transition')):
        n=min(int(state['count'][col]),state['history'].shape[1])
        result[f'hand_speed/{role}/window_count']=n
        result[f'hand_speed/{role}/window_success']=float(state['history'][col].sum())/max(n,1)
        result[f'hand_speed/{role}/consecutive_bad_windows']=int(state['bad_windows'][col])
    for rung in range(5):
        result[f'hand_speed/rung_{rung}/seeded_completed']=int(state['seeded_completed'][rung])
        result[f'hand_speed/rung_{rung}/seeded_successes']=int(state['seeded_successes'][rung])
        for col,role in enumerate(('protected','transition')):
            prefix=f'hand_speed/rung_{rung}/{role}'
            n=int(state['completed'][rung,col])
            result[prefix+'/completed']=n
            result[prefix+'/successes']=int(state['successes'][rung,col])
            result[prefix+'/rolling_success']=float(state['rates'][rung,col])
            result[prefix+'/mean_strict_compliance']=float(state['compliance_sum'][rung,col])/max(n,1)
    return result
