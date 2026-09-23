"""Opt-in empty-ground posture reward and explicitly denominated leader telemetry."""
import math
import torch


def reward_overrides(settings, inherited=None, *, bonus_scale=None, region_scale=None):
    """Resolve finite positive run parameters without altering immutable bank settings."""
    values=dict(inherited or {})
    if set(values)-{'bonus_scale','region_scale'}:raise ValueError('Unknown flat reward override')
    for name,value in [('bonus_scale',bonus_scale),('region_scale',region_scale)]:
        if value is not None:values[name]=value
    if settings is None:
        if values:raise ValueError('Flat reward overrides require a flat-balance bank')
        return None
    result={name:values.get(name,settings[name]) for name in ('bonus_scale','region_scale')}
    for name,value in result.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'Flat {name} must be finite and positive')
    return result


def posture_terms(flat, hands, root_xy, tangent, velocity, lower, upper, bonus_scale=3., region_scale=.15):
    """Raw reward (multiply by control dt once), structurally zero off flat rows."""
    delta=hands[:,:,:2]-root_xy[:,None]
    normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
    local=torch.stack(((delta*tangent[:,None]).sum(-1),(delta*normal[:,None]).sum(-1),hands[:,:,2]),-1)
    distance=(lower-local).clamp_min(0)+(local-upper).clamp_min(0)
    squared=distance.square().sum(-1)
    cost=(squared/(squared+region_scale**2)).mean(-1)
    compliant=(squared<=.05**2+1e-10).all(-1)
    speed=(velocity[:,:2]*tangent).sum(-1)
    bonus=torch.where(flat,bonus_scale*(1-cost)*(speed/.6).clamp(0,1),0.)
    return bonus,dict(cost=cost,compliant=compliant,walking=speed>=.2,speed=speed)


class BalanceMetrics:
    """One-update leader statistics. Streak lives in task across update boundaries."""
    def __init__(self):
        self.rows=[]
        self.all_leader_steps=None
        self.counts=None
        self.max_streak=None
    def append(self, metrics, done, leader, dt):
        if 'balance/flat' not in metrics:return
        count=leader.sum()
        self.all_leader_steps=count if self.all_leader_steps is None else self.all_leader_steps+count
        mask=leader&metrics['balance/flat'];walking=mask&metrics['balance/walking']
        raised=mask&metrics['balance/compliant'];both=walking&raised
        nominal=mask&metrics['balance/nominal']
        fallen=mask&done&metrics['episode/fall']
        values=torch.stack((mask.sum(),walking.sum(),both.sum(),(mask&done).sum(),fallen.sum(),
            torch.where(mask,metrics['balance/speed'],0.).sum(),
            raised.sum(),nominal.sum(),
            torch.where(raised,metrics['balance/foot_balance'],0.).sum(),
            torch.where(nominal,metrics['balance/foot_balance'],0.).sum(),
            torch.where(mask,metrics['balance/bonus'],0.).sum(),
            (walking&nominal).sum(),
            torch.where(both,metrics['balance/foot_balance'],0.).sum(),
            torch.where(walking&nominal,metrics['balance/foot_balance'],0.).sum())).float()
        self.counts=values if self.counts is None else self.counts+values
        maximum=torch.where(mask,metrics['balance/streak'],0).max()
        self.max_streak=maximum if self.max_streak is None else torch.maximum(self.max_streak,maximum)
        self.rows.append(metrics['balance/heights'][mask].detach())
        self.dt=dt
    def result(self):
        if self.counts is None:return {}
        n,walk,both,episodes,falls,speed,raised,nominal,rf,nf,bonus,walking_nominal,wrf,wnf=self.counts.tolist()
        p='balance/leader_'
        out={p+'step_count':n,p+'walking_step_count':walk,p+'walking_compliant_step_count':both,
             p+'completed_episode_count':episodes,p+'fall_episode_count':falls,
             p+'simulated_minutes':n*self.dt/60,p+'raised_step_count':raised,p+'nominal_step_count':nominal,
             p+'longest_raised_walking_seconds':float(self.max_streak)*self.dt}
        out[p+'all_task_step_count']=float(self.all_leader_steps)
        out[p+'flat_step_fraction']=n/max(float(self.all_leader_steps),1)
        out[p+'walking_nominal_step_count']=walking_nominal
        if both:out[p+'foot_balance_walking_raised_reward_per_step']=wrf/both
        if walking_nominal:out[p+'foot_balance_walking_nominal_reward_per_step']=wnf/walking_nominal
        if n:out.update({p+'all_steps_walking_compliant_fraction':both/n,p+'fall_rate_per_simulated_minute':falls/(n*self.dt/60),p+'route_speed_mean_mps':speed/n,p+'posture_bonus_mean_per_step':bonus/n})
        if walk:out[p+'walking_steps_compliant_fraction']=both/walk
        if episodes:out[p+'fall_rate_per_episode']=falls/episodes
        if raised:out[p+'foot_balance_raised_reward_per_step']=rf/raised
        if nominal:out[p+'foot_balance_nominal_reward_per_step']=nf/nominal
        if n:
            heights=torch.cat(self.rows)
            quantiles=torch.quantile(heights,torch.tensor([0.,.1,.5,.9,1.],device=heights.device),dim=0).tolist()
            for j,name in enumerate(('root','left_hand','right_hand')):
                for i,q in enumerate(('min','p10','p50','p90','max')):out[p+name+'_height_'+q+'_m']=quantiles[i][j]
        return out
