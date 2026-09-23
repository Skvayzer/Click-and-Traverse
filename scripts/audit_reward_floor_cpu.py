"""Replay fixed CPU PD traces through production rewards; never train or load a policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import json
from collections import defaultdict
import torch
from audit_protected_objective_cpu import audit, ROOT


def main():
    samples = []
    static = defaultdict(list)
    def capture(obj, action, contacts, label, core, valid, penalty):
        scales = obj.config['reward_config']['scales']
        original = scales['wholebody_hand_contrast_region']
        for weight, width in [(w, 0.) for w in (-1, -3, -5, -10, -20)] + [(-3, .2)]:
            scales['wholebody_hand_contrast_region'] = weight
            obj.config['hand_reward_soft_floor'] = width
            post, terms = obj._rewards(action, contacts)
            pre = sum(terms.values()) * obj.dt
            static[(weight, width, label)].append(dict(pre=float(pre[0]), post=float(post[0]+penalty[0])))
            for i in torch.where(core)[0]:
                samples.append(dict(label=label, weight=weight, width=width,
                    valid=bool(valid[i]), pre=float(pre[i]), post=float(post[i]+penalty[i]),
                    clipped=bool(obj.telemetry['reward_floor_clipped'][i]),
                    slope=float(obj.telemetry['reward_floor_slope'][i]),
                    lift=float(obj.telemetry['reward_soft_floor_lift'][i]),
                    region=float(obj.telemetry['hand_contrast_region_cost'][i]),
                    handsdf=float(terms['handsdf'][i]*obj.dt), collision=float(penalty[i])))
        scales['wholebody_hand_contrast_region'] = original
        obj.config['hand_reward_soft_floor'] = 0.
    result = audit(sample_callback=capture, bank_path=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json')
    summary = []
    for weight, width in [(w, 0.) for w in (-1, -3, -5, -10, -20)] + [(-3, .2)]:
        for valid in (True, False):
            rows = [r for r in samples if r['weight']==weight and r['width']==width and r['label']!='slew_raise' and (not valid or r['valid'])]
            pre = torch.tensor([r['pre'] for r in rows]); slope = torch.tensor([r['slope'] for r in rows])
            summary.append(dict(weight=weight, width=width, through_first_termination=valid,
                n=len(rows), clipped=sum(r['clipped'] for r in rows), negative=int((pre<0).sum()),
                shaped_sum_before_dt=dict(mean=float(pre.mean()/.02),
                    quantiles=dict(zip(('min','p05','p25','p50','p75','p95','max'),(torch.quantile(pre,torch.tensor([0.,.05,.25,.5,.75,.95,1.]))/.02).tolist()))),
                slope_min=float(slope.min()), slope_median=float(slope.median()), slope_mean=float(slope.mean()),
                mean_lift=sum(r['lift'] for r in rows)/len(rows),
                region_mean=sum(r['region'] for r in rows)/len(rows),
                positive_region_bonus_still_clipped=int(((pre+.06)<0).sum()) if weight==-3 and width==0 else None))
    fixed = [r for r in samples if r['width']==.2]
    base = [r for r in samples if r['weight']==-3 and r['width']==0]
    assert len(fixed)==len(base)
    inversions = sum(a['post']*b['post'] < 0 for a,b in zip(base,fixed))
    original = [r for r in samples if r['weight']==-20 and r['width']==0]
    original_inversions = sum(a['post']*b['post'] < 0 for a,b in zip(original,fixed))
    ledger = {}
    for width in (0.,.2):
        a=static[(-3,width,'nominal')]; b=static[(-3,width,'commandable')]
        ledger[str(width)]={k:sum(y[k]-x[k] for x,y in zip(a,b))/len(a) for k in ('pre','post')}
    report=dict(method=result['method'], bank='data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json',
        config_source=result['config_source'], config_sha256=result['config_sha256'],
        heading_weight=-5, dt=.02, summary=summary, static_delta_at_region_minus3=ledger,
        measured_final_reward_sign_inversions=inversions,
        original_minus20_final_reward_sign_inversions=original_inversions, samples=samples)
    out=ROOT/'outputs/protected_reward_floor_cpu'; out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':
    main()
