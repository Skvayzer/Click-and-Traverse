"""Translation-relative response telemetry; no rewards or observation features.

An event starts when either hand is below 20 cm and ends when both exceed
22 cm or the episode ends. Lock the threatened hand and outward SDF normal at
onset. Subtract root translation, NOT rotation, as in the paired poke tool.
Unlike that tool, an online event has no matched no-obstacle control: ordinary
arm swing and commanded root progress are included, not causal attribution.
"""
import torch

PREFIX = 'response_split_'
FIELDS = ('event_count', 'hand_relative_retreat_sum_m', 'root_world_displacement_sum_m',
          'root_retreat_sum_m', 'ratio_sum', 'ratio_valid_count', 'zero_root_count',
          'invalid_event_count')


def initial(root):
    n = len(root)
    return dict(active=torch.zeros(n, dtype=torch.bool, device=root.device),
                hand=torch.zeros(n, dtype=torch.long, device=root.device),
                root=root.clone(), relative=torch.zeros_like(root), normal=torch.zeros_like(root))


def advance(state, hands, root, clearance, normals, done):
    """Return one record per completed event, zeros elsewhere; ratio is signed.

    Ratios with root displacement <=1 mm are undefined and excluded, with an
    explicit zero-root count. Do not turn arm-only responses into huge ratios.
    """
    n = len(root); rows = torch.arange(n, device=root.device)
    finite = (torch.isfinite(hands).all((1, 2)) & torch.isfinite(root).all(-1)
              & torch.isfinite(clearance).all(-1) & torch.isfinite(normals).all((1, 2)))
    start = ~state['active'] & (clearance.amin(-1) < .20) & finite
    hand = clearance.argmin(-1)
    state['hand'] = torch.where(start, hand, state['hand'])
    hand = state['hand']; relative = hands[rows, hand]-root
    normal = normals[rows, hand]
    norm = normal.norm(dim=-1, keepdim=True)
    start &= norm[:, 0] > 1e-6
    for key, value in (('root', root), ('relative', relative), ('normal', normal/norm.clamp_min(1e-6))):
        state[key] = torch.where(start[:, None], value, state[key])
    state['active'] |= start
    finish = state['active'] & ((clearance.amin(-1) >= .22) | done | ~finite)
    valid = finish & finite
    arm = ((relative-state['relative'])*state['normal']).sum(-1)
    displacement = root-state['root']; body = displacement.norm(dim=-1)
    retreat = (displacement*state['normal']).sum(-1)
    ratio_valid = valid & (body > .001)
    values = dict(event_count=finish, hand_relative_retreat_sum_m=torch.where(valid, arm, 0.),
                  root_world_displacement_sum_m=torch.where(valid, body, 0.),
                  root_retreat_sum_m=torch.where(valid, retreat, 0.),
                  ratio_sum=torch.where(ratio_valid, arm/body.clamp_min(.001), 0.),
                  ratio_valid_count=ratio_valid, zero_root_count=valid & ~ratio_valid,
                  invalid_event_count=finish & ~finite)
    state['active'] &= ~finish
    return {k: v.to(root.dtype) for k, v in values.items()}


def summaries(counts):
    result = dict(counts)
    n = counts.get('event_count', 0)-counts.get('invalid_event_count', 0)
    if n > 0:
        for key in ('hand_relative_retreat', 'root_world_displacement', 'root_retreat'):
            result[key+'_mean_m'] = counts[key+'_sum_m']/n
    if counts.get('ratio_valid_count', 0) > 0:
        result['hand_to_root_ratio_mean'] = counts['ratio_sum']/counts['ratio_valid_count']
    return result
