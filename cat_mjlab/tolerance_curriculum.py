"""Opt-in metric ladder; strict 5 cm telemetry/rewards remain independent.

Each role needs a full rolling window of unseeded leader outcomes. Episode
rungs freeze at reset; stale completions cannot advance a newer rung.
"""
import torch

RUNGS = (.20, .15, .10, .07, .05)


def configure(config, *, enabled=False, window=256):
    if type(window) is not int or window < 1:
        raise ValueError('Tolerance curriculum window must be a positive integer')
    if enabled and not config.get('wholebody_hand_contrast'):
        raise ValueError('Tolerance curriculum requires hand contrast')
    # Explicit opt-in on every fresh launch, even with an inherited checkpoint.
    config.pop('hand_tolerance_curriculum', None)
    if enabled:
        from .config import hand_curriculum_threshold
        config['hand_tolerance_curriculum'] = dict(rungs=list(RUNGS), window=window,
            threshold=hand_curriculum_threshold(config))
    return config


def initialize(settings, device):
    w = settings['window']
    return dict(stage=torch.zeros((), dtype=torch.long, device=device),
        history=torch.zeros((2, w), dtype=torch.bool, device=device),
        count=torch.zeros(2, dtype=torch.long, device=device),
        completed=torch.zeros((5, 2), dtype=torch.long, device=device),
        successes=torch.zeros((5, 2), dtype=torch.long, device=device),
        rates=torch.zeros((5, 2), device=device))


def advance(state, settings, *, roles, episode_rungs, eligible, success):
    stage = int(state['stage'])
    # Called only at first resolved outcome. Deterministic environment-index
    # order breaks simultaneous completion ties; no policy RNG is consumed.
    for column, role in enumerate((1, 3)):
        values = success[eligible & (roles == role) & (episode_rungs == stage)]
        n = values.numel()
        if not n:
            continue
        old = int(state['count'][column])
        state['completed'][stage, column] += n
        state['successes'][stage, column] += values.long().sum()
        w = settings['window']
        kept = values[-w:]
        slots = (torch.arange(len(kept), device=values.device) + old + n - len(kept)) % w
        state['history'][column, slots] = kept
        state['count'][column] += n
        state['rates'][stage, column] = state['history'][column].float().sum() / min(old+n, w)
    if (stage < len(RUNGS)-1 and bool((state['count'] >= settings['window']).all())
            and bool((state['rates'][stage] >= settings['threshold']).all())):
        state['stage'] += 1
        state['count'].zero_()
        state['history'].zero_()


def metrics(state):
    stage = int(state['stage'])
    result = {'hand_tolerance/rung': stage, 'hand_tolerance/tolerance_m': RUNGS[stage]}
    for rung in range(len(RUNGS)):
        for col, role in enumerate(('protected', 'transition')):
            prefix = f'hand_tolerance/rung_{rung}/{role}'
            result[prefix+'/completed'] = int(state['completed'][rung, col])
            result[prefix+'/successes'] = int(state['successes'][rung, col])
            result[prefix+'/rolling_success'] = float(state['rates'][rung, col])
    return result
