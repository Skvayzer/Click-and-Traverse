"""Optional per-zone reward metadata, without rewriting immutable field banks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


LEGACY_SDF_KNEE = .05


def passage_parameters(scenes, *, bank_path, override_path=None):
    """Read inline zone parameters or a bank-pinned scene/zone metadata overlay.

    Only certified narrow modules may override the knee or heading. The existing
    field/collision/reset manifests remain immutable and keep their hashes.
    """
    overrides = {}
    if override_path is not None:
        document = json.loads(Path(override_path).read_text())
        if (document.get('schema') != 'cat-passage-rewards-v1'
                or document.get('bank_sha256') != hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()):
            raise ValueError('Passage reward metadata schema/bank hash differs')
        overrides = document['scenes']
        if not isinstance(overrides, dict):
            raise ValueError('Passage reward scenes must be a mapping')
    known = {s['scene_id'] for s in scenes if s is not None}
    if set(overrides) - known:
        raise ValueError('Unknown scene in passage reward metadata')
    result = np.full((len(scenes), 6), LEGACY_SDF_KNEE, dtype=np.float32)
    heading = {key: np.zeros_like(result) for key in ("heading_target_rad", "heading_weight")}
    heading["heading_axis"] = np.zeros(result.shape, dtype=bool)
    heading["heading_override"] = np.zeros(result.shape, dtype=bool)
    for row, scene in enumerate(scenes):
        if scene is None:
            continue
        contrast = scene.get('hand_contrast')
        entries = overrides.get(scene['scene_id'], {})
        zones = [] if contrast is None else contrast['zones']
        if not isinstance(entries, dict) or set(entries) - {str(i) for i in range(len(zones))}:
            raise ValueError('Unknown zone in passage reward metadata')
        for index, zone in enumerate(zones):
            entry = entries.get(str(index), {})
            if not isinstance(entry, dict) or set(entry) - {'sdf_reward_knee', 'heading_target_rad', 'heading_weight', 'heading_axis'}:
                raise ValueError('Unknown passage reward parameter')
            knee = entry.get('sdf_reward_knee', zone.get('sdf_reward_knee', LEGACY_SDF_KNEE))
            if isinstance(knee, bool) or not isinstance(knee, (int, float)) or not math.isfinite(knee) or not 0 <= knee <= LEGACY_SDF_KNEE:
                raise ValueError('SDF reward knee must be finite and within [0, .05]')
            heading_keys = {'heading_target_rad', 'heading_weight', 'heading_axis'}
            values = {key: entry.get(key, zone.get(key)) for key in heading_keys}
            specified = any(value is not None for value in values.values())
            if specified:
                if any(value is None for value in values.values()):
                    raise ValueError('Heading target, weight and axis must be specified together')
                for key in ('heading_target_rad', 'heading_weight'):
                    value = values[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError('Heading angle and weight must be finite numbers')
                if values['heading_weight'] < 0 or type(values['heading_axis']) is not bool:
                    raise ValueError('Heading weight must be nonnegative and axis must be boolean')
                for key, value in values.items():
                    heading[key][row, index] = value
                heading['heading_override'][row, index] = True
            if knee != LEGACY_SDF_KNEE or specified:
                from cat_ppo.furniture.room_navigation import scene_navigation_radius
                scene_navigation_radius(scene)  # Geometry-bound certificate validation.
                modules = contrast.get('modules', [])
                if (len(modules) != len(zones) or modules[index]['role'] != 'narrow'
                        or zone['fade_m'] <= 0):
                    raise ValueError('Passage reward override requires a certified narrow module with a fade')
            result[row, index] = knee
    return dict(sdf_reward_knee=result, **heading)


def passage_knees(scenes, *, bank_path, override_path=None):
    """Compatibility accessor for knee-only consumers."""
    return passage_parameters(scenes, bank_path=bank_path, override_path=override_path)['sdf_reward_knee']


def blended_sdf_knee(metadata, scene_ids, context):
    target = metadata['sdf_reward_knee'][scene_ids, context['zone_index']]
    u = context['phase_weight']
    blend = u.square() * (3 - 2 * u)  # C1 boundary ramp; zero outside the zone.
    return LEGACY_SDF_KNEE + blend * (target - LEGACY_SDF_KNEE)
