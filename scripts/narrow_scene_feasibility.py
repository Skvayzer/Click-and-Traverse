#!/usr/bin/env python3
"""Which CAT-type scenes with narrow openings can be passed WITHOUT touching, at least geometrically?

Per scene: the 2-D footprint of everything occupied at 0.25-0.95 m (the blocking band of cat_mjlab/blocking.py:
hurdles are stepped over, beams ducked under), its distance transform, and a grid search from the start region to
the goal plane through cells farther than R from any obstacle -- R = half the robot's SIDEWAYS depth (0.236 m
collision proxy, standing) + 1 cm. A scene without such a path cannot be crossed cleanly by any posture we
train, so a narrow-passage specialist should not be graded (or taught) on it.
Writes JSON {scene_id: {narrowest_m, feasible}} and prints a summary per skill tag.
usage: narrow_scene_feasibility.py BANK_MANIFEST OUTPUT_JSON
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = .236 / 2 + .01


def main():
    from scipy import ndimage
    from cat_ppo.furniture.generalist_fields import scene_directory, load_generalist_manifest
    import cat_mjlab.packing.field_packing as fp
    mp = Path(sys.argv[1]).resolve(); m = load_generalist_manifest(mp, verify_files=False)
    out = {}
    for r in m['scenes']:
        if r.get('task_kind') != 'cat':
            continue
        d = scene_directory(m, mp, r); shape = r['shape']; o = np.asarray(r['origin']); dx = r['dx']
        sdf = np.asarray(fp.unpack_scalar(np.load(d / 'sdf.npy')), dtype=np.float32).reshape(shape)
        z = o[2] + dx * np.arange(shape[2]); k = (z >= .25) & (z <= .95)          # the blocking band: hurdles (<= 0.20 m) are stepped over, beams (>= 1.0 m) ducked under
        blocked = (sdf[:, :, k] <= 0).any(-1)
        free = ndimage.distance_transform_edt(~blocked) * dx > R
        xs = o[0] + dx * np.arange(shape[0])
        start = np.zeros_like(free); i0 = int(np.argmin(abs(xs - 0.)))
        start[max(0, i0 - 3):i0 + 4, :] = True                      # start line region around x = 0
        goal_i = int(np.argmin(abs(xs - r['goal'][0])))
        seen = np.zeros_like(free); q = deque(map(tuple, np.argwhere(start & free)))
        for c in q: seen[c] = True
        feasible = False
        while q:
            a, b = q.popleft()
            if a >= goal_i:
                feasible = True; break
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                na, nb = a + da, b + db
                if 0 <= na < free.shape[0] and 0 <= nb < free.shape[1] and free[na, nb] and not seen[na, nb]:
                    seen[na, nb] = True; q.append((na, nb))
        widths = []
        for column in blocked:
            if column.any() and not column.all():
                run = best = 0
                for v in ~column:
                    run = run + 1 if v else 0; best = max(best, run)
                widths.append(best * dx)
        out[r['scene_id']] = dict(narrowest_m=round(min(widths), 3) if widths else None, feasible=feasible, family=r['family'])
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
    rows = list(out.values())
    for fam in sorted({v['family'] for v in rows}):
        sel = [v for v in rows if v['family'] == fam]
        narrow = [v for v in sel if v['narrowest_m'] is not None and v['narrowest_m'] < .77]
        print(f"{fam:15s} scenes {len(sel):4d}  narrow(<0.77 m) {len(narrow):4d}  of which feasible {sum(v['feasible'] for v in narrow):4d}"
              f"  | <0.45 m: {sum(v['narrowest_m'] < .45 for v in narrow):4d}, feasible {sum(v['feasible'] for v in narrow if v['narrowest_m'] < .45):4d}")


if __name__ == '__main__':
    main()
