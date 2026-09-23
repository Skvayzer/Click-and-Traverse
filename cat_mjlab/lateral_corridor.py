"""Route-anchored hand protection and conservative whole-arm swept clearance."""
import math
import torch

MARGIN = .07899
SCALE = .10


def scene_faces(room):
    """Intersection of actual parallel cabinet openings, never nominal module width."""
    import numpy as np
    route = np.asarray(room['route'], dtype=float)
    t = route[1] - route[0]; t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0]])
    left, right = [], []
    for b in room['boxes']:
        if b['category'] != 'cabinet': continue
        c, s = np.cos(b['yaw']), np.sin(b['yaw'])
        axes = np.array([[c, -s], [s, c]])
        if abs(axes[:, 0] @ n) > 1e-7:
            raise ValueError('Corridor requires parallel cabinets')
        y = (np.asarray(b['center'][:2])-route[0]) @ n
        h = np.asarray(b['half_size'][:2]) @ np.abs(axes.T @ n)
        (left if y < 0 else right).append(y+h if y < 0 else y-h)
    if not left or not right: raise ValueError('Corridor requires both cabinet sides')
    lower, upper = max(left), min(right)
    if not lower < 0 < upper: raise ValueError('Invalid corridor faces')
    return lower, upper


def lateral(world, origin, tangent):
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
    return ((world[..., :2]-origin[:, None, :])*normal[:, None, :]).sum(-1)


def corridor_cost(y, bound):
    """Mean squared one-sided excess; C1 at boundary, no saturation."""
    return ((y.abs()-bound[:, None]).clamp_min(0)/SCALE).square().mean(-1)


def arm_clearance(endpoints, radii, origin, tangent, lower, upper):
    y = lateral(endpoints.flatten(1, 2), origin, tangent).reshape(endpoints.shape[:3])
    return (torch.minimum(y.amin(-1)-lower[:, None], upper[:, None]-y.amax(-1))-radii).amin(-1)


def combined_pose(y, bound, heading_good, clearance):
    return (torch.isfinite(y).all(-1) & (y.abs() <= bound[:, None]).all(-1)
            & heading_good.bool() & torch.isfinite(clearance) & (clearance >= MARGIN))


class ArmGeometry:
    def __init__(self, model, device):
        import json
        from .collision import PROPOSAL, compile_proposal
        proposal = json.loads(PROPOSAL.read_text())
        proposal['shapes'] = [s for s in proposal['shapes'] if s['group'] == 'arms']
        if not proposal['shapes'] or any(s['kind'] != 'capsule' for s in proposal['shapes']):
            raise ValueError('Whole-arm corridor needs enclosing capsules')
        self.compiled = compile_proposal(proposal, model, device)

    def clearance(self, data, geometry):
        from .collision import transform_primitives
        ep = transform_primitives(data.xpos, data.xmat, self.compiled)['endpoints']
        return arm_clearance(ep, self.compiled['radii'], geometry['origin'], geometry['tangent'],
                             geometry['lower'], geometry['upper'])
