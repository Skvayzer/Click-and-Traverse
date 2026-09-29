"""Context-gated adversarial motion priors (AMP with T-GMP-style context dependence).

Style groups (one discriminator each), from data/motion_library/library_v1:
  0 locomotion  everyday human motion (walk, stand, turn, carry)   -- fallback, always gets the rest
  1 sidle       sidestepping                                      -- gate: narrow lateral gap
  2 duck_step   ducking under / stepping over                     -- gate: head field points down, or feet field up
(3 crawl and 4 protect exist in the library but are disabled here.)

Per control step each environment gets gate weights w_g(context) summing to 1 and a style reward
    r_style = sum_g w_g * max(0, 1 - 0.25 (D_g(f_t, f_t+1) - 1)^2)          in [0, 1]
where f is the 89-D heading-frame feature vector defined in scripts/motion_library/build_library.py.
Each D_g is trained (least-squares GAN + gradient penalty) on library transitions of its group vs
robot transitions collected while that group's gate was open.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

GROUPS = ("locomotion", "sidle", "duck_step")
KEY_SITES = ("head", "left_palm", "right_palm")
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link")
FEATURE_DIM = 89


def robot_features(qpos, qvel, site_xpos, body_xpos, pelvis_xmat):
    """89-D features from simulator state, same layout as the library.

    qpos [N,36], qvel [N,35], site_xpos [N,3,3] (KEY_SITES), body_xpos [N,4,3] (KEY_BODIES),
    pelvis_xmat [N,3,3]. MuJoCo free-joint qvel: linear velocity in the world frame, angular
    velocity in the body frame.
    """
    root = qpos[:, :3]; q = qpos[:, 3:7]
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s = yaw.cos(), yaw.sin()
    def heading(v):          # world -> heading frame, v [...,3]
        x, y = v[..., 0], v[..., 1]
        cc, ss = c.view(-1, *([1] * (v.dim() - 2))), s.view(-1, *([1] * (v.dim() - 2)))
        return torch.stack((cc * x + ss * y, -ss * x + cc * y, v[..., 2]), -1)
    gravity = -pelvis_xmat[:, 2, :]                                  # R^T (0,0,-1)
    keys = torch.cat((site_xpos, body_xpos), 1) - root[:, None]
    return torch.cat((root[:, 2:3], gravity, heading(qvel[:, :3]), qvel[:, 3:6], qpos[:, 7:36], qvel[:, 6:35],
                      heading(keys).reshape(len(qpos), -1)), -1)


def gate_weights(heading_sdf, head_gf_z, feet_gf_z_max, *, gap_open=.75, gap_tight=.55, half_width=.16):
    """[N,3] weights for (locomotion, sidle, duck_step), summing to 1.

    heading_sdf [N,4]: distance-field samples at the counterfactual shoulder probes
    (+normal, -normal, ahead+normal, ahead-normal) used by the heading reward.
    """
    left = torch.minimum(heading_sdf[:, 0], heading_sdf[:, 2]); right = torch.minimum(heading_sdf[:, 1], heading_sdf[:, 3])
    gap = 2 * half_width + left.clamp_min(0) + right.clamp_min(0)
    w_sidle = ((gap_open - gap) / (gap_open - gap_tight)).clamp(0, 1)
    w_duck = ((-head_gf_z - .1) / .3).clamp(0, 1)
    w_step = ((feet_gf_z_max - .1) / .3).clamp(0, 1)
    w_ds = torch.maximum(w_duck, w_step)
    total = (w_sidle + w_ds).clamp_min(1.)                            # obstacle groups never exceed 1 together
    w_sidle, w_ds = w_sidle / total, w_ds / total
    return torch.stack((1 - w_sidle - w_ds, w_sidle, w_ds), -1), gap


class Discriminator(nn.Module):
    def __init__(self, dim=2 * FEATURE_DIM, hidden=(1024, 512)):
        super().__init__()
        layers, last = [], dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]; last = h
        self.body = nn.Sequential(*layers); self.head = nn.Linear(last, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class StylePrior:
    def __init__(self, library, *, device, lr=5e-5, grad_penalty=5., replay_size=500_000, batch=4096, steps=8,
                 store_fraction=.1, min_gate=.3):
        library = Path(library)
        data = __import__("numpy").load(str(library) + ".npz")
        meta = json.loads(Path(str(library) + ".json").read_text())
        names = meta["summary"]["groups"]
        self.device = torch.device(device)
        feats = torch.as_tensor(data["features"].astype("float32"), device=self.device)
        next_ok = torch.as_tensor(data["next_ok"], device=self.device)
        group = torch.as_tensor(data["group"].astype("int64"), device=self.device)
        # Fixed normalisation from the library, applied identically to robot and human features.
        self.mean = feats.mean(0); self.std = feats.std(0).clamp_min(1e-3)
        self.features = (feats - self.mean) / self.std
        self.human_index = []
        for g in GROUPS:
            idx = torch.nonzero(next_ok & (group == names.index(g))).flatten()
            if not len(idx):
                raise ValueError(f"motion library has no transitions for style group {g}")
            self.human_index.append(idx)
        self.minutes = {g: round(len(i) * .02 / 60, 1) for g, i in zip(GROUPS, self.human_index)}
        self.nets = nn.ModuleList([Discriminator() for _ in GROUPS]).to(self.device)
        self.opt = torch.optim.Adam(self.nets.parameters(), lr=lr, betas=(.9, .999))
        self.gp, self.batch, self.steps = grad_penalty, batch, steps
        self.store_fraction, self.min_gate = store_fraction, min_gate
        self.replay = [torch.zeros((replay_size, 2 * FEATURE_DIM), device=self.device) for _ in GROUPS]
        self.filled = [0] * len(GROUPS); self.cursor = [0] * len(GROUPS)

    def normalize(self, f):
        return (f - self.mean) / self.std

    @torch.no_grad()
    def reward(self, f_now, f_next, weights):
        x = torch.cat((self.normalize(f_now), self.normalize(f_next)), -1)
        per = torch.stack([((1 - .25 * (net(x) - 1).square()).clamp_min(0)) for net in self.nets], -1)
        return (weights * per).sum(-1), per

    @torch.no_grad()
    def store(self, f_now, f_next, weights, generator=None):
        x = torch.cat((self.normalize(f_now), self.normalize(f_next)), -1)
        keep = torch.rand(len(x), device=self.device, generator=generator) < self.store_fraction
        for g in range(len(GROUPS)):
            rows = x[keep & (weights[:, g] >= self.min_gate)]
            n = len(rows); cap = len(self.replay[g])
            if not n:
                continue
            n = min(n, cap); idx = (self.cursor[g] + torch.arange(n, device=self.device)) % cap
            self.replay[g][idx] = rows[:n]
            self.cursor[g] = int((self.cursor[g] + n) % cap); self.filled[g] = min(cap, self.filled[g] + n)

    def update(self):
        stats = {}
        for g, net in enumerate(self.nets):
            if self.filled[g] < self.batch:
                continue
            losses = []
            for _ in range(self.steps):
                hi = self.human_index[g][torch.randint(len(self.human_index[g]), (self.batch,), device=self.device)]
                human = torch.cat((self.features[hi], self.features[hi + 1]), -1).requires_grad_(True)
                robot = self.replay[g][torch.randint(self.filled[g], (self.batch,), device=self.device)]
                d_h, d_r = net(human), net(robot)
                grad = torch.autograd.grad(d_h.sum(), human, create_graph=True)[0]
                loss = .5 * (d_h - 1).square().mean() + .5 * (d_r + 1).square().mean() + self.gp * grad.square().sum(-1).mean()
                self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
                losses.append(float(loss))
            name = GROUPS[g]
            stats.update({f"style/{name}/d_loss": sum(losses) / len(losses), f"style/{name}/d_human": float(d_h.mean()),
                          f"style/{name}/d_robot": float(d_r.mean()), f"style/{name}/replay": self.filled[g]})
        return stats

    def state_dict(self):
        return dict(nets=self.nets.state_dict(), opt=self.opt.state_dict())

    def load_state_dict(self, state):
        self.nets.load_state_dict(state["nets"]); self.opt.load_state_dict(state["opt"])


class StyleSchedule:
    """lambda = 0 for `warmup` updates, then linear ramp to `target` over `ramp` updates. The guard
    freezes the ramp (and backs off 10%) while any watched success metric is more than `tolerance`
    below its value when the ramp started."""

    def __init__(self, target=.3, warmup=20, ramp=50, tolerance=.05,
                 watch=("progress/p3_cat_navigation_success", "progress/narrow_zone_success",
                        "progress/protected_zone_success", "progress/clutter_room_success")):
        self.target, self.warmup, self.ramp, self.tolerance, self.watch = target, warmup, ramp, tolerance, watch
        self.value, self.update, self.baseline = 0., 0, {}

    def step(self, metrics):
        self.update += 1
        if self.update == self.warmup:
            self.baseline = {k: metrics[k] for k in self.watch if k in metrics}
        held = [k for k, v in self.baseline.items() if k in metrics and metrics[k] < v - self.tolerance]
        if self.update > self.warmup:
            if held:
                self.value = max(0., self.value * .9)
            else:
                self.value = min(self.target, self.value + self.target / max(self.ramp, 1))
        return self.value, held

    def state_dict(self):
        return dict(value=self.value, update=self.update, baseline=self.baseline)

    def load_state_dict(self, s):
        self.value, self.update, self.baseline = s["value"], s["update"], s["baseline"]
