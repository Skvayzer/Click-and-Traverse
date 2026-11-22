#!/usr/bin/env python3
"""CPU replay and constrained inverse-dynamics references for G1 standing threats."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cat_mjlab.energy import DEFAULT_ALPHA, REFERENCE_PEAK_W, motor_spec
from cat_mjlab.model import assemble_training_xml
from cat_mjlab.sim_cpu import CPUSimulation, INPUT_FIELDS, VIEW_FIELDS
from cat_ppo.furniture.control import JOINT_NAMES
from eval_standing_hands import evaluate
from standing_event_energy import StandingEventEnergy

BUCKETS = ("danger", "anticipation")
STRATEGIES = ("arm", "body", "step")
MODELS = ("mech", "heat")
HAND_RADIUS = .10344617267582552


def segment_distance(point, start, end):
    delta = end - start
    fraction = np.clip(np.sum((point - start) * delta, axis=-1) /
                       np.maximum(np.sum(delta ** 2, axis=-1), 1.e-12), 0., 1.)
    return np.linalg.norm(point - start - np.expand_dims(fraction, -1) * delta, axis=-1)


def minimum_jerk(t):
    t = np.clip(t, 0., 1.)
    return t**3 * (10. + t * (-15. + 6. * t))


def integrate_cost(torque, velocity, time, limits, weights):
    work = torque * velocity
    mech = np.abs(work).sum(-1)
    heat = np.maximum(work.sum(-1), 0.) + DEFAULT_ALPHA * ((torque / limits)**2 * weights).sum(-1)
    dt = np.diff(time)
    return {name: float(np.sum(.5 * (power[:-1] + power[1:]) * dt))
            for name, power in zip(MODELS, (mech, heat))}


class BatchedReplayCPU(CPUSimulation):
    """Mirror the existing CPU MuJoCo bridge in a small, shared scene bank batch."""
    def __init__(self, num_envs, **kwargs):
        super().__init__(num_envs=1, **kwargs)
        self.num_envs = num_envs
        self.worlds = [self.raw] + [mujoco.MjData(self.model) for _ in range(num_envs - 1)]
        for key, value in vars(self.data).items():
            setattr(self.data, key, value.repeat((num_envs,) + (1,) * (value.ndim - 1)))
        self._views = {key: value.numpy() for key, value in vars(self.data).items()}
        self._final = SimpleNamespace(xpos=torch.empty_like(self.data.xpos), xmat=torch.empty_like(self.data.xmat))
        self.final_xyz, self.final_mat = self._final.xpos.numpy(), self._final.xmat.numpy()

    def _apply(self, operation, env_ids=None, push=True):
        ids = range(self.num_envs) if env_ids is None else env_ids.tolist()
        for i in ids:
            world = self.worlds[i]
            if push:
                for key in INPUT_FIELDS:
                    getattr(world, key)[:] = self._views[key][i]
            operation(self.model, world)
            for key in VIEW_FIELDS + ("xmat", "site_xmat"):
                self._views[key][i].flat = getattr(world, key).flat
            self._views["time"][i] = world.time
        self._contact_cache.clear()

    def step(self):
        self._apply(mujoco.mj_step)

    def forward(self, env_ids=None):
        self._apply(mujoco.mj_forward, env_ids)

    def reset_data(self, env_ids=None):
        self._apply(mujoco.mj_resetData, env_ids, push=False)

    def final_collision_data(self):
        for i, world in enumerate(self.worlds):
            self._shadow.qpos[:] = self._views["qpos"][i]
            self._shadow.mocap_pos[:] = world.mocap_pos
            self._shadow.mocap_quat[:] = world.mocap_quat
            mujoco.mj_kinematics(self.model, self._shadow)
            self.final_xyz[i] = self._shadow.xpos
            self.final_mat[i].flat = self._shadow.xmat.flat
        return self._final

    def contact_flags(self, pairs):
        key = id(pairs)
        if key not in self._contact_cache:
            rows = []
            pairs = [tuple(sorted(pair)) for pair in pairs.tolist()]
            for world in self.worlds:
                contact = world.contact
                touching = {tuple(sorted((a, b))) for a, b, dist in zip(contact.geom1, contact.geom2, contact.dist)
                            if dist < 0.}
                rows.append([pair in touching for pair in pairs])
            self._contact_cache[key] = torch.tensor(rows)
        return self._contact_cache[key]

    def capacity_report(self):
        return dict(contacts=max(world.ncon for world in self.worlds), overflow_worlds=0)


class ReferenceTargets(StandingEventEnergy):
    def __init__(self, task):
        super().__init__(task)
        self.qpos = task.data.qpos
        self.initial = torch.zeros_like(self.qpos)
        self.peak = self.qpos.new_zeros(task.num_envs)
        self.endpoint = self.qpos.new_zeros((task.num_envs, 3))
        self.snapshot = self.qpos.new_zeros((task.num_envs, self.qpos.shape[-1] + 14))
        self.captured = torch.zeros_like(self.open)
        self.references = []

    def probe(self, ids, hand, hands, start, end, sizes, kind, reversing):
        active = self.open[ids]
        begun = active & (self.steps[ids] == 0)
        self.initial[ids] = torch.where(begun[:, None], self.qpos[ids], self.initial[ids])
        self.captured[ids] &= ~begun
        self.peak[ids] = torch.where(begun, -torch.inf, self.peak[ids])
        point = hands[self.ids[:len(ids)], hand]
        delta = end - start
        u = (((point - start) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1.e-12)).clamp(0., 1.)
        radius = torch.where(kind == 0, sizes[:, 0],
                             torch.where(kind == 1, sizes.norm(dim=-1), sizes[:, :2].norm(dim=-1)))
        gap = (point - start - u[:, None] * delta).norm(dim=-1) - radius - HAND_RADIUS
        improving = active & ~self.captured[ids] & (gap > self.peak[ids])
        self.peak[ids] = torch.where(improving, gap, self.peak[ids])
        self.endpoint[ids] = torch.where(improving[:, None], point, self.endpoint[ids])
        close = active & reversing & ~self.captured[ids] & (self.steps[ids] > 0)
        idx = ids[close]
        self.captured[idx] = True
        self.snapshot[idx] = torch.cat((self.initial[idx], start[close], end[close],
                                        radius[close, None], self.endpoint[idx], self.peak[idx, None],
                                        self.energy[idx, 0].sum(-1), self.steps[idx, None] * self.sim_dt), dim=1)

    def finish(self, ids, bucket, hand, success):
        self.references.append((self.captured[ids].clone(), self.snapshot[ids].clone()))
        self.captured[ids] = False
        super().finish(ids, bucket, hand, success)

    def summary(self, rows, buckets):
        by_class, details = super().summary(rows, buckets)
        i = 0
        for flags, vectors in self.references:
            for valid, v in zip(flags.tolist(), vectors.tolist()):
                if valid:
                    details[i]["reference"] = dict(qpos=v[:36], sweep_start=v[36:39], sweep_end=v[39:42],
                                                   radius=v[42], achieved_point=v[43:46], achieved_gap=v[46],
                                                   policy_j=dict(zip(MODELS, v[47:49])), seconds=v[49])
                i += 1
        return by_class, details


class InverseReference:
    def __init__(self, samples):
        tree = ET.fromstring(assemble_training_xml())
        world = tree.find("worldbody")
        equality = tree.find("equality")
        if equality is None:
            equality = ET.SubElement(tree, "equality")
        for side in ("left", "right"):
            ET.SubElement(world, "site", name=f"{side}_weld_target", pos="0 0 0", size=".001")
            ET.SubElement(equality, "weld", name=f"{side}_foot_weld", site1=f"{side}_foot",
                          site2=f"{side}_weld_target", active="false")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        self.samples = samples
        self.feet = [self.model.site(f"{s}_foot").id for s in ("left", "right")]
        self.palms = [self.model.site(f"{s}_palm").id for s in ("left", "right")]
        self.anchors = [self.model.site(f"{s}_weld_target").id for s in ("left", "right")]
        # Moving a world site at runtime disables the compiler's identity-frame shortcut.
        self.model.site_sameframe[self.anchors] = 0
        self.welds = [self.model.equality(f"{s}_foot_weld").id for s in ("left", "right")]
        self.root = self.model.body("pelvis").id
        specs = np.array([(tau, speed) for _, tau, speed in map(motor_spec, JOINT_NAMES)])
        self.limits, self.speed = specs.T
        self.weights = self.limits * self.speed / REFERENCE_PEAK_W
        self.joint_range = self.model.jnt_range[1:].copy()
        self.leg_range = self.joint_range[:12].T
        self.waist_range = self.joint_range[12:15].T

    def kine(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)

    def frames(self):
        return self.data.site_xpos[self.feet].copy(), self.data.site_xmat[self.feet].reshape(2, 3, 3).copy()

    def foot_error(self, target, orient):
        pos, rot = self.frames()
        return np.r_[(pos - target).ravel(), (.12 * (rot - orient)).ravel()]

    def roots(self, q0, delta):
        q = q0.copy()
        q[:3] += delta[:3]
        initial = Rotation.from_quat(q0[3:7], scalar_first=True)
        q[3:7] = (Rotation.from_rotvec(delta[3:6]) * initial).as_quat(scalar_first=True)
        q[19:22] = delta[6:]
        return q

    def leg_ik(self, q, foot_pos, foot_rot, initial):
        def error(angles):
            q[7:19] = angles
            self.kine(q)
            return np.r_[self.foot_error(foot_pos, foot_rot), .002 * (angles - initial)]
        fitted = least_squares(error, np.clip(initial, *self.leg_range), bounds=self.leg_range,
                               max_nfev=24, ftol=1.e-5, xtol=1.e-5)
        q[7:19] = fitted.x
        residual = np.abs(error(fitted.x))
        if residual[:6].max() > .004 or residual[6:24].max() > .005:
            return None
        return q

    def arm_paths(self, q0, hand, targets, fraction):
        arm = slice(22 + 7*hand, 29 + 7*hand)
        bounds = self.joint_range[15 + 7*hand:22 + 7*hand].T
        for target in targets:
            for regularizer in (.002, .018):
                q = q0.copy()
                def error(angles):
                    q[arm] = angles
                    self.kine(q)
                    return np.r_[self.data.site_xpos[self.palms[hand]] - target,
                                 regularizer*(angles-q0[arm])]
                fit = least_squares(error, np.clip(q0[arm], *bounds), bounds=bounds, max_nfev=36)
                q[arm] = fit.x
                yield q0 + fraction[:, None] * (q - q0), None

    def body_paths(self, q0, hand, targets, fraction, feet, orient):
        lower = np.r_[[-.14, -.14, -.10, -.35, -.35, -.55], self.waist_range[0], self.leg_range[0]]
        upper = np.r_[[ .14,  .14,  .09,  .35,  .35,  .55], self.waist_range[1], self.leg_range[1]]
        base = np.r_[np.zeros(6), q0[19:22], q0[7:19]]
        for target in targets:
            def error(v):
                q = self.roots(q0, v[:9])
                q[7:19] = v[9:]
                self.kine(q)
                return np.r_[6. * self.foot_error(feet, orient), self.data.site_xpos[self.palms[hand]] - target,
                             .012 * (v-base)]
            fit = least_squares(error, np.clip(base, lower, upper), bounds=(lower,upper), max_nfev=40)
            path = [q0]
            for f in fraction[1:]:
                q = self.leg_ik(self.roots(q0, base[:9] + f*(fit.x[:9]-base[:9])), feet, orient, path[-1][7:19])
                if q is None:
                    break
                path.append(q)
            if len(path) == len(fraction):
                yield np.asarray(path), None

    def step_paths(self, q0, times, feet, orient):
        lateral = Rotation.from_quat(q0[3:7], scalar_first=True).as_matrix()[:,1].copy()
        lateral[2] = 0.
        lateral /= max(np.linalg.norm(lateral), 1.e-12)
        for swing, direction in ((0, 1.), (1, -1.)):
            stance = 1-swing
            for metres in (.12, .22, .30):
                shift = metres * direction * lateral
                stance_shift = .70*(feet[stance,:2]-q0[:2])
                path = [q0]
                for t in times[1:]:
                    f = minimum_jerk((t-.22)/.55)
                    q = q0.copy()
                    q[:2] += stance_shift * (minimum_jerk(t/.22)-minimum_jerk((t-.75)/.25))
                    q[:2] += .65 * shift[:2] * minimum_jerk((t-.73)/.27)
                    target = feet.copy()
                    target[swing] += shift*f
                    phase = np.clip((t-.22)/.55,0.,1.)
                    target[swing,2] += .06*np.sin(np.pi*phase)**2
                    q = self.leg_ik(q, target, orient, path[-1][7:19])
                    if q is None:
                        break
                    path.append(q)
                if len(path) == len(times):
                    support = np.ones((len(times),2), dtype=bool)
                    support[(times > .22) & (times < .77), swing] = False
                    yield np.asarray(path), support

    def inverse(self, path, seconds, support):
        m, d = self.model, self.data
        n = len(path)
        time = np.linspace(0., seconds, n)
        dt = time[1]
        velocity = np.zeros((n,m.nv))
        for i in range(1,n-1):
            mujoco.mj_differentiatePos(m,velocity[i],2*dt,path[i-1],path[i+1])
        if np.any(np.abs(velocity[:,6:]) > self.speed):
            return None
        acceleration = np.gradient(velocity,dt,axis=0)
        torques = np.zeros((n,m.nu))
        if support is None:
            support = np.ones((n,2),dtype=bool)
        jac_pos, jac_rot = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        for i in range(n):
            self.kine(path[i])
            feet, rotations = self.frames()
            active = support[i]
            # Site weld anchors stay at their stance targets. Swing feet have no support wrench.
            if i == 0:
                anchors, orientations = feet.copy(), rotations.copy()
            if i > 0:
                landed = active & ~support[i-1]
                anchors[landed], orientations[landed] = feet[landed], rotations[landed]
            m.site_pos[self.anchors] = anchors
            m.site_quat[self.anchors] = Rotation.from_matrix(orientations).as_quat(scalar_first=True)
            d.eq_active[self.welds] = active
            d.qvel[:], d.qacc[:] = velocity[i], acceleration[i]
            mujoco.mj_inverse(m,d)
            total = d.qfrc_inverse + d.qfrc_constraint
            mujoco.mj_jacBody(m,d,jac_pos,jac_rot,self.root)
            root_jac = np.vstack((jac_pos[:,:6],jac_rot[:,:6]))
            wrench = np.linalg.solve(root_jac.T,total[:6])
            force, moment = wrench[:3], wrench[3:] + np.cross(d.xpos[self.root],wrench[:3])
            if not supported_zmp(force,moment,feet[active],rotations[active]):
                return None
            contact_jac = []
            for foot in np.asarray(self.feet)[active]:
                mujoco.mj_jacSite(m,d,jac_pos,jac_rot,int(foot))
                contact_jac.append(np.vstack((jac_pos,jac_rot)))
            contact_jac = np.concatenate(contact_jac)
            # Compensate finite weld compliance with the minimum-norm stance
            # reaction that balances the remaining floating-base wrench.
            correction = np.linalg.lstsq(contact_jac[:,:6].T,d.qfrc_inverse[:6],rcond=None)[0]
            torque = d.qfrc_inverse[6:] - contact_jac[:,6:].T @ correction
            if not np.isfinite(torque).all() or np.any(np.abs(torque)>1.02*self.limits):
                return None
            torques[i] = torque
        return integrate_cost(torques,velocity[:,6:],time,self.limits,self.weights)

    def solve(self, event):
        ref = event["reference"]
        q0 = np.asarray(ref["qpos"])
        start,end = np.asarray(ref["sweep_start"]),np.asarray(ref["sweep_end"])
        radius = ref["radius"]+HAND_RADIUS
        hand = int(event["hand_side"]=="right")
        self.kine(q0)
        hand0 = self.data.site_xpos[self.palms[hand]].copy()
        feet, orient = self.frames()
        initial = segment_distance(hand0,start,end)-radius
        achieved = ref["achieved_gap"]
        goal = achieved if achieved >= initial+.001 else max(achieved,.10)
        point = np.asarray(ref["achieved_point"])
        axis = end-start
        center = start + axis*np.clip(np.dot(hand0-start,axis)/max(np.dot(axis,axis),1.e-12),0.,1.)
        radial = hand0-center
        radial /= max(np.linalg.norm(radial),1.e-12)
        targets = (point + radial*max(0.,goal-achieved+.002),
                   center + radial*(radius+goal+.003))
        times = np.linspace(0.,1.,self.samples)
        fraction = minimum_jerk(times)
        hold = np.tile(q0,(self.samples,1))
        strategy_paths = (self.arm_paths(q0,hand,targets,fraction),
                          self.body_paths(q0,hand,targets,fraction,feet,orient),
                          self.step_paths(q0,times,feet,orient))
        result = dict(initial_gap_m=float(initial), target_gap_m=float(goal),
                      achieved_target=bool(goal==achieved), energies={})
        for name, candidates in zip(STRATEGIES,strategy_paths):
            best = {metric: None for metric in MODELS}
            if name != "step" and initial >= goal-.003:
                candidates = ((hold,None),)
            for path, support in candidates:
                self.kine(path[-1])
                position = self.data.site_xpos[self.palms[hand]]
                if segment_distance(position,start,end)-radius < goal-.003:
                    continue
                energy = self.inverse(path,ref["seconds"],support)
                if energy is not None:
                    for metric in MODELS:
                        if best[metric] is None or energy[metric] < best[metric]:
                            best[metric] = energy[metric]
            result["energies"][name] = best
        return result


def supported_zmp(force, moment, feet, rotations):
    if force[2] <= 0. or np.linalg.norm(force[:2]) > .75*force[2]:
        return False
    zmp = np.array([-moment[1],moment[0]])/force[2]
    corners = np.array([[x,y,0.] for x in (-.09,.09) for y in (-.035,.035)])
    polygon = (feet[:,None] + corners @ rotations.transpose(0,2,1))[...,:2].reshape(-1,2)
    hull = ConvexHull(polygon)
    return bool(np.all(hull.equations[:,:2] @ zmp + hull.equations[:,2] <= .003))


_SOLVER = None


def worker_init(samples):
    global _SOLVER
    _SOLVER = InverseReference(samples)


def solve_event(event):
    return _SOLVER.solve(event)


def distribution(values):
    if not values:
        return dict(count=0, median=None, iqr=None)
    a,b,c = np.quantile(values,[.25,.50,.75])
    return dict(count=len(values),median=float(b),iqr=[float(a),float(c)])


def matched_watts(events):
    cells = {}
    for e in events:
        if e["response"] != "still_success" or "reference" not in e:
            continue
        ref = e["reference"]
        cell = cells.setdefault((e["bucket"],e["hand_side"]),np.zeros(3))
        cell += [ref["seconds"],*(ref["policy_j"][m] for m in MODELS)]
    return {key: value[1:]/value[0] for key,value in cells.items() if value[0] > 0.}


def select_events(results, n, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for p, result in enumerate(results):
        baseline = matched_watts(result["approaches"])
        for bucket in BUCKETS:
            matching = [e for e in result["approaches"] if e["bucket"]==bucket and "reference" in e]
            for i in rng.choice(len(matching),size=min(n,len(matching)),replace=False):
                e = matching[int(i)].copy()
                e["policy"] = p
                w = baseline.get((bucket,e["hand_side"]))
                e["policy_extra_j"] = {
                    m: float(e["reference"]["policy_j"][m]-e["reference"]["seconds"]*w[j])
                    for j,m in enumerate(MODELS)
                } if w is not None else None
                selected.append(e)
    return selected


def report(events):
    by_strategy = {}
    for strategy in STRATEGIES:
        energies = {m: [e["optimal"]["energies"][strategy][m] for e in events
                        if e["optimal"]["energies"][strategy][m] is not None] for m in MODELS}
        by_strategy[strategy] = dict(feasible_fraction=len(energies["mech"])/len(events),
                                     **{m: distribution(values) for m,values in energies.items()})
    for event in events:
        r = event["optimal"]
        arm,body = (r["energies"][name] for name in ("arm","body"))
        event["k_phys"] = {m: body[m]/arm[m] if arm[m] and body[m] is not None else None for m in MODELS}
        bonus = event["policy_extra_j"]
        event["efficiency_ratio"] = {
            m: bonus[m]/arm[m] if arm[m] and bonus is not None and r["achieved_target"] and
            event["response"]=="arm_first" else None for m in MODELS}
    return dict(events=len(events), strategies=by_strategy,
                k_phys={m: distribution([e["k_phys"][m] for e in events if e["k_phys"][m] is not None]) for m in MODELS},
                efficiency={p:{m: distribution([e["efficiency_ratio"][m] for e in events if e["policy"]==p and
                                               e["efficiency_ratio"][m] is not None]) for m in MODELS}
                            for p in sorted(set(e["policy"] for e in events))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rows-per-bucket", type=int, default=60)
    p.add_argument("--events-per-policy-bucket", type=int, default=75)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(1)
    existing = json.loads(args.benchmark.read_text())[0]["approaches"]
    rng = np.random.default_rng(0)
    rows = []
    for bucket in BUCKETS:
        ids = sorted({e["row"] for e in existing if e["bucket"]==bucket})
        rows.extend(rng.choice(ids,size=min(len(ids),args.rows_per_bucket),replace=False).tolist())
    rows = sorted(set(rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay = []
    for i, checkpoint in enumerate(args.checkpoint):
        value = evaluate(checkpoint,rows,args.steps,"cpu",False,
                         sim_class=BatchedReplayCPU,observer_class=ReferenceTargets)
        replay.append(value)
        (args.out_dir/f"replay_{i}.json").write_text(json.dumps(value,allow_nan=False))
        print(f"CPU replay {i} complete: {len(value['approaches'])} approaches",flush=True)
    selected = select_events(replay,args.events_per_policy_bucket,seed=0)
    if not selected:
        raise RuntimeError("No complete threatened approaches with a usable deadline")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                             initializer=worker_init,initargs=(args.samples,)) as pool:
        for i,(e,result) in enumerate(zip(selected,pool.map(solve_event,selected,chunksize=2)),1):
            e["optimal"] = result
            if i % 20 == 0:
                print(f"Inverse dynamics: {i}/{len(selected)} events",flush=True)
    summary = report(selected)
    summary["checkpoints"] = args.checkpoint
    summary["rows"] = rows
    (args.out_dir/"reference.json").write_text(json.dumps(dict(summary=summary,events=selected),allow_nan=False))
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
