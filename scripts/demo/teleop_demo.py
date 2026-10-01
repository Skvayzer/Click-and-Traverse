#!/usr/bin/env python3
"""Drive the trained policy yourself: keyboard in the terminal, robot and furniture in the browser.

The policy, observations and obstacle fields are exactly the training ones: the unchanged training
task runs on one world with plain MuJoCo on the CPU (cat_mjlab.sim_cpu). The keyboard replaces the
route with a commanded velocity (task.joystick); nothing steers that command away from obstacles,
so any avoidance you see is the policy's own.

Demo-only differences from training (all documented in the status panel):
  * obstacle contact is checked once per control step, every other step (training: every substep);
  * contact does not end the episode -- it is counted and shown; only a fall resets the robot;
  * no pushes, no observation noise, nominal PD gains; no goal hold, no time limit;
  * rewards are not computed.

Run on dep-1 (it needs the scene bank, ~18 GB RAM), then on the laptop:
    ssh -L 8080:localhost:8080 konstantinsmirnov@dep-1      # browser: http://localhost:8080
Keys (terminal): Up/W go, Down/S stop, Left/A and Right/D turn the commanded direction 20 deg
(hold to keep turning), +/- speed, T drive at the nearest table, R reset, N next scene, Q quit.

Browser keyboard (viser has no key events): open http://localhost:8081 instead of 8080 -- a page that
shows the viser view full-screen and captures the keys itself (forward both ports:
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 konstantinsmirnov@dep-1). There the keys are
hold-to-drive like a joystick: hold Up/W to walk forward, release to stop; hold Left/Right (A/D)
to turn the robot (also in place, by small steps); Space stop, +/- speed, T nearest table, R reset,
N next room.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

PRESETS = {
    "demo": ["table-contrast-train-20260924-open-9e3c0119ba82", "random-furniture-dense-train-004003",
             "random-generic_clutter-dense-train-005001-1eaef5f4115f", "random-furniture-dense-train-20261205-c5b0ae199c87",
             "table-contrast-train-20260924-forward_protected-5faeda4369c6"],
}
TURN_IN_PLACE_SPEED = .15     # m/s step toward the target facing while only A/D are held
REGIONS = ("feet", "legs", "trunk", "head", "arms", "hands")
PALETTE = {"wall": (158, 168, 178), "chair": (52, 120, 135), "table": (214, 140, 70), "top": (214, 140, 70)}


def box_color(category):
    for key, rgb in PALETTE.items():
        if key in category:
            return rgb
    return (160, 110, 90)


class Session:
    """One world: policy + unchanged training task on CPU MuJoCo, with a joystick command."""

    def __init__(self, args):
        import torch
        from record_mjlab_rollout import load_policy
        from cat_mjlab.runner import create_task
        from cat_mjlab.sim_cpu import CPUSimulation
        torch.set_num_threads(1)
        self.torch = torch
        self.learner, saved = load_policy(args.checkpoint, device="cpu", policy_id=0)
        env = copy.deepcopy(saved["contract"]["environment_config"])
        env["push_config"]["enable"] = False
        env["noise_config"]["level"] = 0.
        env["dm_rand_config"].update(enable_pd=False, enable_rfi=False)
        env.pop("goal_hold_seconds", None)
        env.pop("teleop", None)                     # the keyboard is the joystick here
        bank = args.bank
        ns = SimpleNamespace(bank_manifest=Path(bank + "_packed/manifest.json"), body_collision_bank=Path(bank + "_collision/manifest.json"),
                             body_collision_resets=Path(bank + "_resets/manifest.json"), device="cpu", seed=0, num_envs=1,
                             compile_task=False, nconmax=saved["contract"]["nconmax"], njmax=saved["contract"]["njmax"],
                             reactive_bank=None, reactive_row=None)
        print("Loading scene bank (about 30 s) ...", flush=True)
        self.task, self.sim, _ = create_task(ns, environment_config=env, sim_class=CPUSimulation)
        task = self.task
        task.collision_every_substep = False
        task.bank.episode_lengths.fill_(10 ** 9)
        # Obstacle check every other control step (8 ms each on CPU); the skipped steps report none.
        checker, self._tick = task.collision, [0]
        def every_other(scene_ids, data):
            self._tick[0] += 1
            if self._tick[0] % 2:
                return checker(scene_ids, data)
            return torch.zeros((1, 6), dtype=torch.bool)
        task.collision = every_other
        # Rewards are not needed; keep the telemetry of one real evaluation for the metrics code.
        rewards, self._components = task._rewards, None
        def no_rewards(action, contacts):
            if self._components is None:
                reward, components = rewards(action, contacts)
                self._components = {k: torch.zeros_like(v) for k, v in components.items()}
                return reward, components
            return torch.zeros(1), dict(self._components)
        task._rewards = no_rewards
        # Contact is shown, not terminal: only a fall (or a numerical failure) resets the robot.
        termination = task._termination
        self.last = dict(regions=torch.zeros(6, dtype=torch.bool), flags={})
        def shown_not_terminal(regions, contacts):
            done, flags = termination(regions, contacts)
            self.last = dict(regions=regions[0].clone(), flags={k: bool(v[0]) for k, v in flags.items()})
            return flags["fall"] | flags["numerical"], flags
        task._termination = shown_not_terminal
        task.joystick = torch.zeros((1, 2))
        self.manifest = json.loads(ns.bank_manifest.read_text())
        self.index = {s["scene_id"]: i for i, s in enumerate(self.manifest["scenes"])}
        from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory
        self._mp = ns.bank_manifest.resolve(); self._loaded = load_generalist_manifest(self._mp, verify_files=False)
        self._scene_directory = scene_directory
        resolve = lambda key: key if key in self.index else next((s for s in self.index if s.startswith(key)), None)
        self.scenes = [r for r in map(resolve, args.scenes) if r is not None]
        if not self.scenes:
            raise ValueError("none of the requested scenes is in the bank")
        self.scene = None
        self.heading, self.speed, self.speed_setting = 0., 0., args.speed
        self.contacts, self.contact_parts, self.falls = 0, {}, 0
        self.lock = threading.Lock()
        self.held, self.hold_mode = set(), False          # browser keys: hold-to-drive
        self.status = {}

    def boxes(self):
        record = self.manifest["scenes"][self.index[self.scene]]
        return json.loads((self._scene_directory(self._loaded, self._mp, record) / "scene.json").read_text())["boxes"]

    def load(self, scene_id):
        with self.lock:
            self.scene = scene_id
            self.task.reset(scene_ids=self.torch.tensor([self.index[scene_id]]))
            self.task.joystick.zero_()
            self.speed, self.contacts, self.contact_parts, self.falls = 0., 0, {}, 0
            self.heading = self.yaw()

    def yaw(self):
        q = self.sim.raw.qpos[3:7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def aim_at_nearest(self, text="top"):
        xy = self.sim.raw.qpos[:2]
        boxes = [b for b in self.boxes() if text in b["category"]]
        if boxes:
            b = min(boxes, key=lambda b: math.dist(b["center"][:2], xy))
            self.heading = math.atan2(b["center"][1] - xy[1], b["center"][0] - xy[0])
            self.speed = self.speed_setting

    def step(self):
        torch = self.torch
        with self.lock:
            if self.hold_mode:
                # Joystick-like. The policy has no yaw-rate input: it walks along the commanded
                # velocity and turns to face it. A/D rotate a target facing (90 deg/s) kept within
                # 60 deg of where the body points; W walks along it; A/D alone command a slow step
                # toward it, which turns the robot almost in place; nothing held = stand.
                yaw = self.yaw()
                turning = ("left" in self.held) - ("right" in self.held)
                if not ("up" in self.held or turning):
                    self.heading = yaw
                self.heading += turning * math.radians(90) * .02
                lag = math.remainder(self.heading - yaw, 2 * math.pi)
                self.heading = yaw + max(-math.radians(60), min(math.radians(60), lag))
                self.speed = self.speed_setting if "up" in self.held else (TURN_IN_PLACE_SPEED if turning else 0.)
            self.task.joystick[0, 0] = self.speed * math.cos(self.heading)
            self.task.joystick[0, 1] = self.speed * math.sin(self.heading)
            action = self.learner.act(self.task.obs, policy_ids=0, deterministic=True)["action"]
            before_falls = self.falls
            result = self.task.step(action)
            flags, regions = self.last["flags"], self.last["regions"]
            touching = [r for r, hit in zip(REGIONS, regions.tolist()) if hit]
            for key, part in (("hand_violation", "hands (envelope)"), ("elbow_violation", "elbows")):
                if flags.get(key):
                    touching.append(part)
            if touching:
                self.contacts += 1
                for part in touching:
                    self.contact_parts[part] = self.contact_parts.get(part, 0) + 1
            if bool(result["done"][0]):
                self.falls += 1; self.speed = 0.; self.heading = self.yaw()
            sdf = self.task.info["sdf"][0, :, 0]
            return dict(touching=touching, body_clearance=float(sdf.min()), hand_clearance=float(sdf[5:7].min()),
                        speed=float(torch.linalg.vector_norm(self.sim.data.qvel[0, :2])), fell=self.falls > before_falls)


def keyboard_loop(session, stop):
    """Raw-mode terminal keys (works over ssh). Arrow keys arrive as ESC [ A/B/C/D."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        buffer = ""
        while not stop.is_set():
            if not select.select([sys.stdin], [], [], .05)[0]:
                continue
            buffer += os.read(fd, 32).decode(errors="ignore")
            while buffer:
                if buffer.startswith("\x1b["):
                    if len(buffer) < 3:
                        break
                    key, buffer = {"A": "up", "B": "down", "C": "right", "D": "left"}.get(buffer[2], ""), buffer[3:]
                elif buffer[0] == "\x1b":
                    if len(buffer) < 2:
                        break
                    key, buffer = "", buffer[1:]
                else:
                    key, buffer = buffer[0].lower(), buffer[1:]
                handle_key(session, key, stop)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def handle_key(session, key, stop):
    s = session
    if key in ("up", "w"):
        s.speed = s.speed_setting
    elif key in ("down", "s", " "):
        s.speed = 0.
    elif key in ("left", "a"):
        s.heading += math.radians(20)
    elif key in ("right", "d"):
        s.heading -= math.radians(20)
    elif key in ("+", "="):
        s.speed_setting = min(1., s.speed_setting + .1); s.speed = s.speed_setting if s.speed else 0.
    elif key in ("-", "_"):
        s.speed_setting = max(.1, s.speed_setting - .1); s.speed = s.speed_setting if s.speed else 0.
    elif key == "t":
        s.aim_at_nearest("top")
    elif key == "r":
        s.load(s.scene)
    elif key == "n":
        s.load(s.scenes[(s.scenes.index(s.scene) + 1) % len(s.scenes)])
    elif key == "q":
        stop.set()


KEY_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Robot teleop</title>
<style>body{margin:0;overflow:hidden;background:#111}iframe{border:0;width:100vw;height:100vh}
#hud{position:fixed;left:12px;bottom:12px;background:rgba(0,0,0,.65);color:#fff;font:14px/1.4 system-ui,sans-serif;
padding:8px 12px;border-radius:8px;max-width:46vw;pointer-events:none}#hud b.c{color:#ff6b6b}#hud b.ok{color:#7ee787}</style></head>
<body><iframe id="v"></iframe><div id="hud">connecting ...</div><script>
document.getElementById('v').src=location.protocol+'//'+location.hostname+':__VISER_PORT__/';
const drive={ArrowUp:'up',KeyW:'up',ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right'};
const act={Space:'stop',ArrowDown:'stop',KeyS:'stop',KeyT:'table',KeyR:'reset',KeyN:'next',Equal:'faster',NumpadAdd:'faster',Minus:'slower',NumpadSubtract:'slower'};
const held=new Set();
const send=o=>fetch('/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});
addEventListener('keydown',e=>{const k=drive[e.code];if(k){e.preventDefault();if(!held.has(k)){held.add(k);send({key:k,down:true});}}
  else if(act[e.code]){e.preventDefault();if(!e.repeat)send({action:act[e.code]});}});
addEventListener('keyup',e=>{const k=drive[e.code];if(k){e.preventDefault();held.delete(k);send({key:k,down:false});}});
// Clicking the 3D view moves keyboard focus into the iframe; release keys and take focus back.
addEventListener('blur',()=>{for(const k of held)send({key:k,down:false});held.clear();setTimeout(()=>window.focus(),60);});
setInterval(()=>{if(document.activeElement&&document.activeElement.tagName==='IFRAME')window.focus();},300);
setInterval(async()=>{try{const s=await (await fetch('/status')).json();
 document.getElementById('hud').innerHTML=(s.touching&&s.touching.length?'<b class=c>CONTACT: '+s.touching.join(', ')+'</b>':'<b class=ok>no contact</b>')+
 '<br>speed setting '+s.setting.toFixed(1)+' m/s, actual '+s.speed.toFixed(2)+' m/s<br>closest: body '+s.body.toFixed(2)+' m, hands '+s.hands.toFixed(2)+
 ' m<br>contact steps '+s.contacts+', falls '+s.falls+'<br><span style="opacity:.7">hold W/Up walk forward, A/D turn (also in place), Space stop, +/- speed, T table, R reset, N next room</span>';}catch(e){}},250);
</script></body></html>"""


def serve_keys(session, port, viser_port, stop):
    import http.server
    page = KEY_PAGE.replace("__VISER_PORT__", str(viser_port)).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, body, kind):
            self.send_response(200); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/status"):
                self._reply(json.dumps(session.status).encode(), "application/json")
            else:
                self._reply(page, "text/html; charset=utf-8")

        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            session.hold_mode = True
            if "key" in msg:
                (session.held.add if msg.get("down") else session.held.discard)(msg["key"])
            action = msg.get("action")
            if action == "stop":
                session.held.clear(); session.speed = 0.
            elif action in ("faster", "slower"):
                session.speed_setting = min(1., session.speed_setting + .1) if action == "faster" else max(.1, session.speed_setting - .1)
            elif action:
                handle_key(session, {"table": "t", "reset": "r", "next": "n"}[action], stop)
                if action == "table":
                    session.hold_mode = False            # T keeps driving at the table until a key is pressed
            self._reply(b"{}", "application/json")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="outputs/expert_rooms_tables_v5_20260930/snapshots/update_0357_final.pt")
    p.add_argument("--bank", default="data/furniture/table_edges_v1", help="Bank prefix (_packed/_collision/_resets)")
    p.add_argument("--scenes", nargs="+", default=PRESETS["demo"])
    p.add_argument("--speed", type=float, default=.5, help="Initial speed setting (m/s)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-keyboard", action="store_true", help="Browser buttons only (no terminal key capture)")
    p.add_argument("--keys-port", type=int, help="Port of the browser keyboard page (default: --port + 1)")
    p.add_argument("--steps", type=int, default=0, help="Headless self-test: run this many steps, print timing, exit")
    args = p.parse_args(argv)
    os.chdir(ROOT)
    session = Session(args)
    session.load(session.scenes[0])

    if args.steps:                                           # headless check: timing + one straight push at a table
        session.aim_at_nearest("top"); t = time.time(); worst = 9.
        for _ in range(args.steps):
            info = session.step(); worst = min(worst, info["body_clearance"])
        dt = (time.time() - t) / args.steps
        print(json.dumps(dict(ms_per_step=round(dt * 1000, 1), realtime_factor=round(.02 / dt, 2), contacts=session.contacts,
                              contact_parts=session.contact_parts, falls=session.falls, closest_body_clearance_m=round(worst, 3))))
        return

    import mujoco
    import numpy as np
    import viser
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="fixed", control_width="large")
    from mjviser import ViserMujocoScene
    robot = ViserMujocoScene(server, session.sim.model, 1)
    # mjviser's own tracking draws the robot shifted to the origin (keeps it centred by moving the
    # scene); the furniture boxes are drawn in world coordinates, so the robot appeared inside the
    # west wall (x = 0). Draw everything in world coordinates and move the camera instead.
    robot.camera_tracking_enabled = False
    vis_data = mujoco.MjData(session.sim.model)
    box_handles = []

    def draw_scene():
        for h in box_handles:
            h.remove()
        box_handles.clear()
        for k, b in enumerate(session.boxes()):
            yaw = b["yaw"]
            box_handles.append(server.scene.add_box(f"/furniture/{k}", color=box_color(b["category"]),
                dimensions=tuple(2 * x for x in b["half_size"]), position=tuple(b["center"]),
                wxyz=(math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)), opacity=.55 if "wall" in b["category"] else 1.))

    status = server.gui.add_markdown("starting ...")
    with server.gui.add_folder("Drive"):
        row = server.gui.add_button_group("Command", ("Turn left", "Go", "Stop", "Turn right"))
        row.on_click(lambda e: handle_key(session, {"Turn left": "a", "Go": "w", "Stop": "s", "Turn right": "d"}[e.target.value], stop))
        speed = server.gui.add_slider("Speed (m/s)", .1, 1., .1, session.speed_setting)
        @speed.on_update
        def _(_):
            session.speed_setting = speed.value
            if session.speed: session.speed = speed.value
        server.gui.add_button("Drive straight at the nearest table").on_click(lambda _: session.aim_at_nearest("top"))
    with server.gui.add_folder("Scene"):
        choice = server.gui.add_dropdown("Room", tuple(session.scenes), initial_value=session.scene)
        @choice.on_update
        def _(_):
            session.load(choice.value); draw_scene()
        server.gui.add_button("Reset").on_click(lambda _: (session.load(session.scene), draw_scene()))
        follow = server.gui.add_checkbox("Camera follows robot", True)
    draw_scene()

    stop = threading.Event()
    if not args.no_keyboard and sys.stdin.isatty():
        threading.Thread(target=keyboard_loop, args=(session, stop), daemon=True).start()
    keys_port = args.keys_port or args.port + 1
    serve_keys(session, keys_port, args.port, stop)
    print(f"\nBrowser with keyboard: http://localhost:{keys_port}   (3D view only: http://localhost:{args.port})")
    print(f"From the laptop: ssh -L {args.port}:localhost:{args.port} -L {keys_port}:localhost:{keys_port} konstantinsmirnov@dep-1")
    print("Keys: Up/W go, Down/S stop, Left/A Right/D turn 20 deg, +/- speed, T at nearest table, R reset, N next scene, Q quit\n", flush=True)
    shown_scene, frame, t_report, n_report = session.scene, 0, time.time(), 0
    try:
        while not stop.is_set():
            start = time.time()
            info = session.step(); frame += 1; n_report += 1
            if session.scene != shown_scene:
                shown_scene = session.scene; draw_scene(); choice.value = session.scene
            if frame % 2 == 0:
                vis_data.qpos[:] = session.sim.raw.qpos; mujoco.mj_kinematics(session.sim.model, vis_data)
                robot.update_from_mjdata(vis_data)
                if follow.value and frame % 6 == 0:
                    # ~8 Hz, smoothed, look_at and position moved together in one message: separate
                    # updates every frame made the view jump between half-applied camera states.
                    target = vis_data.qpos[:3].copy(); target[2] = .8
                    for client in server.get_clients().values():
                        look = np.asarray(client.camera.look_at, dtype=float)
                        shift = .35 * (target - look)
                        if np.linalg.norm(shift) > .005:
                            with client.atomic():
                                client.camera.position = np.asarray(client.camera.position, dtype=float) + shift
                                client.camera.look_at = look + shift
            session.status = dict(touching=info["touching"], setting=session.speed_setting, speed=info["speed"],
                                  body=info["body_clearance"], hands=info["hand_clearance"], contacts=session.contacts, falls=session.falls,
                                  yaw_deg=math.degrees(session.yaw()), xy=[float(v) for v in session.sim.raw.qpos[:2]])
            if frame % 10 == 0:
                rate = n_report * .02 / max(time.time() - t_report, 1e-6); t_report, n_report = time.time(), 0
                touching = info["touching"]
                parts = ", ".join(f"{k} x{v}" for k, v in sorted(session.contact_parts.items())) or "none"
                status.content = (f"**{'CONTACT: ' + ', '.join(touching) if touching else 'no contact'}**  \n"
                                  f"commanded {session.speed:.1f} m/s toward {math.degrees(session.heading) % 360:.0f} deg, "
                                  f"actual {info['speed']:.2f} m/s  \n"
                                  f"closest clearance: body {info['body_clearance']:+.2f} m, hands {info['hand_clearance']:+.2f} m  \n"
                                  f"contact steps so far: {session.contacts} ({parts}); falls/resets: {session.falls}  \n"
                                  f"simulation speed {rate:.2f}x real time")
            time.sleep(max(0., .02 - (time.time() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
