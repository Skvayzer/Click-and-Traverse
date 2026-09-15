#!/usr/bin/env python3
"""Render the audited 15 September CAT setup. Does not access or modify a run.

Requires reportlab, pymupdf, matplotlib, numpy. Embedded plot PDFs and their
audited inputs live in docs/assets/setup-report-20260915. All dates and metrics
refer to that snapshot, not the time at which this document is regenerated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets/setup-report-20260915'
PDF = ROOT / 'docs/CAT_CURRENT_TRAINING_SETUP_20260915.pdf'
W, H = 841.89, 595.28
INK, MUTED, TEAL, BLUE, ORANGE = '#183347', '#596D7B', '#007D76', '#316CA4', '#AD501A'
LIGHT, LINE, GREEN_BG, BLUE_BG, AMBER_BG = '#F4F7F9', '#DCE5EA', '#EAF5F1', '#EEF3FA', '#FFF3E8'
COMMIT = '40b4dd2682dd4e5d75a6b853c24333da99379cf3'
CODE = f'https://github.com/Skvayzer/Click-and-Traverse/blob/{COMMIT}/'
WB = 'https://wandb.ai/skvayzer/CAT-wholebody/runs/1d39c55c'
MODEL = 'https://huggingface.co/Axian12138/Click-and-Traverse/tree/46ce4b57ba0639168d51741b661ff62f7ce6f045/logs_v1/generalist_v1/checkpoints/005033164800'


class Report:
    def __init__(self):
        fontdir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
        pdfmetrics.registerFont(TTFont('Body', str(fontdir / 'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('Bold', str(fontdir / 'DejaVuSans-Bold.ttf')))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
        self.layout = ASSETS / 'layout.pdf'
        self.c = canvas.Canvas(str(self.layout), pagesize=(W, H))
        self.page = 0
        self.charts = []
        self.toc = []

    def text(self, x, top, text, size=10.5, color=INK, bold=False):
        self.c.setFont('Bold' if bold else 'Body', size)
        self.c.setFillColor(colors.HexColor(color))
        self.c.drawString(x, H-top-size, text)

    def para(self, x, top, width, body, size=10.5, color=INK, max_height=100, leading=None):
        style = ParagraphStyle('p', fontName='Body', fontSize=size,
                               leading=leading or size*1.38, textColor=colors.HexColor(color))
        p = Paragraph(body, style)
        _, height = p.wrap(width, H)
        if height > max_height + .1:
            raise ValueError(f'Page {self.page}: {height:.1f} > {max_height}: {body[:100]}')
        p.drawOn(self.c, x, H-top-height)
        return height

    def rect(self, x, top, width, height, fill=LIGHT, radius=6):
        self.c.setFillColor(colors.HexColor(fill))
        self.c.setStrokeColor(colors.HexColor(LINE))
        self.c.setLineWidth(.6)
        self.c.roundRect(x, H-top-height, width, height, radius, stroke=1, fill=1)

    def box(self, x, top, width, height, title, body, fill=LIGHT, color=INK, size=10):
        self.rect(x, top, width, height, fill)
        self.text(x+12, top+10, title, 11, color, True)
        self.para(x+12, top+30, width-24, body, size, color, max_height=height-38)

    def arrow(self, points, color=MUTED):
        self.c.setStrokeColor(colors.HexColor(color)); self.c.setFillColor(colors.HexColor(color))
        self.c.setLineWidth(1.2)
        p = self.c.beginPath(); p.moveTo(points[0][0], H-points[0][1])
        for x, t in points[1:]:
            p.lineTo(x, H-t)
        self.c.drawPath(p)
        end, prev = np.array(points[-1], dtype=float), np.array(points[-2], dtype=float)
        direction = (end-prev)/np.linalg.norm(end-prev)
        side = np.array([-direction[1], direction[0]])
        p = self.c.beginPath(); p.moveTo(end[0], H-end[1])
        for a in (end-direction*6+side*2.5, end-direction*6-side*2.5):
            p.lineTo(a[0], H-a[1])
        p.close(); self.c.drawPath(p, fill=1, stroke=0)

    def begin(self, title, subtitle):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.toc.append([1, title, self.page])
        self.text(32, 17, 'CLICK-AND-TRAVERSE  /  CURRENT IMPLEMENTATION  /  15 SEPTEMBER 2026', 8.2, TEAL, True)
        self.text(32, 39, title, 24, INK, True)
        self.para(33, 77, 775, subtitle, 10.2, MUTED, max_height=30)
        self.c.setStrokeColor(colors.HexColor(LINE)); self.c.line(32, 32, W-32, 32)
        self.text(32, H-24, 'dep-0 · 14:47 UTC snapshot · frozen training source 40b4dd2', 7.6, MUTED)
        self.text(756, H-24, f'{self.page} / 10', 8.2, MUTED)

    def table(self, x, top, widths, rows, row_height=28, size=9.4):
        for i, row in enumerate(rows):
            self.rect(x, top+i*row_height, sum(widths), row_height,
                      BLUE_BG if i == 0 else ('#FFFFFF' if i % 2 else LIGHT), radius=0)
            px = x
            for value, width in zip(row, widths):
                self.para(px+8, top+i*row_height+6, width-16,
                          f'<b>{value}</b>' if i == 0 else str(value), size,
                          max_height=row_height-8, leading=size*1.22)
                px += width

    def chart(self, name, x, top, width, height):
        self.charts.append((self.page-1, name, pymupdf.Rect(x, top, x+width, top+height)))

    def link(self, x, top, label, url, size=9):
        self.text(x, top, label, size, BLUE)
        width = pdfmetrics.stringWidth(label, 'Body', size)
        self.c.linkURL(url, (x, H-top-size-2, x+width, H-top+2), relative=0)

    def finish(self):
        self.c.showPage(); self.c.save()
        doc = pymupdf.open(self.layout)
        for page, name, rect in self.charts:
            with pymupdf.open(ASSETS / (name+'.pdf')) as chart:
                doc[page].show_pdf_page(rect, chart, 0)
        doc.set_metadata(dict(title='CAT: whole-body furniture traversal and hand protection — current setup',
                              author='Konstantin Smirnov', subject='Audited dep-0 implementation and training snapshot, 2026-09-15 14:47 UTC'))
        doc.set_toc(self.toc)
        doc.save(PDF, garbage=4, deflate=True)
        doc.close()
        checks = []
        previews = ASSETS / 'previews'
        previews.mkdir(exist_ok=True)
        with pymupdf.open(PDF) as doc:
            assert len(doc) == 10
            alltext = '\n'.join(p.get_text() for p in doc)
            for required in ('1,138,753,536', '406', '494', '8,192', '97.42%',
                             '39 fixed scenes', 'No human motion', '0.0 m', '50 control steps'):
                assert required in alltext, required
            for page in doc:
                for block in page.get_text('dict')['blocks']:
                    if block['type'] != 0:
                        continue
                    for line in block['lines']:
                        for span in line['spans']:
                            assert page.rect.contains(pymupdf.Rect(span['bbox'])), (page.number+1, span['text'])
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(previews / f'page-{page.number+1:02}.png')
                checks.append(dict(page=page.number+1, text_characters=len(page.get_text()),
                                   links=len(page.get_links()), text_within_page=True))
        (ASSETS/'document-checks.json').write_text(json.dumps(dict(
            pages=checks, required_claims_present=True, vector_charts=len(self.charts),
            pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest()), indent=2)+'\n')
        self.layout.unlink()
        print(PDF)


def compose():
    d = json.loads((ASSETS/'progress-summary.json').read_text())
    assert d['last_logged_transition'] == 1138753536
    assert d['code']['git_commit'] == COMMIT
    r = Report()

    r.begin('What is training on dep-0?',
            'One CAT generalist is being fine-tuned for leg, waist and arm control across original CAT scenes and dense clutter.')
    r.rect(32, 117, 778, 42, GREEN_BG)
    r.text(45, 128, 'LIVE AT SNAPSHOT: 1,138,753,536 transitions · 8,192 environments · one W&B run', 12, TEAL, True)
    r.box(32, 185, 225, 95, 'Released CAT generalist',
          'Native pretrained actor + critic.<br/>Leg/navigation weights restored;<br/>new upper-body outputs added.', BLUE_BG, BLUE)
    r.box(308, 185, 226, 95, 'One trainable whole-body actor',
          '406 observations → 29 actions.<br/>12 leg + 3 waist + 14 arm/wrist.<br/>Dex3 fingers stay fixed.', GREEN_BG, TEAL)
    r.box(585, 185, 225, 95, 'CAT simulator + native PPO',
          '39 fixed scenes in one learner.<br/>50 Hz control, 500 Hz physics.<br/>Continuous updates until stopped.', BLUE_BG, BLUE)
    r.arrow([(257, 232), (308, 232)], BLUE)
    r.arrow([(534, 232), (585, 232)], TEAL)
    r.arrow([(698, 280), (698, 305), (421, 305), (421, 280)], TEAL)
    r.text(474, 311, 'On-policy rollouts + PPO gradients', 9, TEAL)
    r.box(32, 347, 377, 143, 'Implemented additions',
          'Arm and waist actuation; mesh-sized hand envelopes including thumbs; predictive hand-clearance features; hand/arm penalties; dense tables/chairs and generic clutter.<br/><br/>All run together with the original CAT reward and scene sampler.', GREEN_BG, TEAL, 10.3)
    r.box(433, 347, 377, 143, 'What the results currently support',
          'Generic clutter survival improves. Furniture remains difficult: 97.42% of its recent completed episodes terminate early.<br/><br/>No human motion demonstrations or imitation loss are used. Reliable protective hand raising is not yet demonstrated.', AMBER_BG, ORANGE, 10.3)
    r.link(33, 517, 'Open the single live W&B run', WB)
    r.text(342, 517, 'Snapshot: 15 Sep 2026, 14:47 UTC / 18:47 Dubai', 9, MUTED)

    r.begin('The actor sees fields and controls 29 joints',
            'Separate feed-forward actor and critic MLPs preserve CAT’s hidden-layer sizes. Every layer remains trainable.')
    r.box(32, 120, 235, 120, 'Actor input: 406 values',
          '131 proprioception / control<br/>77 original body-field features<br/>198 added hand/arm features<br/><br/>Observation normalization: off', BLUE_BG, BLUE, 10)
    r.box(309, 120, 235, 120, 'Actor MLP',
          '<b>406 → 512 → 256 → 128 → 64</b><br/>Swish / SiLU hidden activations.<br/><br/>58 linear outputs:<br/>29 means + 29 raw scales.', GREEN_BG, TEAL, 10)
    r.box(586, 120, 224, 120, 'Action distribution',
          'σ = softplus(raw scale) + 0.001<br/>Sample Gaussian; apply tanh.<br/><br/>29 bounded actions become<br/>joint position targets.', BLUE_BG, BLUE, 10)
    r.arrow([(267, 181), (309, 181)], BLUE)
    r.arrow([(544, 181), (586, 181)], TEAL)
    r.box(32, 264, 235, 92, 'Critic input: 494 values',
          '208 current / noiseless base<br/>88 privileged values<br/>198 true hand/arm features', LIGHT, INK, 10)
    r.box(309, 264, 501, 92, 'Critic MLP — used during learning',
          '<b>494 → 1,024 → 512 → 256 → 128 → 1 value</b><br/><br/>No recurrent state, Transformer or 15-frame observation stack.', LIGHT, INK, 10)
    r.arrow([(267, 310), (309, 310)])
    r.table(32, 384, [240, 260, 278], [
        ['Feature block', 'Where it comes from', 'Purpose'],
        ['11 original sites × 7 = 77', 'Head, torso, pelvis, limbs', 'Guidance xyz + boundary xyz + SDF'],
        ['22 new probes × 9 = 198', 'Hand envelopes, palms, arms', '3 distances + boundary xyz + 3 metadata'],
    ], row_height=33, size=9.3)
    r.para(33, 501, 775,
           '<b>Warm start:</b> released inputs are 162 actor / 250 critic values, with 12 leg actions. Retained weights are copied; new input weights and action means start at zero, new action standard deviation at 0.05. Mapped retained actor/value outputs match at initialization; physical trajectories need not match. New arm skills must be learned.',
           10, max_height=48)

    r.begin('From field queries to motion: every 20 ms',
            'The policy observes a known static 3D field map. This training run has no camera, depth reconstruction or online SLAM.')
    r.box(32, 123, 173, 105, '1  Observe',
          'Noisy proprioception;<br/>body-site field samples;<br/>hand/arm probes.<br/>Root odometry held 5 steps.', BLUE_BG, BLUE, 9.6)
    r.box(234, 123, 173, 105, '2  Act',
          'One actor samples<br/>29 actions at 50 Hz.<br/><br/>Legs, waist and arms act<br/>in the same policy step.', GREEN_BG, TEAL, 9.6)
    r.box(436, 123, 173, 105, '3  Set joint targets',
          'Legs: incremental.<br/>Upper body: nominal offsets<br/>with a target-rate limit.<br/>Soft joint limits retained.', BLUE_BG, BLUE, 9.6)
    r.box(638, 123, 172, 105, '4  Simulate',
          '10 physics steps × 2 ms.<br/>PD torques recomputed;<br/>torque limits and random<br/>force injection retained.', LIGHT, INK, 9.6)
    for x in (205, 407, 609):
        r.arrow([(x, 177), (x+29, 177)])
    r.box(234, 269, 375, 71, '5  Reward, terminal logic and scene resampling',
          'CAT reward + two clearance penalties; collect transitions.<br/>On episode end, reset and sample a scene for that environment.', GREEN_BG, TEAL, 9.6)
    r.arrow([(724, 228), (724, 303), (609, 303)])
    r.arrow([(234, 303), (117, 303), (117, 228)])
    r.table(32, 367, [166, 327, 285], [
        ['Action group', 'Target update', 'What moves'],
        ['12 leg joints', 'q_target = q_previous + 0.5 × action', 'Original CAT incremental semantics'],
        ['17 upper-body joints', 'q_target = q_nominal + 0.8 × action', '3 waist + 14 arm / wrist joints'],
        ['Upper target rate', '2 rad/s = at most 0.04 rad/control step', 'Smooths target changes before limits'],
        ['Dex3-1 fingers', 'Fixed pose; 7 finger joints welded per hand', '3-finger hardware; no finger actions'],
    ], row_height=29, size=9.3)
    r.para(33, 524, 775, '<b>Timing detail:</b> root odometry refreshes every 0.1 s; articulation and field queries update each control step. The critic uses the current true state. No active odometry-noise injection is applied in this path.', 9.4, MUTED, max_height=28)

    r.begin('The added rooms require repeated narrow passages',
            'Actual current bank geometry: each room is 9.00 × 9.69 m, with six 68–78 cm bottlenecks and obstacles at several heights.')
    r.chart('scene-layouts', 32, 117, 778, 310)
    r.box(32, 438, 250, 104, 'One fixed bank: 39 scenes',
          '36 byte-verified public CAT fields<br/>+ 1 reconstructed missing CAT slot<br/>+ 1 furniture + 1 generic room.<br/>Resets randomize state, not layout.', BLUE_BG, BLUE, 9.6)
    r.box(296, 438, 250, 104, 'Original CAT field pipeline',
          '3D occupancy at 4 cm resolution<br/>→ signed distance field (SDF)<br/>→ boundary gradient + progressive<br/>3D fast-marching guidance to goal.', GREEN_BG, TEAL, 9.6)
    r.box(560, 438, 250, 104, 'Geometry is not a learned path',
          'The dashed 39.85 m route checks<br/>root clearance during construction.<br/>It is not supplied to the actor and<br/>does not prove dynamic feasibility.', AMBER_BG, ORANGE, 9.6)

    r.begin('Hand protection uses geometry and predicted clearance',
            'Dex3-1 envelopes include the thumb and 5 mm padding. Twenty-two probes cover both hands and both arms.')
    r.chart('hand-envelope', 32, 118, 371, 208)
    r.chart('scene-fields', 430, 118, 380, 208)
    r.para(33, 336, 369,
           '<b>Probe placement:</b> each hand has a 3 cm palm sphere plus 8 box-corner points. Four 5 cm spheres cover forearms/elbows: 18 hand + 4 arm probes total.<br/><br/><b>At each probe:</b> clearance now, at 0.2 s and 0.4 s; boundary xyz; observation age; unknown flag; position uncertainty. The last two are zero in this run.',
           10.2, max_height=125)
    r.para(433, 336, 377,
           '<b>Hand penalty:</b> below 12 cm minimum predicted clearance, multiply squared deficit by (1 + urgency), with urgency clipped to 0–10; average over 18 probes, scale −5.<br/><br/><b>Arm penalty:</b> below 8 cm current clearance, average squared deficit over 4 probes, scale −2. Both join the full CAT reward before its 0.02 s scaling and clipping.',
           10.2, max_height=125)
    r.rect(32, 478, 778, 66, AMBER_BG)
    r.para(45, 488, 750,
           '<b>What this can teach:</b> the actor can raise, tuck or reposition arms when that improves return. Prediction extrapolates current velocity; it does not simulate future actions. Point probes do not certify a whole swept volume. There is no explicit human-style hand-raising target or hard action safety filter.',
           10.2, ORANGE, max_height=47)

    r.begin('Native CAT PPO runs continuously on one GPU',
            'The released final generalist has already completed distillation. This run starts direct PPO fine-tuning; there is no live teacher stage.')
    r.box(32, 120, 225, 101, 'Collect an on-policy batch',
          '8,192 parallel environments<br/>× 32 control steps × 4 chunks<br/><br/><b>1,048,576 transitions / update</b>', BLUE_BG, BLUE, 10)
    r.box(309, 120, 225, 101, 'Advantages and optimization',
          'GAE advantages; clipped PPO.<br/>64 minibatches × 4 passes.<br/><br/><b>256 Adam steps / update</b>', GREEN_BG, TEAL, 10)
    r.box(586, 120, 224, 101, 'Log, save and continue',
          'Same actor, critic and optimizer;<br/>same env, RNG and sampler state.<br/><br/>One W&amp;B ID and global-step axis.', BLUE_BG, BLUE, 10)
    r.arrow([(257, 170), (309, 170)], BLUE)
    r.arrow([(534, 170), (586, 170)], TEAL)
    r.arrow([(698, 221), (698, 246), (145, 246), (145, 221)], TEAL)
    r.table(32, 276, [315, 232, 231], [
        ['Setting', 'Released CAT', 'Active dep-0 run'],
        ['Parallel environments', '65,536', '8,192'],
        ['Trajectories per minibatch', '2,048', '512 (16,384 transitions)'],
        ['Transitions per PPO update', '4,194,304', '1,048,576'],
        ['Unroll / minibatches / passes', '32 / 64 / 4', '32 / 64 / 4'],
        ['Learning rate / entropy / discount', '0.0003 / 0.003 / 0.98', 'Same'],
        ['Clip / GAE / max gradient norm', '0.2 / 0.95 / 1', 'Same'],
        ['Reward scaling / advantage normalization', '1 / enabled', 'Same'],
    ], row_height=28, size=9.4)
    r.para(33, 516, 775,
           '<b>Resource change:</b> the batch is four times smaller to fit expanded observations on the 32 GB GPU. Adam initializes once at fine-tuning start. There is no finite step budget, automatic evaluation, stage reset or rollback to best weights.',
           10.1, max_height=30)

    r.begin('Preserved CAT task, with explicit extensions',
            'The full CAT reward, original reset behavior, noise, disturbances, gait logic and adaptive mixed-scene wrapper remain in use.')
    r.box(32, 120, 377, 173, 'Reward: 22 original terms + 2 additions',
          'Original navigation / orientation and body-motion terms; gait/contact/clearance/slip/balance terms; smoothness, joint-limit and torque costs; body guidance and distance-field terms.<br/><br/><b>Task reward = clip(0.02 × weighted sum, 0, 10,000).</b><br/>The new hand and arm costs are inside that sum. CAT’s outer wrapper zeros learner reward on terminal transitions.', BLUE_BG, BLUE, 10.1)
    r.box(433, 120, 377, 173, 'When an environment resets',
          'Falls and invalid state terminate immediately. Selected self-contacts, body-field penetration and added probe penetration terminate after the original grace period.<br/><br/><b>Collision threshold: 0.0 m.</b><br/><b>Grace: 50 control steps (about 1 s).</b><br/>Timeouts are truncations; CAT’s native GAE handling remains.', GREEN_BG, TEAL, 10.1)
    r.box(32, 313, 377, 155, 'Reset and episode length',
          'Original scenes: CAT start distribution (±1 m XY), original randomization, 1,000-step / 20 s time limit.<br/><br/>New rooms: entrance reset with ±8 cm XY jitter, original yaw/joint/velocity randomization, 4,000-step / 80 s limit; goal reward uses the room’s actual goal.', LIGHT, INK, 10.1)
    r.box(433, 313, 377, 155, 'CAT adaptive scene sampling',
          'At each step, update per-scene completed and timeout counts with EMA decay 0.95.<br/><br/>s = timeout EMA / completed EMA<br/>weight = max(1 − s, 0.001); normalize over 39 slots.<br/>A reset samples from these weights. Initial weights are uniform; scene mix evolves with survival.', LIGHT, INK, 10.1)
    r.rect(32, 491, 778, 53, AMBER_BG)
    r.para(45, 501, 750,
           '<b>Critical interpretation:</b> reaching the goal does not itself end an episode. The sampler’s “success” statistic means timeout survival. Training still uses CAT’s flat-floor/feet contact physics; furniture obstacles are fields, not contact-force bodies.',
           10.2, ORANGE, max_height=34)

    r.begin('Runtime, unified logging and checkpoint handling',
            'One detached process on dep-0 uses a frozen source checkout. Reporting here is read-only and does not interrupt training.')
    r.table(32, 121, [174, 202], [
        ['Hardware / software', 'Actual setup'],
        ['Host / process', 'dep-0 (WS012743) / PID 11525'],
        ['GPU', 'RTX 5090 · 32,607 MiB total'],
        ['NVIDIA allocated VRAM', '30,425 MiB = 29.71 GiB'],
        ['JAX peak live / pool limit', '22.64 / 28.26 GiB'],
        ['Epoch throughput', '≈87,600 transitions/s'],
        ['CPU / host RAM', '24 logical CPUs / 125 GiB'],
        ['Python / JAX / Brax', '3.12.9 / 0.4.38 / 0.12.3'],
        ['MuJoCo + MJX / Flax', '3.3.1 / 0.10.4'],
    ], row_height=29, size=9.3)
    r.para(33, 397, 375,
           'VRAM accounting overlaps: the CUDA total includes the allocator pool. Do not add these values. Execution samples reach 99–100% utilization; host checkpoint writes create gaps. The snapshot caught 0% at a boundary. Throughput excludes checkpoint-writing time.',
           10, MUTED, max_height=86)
    r.box(433, 121, 377, 100, 'One W&B experiment',
          'Run ID <b>1d39c55c</b> / project CAT-wholebody.<br/>Pooled episode and optimization metrics plus per-scene diagnostics share one global-step axis. Scene families are not separate runs.', BLUE_BG, BLUE, 10.1)
    r.box(433, 237, 377, 117, 'One selected best model',
          '<b>checkpoints/best</b> contains actor/critic inference state.<br/>Selected by maximum single-update training reward proxy.<br/>Current best: update 6 / 6,291,456 transitions, score 0.350549. It never replaces the live learner.<br/>This is not a held-out furniture-success score.', AMBER_BG, ORANGE, 9.8)
    r.box(433, 370, 377, 117, 'One overwritten full recovery file',
          '<b>resume.msgpack</b> stores current weights, Adam, RNG, normalizer, environment, scene sampler and metric buffers.<br/>Atomic save at initialization and after each PPO update.<br/>About 1.03 GiB; native best model about 6 MB.<br/>Exact resume checks source, fields and configuration.', GREEN_BG, TEAL, 9.8)
    r.para(33, 507, 775,
           '<b>Run:</b> outputs/cat_wholebody_generalist_20260915. <b>Start:</b> 10:37:15 UTC, 15 September. Same process, attempt 1, zero logged nonfinite events through the snapshot. Continuous means no scheduled cutoff; process errors can still stop it.',
           10, max_height=39)

    r.begin('Measured progress: survival improves, furniture lags',
            'Read-only snapshot at 14:47:33 UTC: 1,086 complete PPO updates and 1.556 million completed training episodes.')
    r.chart('progress-curves', 32, 116, 479, 311)
    r.table(532, 126, [148, 60, 70], [
        ['Recent comparison', 'Early', 'Latest'],
        ['All: early termination', '44.30%', '39.10%'],
        ['Original CAT slots', '37.54%', '35.10%'],
        ['Furniture', '99.96%', '97.42%'],
        ['Generic clutter', '99.51%', '53.54%'],
        ['Reward / transition', '0.3322', '0.3270'],
        ['Logged duration (s)', '12.30', '14.85'],
    ], row_height=31, size=8.8)
    r.para(533, 360, 276,
           '<b>Hand objective:</b> proximity-penalty magnitude per logged episode step is 26.38% higher. That is worse, despite longer episodes. Reward per transition is 1.59% lower.',
           10.1, ORANGE, max_height=78)
    r.box(32, 447, 479, 97, 'What the comparison measures',
          'Early = updates 21–70; latest = 1037–1086 (50 updates each). Early is already fine-tuned, not the pretrained baseline. Failure combines falls, collision and invalid state. Counts are episode-weighted after undoing the logger’s rolling means.', LIGHT, INK, 9.8)
    r.box(532, 447, 278, 97, 'What it does not establish',
          'Adaptive training scene mix is not a matched evaluation. Timeout is not goal success. Neither safe hand raising nor reliable furniture traversal is demonstrated.', AMBER_BG, ORANGE, 9.8)

    r.begin('What remains to establish, and how to reproduce it',
            'The implementation is inspectable and the run is reproducible. Learned skill quality still requires controlled evaluation.')
    r.box(32, 119, 377, 150, 'Evidence already obtained',
          '243 full regression checks and 72 later focused recovery checks passed. Tests cover original CAT reset/step behavior, exact mapped actor/critic outputs, fields, sampler parity and mixed-scene reset/step integration.<br/><br/>The actual 8,192-environment GPU profile compiled and has executed 1,086 updates. These are integration checks, not traversal-success tests.', GREEN_BG, TEAL, 10)
    r.box(433, 119, 377, 150, 'Still missing from the experiment',
          'A matched pretrained-versus-current evaluation; goal completion and hand-clearance statistics; unseen room layouts; collision-force evaluation with physical furniture; a dynamic whole-body feasibility check.<br/><br/>No human motion data or imitation loss is integrated. Scene-aligned, retargeted human motion remains a proposed follow-up.', AMBER_BG, ORANGE, 10)
    r.text(33, 291, 'SOURCE AND DATA IDENTITIES', 9.2, TEAL, True)
    r.table(32, 312, [187, 591], [
        ['Component', 'Pinned identity'],
        ['Training source', COMMIT],
        ['Original CAT reference', '866ba392f1c1e84b92ad75fa66550f26e8af8e48'],
        ['Pretrained model', '46ce4b57ba0639168d51741b661ff62f7ce6f045 · checkpoint 005033164800'],
        ['Released field dataset', 'db8c202fa724bc4d3dba2cb9fead1267f15fd811'],
        ['Missing original scene', 'D8G2L3O2S13 reconstructed; unknown original bytes are not claimed'],
    ], row_height=27, size=9.2)
    r.link(33, 492, 'Frozen training entry point', CODE+'train_cat_wholebody.py')
    r.link(297, 492, 'Whole-body task implementation', CODE+'cat_ppo/envs/g1/env_cat_wholebody.py')
    r.link(594, 492, 'Released pretrained checkpoint', MODEL)
    r.para(33, 516, 775,
           'The accompanying Markdown has the exact launch command, operator paths, field/source hashes and software lock. Figure inputs, geometry provenance, validated progress counts and the PDF builder are saved with this report. All plots show the current fields or measured training logs.',
           9.8, MUTED, max_height=30)
    r.finish()


if __name__ == '__main__':
    compose()
