"""Paired CAT/clutter evaluation with bounded, multi-scene world batches (never trains).

Defaults: 400 scenes (all 64 original/published, 336 randomly sampled procedural),
32 distinct certified poses/scene, 16 scenes/batch, 512 parallel worlds.
Explicit --scene-ids also accepts furniture and generic_clutter (separate groups).
--scene-families furniture generic_clutter selects all 61 ordinary-clutter scenes.
--continue-after-clean-goal observes physical episode lengths and raw timeouts;
without it, legacy first-outcome stopping is preserved.
Only the current batch's ragged fields is resident; collision metadata is shared.
At most 60.9 MiB of CAT fields is needed for the recommended 16-scene batch.

Certificates guarantee obstacle clearance, NOT traversal bounds. Exact certified
qpos is installed without pose perturbation. Initial out-of-bounds poses are
explicit zero-step 'other' outcomes for BOTH actors; summaries also show rates
conditional on an in-bounds start. Collision certificate mismatches still abort.

Planning at 512 worlds: 2.5–4 GiB total, ~6.1 GiB conservative allowance within
8 GiB headroom. Throughput hypothesis: 10–30 pairs/min, not GPU-verified. Runtime
checks restore/hash the full batch before each policy and replay fixed actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'data/furniture/cat_flat_balance_v1_20260920'
GROUPS = ('original_published_cat', 'procedural_cat')
CLUTTER_GROUPS = ('furniture', 'generic_clutter')
OUTCOMES = ('clean_goal', 'body_collision', 'fall', 'timeout', 'other')


def actor_observation_for_policy(config, state):
    """Require the native physical-feature contract; no prefix slicing."""
    from cat_mjlab.observation_contract import ACTOR_SIZE, CRITIC_SIZE
    if (config.actor_obs, config.critic_obs, config.action_size) != (ACTOR_SIZE, CRITIC_SIZE, 29):
        raise ValueError('Unsupported policy observation/action contract; expected 222/310/29')
    if state.shape[-1] != ACTOR_SIZE:
        raise ValueError('Evaluator must supply the native actor observation')
    return state


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(source, destination):
    """Open once: the trainer atomically replaces resume.pt, preserving this inode.

    Also reject an in-place writer via fstat; never retry a possibly torn copy.
    """
    with source.open('rb') as src, destination.open('xb') as dst:
        before = os.fstat(src.fileno())
        shutil.copyfileobj(src, dst, 1024 * 1024)
        after = os.fstat(src.fileno())
    # Replacing the path can change the old inode's ctime/link count without
    # changing its contents. Size and mtime detect ordinary in-place writes.
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        destination.unlink()
        raise RuntimeError(f'Checkpoint changed in place while copying: {source}; rerun')
    return dict(source=str(source), frozen=str(destination), sha256=digest(destination),
                bytes=before.st_size, source_mtime_ns=before.st_mtime_ns)


def group(record):
    if record['family'] in ('original_cat', 'published_cat'):
        return GROUPS[0]
    if record['family'] == 'procedural_cat':
        return GROUPS[1]
    if record['family'] in CLUTTER_GROUPS:
        return record['family']
    return None


def new_statistics(groups=GROUPS):
    # Bounded host memory even for millions of pairs; JSONL owns full provenance.
    return {name: dict(pairs=0, paired=Counter(), reset_outside_bounds=0, scenes={}, policies={key: dict(
        outcomes=Counter(), lengths=Counter(), first_goals=Counter(), physical_lengths=Counter(),
        raw_timeouts=0, unsuccessful_timeouts=0, flags=Counter()) for key in ('A', 'B')}) for name in groups}


def record_pair(statistics, row):
    item = statistics[row['group']]
    item['pairs'] += 1
    outside = int(row['initial_outside_bounds'])
    item['reset_outside_bounds'] += outside
    scene = item['scenes'].setdefault(row['scene_id'], dict(pairs=0, eligible=0, A=0, B=0, discordant=0))
    scene['pairs'] += 1
    scene['eligible'] += 1-outside
    for key in ('A', 'B'):
        scene[key] += int(row['outcomes'][key]['outcome'] == 'clean_goal')
    scene['discordant'] += int((row['outcomes']['A']['outcome'] == 'clean_goal') !=
                               (row['outcomes']['B']['outcome'] == 'clean_goal'))
    item['paired'][row['outcomes']['A']['outcome'] + ' -> ' + row['outcomes']['B']['outcome']] += 1
    for key, outcome in row['outcomes'].items():
        policy = item['policies'][key]
        policy['outcomes'][outcome['outcome']] += 1
        policy['lengths'][outcome['length']] += 1
        first_goal = outcome['time_to_first_clean_goal']
        if first_goal is not None:
            policy['first_goals'][first_goal] += 1
        policy['physical_lengths'][outcome['physical_length']] += 1
        policy['raw_timeouts'] += int(outcome['raw_timeout'])
        policy['unsuccessful_timeouts'] += int(outcome['unsuccessful_timeout'])
        if outcome['outcome'] not in ('clean_goal', 'timeout'):
            policy['flags'].update(flag for flag, present in outcome['flags'].items()
                                   if present and flag not in ('reset_replaced', 'goal_reached', 'raw_goal'))


def distribution(histogram, np):
    """Interpolated order statistics without expanding the bounded histogram."""
    sample_count = sum(histogram.values())
    values = np.array(sorted(histogram), dtype=float)
    frequencies = np.array([histogram[int(v)] for v in values], dtype=np.int64)
    lengths = None
    if sample_count:
        ranks = np.array([0, .25, .5, .75, .9, .95, 1.])*(sample_count-1)
        cumulative = frequencies.cumsum()
        lower = values[np.searchsorted(cumulative, np.floor(ranks), side='right')]
        upper = values[np.searchsorted(cumulative, np.ceil(ranks), side='right')]
        quantiles = lower+(upper-lower)*(ranks-np.floor(ranks))
        lengths = dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'p95', 'max'), map(float, quantiles)))
        lengths.update(mean=float((values*frequencies).sum()/sample_count),
                       histogram={str(k): v for k, v in sorted(histogram.items())})
    return lengths


def summarize(statistics, np):
    result = {}
    for name, population in statistics.items():
        count = population['pairs']
        eligible = count-population['reset_outside_bounds']
        item = dict(pairs=count, scenes=len(population['scenes']),
                    initial_outside_bounds=population['reset_outside_bounds'], eligible_pairs=eligible,
                    policies={}, paired_outcomes=dict(population['paired']))
        for key, policy in population['policies'].items():
            counts = policy['outcomes']
            lengths = distribution(policy['lengths'], np)
            item['policies'][key] = dict(episodes=count,
                clean_goal_success_rate=counts['clean_goal']/count if count else None,
                clean_goal_success_rate_given_in_bounds_reset=counts['clean_goal']/eligible if eligible else None,
                outcome_counts={k: counts[k] for k in OUTCOMES},
                failure_rates={k: counts[k]/count if count else None for k in OUTCOMES[1:]},
                failure_flag_counts=dict(policy['flags']), length_steps=lengths,
                time_to_first_clean_goal=distribution(policy['first_goals'], np),
                time_to_first_clean_goal_count=sum(policy['first_goals'].values()),
                physical_length_steps=distribution(policy['physical_lengths'], np),
                raw_timeout_count=policy['raw_timeouts'],
                raw_timeout_rate=policy['raw_timeouts']/count if count else None,
                unsuccessful_timeout_count=policy['unsuccessful_timeouts'],
                unsuccessful_timeout_rate=policy['unsuccessful_timeouts']/count if count else None)
        item['success_rate_B_minus_A'] = ((population['policies']['B']['outcomes']['clean_goal']-
            population['policies']['A']['outcomes']['clean_goal'])/count if count else None)
        item['success_rate_B_minus_A_given_in_bounds_reset'] = (
            (population['policies']['B']['outcomes']['clean_goal']-
             population['policies']['A']['outcomes']['clean_goal'])/eligible if eligible else None)
        item['per_scene'] = population['scenes']
        discordant = sum(r['discordant'] for r in population['scenes'].values())
        item['paired_success_discordant_count'] = discordant
        item['paired_success_discordant_fraction'] = discordant/count if count else None
        item['paired_success_discordant_fraction_given_in_bounds_reset'] = discordant/eligible if eligible else None
        if len(population['scenes']) >= 2:
            records = list(population['scenes'].values())
            differences = np.array([r['B']-r['A'] for r in records], dtype=float)
            sizes = np.array([r['pairs'] for r in records], dtype=float)
            eligible_sizes = np.array([r['eligible'] for r in records], dtype=float)
            weights = np.random.default_rng(7381).multinomial(len(records),
                np.full(len(records), 1/len(records)), size=2000)
            numerator = weights @ differences
            item['scene_cluster_bootstrap_95_delta'] = np.percentile(numerator/(weights @ sizes), [2.5, 97.5]).tolist()
            denominator = weights @ eligible_sizes
            item['scene_cluster_bootstrap_95_delta_given_in_bounds_reset'] = (
                np.percentile(numerator[denominator > 0]/denominator[denominator > 0], [2.5, 97.5]).tolist()
                if np.any(denominator > 0) else None)
            item['uncertainty_note'] = ('Paired scene-cluster percentile bootstrap, 2000 resamples; no finite-population correction. '
                'Exploratory across-scene uncertainty, not a guarantee of small-effect detection or an IID episode interval.')
        result[name] = item
    return result


def select_scenes(scenes, scene_count, seed, explicit_ids=None, families=None):
    available = [i for i, scene in enumerate(scenes) if group(scene) is not None]
    if explicit_ids and families:
        raise ValueError('Choose either scene IDs or scene families, not both')
    if families:
        return [i for i in available if scenes[i]['family'] in families]
    if explicit_ids:
        requested = set(explicit_ids)
        if len(requested) != len(explicit_ids) or not requested <= {scenes[i]['scene_id'] for i in available}:
            raise ValueError('Scene IDs must be unique and belong to supported CAT or clutter families')
        return [i for i in available if scenes[i]['scene_id'] in requested]
    # Implicit selection remains CAT-only, including --scene-count 0.
    available = [i for i in available if group(scenes[i]) in GROUPS]
    if scene_count == 0:
        return available
    reference = [i for i in available if group(scenes[i]) == GROUPS[0]]
    procedural = [i for i in available if group(scenes[i]) == GROUPS[1]]
    if not len(reference) < scene_count <= len(available):
        raise ValueError(f'--scene-count must exceed {len(reference)} and be <= {len(available)}, or 0 for all; use --scene-ids for smaller subsets')
    return sorted(reference + random.Random(seed).sample(procedural, scene_count-len(reference)))


def plan_batch(indices, scenes, pool, poses_per_scene, width, seed, np):
    """Explicit immutable world -> local scene -> global scene/pose mapping.

    Each real scene contributes distinct poses exactly once. Only unused final
    batch slots are padded; their outcomes never enter the sample or reports.
    """
    worlds = []
    for local_id, global_id in enumerate(indices):
        unique = np.unique(np.asarray(pool[global_id]), axis=0, return_index=True)[1]
        if len(unique) < poses_per_scene:
            raise ValueError(f'{scenes[global_id]["scene_id"]}: requested {poses_per_scene} poses, only {len(unique)} distinct certified poses')
        scene_seed = int.from_bytes(hashlib.sha256(
            f'{seed}:{scenes[global_id]["scene_id"]}'.encode()).digest()[:8], 'little') & ((1 << 63)-1)
        chosen = np.random.default_rng(scene_seed).permutation(unique)[:poses_per_scene]
        for repeat, pose_index in enumerate(chosen.tolist()):
            worlds.append(dict(scene_index=global_id, local_scene_index=local_id,
                certified_pose_index=pose_index, repeat=repeat, seed=(scene_seed+repeat) % (1 << 63), padding=False))
    count = len(worlds)
    if not 0 < count <= width:
        raise ValueError('Scene batch exceeds world capacity')
    worlds.extend(dict(worlds[-1], padding=True) for _ in range(width-count))
    return worlds, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-a', type=Path, default=ROOT/'outputs/mjlab_migration_20260919/original-cat-expanded.npz')
    parser.add_argument('--policy-b', type=Path, default=ROOT/'outputs/cat_flat_balance_ppo_37632_20260920/resume.pt')
    parser.add_argument('--bank-manifest', type=Path, default=BANK/'manifest.json')
    parser.add_argument('--collision-manifest', type=Path)
    parser.add_argument('--reset-manifest', type=Path)
    parser.add_argument('--environment-config', type=Path, help='JSON environment mapping; otherwise B checkpoint contract')
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--num-envs', type=int, default=512, help='Maximum parallel worlds across resident scenes')
    parser.add_argument('--scenes-per-batch', type=int, default=16, help='Maximum simultaneously resident scenes')
    parser.add_argument('--scene-count', type=int, default=400,
                        help='All original/published plus a seeded uniform procedural sample; 0 selects all CAT')
    parser.add_argument('--poses-per-scene', '--episodes-per-scene', dest='episodes_per_scene', type=int, default=32,
                        help='Distinct certified poses per scene; no reuse or uncertified pose jitter')
    parser.add_argument('--check-interval', type=int, default=25, help='Steps between host checks; outcomes latch every step')
    parser.add_argument('--compile-task', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verify-replay-steps', type=int, default=2,
                        help='First-batch fixed-action replay check; 0 disables this extra check')
    parser.add_argument('--vram-budget-gib', type=float, default=8., help='Planning budget, not an allocator limit')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--scene-ids', nargs='+', help='Exact CAT or clutter scene IDs, overriding --scene-count')
    selection.add_argument('--scene-families', nargs='+',
                           choices=('original_cat', 'published_cat', 'procedural_cat', *CLUTTER_GROUPS),
                           help='All scenes in these families, overriding --scene-count; no adaptive sampling')
    parser.add_argument('--continue-after-clean-goal', action='store_true',
                        help='Observe physical termination after first success; default stops at first outcome')
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--max-steps', type=int, help='Optional common cap on manifest horizons')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--action-mode', choices=('deterministic', 'stochastic'), default='deterministic')
    parser.add_argument('--policy-id-a', type=int, default=0)
    parser.add_argument('--policy-id-b', type=int, default=0)
    args = parser.parse_args()
    if min(args.num_envs, args.episodes_per_scene, args.scenes_per_batch, args.check_interval) < 1 or (
            args.max_steps is not None and args.max_steps < 1) or args.verify_replay_steps < 0 or args.scene_count < 0:
        parser.error('Counts must be positive; scene-count/replay steps may be zero')
    if args.num_envs < args.episodes_per_scene:
        parser.error('--num-envs must accommodate all selected poses of at least one scene')
    args.bank_manifest = args.bank_manifest.resolve()
    args.collision_manifest = (args.collision_manifest or args.bank_manifest.parent.with_name(
        args.bank_manifest.parent.name + '_collision')/'manifest.json').resolve()
    args.reset_manifest = (args.reset_manifest or args.bank_manifest.parent.with_name(
        args.bank_manifest.parent.name + '_resets')/'manifest.json').resolve()
    # CPU-only manifest helpers must never create a second GPU runtime.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    # Fail before copying large checkpoints if the requested device is absent.
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this execution environment; no evaluation artifacts created')
        torch.cuda.set_device(device)
    from cat_mjlab.collision import CollisionChecker, PROPOSAL
    from cat_mjlab.scene_bank import SceneBank
    from cat_mjlab.task import CATTask
    from cat_mjlab.sim import CATSimulation
    from cat_mjlab.learning import ActorCritic, Learner, LearnerConfig, gaussian_parameters
    from cat_mjlab.conversion import load_array_archive
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    from cat_mjlab.passage_rewards import passage_parameters

    manifest = load_generalist_manifest(args.bank_manifest, verify_files=False)
    scenes = manifest['scenes']
    selected = select_scenes(scenes, args.scene_count, args.seed, args.scene_ids, args.scene_families)
    if not selected:
        raise ValueError('No supported scenes selected')
    for i in selected:
        s = scenes[i]
        expected_kind = 'room' if group(s) in CLUTTER_GROUPS else 'cat'
        if s.get('task_kind', 'cat') != expected_kind or s.get('reset_mode', 'cat') != expected_kind:
            raise ValueError(f'Unsupported scene semantics: {s["scene_id"]}')
        if 'episode_length' not in s or s.get('crossed_mode') not in ('x_plane', 'goal_radius'):
            raise ValueError(f'Missing explicit horizon/goal semantics: {s["scene_id"]}')
    reset_meta = json.loads(args.reset_manifest.read_text())
    collision_meta = json.loads(args.collision_manifest.read_text())
    for key, path in (('field_manifest_sha256', args.bank_manifest),
                      ('collision_bank_sha256', args.collision_manifest), ('proxy_sha256', PROPOSAL)):
        if reset_meta[key] != digest(path):
            raise ValueError(f'Reset certification mismatch: {key}')
    if [s['scene_id'] for s in collision_meta['scenes']] != [s['scene_id'] for s in scenes]:
        raise ValueError('Collision scene order differs from field bank')
    reset_file = args.reset_manifest.parent/reset_meta['file']
    if reset_meta.get('status') != 'complete' or digest(reset_file) != reset_meta['sha256']:
        raise ValueError('Incomplete or modified certified reset pool')
    pool = np.load(reset_file, mmap_mode='r', allow_pickle=False)
    if (pool.shape != (len(scenes), reset_meta['poses_per_scene'], 36)
            or pool.shape[1] < 1 or not np.isfinite(pool).all()):
        raise ValueError('Unexpected reset pool shape or nonfinite poses')
    if args.episodes_per_scene > pool.shape[1]:
        raise ValueError(f'Requested {args.episodes_per_scene} poses per scene, but only {pool.shape[1]} are certified; pose reuse is disabled')
    scene_capacity = min(args.scenes_per_batch, args.num_envs // args.episodes_per_scene, len(selected))
    num_envs = scene_capacity * args.episodes_per_scene
    batches = [selected[i:i+scene_capacity] for i in range(0, len(selected), scene_capacity)]
    plans = [plan_batch(indices, scenes, pool, args.episodes_per_scene, num_envs, args.seed, np) for indices in batches]
    peak_field_bytes = max(sum(int(np.prod(scenes[i]['shape']))*28 for i in indices) for indices in batches)
    estimated_vram = 1.7 + num_envs/1024 + peak_field_bytes/1024**3
    memory_allowance = 4.5 + 3*num_envs/1024 + peak_field_bytes/1024**3
    if not 0 < args.vram_budget_gib <= 8 or memory_allowance > args.vram_budget_gib-1:
        parser.error('Worlds plus resident fields exceed the planning budget with 1 GiB margin; reduce batch size')
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if free < (memory_allowance + 1) * 1024**3:
            raise RuntimeError(f'Need {memory_allowance + 1:.2f} GiB free including 1 GiB margin')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identities = {key: freeze(path.resolve(), args.output_dir/f'policy_{key}{path.suffix}')
                  for key, path in (('A', args.policy_a), ('B', args.policy_b))}

    def load_policy(key):
        path = Path(identities[key]['frozen'])
        if path.suffix == '.npz':
            learner = Learner(LearnerConfig(algorithm='ppo'), device='cpu')
            report = load_array_archive(learner, path, restore_optimizer=False)
            model, contract = learner.model, {}
            identities[key]['conversion'] = report
        else:
            # mmap avoids materializing the live run's enormous simulator/Adam
            # state; only the inference model is copied out of the frozen file.
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            state = payload.get('learner', payload)
            model = ActorCritic(LearnerConfig(**state['config']))
            model.load_state_dict(state['model'], strict=True)
            contract = copy.deepcopy(payload.get('contract', {}))
            if contract.get('normalize_observations') or state.get('normalize_observations'):
                raise ValueError('Observation-normalized checkpoints are unsupported')
            identities[key]['env_steps'] = state.get('env_steps', payload.get('step'))
        from cat_mjlab.observation_contract import ACTOR_SIZE
        actor_observation_for_policy(model.config, torch.empty(0, ACTOR_SIZE))
        identities[key]['observation_input'] = 'native_geometry_v2'
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError('Nonfinite policy weights')
        return model.eval().requires_grad_(False), contract

    models, contracts = {}, {}
    for key in ('A', 'B'):
        models[key], contracts[key] = load_policy(key)
        policy_id = getattr(args, 'policy_id_' + key.lower())
        if policy_id < 0 or policy_id >= (models[key].config.num_policies if models[key].config.algorithm == 'sapg' else 1):
            raise ValueError(f'Invalid policy ID for {key}')
    config = (json.loads(args.environment_config.read_text()) if args.environment_config else
              copy.deepcopy(contracts['B'].get('environment_config')))
    if config is None:
        raise ValueError('B has no environment_config; provide --environment-config explicitly')
    # Both actors use the same current CAT task. Room-only reward machinery is
    # disabled; room route navigation is loaded independently below.
    config.update(randomize_initial_episode_steps=False, wholebody_hand_contrast=False)
    config['wholebody']['body_collision'].update(enabled=True, bank_manifest=str(args.collision_manifest),
                                                reset_manifest=str(args.reset_manifest), proposal=str(PROPOSAL))
    models = {k: v.to(device) for k, v in models.items()}
    sim = CATSimulation(num_envs, device=str(device), nconmax=64, njmax=256)
    shared_collision = CollisionChecker(sim.model, args.collision_manifest,
                                       field_manifest=args.bank_manifest, device=device)

    if args.compile_task:
        shared_collision.enable_compilation()
    from cat_mjlab.fields import sample_ragged_field
    from cat_mjlab.compilation import compile_batched_kernel
    sample_kernel = (compile_batched_kernel(sample_ragged_field, batch_arg='pos')
                     if args.compile_task else sample_ragged_field)

    class PlannedResetPool:
        """Resolve the task's [local scene, random pose] lookup to planned rows.

        qpos contains the exact certified pose, already in its scene coordinates.
        The planned per-world scene IDs are supplied to EVERY reset, including
        the implicit constructor reset. Partial resets are prohibited.
        """
        def __init__(self, poses, local_ids, scene_count):
            self.shape = (scene_count, 1, 36)
            self.poses = torch.as_tensor(poses.copy(), device=device)
            self.local_ids = torch.as_tensor(local_ids, dtype=torch.long, device=device)

        def __getitem__(self, indices):
            if indices[0].shape != (num_envs,) or indices[1].shape != (num_envs,):
                raise ValueError('Planned certified resets require a full world batch')
            if not torch.equal(indices[0], self.local_ids):
                raise AssertionError('Reset scene IDs differ from the planned pose-to-world mapping')
            return self.poses

    class PackedSceneBank(SceneBank):
        """Same ragged layout/indexing as SceneBank; only this scene batch loads."""
        def __init__(self, indices, worlds):
            records = [scenes[i] for i in indices]
            self.path, self.device, self.count = args.bank_manifest, device, len(indices)
            self.manifest = {'scenes': records}
            tensor = lambda x, dtype=None: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
            self.global_ids = tensor(indices, torch.long)
            sizes = [int(np.prod(s['shape'])) for s in records]
            offsets = np.cumsum([0]+sizes[:-1], dtype=np.int64)
            self.fields = {}
            for name, channels in (('sdf', 1), ('bf', 3), ('gf', 3)):
                values = np.empty((sum(sizes), channels), dtype=np.float32)
                for record, offset, size in zip(records, offsets, sizes):
                    directory = scene_directory(manifest, args.bank_manifest, record)
                    field = directory/record['fields'][name]['file']
                    if digest(field) != record['fields'][name]['sha256']:
                        raise ValueError(f'Field hash mismatch: {field}')
                    array = np.load(field, allow_pickle=False, mmap_mode='r')
                    expected = tuple(record['shape']) + (() if channels == 1 else (channels,))
                    if array.dtype != np.float32 or array.shape != expected or not np.isfinite(array).all():
                        raise ValueError(f'Invalid field: {field}')
                    values[offset:offset+size] = array.reshape(-1, channels)
                self.fields[name] = tensor(values)
            self.sample_kernel = sample_kernel
            self.offsets = tensor(offsets, torch.long)
            for name, key, dtype in (
                ('shapes', 'shape', torch.long), ('origins', 'origin', torch.float32),
                ('dxs', 'dx', torch.float32), ('starts', 'start', torch.float32),
                ('reset_xy_scale', 'reset_xy_scale', torch.float32),
                ('reset_yaws', 'reset_yaw', torch.float32), ('goals', 'goal', torch.float32)):
                setattr(self, name, tensor([s[key] for s in records], dtype))
            self.is_cat = tensor([s.get('task_kind', 'cat') == 'cat' for s in records], torch.bool)
            self.reset_is_cat = tensor([s.get('reset_mode', 'cat') == 'cat' for s in records], torch.bool)
            self.crossed_is_plane = tensor([s['crossed_mode'] == 'x_plane' for s in records], torch.bool)
            self.episode_lengths = tensor([min(s['episode_length'], args.max_steps or s['episode_length']) for s in records], torch.long)
            self.weights = tensor([1.]*self.count, torch.float32)
            self.navigation_groups = tensor([0 if group(s) == GROUPS[0] else
                                             2 if group(s) in CLUTTER_GROUPS else 1 for s in records], torch.long)
            empty_rooms = []
            for record in records:
                if record.get('task_kind', 'cat') != 'room':
                    empty_rooms.append(None)
                    continue
                source = record.get('source', {})
                if (source.get('occupancy') != 'conservative-voxel-cell-OBB-intersection-v1'
                        or source.get('room_navigation') != 'ordered-certified-route-v1'):
                    raise ValueError('Room fields predate the conservative geometry/path fix')
                scene_path = scene_directory(manifest, args.bank_manifest, record)/'scene.json'
                if digest(scene_path) != record['scene_sha256']:
                    raise ValueError(f'Room scene hash mismatch: {scene_path}')
                room = json.loads(scene_path.read_text())
                from cat_ppo.furniture.room_geometry import root_cylinder_segment_clearance
                from cat_ppo.furniture.room_navigation import scene_navigation_radius
                route = np.asarray(room['route'])
                if np.any(root_cylinder_segment_clearance(route[:-1], route[1:], room['boxes'],
                                                        radius=scene_navigation_radius(room)) <= 0):
                    raise ValueError(f'Uncertified root route: {record["scene_id"]}')
                empty_rooms.append(room)
            self.rooms = {k: tensor(v) for k, v in pack_room_scenes(empty_rooms).items()}
            self.contrast = {k: tensor(v) for k, v in pack_hand_contrast(empty_rooms).items()}
            self.contrast.update({k: tensor(v) for k, v in passage_parameters(empty_rooms, bank_path=self.path).items()})
            self.has_contrast = self.has_sdf_reward_overrides = False
            self.balance_settings = self.width_curriculum = self.roles = self.levels = self.groups = None
            poses = np.stack([pool[w['scene_index'], w['certified_pose_index']] for w in worlds])
            self.reset_pool = PlannedResetPool(poses, [w['local_scene_index'] for w in worlds], self.count)

    class CertifiedCollision:
        def __init__(self, global_ids):
            self.global_ids, self.force_reset = global_ids, False

        def __call__(self, ids, data):
            if self.force_reset:
                self.force_reset = False
                return torch.ones((len(ids), 6), dtype=torch.bool, device=device)
            # Field IDs are local to the resident batch. Collision grids retain
            # the full bank's global IDs: never pass a local ID straight through.
            return shared_collision(self.global_ids[ids], data)

    class EvaluationTask(CATTask):
        rolling = False
        resetting = False

        def __init__(self, *positional, reset_seeds, **kwargs):
            self.reset_seeds = reset_seeds
            super().__init__(*positional, **kwargs)

        def _rand(self, shape, low=0., high=1.):
            if not self.resetting:
                # Fixed full-batch draws even after some worlds finish: neither
                # actions nor terminal timing change another world's RNG stream.
                return super()._rand(shape, low, high)
            if shape[0] != self.num_envs:
                raise ValueError('Selective reset would invalidate the seed/slot mapping')
            values = np.stack([rng.uniform(low, high, size=shape[1:]) for rng in self.reset_rngs]).astype('float32')
            return torch.as_tensor(values, device=self.device)

        def _observe(self, ids, contacts):
            if self.resetting:
                # Native reset draws phase directly from its batch generator;
                # replace it before observation assembly with each episode seed.
                left = self._rand((self.num_envs,)) < .5
                self.info['phase'] = torch.where(left[:, None],
                    self.data.qpos.new_tensor([0., np.pi]), self.data.qpos.new_tensor([np.pi, 0.]))
            return super()._observe(ids, contacts)

        def reset(self, env_ids=None, scene_ids=None):
            if self.rolling:
                return self.obs  # Scoring and parking own completed rows; no autoreset/RNG draws.
            if env_ids is not None and len(env_ids) != self.num_envs:
                raise ValueError('Evaluation only supports full-batch explicit resets')
            if scene_ids is None:
                scene_ids = self.bank.reset_pool.local_ids
            self.reset_rngs = [np.random.default_rng(seed) for seed in self.reset_seeds]
            self.collision.force_reset = True
            self.resetting = True
            try:
                return super().reset(env_ids, scene_ids)
            finally:
                self.resetting = False

    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return {k: cpu_tree(v) for k, v in value.items()}

    def tree_hash(value):
        h = hashlib.sha256()
        def visit(x):
            if isinstance(x, dict):
                for k in sorted(x):
                    h.update(k.encode()); visit(x[k])
            else:
                a = x.detach().cpu().contiguous().numpy()
                h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
        visit(value)
        return h.hexdigest()

    metadata = dict(schema='cat-paired-evaluation-v4', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    policies=identities, environment_config=config, num_envs=num_envs,
                    manifests={str(p): digest(p) for p in (args.bank_manifest, args.collision_manifest, args.reset_manifest)},
                    script_sha256=digest(__file__), task_source_sha256={p.name: digest(p) for p in sorted((ROOT/'cat_mjlab').glob('*.py'))},
                    selected_scenes=len(selected), excluded_families=dict(Counter(s['family'] for s in scenes if group(s) is None)),
                    resident_scene_capacity=scene_capacity, field_peak_mib=peak_field_bytes/1024**2,
                    selected_scene_ids=[scenes[i]['scene_id'] for i in selected],
                    sampling=('All scenes in explicitly selected families' if args.scene_families else
                              'Exact explicit scene IDs' if args.scene_ids else
                              'All CAT scenes' if args.scene_count == 0 else
                              'All original/published scenes plus seeded uniform procedural sample without replacement'),
                    sample_groups=dict(Counter(group(scenes[i]) for i in selected)),
                    collision_tensor_mib=sum(t.numel()*t.element_size() for t in shared_collision.bank.values())/1024**2,
                    base_plus_worlds_vram_gib=estimated_vram, conservative_allowance_gib=memory_allowance,
                    expected_vram_gib_at_512=[2.5, 4.],
                    memory_budget_gib=args.vram_budget_gib, certified_poses_per_scene=pool.shape[1],
                    expected_pairs_per_minute_at_512=[10, 30], ideal_pairs_per_minute=.12*num_envs,
                    scoring='First clean goal, failure or timeout per row; simultaneous failure wins; body_collision > fall > other; all flags retained',
                    reset_distribution='Distinct exact certified poses without replacement within each scene, with distinct per-episode randomization seeds',
                    dynamic_rng='Native full-batch Torch stream; batch seed and slot recorded. Batch width is part of replay identity.',
                    parking='Finished/padded worlds restored to their initial physical/task row each control step; RNG still draws full batches',
                    action_semantics=args.action_mode, versions={'torch': torch.__version__},
                    notes=['Original family includes one reconstructed-missing-original scene in the default live bank.',
                           'Evaluation uses current mjlab physics and collision rules for both actors.',
                           'VRAM and speed estimates are extrapolations, not measurements.',
                           'Collision certificates do not certify traversal bounds. Initial outside-bounds rows are explicit paired zero-step other outcomes.'])

    def write_metadata():
        (args.output_dir/'run.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')

    def park(active, current, baseline):
        # Explicit per-row masks: never touch a surviving world's state. Compiled
        # together to avoid launching one eager kernel per small history tensor.
        for target, source in zip(current, baseline):
            mask = active.reshape((-1,) + (1,) * (target.ndim - 1))
            target.copy_(torch.where(mask, target, source))

    def score(active, lengths, returns, codes, saved_flags, first_goals, physical_lengths, raw_timeouts,
              reward, flags, terminated, truncated, step_number, horizon):
        failed = terminated | flags[:, failure_columns].any(-1)
        success = flags[:, goal_column] & ~failed
        timed_out = truncated | (step_number >= horizon)
        first_goals = torch.where(active & success & (first_goals < 0), step_number, first_goals)
        first_done = active & (codes < 0) & (failed | success | timed_out)
        done = active & (failed | timed_out | (success & (not args.continue_after_clean_goal)))
        raw_timeouts = raw_timeouts | (active & timed_out)
        value = torch.where(failed, torch.where(flags[:, body_column], 1,
                            torch.where(flags[:, fall_column], 2, 4)), torch.where(success, 0, 3))
        return (active & ~done, torch.where(first_done, step_number, lengths),
                returns + torch.where(active, reward, 0.), torch.where(first_done, value, codes),
                torch.where(first_done[:, None], flags, saved_flags), first_goals,
                torch.where(done, step_number, physical_lengths), raw_timeouts)

    if args.compile_task:
        park = torch.compile(park, fullgraph=True, dynamic=True)
        # Step number is a scalar tensor to avoid one compiled graph per step.
        score = torch.compile(score, fullgraph=True, dynamic=True)

    write_metadata()
    print(json.dumps(dict(event='plan', worlds=num_envs, scenes=len(selected),
        pairs=len(selected)*args.episodes_per_scene, allowance_gib=memory_allowance,
        certified_poses_per_scene=pool.shape[1], scenes_per_batch=scene_capacity, note='No pose reuse within a scene; out-of-bounds certified starts are reported explicitly')), flush=True)
    metadata['replay_check'] = dict(enabled=bool(args.verify_replay_steps), steps=args.verify_replay_steps)
    metadata['metric_semantics'] = dict(
        time_to_first_clean_goal='Control steps to first clean goal; distribution conditional on reaching one; null otherwise',
        unsuccessful_timeout='Observed physical timeout (including evaluation horizon) with no prior clean goal; denominator all episodes',
        length_steps='Legacy first-outcome stopping length',
        physical_length_steps='Observed rollout length; successes censored unless continue-after-clean-goal is enabled',
        raw_timeout='Observed timeout; successful rows censored in default first-outcome mode')
    if not args.verify_replay_steps:
        metadata['notes'].append('Fixed-action replay verification disabled (--verify-replay-steps 0); known restore replay mismatch remains unverified.')
    write_metadata()
    statistics = new_statistics(GROUPS + tuple(g for g in CLUTTER_GROUPS if any(group(scenes[i]) == g for i in selected)))
    total_pairs = 0
    task = None
    started = time.monotonic()

    def write_summary(complete):
        elapsed = time.monotonic()-started
        summary = dict(groups=summarize(statistics, np), pairs=total_pairs, complete=complete,
            replay_verification_enabled=bool(args.verify_replay_steps),
            continue_after_clean_goal=args.continue_after_clean_goal, metric_semantics=metadata['metric_semantics'],
            replay_caveat=None if args.verify_replay_steps else metadata['notes'][-1],
            elapsed_seconds=elapsed, pairs_per_minute=60*total_pairs/max(elapsed, 1e-9),
            torch_peak_allocated_mib=(torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None),
            memory_note='Torch peak excludes Warp/context; memory budget is a planning estimate, not an allocator cap')
        temporary = args.output_dir/'summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2)+'\n')
        temporary.replace(args.output_dir/'summary.json')
        return summary

    with (args.output_dir/'paired.jsonl').open('x') as output, torch.inference_mode():
        for batch_index, (indices, (worlds, count)) in enumerate(zip(batches, plans)):
            batch_started = time.monotonic()
            bank = PackedSceneBank(indices, worlds)
            mapping_hash = hashlib.sha256(json.dumps(worlds, sort_keys=True).encode()).hexdigest()
            batch_seed = int.from_bytes(hashlib.sha256(
                f'{args.seed}:{mapping_hash}'.encode()).digest()[:8], 'little') & ((1 << 63)-1)
            seeds = [w['seed'] for w in worlds]
            if task is None:
                collision = CertifiedCollision(bank.global_ids)
                task = EvaluationTask(sim, bank, config, collision=collision, seed=batch_seed, reset_seeds=seeds)
                if args.compile_task:
                    task.enable_compilation()
                    bank.sample_kernel = sample_kernel
            else:
                task.bank = bank
                task.collision.global_ids = bank.global_ids
                # Scene-dependent statistics must be rebuilt: local scene IDs
                # have new meanings (and the final batch can have fewer scenes).
                task.scene_episode_ema = torch.zeros(bank.count, device=device)
                task.scene_success_ema = torch.zeros(bank.count, device=device)
                task.probabilities = bank.probabilities(stage=task.curriculum_stage)
                for name in ('navigation_counts', 'contrast_counts', 'role_counts',
                             'curriculum_completed', 'curriculum_goals'):
                    getattr(task, name).zero_()
            task.reset_seeds = seeds
            task.generator.manual_seed(batch_seed)
            task.reset(scene_ids=bank.reset_pool.local_ids)
            if not torch.equal(task.scene_ids, bank.reset_pool.local_ids):
                raise AssertionError('Task world-to-scene mapping differs from the batch plan')
            expected_global_ids = torch.tensor([w['scene_index'] for w in worlds], device=device)
            if not torch.equal(bank.global_ids[task.scene_ids], expected_global_ids):
                raise AssertionError('Local-to-global collision scene mapping differs from the batch plan')
            # Check qpos before collision/bounds so a faulty installation cannot
            # be mistaken for a defect in the certified pose bank.
            if not torch.equal(sim.data.qpos, bank.reset_pool.poses):
                raise AssertionError('Reset qpos is not the exact unperturbed certified pose in every world')
            collision_regions = task.collision(task.scene_ids, sim.data)
            initial_outside = task.episode['outside_bounds'].clone()
            reset_diagnostics = []
            for slot, (regions, outside) in enumerate(zip(collision_regions.cpu().tolist(), initial_outside.cpu().tolist())):
                if slot < count and (any(regions) or outside):
                    w = worlds[slot]
                    record = scenes[w['scene_index']]
                    upper = (np.asarray(record['origin'])+(np.asarray(record['shape'])-1)*record['dx']).tolist()
                    reset_diagnostics.append(dict(world_slot=slot, **w, scene_id=record['scene_id'],
                        collision_regions=dict(zip(('feet', 'legs', 'trunk', 'head', 'arms', 'hands'), regions)),
                        outside_bounds=outside, exact_certified_qpos=True,
                        root_xyz=sim.data.qpos[slot, :3].cpu().tolist(),
                        feet_xyz=task.info['positions'][slot, 3:5].cpu().tolist(),
                        field_lower=record['origin'], field_upper=upper))
            with (args.output_dir/'reset_diagnostics.jsonl').open('a') as diagnostics:
                diagnostics.write(json.dumps(dict(batch_index=batch_index, mapping_sha256=mapping_hash,
                    scenes=[scenes[i]['scene_id'] for i in indices], flagged_rows=reset_diagnostics))+'\n')
            if bool(collision_regions.any().item()):
                raise ValueError(f'Collision certificate mismatch in batch {batch_index}; see reset_diagnostics.jsonl. No poses were jittered, replaced or dropped.')
            if reset_diagnostics:
                print(json.dumps(dict(event='initial_outside_bounds', batch_index=batch_index,
                    count=len(reset_diagnostics), treatment='zero-step other for both policies; retained in primary denominator')), flush=True)
            initial = {'simulation': cpu_tree(sim.state_dict()), 'task': cpu_tree(task.state_dict())}
            initial_hash = tree_hash(initial)
            random.seed(batch_seed); np.random.seed(batch_seed % (1 << 32)); torch.manual_seed(batch_seed)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)

            def restore():
                task.rolling = False
                sim.reset_data()  # Clear every world's contact/solver scratch identically.
                sim.load_state_dict(initial['simulation'])
                task.load_state_dict(initial['task'])
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state(rng[3], device)
                if not torch.equal(torch.get_rng_state(), rng[2]) or (
                        rng[3] is not None and not torch.equal(torch.cuda.get_rng_state(device), rng[3])):
                    raise AssertionError('Global Torch RNG state differs after restoration')
                restored_hash = tree_hash({'simulation': sim.state_dict(), 'task': task.state_dict()})
                if restored_hash != initial_hash:
                    raise AssertionError('Full-batch simulator/task/RNG state differs after restoration')
                if not torch.equal(bank.global_ids[task.scene_ids], expected_global_ids):
                    raise AssertionError('Restored world-to-scene mapping differs')
                task.rolling = True
                return restored_hash

            if total_pairs == 0 and args.verify_replay_steps:
                replay = []
                for _ in range(2):
                    restore()
                    for _ in range(args.verify_replay_steps):
                        task.step(torch.zeros((num_envs, 29), device=device))
                    sim.capacity_report()
                    replay.append({'simulation': cpu_tree(sim.state_dict()), 'task': cpu_tree(task.state_dict())})
                if tree_hash(replay[0]) != tree_hash(replay[1]):
                    raise RuntimeError('Multi-scene fixed-action replay differs after restore; no policy results written')
                metadata['replay_check'] = dict(enabled=True, worlds=num_envs, scenes=len(indices), steps=args.verify_replay_steps,
                                              identical=True, state_sha256=tree_hash(replay[0]))
                write_metadata()
                del replay

            flag_names = list(initial['task']['episode'])
            failure_columns = [flag_names.index(k) for k in (
                'fall', 'obstacle', 'self_contact', 'numerical', 'body_collision',
                'hand_violation', 'elbow_violation', 'outside_bounds')]
            goal_column, body_column, fall_column = map(flag_names.index, ('goal_reached', 'body_collision', 'fall'))
            # Per-world horizons are indexed by LOCAL scene ID, just like fields.
            horizons = bank.episode_lengths[bank.reset_pool.local_ids]
            max_horizon = int(horizons.max().item())
            paths = [('simulation', name) for name in initial['simulation']]
            for name in ('info', 'navigation', 'contrast', 'episode', 'obs'):
                paths.extend(('task', name, key) for key in initial['task'][name])
            paths.extend(('task', name) for name in ('scene_ids', 'policy_ids', 'zone_steps', 'outcome_counted', 'episode_reward'))
            def lookup(tree, path):
                for key in path:
                    tree = tree[key]
                return tree
            baseline = tuple(lookup(initial, path).to(device) for path in paths)
            if any(t.ndim == 0 or t.shape[0] != num_envs for t in baseline):
                raise AssertionError('Parking attempted to include shared/non-world state')
            valid = torch.arange(num_envs, device=device) < count
            initially_failed = valid & initial_outside
            initial_flags = torch.stack([initial['task']['episode'][name] for name in flag_names], -1).to(device)
            eligible = valid & ~initial_outside
            has_eligible = bool(eligible.any().item())
            outcomes = {}
            for key in ('A', 'B'):
                restored_hash = restore()
                noise = torch.Generator(device=device).manual_seed(batch_seed ^ 0xCA7AB1)
                active = eligible.clone()
                lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
                first_goals = torch.full_like(lengths, -1)
                physical_lengths = torch.zeros_like(lengths)
                raw_timeouts = torch.zeros(num_envs, dtype=torch.bool, device=device)
                returns = torch.zeros(num_envs, device=device)
                codes = torch.where(initially_failed, 4, -1).long()
                saved_flags = torch.where(initially_failed[:, None], initial_flags, False)
                bad_action = torch.zeros((), dtype=torch.bool, device=device)
                step_number = torch.zeros((), dtype=torch.long, device=device)
                last_progress = time.monotonic()
                for length in range(1, (max_horizon if has_eligible else 0)+1):
                    # Parking never changes an active row or batch RNG draw
                    # shape. Shared scene-level metadata remains immutable.
                    live = {'simulation': vars(sim.data), 'task': vars(task)}
                    park(active, tuple(lookup(live, path) for path in paths), baseline)
                    policy_state = actor_observation_for_policy(models[key].config, task.obs['state'])
                    mean, scale = gaussian_parameters(models[key].logits(policy_state, getattr(args, 'policy_id_'+key.lower())))
                    raw = mean if args.action_mode == 'deterministic' else mean + scale*torch.randn(mean.shape, generator=noise, device=device)
                    bad_action |= (active & ~torch.isfinite(raw).all(-1)).any()
                    action = torch.where(active[:, None], torch.nan_to_num(raw.tanh()), 0.)
                    step = task.step(action)
                    flags = torch.stack([step['metrics']['episode/'+name] for name in flag_names], dim=-1)
                    step_number.fill_(length)
                    active, lengths, returns, codes, saved_flags, first_goals, physical_lengths, raw_timeouts = score(
                        active, lengths, returns, codes, saved_flags, first_goals, physical_lengths, raw_timeouts, step['reward'], flags,
                        step['terminated'], step['truncated'], step_number, horizons)
                    if length % args.check_interval == 0 or length == max_horizon:
                        sim.capacity_report()  # Device overflow was latched every physics substep.
                        if bool(bad_action.item()):
                            raise RuntimeError(f'Nonfinite active action: {key}, batch={batch_index}')
                        if not bool(active.any().item()):
                            break
                        if time.monotonic()-last_progress >= 60:
                            print(json.dumps(dict(event='rollout_progress', batch_index=batch_index,
                                policy=key, step=length, horizon=max_horizon,
                                completed_episodes=int((codes[:count] >= 0).sum().item()), batch_episodes=count)), flush=True)
                            last_progress = time.monotonic()
                task.rolling = False
                lengths_cpu, returns_cpu, codes_cpu, flags_cpu = [t.cpu().tolist() for t in (lengths, returns, codes, saved_flags)]
                if any(code < 0 for code in codes_cpu[:count]):
                    raise AssertionError('Unscored world in completed batch')
                first_cpu, physical_cpu, timeout_cpu = [t.cpu().tolist() for t in (first_goals, physical_lengths, raw_timeouts)]
                outcomes[key] = [dict(outcome=OUTCOMES[codes_cpu[slot]], length=lengths_cpu[slot],
                    seconds=lengths_cpu[slot]*task.dt, reward=returns_cpu[slot],
                    time_to_first_clean_goal=first_cpu[slot] if first_cpu[slot] >= 0 else None,
                    physical_length=physical_cpu[slot], raw_timeout=timeout_cpu[slot],
                    unsuccessful_timeout=timeout_cpu[slot] and first_cpu[slot] < 0,
                    flags=dict(zip(flag_names, flags_cpu[slot])), restored_state_sha256=restored_hash)
                    for slot in range(count)]
            outside_cpu = initial_outside.cpu().tolist()
            horizons_cpu = horizons.cpu().tolist()
            for slot, w in enumerate(worlds[:count]):
                index, repeat, seed = w['scene_index'], w['repeat'], w['seed']
                row = dict(episode_id=f'{scenes[index]["scene_id"]}:{repeat}:{seed}', scene_index=index,
                    scene_id=scenes[index]['scene_id'], local_scene_index=w['local_scene_index'],
                    family=scenes[index]['family'], group=group(scenes[index]),
                    scene_source=scenes[index].get('source'), seed=seed, repeat=repeat,
                    certified_pose_index=w['certified_pose_index'], initial_outside_bounds=outside_cpu[slot],
                    reset_status='certified_clear_but_outside_bounds' if outside_cpu[slot] else 'certified_clear_in_bounds',
                    batch_seed=batch_seed, batch_index=batch_index, batch_mapping_sha256=mapping_hash,
                    resident_scene_indices=indices, world_slot=slot, batch_width=num_envs, batch_valid_episodes=count,
                    reset_qpos=initial['simulation']['qpos'][slot].tolist(),
                    reset_qvel=initial['simulation']['qvel'][slot].tolist(),
                    initial_state_sha256=initial_hash, max_steps=horizons_cpu[slot],
                    outcomes={key: outcomes[key][slot] for key in ('A', 'B')})
                record_pair(statistics, row)
                total_pairs += 1
                output.write(json.dumps(row)+'\n')
            output.flush()
            elapsed = time.monotonic()-batch_started
            print(json.dumps(dict(event='batch_complete', batch_index=batch_index,
                scenes=[scenes[i]['scene_id'] for i in indices], pairs=count,
                total_pairs=total_pairs, batch_seconds=elapsed, batch_pairs_per_minute=60*count/elapsed,
                end_to_end_pairs_per_minute=60*total_pairs/(time.monotonic()-started))), flush=True)
            write_summary(complete=False)
            # Clear every old-bank reference BEFORE constructing the next one;
            # the compiled sampler takes ragged arrays explicitly as arguments.
            task.bank = None
            del bank, baseline, initial
    print(json.dumps(write_summary(complete=True), indent=2))


if __name__ == '__main__':
    main()
