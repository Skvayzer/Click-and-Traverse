"""Select transparent, reproducible good and bad examples from recorded trials.

The six displayed episodes illustrate behavior and are not an unbiased sample.
Report aggregate evaluation.json results separately from this selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for metadata_path in sorted(args.episodes.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        records.append(dict(metadata=metadata, directory=metadata_path.parent.resolve(),
                            metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest()))
    if not records:
        raise ValueError("No recorded episodes found")
    scenes = {number: [item for item in records if item["metadata"]["scene_seed"] == number]
              for number in (4001, 4002, 5001, 5002)}
    if any(not group for group in scenes.values()):
        raise ValueError("Expected recorded scenes 4001, 4002, 5001 and 5002")
    checkpoints = {record["metadata"]["checkpoint_steps"] for record in records}
    if len(checkpoints) != 1:
        raise ValueError("Cannot mix checkpoint steps in a demonstration")
    selected = []
    used = set()

    def choose(scene_seed, requested):
        candidates = [record for record in scenes[scene_seed] if record["directory"] not in used]
        matching = [record for record in candidates
                    if bool(record["metadata"]["outcome"]["goal_reached"]) == requested]
        fallback = not matching
        pool = matching or candidates
        if not pool:
            raise ValueError("Not enough distinct episodes for requested examples")
        successes = [record for record in pool if record["metadata"]["outcome"]["goal_reached"]]
        if successes:
            pool = sorted(successes, key=lambda record: (record["metadata"]["outcome"]["seconds"], record["metadata"]["seed"]))
            chosen = pool[(len(pool) - 1) // 2]
            rule = "Lower median duration among available successful episodes; seed breaks ties"
        else:
            long_enough = [record for record in pool if record["metadata"]["outcome"]["seconds"] >= 5.]
            pool = long_enough or pool
            def rank(record):
                outcome = record["metadata"]["outcome"]
                return (bool(outcome.get("hand_violation") or outcome.get("elbow_violation")),
                        bool(outcome.get("obstacle")), outcome["seconds"], -record["metadata"]["seed"])
            chosen = max(pool, key=rank)
            rule = ("Prefer failures lasting at least 5s when available; then hand/elbow violation, "
                    "other obstacle violation, longest duration, lowest seed")
        used.add(chosen["directory"])
        metadata = chosen["metadata"]
        success = bool(metadata["outcome"]["goal_reached"])
        prefix = "furniture" if metadata["family"] == "furniture" else "mixed_clutter"
        filename = f"{len(selected)+1:02d}_{prefix}{scene_seed}_{'success' if success else 'failure'}_seed{metadata['seed']:02d}.mp4"
        selected.append(dict(episode_dir=str(chosen["directory"]), video=str(args.video_dir.resolve() / filename),
                             scene_seed=scene_seed, seed=metadata["seed"], family=metadata["family"],
                             checkpoint_steps=metadata["checkpoint_steps"], outcome=metadata["outcome"],
                             requested_success=requested, fallback_needed=fallback,
                             selection_rule=rule, metadata_sha256=chosen["metadata_sha256"]))

    for number, success in ((4001, True), (4001, False), (4002, True), (4002, False), (5001, True), (5002, False)):
        choose(number, success)
    counts = {str(number): dict(episodes=len(group), successes=sum(bool(record["metadata"]["outcome"]["goal_reached"]) for record in group))
              for number, group in scenes.items()}
    result = dict(schema="cat-clutter-rollout-example-selection-v1", checkpoint_steps=checkpoints.pop(),
                  population=dict(total_episodes=len(records), scenes=counts), examples=selected,
                  interpretation="Purposefully selected demonstrations of success and failure, not an unbiased estimate of success rate",
                  requested="Good and bad furniture4001/4002, good mixed5001, bad mixed5002; if outcome absent show a truthfully labeled available episode",
                  selection_policy="Success: median duration. Failure: >=5s if possible, hand/elbow then other obstacle faults, longest duration, lowest seed. Never repeat an episode.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
