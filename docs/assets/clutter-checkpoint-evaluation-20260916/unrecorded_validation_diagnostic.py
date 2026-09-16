"""One unmodified-validator diagnostic for the frozen clutter checkpoint."""
import functools
import json
import os
from pathlib import Path
import sys
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"
ROOT = Path("/home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody")
SOURCE = ROOT / "outputs/sources/cat_compact_stabilized_20260916"
OUTPUT = ROOT / "analysis/clutter-checkpoint-evaluation-20260916"
sys.path.insert(0, str(SOURCE))
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import ConfigDict
from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
from cat_ppo.furniture.learning import load_native
from cat_ppo.furniture.retention_validation import RetentionValidator
from cat_ppo.learning.policy.ppo.wholebody_distribution import make_ppo_networks, checkpoint_distribution_config

started = time.monotonic()
run = json.loads((OUTPUT / "run.json").read_text())
selected = json.loads((OUTPUT / "checkpoint/selection.json").read_text())
config = wholebody_config(ConfigDict(run["config"]["env_config"]),
                         bank_manifest=OUTPUT / "bank/manifest.json", stabilization=True)
env = G1CatWholeBodyEnv(config=config)
assert env.observation_contract() == run["observation_contract"]
network_config = json.loads((OUTPUT / "checkpoint/native/ppo_network_config.json").read_text())
assert not network_config["normalize_observations"]
assert checkpoint_distribution_config(network_config) is not None
factory = functools.partial(make_ppo_networks, **network_config["network_factory_kwargs"])
loaded = load_native(OUTPUT / "checkpoint/native")
params = jax.tree.map(jp.asarray, loaded)
dtype_checks = [dict(shape=list(np.shape(a)), loaded_dtype=str(a.dtype), device_dtype=str(b.dtype),
                    equal=bool(np.array_equal(np.asarray(a), np.asarray(b))))
               for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params))]
assert all(check["equal"] for check in dtype_checks)
baseline = json.loads((OUTPUT / "validation_baseline.json").read_text())["result"]["modes"]
validator = RetentionValidator(env, factory, chunk_steps=25)
print(json.dumps(dict(event="started", devices=[str(d) for d in jax.devices()],
                      count=validator.count, step=selected["step"],
                      dtype_pairs=sorted(set((x["loaded_dtype"], x["device_dtype"]) for x in dtype_checks)))), flush=True)
result = validator.evaluate(params, step=selected["step"], baseline=baseline).as_dict()
result["metadata"]["diagnostic"] = "Unmodified RetentionValidator(chunk_steps=25).evaluate; no trajectory recording"
result["metadata"]["dtype_checks"] = dtype_checks
result["metadata"]["elapsed_seconds_including_setup"] = time.monotonic() - started
(OUTPUT / "unrecorded-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(event="finished", modes={mode: {
    key: values[key] for key in ("clutter_goal_success_rate", "cat_goal_success_rate", "episode_count")}
    for mode, values in result["modes"].items()}, selection=result["selection"],
    elapsed_seconds=time.monotonic() - started)), flush=True)
