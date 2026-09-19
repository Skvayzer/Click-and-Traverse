"""Load unchanged robot constants without registering the legacy JAX tasks."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "cat_ppo/envs/g1/constants.py"
_spec = importlib.util.spec_from_file_location("_cat_mjlab_original_constants", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("_")})
