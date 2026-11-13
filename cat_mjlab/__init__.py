"""CAT whole-body tasks on mjlab's MuJoCo Warp simulation.

The original MJX implementation remains the numerical reference. Importing this
package does not initialize CUDA, download assets, or start an experiment.
"""

# CAT_PATH_MAP (HPC): redirect file access to the dep-1 absolute paths pinned in data manifests to the local mirror, for
# every entry point that imports this package (training, evaluation, recording). A no-op when the variable is unset.
from .pathmap import install as _install_path_map

_install_path_map()
