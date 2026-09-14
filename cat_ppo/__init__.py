from cat_ppo.utils import registry
from cat_ppo.utils.logger import LOGGER, update_file_handler
from cat_ppo.constant import get_path_log, get_latest_ckpt

# Load simulation environments when the registry is queried, so scene generation
# and checkpoint inspection do not initialize every simulation dependency.
