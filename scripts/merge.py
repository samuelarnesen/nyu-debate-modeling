from script_utils import ScriptUtils

ScriptUtils.setup_script()

from utils import save_utils

save_utils.save(
    base_model_name="/vast/spa9663/models/base_models/llama3-8b-262k",
    adapter_name="/vast/spa9663/models/trained_models/llama-3-mega",
    merge_name="/vast/spa9663/models/trained_models/llama-3-mega-merged",
)
