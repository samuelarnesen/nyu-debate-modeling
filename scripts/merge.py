from script_utils import ScriptUtils

ScriptUtils.setup_script()

from utils import save_utils

save_utils.save(
    base_model_name="/vast/spa9663/models/trained_models/mixtral-8x7b-unified-merged",
    adapter_name="/vast/spa9663/models/trained_models/mixtral-8x7b-dpo-1",
    merge_name="/vast/spa9663/models/trained_models/mixtral-8x7b-dpo-current",
)
