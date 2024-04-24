from script_utils import ScriptUtils

ScriptUtils.setup_script()

from experiments import ExperimentLoader, ResultsCollector
from utils import logger_utils

from tqdm import tqdm

from datetime import datetime

args = ScriptUtils.get_args()
config = ScriptUtils.get_debate_round_script_config(args)

debate_rounds, experiment = ExperimentLoader.generate_debate_rounds(
    experiment_file_path=config.experiment_file_path, name=config.experiment_name, count=args.num_iters
)

debate_rounds[0].first_debater.model.tokenizer.push_to_hub("samarnesen/nyu-debater-1r-dpo")
