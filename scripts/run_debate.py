from script_utils import ScriptUtils

ScriptUtils.setup_script()

from experiments.experiment_loader import ExperimentLoader
from experiments.results_collector import ResultsCollector
from utils.logger_utils import LoggerUtils

from datetime import datetime
from tqdm import tqdm

args = ScriptUtils.get_args()
config = ScriptUtils.get_debate_round_script_config(args)
start_time = str(datetime.now()).replace(" ", "_")
logger = LoggerUtils.get_default_logger(__name__)

debate_rounds, experiment = ExperimentLoader.generate_debate_rounds(
    experiment_file_path=config.experiment_file_path, name=config.experiment_name, count=args.num_iters
)
results_collector = ResultsCollector(
    experiment=experiment,
    save_file_path_prefix=f"{config.save_path_base}/{start_time}_",
    should_save=(not args.local),
)

for i, debate_round in tqdm(enumerate(debate_rounds)):
    logger.debug(f"Beginning round {i} out of {len(debate_rounds)}")
    save_file_path_prefix = f"{config.save_path_base}/{start_time}_{i}" if not args.local else None
    summary = debate_round(save_file_path_prefix=save_file_path_prefix)
    results_collector.record_result(summary)

if not args.suppress_graphs and not args.bon:
    results_collector.graph_results()
