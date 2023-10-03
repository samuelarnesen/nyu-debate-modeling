from script_utils import ScriptUtils

ScriptUtils.set_parent_as_path()

from experiments.experiment_loader import ExperimentLoader
from experiments.results_collector import GraphConfig, GraphType, PivotType, ResultsCollector

from datetime import datetime

args = ScriptUtils.get_args()
config = ScriptUtils.get_debate_round_script_config(args)
start_time = str(datetime.now()).replace(" ", "_")

results_collector = ResultsCollector(quotes_filepath=config.quotes_file_path)
debate_rounds = ExperimentLoader.generate_debate_rounds(
    experiment_file_path=config.experiment_file_path, name=config.experiment_name, count=args.num_iters
)

for i, debate_round in enumerate(debate_rounds):
    summary = debate_round.run(num_speeches=1, save_file_path=f"{config.save_path_base}/{start_time}_{i}.txt")
    results_collector.record_result(summary)

results_collector.graph_results(GraphConfig(graph_type=GraphType.BRADLEY_TERRY, labels=[PivotType.ORDER, PivotType.CORRECTNESS]))
results_collector.graph_results(GraphConfig(graph_type=GraphType.BAR, labels=[PivotType.ORDER, PivotType.CORRECTNESS]))
results_collector.graph_results(GraphConfig(graph_type=GraphType.QUOTES, labels=[PivotType.ORDER, PivotType.CORRECTNESS]))
