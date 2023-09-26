from script_utils import ScriptUtils

ScriptUtils.set_parent_as_path()

from experiments.experiment_loader import ExperimentLoader
from experiments.results_collector import GraphType, ResultsCollector

from datetime import datetime

args = ScriptUtils.get_args()
start_time = str(datetime.now()).replace(" ", "_")

experiment_name = "Test Experiment 2"
experiment_file_path = "experiments/configs/test_experiment.yaml"
save_path_base = "../debate-data/transcripts"
quotes_filepath = "../debate-data/quotes_dataset.p"
if not args.local:
    experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
    experiment_name = "Test Experiment 2 - HPC"
    save_path_base = "/home/spa9663/debate-data/transcripts"
    quotes_filepath = "/home/spa9663/debate-data/quotes_dataset.p"

results_collector = ResultsCollector(quotes_filepath=quotes_filepath)
debate_rounds = ExperimentLoader.generate_debate_rounds(
    experiment_file_path=experiment_file_path, name=experiment_name, count=args.num_iters
)

for i, debate_round in enumerate(debate_rounds):
    summary = debate_round.run(num_speeches=1, save_file_path=f"{save_path_base}/{start_time}_{i}.txt")
    results_collector.record_result(summary)

results_collector.graph_results(GraphType.ELO)
results_collector.graph_results(GraphType.BAR)
results_collector.graph_results(GraphType.QUOTES)
