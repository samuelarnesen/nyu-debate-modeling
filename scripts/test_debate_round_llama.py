from script_utils import ScriptUtils

ScriptUtils.set_parent_as_path()

from experiments.experiment_loader import ExperimentLoader
from experiments.results_collector import GraphType, ResultsCollector

from datetime import datetime

import sys

args = ScriptUtils.get_args()
results_collector = ResultsCollector()
start_time = str(datetime.now()).replace(" ", "_")

experiment_name = args.configuration or "SFT Experiment Extended - Local"
experiment_file_path = "experiments/configs/sft_experiment.yaml"
save_path_base = "../debate-data/transcripts"
if not args.local:
    experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
    experiment_name = args.configuration or "SFT Experiment Extended - HPC"
    save_path_base = "/home/spa9663/debate-data/transcripts"

debate_rounds = ExperimentLoader.generate_debate_rounds(
    experiment_file_path=experiment_file_path, name=experiment_name, count=args.num_iters
)

for i, debate_round in enumerate(debate_rounds):
    debater_one_wins = debate_round.run(num_speeches=1, save_file_path=f"{save_path_base}/{start_time}_{i}.txt")
    results_collector.record_result(1 if debater_one_wins else 2)

results_collector.graph_results(GraphType.ELO)
results_collector.graph_results(GraphType.BAR)
