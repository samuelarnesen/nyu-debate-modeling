from script_utils import ScriptUtils

ScriptUtils.set_parent_as_path()

from experiments.experiment_loader import ExperimentLoader
from experiments.results_collector import GraphType, ResultsCollector

args = ScriptUtils.get_args()
results_collector = ResultsCollector()

experiment_name = "Test Experiment 2"
experiment_file_path = "experiments/configs/test_experiment.yaml"
if not args.local:
    experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
    experiment_name = "Test Experiment 2 - HPC"

debate_round = ExperimentLoader.generate_debate_round(experiment_file_path=experiment_file_path, name=experiment_name)

for i in range(1_000):
    debate_round.reset()
    debater_one_wins = debate_round.run(num_speeches=3)
    results_collector.record_result(1 if debater_one_wins else 2)

results_collector.graph_results(GraphType.ELO)
results_collector.graph_results(GraphType.BAR)
