from script_utils import set_parent_as_path

set_parent_as_path()

from experiments.experiment_loader import ExperimentLoader
from experiments.results_collector import GraphType, ResultsCollector

results_collector = ResultsCollector()

debate_round = ExperimentLoader.generate_debate_round(
    experiment_file_path="./experiments/configs/test_experiment.yaml", name="Test Experiment 1"
)

for i in range(1_000):
    debate_round.reset()
    debater_one_wins = debate_round.run(num_speeches=3)
    results_collector.record_result(1 if debater_one_wins else 2)

results_collector.graph_results(GraphType.ELO)
results_collector.graph_results(GraphType.BAR)

