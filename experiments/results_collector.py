from utils.logger_utils import LoggerUtils

import matplotlib.pyplot as plt

from enum import Enum


class GraphType(Enum):
    BAR = 0
    ELO = 1


class ResultsCollector:
    def __init__(self):
        self.results = []
        self.logger = LoggerUtils.get_default_logger(__name__)

    def reset(self) -> None:
        self.results = []

    def record_result(self, result: int) -> None:
        self.results.append(result)

    def __graph_bar(self) -> dict[int, int]:
        results_dict = {1: 0, 2: 0}
        for entry in self.results:
            results_dict[entry] += 1
        categories = [str(key) for key in results_dict]
        values = [value for _, value in results_dict.items()]

        plt.bar(categories, values)
        plt.show()
        return results_dict

    def __graph_elo(self) -> dict[int, float]:
        k = 16
        default = 1_000
        elo_one = [default]
        elo_two = [default]
        for result in self.results:
            expected_one = 1 / (1 + 10 ** ((elo_two[-1] - elo_one[-1]) / 400))
            expected_two = 1 - expected_one
            delta = (expected_two * k) if result == 1 else (expected_one * k)
            elo_one.append(elo_one[-1] + ((1 if result == 1 else -1) * delta))
            elo_two.append(elo_two[-1] + ((1 if result == 2 else -1) * delta))

        x_axis = [i for i in range(len(elo_one))]
        plt.plot(x_axis, elo_one, label="Debater_One")
        plt.plot(x_axis, elo_two, label="Debater_Two")
        plt.legend()
        plt.show()
        return {1: elo_one[-1], 2: elo_two[-1]}

    def graph_results(self, graph_type: GraphType) -> None:
        if graph_type == GraphType.BAR:
            results = self.__graph_bar()
            self.logger.info(results)
        elif graph_type == GraphType.ELO:
            results = self.__graph_elo()
            self.logger.info(results)
        else:
            raise Exception(f"Graph type {graph_type} is not implemented")
