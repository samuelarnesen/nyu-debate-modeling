from agents.debate_round import DebateRoundSummary
from data.data import DataRow
from data.loaders.quality_debates_quotes_loader import QualityDebatesQuotesDataset, QualityDebatesQuotesLoader
from utils.logger_utils import LoggerUtils
import utils.constants as constants

import matplotlib.pyplot as plt

from enum import Enum
from typing import Optional


class GraphType(Enum):
    BAR = 0
    ELO = 1
    QUOTES = 2


class ResultsCollector:
    def __init__(self, quotes_filepath: Optional[str]):
        self.results = []
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.quotes_collector = QuotesCollector(quotes_filepath=quotes_filepath) if quotes_filepath else None

    def reset(self) -> None:
        self.results = []

    def record_result(self, summary: DebateRoundSummary) -> None:
        self.results.append(1 if summary.debater_one_wins else 2)
        self.quotes_collector.record_result(summary=summary)

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
        plt.plot(x_axis, elo_one, label=constants.DEFAULT_DEBATER_ONE_NAME)
        plt.plot(x_axis, elo_two, label=constants.DEFAULT_DEBATER_TWO_NAME)
        plt.legend()
        plt.show()
        return {constants.DEFAULT_DEBATER_ONE_NAME: elo_one[-1], constants.DEFAULT_DEBATER_TWO_NAME: elo_two[-1]}

    def __graph_quotes(self) -> dict[int, int]:
        results = self.quotes_collector.get_results()
        result_dict = {
            constants.DEFAULT_DEBATER_ONE_NAME: results[0][0] / results[0][1] if results[0][1] else 0,
            constants.DEFAULT_DEBATER_TWO_NAME: results[1][0] / results[1][1] if results[1][1] else 0,
        }
        plt.ylim(0, 1)
        plt.plot(result_dict.keys(), [item for _, item in result_dict.items()])
        plt.show()
        return result_dict

    def graph_results(self, graph_type: GraphType) -> None:
        if graph_type == GraphType.BAR:
            results = self.__graph_bar()
            self.logger.info(results)
        elif graph_type == GraphType.ELO:
            results = self.__graph_elo()
            self.logger.info(results)
        elif graph_type == GraphType.QUOTES:
            results = self.__graph_quotes()
            self.logger.info(results)
        else:
            raise Exception(f"Graph type {graph_type} is not implemented")


class QuotesCollector:
    def __init__(self, quotes_filepath: str):
        self.quotes_dataset = QualityDebatesQuotesLoader.load(full_dataset_filepath=quotes_filepath)
        self.results = [[0, 0], [0, 0]]

    def record_result(self, summary: DebateRoundSummary) -> None:
        def get_count(speech: str, target: str) -> tuple[float, int]:
            count = 0
            split_target = target.split("\n")
            for quote in split_target:
                count += 1 if quote in speech else 0
            return (count / len(split_target), len(split_target)) if split_target else (0, 0)

        def get_total_quote_count(row: DataRow, summary: DebateRoundSummary) -> tuple[int, int]:
            counts = [[0, 0], [0, 0]]
            for speech in summary.transcript.speeches[0:2]:
                if speech.speaker == constants.DEFAULT_DEBATER_ONE_NAME:
                    match, total = get_count(speech=speech, target=row.speeches[0].text)
                    counts[0][0] += match
                    counts[0][1] += total
                elif speech.speaker == constants.DEFAULT_DEBATER_TWO_NAME:
                    match, total = get_count(speech=speech, target=row.speeches[1].text)
                    counts[1][0] += match
                    counts[1][1] += total
            return counts

        matching_row = self.quotes_dataset.get_example(split=summary.split, idx=summary.question_idx)
        quote_count = get_total_quote_count(row=matching_row, summary=summary)
        for i in range(len(quote_count)):
            for j in range(len(quote_count[i])):
                self.results[i][j] += quote_count[i][j]

    def get_results(self):
        return self.results
