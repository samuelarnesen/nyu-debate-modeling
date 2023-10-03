from agents.debate_round import DebateRoundSummary
from data.data import DataRow
from data.loaders.quality_debates_quotes_loader import QualityDebatesQuotesDataset, QualityDebatesQuotesLoader
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from enum import Enum
from typing import Optional


class GraphType(Enum):
    BAR = 0
    BRADLEY_TERRY = 1
    QUOTES = 2
    ELO = 3

class PivotType(Enum):
    ORDER = 0
    CORRECTNESS = 1 # TODO: add Name

class GraphConfig(BaseModel):
    graph_type: GraphType
    labels: list[PivotType]    


class ResultsCollector:
    def __init__(self, quotes_filepath: Optional[str]):
        self.results = []
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.quotes_collector = QuotesCollector(quotes_filepath=quotes_filepath) if quotes_filepath else None
        self.summaries = []

    def reset(self) -> None:
        self.results = []
        self.summaries = []

    def record_result(self, summary: DebateRoundSummary) -> None:
        self.results.append(1 if summary.debater_one_wins else 2)
        self.summaries.append(summary)
        self.quotes_collector.record_result(summary=summary)


    def __graph_bar(self, config: GraphConfig) -> dict[str, float]:

        def get_winner_for_order_and_correctness(summary: DebateRoundSummary) -> str:
            if summary.debater_one_wins and summary.metadata.debater_one_correct:
                return "1_Correct"
            elif summary.debater_one_wins:
                return "1_Incorrect"
            elif not summary.debater_one_wins and not summary.metadata.debater_one_correct:
                return "2_Correct"
            return "2_Incorrect"

        def get_winner_for_order(summary: DebateRoundSummary) -> str:
            return  "1" if self.debater_one_wins else "2"

        def get_winner_for_correctness(summary: DebateRoundSummary) -> str:
            return "Correct" if summary.debater_one_wins == summary.metadata.debater_one_correct else "Incorrect"


        winner_func = None
        results_dict = {}
        if PivotType.ORDER in config.labels and PivotType.CORRECTNESS in config.labels:
            results_dict = {"1_Correct": 0, "2_Correct": 0, "1_Incorrect": 0, "2_Incorrect": 0}
            winner_func = get_winner_for_order_and_correctness
        elif PivotType.ORDER in config.labels:
            results_dict = {"1": 0, "2": 0}
            winner_func = get_winner_for_order
        elif PivotType.CORRECTNESS in config.labels:
            results_dict = {"Correct": 0, "Incorrect": 0}
            winner_func = get_winner_for_order

        for summary in self.summaries:
            tag = winner_func(summary)
            results_dict[tag] += 1

        categories = [str(key) for key in results_dict]
        values = [value for _, value in results_dict.items()]

        plt.bar(categories, values)
        plt.show()
        return results_dict

    def __graph_bradley_terry(self) -> dict[str, float]:

        def log_likelihood(params, indices):
            log_likelihood = 0
            for summary in self.summaries:
                winner_idx = indices["1"] if summary.debater_one_wins else indices["2"]
                loser_idx = indices["2"] if summary.debater_one_wins else indices["1"]
                log_likelihood += params[winner_idx] - np.logaddexp(params[winner_idx], params[loser_idx])
            return -log_likelihood

        init_params = np.zeros(2)
        indices = {"1": 0, "2": 1}
        optimal_params = scipy.optimize.minimize(lambda x: log_likelihood(x, indices), init_params, method='BFGS').x
        debater_skills = {debater: skill for debater, skill in zip(indices, optimal_params)}

        categories = [str(key) for key in debater_skills]
        values = [value for _, value in debater_skills.items()]
        plt.bar(categories, values)
        plt.show()

        return debater_skills

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

    def graph_results(self, config: GraphConfig) -> None:
        if config.graph_type == GraphType.BAR:
            results = self.__graph_bar(config)
            self.logger.info(results)
        elif config.graph_type == GraphType.ELO:
            results = self.__graph_elo()
            self.logger.info(results)
        elif config.graph_type == GraphType.QUOTES:
            results = self.__graph_quotes()
            self.logger.info(results)
        elif config.graph_type == GraphType.BRADLEY_TERRY:
            results = self.__graph_bradley_terry()
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

        matching_row = self.quotes_dataset.get_example(split=summary.metadata.split, idx=summary.metadata.question_idx)
        quote_count = get_total_quote_count(row=matching_row, summary=summary)
        for i in range(len(quote_count)):
            for j in range(len(quote_count[i])):
                self.results[i][j] += quote_count[i][j]

    def get_results(self):
        return self.results
