from agents.debate_round import DebateRoundSummary
from data.data import DataRow
from data.loaders.quality_debates_loader import QualityDebatesLoader, QualityDebatesDataset
from experiments.experiment_loader import ExperimentConfig, ExperimentLoader
from experiments.quotes_collector import QuotesCollector
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.optimize
import scipy.stats

from enum import Enum
from typing import Optional, Union
import math
import re


class WinStats(BaseModel):
    matches: int
    wins: int
    correct_matches: int
    correct_wins: int
    first_matches: int
    first_wins: int


class JudgeStats(BaseModel):
    matches: int
    correct_calls: int
    first_calls: int


class ResultsCollector:
    def __init__(
        self, experiment: Optional[ExperimentConfig], save_file_path_prefix: Optional[str] = None, should_save: bool = True
    ):
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.quotes_collector = QuotesCollector(experiment=experiment) if experiment else None
        self.num_debaters = len(set([debater.alias for debater in experiment.agents.debaters])) if experiment else 2
        self.save_file_path_prefix = save_file_path_prefix
        self.should_save = should_save
        self.summaries = []

    def reset(self) -> None:
        self.summaries = []

    def __save(self, name: str):
        if self.save_file_path_prefix and self.should_save:
            plt.savefig(f"{self.save_file_path_prefix}{name}.png")

    def record_result(self, summaries: Union[DebateRoundSummary, list[DebateRoundSummary]]) -> None:
        summaries = summaries if type(summaries) == list else [summaries]
        for summary in summaries:
            self.summaries.append(summary)
            self.quotes_collector.record_result(summary=summary)

    def __get_num_debaters(self):
        return len(set([debater.alias for debater in self.experiment.agents.debaters]))

    def __graph_judge(self) -> dict[str, float]:
        alias_to_stats = {}
        for summary in self.summaries:
            if summary.judge_alias not in alias_to_stats:
                alias_to_stats[summary.judge_alias] = JudgeStats(matches=0, correct_calls=0, first_calls=0)

            alias_to_stats[summary.judge_alias].matches += 1

            if (summary.metadata.first_debater_correct and summary.first_debater_wins) or (
                not summary.metadata.first_debater_correct and not summary.first_debater_wins
            ):
                alias_to_stats[summary.judge_alias].correct_calls += 1

            if summary.first_debater_wins:
                alias_to_stats[summary.judge_alias].first_calls += 1

        fig, axs = plt.subplots(1, 2)

        axs[0].bar(alias_to_stats.keys(), [val.correct_calls / val.matches for _, val in alias_to_stats.items()])
        axs[0].set_title("Percent Correct")
        axs[0].set_ylim(0, 1)

        axs[1].bar(alias_to_stats.keys(), [val.first_calls / val.matches for _, val in alias_to_stats.items()])
        axs[1].set_title("Percent Chose First Debater")
        axs[1].set_ylim(0, 1)

        fig.suptitle("Judge Metrics")
        plt.tight_layout()
        self.__save("Judge")
        plt.show()
        return alias_to_stats

    def __graph_wins(self) -> dict[str, float]:
        def bayesian_credible_interval(wins: int, games: int, confidence: float = 0.95):
            alpha = (games / 2) + wins
            beta = (games / 2) + games - wins

            lower_bound = scipy.stats.beta.ppf((1 - confidence) / 2, alpha, beta)
            upper_bound = scipy.stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)

            return lower_bound, upper_bound

        alias_to_stats = {}
        for summary in self.summaries:
            if summary.first_debater_alias not in alias_to_stats:
                alias_to_stats[summary.first_debater_alias] = WinStats(
                    matches=0, wins=0, correct_matches=0, correct_wins=0, first_matches=0, first_wins=0
                )
            if summary.second_debater_alias not in alias_to_stats:
                alias_to_stats[summary.second_debater_alias] = WinStats(
                    matches=0, wins=0, correct_matches=0, correct_wins=0, first_matches=0, first_wins=0
                )

            alias_to_stats[summary.first_debater_alias].matches += 1
            alias_to_stats[summary.second_debater_alias].matches += 1
            if summary.first_debater_wins:
                alias_to_stats[summary.first_debater_alias].wins += 1
                alias_to_stats[summary.first_debater_alias].first_wins += 1
                alias_to_stats[summary.first_debater_alias].first_matches += 1
                if summary.metadata.first_debater_correct:
                    alias_to_stats[summary.first_debater_alias].correct_matches += 1
                    alias_to_stats[summary.first_debater_alias].correct_wins += 1
                else:
                    alias_to_stats[summary.second_debater_alias].correct_matches += 1
            else:
                alias_to_stats[summary.second_debater_alias].wins += 1
                alias_to_stats[summary.first_debater_alias].first_matches += 1
                if not summary.metadata.first_debater_correct:
                    alias_to_stats[summary.second_debater_alias].correct_matches += 1
                    alias_to_stats[summary.second_debater_alias].correct_wins += 1
                else:
                    alias_to_stats[summary.first_debater_alias].correct_matches += 1

        categories = ["Overall", "Correct", "Incorrect", "First", "Second"]
        index = np.arange(len(categories))
        bar_width = 0.7 / self.num_debaters

        for i, alias in enumerate(alias_to_stats):
            stats = alias_to_stats[alias]
            values = [
                stats.wins / max(stats.matches, 1),
                stats.correct_wins / max(stats.correct_matches, 1),
                (stats.wins - stats.correct_wins) / max((stats.matches - stats.correct_matches), 1),
                stats.first_wins / max(stats.first_matches, 1),
                (stats.wins - stats.first_wins) / max((stats.matches - stats.first_matches), 1),
            ]

            intervals = [
                bayesian_credible_interval(stats.wins, stats.matches),
                bayesian_credible_interval(stats.correct_wins, stats.correct_matches),
                bayesian_credible_interval((stats.wins - stats.correct_wins), (stats.matches - stats.correct_matches)),
                bayesian_credible_interval(stats.first_wins, stats.first_matches),
                bayesian_credible_interval((stats.wins - stats.first_wins), (stats.matches - stats.first_matches)),
            ]
            assymetric_intervals = [
                [abs(values[i] - interval[0]) for i, interval in enumerate(intervals)],
                [abs(interval[1] - values[i]) for i, interval in enumerate(intervals)],
            ]

            plt.bar(index + (i * bar_width), values, bar_width, label=alias, yerr=assymetric_intervals)

        plt.title("Win Rates")
        plt.xticks(index + ((len(alias_to_stats) - 1) * bar_width) / self.num_debaters, categories)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        self.__save("Win_Rates")
        plt.show()
        return alias_to_stats

    def __graph_bradley_terry(self) -> dict[str, float]:
        def log_likelihood(params, indices):
            log_likelihood = 0
            for summary in self.summaries:
                winner_idx = indices[summary.winning_alias]
                loser_idx = indices[summary.losing_alias]
                log_likelihood += params[winner_idx] - np.logaddexp(params[winner_idx], params[loser_idx])
            return -log_likelihood

        init_params = np.zeros(self.num_debaters)

        indices = {}
        for summary in self.summaries:
            if summary.first_debater_alias not in indices:
                indices[summary.first_debater_alias] = len(indices)
            if summary.second_debater_alias not in indices:
                indices[summary.second_debater_alias] = len(indices)

        debater_skills = {alias: 0 for alias in indices}
        if len(indices) > 1:
            optimal_params = scipy.optimize.minimize(lambda x: log_likelihood(x, indices), init_params, method="BFGS").x
            debater_skills = {debater: math.exp(skill) for debater, skill in zip(indices, optimal_params)}

        categories = [str(key) for key in debater_skills]
        values = [value for _, value in debater_skills.items()]
        plt.bar(categories, values)
        plt.title("Bradley-Terry Scores")
        self.__save("BT")

        plt.show()
        plt.clf()

        computed_win_rate_matrix = []
        for first in categories:
            computed_win_rate_matrix.append([])
            for second in categories:
                computed_win_rate = debater_skills[first] / (debater_skills[first] + debater_skills[second])
                computed_win_rate_matrix[-1].append(computed_win_rate)
        ax = sns.heatmap(computed_win_rate_matrix, annot=True, fmt=".1%", cmap="coolwarm_r", cbar=False)

        ax.set_xticklabels(categories)
        ax.set_yticklabels(categories)
        ax.set_xlabel("Losing Team")
        ax.set_ylabel("Winning Team")
        ax.set_title("Computed Win Rates")
        self.__save("Computed_Win_Rates")
        plt.show()

        return debater_skills

    def __graph_quotes(self, win_stats_dict: dict[str, WinStats]) -> dict[int, int]:
        def get_accuracy(results, name, key):
            return (
                results[name][key].number_of_valid_quotes / results[name][key].number_of_quotes
                if results[name][key].number_of_quotes > 0
                else 0
            )

        results = self.quotes_collector.get_results()
        all_categories = [constants.OVERALL, constants.WINNER, constants.LOSER, constants.CORRECT, constants.INCORRECT]

        category_to_counts = {}
        for name in results:
            win_stats = win_stats_dict[name]
            category_to_counts[name] = {
                constants.OVERALL: win_stats.matches,
                constants.WINNER: win_stats.wins,
                constants.LOSER: win_stats.matches - win_stats.wins,
                constants.CORRECT: win_stats.correct_matches,
                constants.INCORRECT: win_stats.matches - win_stats.correct_matches,
            }

        quote_accuracy = {}
        total_quote_counts = {}
        valid_quote_counts = {}
        valid_quote_word_counts = {}
        for key in all_categories:
            quote_accuracy[key] = {name: get_accuracy(results, name, key) for name in results}
            total_quote_counts[key] = {name: results[name][key].number_of_quotes for name in results}
            valid_quote_counts[key] = {name: results[name][key].number_of_valid_quotes for name in results}
            valid_quote_word_counts[key] = {name: results[name][key].total_valid_quote_length for name in results}

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        index = np.arange(len(results))
        bar_width = 0.3 / self.num_debaters

        for i, key in enumerate(all_categories):
            axs[0, 0].set_ylim(0, 1)
            axs[0, 0].bar(index + (i * bar_width), [item for _, item in quote_accuracy[key].items()], bar_width, label=key)
            axs[0, 0].set_xticks(index + (bar_width * (len(all_categories) - 1)) / self.num_debaters, results.keys())
            axs[0, 0].set_title("Valid Quote Percentage")
            axs[0, 0].legend()

            axs[0, 1].bar(
                index + (i * bar_width),
                [item / max(category_to_counts[alias][key], 1) for alias, item in total_quote_counts[key].items()],
                bar_width,
                label=key,
            )
            axs[0, 1].set_xticks(index + (bar_width * (len(all_categories) - 1)) / self.num_debaters, results.keys())
            axs[0, 1].set_title("Total Quotes")
            axs[0, 1].legend()

            axs[1, 0].bar(
                index + (i * bar_width),
                [item / max(category_to_counts[alias][key], 1) for alias, item in valid_quote_counts[key].items()],
                bar_width,
                label=key,
            )
            axs[1, 0].set_xticks(index + (bar_width * (len(all_categories) - 1)) / self.num_debaters, results.keys())
            axs[1, 0].set_title("Valid Quotes")
            axs[1, 0].legend()

            axs[1, 1].bar(
                index + (i * bar_width),
                [item / max(category_to_counts[alias][key], 1) for alias, item in valid_quote_word_counts[key].items()],
                bar_width,
                label=key,
            )
            axs[1, 1].set_xticks(index + (bar_width * (len(all_categories) - 1)) / self.num_debaters, results.keys())
            axs[1, 1].set_title("Valid Quote Word Count")
            axs[1, 1].legend()

        fig.suptitle("Quotes")
        plt.tight_layout()
        plt.legend()
        self.__save("Quotes")
        plt.show()

        return results

    def graph_results(self) -> None:

        bt_results = self.__graph_bradley_terry()
        self.logger.info(bt_results)

        plt.clf()
        win_results = self.__graph_wins()
        self.logger.info(win_results)

        plt.clf()
        quote_results = self.__graph_quotes(win_stats_dict=win_results)
        self.logger.info(quote_results)

        plt.clf()
        judge_results = self.__graph_judge()
        self.logger.info(judge_results)
