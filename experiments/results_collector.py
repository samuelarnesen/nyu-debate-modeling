from agents import DebateRoundSummary
from experiments.annotator import Annotator
from experiments.experiment_loader import ExperimentConfig, ExperimentLoader
from experiments.quotes_collector import QuotesCollector
from utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize
import scipy.stats

from enum import Enum
from typing import Optional, Union, Any
import json
import math
import os
import uuid


class WinStats(BaseModel):
    matches: int = 0
    wins: int = 0
    correct_matches: int = 0
    correct_wins: int = 0
    first_matches: int = 0
    first_wins: int = 0


class JudgeStats(BaseModel):
    matches: int = 0
    correct_calls: int = 0
    first_calls: int = 0


class ResultsRow(BaseModel):
    first_debater_alias: str
    second_debater_alias: str
    first_debater_settings: dict[Any, Any]
    second_debater_settings: dict[Any, Any]
    judge_settings: dict[Any, Any]
    run_idx: int
    debate_identifier: str
    question: str
    first_debater_answer: str
    second_debater_answer: str
    first_debater_correct: bool
    second_debater_correct: bool
    first_debater_win_prob: float
    second_debater_win_prob: float
    first_debater_wins: bool
    second_debater_wins: bool
    correct_side_win_prob: float
    incorrect_side_win_prob: float
    correct_side_wins: bool


class ResultsCollector:
    def __init__(
        self,
        experiment: ExperimentConfig,
        graphs_path_prefix: Optional[str] = None,
        full_record_path_prefix: Optional[str] = None,
        stats_path_prefix: Optional[str] = None,
        should_save: bool = True,
    ):
        """
        Collects metrics after a series of debate rounds are run.

        Params:
            experiment: the config used to generate the debate rounds
            graphs_path_prefix: the directory and experiment name where the graphs are to be saved
            full_record_path_prefix: the directory and experiment name where the csv of the full run is to be saved
            stats_path_prefix: the directory and experiment name where the summary stats are to be saved
            should_save: whether or not to actually save the metrics
        """
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.quotes_collector = QuotesCollector(experiment=experiment) if experiment else None
        self.annotator = Annotator(model_path=experiment.annotations_classifier_file_path) if should_save else None
        self.num_debaters = (
            len(set([debater.model_settings.alias for debater in experiment.agents.debaters])) if experiment else 2
        )
        self.aliases = sorted(list(set([debater.model_settings.alias for debater in experiment.agents.debaters])))
        self.graphs_path_prefix = graphs_path_prefix
        self.full_record_path_prefix = full_record_path_prefix
        self.stats_path_prefix = stats_path_prefix
        self.should_save = should_save
        self.experiment = experiment
        self.summaries = []

        self.create_output_directories()

    def create_output_directories(self) -> None:
        """Creates the directories needed to output the log files"""

        def create_dir(path: Optional[str]) -> None:
            if path:
                parent_dir = os.path.dirname(path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

        create_dir(self.graphs_path_prefix)
        create_dir(self.full_record_path_prefix)
        create_dir(self.stats_path_prefix)

    def reset(self) -> None:
        """removes all the records of previous debate rounds"""
        self.summaries = []

    def __save_graph(self, name: str):
        if self.graphs_path_prefix and self.should_save:
            plt.savefig(f"{self.graphs_path_prefix}{name}.png")

    def record_result(self, summaries: DebateRoundSummary | list[DebateRoundSummary]) -> None:
        """Adds metrics from that debate round to local store"""
        summaries = summaries if type(summaries) == list else [summaries]
        for summary in summaries:
            self.summaries.append(summary)
            if self.quotes_collector:
                self.quotes_collector.record_result(summary=summary)
            if self.annotator:
                self.annotator.classify_debate_round(summary=summary)

    def __graph_judge(self) -> dict[str, float]:
        """Graphs the judge accuracy statistics"""
        matchup_to_stats = {}
        for summary in self.summaries:
            pair = "_v_".join(sorted([summary.first_debater_alias, summary.second_debater_alias]))
            if pair not in matchup_to_stats:
                matchup_to_stats[pair] = JudgeStats()
            matchup_to_stats[pair].matches += 1
            matchup_to_stats[pair].correct_calls += (
                summary.first_debater_win_prob if summary.metadata.first_debater_correct else summary.second_debater_win_prob
            )
            matchup_to_stats[pair].first_calls += summary.first_debater_win_prob

        fig, axs = plt.subplots(1, 2)

        axs[0].bar(matchup_to_stats.keys(), [val.correct_calls / val.matches for _, val in matchup_to_stats.items()])
        axs[0].set_xticklabels(matchup_to_stats.keys(), rotation="vertical")
        axs[0].xaxis.set_major_locator(ticker.FixedLocator(range(len(matchup_to_stats))))
        axs[0].set_title("Percent Correct")
        axs[0].set_ylim(0, 1)

        axs[1].bar(matchup_to_stats.keys(), [val.first_calls / val.matches for _, val in matchup_to_stats.items()])
        axs[1].set_xticklabels(matchup_to_stats.keys(), rotation="vertical")
        axs[1].xaxis.set_major_locator(ticker.FixedLocator(range(len(matchup_to_stats))))
        axs[1].set_title("Percent Chose First Debater")
        axs[1].set_ylim(0, 1)

        fig.suptitle("Judge Metrics")
        plt.tight_layout()
        self.__save_graph("Judge")
        plt.show()
        return matchup_to_stats

    def __graph_wins(self) -> dict[str, float]:
        def bayesian_credible_interval(wins: int, games: int, confidence: float = 0.95):
            if games == 0:
                return 0, 0
            alpha = (games / 2) + wins
            beta = (games / 2) + games - wins

            lower_bound = scipy.stats.beta.ppf((1 - confidence) / 2, alpha, beta)
            upper_bound = scipy.stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)

            return lower_bound, upper_bound

        alias_to_stats = {}

        for summary in self.summaries:
            if summary.first_debater_alias not in alias_to_stats:
                alias_to_stats[summary.first_debater_alias] = WinStats()
            if summary.second_debater_alias not in alias_to_stats:
                alias_to_stats[summary.second_debater_alias] = WinStats()

            alias_to_stats[summary.first_debater_alias].matches += 1
            alias_to_stats[summary.second_debater_alias].matches += 1

            alias_to_stats[summary.first_debater_alias].first_matches += 1

            alias_to_stats[summary.first_debater_alias].wins += summary.first_debater_win_prob
            alias_to_stats[summary.second_debater_alias].wins += summary.second_debater_win_prob

            alias_to_stats[summary.first_debater_alias].first_wins += summary.first_debater_win_prob

            if summary.metadata.first_debater_correct:
                alias_to_stats[summary.first_debater_alias].correct_matches += 1
                alias_to_stats[summary.first_debater_alias].correct_wins += summary.first_debater_win_prob
            else:
                alias_to_stats[summary.second_debater_alias].correct_matches += 1
                alias_to_stats[summary.second_debater_alias].correct_wins += summary.second_debater_win_prob

        categories = ["Overall", "Correct", "Incorrect", "First", "Second"]
        index = np.arange(len(categories))
        bar_width = 0.7 / self.num_debaters

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

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

            ax1.bar(index + (i * bar_width), values, bar_width, label=alias, yerr=assymetric_intervals)

        ax1.set_xticks(index + ((len(alias_to_stats) - 1) * bar_width) / self.num_debaters, categories)
        ax1.set_ylim(0, 1)
        ax1.set_title("Overall Win Rates")
        ax1.legend()

        win_rate_map = {}
        for summary in self.summaries:
            if summary.winning_alias not in win_rate_map:
                win_rate_map[summary.winning_alias] = {}
            if summary.losing_alias not in win_rate_map[summary.winning_alias]:
                win_rate_map[summary.winning_alias][summary.losing_alias] = 0
            win_rate_map[summary.winning_alias][summary.losing_alias] += 1

        win_rate_matrix = []
        for first in self.aliases:
            win_rate_matrix.append([])
            for second in self.aliases:
                wins = win_rate_map.get(first, {}).get(second, 0)
                losses = win_rate_map.get(second, {}).get(first, 0)
                win_rate_matrix[-1].append((wins / (wins + losses)) if (wins + losses > 0) else 0.5)

        sns.heatmap(win_rate_matrix, annot=True, fmt=".1%", cmap="coolwarm_r", cbar=False, ax=ax2)

        ax2.set_xticklabels(self.aliases)
        ax2.set_yticklabels(self.aliases)
        ax2.set_xlabel("Losing Team")
        ax2.set_ylabel("Winning Team")
        ax2.set_title("Head to Head Win Rates")

        plt.tight_layout()

        self.__save_graph("Win_Rates")
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        categories = [str(key) for key in debater_skills]
        values = [value for _, value in debater_skills.items()]
        ax1.bar(categories, values)
        ax1.set_title("Bradley-Terry Scores")

        computed_win_rate_matrix = []
        for first in categories:
            computed_win_rate_matrix.append([])
            for second in categories:
                computed_win_rate = (
                    debater_skills[first] / (debater_skills[first] + debater_skills[second])
                    if (debater_skills[first] + debater_skills[second]) > 0
                    else 0.5
                )
                computed_win_rate_matrix[-1].append(computed_win_rate)
        sns.heatmap(computed_win_rate_matrix, annot=True, fmt=".1%", cmap="coolwarm_r", cbar=False, ax=ax2)

        ax2.set_xticklabels(categories)
        ax2.set_yticklabels(categories)
        ax2.set_xlabel("Losing Team")
        ax2.set_ylabel("Winning Team")
        ax2.set_title("Computed Win Rates")
        self.__save_graph("Computed_Win_Rates")
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
                constants.OVERALL: win_stats.matches if self.num_debaters > 1 else int(win_stats.matches / 2),
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
        self.__save_graph("Quotes")
        plt.show()

        return results

    def __graph_features(self):
        average, lower, upper = self.annotator.get_results()

        common_tags = ["statement", "summary", "analysis", "quote", "q_context"]
        rare_tags = ["flourish", "framing", "refutation", "promise", "logic", "reply"]

        aliases = list(average.keys())

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        for i in range(2):
            tags = common_tags if i == 0 else rare_tags
            index = np.arange(len(tags))
            bar_width = 4 / (self.num_debaters * len(tags))
            for j, alias in enumerate(aliases):
                average_values = [average[alias][feature] for feature in tags]
                lower_errors = [max(average[alias][feature] - lower[alias][feature], 0) for feature in tags]
                upper_errors = [max(upper[alias][feature] - average[alias][feature], 0) for feature in tags]
                axs[i].bar(index + j * bar_width, average_values, bar_width, label=alias, yerr=[lower_errors, upper_errors])
            axs[i].set_ylabel("Frequency")
            axs[i].set_xticks(index + (bar_width * (len(aliases) - 1) / 2))
            axs[i].set_xticklabels(tags)
            axs[i].set_ylim(0)
            if i == 0:
                axs[i].set_title("Frequency of Attributes")
            else:
                axs[i].set_xlabel("Features")
            axs[i].legend()

        plt.show()
        self.__save_graph("Features")
        return {"average": average, "lower": lower, "upper": upper}

    def __organize_into_df(self):
        def construct_debater_settings(alias: str) -> dict[Any, Any]:
            for debater in filter(lambda x: x.model_settings.alias == alias, self.experiment.agents.debaters):
                return debater.dict()
            return None

        run_idx = uuid.uuid4()
        rows = []
        for summary in self.summaries:
            rows.append(
                ResultsRow(
                    first_debater_alias=summary.first_debater_alias,
                    second_debater_alias=summary.second_debater_alias,
                    first_debater_settings=construct_debater_settings(summary.first_debater_alias),
                    second_debater_settings=construct_debater_settings(summary.second_debater_alias),
                    judge_settings=self.experiment.agents.judge,
                    run_idx=run_idx,
                    debate_identifier=summary.metadata.debate_identifier,
                    question=summary.metadata.question,
                    first_debater_answer=summary.metadata.first_debater_answer,
                    second_debater_answer=summary.metadata.second_debater_answer,
                    first_debater_correct=summary.metadata.first_debater_correct,
                    second_debater_correct=not summary.metadata.first_debater_correct,
                    first_debater_win_prob=summary.first_debater_win_prob,
                    second_debater_win_prob=summary.second_debater_win_prob,
                    first_debater_wins=summary.first_debater_wins,
                    second_debater_wins=not summary.first_debater_wins,
                    correct_side_win_prob=summary.first_debater_win_prob
                    if summary.metadata.first_debater_correct
                    else summary.second_debater_win_prob,
                    incorrect_side_win_prob=summary.first_debater_win_prob
                    if not summary.metadata.first_debater_correct
                    else summary.second_debater_win_prob,
                    correct_side_wins=summary.first_debater_wins
                    if summary.metadata.first_debater_correct
                    else not summary.first_debater_wins,
                ).dict()
            )

        df = pd.DataFrame.from_dict(rows)
        if self.should_save and self.full_record_path_prefix:
            df.to_csv(f"{self.full_record_path_prefix}run.csv")

    def graph_results(self) -> None:
        """
        Graphs and displays the collected metrics.
        Currently supports:
            1. Raw Bradley-Terry Scores
            2. Win rates
            3. Judge accuracy and tendencies
            4. Quote statistics
            5. Stylistic info
        """

        all_stats = []

        bt_results = self.__graph_bradley_terry()
        all_stats.append(bt_results)
        self.logger.info(bt_results)

        plt.clf()
        win_results = self.__graph_wins()
        converted_win_results = {key: value.dict() for key, value in win_results.items()}
        all_stats.append(converted_win_results)
        self.logger.info(win_results)

        plt.clf()
        judge_results = self.__graph_judge()
        converted_judge_results = {key: value.dict() for key, value in judge_results.items()}
        all_stats.append(converted_judge_results)
        self.logger.info(judge_results)

        if self.quotes_collector:
            plt.clf()
            quote_results = self.__graph_quotes(win_stats_dict=win_results)
            converted_quote_results = {key: {k: v.dict() for k, v in value.items()} for key, value in quote_results.items()}
            all_stats.append(converted_quote_results)
            self.logger.info(quote_results)

        if self.annotator:
            plt.clf()
            classifier_results = self.__graph_features()
            converted_classifier_results = {key: value for key, value in classifier_results.items()}
            all_stats.append(converted_classifier_results)
            self.logger.info(classifier_results)

        self.__organize_into_df()

        if self.should_save and self.stats_path_prefix:
            with open(f"{self.stats_path_prefix}.json", "w") as f:
                json.dump(all_stats, f)
