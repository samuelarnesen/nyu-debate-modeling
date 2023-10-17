from agents.debate_round import DebateRoundSummary
from data.data import DataRow
from data.loaders.quality_debates_loader import QualityDebatesLoader, QualityDebatesDataset
from experiments.experiment_loader import ExperimentConfig, ExperimentLoader
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from enum import Enum
from typing import Optional, Union
import re


class QuoteStats(BaseModel):
    number_of_quotes: int
    number_of_valid_quotes: int
    total_valid_quote_length: int


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
        bar_width = 0.35

        for i, alias in enumerate(alias_to_stats):
            stats = alias_to_stats[alias]
            values = [
                stats.wins / max(stats.matches, 1),
                stats.correct_wins / max(stats.correct_matches, 1),
                (stats.wins - stats.correct_wins) / max((stats.matches - stats.correct_matches), 1),
                stats.first_wins / max(stats.first_matches, 1),
                (stats.wins - stats.first_wins) / max((stats.matches - stats.first_matches), 1),
            ]
            plt.bar(index + (i * bar_width), values, bar_width, label=alias)

        plt.title("Win Rates")
        plt.xticks(index + ((len(alias_to_stats) - 1) * bar_width) / 2, categories)
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

        init_params = np.zeros(2)

        indices = {}
        for summary in self.summaries:
            if summary.first_debater_alias not in indices:
                indices[summary.first_debater_alias] = len(indices)
            if summary.second_debater_alias not in indices:
                indices[summary.second_debater_alias] = len(indices)

        debater_skills = {alias: 0 for alias in indices}
        if len(indices) > 1:
            optimal_params = scipy.optimize.minimize(lambda x: log_likelihood(x, indices), init_params, method="BFGS").x
            debater_skills = {debater: skill for debater, skill in zip(indices, optimal_params)}

        categories = [str(key) for key in debater_skills]
        values = [value for _, value in debater_skills.items()]
        plt.bar(categories, values)
        plt.title("Bradley-Terry Scores")
        self.__save("BT")
        plt.show()

        return debater_skills

    def __graph_quotes(self) -> dict[int, int]:
        def get_accuracy(results, name, key):
            return (
                results[name][key].number_of_valid_quotes / results[name][key].number_of_quotes
                if results[name][key].number_of_quotes > 0
                else 0
            )

        results = self.quotes_collector.get_results()
        all_categories = [constants.OVERALL, constants.WINNER, constants.LOSER, constants.CORRECT, constants.INCORRECT]

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
        bar_width = 0.15

        for i, key in enumerate(all_categories):
            axs[0, 0].set_ylim(0, 1)
            axs[0, 0].bar(index + (i * bar_width), [item for _, item in quote_accuracy[key].items()], bar_width, label=key)
            axs[0, 0].set_xticks(index + (bar_width * (len(all_categories) - 1)) / 2, results.keys())
            axs[0, 0].set_title("Valid Quote Percentage")
            axs[0, 0].legend()

            axs[0, 1].bar(
                index + (i * bar_width),
                [item / len(self.summaries) for _, item in total_quote_counts[key].items()],
                bar_width,
                label=key,
            )
            axs[0, 1].set_xticks(index + (bar_width * (len(all_categories) - 1)) / 2, results.keys())
            axs[0, 1].set_title("Total Quotes")
            axs[0, 1].legend()

            axs[1, 0].bar(
                index + (i * bar_width),
                [item / len(self.summaries) for _, item in valid_quote_counts[key].items()],
                bar_width,
                label=key,
            )
            axs[1, 0].set_xticks(index + (bar_width * (len(all_categories) - 1)) / 2, results.keys())
            axs[1, 0].set_title("Valid Quotes")
            axs[1, 0].legend()

            axs[1, 1].bar(
                index + (i * bar_width),
                [item / len(self.summaries) for _, item in valid_quote_word_counts[key].items()],
                bar_width,
                label=key,
            )
            axs[1, 1].set_xticks(index + (bar_width * (len(all_categories) - 1)) / 2, results.keys())
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

        win_results = self.__graph_wins()
        self.logger.info(win_results)

        quote_results = self.__graph_quotes()
        self.logger.info(quote_results)

        judge_results = self.__graph_judge()
        self.logger.info(judge_results)


class QuotesCollector:
    def __init__(self, experiment: ExperimentConfig):
        self.dataset = ExperimentLoader.create_dataset(experiment)
        self.split = ExperimentLoader.get_split(experiment)
        self.alias_to_results = {}

    def record_result(self, summary: DebateRoundSummary) -> None:
        def add_new_alias(alias):
            self.alias_to_results[alias] = {}
            self.alias_to_results[alias][constants.OVERALL] = QuoteStats(
                number_of_quotes=0, number_of_valid_quotes=0, total_valid_quote_length=0
            )
            self.alias_to_results[alias][constants.CORRECT] = QuoteStats(
                number_of_quotes=0, number_of_valid_quotes=0, total_valid_quote_length=0
            )
            self.alias_to_results[alias][constants.WINNER] = QuoteStats(
                number_of_quotes=0, number_of_valid_quotes=0, total_valid_quote_length=0
            )
            self.alias_to_results[alias][constants.LOSER] = QuoteStats(
                number_of_quotes=0, number_of_valid_quotes=0, total_valid_quote_length=0
            )
            self.alias_to_results[alias][constants.INCORRECT] = QuoteStats(
                number_of_quotes=0, number_of_valid_quotes=0, total_valid_quote_length=0
            )

        def is_correct(speaker: str):
            return (speaker == constants.DEFAULT_DEBATER_A_NAME and summary.metadata.first_debater_correct) or (
                speaker != constants.DEFAULT_DEBATER_A_NAME and not summary.metadata.first_debater_correct
            )

        def is_winner(speaker: str):
            return (speaker == constants.DEFAULT_DEBATER_A_NAME and summary.first_debater_wins) or (
                speaker != constants.DEFAULT_DEBATER_A_NAME and not summary.first_debater_wins
            )

        def get_alias_from_speaker(speaker: str):
            if speech.speaker == constants.DEFAULT_DEBATER_A_NAME:
                return summary.first_debater_alias
            elif speech.speaker == constants.DEFAULT_DEBATER_B_NAME:
                return summary.second_debater_alias
            else:
                return constants.DEFAULT_JUDGE_NAME

        def simplify_text(text: str):
            return text.replace(",", "").replace(".", "").replace('"', "").replace("'", "").lower()

        if summary.first_debater_alias not in self.alias_to_results:
            add_new_alias(summary.first_debater_alias)

        if summary.second_debater_alias not in self.alias_to_results:
            add_new_alias(summary.second_debater_alias)

        data = self.dataset.get_example(idx=summary.metadata.question_idx, split=self.split)
        background_text = data.background_text
        for speech in summary.transcript.speeches:
            outputted_quotes = re.findall("<quote>(.*?)</quote>", speech.content)
            alias = get_alias_from_speaker(speech.speaker)
            if alias == constants.DEFAULT_JUDGE_NAME:
                continue
            correct = is_correct(speech.speaker)
            winner = is_winner(speech.speaker)
            for quote in outputted_quotes:
                if quote in background_text or simplify_text(quote) in simplify_text(background_text):
                    self.alias_to_results[alias][constants.OVERALL].number_of_valid_quotes += 1
                    self.alias_to_results[alias][constants.OVERALL].total_valid_quote_length += len(quote.split())
                    if winner:
                        self.alias_to_results[alias][constants.WINNER].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.WINNER].total_valid_quote_length += len(quote.split())
                    else:
                        self.alias_to_results[alias][constants.LOSER].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.LOSER].total_valid_quote_length += len(quote.split())
                    if correct:
                        self.alias_to_results[alias][constants.CORRECT].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.CORRECT].total_valid_quote_length += len(quote.split())
                    else:
                        self.alias_to_results[alias][constants.INCORRECT].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.INCORRECT].total_valid_quote_length += len(quote.split())

                self.alias_to_results[alias][constants.OVERALL].number_of_quotes += 1
                if alias == summary.winning_alias:
                    self.alias_to_results[alias][constants.WINNER].number_of_quotes += 1
                if alias == summary.losing_alias:
                    self.alias_to_results[alias][constants.LOSER].number_of_quotes += 1
                if correct:
                    self.alias_to_results[alias][constants.CORRECT].number_of_quotes += 1
                else:
                    self.alias_to_results[alias][constants.INCORRECT].number_of_quotes += 1

    def get_results(self):
        return self.alias_to_results
