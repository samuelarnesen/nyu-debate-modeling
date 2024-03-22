from debate import DebateRoundSummary
from experiments.experiment_loader import ExperimentConfig, ExperimentLoader
from utils import LoggerUtils, QuoteUtils
import utils.constants as constants

from pydantic import BaseModel

import copy
import re
import sys


class QuoteStats(BaseModel):
    number_of_quotes: int
    number_of_valid_quotes: int
    total_valid_quote_length: int
    quote_length_to_accuracy: list[list[int]]


class QuotesCollector:
    MAX_TRACKED_QUOTE_LENGTH = 300

    def __init__(self, experiment: ExperimentConfig):
        """Collects metrics about quotation usage from debate rounds"""
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.dataset = ExperimentLoader.create_dataset(experiment)
        self.alias_to_results = {}

    def record_result(self, summary: DebateRoundSummary) -> None:
        """Records metrics on the use of quotations in the inputted debate round and stores it"""

        def add_new_alias(alias):
            default = QuoteStats(
                number_of_quotes=0,
                number_of_valid_quotes=0,
                total_valid_quote_length=0,
                quote_length_to_accuracy=[[0, 0] for i in range(QuotesCollector.MAX_TRACKED_QUOTE_LENGTH)],
            )
            self.alias_to_results[alias] = {}
            self.alias_to_results[alias][constants.OVERALL] = copy.deepcopy(default)
            self.alias_to_results[alias][constants.CORRECT] = copy.deepcopy(default)
            self.alias_to_results[alias][constants.WINNER] = copy.deepcopy(default)
            self.alias_to_results[alias][constants.LOSER] = copy.deepcopy(default)
            self.alias_to_results[alias][constants.INCORRECT] = copy.deepcopy(default)

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

        if summary.first_debater_alias not in self.alias_to_results:
            add_new_alias(summary.first_debater_alias)

        if summary.second_debater_alias not in self.alias_to_results:
            add_new_alias(summary.second_debater_alias)

        for speech in summary.transcript.speeches:
            outputted_quotes = QuoteUtils.extract_quotes(speech.content)
            alias = get_alias_from_speaker(speech.speaker)
            if alias == constants.DEFAULT_JUDGE_NAME:
                continue
            correct = is_correct(speech.speaker)
            winner = is_winner(speech.speaker)

            num_valid = 0
            total = 0
            for quote in outputted_quotes:
                total += 1
                quote_length = len(quote.split())
                if QuoteUtils.validate_quote(quote, summary.metadata.background_text, speech.content):
                    num_valid += 1
                    self.alias_to_results[alias][constants.OVERALL].number_of_valid_quotes += 1
                    self.alias_to_results[alias][constants.OVERALL].total_valid_quote_length += quote_length
                    self.alias_to_results[alias][constants.OVERALL].quote_length_to_accuracy[quote_length][0] += 1
                    if winner:
                        self.alias_to_results[alias][constants.WINNER].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.WINNER].total_valid_quote_length += quote_length
                        self.alias_to_results[alias][constants.WINNER].quote_length_to_accuracy[quote_length][0] += 1
                    else:
                        self.alias_to_results[alias][constants.LOSER].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.LOSER].total_valid_quote_length += quote_length
                        self.alias_to_results[alias][constants.LOSER].quote_length_to_accuracy[quote_length][0] += 1
                    if correct:
                        self.alias_to_results[alias][constants.CORRECT].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.CORRECT].total_valid_quote_length += quote_length
                        self.alias_to_results[alias][constants.CORRECT].quote_length_to_accuracy[quote_length][0] += 1
                    if not correct:
                        self.alias_to_results[alias][constants.INCORRECT].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.INCORRECT].total_valid_quote_length += quote_length
                        self.alias_to_results[alias][constants.INCORRECT].quote_length_to_accuracy[quote_length][0] += 1
                else:
                    self.logger.debug("The following quote was invalid:\n{}".format(quote))

                self.alias_to_results[alias][constants.OVERALL].number_of_quotes += 1
                self.alias_to_results[alias][constants.OVERALL].quote_length_to_accuracy[quote_length][1] += 1
                if winner:
                    self.alias_to_results[alias][constants.WINNER].number_of_quotes += 1
                    self.alias_to_results[alias][constants.WINNER].quote_length_to_accuracy[quote_length][1] += 1
                if not winner:
                    self.alias_to_results[alias][constants.LOSER].number_of_quotes += 1
                    self.alias_to_results[alias][constants.LOSER].quote_length_to_accuracy[quote_length][1] += 1
                if correct:
                    self.alias_to_results[alias][constants.CORRECT].number_of_quotes += 1
                    self.alias_to_results[alias][constants.CORRECT].quote_length_to_accuracy[quote_length][1] += 1
                if not correct:
                    self.alias_to_results[alias][constants.INCORRECT].number_of_quotes += 1
                    self.alias_to_results[alias][constants.CORRECT].quote_length_to_accuracy[quote_length][1] += 1

    def get_results(self) -> dict[str, dict[str, QuoteStats]]:
        """
        Returns the stored results

        Returns:
            alias_to_results: a dictionary that maps a model alias to another dictionary, where the keys are different
                slices of the data (e.g 'overall', 'winner', 'correct') and the values are raw counts.
        """
        simplified_results = {}
        for alias in self.alias_to_results:
            simplified_results[alias] = copy.deepcopy(self.alias_to_results[alias])
            for key in simplified_results[alias]:
                vals = [
                    idx
                    for idx, pair in filter(
                        lambda x: x[1][1] > 0, enumerate(simplified_results[alias][key].quote_length_to_accuracy)
                    )
                ]
                max_val = max(vals) if vals else 0
                simplified_results[alias][key].quote_length_to_accuracy = simplified_results[alias][
                    key
                ].quote_length_to_accuracy[: (max_val + 1)]
        return simplified_results
