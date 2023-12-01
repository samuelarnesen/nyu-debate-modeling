from agents import DebateRoundSummary
from experiments.experiment_loader import ExperimentConfig, ExperimentLoader
from utils import LoggerUtils, QuoteUtils
import utils.constants as constants

from pydantic import BaseModel

import re


class QuoteStats(BaseModel):
    number_of_quotes: int
    number_of_valid_quotes: int
    total_valid_quote_length: int


class QuotesCollector:
    def __init__(self, experiment: ExperimentConfig):
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.dataset = ExperimentLoader.create_dataset(experiment)
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
            only_one_alias = summary.winning_alias == summary.losing_alias

            for quote in outputted_quotes:
                if QuoteUtils.validate_quote(quote, summary.metadata.background_text, speech.content):
                    self.alias_to_results[alias][constants.OVERALL].number_of_valid_quotes += 1
                    self.alias_to_results[alias][constants.OVERALL].total_valid_quote_length += len(quote.split())
                    if winner or only_one_alias:
                        self.alias_to_results[alias][constants.WINNER].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.WINNER].total_valid_quote_length += len(quote.split())
                    if not winner or only_one_alias:
                        self.alias_to_results[alias][constants.LOSER].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.LOSER].total_valid_quote_length += len(quote.split())
                    if correct or only_one_alias:
                        self.alias_to_results[alias][constants.CORRECT].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.CORRECT].total_valid_quote_length += len(quote.split())
                    if not correct or only_one_alias:
                        self.alias_to_results[alias][constants.INCORRECT].number_of_valid_quotes += 1
                        self.alias_to_results[alias][constants.INCORRECT].total_valid_quote_length += len(quote.split())
                else:
                    self.logger.debug("The following quote was invalid:\n{}".format(quote))

                self.alias_to_results[alias][constants.OVERALL].number_of_quotes += 1
                if winner or only_one_alias:
                    self.alias_to_results[alias][constants.WINNER].number_of_quotes += 1
                if not winner or only_one_alias:
                    self.alias_to_results[alias][constants.LOSER].number_of_quotes += 1
                if correct or only_one_alias:
                    self.alias_to_results[alias][constants.CORRECT].number_of_quotes += 1
                if not correct or only_one_alias:
                    self.alias_to_results[alias][constants.INCORRECT].number_of_quotes += 1

    def get_results(self):
        return self.alias_to_results
