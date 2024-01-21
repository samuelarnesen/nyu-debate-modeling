from agents.agent import Agent
from agents.debater import Debater
from agents.judge import Judge, JudgeType
from agents.models import ModelResponse
from agents.transcript import Transcript
from prompts import Prompt, PromptConfig, PromptParser
from utils import LoggerUtils, QuoteUtils
import utils.constants as constants

from pydantic import BaseModel

from enum import Enum
from typing import Optional, Any, Union
import copy
import random


class QuestionMetadata(BaseModel):
    first_debater_correct: bool
    question_idx: int
    background_text: str
    question: str
    first_debater_answer: str
    second_debater_answer: str
    debate_identifier: str


class DebateRoundSummary(BaseModel):
    metadata: QuestionMetadata
    transcript: Any
    winning_alias: str
    losing_alias: str
    first_debater_alias: str
    second_debater_alias: str
    first_debater_wins: bool
    judge_alias: str
    winning_debater_prob: float = 1.0
    first_debater_win_prob: float = 0.5
    second_debater_win_prob: float = 0.5
    failed: bool = False


class SplittingRule(Enum):
    OPENING_ONLY = 1
    ALL_RANDOM = 2


class DebateRound:
    def __init__(
        self,
        first_debater: Debater,
        second_debater: Debater,
        judge: Judge,
        metadata: QuestionMetadata | list[QuestionMetadata],
    ):
        """An abstraction that coordinates the ordered generation of speeches by the debaters and the judge."""
        self.first_debater = first_debater
        self.second_debater = second_debater
        self.judge = judge
        self.metadata = metadata if type(metadata) == list else [metadata]
        self.name_to_agent = {
            self.first_debater.name: self.first_debater,
            self.second_debater.name: self.second_debater,
            self.judge.name: self.judge,
        }
        self.logger = LoggerUtils.get_default_logger(__name__)

    def set_first_debater(self, debater: Debater):
        """Changes the identity of the first debater in the debate."""
        self.first_debater = debater
        self.name_to_agent[self.first_debater.name] = debater

    def set_second_debater(self, debater: Debater):
        """Changes the identity of the second debater in the debate."""
        self.second_debater = debater
        self.name_to_agent[self.second_debater.name] = debater

    def set_judge(self, judge: Judge):
        """Changes the identity of the judge in the debate."""
        self.judge = judge
        self.name_to_agent[self.judge.name] = judge

    def run_round(self) -> tuple[list[str], ModelResponse]:
        """
        Each debater generates speeches until the judge renders their decision.

        Returns:
            last_output: a list of strings with the name of the agent that won each debate in the batch
            last_model_output: the model generation from the judge's decision. This is useful if the judge
                also returns the probability that a given debater won.
        """
        last_output = None
        last_model_output = None
        next_speaker = self.judge.get_next_expected_speaker()
        while next_speaker:
            speaker = self.name_to_agent[next_speaker]
            try:
                batch_response, model_output = speaker()
            except Exception as e:
                self.logger.error("Received an error while trying to generate a speech %s", str(e), exc_info=True)
                return None, None

            for idx, response in enumerate(batch_response):
                validated_response = str(response)
                if speaker.quotes_require_validation:
                    validated_response = QuoteUtils.validate_and_replace_quotes(
                        speech_content=str(response),
                        background_text=self.metadata[min(idx, len(self.metadata) - 1)].background_text,
                    )
                for _, agent in self.name_to_agent.items():
                    response_to_use = validated_response if agent.receive_validated_quotes else response
                    agent.receive_message(speaker=speaker.name, content=response_to_use, idx=idx)

            next_speaker = self.judge.get_next_expected_speaker()
            last_output = batch_response
            last_model_output = model_output
        return last_output, last_model_output

    def record_winners(
        self,
        last_output: Optional[list[str]],
        last_model_output: Optional[list[ModelResponse]],
        save_file_path_prefix: Optional[str] = None,
    ) -> list[DebateRoundSummary]:
        """Generates a full summary of the debate round including the winner, transcript, metadata, and aliases of all the participating models"""
        if not last_output:
            return []

        first_debater_win_list = []
        winning_probability_list = []
        for i, (debater_a_wins, model_output) in enumerate(zip(last_output, last_model_output)):
            winner = constants.DEFAULT_DEBATER_A_NAME if debater_a_wins else constants.DEFAULT_DEBATER_B_NAME
            first_debater_win_list.append(winner == self.first_debater.name)
            string_value = self.judge.get_transcript(idx=i).full_string_value()
            winning_probability_list.append(
                1.0 if not model_output.probabilistic_decision else model_output.probabilistic_decision[winner]
            )
            self.logger.debug(string_value)

        if save_file_path_prefix:
            self.name_to_agent[self.judge.expected_saver].save(
                save_file_path_prefix=save_file_path_prefix, metadata=[item.dict() for item in self.metadata]
            )

        return [
            DebateRoundSummary(
                metadata=self.metadata[i % len(self.metadata)],
                transcript=self.judge.get_transcript(idx=i),
                winning_alias=self.first_debater.get_alias() if first_debater_wins else self.second_debater.get_alias(),
                losing_alias=self.first_debater.get_alias() if not first_debater_wins else self.second_debater.get_alias(),
                first_debater_alias=self.first_debater.get_alias(),
                second_debater_alias=self.second_debater.get_alias(),
                first_debater_wins=first_debater_wins,
                judge_alias=self.judge.get_alias(),
                winning_debater_prob=winning_probability_list[i],
                first_debater_win_prob=winning_probability_list[i]
                if first_debater_wins
                else (1 - winning_probability_list[i]),
                second_debater_win_prob=(1 - winning_probability_list[i])
                if first_debater_wins
                else winning_probability_list[i],
            )
            for i, first_debater_wins in enumerate(first_debater_win_list)
        ]

    def __call__(self, save_file_path_prefix: Optional[str] = None) -> list[DebateRoundSummary]:
        """Runs the round and generates a summary of the results"""
        last_output, last_model_output = self.run_round()
        return self.record_winners(
            last_output=last_output, last_model_output=last_model_output, save_file_path_prefix=save_file_path_prefix
        )


class SplittableDebateRound:
    """
    This class is used to generate debate rounds that need to be replayed up to a specific point in the round.
    This is useful if one wants to see who wins if a different speech was given somewhere far into the round"""

    @classmethod
    def run_split_round(
        cls, debate_round: DebateRound, splitting_rule: SplittingRule, save_file_path_prefix: Optional[str] = None
    ):
        """Splits a round at the specified point and reruns it"""
        first_round_summary = debate_round(save_file_path_prefix=save_file_path_prefix)

        truncation_index = SplittableDebateRound.__get_truncation_index(
            splitting_rule=splitting_rule, debate_round=debate_round
        )

        second_round = DebateRound(
            first_debater=debate_round.first_debater.copy(
                transcripts=SplittableDebateRound.__truncate_transcript(
                    agent=debate_round.first_debater, idx=truncation_index
                )
            ),
            second_debater=debate_round.second_debater.copy(
                transcripts=SplittableDebateRound.__truncate_transcript(
                    agent=debate_round.second_debater, idx=truncation_index
                )
            ),
            judge=debate_round.judge.copy(
                transcripts=SplittableDebateRound.__truncate_transcript(agent=debate_round.judge, idx=truncation_index)
            ),
            metadata=copy.deepcopy(debate_round.metadata),
        )

        second_round_summary = second_round(save_file_path_prefix=save_file_path_prefix)

        return first_round_summary, second_round_summary

    @classmethod
    def __get_truncation_index(self, splitting_rule: SplittingRule, debate_round: DebateRound):
        if splitting_rule == SplittingRule.OPENING_ONLY:
            return 0
        elif splitting_rule == SplittingRule.ALL_RANDOM:
            total_speeches = debate_round.first_debater.transcripts[0].get_speech_count(debaters_only=True)
            if len(total_speeches) <= 2:
                return 0
            random_index = random.randrange(0, total_speeches - 1)
            if random_index <= 1:
                return 0
            return random_index
        raise Exception(f"Splitting rule {splitting_rule} is not recognized")

    @classmethod
    def __truncate_transcript(cls, agent: Agent, idx: int) -> list[Transcript]:
        return [transcript.copy().truncate(idx=idx, debaters_only=True) for transcript in agent.transcripts]
