from agents.agent import Agent
from agents.debater import Debater
from agents.judge import Judge, JudgeType
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


class DebateRoundSummary(BaseModel):
    metadata: QuestionMetadata
    transcript: Any
    winning_alias: str
    losing_alias: str
    first_debater_alias: str
    second_debater_alias: str
    first_debater_wins: bool
    judge_alias: str


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
        self.first_debater = debater
        self.name_to_agent[self.first_debater.name] = debater

    def set_second_debater(self, debater: Debater):
        self.second_debater = debater
        self.name_to_agent[self.second_debater.name] = debater

    def set_judge(self, judge: Judge):
        self.judge = judge
        self.name_to_agent[self.judge.name] = judge

    def run_round(self) -> list[str]:
        last_output = None
        next_speaker = self.judge.get_next_expected_speaker()
        while next_speaker:
            speaker = self.name_to_agent[next_speaker]
            batch_response = speaker()
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
        return last_output

    def record_winners(
        self, last_output: list[str], save_file_path_prefix: Optional[str] = None
    ) -> list[DebateRoundSummary]:
        first_debater_win_list = []
        for i, debater_a_wins in enumerate(last_output):
            winner = constants.DEFAULT_DEBATER_A_NAME if debater_a_wins else constants.DEFAULT_DEBATER_B_NAME
            first_debater_win_list.append(winner == self.first_debater.name)
            string_value = self.judge.get_transcript(idx=i).full_string_value()
            self.logger.debug(string_value)

        if save_file_path_prefix:
            self.name_to_agent[self.judge.expected_saver].save(save_file_path_prefix=save_file_path_prefix)

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
            )
            for i, first_debater_wins in enumerate(first_debater_win_list)
        ]

    def __call__(self, save_file_path_prefix: Optional[str] = None) -> list[DebateRoundSummary]:
        last_output = self.run_round()
        return self.record_winners(last_output=last_output, save_file_path_prefix=save_file_path_prefix)


class SplittableDebateRound:
    @classmethod
    def run_split_round(
        cls, debate_round: DebateRound, splitting_rule: SplittingRule, save_file_path_prefix: Optional[str] = None
    ):
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
