from agents.debater import Debater
from agents.judge import Judge, JudgeType
from agents.prompt import Prompt, PromptConfig, PromptParser
from agents.transcript import Transcript
from data.data import RawDataset, SplitType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel

from enum import Enum
from typing import Optional, Any, Union


class QuestionMetadata(BaseModel):
    first_debater_correct: bool
    question_idx: int
    split: SplitType = SplitType.TRAIN


class DebateRoundSummary(BaseModel):
    metadata: QuestionMetadata
    transcript: Union[Any]
    winning_alias: str
    losing_alias: str
    first_debater_alias: str
    second_debater_alias: str
    first_debater_wins: bool
    judge_alias: str


class DebateRound:
    def __init__(
        self,
        first_debater: Debater,
        second_debater: Debater,
        judge: Judge,
        metadata: Union[QuestionMetadata, list[QuestionMetadata]],
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
                for _, agent in self.name_to_agent.items():
                    agent.receive_message(speaker=speaker.name, content=response, idx=idx)
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
            self.logger.debug(self.judge.get_transcript(idx=i).full_string_value())

        if save_file_path_prefix:
            if self.judge.judge_type == JudgeType.BEST_OF_N:
                self.first_debater.save(save_file_path_prefix=save_file_path_prefix)
            else:
                self.judge.save(save_file_path_prefix=save_file_path_prefix)

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

    def __call__(self, save_file_path_prefix: Optional[str] = None) -> list[DebateRoundSummary]:  # TODO: remote num_speeches
        last_output = self.run_round()
        return self.record_winners(last_output=last_output, save_file_path_prefix=save_file_path_prefix)
