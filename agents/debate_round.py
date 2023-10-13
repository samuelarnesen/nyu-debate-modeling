from agents.debater import Debater
from agents.judge import Judge
from agents.prompt import Prompt, PromptConfig, PromptParser
from agents.transcript import Transcript
from data.data import RawDataset, SplitType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel

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
        self.debaters = [self.first_debater, self.second_debater]
        self.participants = [self.first_debater, self.second_debater, self.judge]
        self.logger = LoggerUtils.get_default_logger(__name__)

    def run(self, num_speeches=3, save_file_path_prefix: str = None) -> list[DebateRoundSummary]:
        for speech_num in range(num_speeches):
            responses = {}
            for debater in self.debaters:
                batch_response = debater.generate()
                for idx, response in enumerate(batch_response):
                    if speech_num == 0:
                        responses.setdefault(debater.name, []).append((response, idx))
                    else:
                        for participant in self.participants:
                            participant.receive_message(speaker=debater.name, content=response, idx=idx)
            if speech_num == 0:
                for participant in self.participants:
                    if participant.name in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_JUDGE_NAME]:
                        for response, idx in responses[constants.DEFAULT_DEBATER_A_NAME]:
                            participant.receive_message(speaker=constants.DEFAULT_DEBATER_A_NAME, content=response, idx=idx)
                        for response, idx in responses[constants.DEFAULT_DEBATER_B_NAME]:
                            participant.receive_message(speaker=constants.DEFAULT_DEBATER_B_NAME, content=response, idx=idx)
                    else:
                        for response, idx in responses[constants.DEFAULT_DEBATER_B_NAME]:
                            participant.receive_message(speaker=constants.DEFAULT_DEBATER_B_NAME, content=response, idx=idx)
                        for response, idx in responses[constants.DEFAULT_DEBATER_A_NAME]:
                            participant.receive_message(speaker=constants.DEFAULT_DEBATER_A_NAME, content=response, idx=idx)

            if speech_num < (num_speeches - 1):
                batch_response = self.judge.generate()
                for participant in self.participants:
                    for idx, response in enumerate(batch_response):
                        participant.receive_message(speaker=self.judge.name, content=response, idx=idx)

        responses = self.judge.judge()
        first_debater_win_list = []
        for i, debater_a_wins in enumerate(responses):
            winner = constants.DEFAULT_DEBATER_A_NAME if debater_a_wins else constants.DEFAULT_DEBATER_B_NAME
            response_to_use = f"{winner} wins the debate"
            first_debater_win_list.append(winner == self.first_debater.name)
            self.judge.receive_message(speaker=self.judge.name, content=response_to_use, idx=i)
            self.logger.debug(self.judge.get_transcript(idx=i).full_string_value())

        if save_file_path_prefix:
            self.judge.save(save_file_path_prefix=save_file_path_prefix)

        return [
            DebateRoundSummary(
                metadata=self.metadata[i],
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
