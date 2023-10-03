from agents.agent import Debater, Judge
from agents.prompt import Prompt, PromptConfig, PromptParser
from agents.transcript import Transcript
from data.data import RawDataset, SplitType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel

from typing import Optional, Any, Union


class DebateRoundMetadata(BaseModel):
    debater_one_correct: bool
    question_idx: int
    split: SplitType = SplitType.TRAIN


class DebateRoundSummary(BaseModel):
    debater_one_wins: Union[Any, bool]
    metadata: DebateRoundMetadata
    transcript: Union[Any]


class DebateRound:
    def __init__(self, first_debater: Debater, second_debater: Debater, judge: Judge, metadata: DebateRoundMetadata):
        self.first_debater = first_debater
        self.second_debater = second_debater
        self.judge = judge
        self.metadata = metadata
        self.debaters = [self.first_debater, self.second_debater]
        self.participants = [self.first_debater, self.second_debater, self.judge]
        self.logger = LoggerUtils.get_default_logger(__name__)

    def reset(self):
        for participant in self.participants:
            participant.reset()

    def run(self, num_speeches=3, save_file_path: str = None) -> bool:
        self.first_debater.reset()
        self.second_debater.reset()
        self.judge.reset()

        self.logger.debug(self.judge.get_transcript().full_string_value())

        for speech_num in range(num_speeches):
            responses = []
            for debater in self.debaters:
                response = debater.generate()
                responses.append((debater.name, response))
                if speech_num > 0:
                    for participant in self.participants:
                        participant.receive_message(speaker=debater.name, content=response)
                self.logger.debug(f"{debater.name}: {response}\n")
            if speech_num == 0:
                for participant in self.participants:
                    for speaker, response in responses:
                        participant.receive_message(speaker=speaker, content=response)

        response, debater_one_wins = self.judge.generate()
        self.logger.debug(f"{self.judge.name}: {response}\n")
        self.logger.debug(
            f"Winner is {constants.DEFAULT_DEBATER_ONE_NAME if debater_one_wins else constants.DEFAULT_DEBATER_TWO_NAME}"
        )

        if save_file_path:
            self.judge.save(save_file_path=save_file_path)

        return DebateRoundSummary(
            debater_one_wins=debater_one_wins, transcript=self.judge.get_transcript(), metadata=self.metadata
        )
