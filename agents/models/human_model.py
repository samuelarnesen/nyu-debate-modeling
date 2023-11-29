from __future__ import annotations

from agents.model import Model, ModelInput
from data.data import SpeakerType, SpeechData
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from typing import Union, Optional


class HumanModel(Model):
    def __init__(self, alias: str, is_debater: bool, debater_name: str, speeches: list[SpeechData], **kwargs):
        super().__init__(alias=alias, is_debater=is_debater)
        position = 0 if debater_name == constants.DEFAULT_DEBATER_A_NAME else 1
        self.speeches = [
            speech for speech in filter(lambda x: x.speaker_type == SpeakerType.DEBATER and x.position == position, speeches)
        ]
        self.speech_idx = 0
        self.debater_name = debater_name
        self.logger = LoggerUtils.get_default_logger(__name__)

    def predict(self, inputs: list[list[ModelInput]], **kwargs) -> str:
        speech = ""
        if self.speech_idx < len(self.speeches):
            speech = self.speeches[self.speech_idx].text
            self.speech_idx += 1
        else:
            logger.warn(
                f"Human debater {self.alias} was unable to generate a speech. Current index is {self.speech_idx} but there are only {len(self.speeches)} in its speech list."
            )
        return [speech]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> HumanModel:
        return HumanModel(alias=alias, is_debater=is_debater, debater_name=self.debater_name, speeches=self.speeches)
