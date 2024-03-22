from __future__ import annotations

from models.model import Model, ModelInput, ModelResponse
from data import SpeakerType, SpeechData
from utils import LoggerUtils
import utils.constants as constants

from typing import Union, Optional


class HumanModel(Model):
    def __init__(self, alias: str, is_debater: bool, debater_name: str, speeches: list[SpeechData], **kwargs):
        """
        A human model returns the text that the human debaters generated during the human debate experiments.

        Args:
            alias: String that identifies the model for metrics and deduplication
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
            speeches: List of speeches delivered by the human debaters. These speeches **must be in the same order
                as the subsequent debate rounds**
        """
        super().__init__(alias=alias, is_debater=is_debater)
        position = 0 if debater_name == constants.DEFAULT_DEBATER_A_NAME else 1
        self.speeches = [
            speech for speech in filter(lambda x: x.speaker_type == SpeakerType.DEBATER and x.position == position, speeches)
        ]
        self.speech_idx = 0
        self.debater_name = debater_name
        self.logger = LoggerUtils.get_default_logger(__name__)

    def predict(self, inputs: list[list[ModelInput]], **kwargs) -> ModelResponse:
        """
        Generates a list of texts in response to the given input. This does not support batch processing.

        Args:
            Inputs: **This input is ignored**. This model returns a deterministic response so the content of the input
                does not matter. It is maintained only to be consistent with the interface.

        Returns:
            A list of length 1 containing the text of the corresponding speech from the human debates.

        Raises:
            Exception: Raises an exception if the batch size is greater than 1.
        """
        if len(inputs) > 1:
            raise Exception(f"The HumanModel does not support batch processing. Input was of length {len(inputs)}")

        speech = ""
        if self.speech_idx < len(self.speeches):
            speech = self.speeches[self.speech_idx].text
            self.speech_idx += 1
        else:
            logger.warn(
                f"Human debater {self.alias} was unable to generate a speech. Current index is {self.speech_idx} but there are only {len(self.speeches)} in its speech list."
            )
        return [ModelResponse(speech=speech, prompt="\n".join(model_input.content for model_input in inputs[0]))]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> HumanModel:
        """Generates a deepcopy of this model"""
        return HumanModel(alias=alias, is_debater=is_debater, debater_name=self.debater_name, speeches=self.speeches)
