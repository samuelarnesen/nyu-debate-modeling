from __future__ import annotations

from agents.model import Model, ModelInput, SpeechStructure
import utils.constants as constants

from typing import Union, Optional
import random


class DeterministicModel(Model):
    def __init__(self, alias: str, is_debater: bool = False):
        super().__init__(alias=alias, is_debater=is_debater)
        self.text = "My position is correct. You have to vote for me."
        self.text_length = len(self.text.split())

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=250,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> str:
        if speech_structure == SpeechStructure.DECISION:
            return [utils.DEFAULT_DEBATER_A_NAME for i in range(len(inputs))]
        elif speech_structure == SpeechStructure.PREFERENCE:
            return [str(5) for i in range(len(inputs))]
        text_to_repeat = "\n".join([self.text for i in range(int(max_new_tokens / self.text_length))])
        return [text_to_repeat for i in range(len(inputs))]

    def copy(self, alias: str, is_debater: Optional[bool] = None) -> DeterministicModel:
        return DeterministicModel(alias=alias, is_debater=is_debater if is_debater is not None else False)
