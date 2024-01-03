from __future__ import annotations

from agents.models.model import Model, ModelInput, ModelResponse, SpeechStructure
import utils.constants as constants

from typing import Union, Optional
import random


class DeterministicModel(Model):
    def __init__(self, alias: str, is_debater: bool = False):
        """
        A deterministic model responds with the same deterministic string in response to every input. Useful for testing.

        Args:
            alias: string that identifies the model for metrics and deduplication
            is_debater: boolean indicating whether the model is a debater (true) or judge (false)
        """
        super().__init__(alias=alias, is_debater=is_debater)
        self.text = "My position is correct. You have to vote for me." if self.is_debater else "I am a judge. Let me think."
        self.text_length = len(self.text.split())

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = 250,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> ModelResponse:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates. Since the model will return the same
                deterministic response no matter what, the content of the input does not matter.
            max_new_tokens: The total number of new tokens to generate. The canned line will be repeated
                until it reaches that limit.
            speech_structure: The format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text), "preference" (which means a number is expected), and "decision"
                (which means a boolean is expected)
            num_return_sequences: The number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of ModelResponses, with one response for each entry in the batch (or for as many sequences are specified
            to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """
        if len(inputs) > 1 and num_return_sequences > 1:
            raise Exception(
                f"Length of input ({len(inputs)}) and num_return_sequences ({num_return_sequences}) cannot both be greater than 1."
            )

        if speech_structure == SpeechStructure.DECISION:
            return [ModelResponse(decision=constants.DEFAULT_DEBATER_A_NAME) for i in range(len(inputs))]
        elif speech_structure == SpeechStructure.PREFERENCE:
            return [ModelResponse(preference=5.0) for i in range(len(inputs))]
        text_to_repeat = "\n".join([self.text for i in range(int(max_new_tokens / self.text_length))])

        num_return_sequences = len(inputs) if len(inputs) > 1 else num_return_sequences
        return [ModelResponse(speech=text_to_repeat) for i in range(num_return_sequences)]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> DeterministicModel:
        """Generates a deepcopy of this model"""
        return DeterministicModel(alias=alias, is_debater=is_debater if is_debater is not None else False)
