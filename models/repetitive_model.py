from __future__ import annotations

from models.model import Model, ModelInput, ModelResponse, SpeechStructure
import utils.constants as constants

from typing import Union, Optional
import random
import sys


class RepetitiveModel(Model):
    def __init__(self, alias: str, is_debater: bool = False):
        """
        An repetitive model only works for judging and always responds with whatever letter appeared most frequently in the previous speech.
        Useful for evaluating whether an open debater chose the correct side and for debugging

        Args:
            alias: string that identifies the model for metrics and deduplication
            is_debater: boolean indicating whether the model is a debater (true) or judge (false)
        """
        super().__init__(alias=alias, is_debater=is_debater)

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
                (which is just free text), and "decision" (which means a boolean is expected)
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

        outputs = []
        if speech_structure == SpeechStructure.DECISION:
            for model_input in inputs:
                content = (
                    constants.DEFAULT_DEBATER_A_NAME
                    if model_input[-1].content.lower().rfind("a") > model_input[-1].content.lower().rfind("b")
                    else constants.DEFAULT_DEBATER_B_NAME
                )
                outputs.append(ModelResponse(decision=content))
        else:
            for model_input in inputs:
                outputs.append(ModelResponse(speech="A" if random.random() < 0.5 else "B"))
        return outputs

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> DeterministicModel:
        """Generates a deepcopy of this model"""
        return RepetitiveModel(alias=alias, is_debater=is_debater if is_debater is not None else False)
