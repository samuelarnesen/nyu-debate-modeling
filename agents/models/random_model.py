from __future__ import annotations

from agents.models.model import Model, ModelInput, ModelResponse, SpeechStructure
import utils.constants as constants

from typing import Union, Optional
import random
import re


class RandomModel(Model):
    def __init__(self, alias: str, is_debater: bool = False, **kwargs):
        """
        A random model responds with a random string in response to every input. Useful for testing.

        Args:
            alias: string that identifies the model for metrics and deduplication
            is_debater: boolean indicating whether the model is a debater (true) or judge (false)
        """
        super().__init__(alias=alias, is_debater=is_debater)
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=250,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[str]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates. Since the model will return a random
                response no matter what, the content of the input does not matter.
            max_new_tokens: The total number of new tokens to generate.
            speech_structure: The format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text) and "decision" (which means a boolean is expected)
            num_return_sequences: The number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of text, with one string for each entry in the batch (or for as many sequences are specified
            to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """

        def generate_random_text():
            return " ".join(
                [
                    "".join(random.choices(self.alphabet, k=random.randrange(1, 8)))
                    for i in range(random.randrange(1, max_new_tokens))
                ]
            )

        def generate_random_decision():
            a_odds = random.random()
            b_odds = 1 - a_odds
            decision = constants.DEFAULT_DEBATER_A_NAME if a_odds > 0.5 else constants.DEFAULT_DEBATER_B_NAME
            return decision, (a_odds, b_odds)

        if len(inputs) > 1 and num_return_sequences > 1:
            raise Exception(
                f"Length of input ({len(inputs)}) and num_return_sequences ({num_return_sequences}) cannot both be greater than 1."
            )

        if speech_structure == SpeechStructure.DECISION:
            decisions = []
            for i in range(len(inputs)):
                decision, (a_odds, b_odds) = generate_random_decision()
                decisions.append(
                    ModelResponse(
                        decision=decision,
                        probabilistic_decision={
                            constants.DEFAULT_DEBATER_A_NAME: a_odds,
                            constants.DEFAULT_DEBATER_B_NAME: b_odds,
                        },
                        prompt="\n".join([model_input.content for model_input in inputs[i]]),
                    )
                )
            return decisions

        num_return_sequences = max(num_return_sequences, len(inputs))
        return [
            ModelResponse(
                speech=generate_random_text(), prompt="\n".join([model_input.content for model_input in inputs[i]])
            )
            for i in range(num_return_sequences)
        ]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> RandomModel:
        """Generates a deepcopy of this model"""
        return RandomModel(alias=alias, is_debater=is_debater if is_debater is not None else False)
