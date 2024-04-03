from __future__ import annotations

from models.model import Model, ModelInput, ModelResponse, RoleType, SpeechStructure
from utils import logger_utils
import utils.constants as constants

import anthropic
import backoff

from concurrent.futures import ThreadPoolExecutor
from typing import Union, Optional
import logging
import os
import math
import random
import re


class AnthropicModel(Model):
    MAX_PARALLEL_REQUESTS = 16
    DEFAULT_MODEL_ENDPOINT = "claude-3-opus-20240229"

    def __init__(self, alias: str, is_debater: bool = True, endpoint: Optional[str] = None, **kwargs):
        """
        An AnthropicModel calls Claude to generate the appropriate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
        """
        super().__init__(alias=alias, is_debater=is_debater)
        self.client = anthropic.Anthropic()
        self.endpoint = endpoint if endpoint else AnthropicModel.DEFAULT_MODEL_ENDPOINT
        self.logger = logger_utils.get_default_logger(__name__)

    def predict(
        self,
        inputs: list[list[ModelInput] | str],
        max_new_tokens=200,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> list[ModelResponse]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates.
            max_new_tokens: The maximum total number of new tokens to generate.
            speech_structure: the format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text), and "decision" (which means a boolean is expected)

        Returns:
            A list of model responses, with one string for each entry in the batch.
        """
        with ThreadPoolExecutor(max_workers=AnthropicModel.MAX_PARALLEL_REQUESTS) as executor:
            futures = [
                executor.submit(
                    self.predict_single_input,
                    model_input_list=input_value,
                    max_new_tokens=max_new_tokens,
                    speech_structure=speech_structure,
                )
                for input_value in inputs
            ]
            results = [future.result() for future in futures]

        return results

    def predict_single_input(
        self,
        model_input_list: list[ModelInput] | str,
        max_new_tokens=200,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> ModelResponse:
        """
        Generates a list of texts in response to a single given input.

        Args:
            model_input_list: A list of model inputs. Each ModelInput corresponds roughly to one command
            max_new_tokens: The maximum total number of new tokens to generate.
            speech_structure: the format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text) and "decision" (which means a boolean is expected)

        Returns:
            A list of model responses, with one string for each entry in the batch.
        """

        def extract_response_from_structured_speech(message: str, regex_str: str, default: str) -> str:
            match = re.match(regex_str, message)
            if match:
                return match.group(1)
            else:
                self.logger.warn("The regex {} did not match the following message: {}".format(regex_str, message))
                return default

        def process_logprobs(completion: dict) -> tuple[float, float]:
            """This exists to maintain parity with the OpenAI model functionality even though the Anthropic API
            does not support logprobs yet"""

            if re.search(constants.DEFAULT_DEBATER_A_NAME, completion.content[0].text):
                return 1.0, 0.0
            elif re.search(constants.DEFAULT_DEBATER_B_NAME, completion.content[0].text):
                return 0.0, 1.0
            print("uh oh!", completion.content[0].text)
            return 0.5, 0.5

        system, messages = AnthropicModel.generate_llm_input_from_model_inputs(input_list=model_input_list)

        try:
            completion = self.call_anthropic(
                system=system, messages=messages, max_new_tokens=max_new_tokens, speech_structure=speech_structure
            )
        except Exception as e:
            self.logger.warn(f"Anthropic API returned an API Error: {e}")
            self.logger.warn(e)
            return ModelResponse(failed=True)

        message = completion.content[0].text

        if speech_structure == SpeechStructure.DECISION:
            a_odds, b_odds = process_logprobs(completion)
            message = (
                constants.DEFAULT_DEBATER_A_NAME
                if a_odds > b_odds
                else (
                    constants.DEFAULT_DEBATER_B_NAME
                    if (b_odds > a_odds or random.random() > 0.5)
                    else constants.DEFAULT_DEBATER_A_NAME
                )
            )
            self.logger.debug(f"Debater A's odds: {a_odds}, Debater B's odds: {b_odds}, Winner: {message}")
            return ModelResponse(
                decision=message,
                probabilistic_decision={
                    constants.DEFAULT_DEBATER_A_NAME: a_odds,
                    constants.DEFAULT_DEBATER_B_NAME: b_odds,
                },
                prompt="\n".join(model_input.content for model_input in model_input_list),
            )

        return ModelResponse(speech=message, prompt="\n".join(model_input.content for model_input in model_input_list))

    # @backoff.on_exception(backoff.expo, backoff.on_exception, max_tries=4)
    def call_anthropic(
        self, system: str, messages: list[dict[str, str]], speech_structure: SpeechStructure, max_new_tokens: int
    ):
        return self.client.messages.create(
            model=self.endpoint,  # "claude-3-haiku-20240307", #"claude-3-opus-20240229",
            max_tokens=max_new_tokens,
            system=system,
            messages=messages,
            temperature=0.0 if speech_structure == SpeechStructure.DECISION else 0.5,
        )

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> AnthropicModel:
        """Generates a deepcopy of this model"""
        return AnthropicModel(alias=alias, is_debater=is_debater, endpoint=self.endpoint)

    @classmethod
    def generate_llm_input_from_model_inputs(
        cls, input_list: list[ModelInput], extra_suffix: str = ""
    ) -> tuple[str, dict[str, list[dict[str, str]]]]:
        """Converts a ModelInput into the format that the Anthropic API expects. The first output
        is the system prompt and the second is the messages list"""

        def model_input_to_anthropic_format(model_input: ModelInput | str) -> dict[str, str]:
            if isinstance(model_input, str):
                return {"role": RoleType.USER.name.lower(), "content": model_input}
            return {"role": model_input.role.name.lower(), "content": model_input.content}

        def add_actual_speech(messages: list[dict[str, str]], actual_speech: str) -> None:
            messages.append({"role": "assistant", "content": actual_speech})

        messages = [model_input_to_anthropic_format(model_input) for model_input in input_list]
        if extra_suffix:
            add_actual_speech(messages=messages, actual_speech=extra_suffix)

        if messages[0]["role"] == RoleType.SYSTEM.name.lower():
            return messages[0]["content"], messages[1:]
        return "", messages
