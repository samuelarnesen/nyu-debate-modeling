from __future__ import annotations

from models.model import Model, ModelInput, ModelResponse, RoleType, SpeechStructure
from utils import LoggerUtils
import utils.constants as constants

import backoff
import openai

from concurrent.futures import ThreadPoolExecutor
from typing import Union, Optional
import logging
import os
import math
import random
import re


class OpenAIModel(Model):
    decision_regex = ".*Winner: (Debater_[AB])"

    MAX_PARALLEL_REQUESTS = 16
    INSTRUCTION_SUFFIX = ""
    DEFAULT_OPENAI_MODEL_ENDPOINT = "gpt-4-0125-preview"

    def __init__(self, alias: str, is_debater: bool = True, endpoint: Optional[str] = None, **kwargs):
        """
        An OpenAIModel calls GPT4 to generate the appropriate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
        """
        super().__init__(alias=alias, is_debater=is_debater)
        self.__configure()
        self.client = openai.OpenAI()
        self.endpoint = endpoint if endpoint else OpenAIModel.DEFAULT_OPENAI_MODEL_ENDPOINT
        self.logger = LoggerUtils.get_default_logger(__name__)

    def __configure(self):
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.getenv("OPENAI_API_KEY")

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
        with ThreadPoolExecutor(max_workers=OpenAIModel.MAX_PARALLEL_REQUESTS) as executor:
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
            debater_suffixes = ["_A", "_B"]
            logprobs = completion.choices[0].logprobs.content
            for entry in logprobs:
                if entry.token in debater_suffixes:
                    scores = {suffix: 0 for suffix in debater_suffixes}
                    for option in filter(lambda x: x.token in debater_suffixes, entry.top_logprobs):
                        scores[option.token] = math.exp(float(option.logprob))
                    total_probs = sum(scores.values())
                    renormalized_scores = {suffix: scores[suffix] / total_probs for suffix in scores}
                    return (
                        renormalized_scores[debater_suffixes[0]],
                        renormalized_scores[debater_suffixes[1]],
                    )
            return 0.5, 0.5

        messages = OpenAIModel.generate_llm_input_from_model_inputs(input_list=model_input_list)

        try:
            completion = self.call_openai(
                messages=messages, max_new_tokens=max_new_tokens, speech_structure=speech_structure
            )
        except openai.APIError as e:
            self.logger.warn(f"OpenAI API returned an API Error: {e}")
            self.logger.warn(e)
            return ModelResponse(failed=True)
        except openai.APIConnectionError as e:
            self.logger.warn(f"Failed to connect to OpenAI API: {e}")
            self.logger.warn(e)
            return ModelResponse(failed=True)
        except openai.RateLimitError as e:
            self.logger.warn(f"OpenAI API request exceeded rate limit: {e}")
            self.logger.warn(e)
            return ModelResponse(failed=True)

        message = completion.choices[0].message.content

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

    @backoff.on_exception(backoff.expo, backoff.on_exception, max_tries=4)
    def call_openai(
        self, messages: list[dict[str, str]], speech_structure: SpeechStructure, max_new_tokens: int
    ) -> openai.ChatCompletion:
        return self.client.chat.completions.create(
            model=self.endpoint,
            messages=messages,
            max_tokens=max_new_tokens,
            logprobs=(speech_structure != SpeechStructure.OPEN_ENDED),
            top_logprobs=5 if (speech_structure != SpeechStructure.OPEN_ENDED) else None,
        )

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> OpenAIModel:
        """Generates a deepcopy of this model"""
        return OpenAIModel(alias=alias, is_debater=is_debater)

    @classmethod
    def generate_llm_input_from_model_inputs(
        cls, input_list: list[ModelInput], extra_suffix: str = ""
    ) -> dict[str, list[dict[str, str]]]:
        """Converts a ModelInput into the jsonl format for GPT finetuning that's expected by the model"""

        def model_input_to_openai_format(model_input: ModelInput | str) -> dict[str, str]:
            if isinstance(model_input, str):
                return {"role": RoleType.USER.name.lower(), "content": model_input}
            return {"role": model_input.role.name.lower(), "content": model_input.content}

        def add_actual_speech(messages: list[dict[str, str]], actual_speech: str) -> None:
            messages.append({"role": "assistant", "content": actual_speech})

        messages = [model_input_to_openai_format(model_input) for model_input in input_list]
        if extra_suffix:
            add_actual_speech(messages=messages, actual_speech=extra_suffix)
        return messages
