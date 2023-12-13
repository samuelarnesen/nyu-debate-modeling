from __future__ import annotations

from agents.models.model import Model, ModelInput, RoleType, SpeechStructure
from utils import LoggerUtils
import utils.constants as constants

import openai

from typing import Union
import logging
import os
import random
import re


class OpenAIModel(Model):
    decision_addendum = (
        "\n\nNow please give your answer exclusively in this format:\n"
        + "Winner:[DEBATER_NAME] (Example 1 - Winner: Debater_A. Example 2: Winner: Debater_B)\n"
        + "Do not contain any other text at all."
    )

    preference_addendum = (
        "\nNow please give your answer exclusively in this format:\n"
        + "Overall Score: [0-10] (Example 1 - Overall Score: 3. Example 2: Overall Score: 7)\n"
        + "Do not contain any other text at all."
    )

    decision_regex = ".*Winner: (Debater_[AB])"

    preference_regex = ".*Overall Score: (\\d+(\\.\\d+)?)"

    def __init__(self, alias: str, is_debater: bool = True, **kwargs):
        """
        An OpenAIModel calls GPT4 to generate the appropriate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
        """
        super().__init__(alias=alias, is_debater=is_debater)
        self.__configure()
        self.logger = LoggerUtils.get_default_logger(__name__)

    def __configure(self):
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def predict(
        self,
        inputs: list[list[ModelInput] | str],
        max_new_tokens=450,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> list[str]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates.
            max_new_tokens: The maximum total number of new tokens to generate.
            speech_structure: the format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text), "preference" (which means a number is expected), and "decision"
                (which means a boolean is expected)

        Returns:
            A list of text, with one string for each entry in the batch.
        """

        def model_input_to_openai_format(model_input: ModelInput | str) -> dict[str, str]:
            if isinstance(model_input, str):
                return {"role": RoleType.USER.name.lower(), "content": model_input}
            print(model_input)
            return {"role": model_input.role.name.lower(), "content": model_input.content}

        def add_addendum(messages: list[dict[str, str]], addendum: str) -> None:
            if messages[-1]["role"] == "user":
                messages[-1]["content"] = "\n".join([messages[-1]["content"], addendum])
            else:
                messages.append({"role": "user", "content": addendum})

        def extract_response_from_structured_speech(message: str, regex_str: str, default: str) -> str:
            match = re.match(regex_str, message)
            if match:
                return match.group(1)
            else:
                self.logger.warn("The regex {} did not match the following message: {}".format(regex_str, message))
                return default

        responses = []
        for model_input_list in inputs:
            messages = [model_input_to_openai_format(model_input) for model_input in model_input_list]

            if speech_structure == SpeechStructure.DECISION:
                add_addendum(messages=messages, addendum=OpenAIModel.decision_addendum)
            elif speech_structure == SpeechStructure.PREFERENCE:
                add_addendum(messages=messages, addendum=OpenAIModel.preference_addendum)

            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=messages,
                    max_tokens=max_new_tokens,
                )
            except Exception as e:
                self.logger.warn(f"Received an error while calling OpenAI: {e}")
                completion = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=messages,
                    max_tokens=max_new_tokens,
                )

            message = completion.choices[0].message["content"]

            self.logger.info(f"Received message")

            if speech_structure == SpeechStructure.DECISION:
                message = extract_response_from_structured_speech(
                    message=message,
                    regex_str=OpenAIModel.decision_regex,
                    default=constants.DEFAULT_DEBATER_A_NAME if random.random() < 0.5 else constants.DEFAULT_DEBATER_B_NAME,
                )
            elif speech_structure == SpeechStructure.PREFERENCE:
                message = extract_response_from_structured_speech(
                    message=message,
                    regex_str=OpenAIModel.preference_regex,
                    default=str(-1),
                )

            responses.append(message)

        return responses

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> HumanModel:
        """Generates a deepcopy of this model"""
        return HumanModel(alias=alias, is_debater=is_debater)
