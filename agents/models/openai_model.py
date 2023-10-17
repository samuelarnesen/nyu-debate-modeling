from __future__ import annotations

from agents.model import Model, ModelInput
from utils.logger_utils import LoggerUtils
import utils.constants as constants

import openai

import os
import random
import re


class OpenAIModel(Model):
    def __init__(self, alias: str, is_debater: bool = True):
        super().__init__(alias=alias, is_debater=is_debater)
        self.__configure()
        self.logger = LoggerUtils.get_default_logger(__name__)

    def __configure(self):
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens=450, decide: bool = False, **kwargs) -> list[str]:
        def model_input_to_openai_format(model_input: ModelInput) -> dict[str, str]:
            return {"role": model_input.role.name.lower(), "content": model_input.content}

        decision_addendum = (
            "\n\nPlease answer exclusively in this format:\n"
            + "Winner:[DEBATER_NAME] (Example 1 - Winner: Debater_A. Example 2: Winner: Debater_B)"
        )

        responses = []
        for model_input_list in inputs:
            messages = [model_input_to_openai_format(model_input) for model_input in model_input_list]
            if not self.is_debater and decide:
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] = "\n".join([messages[-1]["content"], decision_addendum])
                else:
                    messages.append({"role": "user", "content": decision_addendum})

            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=max_new_tokens,
                )
            except Exception as e:
                self.logger.warn(f"Received an error while calling OpenAI: {e}")
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=max_new_tokens,
                )

            message = completion.choices[0].message["content"]

            if decide:
                self.logger.debug("Deciding on a winner. The message was {}".format(message))
                match = re.match(".*Winner: (Debater_[AB])", message)
                if match:
                    responses.append(match.group(1))
                    self.logger.debug(f"The logged winner was {responses[-1]}")
                else:
                    responses.append(
                        constants.DEFAULT_DEBATER_A_NAME if random.random() < 0.5 else constants.DEFAULT_DEBATER_B_NAME
                    )

                    self.logger.warn("A precise winner could not be extracted from the following text: {}".format(message))
            else:
                responses.append(message)

        return responses
