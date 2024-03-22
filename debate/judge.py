from __future__ import annotations

from debate.agent import Agent, ScratchpadConfig
from debate.speech_format import SpeechFormat, SpeechFormatType
from debate.transcript import SpeechFormat, SpeechType, Transcript
from models import Model, ModelResponse, SpeechStructure
from prompts import Prompt, PromptTag
from utils import LoggerUtils
import utils.constants as constants

from enum import Enum
from typing import Any, Optional, Union
import copy
import json
import os
import random
import re


class Judge(Agent):
    def __init__(
        self,
        name: str,
        prompt: Prompt | list[Prompt],
        model: Model,
        num_speeches: int,
        speech_format: Optional[SpeechFormat] = None,
        speech_structure: SpeechStructure = SpeechStructure.DECISION,
        expected_saver: str = constants.DEFAULT_JUDGE_NAME,
        scratchpad_config: ScratchpadConfig = ScratchpadConfig(),
    ):
        """
        The abstraction used to both judge rounds and determine who speaks next.

        Params:
            name: The name of the judge (just needs to be unique within the round)
            prompt: The prompt structure used to generate inputs to the model
            model: The model that actually generates text
            num_speeches: The number of speeches each debater is expected to deliver
            speech_format: the order of speeches the judge expects to hear
            speech_structure: the default way the judge is to supposed to generate text
            expected_saver: whether the judge or the debater is in charge of saving the transcript
            scratchpad_config: configuration that specifies if and how to use a scratchpad
        """
        super().__init__(
            name=name,
            is_debater=False,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            quotes_require_validation=False,
            receive_validated_quotes=True,
            speech_format=speech_format
            if speech_format
            else SpeechFormatType.DEFAULT_DEBATE_JUDGE.get_speech_format(
                name=name, num_speeches=num_speeches, use_scratchpad=scratchpad_config.use_scratchpad
            ),
        )
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.speech_structure = speech_structure
        self.expected_saver = expected_saver
        self.scratchpad_config = scratchpad_config
        self.num_speeches = num_speeches

    def generate(
        self, max_new_tokens: Optional[int] = None, speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED
    ) -> [list[ModelResponse]]:
        """Calls the underlying model to generate text"""
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        max_new_tokens = max_new_tokens if speech_structure == SpeechStructure.OPEN_ENDED else 15
        return self.model.predict(
            inputs=model_inputs,
            max_new_tokens=max_new_tokens or self.speech_format.tokens_per_speech,
            speech_structure=speech_structure,
        )

    def __call__(self) -> list[str | bool]:
        """
        Calls the underlying model to generate text for each element in the batch.

        Returns:
            Either a string with the text it generated or a boolean indicating whether the first debater won
            (depending on whether the speech was a decision or open-ended) for each element in the batch.
        """
        if self.scratchpad_config.use_scratchpad or not self.transcripts[0].only_decision_remains():
            batch_generation = self.generate(max_new_tokens=250, speech_structure=SpeechStructure.OPEN_ENDED)
            batch_reasoning = [response.speech for response in batch_generation]
        if self.transcripts[0].only_decision_remains():  # all formats should be the same so we can use any transcript
            if self.scratchpad_config.use_scratchpad:
                for i, reasoning in enumerate(batch_reasoning):
                    super().receive_message(speaker=self.name, content=reasoning, idx=i)
            batch_predictions = self.generate(max_new_tokens=15, speech_structure=self.speech_structure)
            validated_predictions = self.validate_responses(batch_predictions)
            returned_response = self.process_responses(validated_predictions)
            if self.scratchpad_config.use_scratchpad:
                for generation, prediction in zip(batch_generation, batch_predictions):
                    prediction.failed = generation.failed or prediction.failed

            return returned_response, batch_predictions
        return batch_reasoning, batch_generation

    def validate_responses(self, responses: list[ModelResponse]) -> None:
        """Confirms that the responses matched the expected format"""
        for response in filter(lambda x: not x.decision, responses):
            self.logger.warn('Response of "{}" was invalid. Must be a debater name.'.format(response))
        return responses

    def process_responses(self, responses: list[ModelResponse]) -> list[Any]:
        """Converts a text response to a list of booleans indicating if Debater_A won"""
        return [constants.DEFAULT_DEBATER_A_NAME in response.decision for response in responses]

    def copy(self, transcripts: Optional[list[Transcript]] = None, prompts: Optional[list[Prompt] | Prompt] = None) -> Judge:
        """Deep copies everything except the underlying model"""
        judge = Judge(
            name=self.name,
            prompt=prompts if prompts else [copy.deepcopy(prompt) for prompt in self.prompts],
            model=self.model,
            num_speeches=self.num_speeches,
            speech_format=self.speech_format,
            speech_structure=self.speech_structure,
            expected_saver=self.expected_saver,
            scratchpad_config=self.scratchpad_config,
        )
        if transcripts:
            judge.transcripts = [transcript.copy() for transcript in transcripts]
        return judge
