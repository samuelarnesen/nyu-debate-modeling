from __future__ import annotations

from agents.agent import Agent
from agents.models import Model, SpeechStructure
from agents.transcript import SpeechFormat, SpeechType, Transcript
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


class JudgeType(Enum):
    STANDARD = 1
    BEST_OF_N = 2


class Judge(Agent):
    def __init__(
        self,
        name: str,
        prompt: Prompt | list[Prompt],
        model: Model,
        num_speeches: int,
        speech_format: Optional[SpeechFormat] = None,
        speech_structure: SpeechStructure = SpeechStructure.DECISION,
        judge_type: JudgeType = JudgeType.STANDARD,
        expected_saver: str = constants.DEFAULT_JUDGE_NAME,
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
            judge_type: whether the judge is a best-of-n judge or not
            expected_saver: whether the judge or the debater is in charge of saving the transcript
        """
        super().__init__(
            name=name,
            is_debater=False,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            quotes_require_validation=False,
            receive_validated_quotes=True,
            speech_format=speech_format if speech_format else JudgeUtils.get_default_speech_format(num_speeches),
        )
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.speech_structure = speech_structure
        self.judge_type = judge_type
        self.expected_saver = expected_saver

    def generate(
        self, max_new_tokens: int = 150, speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED
    ) -> [list[str]]:
        """Calls the underlying model to generate text"""
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        max_new_tokens = max_new_tokens if speech_structure == SpeechStructure.OPEN_ENDED else 15
        return self.model.predict(inputs=model_inputs, max_new_tokens=max_new_tokens, speech_structure=speech_structure)

    def __call__(self) -> list[str | bool]:
        """
        Calls the underlying model to generate text for each element in the batch.

        Returns:
            Either a string with the text it generated or a boolean indicating whether the first debater won
            (depending on whether the speech was a decision or open-ended) for each element in the batch.
        """
        batch_reasoning = self.generate(max_new_tokens=450, speech_structure=SpeechStructure.OPEN_ENDED)
        if self.transcripts[0].only_decision_remains():  # all formats should be the same so we can use any transcript
            for i, reasoning in enumerate(batch_reasoning):
                super().receive_message(speaker=self.name, content=reasoning, idx=i)

            batch_predictions = self.generate(max_new_tokens=15, speech_structure=self.speech_structure)
            validated_predictions = self.validate_responses(batch_predictions)
            returned_response = self.process_responses(validated_predictions)
            return returned_response
        return batch_reasoning

    def validate_responses(self, responses: list[str]) -> None:
        """Confirms that the responses matched the expected format"""
        for response in filter(
            lambda x: x not in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME], responses
        ):
            self.logger.warn('Response of "{}" was invalid. Must be a debater name.'.format(response))
        return responses

    def process_responses(self, responses: list[str]) -> list[Any]:
        """Converts a text response to a list of booleans indicating if Debater_A won"""
        return [constants.DEFAULT_DEBATER_A_NAME in response for response in responses]

    def copy(self, transcripts: Optional[list[Transcript]] = None) -> Judge:
        """Deep copies everything except the underlying model"""
        judge = Judge(
            name=self.name,
            prompt=[copy.deepcopy(prompt) for prompt in self.prompts],
            model=self.model,
            num_speeches=self.num_speeches,
            speech_format=self.speech_format,
            speech_structure=self.speech_structure,
            judge_type=self.judge_type,
            expected_saver=self.expected_saver,
        )
        if transcripts:
            judge.transcripts = [transcript.copy() for transcript in transcripts]
        return judge


class BoNJudge(Judge):
    def __init__(self, judge: Judge, n: int, debater_a: bool):
        """
        Abstraction for a judge that generates preferences between different versions of the same speech.

        Params:
            judge: The underlying judge object that is being converted to a BoN judge
            n: The expected number of samples of each speech
            debater_a: Boolean indicating whether the judge is to judge multiple versions of Debater_A or Debater_B's speech.
        """

        super().__init__(
            name=judge.name,
            prompt=[copy.deepcopy(judge.prompts[0]) for i in range(n)],
            model=judge.model,
            num_speeches=1,
            speech_format=JudgeUtils.get_bon_speech_format(debater_a=debater_a),
            speech_structure=SpeechStructure.PREFERENCE,
            judge_type=JudgeType.BEST_OF_N,
            expected_saver=constants.DEFAULT_DEBATER_A_NAME if debater_a else constants.DEFAULT_DEBATER_B_NAME,
        )
        self.internal_results = []
        self.debater_a = debater_a

    def validate_responses(self, responses: list[str]) -> list[str]:
        """Verifies that the response is of the correct format"""
        validated_responses = []
        for i, response in enumerate(responses):
            if not re.search("\\d+(\\.\\d+)?", response):
                self.logger.warn('Response of "{}" was invalid. Must be a number.'.format(response))
                validated_responses.append(-1)
            else:
                validated_responses.append(response)
        return validated_responses

    def process_responses(self, responses: list[str]) -> list[Any]:
        """Converts the string response to a scale of [0-10] where 0 is bad and 10 is good"""
        scores = [(float(response) if self.debater_a else (constants.MAX_SCORE - float(response))) for response in responses]
        speeches = [transcript.get_last_external_speech() for transcript in self.transcripts]
        self.internal_results.append(
            [
                {"speaker": speech.speaker, "content": speech.content, "score": score}
                for speech, score in zip(speeches, scores)
            ]
        )
        return scores

    def save(self, save_file_path_prefix: str) -> None:
        """Saves the model to the specified path"""
        idx = 0
        while os.path.exists(f"{save_file_path_prefix}_{idx}.txt"):
            idx += 1

        with open(f"{save_file_path_prefix}_{idx}.txt", "w") as f:
            json.dump(self.internal_results, f)


class JudgeUtils:
    pre_debate_speech_format = (
        SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
        .add(prompt_tag=PromptTag.JUDGE_SYSTEM)
        .add(prompt_tag=PromptTag.PRE_DEBATE_JUDGE)
    )

    opening_speech_speech_format = (
        SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_A_NAME)
        .add(prompt_tag=PromptTag.PRE_DEBATER_B_SPEECH_JUDGE)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_B_NAME)
    )

    argument_speech_format = (
        SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.JUDGE_QUESTION_INSTRUCTIONS)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_A_NAME)
        .add(prompt_tag=PromptTag.PRE_DEBATER_B_SPEECH_JUDGE)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_B_NAME)
    )

    decision_speech_format = (
        SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.POST_ROUND_JUDGE)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
    )

    @classmethod
    def get_default_speech_format(cls, num_speeches: int):
        """Gets the speech order for non-BoN judges"""
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_speech_speech_format)
            .add_format(speech_format=JudgeUtils.argument_speech_format, repeats=(num_speeches - 1))
            .add_format(speech_format=JudgeUtils.decision_speech_format)
        )

    @classmethod
    def get_bon_speech_format(cls, debater_a: bool):
        """Gets the speech order for BoN judges"""
        bon_speech_format = (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add(prompt_tag=PromptTag.BEST_OF_N_JUDGE_INSTRUCTION)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_speech_speech_format)
            .add_format(speech_format=bon_speech_format)
        )
