from __future__ import annotations

from agents.agent import Agent
from agents.models import Model, ModelResponse, SpeechStructure
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
    PREFERENCE = 2


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
        chain_of_thought: bool = True,
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
            judge_type: whether the judge is a preference judge or not
            expected_saver: whether the judge or the debater is in charge of saving the transcript
            chain_of_thought: whether the judge gets to use chain-of-thought before generating the response
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
            else JudgeUtils.get_default_speech_format(num_speeches=num_speeches, chain_of_thought=chain_of_thought),
        )
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.speech_structure = speech_structure
        self.judge_type = judge_type
        self.expected_saver = expected_saver
        self.chain_of_thought = chain_of_thought

    def generate(
        self, max_new_tokens: int = 150, speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED
    ) -> [list[ModelResponse]]:
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
        if self.chain_of_thought or not self.transcripts[0].only_decision_remains():
            batch_generation = self.generate(max_new_tokens=450, speech_structure=SpeechStructure.OPEN_ENDED)
            batch_reasoning = [response.speech for response in batch_generation]
        if self.transcripts[0].only_decision_remains():  # all formats should be the same so we can use any transcript
            if self.chain_of_thought:
                for i, reasoning in enumerate(batch_reasoning):
                    super().receive_message(speaker=self.name, content=reasoning, idx=i)
            batch_predictions = self.generate(max_new_tokens=15, speech_structure=self.speech_structure)
            validated_predictions = self.validate_responses(batch_predictions)
            returned_response = self.process_responses(validated_predictions)
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
            judge_type=self.judge_type,
            expected_saver=self.expected_saver,
            chain_of_thought=self.chain_of_thought,
        )
        if transcripts:
            judge.transcripts = [transcript.copy() for transcript in transcripts]
        return judge


class PreferenceJudge(Judge):
    def __init__(self, judge: Judge, n: int, debater_a: bool):
        """
        Abstraction for a judge that generates preferences between different versions of the same speech.

        Params:
            judge: The underlying judge object that is being converted to a preference judge
            n: The expected number of samples of each speech
            debater_a: Boolean indicating whether the judge is to judge multiple versions of Debater_A or Debater_B's speech.
        """

        super().__init__(
            name=judge.name,
            prompt=[copy.deepcopy(judge.prompts[0]) for i in range(n)],
            model=judge.model,
            num_speeches=1,
            speech_format=JudgeUtils.get_preference_speech_format(debater_a=debater_a),
            speech_structure=SpeechStructure.PREFERENCE,
            judge_type=JudgeType.PREFERENCE,
            expected_saver=constants.DEFAULT_DEBATER_A_NAME if debater_a else constants.DEFAULT_DEBATER_B_NAME,
        )
        self.internal_results = []
        self.debater_a = debater_a

    def validate_responses(self, responses: list[ModelResponse]) -> list[str]:
        """Verifies that the response is of the correct format"""
        validated_responses = []
        for i, response in enumerate(responses):
            if not response.preference:
                self.logger.warn(f'Response of "{response.preference}" was invalid. Must be a number.')
                response.preference = -1
            validated_responses.append(response)
        return validated_responses

    def process_responses(self, responses: list[ModelResponse]) -> list[Any]:
        """Converts the string response to a scale of [0-10] where 0 is bad and 10 is good"""
        scores = [
            (response.preference if self.debater_a else (constants.MAX_SCORE - response.preference))
            for response in responses
        ]
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

    @classmethod
    def get_decision_speech_format(cls, chain_of_thought: bool):
        """gets the speech format for the judge prior to making a decision"""
        decision_speech_format = SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        if chain_of_thought:
            return (
                decision_speech_format.add(prompt_tag=PromptTag.POST_ROUND_JUDGE)
                .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
                .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
            )
        return decision_speech_format.add(prompt_tag=PromptTag.POST_ROUND_JUDGE_WITHOUT_REASONING).add_user_inputted_speech(
            expected_speaker=constants.DEFAULT_JUDGE_NAME
        )

    @classmethod
    def get_default_speech_format(cls, num_speeches: int, chain_of_thought):
        """Gets the speech order for non-preference judges"""
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_speech_speech_format)
            .add_format(speech_format=JudgeUtils.argument_speech_format, repeats=(num_speeches - 1))
            .add_format(speech_format=JudgeUtils.get_decision_speech_format(chain_of_thought=chain_of_thought))
        )

    @classmethod
    def get_preference_speech_format(cls, debater_a: bool):
        """Gets the speech order for preference judges"""
        preference_speech_format = (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add(prompt_tag=PromptTag.PREFERENCE_JUDGE_INSTRUCTION)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_speech_speech_format)
            .add_format(speech_format=preference_speech_format)
        )
