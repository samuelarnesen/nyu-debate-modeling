from agents.agent import Agent
from agents.model import Model, SpeechStructure
from agents.prompt import Prompt, PromptTag
from agents.transcript import SpeechFormat, SpeechType, Transcript
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from enum import Enum
from typing import Any, Optional, Union
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
        prompt: Union[Prompt, list[Prompt]],
        model: Model,
        num_speeches: int,
        speech_format: Optional[SpeechFormat] = None,
    ):
        super().__init__(
            name=name,
            is_debater=False,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            speech_format=speech_format if speech_format else JudgeUtils.get_default_speech_format(num_speeches),
        )
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.speech_structure = SpeechStructure.DECISION
        self.judge_type = JudgeType.STANDARD

    def generate(
        self, max_new_tokens: int = 150, speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED
    ) -> [list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=max_new_tokens, speech_structure=speech_structure)

    def __call__(self) -> list[Union[str, bool]]:
        def validate_responses(predictions) -> None:
            for prediction in filter(
                lambda x: x not in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME], predictions
            ):
                self.logger.warn("Response of {} was invalid".format(prediction))

        batch_reasoning = self.generate(max_new_tokens=150, speech_structure=SpeechStructure.OPEN_ENDED)
        if self.transcripts[0].only_decision_remains():  # all formats should be the same so we can use any transcript
            for i, reasoning in enumerate(batch_reasoning):
                super().receive_message(speaker=self.name, content=reasoning, idx=i)

            batch_predictions = self.generate(max_new_tokens=15, speech_structure=self.speech_structure)
            self.validate_responses(batch_predictions)

            return self.process_responses(batch_predictions)
        return batch_reasoning

    def validate_responses(self, responses: list[str]) -> None:
        for response in filter(
            lambda x: x not in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME], responses
        ):
            self.logger.warn("Response of {} was invalid".format(response))

    def process_responses(self, responses: list[str]) -> list[Any]:
        return [constants.DEFAULT_DEBATER_A_NAME in response for response in responses]


class BoNJudge(Judge):
    def __init__(self, judge: Judge, n: int):
        super().__init__(
            name=judge.name,
            prompt=[judge.prompts[0] for i in range(n)],
            model=judge.model,
            num_speeches=1,
            speech_format=JudgeUtils.get_bon_speech_format(),
        )
        self.speech_structure = SpeechStructure.PREFERENCE
        self.judge_type = JudgeType.BEST_OF_N
        self.internal_results = []

    def receive_message(self, speaker: str, content: str, idx: int = 0):
        if speaker != self.name:  # have to replicate debater speeches to all transcripts
            for transcript in self.transcripts:
                transcript.add_speech(speaker=speaker, content=content)
        else:
            self.transcripts[idx].add_speech(speaker=speaker, content=content)

    def validate_responses(self, responses: list[str]) -> None:
        for i, response in enumerate(filter(lambda x: not re.match("\\d+(\\.\\d+)?", x), responses)):
            self.logger.warn("Response of {} was invalid".format(response))
            responses[i] = None

    def process_responses(self, responses: list[str]) -> list[Any]:
        scores = [float(response) for response in filter(lambda x: x is not None, responses)]
        speeches = [transcript.get_last_external_speech() for transcript in self.transcripts]
        self.internal_results.append(
            [
                {"speaker": speech.speaker, "content": speech.content, "score": score}
                for speech, score in zip(speeches, scores)
            ]
        )
        return scores

    def save(self, save_file_path_prefix: str) -> None:
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

    bon_opening_speech_format = (
        SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_A_NAME)
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

    bon_speech_format = (
        SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        .add(prompt_tag=PromptTag.BEST_OF_N_JUDGE_INSTRUCTION)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
    )

    @classmethod
    def get_default_speech_format(cls, num_speeches: int):
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_speech_speech_format)
            .add_format(speech_format=JudgeUtils.argument_speech_format, repeats=(num_speeches - 1))
            .add_format(speech_format=JudgeUtils.decision_speech_format)
        )

    @classmethod
    def get_bon_speech_format(cls):
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.bon_opening_speech_format)
            .add_format(speech_format=JudgeUtils.bon_speech_format)
        )
