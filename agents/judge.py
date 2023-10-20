from agents.agent import Agent
from agents.model import Model, SpeechStructure
from agents.prompt import Prompt, PromptTag
from agents.transcript import SpeechFormat, SpeechType, Transcript
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from typing import Any, Optional, Union
import random
import re


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

    def generate(
        self, max_new_tokens: int = 150, speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED
    ) -> [list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=max_new_tokens, speech_structure=speech_structure)

    def judge(self) -> list[bool]:
        def validate_responses(predictions) -> None:
            for prediction in filter(
                lambda x: x not in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME], predictions
            ):
                self.logger.warn("Response of {} was invalid".format(prediction))

        batch_reasoning = self.generate(max_new_tokens=150, speech_structure=SpeechStructure.OPEN_ENDED)
        for i, reasoning in enumerate(batch_reasoning):
            super().receive_message(speaker=self.name, content=reasoning, idx=i)

        batch_predictions = self.generate(max_new_tokens=15, speech_structure=self.speech_structure)
        self.validate_responses(batch_predictions)

        return self.process_responses(batch_predictions)

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

    def validate_responses(self, responses: list[str]) -> None:
        for i, response in enumerate(filter(lambda x: not re.match("\\d+(\\.\\d+)?", x), responses)):
            self.logger.warn("Response of {} was invalid".format(response))
            responses[i] = None

    def process_responses(self, responses: list[str]) -> list[Any]:
        return [float(response) for response in filter(lambda x: x is not None, responses)]


class JudgeUtils:
    pre_debate_speech_format = (
        SpeechFormat()
        .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
        .add(prompt_tag=PromptTag.JUDGE_SYSTEM)
        .add(prompt_tag=PromptTag.PRE_DEBATE_JUDGE)
    )

    opening_statement_speech_format = (
        SpeechFormat()
        .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)
        .add_user_inputted_speech()
        .add(prompt_tag=PromptTag.PRE_DEBATER_B_SPEECH_JUDGE)
        .add_user_inputted_speech()
    )

    argument_speech_format = (
        SpeechFormat()
        .add(prompt_tag=PromptTag.JUDGE_QUESTION_INSTRUCTIONS)
        .add_user_inputted_speech()
        .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)
        .add_user_inputted_speech()
        .add(prompt_tag=PromptTag.PRE_DEBATER_B_SPEECH_JUDGE)
        .add_user_inputted_speech()
    )

    decision_speech_format = (
        SpeechFormat().add(prompt_tag=PromptTag.POST_ROUND_JUDGE).add_user_inputted_speech().add_user_inputted_speech()
    )

    bon_speech_format = (
        SpeechFormat()
        .add(prompt_tag=PromptTag.BEST_OF_N_JUDGE_INSTRUCTION)
        .add_user_inputted_speech()
        .add_user_inputted_speech()
    )

    @classmethod
    def get_default_speech_format(cls, num_speeches: int):
        return (
            SpeechFormat()
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_statement_speech_format)
            .add_format(speech_format=JudgeUtils.argument_speech_format, repeats=(num_speeches - 1))
            .add_format(speech_format=JudgeUtils.decision_speech_format)
        )

    @classmethod
    def get_bon_speech_format(cls):
        (
            SpeechFormat()
            .add_format(speech_format=JudgeUtils.pre_debate_speech_format)
            .add_format(speech_format=JudgeUtils.opening_statement_speech_format)
            .add_format(speech_format=JudgeUtils.argument_speech_format, repeats=1)
            .add_format(speech_format=JudgeUtils.bon_speech_format)
        )
