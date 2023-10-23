from agents.agent import Agent
from agents.model import Model
from agents.models.offline_model import OfflineModel
from agents.prompt import Prompt, PromptTag
from agents.transcript import SpeechFormat, Transcript
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from typing import Optional, Union


class Debater(Agent):
    def __init__(
        self,
        name: str,
        prompt: Union[Prompt, list[Prompt]],
        model: Model,
        num_speeches: int,
        speech_format: Optional[SpeechFormat] = None,
        use_scratchpad: bool = True,
        best_of_n: bool = False,
    ):
        super().__init__(
            name=name,
            is_debater=True,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            speech_format=speech_format
            if speech_format
            else DebaterUtils.get_default_speech_format(name, num_speeches, use_scratchpad, best_of_n),
        )
        self.use_scratchpad = use_scratchpad
        self.logger = LoggerUtils.get_default_logger(__name__)

    def generate(self, max_new_tokens=300) -> Optional[list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=max_new_tokens, debater_name=self.name)

    def __call__(self) -> Optional[list[str]]:
        if self.use_scratchpad:
            batch_reasoning = self.generate(max_new_tokens=300)
            for i, reasoning in enumerate(batch_reasoning):
                super().receive_message(speaker=self.name, content=reasoning, idx=i)
                self.logger.debug(reasoning)

        return self.generate(max_new_tokens=300)


class OfflineDebater(Debater):
    def __init__(self, debater: Debater, file_path: str, first_debater_prompt: Prompt):
        super().__init__(
            name=debater.name,
            prompt=debater.prompts,
            model=OfflineModel(
                alias=debater.model.alias, is_debater=debater.is_debater, file_path=file_path, prompt=first_debater_prompt
            ),
            num_speeches=debater.num_speeches,
        )


class DebaterUtils:
    @classmethod
    def get_own_speech_format(cls, name: str, use_scratchpad: bool, opening_speech: bool):
        scratchpad = (
            SpeechFormat(name).add(prompt_tag=PromptTag.DEBATER_SCRATCHPAD).add_user_inputted_speech(expected_speaker=name)
        )
        pre_speech_tag = PromptTag.PRE_OPENING_SPEECH if opening_speech else PromptTag.PRE_SPEECH
        own_speech = SpeechFormat(name).add(prompt_tag=pre_speech_tag).add_user_inputted_speech(expected_speaker=name)
        if use_scratchpad:
            own_speech = scratchpad.add_format(speech_format=own_speech)
        return own_speech

    @classmethod
    def get_default_speech_format(cls, name: str, num_speeches: int, use_scratchpad: bool, best_of_n: bool = False):
        opponent_name = (
            constants.DEFAULT_DEBATER_A_NAME
            if name == constants.DEFAULT_DEBATER_B_NAME
            else constants.DEFAULT_DEBATER_B_NAME
        )
        pre_debate = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.DEBATER_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE)
        )

        judge_questions = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_JUDGE_QUESTIONS)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

        scratchpad = (
            SpeechFormat(name).add(prompt_tag=PromptTag.DEBATER_SCRATCHPAD).add_user_inputted_speech(expected_speaker=name)
        )
        own_speech = SpeechFormat(name).add(prompt_tag=PromptTag.PRE_SPEECH).add_user_inputted_speech(expected_speaker=name)
        if use_scratchpad:
            own_speech = scratchpad.add_format(speech_format=own_speech)

        opponent_speech = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_OPPONENT_SPEECH)
            .add_user_inputted_speech(expected_speaker=opponent_name)
        )

        opening_statements = (
            SpeechFormat(name).add(prompt_tag=PromptTag.PRE_OPENING_SPEECH).add_format(speech_format=own_speech)
        )

        if not best_of_n:
            opening_statements = opening_statements.add_format(speech_format=opponent_speech)

        later_arguments = (
            SpeechFormat(name)
            .add_format(speech_format=judge_questions)
            .add_format(speech_format=own_speech if name == constants.DEFAULT_DEBATER_A_NAME else opponent_speech)
            .add_format(speech_format=opponent_speech if name == constants.DEFAULT_DEBATER_A_NAME else own_speech)
        )

        decision = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.JUDGE_DECISION_FOR_DEBATER)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

        return (
            SpeechFormat(name)
            .add_format(speech_format=pre_debate)
            .add_format(speech_format=opening_statements)
            .add_format(speech_format=later_arguments, repeats=(num_speeches - 1))
            .add_format(speech_format=decision)
        )
