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
    ):
        super().__init__(
            name=name,
            is_debater=True,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            speech_format=speech_format
            if speech_format
            else Debater.get_default_speech_format(num_speeches, name, use_scratchpad),
        )
        self.use_scratchpad = use_scratchpad
        self.logger = LoggerUtils.get_default_logger(__name__)

    def generate(self, max_new_tokens=300) -> Optional[list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=max_new_tokens, debater_name=self.name)

    def debate(self) -> Optional[list[str]]:
        if self.use_scratchpad:
            batch_reasoning = self.generate(max_new_tokens=300)
            for i, reasoning in enumerate(batch_reasoning):
                super().receive_message(speaker=self.name, content=reasoning, idx=i)
                self.logger.debug(reasoning)

        return self.generate(max_new_tokens=300)

    @classmethod
    def get_default_speech_format(cls, num_speeches: int, name: str, use_scratchpad: bool):
        pre_debate = (
            SpeechFormat()
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.DEBATER_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE)
            .add(prompt_tag=PromptTag.PRE_OPENING_SPEECH)
        )

        judge_questions = SpeechFormat().add(prompt_tag=PromptTag.PRE_JUDGE_QUESTIONS).add_user_inputted_speech()

        scratchpad = SpeechFormat().add(prompt_tag=PromptTag.DEBATER_SCRATCHPAD).add_user_inputted_speech()
        own_speech = SpeechFormat().add(prompt_tag=PromptTag.DEBATER_SCRATCHPAD).add_user_inputted_speech()
        if use_scratchpad:
            own_speech = scratchpad.add_format(speech_format=own_speech)

        opponent_speech = SpeechFormat().add(prompt_tag=PromptTag.PRE_OPPONENT_SPEECH).add_user_inputted_speech()

        opening_statements = SpeechFormat().add_format(speech_format=own_speech).add_format(speech_format=opponent_speech)

        later_arguments = (
            SpeechFormat()
            .add_format(speech_format=judge_questions)
            .add_format(speech_format=own_speech if name == constants.DEFAULT_DEBATER_A_NAME else opponent_speech)
            .add_format(speech_format=opponent_speech if name == constants.DEFAULT_DEBATER_A_NAME else own_speech)
        )

        return (
            SpeechFormat()
            .add_format(speech_format=pre_debate)
            .add_format(speech_format=opening_statements)
            .add_format(speech_format=later_arguments, repeats=(num_speeches - 1))
        )


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
