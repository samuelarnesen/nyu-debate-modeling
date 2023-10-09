from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import Transcript
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from typing import Optional, Union

import random


class Agent:
    def __init__(self, name: str, is_debater: bool, prompt: Union[Prompt, list[Prompt]], model: Model):
        self.name = name
        self.is_debater = is_debater
        self.model = model

        self.prompts = prompt if type(prompt) == list else [prompt]
        self.transcripts = [Transcript(debater_name=self.name, is_debater=self.is_debater, prompt=p) for p in self.prompts]

    def receive_message(self, speaker: str, content: str, idx: int = 0):
        self.transcripts[idx].add_speech(speaker=speaker, content=content)

    def generate(self) -> Optional[list[str]]:
        pass

    def save(self, save_file_path_prefix: str):
        for i, transcript in enumerate(self.transcripts):
            transcript.save(save_file_path=f"{save_file_path_prefix}_{i}.txt")

    def get_transcript(self, idx: int = 0) -> Transcript:
        return self.transcripts[idx]

    def get_alias(self) -> str:
        return self.model.alias


class Debater(Agent):
    def __init__(self, name: str, prompt: Union[Prompt, list[Prompt]], model: Model):
        super().__init__(name=name, is_debater=True, prompt=prompt, model=model)

    def generate(self) -> Optional[list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=450)


class Judge(Agent):
    def __init__(self, name: str, prompt: Union[Prompt, list[Prompt]], model: Model):
        super().__init__(name=name, is_debater=False, prompt=prompt, model=model)
        self.logger = LoggerUtils.get_default_logger(__name__)

    def generate(self) -> [list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=150, decide=False)

    def judge(self) -> list[bool]:
        def validate_responses(predictions) -> None:
            for prediction in filter(lambda x: x not in ["Debater_A", "Debater_B"], predictions):
                self.logger.warn("Response of {} was invalid".format(prediction))

        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        predictions = self.model.predict(inputs=model_inputs, max_new_tokens=15, decide=True)
        validate_responses(predictions)

        return ["Debater_A" in prediction for prediction in predictions]
