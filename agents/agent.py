from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import Transcript
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

    def generate(self) -> Optional[str]:
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

    def generate(self) -> Optional[str]:
        if self.transcripts:
            model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
            return self.model.predict(model_inputs)
        else:
            return None


class Judge(Agent):
    def __init__(self, name: str, prompt: Union[Prompt, list[Prompt]], model: Model):
        super().__init__(name=name, is_debater=False, prompt=prompt, model=model)

    def generate(self) -> list[tuple[str, bool]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        text_lists = self.model.predict(inputs=model_inputs, max_new_tokens=50)

        result_list = []
        for texts in text_lists:
            results = {constants.DEFAULT_DEBATER_A_NAME: 0, constants.DEFAULT_DEBATER_B_NAME: 0}
            for text in texts:
                if constants.DEFAULT_DEBATER_A_NAME in text:
                    results[constants.DEFAULT_DEBATER_A_NAME] += 1
                if constants.DEFAULT_DEBATER_B_NAME in text:
                    results[constants.DEFAULT_DEBATER_B_NAME] += 1
            debater_a_win = (
                True
                if results[constants.DEFAULT_DEBATER_A_NAME] > results[constants.DEFAULT_DEBATER_B_NAME]
                else (
                    False
                    if results[constants.DEFAULT_DEBATER_A_NAME] < results[constants.DEFAULT_DEBATER_B_NAME]
                    else random.random() < 0.5
                )
            )
            result_list.append((f"Judge decision: debater_a_win: {debater_a_win}", debater_a_win))

        return result_list
