from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import Transcript
import utils.constants as constants

from typing import Optional

import random


class Agent:
    def __init__(self, name: str, is_debater: bool, prompt: Prompt, model: Model):
        self.name = name
        self.is_debater = is_debater
        self.prompt = prompt
        self.model = model
        self.transcript = Transcript(debater_name=self.name, is_debater=self.is_debater, prompt=self.prompt)

    def receive_message(self, speaker: str, content: str):
        self.transcript.add_speech(speaker=speaker, content=content)

    def generate(self) -> Optional[str]:
        pass

    def reset(self):
        self.transcript = Transcript(debater_name=self.name, is_debater=self.is_debater, prompt=self.prompt)

    def save(self, save_file_path: str):
        self.transcript.save(save_file_path=save_file_path)

    def get_transcript(self) -> Transcript:
        return self.transcript


class Debater(Agent):
    def __init__(self, name: str, prompt: Prompt, model: Model):
        super().__init__(name=name, is_debater=True, prompt=prompt, model=model)

    def generate(self) -> Optional[str]:
        if self.transcript:
            model_input = self.transcript.to_model_input()
            return self.model.predict(model_input)
        else:
            return None


class Judge(Agent):
    def __init__(self, name: str, prompt: Prompt, model: Model):
        super().__init__(name=name, is_debater=False, prompt=prompt, model=model)

    def generate(self) -> Optional[tuple[str, bool]]:
        model_input = self.transcript.to_model_input()
        texts = self.model.predict(inputs=model_input)

        results = {constants.DEFAULT_DEBATER_ONE_NAME: 0, constants.DEFAULT_DEBATER_TWO_NAME: 0}
        for text in texts:
            if constants.DEFAULT_DEBATER_ONE_NAME in text:
                results[constants.DEFAULT_DEBATER_ONE_NAME] += 1
            if constants.DEFAULT_DEBATER_TWO_NAME in text:
                results[constants.DEFAULT_DEBATER_TWO_NAME] += 1
        debater_a_wins = (
            True
            if results[constants.DEFAULT_DEBATER_ONE_NAME] > results[constants.DEFAULT_DEBATER_TWO_NAME]
            else (
                False
                if results[constants.DEFAULT_DEBATER_ONE_NAME] < results[constants.DEFAULT_DEBATER_TWO_NAME]
                else random.random() < 0.5
            )
        )
        return "\t".join(texts), debater_a_wins
