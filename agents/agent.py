from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import Transcript

from typing import Optional

import random  # remove when judge is no longer stupid


class Agent:
    def __init__(self, name: str, is_debater: bool, prompt: Prompt, model: Model):
        self.name = name
        self.is_debater = is_debater
        self.prompt = prompt
        self.model = model
        self.transcript = Transcript(debater_name=self.name, is_debater=self.is_debater, prompt=self.prompt)

    def receive_message(self, speaker: str, content: str):
        self.transcript.add_speech(speaker=speaker, content=content)

    def reset(self):
        self.transcript = Transcript(debater_name=self.name, is_debater=self.is_debater, prompt=self.prompt)

    def generate(self) -> Optional[str]:
        if self.transcript:
            model_input = self.transcript.to_model_input()
            return self.model.predict(model_input)
        else:
            return None


class Debater(Agent):
    def __init__(self, name: str, prompt: Prompt, model: Model):
        super().__init__(name=name, is_debater=True, prompt=prompt, model=model)


class Judge(Agent):
    def __init__(self, name: str, prompt: Prompt, model: Model):
        super().__init__(name=name, is_debater=False, prompt=prompt, model=model)

    def generate(self) -> Optional[tuple[str, bool]]:
        if self.transcript:
            text = super().generate()
            return text, random.random() < 0.5
        return None
