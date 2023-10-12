from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import Transcript

from typing import Optional, Union


class Agent:
    def __init__(self, name: str, is_debater: bool, prompt: Union[Prompt, list[Prompt]], model: Model, num_speeches: int):
        self.name = name
        self.is_debater = is_debater
        self.model = model
        self.num_speeches = num_speeches

        self.prompts = prompt if type(prompt) == list else [prompt]
        self.transcripts = [
            Transcript(debater_name=self.name, is_debater=self.is_debater, prompt=p, num_speeches=num_speeches)
            for p in self.prompts
        ]

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
