from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import SpeechFormat, Transcript

from typing import Optional, Union


class Agent:
    def __init__(
        self,
        name: str,
        is_debater: bool,
        prompt: Union[Prompt, list[Prompt]],
        model: Model,
        num_speeches: int,
        speech_format: SpeechFormat,
    ):
        self.name = name
        self.is_debater = is_debater
        self.model = model
        self.num_speeches = num_speeches

        self.prompts = prompt if type(prompt) == list else [prompt]
        self.transcripts = [Transcript(name=self.name, prompt=p, speech_format=speech_format) for p in self.prompts]

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
