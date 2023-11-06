from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import SpeechFormat, Transcript
from utils.logger_utils import LoggerUtils

from typing import Optional, Union


class Agent:
    def __init__(
        self,
        name: str,
        is_debater: bool,
        prompt: Union[Prompt, list[Prompt]],
        model: Model,
        num_speeches: int,
        validate_quotes: bool,
        speech_format: SpeechFormat,
    ):
        self.name = name
        self.is_debater = is_debater
        self.model = model
        self.num_speeches = num_speeches
        self.validate_quotes = validate_quotes
        self.speech_format = speech_format

        self.prompts = prompt if type(prompt) == list else [prompt]
        self.transcripts = [
            Transcript(name=self.name, prompt=p, speech_format=speech_format, index=i) for i, p in enumerate(self.prompts)
        ]
        self.cached_messages = {}

    def receive_message(self, speaker: str, content: str, idx: int):
        if idx >= len(self.transcripts):
            return

        self.cached_messages.setdefault(speaker, {}).setdefault(idx, []).append(content)
        expected_speaker = self.get_next_expected_speaker(idx=idx)
        while self.cached_messages.get(expected_speaker, {}).get(idx):
            for message in self.cached_messages[expected_speaker][idx]:
                self.transcripts[idx].add_speech(speaker=expected_speaker, content=message)
            del self.cached_messages[expected_speaker][idx]
            expected_speaker = self.get_next_expected_speaker(idx=idx)

    def __call__(self) -> Optional[list[str]]:
        pass

    def save(self, save_file_path_prefix: str):
        for i, transcript in enumerate(self.transcripts):
            transcript.save(save_file_path=f"{save_file_path_prefix}_{i}.txt")

    def get_transcript(self, idx: int = 0) -> Transcript:
        return self.transcripts[idx]

    def get_alias(self) -> str:
        return self.model.alias

    def get_next_expected_speaker(self, idx: int = 0) -> Optional[str]:
        return self.transcripts[idx].get_next_expected_speaker()
