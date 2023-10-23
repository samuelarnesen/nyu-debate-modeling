from __future__ import annotations

from agents.model import ModelInput, RoleType
from agents.prompt import Prompt, PromptTag
import utils.constants as constants

from pydantic import BaseModel

from enum import Enum
from typing import Callable, Optional, Union


class Speech(BaseModel):
    speaker: str
    content: str


class SpeechType(Enum):
    PRE_FILLED = 1
    USER_INPUTTED = 2


class SpeechFormatEntry(BaseModel):
    speech_type: SpeechType
    prompt_tag: Optional[PromptTag]
    expected_speaker: Optional[str]


class SpeechFormat:
    def __init__(self, name: str):
        self.progression = []
        self.name = name

    def add(
        self,
        speech_type: Optional[SpeechType] = None,
        prompt_tag: Optional[PromptTag] = None,
        expected_speaker: Optional[str] = None,
    ):
        speech_type = speech_type if speech_type else (SpeechType.PRE_FILLED if prompt_tag else SpeechType.USER_INPUTTED)
        self.progression.append(
            SpeechFormatEntry(speech_type=speech_type, prompt_tag=prompt_tag, expected_speaker=expected_speaker)
        )
        return self

    def add_user_inputted_speech(self, expected_speaker: Optional[str] = None):
        return self.add(
            speech_type=SpeechType.USER_INPUTTED,
            prompt_tag=None,
            expected_speaker=expected_speaker if expected_speaker else self.name,
        )

    def add_format(self, speech_format: SpeechFormat, repeats: num_repetitions = 1):
        for i in range(repeats):
            for speech_type, prompt_tag, expected_speaker in speech_format:
                self.add(
                    speech_type=speech_type,
                    prompt_tag=prompt_tag,
                    expected_speaker=expected_speaker,
                )
        return self

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.progression):
            entry = self.progression[self.idx]
            self.idx += 1
            return entry.speech_type, entry.prompt_tag, entry.expected_speaker
        else:
            raise StopIteration


class Transcript:
    def __init__(self, name: str, prompt: Prompt, speech_format: SpeechFormat):
        self.prompt = prompt
        self.name = name
        self.speeches = []
        self.speech_format = speech_format

    def reset(self) -> None:
        self.speeches = []

    def add_speech(self, speaker: str, content: str) -> None:
        self.speeches.append(Speech(speaker=speaker, content=content))

    def save(self, save_file_path: str) -> None:
        with open(save_file_path, "w") as f:
            f.write(str(self.full_string_value()))

    def to_model_input(self):
        def add_to_model_inputs(model_inputs: list[ModelInput], new_addition: ModelInput) -> None:
            if model_inputs and model_inputs[-1].role == new_addition.role:
                model_inputs[-1] = ModelInput(
                    role=new_addition.role, content=f"{model_inputs[-1].content}\n\n{new_addition.content}"
                )
            else:
                model_inputs.append(new_addition)

        model_inputs = []
        index = 0
        for i, (speech_type, prompt_tag, expected_speaker) in enumerate(self.speech_format):
            if speech_type == SpeechType.PRE_FILLED:
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(
                        role=RoleType.SYSTEM if i < 2 else RoleType.USER,
                        content=self.prompt.messages[prompt_tag].content,
                    ),
                )
            else:
                if index >= len(self.speeches):
                    break
                role = RoleType.USER if self.speeches[index].speaker != self.name else RoleType.ASSISTANT
                add_to_model_inputs(model_inputs, ModelInput(role=role, content=self.speeches[index].content))
                index += 1

        return model_inputs

    def get_last_external_speech(self) -> Optional[str]:
        for i in range(len(self.speeches)):
            speech = self.speeches[len(self.speeches) - i - 1]
            if speech.speaker != self.name:
                return speech
        return ""

    def get_next_expected_speaker(self) -> Optional[str]:
        expected_speakers = [expected_speaker for _, _, expected_speaker in filter(lambda x: x[-1], self.speech_format)]
        expected_speaker = expected_speakers[len(self.speeches)] if len(self.speeches) < len(expected_speakers) else None

        tags = [(tag, expected_speaker) for _, tag, expected_speaker in filter(lambda x: x[-1], self.speech_format)]
        return expected_speaker

    def only_decision_remains(self) -> bool:
        expected_speakers = [expected_speaker for _, _, expected_speaker in filter(lambda x: x[-1], self.speech_format)]
        remaining_speakers = (
            set(expected_speakers[len(self.speeches) :]) if len(self.speeches) < len(expected_speakers) else set()
        )
        return constants.DEFAULT_JUDGE_NAME in remaining_speakers and len(remaining_speakers) == 1

    def pop(self):
        if len(self.speeches) > 0:
            self.speeches = self.speeches[:-1]

    def __str__(self):
        return f"Name: {self.name}\n\n" + "\n\n".join([str(speech) for speech in self.speeches])

    def full_string_value(self):
        return "\n\n".join([x.content for x in self.to_model_input()])
