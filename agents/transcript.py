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


class SpeechFormat:
    def __init__(self):
        self.progression = []

    def add(self, speech_type: Optional[SpeechType] = None, prompt_tag: Optional[PromptTag] = None):
        speech_type = speech_type if speech_type else (SpeechType.PRE_FILLED if prompt_tag else SpeechType.USER_INPUTTED)
        if speech_type == SpeechType.PRE_FILLED and prompt_tag:
            self.progression.append((speech_type, prompt_tag))
        else:
            self.progression.append((speech_type, None))
        return self

    def add_user_inputted_speech(self):
        self.progression.append((SpeechType.USER_INPUTTED, None))
        return self

    def add_format(self, speech_format: SpeechFormat, repeats: num_repetitions = 1):
        for i in range(repeats):
            for speech_type, prompt_tag in speech_format:
                self.add(speech_type=speech_type, prompt_tag=prompt_tag)
        return self

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.progression):
            speech_type, prompt_tag = self.progression[self.idx]
            self.idx += 1
            return speech_type, prompt_tag
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
        for i, (speech_type, tag) in enumerate(self.speech_format):
            if speech_type == SpeechType.PRE_FILLED:
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(role=RoleType.SYSTEM if i < 2 else RoleType.USER, content=self.prompt.messages[tag].content),
                )
            else:
                if index >= len(self.speeches):
                    break
                role = RoleType.USER if self.speeches[index].speaker != self.name else RoleType.ASSISTANT
                add_to_model_inputs(model_inputs, ModelInput(role=role, content=self.speeches[index].content))
                index += 1

        return model_inputs

    def pop(self):
        if len(self.speeches) > 0:
            self.speeches = self.speeches[:-1]

    def __str__(self):
        return f"Name: {self.name}\n\n" + "\n\n".join([str(speech) for speech in self.speeches])

    def full_string_value(self):
        return "\n\n".join([x.content for x in self.to_model_input()])
