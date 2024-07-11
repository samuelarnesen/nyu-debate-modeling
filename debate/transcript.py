from __future__ import annotations

from debate.speech_format import Speech, SpeechType, SpeechFormatEntry, SpeechFormat
from models import ModelInput, ModelResponse
from prompts import Prompt, RoleType
import utils.constants as constants

from pydantic import BaseModel

from enum import Enum
from typing import Any, Callable, Optional, Union
import copy
import json


class Transcript:
    def __init__(
        self, name: str, prompt: Prompt, speech_format: SpeechFormat, index: int = 0, alternate_prompts: bool = False
    ):
        """
        An abstraction that tracks the commands and speeches delivered in the round. This can then
        be used to construct an input to a model.

        Params:
            name: The name of the debater who is to use this transcript.
            prompt: The prompt that is used to generate commands.
            speech_format: The order of speeches and commands that the debater expects to receive.
            index: The index corresponding to which element in the batch this transcript is being used for.
            alternate_prompts: False if the default prompts should always be used (good for validation). True if one wants to
                mix in alternate prompts (good for training)
        """
        self.prompt = prompt
        self.name = name
        self.speeches = []
        self.speech_format = speech_format
        self.alternate_prompts = alternate_prompts

    def reset(self) -> None:
        """Removes all the given speeches"""
        self.speeches = []

    def add_speech(
        self, speaker: str, content: str, supplemental: Optional[ModelResponse | list[ModelResponse]] = None
    ) -> None:
        """
        Adds an agent-generated speech to the transcript

        Args:
            speaker: The name of the debater (Debater_A, Debater_B) that gave the speech
            content: The text of the speech
            supplemental: Any additional metadata that one wants to tag along with the speech
        """
        self.speeches.append(Speech(speaker=speaker, content=content, supplemental=supplemental))

    def save(self, save_file_path_prefix: str, metadata: Optional[dict[Any, Any]]) -> None:
        """Saves to the specified path"""
        """
        with open(save_file_path_prefix + ".txt", "w") as f:
            f.write(str(self.full_string_value()))
        """
        with open(save_file_path_prefix + ".json", "w") as f:
            json.dump(self.json_value(metadata=metadata), f)

    def to_model_input(self) -> list[ModelInput]:
        """Converts the speech to a list of inputs that can be used to generate more text by models"""

        def add_to_model_inputs(model_inputs: list[ModelInput], new_addition: ModelInput) -> None:
            if model_inputs and model_inputs[-1].role == new_addition.role:
                model_inputs[-1] = ModelInput(
                    role=new_addition.role, content=f"{model_inputs[-1].content}\n\n{new_addition.content}"
                )
            else:
                model_inputs.append(new_addition)

        model_inputs = []
        index = 0
        for i, (speech_type, prompt_tag, last_only_prompt_tag, expected_speaker) in enumerate(self.speech_format):
            if speech_type == SpeechType.PRE_FILLED:
                prompt_tag_to_use = (
                    prompt_tag if (index < len(self.speeches) or not last_only_prompt_tag) else last_only_prompt_tag
                )

                content_idx = index % len(self.prompt.messages[prompt_tag_to_use].content) if self.alternate_prompts else 0
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(
                        role=RoleType.SYSTEM if i < 2 else RoleType.USER,
                        content=self.prompt.messages[prompt_tag_to_use].content[content_idx],
                    ),
                )
            else:
                if index >= len(self.speeches):
                    break
                role = (
                    RoleType.USER
                    if self.speeches[index].speaker != self.name or index < len(self.speeches)
                    else RoleType.ASSISTANT
                )

                add_to_model_inputs(model_inputs, ModelInput(role=role, content=str(self.speeches[index].content)))
                index += 1

        return model_inputs

    def get_last_external_speech(self) -> Optional[str]:
        """Get the text of the last speech that was delivered by a different agent"""
        for i in range(len(self.speeches)):
            speech = self.speeches[len(self.speeches) - i - 1]
            if speech.speaker != self.name:
                return speech
        return ""

    def get_last_internal_speech(self) -> Optional[str]:
        """Get the text of the last speech that was delivered by one self"""
        for i in range(len(self.speeches)):
            speech = self.speeches[len(self.speeches) - i - 1]
            if speech.speaker == self.name:
                return speech
        return ""

    def get_speakers(self) -> set[str]:
        """Gets a list of all the speakers who appear in the transcript"""
        return set([speech.speaker for speech in self.speeches])

    def get_next_expected_speaker(self) -> Optional[str]:
        """Gets the name of the next agent that is expected to deliver a speech"""
        expected_speakers = [expected_speaker for _, _, _, expected_speaker in filter(lambda x: x[-1], self.speech_format)]
        return expected_speakers[len(self.speeches)] if len(self.speeches) < len(expected_speakers) else None

    def only_decision_remains(self) -> bool:
        """Returns true if there are no more speeches that are expected to be delivered besides the
        judge's final verdict"""
        expected_speakers = [expected_speaker for _, _, _, expected_speaker in filter(lambda x: x[-1], self.speech_format)]
        remaining_speakers = (
            set(expected_speakers[len(self.speeches) :]) if len(self.speeches) < len(expected_speakers) else set()
        )
        return constants.DEFAULT_JUDGE_NAME in remaining_speakers and len(remaining_speakers) == 1

    def full_string_value(self) -> str:
        """Converts the transcript into a string for logging and saving"""
        return "\n\n".join([x.content for x in self.to_model_input()])

    def json_value(self, metadata: Optional[dict[Any, Any]] = None) -> str:
        """Converts the transcript into a json object that can be parsed for offline processing"""

        def clean(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for key, val in obj.items():
                    if isinstance(val, dict):
                        new_dict[key] = clean(val)
                    elif "token" in key and isinstance(val, list) and val and isinstance(val[0], int):
                        pass
                    elif isinstance(val, list):
                        new_dict[key] = [clean(item) for item in val]
                    else:
                        new_dict[key] = val
            return obj

        speeches = []
        index = 0
        for i, (speech_type, prompt_tag, _, expected_speaker) in enumerate(self.speech_format):
            supplemental = None
            if speech_type == SpeechType.PRE_FILLED:
                content = self.prompt.messages[prompt_tag].content[index % len(self.prompt.messages[prompt_tag].content)]
            else:
                if index >= len(self.speeches):
                    break
                content = self.speeches[index].content
                supplemental = clean(self.speeches[index].supplemental)
                # supplemental = {k: v for k, v in filter(lambda x: "token" not in x, self.speeches[index].supplemental)}
                index += 1
            speeches.append(Speech(speaker=expected_speaker or "Prompt", content=content, supplemental=supplemental).dict())

        return {"metadata": metadata, "speeches": speeches}

    def copy(self) -> Transcript:
        """Deepcopies this objects"""
        return copy.deepcopy(self)

    def get_external_speech_count(self) -> int:
        """Returns the number of external speeches that have been added to the transcript"""
        return len(self.speeches)

    def truncate(self, idx: int, debaters_only: bool = False) -> None:
        """
        Removes all the speeches after the specified index.

        Params:
            idx: The last speech in the round to include before removing the rest
            debaters_only: whether the idx refers to only speeches given by the debaters
        """
        max_idx = len(self.speeches)
        if debaters_only:
            counter = 0
            max_idx = 0
            idx_to_true_idx = {}
            for i, speech in enumerate(self.speeches):
                if speech.speaker != constants.DEFAULT_JUDGE_NAME:
                    idx_to_true_idx[counter] = i
                    max_idx = counter
                    counter += 1
            idx = idx_to_true_idx[idx]
        return self.speeches[: min(idx, max_idx)]

    def get_speech_count(self, debaters_only: bool = False) -> int:
        """Returns the number of speeches that have already been added (only includes speeches by debaters
        if the debaters_only parameter is true)"""
        if not debaters_only:
            return len(self.speeches)
        else:
            return len([speech for speech in filter(lambda x: x.speaker != constants.DEFAULT_JUDGE_NAME, self.speeches)])

    def __str__(self):
        """Shorter string representation as compared to full_string_value()"""
        return f"Name: {self.name}\n\n" + "\n\n".join([str(speech) for speech in self.speeches])
