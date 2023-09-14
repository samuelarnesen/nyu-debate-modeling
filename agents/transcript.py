from agents.model import ModelInput, RoleType
from agents.prompt import Prompt, PromptTag

from pydantic import BaseModel


class Speech(BaseModel):
    speaker: str
    content: str


class Transcript:
    def __init__(self, is_debater: bool, debater_name: str, prompt: Prompt):
        self.prompt = prompt
        self.is_debater = is_debater
        self.debater_name = debater_name
        self.speeches = []

    def reset(self):
        self.speeches = []

    def add_speech(self, speaker: str, content: str):
        self.speeches.append(Speech(speaker=speaker, content=content))

    # Note: this only works for debaters
    def to_model_input(self) -> list[ModelInput]:
        def add_to_model_inputs(model_inputs: list[ModelInput], new_addition: ModelInput) -> None:
            if model_inputs and model_inputs[-1].role == new_addition.role:
                model_inputs[-1] = ModelInput(
                    role=new_addition.role, content=f"{model_inputs[-1].content}\n{new_addition.content}"
                )
            else:
                model_inputs.append(new_addition)

        model_inputs = []
        if self.prompt.messages[PromptTag.OVERALL_SYSTEM]:
            add_to_model_inputs(
                model_inputs,
                ModelInput(role=RoleType.SYSTEM, content=self.prompt.messages[PromptTag.OVERALL_SYSTEM].content),
            )

        if self.is_debater and self.prompt.messages[PromptTag.DEBATER_SYSTEM]:
            add_to_model_inputs(
                model_inputs,
                ModelInput(role=RoleType.SYSTEM, content=self.prompt.messages[PromptTag.DEBATER_SYSTEM].content),
            )

        if not self.is_debater and self.prompt.messages[PromptTag.JUDGE_SYSTEM]:
            add_to_model_inputs(
                model_inputs,
                ModelInput(role=RoleType.SYSTEM, content=self.prompt.messages[PromptTag.JUDGE_SYSTEM].content),
            )

        if self.prompt.messages[PromptTag.PRE_DEBATE]:
            add_to_model_inputs(
                model_inputs, ModelInput(role=RoleType.USER, content=self.prompt.messages[PromptTag.PRE_DEBATE].content)
            )

        for i, speech in enumerate(self.speeches):
            role = RoleType.USER if speech.speaker != self.debater_name else RoleType.ASSISTANT
            if speech.speaker == self.debater_name:
                tag = PromptTag.PRE_OPENING_SPEECH if i == 0 else PromptTag.PRE_LATER_SPEECH
                add_to_model_inputs(model_inputs, ModelInput(role=RoleType.USER, content=self.prompt.messages[tag].content))
                add_to_model_inputs(model_inputs, ModelInput(role=RoleType.ASSISTANT, content=speech.content))
            else:
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(role=RoleType.USER, content=self.prompt.messages[PromptTag.PRE_OPPONENT_SPEECH].content),
                )
                add_to_model_inputs(model_inputs, ModelInput(role=RoleType.USER, content=speech.content))

        tag = PromptTag.PRE_OPENING_SPEECH if not self.speeches else PromptTag.PRE_LATER_SPEECH
        add_to_model_inputs(model_inputs, ModelInput(role=RoleType.USER, content=self.prompt.messages[tag].content))

        return model_inputs
