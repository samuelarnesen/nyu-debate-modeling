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

    def reset(self) -> None:
        self.speeches = []

    def add_speech(self, speaker: str, content: str) -> None:
        self.speeches.append(Speech(speaker=speaker, content=content))

    def save(self, save_file_path: str) -> None:
        with open(save_file_path, "w") as f:
            f.write(str(self))

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

        if self.is_debater:
            if self.prompt.messages[PromptTag.PRE_DEBATE]:
                add_to_model_inputs(
                    model_inputs, ModelInput(role=RoleType.USER, content=self.prompt.messages[PromptTag.PRE_DEBATE].content)
                )
        else:
            if self.prompt.messages[PromptTag.PRE_DEBATE_JUDGE]:
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(role=RoleType.USER, content=self.prompt.messages[PromptTag.PRE_DEBATE_JUDGE].content),
                )

        for i, speech in enumerate(self.speeches):
            role = RoleType.USER if speech.speaker != self.debater_name else RoleType.ASSISTANT
            if self.is_debater:
                if speech.speaker == self.debater_name:
                    tag = PromptTag.PRE_OPENING_SPEECH if i == 0 else PromptTag.PRE_LATER_SPEECH
                    add_to_model_inputs(
                        model_inputs, ModelInput(role=RoleType.USER, content=self.prompt.messages[tag].content)
                    )
                    add_to_model_inputs(model_inputs, ModelInput(role=RoleType.ASSISTANT, content=speech.content))
                else:
                    add_to_model_inputs(
                        model_inputs,
                        ModelInput(role=RoleType.USER, content=self.prompt.messages[PromptTag.PRE_OPPONENT_SPEECH].content),
                    )
                    add_to_model_inputs(model_inputs, ModelInput(role=RoleType.USER, content=speech.content))
            else:
                tag = (
                    PromptTag.PRE_DEBATER_A_SPEECH_JUDGE
                    if speech.speaker == self.debater_name
                    else PromptTag.PRE_DEBATER_B_SPEECH_JUDGE
                )
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(role=RoleType.USER, content=self.prompt.messages[tag].content),
                )
                add_to_model_inputs(model_inputs, ModelInput(role=RoleType.USER, content=speech.content))
        if not self.is_debater:
            add_to_model_inputs(
                model_inputs,
                ModelInput(role=RoleType.USER, content=self.prompt.messages[PromptTag.POST_ROUND_JUDGE].content),
            )

        return model_inputs

    def __str__(self):
        return f"Name: {self.debater_name}\n\n" + "\n\n".join([str(speech) for speech in self.speeches])

    def full_string_value(self):
        return "\n\n".join([x.content for x in self.to_model_input()])
