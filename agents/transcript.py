from agents.model import ModelInput, RoleType
from agents.prompt import Prompt, PromptTag
import utils.constants as constants

from pydantic import BaseModel


class Speech(BaseModel):
    speaker: str
    content: str


class Transcript:
    def __init__(self, is_debater: bool, debater_name: str, prompt: Prompt, num_speeches: int):
        self.prompt = prompt
        self.is_debater = is_debater
        self.debater_name = debater_name
        self.speeches = []
        self.complete = False
        self.num_speeches = num_speeches
        self.progression = self.__get_progression()

    def reset(self) -> None:
        self.speeches = []

    def add_speech(self, speaker: str, content: str) -> None:
        self.speeches.append(Speech(speaker=speaker, content=content))

    def save(self, save_file_path: str) -> None:
        with open(save_file_path, "w") as f:
            f.write(str(self.full_string_value()))

    def __get_progression(self):
        if self.is_debater:
            progression = [
                PromptTag.OVERALL_SYSTEM,
                PromptTag.DEBATER_SYSTEM,
                PromptTag.PRE_DEBATE,
                PromptTag.PRE_OPENING_SPEECH,
                None,
                PromptTag.PRE_OPPONENT_SPEECH,
                None,
            ]  # Note: this will interact poorly for Debater B on the opening speech order
            speech_progression = []
            for i in range(self.num_speeches - 1):
                speech_progression += [PromptTag.PRE_JUDGE_QUESTIONS, None]
                if self.debater_name == constants.DEFAULT_DEBATER_A_NAME:
                    speech_progression += [PromptTag.PRE_LATER_SPEECH, None, PromptTag.PRE_OPPONENT_SPEECH, None]
                else:
                    speech_progression += [PromptTag.PRE_OPPONENT_SPEECH, None, PromptTag.PRE_LATER_SPEECH, None]

            return progression + speech_progression
        else:
            progression = [
                PromptTag.OVERALL_SYSTEM,
                PromptTag.JUDGE_SYSTEM,
                PromptTag.PRE_DEBATE_JUDGE,
            ]

            speech_progression = []
            for i in range(self.num_speeches):
                if i > 0:
                    speech_progression += [PromptTag.JUDGE_QUESTION_INSTRUCTIONS, None]
                speech_progression += [
                    PromptTag.PRE_DEBATER_A_SPEECH_JUDGE,
                    None,
                    PromptTag.PRE_DEBATER_B_SPEECH_JUDGE,
                    None,
                ]

            decision_progression = [PromptTag.POST_ROUND_JUDGE, None, None]

            return progression + speech_progression + decision_progression

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
        for i, tag in enumerate(self.progression):
            if tag:
                add_to_model_inputs(
                    model_inputs,
                    ModelInput(role=RoleType.SYSTEM if i < 2 else RoleType.USER, content=self.prompt.messages[tag].content),
                )
            else:
                if index >= len(self.speeches):
                    break
                role = RoleType.USER if self.speeches[index].speaker != self.debater_name else RoleType.ASSISTANT
                add_to_model_inputs(model_inputs, ModelInput(role=role, content=self.speeches[index].content))
                index += 1

        return model_inputs

    def __str__(self):
        return f"Name: {self.debater_name}\n\n" + "\n\n".join([str(speech) for speech in self.speeches])

    def full_string_value(self):
        return "\n\n".join([x.content for x in self.to_model_input()])
