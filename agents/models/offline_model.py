from __future__ import annotations

from agents.models.model import Model, ModelInput
from prompts import Prompt, PromptTag
from utils import InputUtils
import utils.constants as constants

from typing import Union, Optional
import os
import random


class OfflineModel(Model):
    def __init__(self, alias: str, is_debater: bool, file_path: str, prompt: Prompt, **kwargs):
        super().__init__(alias=alias, is_debater=is_debater)
        self.speeches = self.__load(file_path=file_path, prompt=prompt)
        self.speech_idx = 0
        self.last_round_idx = -1
        self.last_debater_name = ""
        self.prompt = prompt

    def predict(
        self, inputs: list[list[ModelInput]], max_new_tokens: int = 250, debater_name: str = "", round_idx: int = 0, **kwargs
    ) -> str:
        if not debater_name:
            raise Exception(
                "Debater name cannot be empty -- did you try using the OfflineModel as a judge? That's not supported"
            )
        if self.last_debater_name != debater_name:
            self.last_round_idx = -1
        if self.last_round_idx != round_idx:
            self.speech_idx = 0
            self.last_round_idx = round_idx

        speech = self.speeches[round_idx][debater_name][self.speech_idx]

        self.speech_idx += 1
        return [speech]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> OfflineModel:
        return OfflineModel(alias=alias, is_debater=is_debater, prompt=self.prompt)

    def __load(self, file_path: str, prompt: Prompt):
        file_texts = InputUtils.read_file_texts(base_path=file_path)
        debate_rounds = [self.__extract_speeches(text=text, prompt=prompt) for text in file_texts]
        debater_to_speech_map = []
        for i, debate_round in enumerate(debate_rounds):
            debater_to_speech_map.append({})
            for speaker, speech in debate_round:
                debater_to_speech_map[i].setdefault(speaker, [])
                debater_to_speech_map[i][speaker].append(speech)

        return debater_to_speech_map

    def __extract_speeches(self, text: str, prompt: Prompt) -> list[str]:
        def get_index(text, targets):
            max_length = 100
            for target in targets:
                index = text.find(target[: min(len(target), max_length)])
                if index > -1:
                    return index, target
            return float("inf"), None

        start_text = prompt.messages[PromptTag.PRE_DEBATER_A_SPEECH_JUDGE].content
        mid_text = prompt.messages[PromptTag.PRE_DEBATER_B_SPEECH_JUDGE].content
        end_text_one = prompt.messages[PromptTag.JUDGE_QUESTION_INSTRUCTIONS].content
        end_text_two = prompt.messages[PromptTag.POST_ROUND_JUDGE].content

        speeches = []
        keep_parsing = True
        text_to_parse = text
        while keep_parsing:
            first_speech_start, used_start_target = get_index(text_to_parse, start_text)
            first_speech_end, used_mid_target = get_index(text_to_parse, mid_text)
            second_speech_end, used_end_target = min(
                get_index(text_to_parse, end_text_one), get_index(text_to_parse, end_text_two)
            )
            speeches.append(
                (
                    constants.DEFAULT_DEBATER_A_NAME,
                    (text_to_parse[first_speech_start + len(used_start_target) : first_speech_end].lstrip().rstrip()),
                )
            )
            speeches.append(
                (
                    constants.DEFAULT_DEBATER_B_NAME,
                    (text_to_parse[first_speech_end + len(used_mid_target) : second_speech_end].lstrip().rstrip()),
                )
            )

            text_to_parse = text_to_parse[second_speech_end + len(used_end_target) :]
            _, remaining_end_target = get_index(text_to_parse, end_text_two)
            keep_parsing = remaining_end_target is not None

        return speeches
