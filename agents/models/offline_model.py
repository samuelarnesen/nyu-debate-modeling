from __future__ import annotations

from agents.model import Model, ModelInput
from agents.prompt import Prompt, PromptTag
import utils.constants as constants

from typing import Union, Optional
import os
import random


class OfflineModel(Model):
    def __init__(self, alias: str, is_debater: bool, file_path: str, prompt: Prompt):
        super().__init__(alias=alias, is_debater=is_debater)
        self.speeches = self.__load(file_path=file_path, prompt=prompt)
        self.speech_idx = 0
        self.last_round_idx = -1
        self.last_debater_name = ""

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

    def copy(self, alias: str, is_debater: Optional[bool] = None) -> OfflineModel:
        return OfflineModel(alias=alias, is_debater=is_debater, speeches=self.speeches)

    def __load(self, file_path: str, prompt: Prompt):
        file_texts = self.__get_file_texts(base_path=file_path)
        debate_rounds = [self.__extract_speeches(text=text, prompt=prompt) for text in file_texts]
        debater_to_speech_map = []
        for i, debate_round in enumerate(debate_rounds):
            debater_to_speech_map.append({})
            for speaker, speech in debate_round:
                debater_to_speech_map[i].setdefault(speaker, [])
                debater_to_speech_map[i][speaker].append(speech)
                if i == 0 and speaker == "Debater_B" and len(debater_to_speech_map[i][speaker]) == 1:
                    in_there = "This is what Debater_B said during their speech" in debater_to_speech_map[0]["Debater_B"][0]

        return debater_to_speech_map

    def __get_file_texts(self, base_path: str) -> list[str]:
        round_idx = 0
        batch_idx = 0
        keep_extracting = True
        file_texts = []
        while keep_extracting:
            candidate_path = f"{base_path}_{round_idx}_{batch_idx}.txt"
            if os.path.exists(candidate_path):
                with open(candidate_path) as f:
                    file_texts.append(f.read())
                batch_idx += 1
            elif batch_idx == 0:
                keep_extracting = False
            else:
                round_idx += 1
                batch_idx = 0
        return file_texts

    def __extract_speeches(self, text: str, prompt: Prompt) -> list[str]:
        def get_index(text, target):
            index = text.find(target)
            if index == -1:
                return float("inf")
            return index

        start_text = prompt.messages[PromptTag.PRE_DEBATER_A_SPEECH_JUDGE].content
        mid_text = prompt.messages[PromptTag.PRE_DEBATER_B_SPEECH_JUDGE].content
        end_text_one = prompt.messages[PromptTag.JUDGE_QUESTION_INSTRUCTIONS].content
        end_text_two = prompt.messages[PromptTag.POST_ROUND_JUDGE].content

        speeches = []
        keep_parsing = True
        text_to_parse = text
        while keep_parsing:
            first_speech_start = get_index(text_to_parse, start_text)
            first_speech_end = get_index(text_to_parse, mid_text)
            second_speech_end = min(get_index(text_to_parse, end_text_one), get_index(text_to_parse, end_text_two))
            speeches.append(
                (
                    constants.DEFAULT_DEBATER_A_NAME,
                    (text_to_parse[first_speech_start + len(start_text) : first_speech_end].lstrip().rstrip()),
                )
            )
            speeches.append(
                (
                    constants.DEFAULT_DEBATER_B_NAME,
                    (text_to_parse[first_speech_end + len(mid_text) : second_speech_end].lstrip().rstrip()),
                )
            )

            text_to_parse = text_to_parse[second_speech_end + min(len(end_text_one), len(end_text_two)) :]
            keep_parsing = end_text_two in text_to_parse

        return speeches
