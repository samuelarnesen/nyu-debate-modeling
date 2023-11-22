from agents.prompt import Prompt, PromptParser, PromptTag
from data.data import DatasetType, JudgePreferenceDataRow, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
from utils.input_utils import InputUtils
import utils.constants as constants

from typing import Any, Optional
import json
import os
import re
import sys


class JudgePreferencesDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        super().__init__(DatasetType.JUDGE_PREFERENCES)
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[JudgePreferenceDataRow]:
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[JudgePreferenceDataRow]:
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> JudgePreferenceDataRow:
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, train_data: list[tuple[str, str, str]]):
        return [
            JudgePreferenceDataRow(prompt=instruction, chosen=chosen, rejected=rejected)
            for instruction, chosen, rejected in train_data
        ]


class JudgePreferencesLoader(RawDataLoader):
    @classmethod
    def load(cls, full_dataset_filepath: str, **kwargs) -> JudgePreferencesDataset:
        train_data = []
        input_texts = InputUtils.read_file_texts(base_path=full_dataset_filepath, group_by_batch=True)
        for batch in input_texts:
            position = 0 if constants.DEBATER_A_IDENTIFICATION in batch[0] else 1
            parsed_texts = []
            previous_instruction = None
            for example in batch:
                instruction_end_text = re.search("|".join(constants.BEGIN_SPEECH_OPTIONS), example, re.DOTALL).group()
                verdict_start_text = re.search("|".join(constants.BEGIN_JUDGING_OPTIONS), example, re.DOTALL).group()
                instruction_end = example.find(instruction_end_text) + len(instruction_end_text)
                verdict_start = example.find(verdict_start_text)
                verdict_end = verdict_start + len(verdict_start_text)
                instruction = example[:instruction_end].strip()
                speech = example[instruction_end:verdict_start].strip()
                verdict = (
                    float(example[verdict_end:])
                    if constants.DEBATER_A_IDENTIFICATION
                    else constants.MAX_SCORE - float(example[verdict_end:])
                )
                parsed_texts.append((instruction, speech, verdict))

                if previous_instruction:
                    assert instruction == previous_instruction
                previous_instruction = instruction

            sorted_parsed_texts = sorted(parsed_texts, key=lambda x: x[2], reverse=True)
            sorted_speeches = [speech for _, speech, _ in sorted_parsed_texts]
            train_data.append((instruction, sorted_speeches[0], sorted_speeches[-1]))
            if len(sorted_speeches) > 4:
                train_data.append((instruction, sorted_speeches[1], sorted_speeches[-2]))

        return JudgePreferencesDataset(
            train_data=train_data,
            val_data=[],
            test_data=[],
        )
