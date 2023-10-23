from agents.prompt import Prompt, PromptParser, PromptTag
from data.data import JudgePreferenceDataRow, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
from utils.input_utils import InputUtils
import utils.constants as constants

from typing import Any, Optional
import json
import os


class JudgePreferencesDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
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
            JudgePreferenceDataRow(instruction=instruction, chosen=chosen, rejected=rejected)
            for instruction, chosen, rejected in train_data
        ]


class JudgePreferencesLoader(RawDataLoader):
    @classmethod
    def load(cls, full_dataset_filepath: str, prompts_file_path: str, prompt_name: str, **kwargs) -> JudgePreferencesDataset:
        train_data = []
        input_texts = InputUtils.read_file_texts(base_path=full_dataset_filepath, group_by_batch=True)
        for batch in input_texts:
            position = 0 if constants.DEBATER_A_IDENTIFICATION in batch[0] else 1
            prompt_config = PromptParser.convert_data_row_to_default_prompt_config(row=row, position=speech.position)
            prompt = PromptParser.parse(prompts_file_path=prompts_file_path, prompt_config=prompt_config, name=prompt_name)

            parsed_texts = []
            instruction = None
            instruction_end = prompt.messages[PromptTag.PRE_SPEECH].content
            verdict_begin = prompt.messages[PromptTag.JUDGE_DECISION_FOR_DEBATER]
            for example in batch:
                if not instruction:
                    instruction_index = text.find(instruction_end)
                    assert instruction_index != -1
                    instruction = example[0 : instruction_index + len(instruction_end)]

                verdict_index = text.find(verdict_begin)
                speech = example[instruction_index + len(instruction_end) : verdict_index].strip()
                verdict = float(example[verdict_index + len(verdict_begin) :].strip())
                parsed_texts.append((instruction, speech, verdict))

            sorted_parsed_texts = sorted(parsed_texts, key=lambda x: x[2], reversed=True)
            sorted_speeches = [speech for _, speech, _ in sorted_parsed_texts]
            train_data.append((instruction, sorted_speeches[0], sorted_speeches[-1]))
            if i > 4:
                train_data.append((instruction, sorted_speeches[0], sorted_speeches[-1]))

        return JudgePreferencesDataset(
            train_data=train_data,
            val_data=[],
            test_data=[],
        )
