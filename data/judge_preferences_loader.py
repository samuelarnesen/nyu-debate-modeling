from data.dataset import DataRow, DatasetType, JudgePreferenceDataRow, RawDataLoader, RawDataset, SplitType
from data.quality_loader import QualityLoader
from utils import InputType, input_utils, quote_utils
import utils.constants as constants

from enum import Enum, auto
from typing import Any, Optional
import json, math, random


class RewardType(Enum):
    LOG_PROB = auto()
    PROB = auto()
    LOGIT = auto()
    SIGMOID = auto()
    BINARY = auto()


class JudgePreferencesDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        """
        A dataset of judge preferences from a previous best-of-n run. Each row is a pair of speeches with one
        labelled as the chosen speech and the other as the rejected speech.
        """
        super().__init__(DatasetType.JUDGE_PREFERENCES)
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[JudgePreferenceDataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[JudgePreferenceDataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> JudgePreferenceDataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def merge(self, other: RawDataset):
        """Combines the data from two datasets"""
        for key in filter(lambda x: x in other.data, self.data):
            self.data[key] += other.data[key]

    def __convert_batch_to_rows(self, train_data: list[tuple[str, str, str, float]]):
        return [
            JudgePreferenceDataRow(prompt=instruction, chosen=chosen, rejected=rejected, preference=preference)
            for instruction, chosen, rejected, preference in train_data
        ]


class JudgePreferencesLoader(RawDataLoader):
    MIN_GAP = 0.00

    @classmethod
    def process_row(
        cls, data: dict[Any, Any], reward_type: RewardType = RewardType.LOG_PROB, **kwargs
    ) -> list[tuple[str, str, str, float]]:
        def clean_speech(speech: str) -> str:
            speech = speech.replace(constants.INVALID_QUOTE_TAG, constants.QUOTE_TAG).replace(
                constants.INVALID_UNQUOTE_TAG, constants.UNQUOTE_TAG
            )
            return quote_utils.clean_up_quotes(speech_content=speech)

        def get_preference(selected: dict, rejected: dict) -> float:
            selected_pref = selected["supplemental"]["preference"]
            rejected_pref = rejected["preference"]
            if reward_type == RewardType.LOGIT:
                selected_over_rejected = selected_pref * (1 - rejected_pref)
                rejected_over_selected = rejected_pref * (1 - selected_pref)
                return selected_over_rejected / (selected_over_rejected + rejected_over_selected)
            elif reward_type == RewardType.PROB:
                multiplier = kwargs.get("multiplier", 5.75)
                return math.exp(multiplier * selected_pref) / (
                    math.exp(multiplier * selected_pref) + math.exp(multiplier * rejected_pref)
                )
            elif reward_type == RewardType.SIGMOID:
                multiplier = kwargs.get("multiplier", 5)
                temperature = kwargs.get("temperature", 0.125)
                mean = 0.5
                selected_reward = multiplier / (1 + math.exp(-((selected_pref - mean) / temperature)))
                rejected_reward = multiplier / (1 + math.exp(-((rejected_pref - mean) / temperature)))
                return math.exp(selected_reward) / (math.exp(selected_reward) + math.exp(rejected_reward))
            elif reward_type == RewardType.BINARY:
                return 1.0
            else:
                multiplier = kwargs.get("multiplier", 2.25)
                return (selected_pref**multiplier) / ((rejected_pref**multiplier) + (selected_pref**multiplier))

        outputs = []
        for selected in filter(
            lambda x: x["speaker"] in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME],
            data["speeches"],
        ):
            instruction = selected["supplemental"]["prompt"]
            rejected = sorted(selected["supplemental"]["rejected_responses"], key=lambda x: x["preference"])[0]
            if selected["supplemental"]["preference"] - rejected["preference"] > JudgePreferencesLoader.MIN_GAP:
                selected_speech = clean_speech(selected["content"])
                rejected_speech = clean_speech(rejected["speech"])
                preference = get_preference(selected, rejected)
                outputs.append((instruction, selected_speech, rejected_speech, preference))
        return outputs

    @classmethod
    def load(
        cls, full_dataset_filepath: str | list[str], reward_type: RewardType = RewardType.LOG_PROB, **kwargs
    ) -> JudgePreferencesDataset:
        """
        Constructs a JudgePreferencesDataset.

        Params:
            full_dataset_filepath: This is the *prefix* of the files with all the Best-of-N generations.

        Returns:
            A JudgePreferencesDataset where each row has a chosen and a rejected speech.
        """

        train_data = []
        input_texts = input_utils.read_file_texts(base_path=full_dataset_filepath, input_type=InputType.JSON_TRANSCRIPT)
        for text in input_texts:
            train_data.extend(JudgePreferencesLoader.process_row(json.loads(text), reward_type=reward_type, **kwargs))

        return JudgePreferencesDataset(
            train_data=train_data,
            val_data=[],
            test_data=[],
        )


class CorrectnessJudgePreferencesLoader(RawDataLoader):
    @classmethod
    def load(cls, full_dataset_filepath: str | list[str], **kwargs) -> JudgePreferencesDataset:
        """
        Constructs a CorrectnessJudgePreferencesDataset. This is a modified JudgePreferencesDataset where a speech is marked
        as "chosen" if it defends the correct side and "rejected" if it defends the incorrect side. The "preference" is just
        the outright win probability.

        Params:
            full_dataset_filepath: This is the *prefix* of the files with all the Best-of-N generations.

        Returns:
            A JudgePreferencesDataset where each row has a chosen and a rejected speech.
        """

        def clean_speech(speech: str) -> str:
            speech = speech.replace(constants.INVALID_QUOTE_TAG, constants.QUOTE_TAG).replace(
                constants.INVALID_UNQUOTE_TAG, constants.UNQUOTE_TAG
            )
            return quote_utils.clean_up_quotes(speech_content=speech)

        def get_actual_judge_score(speeches: list[dict[Any, Any]], name: str) -> Optional[float]:
            for i in range(len(speeches)):
                speech = speeches[len(speeches) - i - 1]
                if speech["speaker"] == constants.DEFAULT_JUDGE_NAME and speech["supplemental"]["probabilistic_decision"]:
                    return speech["supplemental"]["probabilistic_decision"][name]
            return None

        train_data = []
        input_texts = input_utils.read_file_texts(base_path=full_dataset_filepath, input_type=InputType.JSON_TRANSCRIPT)
        for text in input_texts:
            data = json.loads(text)
            for selected in filter(
                lambda x: x["speaker"] in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME],
                data["speeches"],
            ):
                instruction = selected["supplemental"]["prompt"]

                speech_preference_pairs = [(selected["content"], selected["supplemental"]["preference"])] + [
                    (rejected["speech"], rejected["preference"])
                    for rejected in selected["supplemental"]["rejected_responses"]
                ]
                random_selected_speech, random_selected_preference = random.choice(speech_preference_pairs)

                if not selected["supplemental"]["preference"]:
                    random_selected_preference = get_actual_judge_score(data["speeches"], selected["speaker"])

                is_correct = (
                    data["metadata"]["first_debater_correct"] and selected["speaker"] == constants.DEFAULT_DEBATER_A_NAME
                ) or (
                    not data["metadata"]["first_debater_correct"] and selected["speaker"] == constants.DEFAULT_DEBATER_B_NAME
                )
                if is_correct:
                    train_data.append((instruction, clean_speech(random_selected_speech), "", random_selected_preference))
                else:
                    train_data.append((instruction, "", clean_speech(random_selected_speech), random_selected_preference))

        return JudgePreferencesDataset(
            train_data=train_data,
            val_data=[],
            test_data=[],
        )
