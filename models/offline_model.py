from __future__ import annotations

from data import DataRow, RawDataset, SplitType
from models.model import BestOfNConfig, Model, ModelInput, ModelResponse
from utils import InputType, InputUtils
import utils.constants as constants

from abc import ABC, abstractmethod
from enum import auto, Enum
from typing import Optional
import copy
import json
import os
import random
import sys


class OfflineDataFormatParser(ABC):
    @abstractmethod
    def get_first_debater_answer(self, entry) -> str:
        pass

    @abstractmethod
    def get_second_debater_answer(self, entry) -> str:
        pass

    @abstractmethod
    def get_debate_identifier(self, entry) -> str:
        pass

    @abstractmethod
    def get_question(self, entry) -> str:
        pass

    @abstractmethod
    def get_speeches(self, entry) -> list[dict]:
        pass

    @abstractmethod
    def get_speaker_name(self, speech) -> str:
        pass

    @abstractmethod
    def get_text(self, speech) -> str:
        pass

    @abstractmethod
    def validate(self, file) -> bool:
        pass


class ExpandedDataFormatParser(ABC):
    def get_first_debater_answer(self, entry) -> str:
        return entry["metadata"]["first_debater_answer"]

    def get_second_debater_answer(self, entry) -> str:
        return entry["metadata"]["second_debater_answer"]

    def get_debate_identifier(self, entry) -> str:
        return entry["metadata"]["debate_identifier"]

    def get_question(self, entry) -> str:
        return entry["metadata"]["question"]

    def get_speeches(self, entry) -> list[dict]:
        return entry["speeches"]

    def get_speaker_name(self, speech) -> str:
        return speech["speaker"]

    def get_text(self, speech) -> str:
        return speech["content"]

    def validate(self, file) -> bool:
        return "metadata" in file and "speeches" in file


class AbbreviatedDataFormatParser(ABC):
    def get_first_debater_answer(self, entry) -> str:
        return entry["answers"][0]

    def get_second_debater_answer(self, entry) -> str:
        return entry["answers"][1]

    def get_debate_identifier(self, entry) -> str:
        return "_".join([entry["storyTitle"], entry["question"]])

    def get_question(self, entry) -> str:
        return entry["question"]

    def get_speeches(self, entry) -> list[dict]:
        return entry["turns"]

    def get_speaker_name(self, speech) -> str:
        return (
            constants.DEFAULT_DEBATER_A_NAME
            if speech["index"] == 0 and speech["role"] == "Debater"
            else (constants.DEFAULT_DEBATER_B_NAME if speech["role"] == "Debater" else constants.DEFAULT_JUDGE_NAME)
        )

    def get_text(self, speech) -> str:
        return speech["text"]

    def validate(self, file) -> bool:
        return "answers" in file and "question" in file and "turns" in file


class OfflineDataFormat(Enum):
    EXPANDED = ExpandedDataFormatParser()
    ABBREVIATED = AbbreviatedDataFormatParser()

    def get_parser(self):
        return self.value


class OfflineModel(Model):
    def __init__(
        self,
        alias: str,
        speeches: list[str] | list[list[str]],
        **kwargs,
    ):
        """
        An offline model returns the text that was previously generated during an earlier run. This is useful if you
        want to re-judge a round.

        Args:
            alias: String that identifies the model for metrics and deduplication
            speeches: a list of strings corresponding to the speeches the model will output
        """
        super().__init__(alias=alias, is_debater=True)
        self.speech_idx = 0
        self.speeches = speeches if speeches and isinstance(speeches[0], list) else ([speeches] if speeches else [])

    def predict(self, inputs: list[list[ModelInput] | str], **kwargs) -> ModelResponse:
        """Generates a list of texts in response to the given input."""
        speech = [s[self.speech_idx] for s in self.speeches]
        self.speech_idx += 1
        return [
            ModelResponse(speech=speech[i], prompt="\n".join(model_input.content for model_input in inputs[i]))
            for i in range(len(speech))
        ]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> OfflineModel:
        """Generates a deepcopy of this model"""
        return OfflineModel(alias=alias, speeches=self.speeches)

    def can_merge(self, other: Model) -> bool:
        """Determines whether this model can be 'merged' (aka the model associated with one element of the
        batch can be combined with the model associated with another element so that one can do batch
        processing."""
        return isinstance(other, OfflineModel)

    def merge(self, other: Model) -> Model:
        if self.can_merge(other):
            self.speeches.extend(other.speeches)
            return self
        raise Exception("Cannot merge across models")


class BestOfNOfflineModel(Model):
    def __init__(
        self,
        alias: str,
        speeches: list[list[str]],
        opponent_speeches: list[list[str]],
        **kwargs,
    ):
        """
        An offline model returns the text that was previously generated during an earlier Best of N run. This is useful if you
        want to re-judge a round.

        NOTE: ONLY USE THIS MODEL IF YOU WANT TO RECOMPUTE THE BEST_OF_N SCORES. IF YOU WANT TO USE THE SAME
        BEST_OF_N SCORES AS DURING THE INITIAL RUN, THEN A NORMAL OFFLINE MODEL WILL SUFFICE

        Args:
            alias: String that identifies the model for metrics and deduplication
            speeches: a list of strings corresponding to the speeches the model will output.
            opponent_speeches: a list of strings corresponding to the speeches that are forecast for the opponent.
                The dimensions for both this and speeches is: speeches_per_round x N/M.
        """
        super().__init__(alias=alias, is_debater=True)
        self.speech_idx = 0
        self.speeches = speeches
        self.opponent_speeches = opponent_speeches
        self.use_opponent_speeches = False

    def predict(self, inputs: list[list[ModelInput] | str], **kwargs) -> ModelResponse:
        """Generates a list of texts in response to the given input."""
        sample_set = self.speeches if not self.use_opponent_speeches else self.opponent_speeches
        speech = [s for s in sample_set[self.speech_idx]]
        if self.use_opponent_speeches:
            self.speech_idx += 1
        self.use_opponent_speeches = not self.use_opponent_speeches
        return [
            ModelResponse(speech=speech[i], prompt="\n".join(model_input.content for model_input in inputs[i]))
            for i in range(len(speech))
        ]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> OfflineModel:
        """Generates a deepcopy of this model"""
        return OfflineModel(alias=alias, speeches=self.speeches)

    def can_merge(self, other: Model) -> bool:
        """Determines whether this model can be 'merged' (aka the model associated with one element of the
        batch can be combined with the model associated with another element so that one can do batch
        processing."""
        return isinstance(other, BestOfNOfflineModel)

    def merge(self, other: Model) -> Model:
        if self.can_merge(other):
            self.speeches.extend(other.speeches)
            return self
        raise Exception("Cannot merge across models")


class OfflineModelHelper:
    def __init__(self, file_path_prefix: str, dataset: RawDataset, split_type: SplitType):
        """
        This class is used to generate the data and models for offline processing.

        Args:
            file_path_prefix: Either the full path of the transcript jsons (not including the numbers at the end) or just
                the timestamp of the files.
            dataset: The dataset that was used to generate the original prompts
            split_type: Which split of the dataset will be looked at
        """
        self.file_path_prefix = file_path_prefix

        self.data = []

        json_texts = InputUtils.read_file_texts(base_path=file_path_prefix, input_type=InputType.JSON_TRANSCRIPT)
        if json_texts:
            self.data.extend([json.loads(text) for text in json_texts])

        jsonl_texts = InputUtils.read_file_texts(base_path=file_path_prefix, input_type=InputType.JSON_LIST)
        if jsonl_texts:
            self.data.extend([json.loads(line) for line in filter(lambda x: x, jsonl_texts[0].split("\n"))])

        self.dataset = dataset
        self.split_type = split_type

        self.data_format = (
            OfflineDataFormat.EXPANDED
            if OfflineDataFormat.EXPANDED.get_parser().validate(self.data[0])
            else OfflineDataFormat.ABBREVIATED
        )
        self.parser = self.data_format.get_parser()

        self.prune_data()

    @classmethod
    def reduce_to_common_rounds(cls, helper_one: OfflineModelHelper, helper_two: OfflineModelHelper) -> None:
        """
        This takes in two offline model helpers and trims their internal data objects to only contain topics that both
        offline model helpers have rounds for (and so that it contains the topics in the same order.
        This is useful in the case where you have two models that are using two different sets of transcripts and you
        only want to select topics where both models have an example speech from.
        """

        def get_trimmed_data(
            main: list[dict],
            other: list[dict],
            order_by_main: bool,
            main_parser: OfflineDataFormatParser,
            other_parser: OfflineDataFormatParser,
        ) -> list[dict]:
            new_data = []
            opposite = other if order_by_main else main
            primary_parser = main_parser if order_by_main else other_parser
            opposite_parser = other_parser if order_by_main else main_parser
            for title, entry in main.items() if order_by_main else other.items():
                primary_answer = primary_parser.get_first_debater_answer(entry)
                opposite_answer = opposite_parser.get_first_debater_answer(opposite[title]) if title in opposite else None
                if title in opposite and primary_answer == opposite_answer:
                    new_data.append(entry if order_by_main else opposite[title])
            return new_data

        if helper_one == helper_two:
            return

        title_to_entry_one = {helper_one.parser.get_debate_identifier(entry): entry for entry in helper_one.data}
        title_to_entry_two = {helper_two.parser.get_debate_identifier(entry): entry for entry in helper_two.data}

        helper_one.data = get_trimmed_data(
            title_to_entry_one,
            title_to_entry_two,
            order_by_main=True,
            main_parser=helper_one.parser,
            other_parser=helper_two.parser,
        )
        helper_two.data = get_trimmed_data(
            title_to_entry_two,
            title_to_entry_one,
            order_by_main=False,
            main_parser=helper_two.parser,
            other_parser=helper_one.parser,
        )

    def prune_data(self) -> None:
        """Removes data rows that could not be found in the reference dataset"""

        def validate_idx(idx: int) -> bool:
            debate_identifier = self.parser.get_debate_identifier(self.data[idx % len(self.data)])
            question = self.parser.get_question(self.data[idx % len(self.data)])
            story_title = debate_identifier.replace("_" + question, "")
            for row in self.dataset.get_data(split=self.split_type):
                if (
                    row.story_title.strip() == story_title.strip()
                    and row.question.strip() == question.strip()
                    and (
                        row.positions[0] == self.parser.get_first_debater_answer(self.data[idx % len(self.data)])
                        or row.positions[0] == self.parser.get_second_debater_answer(self.data[idx % len(self.data)])
                    )
                    and (
                        row.positions[1] == self.parser.get_first_debater_answer(self.data[idx % len(self.data)])
                        or row.positions[1] == self.parser.get_second_debater_answer(self.data[idx % len(self.data)])
                    )
                ):
                    return True
            return False

        idxs_to_keep = [validate_idx(idx) for idx in range(len(self.data))]
        self.data = [row for idx, row in filter(lambda x: idxs_to_keep[x[0]], enumerate(self.data))]
        if not self.data:
            raise Exception("No eligible data was found")

    def get_example(self, idx: int, split_type: SplitType = SplitType.TRAIN) -> DataRow:
        """
        Gets the row of the dataset that was used to generate the original round.

        Args:
            idx: The index of the round to be replayed (not the index in the dataset)
            split_type: The split of the dataset that the original round was from

        Returns:
            The corresponding row in the dataset that was used to generate the original round.
        """

        debate_identifier = self.parser.get_debate_identifier(self.data[idx % len(self.data)])
        question = self.parser.get_question(self.data[idx % len(self.data)])
        story_title = debate_identifier.replace("_" + question, "")
        for row in self.dataset.get_data(split=split_type):
            if row.story_title.strip() == story_title.strip() and row.question.strip() == question.strip():
                if row.positions[0] != self.parser.get_first_debater_answer(self.data[idx % len(self.data)]):
                    correct_answer = row.positions[row.correct_index]
                    row.positions = (row.positions[1], row.positions[0])
                    row.correct_index = 0 if correct_answer == row.positions[0] else 1
                return row

        raise Exception(f"A row with title {story_title} and question {question} could not be found in the dataset")

    def get_size(self):
        return len(self.data)

    def create_offline_model(
        self,
        alias: str,
        debater_name: str,
        idx: int,
        positions: tuple[str, str],
        best_of_n_config: Optional[BestOfNConfig] = None,
    ) -> OfflineModel:
        """
        Generates an OfflineModel

        Args:
            alias: The alias of the model
            debater_name: The name of the debater (Debater_A, Debater_B) that will use the model since
                OfflineModels are single use
            idx: The index of the original round that was used (not the dataset index)
            positions: tuple of (debater_a_position, debater_b_position)
            best_of_n_config: an optional config if one wants to resample from the multiple generated speeches.
                For example, let's say you generated 16 speeches as part of a best-of-n generation but now want to check
                how the model would've performed if you only sampled 4 speeches, you would pass in a best_of_n config
                that specifies that 4 speeches are to be sampled. Please be careful to (a) avoid passing in an
                n that is greater than the original n and (b) to avoid doing this for multi-stage debate rounds
                [since debaters may be responding to a speech that was not selected]. This is only supported
                for the expanded data format.

        Returns:
            An offline model that can be used once
        """

        if self.data_format != OfflineDataFormat.EXPANDED and best_of_n_config:
            raise Exception("Best of N is only supported with the expanded offline data format")

        entry = self.data[idx % len(self.data)]
        all_speeches = self.parser.get_speeches(entry)
        debater_a_position = self.parser.get_first_debater_answer(entry)
        debater_b_position = self.parser.get_second_debater_answer(entry)

        all_speakers = set([self.parser.get_speaker_name(speech) for speech in all_speeches])  # needed for consultancy

        is_flipped = debater_a_position == positions[1] and debater_b_position == positions[0]
        debater_name_to_use = (
            debater_name
            if not is_flipped or constants.DEFAULT_DEBATER_A_NAME not in all_speakers
            else (
                constants.DEFAULT_DEBATER_A_NAME
                if debater_name == constants.DEFAULT_DEBATER_B_NAME
                else constants.DEFAULT_DEBATER_B_NAME
            )
        )

        relevant_speeches = [
            speech for speech in filter(lambda x: self.parser.get_speaker_name(x) == debater_name_to_use, all_speeches)
        ]

        selected_speeches = []
        contender_speeches = []
        opponent_speeches = []
        if best_of_n_config:
            for speech in relevant_speeches:
                supplemental = speech["supplemental"]
                contenders = [(supplemental["speech"], supplemental["preference"])]
                for rejected_speech in supplemental["rejected_responses"]:
                    contenders.append((rejected_speech["speech"], rejected_speech["preference"]))
                options = random.sample(contenders, k=best_of_n_config.n)
                best_option = sorted(options, key=lambda x: float(x[1]), reverse=True)[0][0]
                if best_option:
                    selected_speeches.append(best_option)
                contender_speeches.append([c[0] for c in contenders])
                if "bon_opposing_model_responses" in supplemental:
                    opponent_speeches.append(
                        [
                            resp["speech"]
                            for resp in filter(lambda x: x is not None, supplemental["bon_opposing_model_responses"])
                        ]
                    )
        else:
            selected_speeches = [self.parser.get_text(speech) for speech in relevant_speeches]

        if best_of_n_config and best_of_n_config.recompute:
            return BestOfNOfflineModel(alias=alias, speeches=contender_speeches, opponent_speeches=opponent_speeches)

        return OfflineModel(
            alias=alias,
            speeches=selected_speeches,
        )
