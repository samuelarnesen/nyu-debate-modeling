from __future__ import annotations

from agents.models.model import BestOfNConfig, Model, ModelInput, ModelResponse
from data import DataRow, RawDataset, SplitType
from utils import InputType, InputUtils
import utils.constants as constants

from typing import Optional
import copy
import json
import os
import random


class OfflineModel(Model):
    def __init__(self, alias: str, speeches: list[str] | list[list[str]], **kwargs):
        """
        An offline model returns the text that was previously generated during an earlier run. This is useful if you
        want to re-judge a round.

        Args:
            alias: String that identifies the model for metrics and deduplication
            speeches: a list of strings corresponding to the speeches the model will output
        """
        super().__init__(alias=alias, is_debater=True)
        self.speech_idx = 0
        self.speeches = speeches if speeches and isinstance(speeches[0], list) else [speeches]

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


class OfflineModelHelper:
    def __init__(self, file_path_prefix: str, dataset: RawDataset):
        """
        This class is used to generate the data and models for offline processing.

        Args:
            file_path_prefix: Either the full path of the transcript jsons (not including the numbers at the end) or just
                the timestamp of the files.
            dataset: The dataset that was used to generate the original prompts
        """
        self.file_path_prefix = file_path_prefix
        self.data = [
            json.loads(text)
            for text in InputUtils.read_file_texts(base_path=file_path_prefix, input_type=InputType.JSON_TRANSCRIPT)
        ]
        self.dataset = dataset

    @classmethod
    def reduce_to_common_rounds(cls, helper_one: OfflineModelHelper, helper_two: OfflineModelHelper) -> None:
        """
        This takes in two offline model helpers and trims their internal data objects to only contain topics that both
        offline model helpers have rounds for (and so that it contains the topics in the same order.
        This is useful in the case where you have two models that are using two different sets of transcripts and you
        only want to select topics where both models have an example speech from.
        """

        def get_trimmed_data(main: list[dict], other: list[dict], order_by_main: bool) -> list[dict]:
            new_data = []
            opposite = other if order_by_main else main
            for title, entry in main.items() if order_by_main else other.items():
                get_position = lambda x: x["metadata"]["first_debater_answer"]
                if title in opposite and get_position(entry) == get_position(opposite[title]):
                    new_data.append(entry)
            return new_data

        title_to_entry_one = {entry["metadata"]["debate_identifier"]: entry for entry in helper_one.data}
        title_to_entry_two = {entry["metadata"]["debate_identifier"]: entry for entry in helper_two.data}

        helper_one.data = get_trimmed_data(title_to_entry_one, title_to_entry_two, order_by_main=True)
        helper_two.data = get_trimmed_data(title_to_entry_two, title_to_entry_one, order_by_main=False)

    def get_example(self, idx: int, split_type: SplitType = SplitType.TRAIN) -> DataRow:
        """
        Gets the row of the dataset that was used to generate the original round.

        Args:
            idx: The index of the round to be replayed (not the index in the dataset)
            split_type: The split of the dataset that the original round was from

        Returns:
            The corresponding row in the dataset that was used to generate the original round.
        """
        debate_identifier = self.data[idx % len(self.data)]["metadata"]["debate_identifier"]
        question = self.data[idx % len(self.data)]["metadata"]["question"]
        story_title = debate_identifier.replace("_" + question, "")
        for row in self.dataset.get_data(split=split_type):
            if row.story_title == story_title and row.question == question:
                if row.positions[0] != self.data[idx % len(self.data)]["metadata"]["first_debater_answer"]:
                    if row.speeches:
                        raise Exception("The speech orders are incompatible")
                    row.positions = (row.positions[1], row.positions[0])
                    row.correct_index = 1 if row.correct_index == 0 else 1
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
        identifier: str = "",
    ) -> OfflineModel:
        """
        Generates an OfflineModel

        Args:
            alias: The alias of the model
            debater_name: The name of the debater (Debater_A, Debater_B) that will use the model since
                OfflineModels are single use
            idx: The index of the original round that was used (not the datsset index)
            positions: tuple of (debater_a_position, debater_b_position)
            best_of_n_config: an optional config if one wants to resample from the multiple generated speeches.
                For example, let's say you generated 16 speeches as part of a best-of-n generation but now want to check
                how the model would've performed if you only sampled 4 speeches, you would pass in a best_of_n config
                that specifies that 4 speeches are to be sampled. Please be careful to (a) avoid passing in an
                n that is greater than the original n and (b) to avoid doing this for multi-stage debate rounds
                [since debaters may be responding to a speech that was not selected]

        Returns:
            An offline model that can be used once
        """
        entry = self.data[idx % len(self.data)]
        all_speeches = entry["speeches"]
        debater_a_position = entry["metadata"]["first_debater_answer"]
        debater_b_position = entry["metadata"]["second_debater_answer"]

        is_flipped = debater_a_position == positions[1] and debater_b_position == positions[0]
        debater_name_to_use = (
            debater_name
            if not is_flipped
            else (
                constants.DEFAULT_DEBATER_A_NAME
                if debater_name == constants.DEFAULT_DEBATER_B_NAME
                else constants.DEFAULT_DEBATER_B_NAME
            )
        )

        relevant_speeches = [speech for speech in filter(lambda x: x["speaker"] == debater_name_to_use, all_speeches)]
        selected_speeches = []
        if best_of_n_config:
            for speech in relevant_speeches:
                supplemental = speech["supplemental"]
                contenders = [(supplemental["speech"], supplemental["preference"])]
                for rejected_speech in supplemental["rejected_responses"]:
                    contenders.append((rejected_speech["speech"], rejected_speech["preference"]))
                options = random.choices(contenders, k=best_of_n_config.n)
                best_option = sorted(options, key=lambda x: float(x[1]), reverse=True)[0][0]
                selected_speeches.append(best_option)
        else:
            selected_speeches = [speech["content"] for speech in relevant_speeches]

        return OfflineModel(
            alias=alias,
            speeches=selected_speeches,
        )
