from __future__ import annotations

from agents.models.model import Model, ModelInput, ModelResponse
from data import DataRow, RawDataset, SplitType
from utils import InputUtils
import utils.constants as constants

from typing import Optional
import json
import os


class OfflineModel(Model):
    def __init__(self, alias: str, speeches: list[str], **kwargs):
        """
        An offline model returns the text that was previously generated during an earlier run. This is useful if you
        want to re-judge a round.

        Args:
            alias: String that identifies the model for metrics and deduplication
            speeches: a list of strings corresponding to the speeches the model will output
        """
        super().__init__(alias=alias, is_debater=True)
        self.speech_idx = 0
        self.speeches = speeches

    def predict(self, inputs: list[list[ModelInput] | str], **kwargs) -> ModelResponse:
        """Generates a list of texts in response to the given input."""

        speech = self.speeches[self.speech_idx]
        self.speech_idx += 1
        return [ModelResponse(speech=speech, prompt="\n".join(model_input.content for model_input in inputs[0]))]

    def copy(self, alias: str, is_debater: Optional[bool] = None, **kwargs) -> OfflineModel:
        """Generates a deepcopy of this model"""
        return OfflineModel(alias=alias, speeches=self.speeches)


class OfflineModelHelper:
    def __init__(self, file_path_prefix: str, dataset: RawDataset):
        """
        This class is used to generate the data and models for offline processing.

        Args:
            file_path_prefix: Either the full path of the transcript jsons (not including the numbers at the end) or just
                the timestamp of the files
            dataset: The dataset that was used to generate the original prompts
        """
        self.data = [json.loads(text) for text in InputUtils.read_file_texts(base_path=file_path_prefix, extension="json")]
        self.dataset = dataset

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
                return row
        raise Exception(f"A row with title {story_title} and question {question} could not be found in the dataset")

    def create_offline_model(self, alias: str, debater_name: str, idx: int) -> OfflineModel:
        """
        Generates an OfflineModel

        Args:
            alias: The alias of the model
            debater_name: The name of the debater (Debater_A, Debater_B) that will use the model since
                OfflineModels are single use
            idx: The index of the original round that was used (not the datsset index)

        Returns:
            An offline model that can be used once
        """
        return OfflineModel(
            alias=alias,
            speeches=[
                speech["content"]
                for speech in filter(lambda x: x["speaker"] == debater_name, self.data[idx % len(self.data)]["speeches"])
            ],
        )
