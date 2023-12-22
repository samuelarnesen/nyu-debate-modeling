from data.dataset import (
    AnnotationBracket,
    AnnotationData,
    AnnotationTag,
    DataRow,
    DatasetType,
    RawDataLoader,
    RawDataset,
    SpeakerType,
    SpeechData,
    SplitType,
)
from data.quality_debates_loader import QualityDebatesDataset, QualityDebatesLoader
import utils.constants as constants

from pydantic import BaseModel

from difflib import SequenceMatcher
from enum import Enum
from typing import Optional, Union
import copy
import pickle
import os
import re
import sys


class Annotation(BaseModel):
    text: str
    clean: str
    metrics: dict[str | AnnotationTag, float]


class AnnotatedQualityDebatesDataset(RawDataset):
    def __init__(self, dataset: QualityDebatesDataset, annotations_file_path: str):
        """
        Dataset where the transcripts of the human debates are annotated with stylistic tags (e.g. statement, rebuttal)

        Params:
            dataset: a normal QualityDebateDataset
            annotations_file_path: path to the file with all the annotations. We will match the speeches in this
                file to the speeches in the normal QualityDebateDataset to attach the annotations
        """

        super().__init__(DatasetType.ANNOTATED_QUALITY_DEBATES)
        self.data = {
            SplitType.TRAIN: dataset.get_data(SplitType.TRAIN),
            SplitType.VAL: dataset.get_data(SplitType.VAL),
            SplitType.TEST: dataset.get_data(SplitType.TEST),
        }
        self.__add_annotation(annotations_file_path=annotations_file_path)

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[DataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[DataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    @classmethod
    def meets_threshold(
        cls, tag: AnnotationTag, bracket: AnnotationBracket, threshold: float, positive: bool, speech: SpeechData
    ):
        """
        Checks whether a given speech meets all the required filters.

        Params:
            tag: Which annotation tag to filter for (e.g. refutuation)
            bracket: When combined with the `threshold' and 'tag' params, this determines what qualifies as an eligible example.
                For example, a tag of 'refutation', a threshold of 0.9, and a bracket of 'high' means that you want samples
                that are in at least the 90th percentile of having the most refutation. By contrast, a bracket of 'low' in that
                situation would mean one would want to be in the 90th percentile of having the least refutation.
            threshold: When combined with the `threshold' and 'bracket' params, this determines what qualifies as an eligible example.
            positive: If one wants to filter for rows that explicitly meet the other criteria (true) or explicitly
                meet the opposite of the criteria (false)
            speech: The speech that is being evaluated for those criteria.
        """
        if not speech.annotation or not speech.annotation.percentiles or tag not in speech.annotation.percentiles:
            return False

        if positive:
            if bracket == AnnotationBracket.HIGH:
                return speech.annotation.percentiles[tag] > threshold
            elif bracket == AnnotationBracket.LOW:
                return speech.annotation.percentiles[tag] < (1 - threshold)
            else:
                return speech.annotation.percentiles[tag] <= threshold and speech.annotation.percentiles[tag] >= (
                    1 - threshold
                )
        else:
            if bracket == AnnotationBracket.HIGH:
                return speech.annotation.percentiles[tag] < (1 - threshold)
            elif bracket == AnnotationBracket.LOW:
                return speech.annotation.percentiles[tag] > threshold
            else:
                return speech.annotation.percentiles[tag] <= threshold and speech.annotation.percentiles[tag] >= (
                    1 - threshold
                )

    def get_annotation_examples(
        self,
        tag: AnnotationTag,
        bracket: AnnotationBracket,
        threshold: float,
        positive: bool,
        source_row: Optional[DataRow] = None,
    ) -> list[SpeechData]:
        """
        Filters the dataset to provide some few-shot examples of the same tag. For instance, if one wants to instruct a model
        to 'use a style that has a lot of refutation', one could use this method to fetch some examples of other
        speeches from the dataset that also have a lot of refutation.

        Params:
            tag: Which annotation tag to filter for (e.g. refutuation)
            bracket: When combined with the `threshold' and 'tag' params, this determines what qualifies as an eligible example.
                For example, a tag of 'refutation', a threshold of 0.9, and a bracket of 'high' means that you want samples
                that are in at least the 90th percentile of having the most refutation. By contrast, a bracket of 'low' in that
                situation would mean one would want to be in the 90th percentile of having the least refutation.
            threshold: When combined with the `threshold' and 'bracket' params, this determines what qualifies as an eligible example.
            positive: If one wants to filter for rows that explicitly meet the other criteria (true) or explicitly
                meet the opposite of the criteria (false)
            source_row: An optional row to exclude in case one is using this for supervised finetuning. If one
                is trying to induce a model to generate a particular speech using few-shot examples, one shouldn't pass
                in the target speech as one of the examples.

        Returns:
            A list of speeches that meet the specified criteria.
        """
        eligible_examples = []
        for row in filter(lambda x: not source_row or source_row.story_title != x.story_title, self.data[SplitType.TRAIN]):
            for speech in filter(
                lambda x: AnnotatedQualityDebatesDataset.meets_threshold(tag, bracket, threshold, positive, x), row.speeches
            ):
                eligible_examples.append(speech)
        return eligible_examples

    def __add_annotation(self, annotations_file_path: str) -> None:
        def match_speeches(speech: SpeechData, annotations: list[Annotation]):
            cleaned_speech = re.sub("\s+", " ", speech.text)
            for annotation in annotations:
                cleaned_annotation = re.sub("\s+", " ", annotation.clean).lstrip().rstrip()
                ratio = SequenceMatcher(None, cleaned_annotation, cleaned_speech).ratio()
                if cleaned_annotation == cleaned_speech or ratio > 0.99:
                    return annotation
            return None

        with open(annotations_file_path, "rb") as f:
            id_to_speeches = pickle.load(f)

        annotated_speeches = []
        for split in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST]:
            for i, row in enumerate(self.data[split]):
                annotations = [Annotation(**entry) for entry in id_to_speeches[row.debate_id]]
                for annotation in annotations:
                    annotation.metrics = {AnnotationTag[key.upper()]: value for key, value in annotation.metrics.items()}
                for speech in row.speeches:
                    matching = match_speeches(speech, annotations)
                    speech.annotation = AnnotationData(percents={}, percentiles={})
                    if matching:
                        speech.annotation = AnnotationData(percents=copy.deepcopy(matching.metrics), percentiles={})
                        annotated_speeches.append(speech)

        for tag in AnnotationTag:
            distribution = sorted([speech.annotation.percents[tag] for speech in annotated_speeches])
            for idx, speech in enumerate(annotated_speeches):
                for i in range(len(distribution)):
                    if speech.annotation.percents[tag] <= distribution[i]:
                        speech.annotation.percentiles[tag] = i / len(distribution)
                        break
                if speech.annotation.percents[tag] > distribution[-1]:
                    speech.annotation.percentiles[tag] = 1


class AnnotatedQualityDebatesLoader(RawDataLoader):
    DEFAULT_ANNOTATIONS_FILE_PATH = (
        os.environ[constants.SRC_ROOT] + "data/datasets/annotated-quality-debates/annotated-data-set.p"
    )

    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        deduplicate: bool = False,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AnnotatedQualityDebatesDataset:
        """Constructs an AnnotatedQualityDebatesDataset"""
        annotations_file_path = (
            supplemental_file_paths.get("annotations_file_path", AnnotatedQualityDebatesLoader.DEFAULT_ANNOTATIONS_FILE_PATH)
            if supplemental_file_paths
            else AnnotatedQualityDebatesLoader.DEFAULT_ANNOTATIONS_FILE_PATH
        )

        quality_debates_dataset = QualityDebatesLoader.load(
            full_dataset_filepath=full_dataset_filepath, deduplicate=deduplicate
        )
        return AnnotatedQualityDebatesDataset(dataset=quality_debates_dataset, annotations_file_path=annotations_file_path)
