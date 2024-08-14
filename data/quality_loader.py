from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SplitType
from data.quality_debates_loader import (
    QualityDebatesDataset,
    QualityConsultancyLoader,
    QualityDebatesLoader,
    QualityModelBasedDebateLoader,
    QualityTranscriptsLoader,
)
import utils.constants as constants

from typing import Any, Optional
import json
import itertools
import os
import random
import re
import statistics


class QualityDataset(RawDataset):
    def __init__(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        test_data: list[dict[str, Any]],
        override_type: Optional[DatasetType] = None,
        allow_multiple_positions_per_question: bool = False,
        dedupe_dataset: Optional[list[tuple[QualityDebatesDataset, bool]]] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
    ):
        """
        Dataset where each row contains a question and positions from the Quality dataset

        Params:
            train_data: list of training data loaded from quality jsonl file
            val_data: list of validation data loaded from quality jsonl file
            test_data: list of testing data loaded from quality jsonl file
            override_type: if this is being used as a parent class, this is the dataset type of the child
            allow_multiple_positions_per_question: many quality questions have more than two answers. By default,
                we will select the best distractor If this parameter is set to true, we will create
                a separate row for every single combination of positions
            dedupe_dataset: The dataset to dedupe from. This is used because the NYU Human Debate experiments used
                questions from the Quality dataset so if one trained on that data, then one needs to remove those
                rows from the validation set. Each entry is a dataset and a boolean indicating if one should dedupe
                all questions that share the same story (True) or not (False).
            flip_sides: Whether the ordering of the positions should be flipped (aka two rounds per question)
            shuffle_deterministically: Whether to use a fixed random seed for shuffling the dataset
        """
        super().__init__(override_type or DatasetType.QUALITY)
        if shuffle_deterministically:
            random.seed(a=123456789)
        self.allow_multiple_positions_per_question = allow_multiple_positions_per_question
        self.flip_sides = flip_sides
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__dedupe_rows(self.__convert_batch_to_rows(val_data), dedupe_dataset),
            SplitType.TEST: self.__dedupe_rows(self.__convert_batch_to_rows(test_data), dedupe_dataset),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}
        if not self.data[SplitType.TEST]:  # Adding b/c Quality Test Set does not have gold labels
            self.__split_validation_and_test_sets()

        self.data[SplitType.TRAIN] = self.__reorder(self.data[SplitType.TRAIN])
        self.data[SplitType.VAL] = self.__reorder(self.data[SplitType.VAL])
        self.data[SplitType.TEST] = self.__reorder(self.data[SplitType.TEST])
        self.shuffle_deterministically = shuffle_deterministically

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
        return [x for x in data_to_return]

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, batch: list[dict[str, Any]]):
        rows = []
        for entry in batch:
            for i in range(len(entry["questions"])):
                rows_to_add = self.__example_to_row(entry, i)
                if rows_to_add:
                    rows.extend(rows_to_add)
        return rows

    def __example_to_row(self, entry: dict[str, Any], question_idx: int) -> list[DataRow]:
        def fix_line_spacing(text):
            """
            Some stories in QuALITY are in a strange format where each line has a maximum number of words (73),
            after which there is a newline. This inconsistency in formats makes exact quoting a little trickier
            so we simplify things by stripping out the excess newlines.
            """
            pattern = r"\n+"
            max_line_length = max([len(line) for line in text.split("\n")])
            if max_line_length < 100:
                text = re.sub(pattern, lambda match: " " if len(match.group(0)) == 1 else match.group(0), text)
            return text

        question = entry["questions"][question_idx]
        if "gold_label" not in question or "difficult" not in question or question["difficult"] == 0:
            return None

        correct_answer = int(question["gold_label"]) - 1

        if self.allow_multiple_positions_per_question:
            incorrect_answers = [i for i in filter(lambda x: x != correct_answer, range(len(question["options"])))]
            possible_position_pairs = [
                (correct_answer, incorrect_answer, True) for incorrect_answer in incorrect_answers
            ] + [(incorrect_answer, correct_answer, False) for incorrect_answer in incorrect_answers]
        else:
            best_wrong_guesses = [
                val["untimed_best_distractor"]
                if (val["untimed_answer"] == question["gold_label"])
                else val["untimed_answer"]
                for val in question["validation"]
            ]
            incorrect_answer = statistics.mode(best_wrong_guesses) - 1
            possible_position_pairs = [(correct_answer, incorrect_answer, True), (incorrect_answer, correct_answer, False)]
            random.shuffle(possible_position_pairs)

        rows = []
        for first, second, first_correct in possible_position_pairs:
            rows.append(
                DataRow(
                    background_text=fix_line_spacing(entry["article"]),
                    question=question["question"],
                    correct_index=0 if first_correct else 1,
                    positions=(
                        question["options"][first],
                        question["options"][second],
                    ),
                    story_title=entry["title"],
                    debate_id="_".join([entry["title"], question["question"]]),
                )
            )
            if not self.flip_sides:
                break
        return rows

    def __split_validation_and_test_sets(self):
        second_half = self.data[SplitType.VAL][int(len(self.data[SplitType.VAL]) / 2) :]
        self.data[SplitType.VAL] = self.data[SplitType.VAL][0 : int(len(self.data[SplitType.VAL]) / 2)]
        val_stories = set([row.story_title for row in self.data[SplitType.VAL]])

        test_data = []
        added_count = 0
        for row in second_half:
            if row.story_title not in val_stories:
                test_data.append(row)
            else:
                self.data[SplitType.VAL].append(row)
        self.data[SplitType.TEST] = test_data

    def __dedupe_rows(self, rows: list[DataRow], dedupe_dataset: Optional[list[QualityDebatesDataset]] = None) -> None:
        if not dedupe_dataset:
            return rows

        used_stories = []
        used_debate_identifiers = []
        for ds, dedupe_stories in dedupe_dataset:
            for other_split in SplitType:
                if dedupe_stories:
                    used_stories += [row.story_title for row in ds.get_data(split=other_split)]
                used_debate_identifiers += [row.debate_id for row in ds.get_data(split=other_split)]

        return [
            row
            for row in filter(
                lambda x: x.story_title not in used_stories and x.debate_id not in used_debate_identifiers, rows
            )
        ]

    def __reorder(self, rows: list[DataRow]) -> list[DataRow]:
        if len(rows) == 0:
            return rows

        random.shuffle(rows)
        story_to_rows = {}
        for row in rows:
            if row.story_title not in story_to_rows:
                story_to_rows[row.story_title] = []
            story_to_rows[row.story_title].append(row)

        final_order = []
        max_index = max([len(story_to_rows[row.story_title]) for row in rows])
        for index in range(max_index):
            for story in filter(lambda x: len(story_to_rows[x]) > index, story_to_rows):
                final_order.append(story_to_rows[story][index])
        return final_order


class QualityLoader(RawDataLoader):
    DEFAULT_TRAIN_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/quality/QuALITY.v1.0.1.htmlstripped.train"
    DEFAULT_VAL_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/quality/QuALITY.v1.0.1.htmlstripped.dev"
    DEFAULT_TEST_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/quality/QuALITY.v1.0.1.htmlstripped.test"

    @classmethod
    def get_splits(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> tuple[list[dict]]:
        """Splits the data in train, val, and test sets"""

        def __load_individual_file(filepath: str) -> list[str, Any]:
            entries = []
            if filepath:
                with open(filepath) as f:
                    for line in f.readlines():
                        entries.append(json.loads(line))
            return entries

        train_filepath = train_filepath or QualityLoader.DEFAULT_TRAIN_PATH
        val_filepath = val_filepath or QualityLoader.DEFAULT_VAL_PATH
        test_filepath = test_filepath or QualityLoader.DEFAULT_TEST_PATH

        train_split = __load_individual_file(train_filepath)
        val_split = __load_individual_file(val_filepath)
        test_split = __load_individual_file(test_filepath)
        return train_split, val_split, test_split

    @classmethod
    def load(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        allow_multiple_positions_per_question: bool = False,
        deduplicate_with_quality_debates: bool = True,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
        **kwargs,
    ) -> QualityDataset:
        """Constructs a QualityDataset"""
        train_split, val_split, test_split = QualityLoader.get_splits(
            train_filepath=train_filepath, val_filepath=val_filepath, test_filepath=test_filepath
        )

        dedupe_datasets = None
        if deduplicate_with_quality_debates:
            quality_debates_filepath = (supplemental_file_paths or {}).get(
                "quality_debates_file_path", QualityTranscriptsLoader.DEFAULT_FILE_PATH
            )
            quality_debates_dataset = QualityDebatesLoader.load(
                full_dataset_filepath=quality_debates_filepath, deduplicate=True
            )
            quality_consultancy_dataset = QualityConsultancyLoader.load(
                full_dataset_filepath=quality_debates_filepath, deduplicate=True
            )
            dedupe_datasets = [(quality_debates_dataset, True), (quality_consultancy_dataset, True)]

        if supplemental_file_paths and "previous_runs" in supplemental_file_paths:
            dedupe_datasets = [] if not dedupe_datasets else dedupe_datasets
            previous_runs = (
                supplemental_file_paths["previous_runs"]
                if isinstance(supplemental_file_paths["previous_runs"], list)
                else [supplemental_file_paths["previous_runs"]]
            )
            for fp in previous_runs:
                dedupe_datasets.append((QualityModelBasedDebateLoader.load(full_dataset_filepath=fp), False))
        return QualityDataset(
            train_data=train_split,
            val_data=val_split,
            test_data=test_split,
            allow_multiple_positions_per_question=allow_multiple_positions_per_question,
            dedupe_dataset=dedupe_datasets,
            flip_sides=flip_sides,
            shuffle_deterministically=shuffle_deterministically,
        )
