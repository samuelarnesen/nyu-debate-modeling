from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SplitType
import utils.constants as constants

from typing import Any, Optional
import json
import itertools
import os
import random
import statistics


class QualityDataset(RawDataset):
    def __init__(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        test_data: list[dict[str, Any]],
        override_type: Optional[DatasetType] = None,
        allow_multiple_positions_per_question: bool = False,
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
        """
        super().__init__(override_type or DatasetType.QUALITY)
        self.allow_multiple_positions_per_question = allow_multiple_positions_per_question
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

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
        random.shuffle(rows)
        return rows

    def __example_to_row(self, entry: dict[str, Any], question_idx: int) -> tuple[str, Any]:
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
                    background_text=entry["article"],
                    question=question["question"],
                    correct_index=0 if first_correct else 1,
                    positions=(
                        question["options"][first],
                        question["options"][second],
                    ),
                    story_title=entry["title"],
                )
            )
        return rows


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
        **kwargs,
    ) -> QualityDataset:
        """Constructs a QualityDataset"""

        train_split, val_split, test_split = QualityLoader.get_splits(
            train_filepath=train_filepath, val_filepath=val_filepath, test_filepath=test_filepath
        )
        return QualityDataset(
            train_data=train_split,
            val_data=val_split,
            test_data=test_split,
            allow_multiple_positions_per_question=allow_multiple_positions_per_question,
        )
