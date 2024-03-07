from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
import utils.constants as constants

from typing import Any, Callable, Optional, Type
import json
import os
import re


class QualityDebatesDataset(RawDataset):
    def __init__(
        self,
        train_data: list[str, Any],
        val_data: list[str, Any],
        test_data: list[str, Any],
        override_type: Optional[DatasetType] = None,
    ):
        """
        Builds a dataset of all the questions and speeches from the human debate experiments. Each row
        is a question along with the assigned sides and a list of speeches.

        Params:
            train_data: a list of json objects corresponding to transcripts in the training set.
            val_data: a list of json objects corresponding to transcripts in the validation set.
            test_data: a list of json objects corresponding to transcripts in the test set.
            override_type: if a child class inherits from this dataset, this is the dataset_type to
                pass to the parent RawDataset constructor.
        """

        super().__init__(override_type or DatasetType.QUALITY_DEBATES)
        self.data = {
            SplitType.TRAIN: self.convert_batch_to_rows(train_data),
            SplitType.VAL: self.convert_batch_to_rows(val_data),
            SplitType.TEST: self.convert_batch_to_rows(test_data),
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
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def convert_batch_to_rows(self, batch: list[dict[str, Any]]):
        return [self.__example_to_row(entry) for entry in batch]

    def __get_correct_answer(self, entry: dict[str, Any]) -> int:
        # the GPT debates don't have judge probabilities but do have the correct answer marked
        if "correctAnswer" in entry and "answers" in entry and entry["correctAnswer"] in entry["answers"]:
            return entry["answers"].index(entry["correctAnswer"])

        judge_probs = entry["turns"][-1]["probabilities"]
        judge_correct = entry["isJudgeCorrect"]
        judge_probs = [
            turn["probabilities"] for turn in filter(lambda x: "Judge" in x["role"] and x["probabilities"], entry["turns"])
        ][-1]
        return (
            0
            if (
                (judge_probs[0] > judge_probs[1] and judge_correct)
                or (judge_probs[0] < judge_probs[1] and not judge_correct)
            )
            else 1
        )

    def __example_to_row(self, entry: dict[str, Any]) -> tuple[str, Any]:
        return DataRow(
            background_text=entry["story"],
            question=entry["question"],
            positions=entry["answers"],
            speeches=[
                SpeechData(
                    text=turn["text"],
                    position=turn["index"] if turn["role"] == "Debater" else -1,
                    speaker_type=SpeakerType.DEBATER if turn["role"] == "Debater" else SpeakerType.JUDGE,
                    probabilities=None if turn["role"] == "Debater" else turn["probabilities"],
                )
                for turn in entry["turns"]
            ],
            correct_index=self.__get_correct_answer(entry),
            debate_id="_".join([entry["storyTitle"], entry["debateId"]]),
            story_title=entry["storyTitle"],
        )


class QualityTranscriptsLoader:
    DEFAULT_FILE_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/quality-debates/debates-readable.jsonl"

    @classmethod
    def get_splits(
        cls,
        file_path: str,
        deduplicate: bool = False,
        combine_train_and_val: bool = False,
        should_keep: Optional[Callable[dict[str, Any], bool]] = None,
    ) -> tuple[list[dict]]:
        """
        Filters the dataset and splits it into train, val, and test splits. Consultancy,
        offline debates, and gpt4 debates are excluded.

        Params:
            file_path: the path to the debate transcripts.
            deduplicate: whether we should return only one transcript for a given question.
            should_keep: a function that accepts a row and returns whether it should be kept (not filtered)

        Returns:
            train_data: a list of json rows corresponding to the filtered training data
            val_data: a list of json rows corresponding to the filtered validation data
            test_data: a list of json rows corresponding to the filtered test data
        """

        def __get_filtered_rows(file_path: str):
            rows = []
            with open(file_path) as f:
                for line in f.readlines():
                    rows.append(json.loads(line))
            return [row for row in filter(should_keep, rows)]

        def __create_splits(filtered_rows: list[dict], combine_train_and_val: bool = False):
            story_to_row = {}
            story_to_question = {}
            for row in filtered_rows:
                if row["storyTitle"] not in story_to_row:
                    story_to_row[row["storyTitle"]] = []
                    story_to_question[row["storyTitle"]] = []
                if row["question"] not in story_to_question[row["storyTitle"]] or not deduplicate:
                    story_to_row[row["storyTitle"]].append(row)
                    story_to_question[row["storyTitle"]].append(row["question"])
            train = []
            val = []
            test = []
            for i, story in enumerate(story_to_row):
                if i < int(0.8 * len(story_to_row)):
                    train.extend(story_to_row[story])
                elif i < int(0.9 * len(story_to_row)):
                    val.extend(story_to_row[story])
                else:
                    test.extend(story_to_row[story])

            if combine_train_and_val:
                train = train + val

            return train, val, test

        filtered_rows = __get_filtered_rows(file_path=file_path)
        return __create_splits(filtered_rows=filtered_rows, combine_train_and_val=combine_train_and_val)

    @classmethod
    def load(
        cls,
        constructor_cls: Type[RawDataLoader],
        full_dataset_filepath: Optional[str] = None,
        deduplicate: bool = False,
        combine_train_and_val: bool = False,
        **kwargs,
    ) -> QualityDebatesDataset:
        """Constructs a QualityDebatesDataset"""
        full_dataset_filepath = full_dataset_filepath or QualityTranscriptsLoader.DEFAULT_FILE_PATH
        train, val, test = constructor_cls.get_splits(
            file_path=full_dataset_filepath, deduplicate=deduplicate, combine_train_and_val=combine_train_and_val
        )
        return QualityDebatesDataset(
            train_data=train,
            val_data=val,
            test_data=test,
        )


class QualityConsultancyLoader(RawDataLoader):
    @classmethod
    def get_splits(cls, file_path: str, deduplicate: bool = False, combine_train_and_val: bool = False):
        def should_keep(row: dict[str, Any]) -> bool:
            roles = [turn["role"] for turn in row["turns"]]
            positions = set([turn.get("index") for turn in filter(lambda x: x.get("index") is not None, row["turns"])])
            return len(set(roles)) >= 2 and "GPT-4" not in roles and "Offline Judge" not in roles and len(positions) == 1

        return QualityTranscriptsLoader.get_splits(
            file_path=file_path,
            deduplicate=deduplicate,
            combine_train_and_val=combine_train_and_val,
            should_keep=should_keep,
        )

    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        deduplicate: bool = False,
        combine_train_and_val: bool = False,
        **kwargs,
    ) -> QualityDebatesDataset:
        """Constructs a QualityDebatesDataset"""
        return QualityTranscriptsLoader.load(
            constructor_cls=cls,
            full_dataset_filepath=full_dataset_filepath,
            deduplicate=deduplicate,
            combine_train_and_val=combine_train_and_val,
        )


class QualityDebatesLoader(RawDataLoader):
    @classmethod
    def get_splits(cls, file_path: str, deduplicate: bool = False, combine_train_and_val: bool = False):
        def should_keep(row: dict[str, Any]) -> bool:
            roles = [turn["role"] for turn in row["turns"]]
            positions = set([turn.get("index") for turn in row["turns"]])
            return (
                len(roles) >= 3
                and "GPT-4" not in roles
                and "Offline Judge" not in roles
                and 0 in positions
                and 1 in positions
            )

        return QualityTranscriptsLoader.get_splits(
            file_path=file_path,
            deduplicate=deduplicate,
            combine_train_and_val=combine_train_and_val,
            should_keep=should_keep,
        )

    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        deduplicate: bool = False,
        combine_train_and_val: bool = False,
        **kwargs,
    ) -> QualityDebatesDataset:
        """Constructs a QualityDebatesDataset"""
        return QualityTranscriptsLoader.load(
            constructor_cls=cls,
            full_dataset_filepath=full_dataset_filepath,
            deduplicate=deduplicate,
            combine_train_and_val=combine_train_and_val,
        )
