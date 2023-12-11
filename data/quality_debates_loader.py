from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType

from typing import Any, Optional
import json
import re


class QualityDebatesDataset(RawDataset):
    def __init__(
        self,
        train_data: list[str, Any],
        val_data: list[str, Any],
        test_data: list[str, Any],
        override_type: DatasetType = None,
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
                    position=turn["index"] if turn["role"] == "Debater" else 0,
                    speaker_type=SpeakerType.DEBATER if turn["role"] == "Debater" else SpeakerType.JUDGE,
                )
                for turn in entry["turns"]
            ],
            correct_index=self.__get_correct_answer(entry),
            debate_id="_".join([entry["storyTitle"], entry["debateId"]]),
            story_title=entry["storyTitle"],
        )


class QualityDebatesLoader(RawDataLoader):
    @classmethod
    def get_splits(cls, file_path: str, deduplicate: bool = False) -> tuple[list[dict]]:
        """
        Filters the dataset and splits it into train, val, and test splits. Consultancy,
        offline debates, and gpt4 debates are excluded.

        Params:
            file_path: the path to the debate transcripts.
            deduplicate: whether we should return only one transcript for a given question.

        Returns:
            train_data: a list of json rows corresponding to the filtered training data
            val_data: a list of json rows corresponding to the filtered validation data
            test_data: a list of json rows corresponding to the filtered test data
        """

        def __should_keep(row: dict[str, Any]) -> bool:
            roles = [turn["role"] for turn in row["turns"]]
            positions = set([turn.get("index") for turn in row["turns"]])
            return (
                len(roles) >= 3
                and "GPT-4" not in roles
                and "Offline Judge" not in roles
                and 0 in positions
                and 1 in positions
            )

        def __get_filtered_rows(file_path: str):
            rows = []
            with open(file_path) as f:
                for line in f.readlines():
                    rows.append(json.loads(line))
            return [row for row in filter(__should_keep, rows)]

        def __create_splits(filtered_rows: list[dict]):
            story_to_row = {}
            story_to_question = {}
            for row in filtered_rows:
                if row["story"] not in story_to_row:
                    story_to_row[row["story"]] = []
                    story_to_question[row["story"]] = []
                if row["question"] not in story_to_question[row["story"]] or not deduplicate:
                    story_to_row[row["story"]].append(row)
                    story_to_question[row["story"]].append(row["question"])
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
            return train, val, test

        filtered_rows = __get_filtered_rows(file_path=file_path)
        return __create_splits(filtered_rows)

    @classmethod
    def load(
        cls,
        full_dataset_filepath: str,
        deduplicate: bool = False,
        **kwargs,
    ) -> QualityDebatesDataset:
        """Constructs a QualityDebatesDataset"""
        train, val, test = QualityDebatesLoader.get_splits(file_path=full_dataset_filepath, deduplicate=deduplicate)
        return QualityDebatesDataset(
            train_data=train,
            val_data=val,
            test_data=test,
        )
