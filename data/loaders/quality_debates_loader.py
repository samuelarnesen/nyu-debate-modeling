from data.data import DataRow, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType

from typing import Any, Optional
import json


class QualityDebatesDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[DataRow]:
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[DataRow]:
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, batch: list[dict[str, Any]]):
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
                    position=turn["index"] if turn["role"] == "Debater" else -1,
                    speaker_type=SpeakerType.DEBATER if turn["role"] == "Debater" else SpeakerType.JUDGE,
                )
                for turn in entry["turns"]
            ],
            correct_index=self.__get_correct_answer(entry),
        )


class QualityDebatesLoader(RawDataLoader):
    @classmethod
    def load(
        cls,
        full_dataset_filepath: str,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> QualityDebatesDataset:
        def __should_keep(row: dict[str, Any]) -> bool:
            roles = [turn["role"] for turn in row["turns"]]
            positions = set([turn.get("index") for turn in row["turns"]])
            return len(roles) >= 3 and "GPT-4" not in roles and 0 in positions and 1 in positions

        def __get_filtered_rows(file_path: str):
            rows = []
            with open(file_path) as f:
                for line in f.readlines():
                    rows.append(json.loads(line))
            return [row for row in filter(__should_keep, rows)]

        filtered_rows = __get_filtered_rows(file_path=full_dataset_filepath)
        length = len(filtered_rows)
        return QualityDebatesDataset(
            train_data=filtered_rows[0 : int(0.8 * length)],
            val_data=filtered_rows[int(0.8 * length) : int(0.9 * length)],
            test_data=filtered_rows[int(0.9 * length) :],
        )
