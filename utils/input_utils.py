from typing import Union
import os


class InputUtils:
    @classmethod
    def read_file_texts(cls, base_path: str, group_by_batch: bool = False) -> list[Union[str, list[str]]]:
        def add_batch(current_batch: list[str], file_texts: list[Union[str, list[str]]]):
            if current_batch:
                if group_by_batch:
                    file_texts.append([example for example in current_batch])
                else:
                    file_texts += [example for example in current_batch]
                current_batch.clear()

        round_idx = 0
        batch_idx = 0
        keep_extracting = True
        file_texts = []
        current_batch = []
        while keep_extracting:
            candidate_path = f"{base_path}_{round_idx}_{batch_idx}.txt"
            if os.path.exists(candidate_path):
                with open(candidate_path) as f:
                    current_batch.append(f.read())
                batch_idx += 1
            elif batch_idx == 0:
                keep_extracting = False
                add_batch(current_batch, file_texts)
            else:
                round_idx += 1
                batch_idx = 0
                add_batch(current_batch, file_texts)
        return file_texts
