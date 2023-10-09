from __future__ import annotations

from agents.model import Model, ModelInput

from typing import Union, Optional
import random


class RandomModel(Model):
    def __init__(self, alias: str, is_debater: bool = False):
        super().__init__(alias=alias, is_debater=is_debater)
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens=250) -> str:
        def generate_random_text():
            return " ".join(
                [
                    "".join(random.choices(self.alphabet, k=random.randrange(1, 8)))
                    for i in range(random.randrange(1, max_new_tokens))
                ]
            )

        return [generate_random_text() for i in range(len(inputs))]

    def copy(self, alias: str, is_debater: Optional[bool] = None) -> RandomModel:
        return RandomModel(alias=alias, is_debater=is_debater if is_debater is not None else False)
