from agents.model import Model, ModelInput

import random


class RandomModel(Model):
    def __init__(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def predict(self, inputs: list[ModelInput], max_length=250) -> str:
        return " ".join(
            [
                "".join(random.choices(self.alphabet, k=random.randrange(1, 8)))
                for i in range(random.randrange(1, max_length))
            ]
        )
