from __future__ import annotations

from pydantic import BaseModel, validator

from prompts import RoleType
import utils.constants as constants

from abc import ABC
from enum import Enum
from typing import Literal, Optional


class ModelInput(BaseModel):
    role: RoleType
    content: str


class ModelResponse(BaseModel):
    speech: str = ""
    decision: Literal[constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME, ""] = ""
    probabilistic_decision: Optional[dict[str, float]] = None
    preference: Optional[float] = None

    @validator("probabilistic_decision")
    def check_keys(cls, v):
        if v:
            if not constants.DEFAULT_DEBATER_A_NAME in v:
                raise ValueError(f"Probabilistic decision is missing required key: {constants.DEFAULT_DEBATER_A_NAME}")
            if not constants.DEFAULT_DEBATER_B_NAME in v:
                raise ValueError(f"Probabilistic decision is missing required key: {constants.DEFAULT_DEBATER_B_NAME}")
            if len(v) > 2:
                all_keys = ", ".join(v.keys())
                raise ValueError(f"There are too many keys in the probabilistic decision map. Keys: {all_keys}")

            eps = 0.001
            total_prob = sum(v.values())
            if total_prob < 1 - eps or total_prob > 1 + eps:
                raise ValueError(f"Total probability does not sum to 1 -- it sums to {total_prob}. Map is {v}")

        return v


class SpeechStructure(Enum):
    OPEN_ENDED = 1
    DECISION = 2
    PREFERENCE = 3


class Model(ABC):
    def __init__(self, alias: str, is_debater: bool = False):
        self.alias = alias
        self.is_debater = is_debater

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens: 250, **kwargs) -> ModelResponse:
        pass

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> Model:
        return self
