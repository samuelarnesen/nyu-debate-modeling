from __future__ import annotations

from pydantic import BaseModel

from prompts import RoleType

from abc import ABC
from enum import Enum
from typing import Optional


class ModelInput(BaseModel):
    role: RoleType
    content: str


class SpeechStructure(Enum):
    OPEN_ENDED = 1
    DECISION = 2
    PREFERENCE = 3


class Model(ABC):
    def __init__(self, alias: str, is_debater: bool = False):
        self.alias = alias
        self.is_debater = is_debater

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens: 250, **kwargs) -> str:
        pass

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> Model:
        return self
