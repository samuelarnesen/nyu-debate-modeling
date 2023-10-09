from __future__ import annotations

from pydantic import BaseModel

from abc import ABC
from enum import Enum
from typing import Optional


class RoleType(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


class ModelInput(BaseModel):
    role: RoleType
    content: str


class Model(ABC):
    def __init__(self, alias: str, is_debater: bool = False):
        self.alias = alias
        self.is_debater = is_debater

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens: 250) -> str:
        pass

    def copy(self, is_debater: Optional[bool] = None) -> Model:
        pass
