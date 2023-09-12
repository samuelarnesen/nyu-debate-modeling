from pydantic import BaseModel

from abc import ABC
from enum import Enum


class RoleType(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


class ModelInput(BaseModel):
    role: RoleType
    content: str


class Model(ABC):
    def predict(self, inputs: list[ModelInput], max_length: 250) -> str:
        pass
