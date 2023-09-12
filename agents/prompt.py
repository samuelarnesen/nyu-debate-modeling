from agents.model import RoleType

from pydantic import BaseModel
import yaml

from enum import Enum
from typing import Any, Union


class Message(BaseModel):
    role: Union[str, RoleType]
    content: str


class PromptConfig(BaseModel):
    name: str
    opponent_name: str
    judge_name: str
    word_limit: int
    position: str
    opponent_position: str
    topic: str


class PromptTag(Enum):
    PRE_DEBATE = 1
    PRE_TURN = 2
    PRE_OPPONENT_TURN = 3
    DEBATER_SYSTEM = 4
    JUDGE_SYSTEM = 5
    OVERALL_SYSTEM = 6


class Prompt(BaseModel):
    name: str
    messages: Union[dict[str, dict[str, Any]], dict[PromptTag, Message]]


class PromptParser:
    @classmethod
    def parse(cls, prompts_file_path: str, prompt_config: PromptConfig, name: str) -> Prompt:
        with open(prompts_file_path) as f:
            loaded_yaml = yaml.safe_load(f)

        prompt = Prompt(name=name, messages=loaded_yaml[name])
        prompt.messages = {PromptTag[tag.upper()]: Message(**message) for tag, message in prompt.messages.items()}

        for prop, value in prompt_config:
            key = f"<{prop.upper()}>"
            for tag, message in prompt.messages.items():
                prompt.messages[tag].content = prompt.messages[tag].content.replace(key, str(value))

        return prompt

    @classmethod
    def generate_opponent_config(cls, config: PromptConfig) -> PromptConfig:
        return PromptConfig(
            name=config.opponent_name,
            opponent_name=config.name,
            judge_name=config.judge_name,
            word_limit=config.word_limit,
            position=config.opponent_position,
            opponent_position=config.position,
            topic=config.topic,
        )
