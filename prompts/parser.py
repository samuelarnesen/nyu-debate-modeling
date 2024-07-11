from data import AnnotationBracket, AnnotatedQualityDebatesDataset, AnnotationTag, DataRow, SplitType
import utils.constants as constants

from pydantic import BaseModel
import yaml

from enum import Enum
from typing import Any, Optional
import os
import re


class HardcodedTopicConfig(BaseModel):
    topic: str
    positions: tuple[str, str]


class PromptLoadingConfig(BaseModel):
    file_path: Optional[str] = None
    default_prompt_name: str = "Debate Prompt"
    use_hardcoded_topics: bool = False
    hardcoded_topic_config: Optional[HardcodedTopicConfig] = None
    is_memorized: bool = False


class PromptTag(Enum):
    PRE_DEBATE = 1
    PRE_OPENING_SPEECH = 2
    PRE_OPPONENT_SPEECH = 3
    DEBATER_SYSTEM = 4
    JUDGE_SYSTEM = 5
    OVERALL_SYSTEM = 6
    PRE_SPEECH = 7
    PRE_DEBATE_JUDGE = 8
    PRE_DEBATER_A_SPEECH_JUDGE = 9
    PRE_DEBATER_B_SPEECH_JUDGE = 10
    POST_ROUND_JUDGE = 11
    JUDGE_QUESTION_INSTRUCTIONS = 12
    PRE_JUDGE_QUESTIONS = 13
    JUDGE_DECISION = 14
    DEBATER_SCRATCHPAD = 16
    JUDGE_DECISION_FOR_DEBATER = 17
    PREVIOUS_DEBATER_SCRATCHPAD = 18
    PRE_PREVIOUS_SPEECH = 19
    POST_ROUND_JUDGE_WITHOUT_REASONING = 20


class RoleType(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


class ExamplesTag(Enum):
    POSITIVE_EXAMPLES = 1
    NEGATIVE_EXAMPLES = 2


class Message(BaseModel):
    role: str | RoleType
    content: str | list[str]


class Prompt(BaseModel):
    name: str
    messages: dict[str, dict[str, Any]] | dict[PromptTag, Message]


class PromptConfig(BaseModel):
    name: str
    opponent_name: str
    position: str
    opponent_position: str
    topic: str
    background_text: str


class PromptParser:
    DEFAULT_PROMPT_FILE_PATH = os.environ[constants.SRC_ROOT] + "prompts/configs/prompts.yaml"
    DEFAULT_PROMPT_NAME = "Debate Prompt"

    try:
        with open(DEFAULT_PROMPT_FILE_PATH) as f:
            DEFAULT_YAML = yaml.safe_load(f)
    except:
        DEFAULT_YAML = None

    @classmethod
    def parse(
        cls,
        prompt_config: PromptConfig,
        prompts_file_path: Optional[str] = None,
        name: str = "Debate Prompt",
    ) -> Prompt:
        """
        Constructs a Prompt object that can then be used by a Debater or Judge to generate text.

        Params:
            prompt_config: configuration containing the values to fill in the prompt with
                (e.g. names of the debaters, topic to be debated, background text)
            prompts_file_path: path to where the prompt messages are listed
            name: the specific prompt name to use (aka which messages to select from the prompt file)

        Returns:
            prompt: a prompt object containing a list of messages that the agents use to run a debate round
        """
        if not prompts_file_path or prompts_file_path == PromptParser.DEFAULT_PROMPT_FILE_PATH and DEFAULT_YAML:
            loaded_yaml = PromptParser.DEFAULT_YAML
        else:
            prompts_file_path = prompts_file_path or PromptParser.DEFAULT_PROMPT_FILE_PATH
            with open(prompts_file_path) as f:
                loaded_yaml = yaml.safe_load(f)

        name = name or PromptParser.DEFAULT_PROMPT_NAME
        prompt = Prompt(name=name, messages=loaded_yaml[name])
        prompt.messages = {PromptTag[tag.upper()]: Message(**message) for tag, message in prompt.messages.items()}

        base_prompt = Prompt(name=name, messages=loaded_yaml[name])
        base_prompt.messages = {PromptTag[tag.upper()]: Message(**message) for tag, message in base_prompt.messages.items()}

        for prop, value in prompt_config:
            key = f"<{prop.upper()}>"
            for tag, messages in prompt.messages.items():
                for i, message in enumerate(messages.content):
                    prompt.messages[tag].content[i] = message.replace(key, str(value))
            for tag, messages in base_prompt.messages.items():
                for i, message in enumerate(messages.content):
                    base_prompt.messages[tag].content[i] = message.replace(key, str(value))
                if tag not in prompt.messages:
                    prompt.messages[tag] = base_prompt.messages[tag]

        return prompt

    @classmethod
    def generate_opponent_config(cls, config: PromptConfig) -> PromptConfig:
        """Generates a prompt config using the config from an opposing debater"""
        return PromptConfig(
            name=config.opponent_name,
            opponent_name=config.name,
            position=config.opponent_position,
            opponent_position=config.position,
            topic=config.topic,
            background_text=config.background_text,
        )

    @classmethod
    def convert_data_row_to_default_prompt_config(
        cls, row: DataRow, position: int, use_title_as_background_text: bool = False
    ) -> PromptConfig:
        """Generates a default prompt config using a data row -- used in training"""
        position = max(position, 0)
        return PromptConfig(
            name=constants.DEFAULT_DEBATER_A_NAME if position == 0 else constants.DEFAULT_DEBATER_B_NAME,
            opponent_name=constants.DEFAULT_DEBATER_B_NAME if position == 0 else constants.DEFAULT_DEBATER_A_NAME,
            position=row.positions[position],
            opponent_position=row.positions[(position - 1) * -1],
            topic=row.question,
            background_text=row.background_text if not use_title_as_background_text else row.story_title,
        )
