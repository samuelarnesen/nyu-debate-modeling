from data import AnnotationBracket, AnnotatedQualityDebatesDataset, AnnotationTag, DataRow, SplitType
import utils.constants as constants

from pydantic import BaseModel
import yaml

from enum import Enum
from typing import Any, Optional, Union
import random
import re


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
    BEST_OF_N_JUDGE_INSTRUCTION = 17
    JUDGE_DECISION_FOR_DEBATER = 18
    BEST_OF_N_JUDGE_OVERVIEW_FOR_DEBATER_A = 19
    BEST_OF_N_JUDGE_OVERVIEW_FOR_DEBATER_B = 20


class RoleType(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


class DynamicEligibilityCriteria(BaseModel):
    tag: str
    bracket: Union[str, AnnotationBracket]
    threshold: float


class DynamicDisplayConfig(BaseModel):
    positive_examples: int
    negative_examples: int


class DynamicPrompt(BaseModel):
    eligibility_criteria: DynamicEligibilityCriteria
    display: DynamicDisplayConfig
    counterpart: list[str]
    reference_prompt: str
    messages: dict[Union[PromptTag, str], str]


class ExamplesTag(Enum):
    POSITIVE_EXAMPLES = 1
    NEGATIVE_EXAMPLES = 2


class Message(BaseModel):
    role: Union[str, RoleType]
    content: Union[str, list[str]]


class Prompt(BaseModel):
    name: str
    messages: Union[dict[str, dict[str, Any]], dict[PromptTag, Message]]


class PromptConfig(BaseModel):
    name: str
    opponent_name: str
    word_limit: int
    position: str
    opponent_position: str
    topic: str
    background_text: str


class PromptParser:
    @classmethod
    def parse(cls, prompts_file_path: str, prompt_config: PromptConfig, name: str) -> Prompt:
        with open(prompts_file_path) as f:
            loaded_yaml = yaml.safe_load(f)

        prompt = Prompt(name=name, messages=loaded_yaml[name])
        prompt.messages = {PromptTag[tag.upper()]: Message(**message) for tag, message in prompt.messages.items()}

        base_prompt = Prompt(name=constants.BASE_PROMPT, messages=loaded_yaml[constants.BASE_PROMPT])
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
        return PromptConfig(
            name=config.opponent_name,
            opponent_name=config.name,
            word_limit=config.word_limit,
            position=config.opponent_position,
            opponent_position=config.position,
            topic=config.topic,
            background_text=config.background_text,
        )

    @classmethod
    def convert_data_row_to_default_prompt_config(cls, row: DataRow, position: int) -> PromptConfig:
        return PromptConfig(
            name=constants.DEFAULT_DEBATER_A_NAME if position == 0 else constants.DEFAULT_DEBATER_B_NAME,
            opponent_name=constants.DEFAULT_DEBATER_B_NAME if position == 0 else constants.DEFAULT_DEBATER_A_NAME,
            word_limit=constants.DEFAULT_WORD_LIMIT,
            position=row.positions[position],
            opponent_position=row.positions[(position - 1) * -1],
            topic=row.question,
            background_text=row.background_text,
        )


class DynamicPromptParser:
    @classmethod
    def get_dynamic_prompt(
        cls,
        row: DataRow,
        prompt_config: PromptConfig,
        dynamic_prompts: list[DynamicPrompt],
        dataset: AnnotatedQualityDebatesDataset,
    ) -> Optional[DynamicPrompt]:
        predicate = lambda speech: speech.position == (
            constants.DEBATER_A_POSITION
            if prompt_config.name == constants.DEFAULT_DEBATER_A_NAME
            else constants.DEBATER_B_POSITION
        )

        # TODO: -- note that this only works for opening speeches!!!!!
        # TODO: -- note that this only works for single dynamic prompts (no combos)
        speech_to_use = None
        for speech in filter(lambda x: x.annotation and x.annotation.percentiles and predicate(x), row.speeches):
            speech_to_use = speech
            break

        if not speech_to_use:
            return

        for prompt in dynamic_prompts:
            tag = AnnotationTag[prompt.eligibility_criteria.tag.upper()]
            meets_threshold = AnnotatedQualityDebatesDataset.meets_threshold(
                tag=tag,
                bracket=AnnotationBracket[prompt.eligibility_criteria.bracket.upper()],
                threshold=prompt.eligibility_criteria.threshold,
                positive=True,
                speech=speech_to_use,
            )
            if meets_threshold:
                examples = dataset.get_annotation_examples(
                    tag=tag,
                    bracket=AnnotationBracket[prompt.eligibility_criteria.bracket.upper()],
                    threshold=prompt.eligibility_criteria.threshold,
                    positive=True,
                    source_row=row,
                )

                examples = random.sample(examples, k=prompt.display.positive_examples)
                example_string = "\n\n".join(
                    [
                        "{}. {}".format(i + 1, re.sub("\n+", " ", speech_data.text, flags=re.DOTALL))
                        for i, speech_data in enumerate(examples)
                    ]
                )
                for key, value in prompt.messages.items():
                    prompt.messages[key] = value.replace(f"<{ExamplesTag.POSITIVE_EXAMPLES.name}>", example_string)

                return prompt

        return None

    @classmethod
    def convert_to_dynamic_prompt(
        cls,
        dynamic_prompt_file_path: str,
        prompt: Prompt,
        prompt_config: PromptConfig,
        dataset: AnnotatedQualityDebatesDataset,
        index: int,
        split: SplitType,
        row: DataRow,
        dynamic_prompt_name: str,
    ) -> Prompt:
        row = dataset.get_example(split=split, idx=index)

        with open(dynamic_prompt_file_path) as f:
            dynamic_loaded_yaml = yaml.safe_load(f)

        dynamic_prompts = [
            DynamicPrompt(**dynamic_loaded_yaml[dynamic_prompt_name][name])
            for name in dynamic_loaded_yaml[dynamic_prompt_name]
        ]

        dynamic_prompt = DynamicPromptParser.get_dynamic_prompt(
            row=row, prompt_config=prompt_config, dynamic_prompts=dynamic_prompts, dataset=dataset
        )

        if dynamic_prompt:
            for tag, message in dynamic_prompt.messages.items():
                tag_to_use = PromptTag[tag.upper()]
                if tag_to_use in prompt.messages:
                    for i, existing_message in enumerate(prompt.messages[tag_to_use].content):
                        prompt.messages[tag_to_use].content[i] = f"{existing_message}\n{message}"

        return prompt
