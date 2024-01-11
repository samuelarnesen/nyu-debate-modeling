from data import AnnotationBracket, AnnotatedQualityDebatesDataset, AnnotationTag, DataRow, SplitType
import utils.constants as constants

from pydantic import BaseModel
import yaml

from enum import Enum
from typing import Any, Optional
import os
import random
import re


class HardcodedTopicConfig(BaseModel):
    topic: str
    positions: tuple[str, str]


class DynamicPromptsLoadingConfig(BaseModel):
    dynamic_prompts_file_path: Optional[str] = None
    dynamic_prompt_name: Optional[str] = None


class PromptLoadingConfig(BaseModel):
    file_path: Optional[str] = None
    default_prompt_name: str = "Base Prompt"
    use_dynamic_prompt: bool = False
    dynamic_prompts_config: DynamicPromptsLoadingConfig = DynamicPromptsLoadingConfig()
    use_hardcoded_topics: bool = False
    hardcoded_topic_config: Optional[HardcodedTopicConfig] = None


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
    PREFERENCE_JUDGE_INSTRUCTION = 17
    JUDGE_DECISION_FOR_DEBATER = 18
    PREFERENCE_JUDGE_OVERVIEW_FOR_DEBATER_A = 19
    PREFERENCE_JUDGE_OVERVIEW_FOR_DEBATER_B = 20
    PREVIOUS_DEBATER_SCRATCHPAD = 21
    PRE_PREVIOUS_SPEECH = 22
    POST_ROUND_JUDGE_WITHOUT_REASONING = 23


class RoleType(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


class DynamicEligibilityCriteria(BaseModel):
    tag: str
    bracket: str | AnnotationBracket
    threshold: float


class DynamicDisplayConfig(BaseModel):
    positive_examples: int
    negative_examples: int


class DynamicPrompt(BaseModel):
    eligibility_criteria: DynamicEligibilityCriteria
    display: DynamicDisplayConfig
    counterpart: list[str]
    reference_prompt: str
    messages: dict[PromptTag | str, str]


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
    word_limit: int
    position: str
    opponent_position: str
    topic: str
    background_text: str


class PromptParser:
    DEFAULT_PROMPT_FILE_PATH = os.environ[constants.SRC_ROOT] + "prompts/configs/prompts.yaml"
    DEFAULT_PROMPT_NAME = "Base Prompt"

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
        name: str = "Base Prompt",
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
        """Generates a prompt config using the config from an opposing debater"""
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
    def convert_data_row_to_default_prompt_config(
        cls, row: DataRow, position: int, use_title_as_background_text: bool = False
    ) -> PromptConfig:
        """Generates a default prompt config using a data row -- used in training"""
        return PromptConfig(
            name=constants.DEFAULT_DEBATER_A_NAME if position == 0 else constants.DEFAULT_DEBATER_B_NAME,
            opponent_name=constants.DEFAULT_DEBATER_B_NAME if position == 0 else constants.DEFAULT_DEBATER_A_NAME,
            word_limit=constants.DEFAULT_WORD_LIMIT,
            position=row.positions[position],
            opponent_position=row.positions[(position - 1) * -1],
            topic=row.question,
            background_text=row.background_text if not use_title_as_background_text else row.story_title,
        )


class DynamicPromptParser:
    DEFAULT_DYNAMIC_PROMPT_FILE_PATH = os.environ["SRC_ROOT"] + "/prompts/configs/dynamic_prompts.yaml"

    @classmethod
    def get_dynamic_prompt(
        cls,
        row: DataRow,
        prompt_config: PromptConfig,
        dynamic_prompts: list[DynamicPrompt],
        dataset: AnnotatedQualityDebatesDataset,
    ) -> Optional[DynamicPrompt]:
        """
        A Dynamic Prompt is a prompt whose messages depends on the attributes of the underlying data row.
        This is primarily used during training. For example, we might want to attach extra text to
        a speech that contains lots of quotes, instructing it to "have lots of quotes" in the hope that
        the model learns to use the stylistic cues from the prompt during generation.

        Params:
            row: the row in the dataset that we are constructing the prompt for
            prompt_config: the set of values to be filled into the prompt (e.g. background text, topic)
            dynamic_prompts: a configuration for handling these varied prompts
            dataset: the dataset that the row was sampled from. This is needed so that we can determine
                where the inputted row is within the distribution of all rows in the dataset.

        Returns:
            dynamic_prompt: a prompt containing a set of messages for the debaters to use during generation
                (similar to a normal prompt object) along with the associated criteria to determine whether
                the prompt should be applied.
        """
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
        prompt: Prompt,
        prompt_config: PromptConfig,
        dataset: AnnotatedQualityDebatesDataset,
        row: DataRow,
        dynamic_prompt_file_path: Optional[str] = None,
        dynamic_prompt_name: str = "Default Dynamic Prompt",
    ) -> Prompt:
        """
        Constructs a dynamic prompt based on the inputted prompt. See DynamicPromptParser.get_dynamic_prompt()
        for a more detailed explanation on what a dynamic prompt is.

        Params:
            dynamic_prompt_file_path: path to the file containing the set of dynamic prompt messages
            prompt: the default messages to be filled in
            prompt_config: the default values to be filled into the prompt (e.g. background text, names, topic)
            dataset: the dataset that the row was sampled from. This is needed so that we can determine
                where the inputted row is within the distribution of all rows in the dataset.
            row: the row in the dataset that we are constructing the prompt for
            dynamic_prompt_name: the name of the subset of messages to use in the file found at dynamic_prompt_file_path

        Returns:
            prompt: A prompt object that can be used like any other prompt to generate speeches for a debater.

        """
        dynamic_prompt_file_path = dynamic_prompt_file_path or DynamicPromptParser.DEFAULT_DYNAMIC_PROMPT_FILE_PATH
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
