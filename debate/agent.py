from debate.transcript import SpeechFormat, Transcript
from models import BestOfNConfig, Model, ModelSettings
from prompts import Prompt
from utils import logger_utils
import utils.constants as constants

from pydantic import BaseModel, ConfigDict, model_validator

from typing import Any, Optional, Union


class ScratchpadConfig(BaseModel):
    use_scratchpad: bool = False
    scratchpad_word_limit: Optional[int] = None
    scratchpad_public: bool = False

    @model_validator(mode="before")
    def check_one_true_and_a_none(cls, values):
        if (
            not values.get("use_scratchpad")
            and (values.get("scratchpad_word_limit") is not None and values.get("scratchpad_word_limit") > 0)
            and values.get("scratchpad_public")
        ):
            raise ValueError("If use_scratchpad=False, then one should not set scratchpad_word_limit or scratchpad_public")
        return values


class AgentConfig(BaseModel):
    model_settings: ModelSettings
    scratchpad: ScratchpadConfig = ScratchpadConfig()
    best_of_n: Optional[BestOfNConfig] = None

    model_config = ConfigDict(protected_namespaces=("protect_me_", "also_protect_"))


class Agent:
    def __init__(
        self,
        name: str,
        is_debater: bool,
        prompt: Prompt | list[Prompt],
        model: Model,
        num_speeches: int,
        receive_validated_quotes: bool,
        quotes_require_validation: bool,
        speech_format: SpeechFormat,
    ):
        """
        An abstraction that controls access to the underlying models. It maintains a prompt and transcript
        to determine what to send to the underlying models. A Debater and a Judge are examples of an agent.

        Params:
            name: A string to identify the agent. It needs only to be unique within its own debate round.
            is_debater: Boolean indicating whether the agent is a debater or a judge.
            prompt: The Prompt structure that controls the inputs to the models. A list is passed in for batch processing.
            model: The model that actually performs the text generation.
            num_speeches: The number of speeches each debater will generate in the round.
            receive_validated_quotes: Whether speeches delivered by others should be corrected to show the validation status
                of their quotes. This is generally true for judges (they see whether a debater made up a quote) but not for
                debaters (they should learn not to make up quotes).
            quotes_require_validation: Whether or not the speeches generated by this agent already have had their quotes
                validated. Quote validation takes some time, so this helps us perform validation only when necessary. This
                is true for speeches generated by the HumanModel and false for the other models.
            speech_format: The order of speeches that the debater is expecting to receive.
        """
        self.name = name
        self.is_debater = is_debater
        self.model = model
        self.num_speeches = num_speeches
        self.receive_validated_quotes = receive_validated_quotes
        self.quotes_require_validation = quotes_require_validation

        self.speech_format = speech_format

        self.prompts = prompt if type(prompt) == list else [prompt]
        self.transcripts = [
            Transcript(name=self.name, prompt=p, speech_format=speech_format, index=i) for i, p in enumerate(self.prompts)
        ]
        self.cached_messages = {}

    def receive_message(
        self, speaker: str, content: str, idx: int, supplemental: Optional[dict[Any, Any] | list[dict[Any, Any]]] = None
    ):
        """
        The agent takes in a speech from another agent (or itself) and adds it to its internal transcript:

        Params:
            speaker: The name of the agent who delivered the speech
            content: The text of the speech
            idx: The index corresponding to which debate in the batch this speech is a part of.
            supplemental: Any additional data that one wants to associate with the speech
        """
        if idx >= len(self.transcripts):
            return

        self.cached_messages.setdefault(speaker, {}).setdefault(idx, []).append((content, supplemental))
        expected_speaker = self.get_next_expected_speaker(idx=idx)
        while self.cached_messages.get(expected_speaker, {}).get(idx):
            for message, supplemental in self.cached_messages[expected_speaker][idx]:
                self.transcripts[idx].add_speech(speaker=expected_speaker, content=message, supplemental=supplemental)
            del self.cached_messages[expected_speaker][idx]
            expected_speaker = self.get_next_expected_speaker(idx=idx)

    def __call__(self) -> Optional[list[str]]:
        """This must be implemented in each agent. This is where they should generate text"""
        pass

    def save(self, save_file_path_prefix: str, metadata: Optional[list[dict[Any, Any]]] = None):
        """Saves the transcripts to the specified location, with a separate file for each element in the batch"""
        metadata = (metadata or []) + [{} for i in range(len(self.transcripts) - len((metadata or [])))]
        for i, (transcript, metadata) in enumerate(zip(self.transcripts, metadata)):
            transcript.save(save_file_path_prefix=f"{save_file_path_prefix}_{i}", metadata=metadata)

    def get_transcript(self, idx: int = 0) -> Transcript:
        """Returns the transcript at the specified index"""
        return self.transcripts[idx]

    def get_alias(self) -> str:
        """Gets the alias of the model underpinning the agent"""
        return self.model.alias if self.model else constants.DEFAULT_ALIAS

    def get_next_expected_speaker(self, idx: int = 0) -> Optional[str]:
        """Gets the name of the agent that this agent expects to deliver the next speech"""
        return self.transcripts[idx].get_next_expected_speaker()

    def post_speech_processing(self) -> None:
        """Handles any post-speech logic. This should mostly be a no-op but is needed for some multi-round
        branching cases where the judge needs to handle speeches coming from different rounds"""
        pass
