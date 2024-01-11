from __future__ import annotations

from agents.agent import Agent, BestOfNConfig, ScratchpadConfig
from agents.judge import JudgeUtils
from agents.models import HumanModel, Model, ModelResponse, OfflineModel, SpeechStructure
from agents.transcript import SpeechFormat, Transcript
from prompts import Prompt, PromptTag
from utils import LoggerUtils, QuoteUtils
import utils.constants as constants

from pydantic import BaseModel

from typing import Optional, Union
import copy


class Debater(Agent):
    def __init__(
        self,
        name: str,
        prompt: Prompt | list[Prompt],
        model: Model,
        num_speeches: int,
        speech_format: Optional[SpeechFormat] = None,
        scratchpad_config: ScratchpadConfig = ScratchpadConfig(),
        quotes_require_validation: bool = True,
    ):
        """
        An abstraction that corresponds to a debater in the round.

        Params:
            name: A string to identify the debater. It needs only to be unique within its own debate round.
            is_debater: Boolean indicating whether the agent is a debater or a judge.
            prompt: The Prompt structure that controls the inputs to the models. A list is passed in for batch processing.
            model: The model that actually performs the text generation.
            num_speeches: The number of speeches each debater will generate in the round.
            speech_format: The order of speeches that the debater is expecting to receive.
            scratchpad_word_limit: Number of words that should be generated in the scratchpad. If this is None or 0,
                then a scratchpad will not be used.
            scratchpad_public: Whether the scratchpad generation should be concatenated to the regular generation when
                sharing the results with other agents.
            quotes_require_validation: Whether or not the speeches generated by this agent already have had their quotes
                validated. Quote validation takes some time, so this helps us perform validation only when necessary. This
                is true for speeches generated by the HumanModel and false for the other models.
        """
        super().__init__(
            name=name,
            is_debater=True,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            receive_validated_quotes=False,
            quotes_require_validation=quotes_require_validation,
            speech_format=speech_format
            if speech_format
            else DebaterUtils.get_default_speech_format(name, num_speeches, scratchpad_config.use_scratchpad),
        )
        self.scratchpad_config = scratchpad_config
        self.quotes_require_validation = quotes_require_validation
        self.logger = LoggerUtils.get_default_logger(__name__)

    def generate(self, max_new_tokens=300, round_idx: int = 0) -> Optional[list[ModelResponse]]:
        """Generates new text using the pre-existing transcript as input"""
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(
            inputs=model_inputs, max_new_tokens=max_new_tokens, debater_name=self.name, round_idx=round_idx
        )

    def copy(
        self, transcripts: Optional[list[Transcript]] = None, prompts: Optional[list[Prompt] | Prompt] = None
    ) -> Debater:
        """Deepcopies the debater (except for the model, which is a shallow copy)"""
        debater = Debater(
            name=self.name,
            prompt=prompts if prompts else [copy.deepcopy(prompt) for prompt in self.prompts],
            model=self.model,
            num_speeches=self.num_speeches,
            speech_format=self.speech_format,
            scratchpad_config=self.scratchpad_config,
            quotes_require_validation=self.quotes_require_validation,
        )
        if transcripts:
            debater.transcripts = [transcript.copy() for transcript in transcripts]
        return debater

    def __call__(self) -> tuple[list[str], Optional[list[ModelResponse]]]:
        """Generates new text using the pre-existing transcript as input. If it has access to a
        scratchpad, it will use that but keep those results hidden."""
        batch_reasoning = []
        if self.scratchpad_config.use_scratchpad:
            batch_reasoning = [
                reasoning.speech for reasoning in self.generate(max_new_tokens=self.scratchpad_config.scratchpad_word_limit)
            ]
            for i, reasoning in enumerate(batch_reasoning):
                super().receive_message(speaker=self.name, content=reasoning, idx=i)
                self.logger.debug(reasoning)

        generation = self.generate(max_new_tokens=300)
        all_speeches = [gen.speech for gen in generation]

        if self.scratchpad_config.use_scratchpad and self.scratchpad_config.scratchpad_public:
            all_speeches = [
                constants.LINE_SEPARATOR.join([reasoning, speech]) for reasoning, speech in zip(all_speeches, generation)
            ]

        return all_speeches, generation


class BestOfNDebater(Debater):
    def __init__(self, debater: Debater, opposing_debater: Debater, judge: Judge, best_of_n_config: BestOfNConfig):
        super().__init__(
            name=debater.name,
            prompt=debater.prompts,
            model=debater.model,
            num_speeches=debater.num_speeches,
            speech_format=debater.speech_format,
        )
        self.opposing_debater = opposing_debater
        self.base_opponent_transcript = copy.deepcopy(opposing_debater.transcripts[0])
        self.judge = judge
        self.config = best_of_n_config

    def __call__(self):
        # just doing round 1 for now and unbatched inputs
        model_responses = self.model.predict(
            inputs=[self.transcripts[0].to_model_input() for _ in range(self.config.n)],
            max_new_tokens=300,
            debater_name=self.name,
        )

        opposing_debater_responses = self.model.predict(
            inputs=[self.base_opponent_transcript.to_model_input() for _ in range(self.config.opponent_n)],
            max_new_tokens=300,
            debater_name=self.opposing_debater.name,
        )

        judge_inputs = []
        for response in model_responses:
            for opposing_response in opposing_debater_responses:
                judge_transcript = Transcript(
                    name=self.judge.transcripts[0].name,
                    prompt=self.judge.transcripts[0].prompt,
                    speech_format=JudgeUtils.get_default_speech_format(
                        num_speeches=self.judge.num_speeches, chain_of_thought=False
                    ),
                )
                if self.name == constants.DEFAULT_DEBATER_A_NAME:
                    judge_transcript.add_speech(speaker=self.name, content=response.speech)
                    judge_transcript.add_speech(speaker=self.opposing_debater.name, content=opposing_response.speech)
                else:
                    judge_transcript.add_speech(speaker=self.opposing_debater.name, content=opposing_response.speech)
                    judge_transcript.add_speech(speaker=self.name, content=response.speech)

                judge_inputs.append(judge_transcript.to_model_input())

        judge_model_response = self.judge.model.predict(
            inputs=judge_inputs, max_new_tokens=15, speech_structure=SpeechStructure.DECISION
        )

        split_judge_response = [
            [resp.probabilistic_decision[self.name] for resp in judge_model_response[i : i + self.config.opponent_n]]
            for i in range(0, len(judge_model_response), self.config.opponent_n)
        ]
        scores = [min(option) if self.config.maxmin else sum(option) / len(option) for option in split_judge_response]
        selection_idx = sorted(zip(scores, range(len(model_responses))), key=lambda x: x[0], reverse=True)[0][1]
        best_model_response = model_responses[selection_idx]

        for i, (model_response, score) in enumerate(zip(model_responses, scores)):
            if i == selection_idx:
                self.logger.debug("Chosen Speech (Score of {}):\n{}\n".format(round(score, 2), model_response.speech))
            else:
                self.logger.debug("Rejected Speech (Score of {}):\n{}\n".format(round(score, 2), model_response.speech))

        for opposing_response in opposing_debater_responses:
            self.logger.debug("Opponent Speech:\n{}\n".format(opposing_response.speech))

        return [best_model_response.speech], [best_model_response]

    def copy(
        self, transcripts: Optional[list[Transcript]] = None, prompts: Optional[list[Prompt] | Prompt] = None
    ) -> Debater:
        """Deepcopies the debater (except for the model, which is a shallow copy)"""
        debater = super().copy(transcripts=transcripts, prompts=prompts)
        return BestOfNDebater(
            debater=debater, opposing_debater=self.opposing_debater, judge=self.judge, best_of_n_config=self.config
        )


class PreferenceDebater(Debater):
    def __init__(self, debater: Debater, n: int, prompts: Optional[list[Prompt]] = None, evaluated: bool = True):
        """
        A debater model that generates multiple versions of the same speech for preference judgment comparisons.

        Params:
            debater: The underlying debater that is to be converted to a PreferenceDebater
            n: The number of speeches to generate for each input.
            prompts: The Prompt structure that controls the inputs to the models. A list is passed in for batch processing.
            evaluated: indicates whether the round has already been judged.
        """
        super().__init__(
            name=debater.name,
            prompt=PreferenceDebater.construct_prompts(debater=debater, n=n if evaluated else 1, prompts=prompts),
            model=debater.model,
            num_speeches=debater.num_speeches,
            speech_format=DebaterUtils.get_preference_speech_format(
                debater.name, debater.num_speeches, debater.scratchpad_config.use_scratchpad
            ),
        )
        self.n = n
        self.evaluated = evaluated

    @classmethod
    def construct_prompts(cls, debater: Debater, n: int, prompts: Optional[list[Prompt]]):
        """Copies the prompts from the existing debater"""
        prompts = prompts if prompts else debater.prompts
        return [copy.deepcopy(prompts[i % len(prompts)]) for i in range(n)]

    def copy(
        self, transcripts: Optional[list[Transcript]] = None, prompts: Optional[list[Prompt] | Prompt] = None
    ) -> Debater:
        """Deepcopies the debater (except for the model, which is a shallow copy)"""
        debater = super().copy(transcripts=self.transcripts, prompts=prompts)
        return PreferenceDebater(debater=debater, n=self.n, prompts=self.prompts, evaluated=self.evaluated)

    def generate(self, max_new_tokens=300) -> Optional[list[str]]:
        """Generates new text based on the pre-existing transcripts"""
        prediction = []
        for transcript in self.transcripts:
            prediction.extend(
                self.model.predict(
                    inputs=[transcript.to_model_input()],
                    max_new_tokens=max_new_tokens,
                    debater_name=self.name,
                )
            )
        if not self.evaluated and self.n > len(prediction):
            prediction.extend([copy.deepcopy(prediction[0]) for i in range(self.n - len(prediction))])
        return prediction


class OfflineDebater(Debater):
    def __init__(self, debater: Debater, file_path: str, first_debater_prompt: Prompt, round_idx: int = 0):
        """
        A separate abstraction for a debater that uses an OfflineModel.

        Params:
            debater: The underlying debater that is to be converted to an OfflineDebater
            file_path: The path to the transcripts that the OfflineModel will use to replay text
            first_debater_prompt: The prompt of the first debater in te debate round (needed for parsing)
            round_idx: The index of the round within the context off all the rounds found at the file_path.
        """
        super().__init__(
            name=debater.name,
            prompt=debater.prompts,
            model=OfflineModel(
                alias=debater.model.alias, is_debater=debater.is_debater, file_path=file_path, prompt=first_debater_prompt
            ),
            num_speeches=debater.num_speeches,
            speech_format=debater.speech_format,
            quotes_require_validation=False,
        )
        self.round_idx = round_idx
        self.file_path = file_path
        self.first_debater_prompt = first_debater_prompt

    def copy(
        self, transcripts: Optional[list[Transcript]] = None, prompts: Optional[list[Prompt] | Prompt] = None
    ) -> Debater:
        """Deepcopies the debater"""
        debater = super().copy(transcripts=self.transcripts, prompts=prompts)
        return OfflineDebater(
            debater=debater,
            file_path=self.file_path,
            first_debater_prompt=self.first_debater_prompt,
            round_idx=self.round_idx,
        )

    def __call__(self) -> Optional[list[str]]:
        """Generates new text using the OfflineModel"""
        generation = self.generate(max_new_tokens=300, round_idx=self.round_idx)
        return [gen.speech for gen in generation], generation


class HumanDebater(Debater):
    def __init__(self, debater: Debater, speeches: list[SpeechData]):
        """
        A separate abstraction for a debater that uses a HumanModel.

        Params:
            debater: The undelrying debater that is to be converted to a HumanDebater.
            speeches: The list of speeches from the dataset that are to be delivered when text is generated
        """
        super().__init__(
            name=debater.name,
            prompt=debater.prompts,
            model=HumanModel(
                alias=debater.model.alias, is_debater=debater.is_debater, debater_name=debater.name, speeches=speeches
            ),
            num_speeches=debater.num_speeches,
            speech_format=debater.speech_format,
            quotes_require_validation=False,
        )


class DebaterUtils:
    @classmethod
    def get_speech_format(cls, name: str, num_speeches: int, use_scratchpad: bool, preference: bool = False):
        """Generates the order of speeches that the debater expects to receive"""
        opponent_name = (
            constants.DEFAULT_DEBATER_A_NAME
            if name == constants.DEFAULT_DEBATER_B_NAME
            else constants.DEFAULT_DEBATER_B_NAME
        )
        pre_debate = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.DEBATER_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE)
        )

        judge_questions = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_JUDGE_QUESTIONS)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

        scratchpad = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PREVIOUS_DEBATER_SCRATCHPAD, last_only_prompt_tag=PromptTag.DEBATER_SCRATCHPAD)
            .add_user_inputted_speech(expected_speaker=name)
        )
        own_speech = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_PREVIOUS_SPEECH, last_only_prompt_tag=PromptTag.PRE_SPEECH)
            .add_user_inputted_speech(expected_speaker=name)
        )
        if use_scratchpad:
            own_speech = scratchpad.add_format(speech_format=own_speech)

        opponent_speech = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_OPPONENT_SPEECH)
            .add_user_inputted_speech(expected_speaker=opponent_name)
        )

        opening_statements = (
            SpeechFormat(name).add(prompt_tag=PromptTag.PRE_OPENING_SPEECH).add_format(speech_format=own_speech)
        )

        if not preference:
            opening_statements = opening_statements.add_format(speech_format=opponent_speech)

        later_arguments = (
            SpeechFormat(name)
            .add_format(speech_format=judge_questions)
            .add_format(speech_format=own_speech if name == constants.DEFAULT_DEBATER_A_NAME else opponent_speech)
            .add_format(speech_format=opponent_speech if name == constants.DEFAULT_DEBATER_A_NAME else own_speech)
        )

        decision = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.JUDGE_DECISION_FOR_DEBATER)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

        return (
            SpeechFormat(name)
            .add_format(speech_format=pre_debate)
            .add_format(speech_format=opening_statements)
            .add_format(speech_format=later_arguments, repeats=(num_speeches - 1))
            .add_format(speech_format=decision)
        )

    @classmethod
    def get_default_speech_format(cls, name: str, num_speeches: int, use_scratchpad: bool):
        """Gets the speech orders for a normal (non-Preference) debater"""
        return DebaterUtils.get_speech_format(
            name=name, num_speeches=num_speeches, use_scratchpad=use_scratchpad, preference=False
        )

    @classmethod
    def get_preference_speech_format(cls, name: str, num_speeches: int, use_scratchpad: bool):
        """Gets the speech order for a Preference debater"""
        return DebaterUtils.get_speech_format(
            name=name, num_speeches=num_speeches, use_scratchpad=use_scratchpad, preference=True
        )
