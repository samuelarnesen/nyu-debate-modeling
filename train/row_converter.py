from agents import Debater, Judge, LLMInput, LLMType, SpeechFormatType, SpeechFormatStructure, Transcript
from data import DataRow, DatasetType, RawDataset, SpeakerType, SpeechData, SplitType
from prompts import Prompt, PromptParser, PromptTag
from train.train_utils import TrainingConfig, TrainingTarget
import utils.constants as constants

from typing import Any, Callable, Optional


class RowConverter:
    @classmethod
    def generate_prompt_from_speech(
        cls,
        row: DataRow,
        speech: SpeechData,
        config: TrainingConfig,
        dataset: RawDataset,
        speech_structure: SpeechFormatStructure,
    ) -> Prompt:
        """Constructs a prompt from a given speech and row in the dataset"""
        position = speech.position
        if config.target == TrainingTarget.JUDGE and speech_structure.num_participants == 1:
            all_positions = set([speech.position for speech in row.speeches])
            position = 0 if 0 in all_positions else 1

        prompt_config = PromptParser.convert_data_row_to_default_prompt_config(
            row=row,
            position=position,
            use_title_as_background_text=config.prompt_config.is_memorized,
        )
        return PromptParser.parse(
            prompt_config=prompt_config,
            prompts_file_path=config.prompt_config.file_path,
            name=speech_structure.default_prompt_name or config.prompt_config.default_prompt_name,
        )

        return prompt

    @classmethod
    def get_speaker_from_speech(cls, speech: SpeechData) -> str:
        """Returns the name (Debater_A, Debater_B) from the speech"""
        return (
            constants.DEFAULT_DEBATER_A_NAME
            if speech.position == 0
            else (constants.DEFAULT_DEBATER_B_NAME if speech.position == 1 else constants.DEFAULT_JUDGE_NAME)
        )

    @classmethod
    def get_default_speeches(cls) -> list[SpeechData]:
        """Returns empty speeches"""
        return [
            SpeechData(
                text="",
                position=0,
                speaker_type=SpeakerType.DEBATER,
            ),
            SpeechData(
                text="",
                position=1,
                speaker_type=SpeakerType.DEBATER,
            ),
            SpeechData(
                text="",
                position=0,
                speaker_type=SpeakerType.JUDGE,
            ),
        ]

    @classmethod
    def convert_transcript(
        cls,
        row: DataRow,
        config: TrainingConfig,
        skipping_func: Callable[[SpeechData], bool],
        is_debater: bool,
        dataset: RawDataset,
        speech_structure: SpeechFormatStructure,
        filter_empty_speeches: bool = True,
        use_gold_labels: Optional[bool] = None,
    ) -> list[LLMInput]:
        """
        Returns a list of inputs that can be used as rows in an actual training dataset.

        Params:
            row: the row in the dataset (abstraction from our code) that is to be converted into a row
                that can be used by a Trainer object
            config: the configuration for the training run (contains hyperparameters, prompt names, etc)
            skipping_func: function that determines whether a given speech should be excluded from the dataset
                (useful if we want to exclude things like pre-debate judge probabilities)
            is_debater: whether the row is being converted for training a debater (true) or judge (false)
            dataset: the dataset (abstraction from our code) that the row is sampled from
            speech_structure: the format that that the round should be converted to (debate or consultancy)
            use_gold_labels: whether the judge should use gold labels (True) or human judgment (False). Only applicable
                if is_debater is False

        Returns:
            llm_inputs: a list of inputs of type LLMInput that can be easily converted into a dataset that
                the Trainer objects can process.
        """
        llm_class = LLMType[config.llm_type.upper()].get_llm_class()
        llm_inputs = []

        only_judge_has_spoken = True
        previous_speaker_type = SpeakerType.JUDGE
        speeches_so_far = []
        rounds = 1
        for i, speech in enumerate(row.speeches or RowConverter.get_default_speeches()):
            # we want to skip whatever judgment the judge made before the round started
            if only_judge_has_spoken and speech.speaker_type == SpeakerType.JUDGE:
                continue
            only_judge_has_spoken = False

            if speech.speaker_type == SpeakerType.JUDGE and (previous_speaker_type == SpeakerType.DEBATER or not is_debater):
                rounds += 1

            if config.opening_speeches_only and rounds > (1 if is_debater else 2):
                return llm_inputs

            if skipping_func(speech):
                speeches_so_far.append(speech)
                continue

            name = RowConverter.get_speaker_from_speech(speech)
            prompt = RowConverter.generate_prompt_from_speech(
                row=row, speech=speech, config=config, dataset=dataset, speech_structure=speech_structure
            )

            transcript = Transcript(
                name=name,
                prompt=prompt,
                speech_format=(
                    speech_structure.debater_format.get_speech_format(
                        name=name, num_speeches=rounds, use_scratchpad=config.scratchpad_config.use_scratchpad
                    )
                    if is_debater
                    else speech_structure.judge_format.get_speech_format(
                        name=constants.DEFAULT_JUDGE_NAME, num_speeches=(rounds - 1), use_scratchpad=False, flipped=False
                    )
                ),
            )

            if rounds > 1:  # this conditional lets us handle the simultaneity of the first round
                for previous_speech in speeches_so_far:
                    speaker = RowConverter.get_speaker_from_speech(speech=previous_speech)
                    if config.scratchpad_config.use_scratchpad and speaker == name:
                        transcript.add_speech(speaker=speaker, content=previous_speech.scratchpad)
                    transcript.add_speech(speaker=speaker, content=previous_speech.text)

            if config.scratchpad_config.use_scratchpad:
                llm_inputs.append(
                    llm_class.generate_llm_input_from_model_inputs(
                        input_list=transcript.to_model_input(), extra_suffix=speech.scratchpad
                    ).dict()
                )
                transcript.add_speech(speaker=name, content=speech.scratchpad)

            speech_texts = [speech.text]
            if not is_debater:
                if use_gold_labels:
                    speech_texts = ["Winner: Debater_A"] if row.correct_index == 0 else ["Winner: Debater_B"]
                else:
                    winner = "A" if speech.probabilities[0] > 0.5 else "B"
                    loser = "B" if speech.probabilities[0] > 0.5 else "A"
                    max_probability = max(speech.probabilities)
                    speech_texts = [f"Debater_{winner}"] + [
                        f"Debater_{winner}" if max_probability >= 0.75 else f"Debater_{loser}"
                    ]

            local_llm_input_list = [
                llm_class.generate_llm_input_from_model_inputs(
                    input_list=transcript.to_model_input(), extra_suffix=speech_text
                )
                for speech_text in speech_texts
            ]

            if (
                isinstance(local_llm_input_list[0], LLMInput)
                and local_llm_input_list[0].extra_suffix
                and isinstance(local_llm_input_list[0].extra_suffix, str)
                or not filter_empty_speeches
            ):
                llm_inputs.extend([llm_input.dict() for llm_input in local_llm_input_list])
            elif (
                isinstance(local_llm_input_list[0], list)
                and isinstance(local_llm_input_list[0][-1], dict)
                and local_llm_input_list[0][-1].get("role", "") == "assistant"
            ):
                llm_inputs.extend(local_llm_input_list)

            previous_speaker_type = speech.speaker_type
            speeches_so_far.append(speech)
        return llm_inputs

    @classmethod
    def convert_all_speeches_for_debater(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, speech_structure: SpeechFormatStructure
    ) -> list[LLMInput]:
        """Returns a list of inputs that can be used as rows in an actual training dataset that can be
        used to train a debater. See convert_transcript() for more details"""
        return RowConverter.convert_transcript(
            row=row,
            config=config,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.JUDGE,
            is_debater=True,
            dataset=dataset,
            speech_structure=speech_structure,
        )

    @classmethod
    def convert_all_speeches_for_judge(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, speech_structure: SpeechFormatStructure
    ) -> list[dict[str, str]]:
        """Returns a list of inputs that can be used as rows in an actual training dataset that can be
        used to train a judge. See convert_transcript() for more details"""
        return RowConverter.convert_transcript(
            row=row,
            config=config,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.DEBATER,
            is_debater=False,
            dataset=dataset,
            speech_structure=speech_structure,
            use_gold_labels=False,
        )

    @classmethod
    def convert_row(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, speech_structure: SpeechFormatStructure
    ) -> list[dict[str, str]]:
        """Returns a list of inputs that can be used as rows in an actual training dataset. See
        convert_transcript() for more details"""
        if config.target == TrainingTarget.DEBATER:
            return RowConverter.convert_all_speeches_for_debater(
                row=row, config=config, dataset=dataset, speech_structure=speech_structure
            )
        elif config.target == TrainingTarget.JUDGE:
            return RowConverter.convert_all_speeches_for_judge(
                row=row, config=config, dataset=dataset, speech_structure=speech_structure
            )
        else:
            raise Exception(
                f"Tried to train on an ineligible training target of {config.target}. This line should not be reached."
            )
