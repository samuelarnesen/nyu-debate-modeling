from agents import Debater, DebaterUtils, Judge, JudgeUtils, LLMInput, LLMType, Transcript
from data import AnnotatedQualityDebatesDataset, DataRow, DatasetType, RawDataset, SpeakerType, SpeechData, SplitType
from prompts import DynamicPromptParser, Prompt, PromptParser, PromptTag
from train.train_utils import TrainingConfig, TrainingTarget
import utils.constants as constants

from typing import Any, Callable


class RowConverter:
    @classmethod
    def generate_dynamic_prompt_from_speech(
        cls,
        config: TrainingConfig,
        prompt: Prompt,
        row: DataRow,
        speech: SpeechData,
        dataset: AnnotatedQualityDebatesDataset,
        use_title_as_background_text: bool = False,
    ) -> Prompt:
        """Generates a dynamic prompt using the speech. See PromptParser.get_dynamic_prompt() for a more detailed explanation
        on what a dynamic prompt is."""
        return DynamicPromptParser.convert_to_dynamic_prompt(
            dynamic_prompt_file_path=config.prompt_config.dynamic_prompts_config.dynamic_prompts_file_path,
            prompt=prompt,
            prompt_config=PromptParser.convert_data_row_to_default_prompt_config(row=row, position=speech.position),
            dataset=dataset,
            row=row,
            dynamic_prompt_name=config.prompt_config.dynamic_prompts_config.dynamic_prompt_name,
        )

    @classmethod
    def is_dynamic_prompt(cls, config: TrainingConfig, dataset: RawDataset) -> bool:
        """Returns whether the config requires dynamic prompting.
        See PromptParser.get_dynamic_prompt() for a more detailed explanation"""
        return (
            config.prompt_config.use_dynamic_prompt and dataset.get_dataset_type() == DatasetType.ANNOTATED_QUALITY_DEBATES
        )

    @classmethod
    def generate_prompt_from_speech(
        cls, row: DataRow, speech: SpeechData, config: TrainingConfig, dataset: RawDataset
    ) -> Prompt:
        """Constructs a prompt from a given speech and row in the dataset"""
        prompt_config = PromptParser.convert_data_row_to_default_prompt_config(
            row=row, position=speech.position, use_title_as_background_text=config.prompt_config.is_memorized
        )
        prompt = PromptParser.parse(
            prompt_config=prompt_config,
            prompts_file_path=config.prompt_config.file_path,
            name=config.prompt_config.default_prompt_name,
        )

        if RowConverter.is_dynamic_prompt(config=config, dataset=dataset):
            return RowConverter.generate_dynamic_prompt_from_speech(
                config=config,
                prompt=prompt,
                row=row,
                speech=speech,
                dataset=dataset,
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
    def get_dummy_name_for_speaker(cls, name: str) -> str:
        """Returns a placeholder speech (useful for dynamic prompting)"""
        return f"<{name}_Speech>"

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
        use_dummy: bool = False,
        filter_empty_speeches: bool = True,
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
            use_dummy: whether to use a dummy speech instead of the text of a real speech (for dynamic prompting only)

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
            prompt = RowConverter.generate_prompt_from_speech(row=row, speech=speech, config=config, dataset=dataset)

            transcript = Transcript(
                name=name,
                prompt=prompt,
                speech_format=(
                    DebaterUtils.get_speech_format(
                        name=name, num_speeches=rounds, use_scratchpad=config.scratchpad_config.use_scratchpad
                    )
                    if is_debater
                    else JudgeUtils.get_default_speech_format(num_speeches=(rounds - 1))
                ),
            )

            if rounds > 1:  # this conditional lets us handle the simultaneity of the first round
                for previous_speech in speeches_so_far:
                    speaker = RowConverter.get_speaker_from_speech(speech=previous_speech)
                    dummy_text = RowConverter.get_dummy_name_for_speaker(name=speaker)
                    if config.scratchpad_config.use_scratchpad and speaker == name:
                        transcript.add_speech(
                            speaker=speaker, content=previous_speech.scratchpad if not use_dummy else (dummy_text + "\n")
                        )
                    transcript.add_speech(
                        speaker=speaker, content=previous_speech.text if not use_dummy else (dummy_text + "\n")
                    )

            if config.scratchpad_config.use_scratchpad:
                llm_inputs.append(
                    llm_class.generate_llm_input_from_model_inputs(
                        input_list=transcript.to_model_input(), extra_suffix=speech.scratchpad
                    ).dict()
                )
                transcript.add_speech(speaker=name, content=speech.scratchpad if not use_dummy else (dummy_text + "\n"))

            llm_input = llm_class.generate_llm_input_from_model_inputs(
                input_list=transcript.to_model_input(), extra_suffix=speech.text
            )

            if (llm_input.extra_suffix and isinstance(llm_input.extra_suffix, str)) or not filter_empty_speeches:
                llm_inputs.append(llm_input.dict())

            previous_speaker_type = speech.speaker_type
            speeches_so_far.append(speech)
        return llm_inputs

    @classmethod
    def convert_all_speeches_for_debater(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, use_dummy: bool = False
    ) -> list[LLMInput]:
        """Returns a list of inputs that can be used as rows in an actual training dataset that can be
        used to train a debater. See convert_transcript() for more details"""
        return RowConverter.convert_transcript(
            row=row,
            config=config,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.JUDGE,
            is_debater=True,
            dataset=dataset,
            use_dummy=use_dummy,
        )

    @classmethod
    def convert_all_speeches_for_judge(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, use_dummy: bool = False
    ) -> list[dict[str, str]]:
        """Returns a list of inputs that can be used as rows in an actual training dataset that can be
        used to train a judge. See convert_transcript() for more details"""
        return RowConverter.convert_transcript(
            row=row,
            config=config,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.DEBATER,
            is_debater=False,
            dataset=dataset,
            use_dummy=use_dummy,
        )

    @classmethod
    def convert_row(
        cls,
        row: DataRow,
        target: TrainingTarget,
        config: TrainingConfig,
        dataset: RawDataset,
        use_dummy: bool = False,
    ) -> list[dict[str, str]]:
        """Returns a list of inputs that can be used as rows in an actual training dataset. See
        convert_transcript() for more details"""
        if target == TrainingTarget.DEBATER:
            return RowConverter.convert_all_speeches_for_debater(
                row=row, config=config, dataset=dataset, use_dummy=use_dummy
            )
        elif target == TrainingTarget.JUDGE:
            return RowConverter.convert_all_speeches_for_judge(row=row, config=config, dataset=dataset, use_dummy=use_dummy)
        else:
            raise Exception(f"Tried to train on an ineligible training target of {target}. This line should not be reached.")
