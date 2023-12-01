from agents import Debater, DebaterUtils, Judge, JudgeUtils, LlamaInput, LlamaModel, Transcript
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
        index: int,
    ) -> Prompt:
        return DynamicPromptParser.convert_to_dynamic_prompt(
            dynamic_prompt_file_path=config.prompt_config.dynamic_prompts_file_path,
            prompt=prompt,
            prompt_config=PromptParser.convert_data_row_to_default_prompt_config(row=row, position=speech.position),
            dataset=dataset,
            index=index,
            split=SplitType.TRAIN,
            row=row,
            dynamic_prompt_name=config.prompt_config.dynamic_prompt_name,
        )

    @classmethod
    def is_dynamic_prompt(cls, config: TrainingConfig, dataset: RawDataset) -> bool:
        return (
            config.prompt_config.dynamic_prompts_file_path
            and config.prompt_config.dynamic_prompt_name
            and dataset.get_dataset_type() == DatasetType.ANNOTATED_QUALITY_DEBATES
        )

    @classmethod
    def generate_prompt_from_speech(
        cls, row: DataRow, speech: SpeechData, config: TrainingConfig, dataset: RawDataset, index: int
    ) -> Prompt:
        prompt_config = PromptParser.convert_data_row_to_default_prompt_config(row=row, position=speech.position)
        prompt = PromptParser.parse(
            prompts_file_path=config.prompt_config.prompts_file_path,
            prompt_config=prompt_config,
            name=config.prompt_config.prompt_name,
        )

        if RowConverter.is_dynamic_prompt(config=config, dataset=dataset):
            return RowConverter.generate_dynamic_prompt_from_speech(
                config=config,
                prompt=prompt,
                row=row,
                speech=speech,
                dataset=dataset,
                index=index,
            )
        return prompt

    @classmethod
    def get_speaker_from_speech(cls, speech: SpeechData) -> str:
        return (
            constants.DEFAULT_DEBATER_A_NAME
            if speech.position == 0
            else (constants.DEFAULT_DEBATER_B_NAME if speech.position == 1 else constants.DEFAULT_JUDGE_NAME)
        )

    @classmethod
    def get_dummy_name_for_speaker(cls, name: str) -> str:
        return f"<{name}_Speech>"

    @classmethod
    def get_default_speeches(cls) -> list[SpeechData]:
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
        index: int = 0,
        use_dummy: bool = False,
    ):
        llama_inputs = []

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
                return llama_inputs

            if skipping_func(speech):
                speeches_so_far.append(speech)
                continue

            name = RowConverter.get_speaker_from_speech(speech)
            prompt = RowConverter.generate_prompt_from_speech(
                row=row, speech=speech, config=config, dataset=dataset, index=index
            )

            transcript = Transcript(
                name=name,
                prompt=prompt,
                speech_format=(
                    DebaterUtils.get_default_speech_format(name=name, num_speeches=rounds, use_scratchpad=False)
                    if is_debater
                    else JudgeUtils.get_default_speech_format(num_speeches=(rounds - 1))
                ),
            )

            if rounds > 1:  # this conditional lets us handle the simultaneity of the first round
                for previous_speech in speeches_so_far:
                    speaker = RowConverter.get_speaker_from_speech(speech=previous_speech)
                    dummy_text = RowConverter.get_dummy_name_for_speaker(name=speaker)
                    transcript.add_speech(
                        speaker=speaker, content=previous_speech.text if not use_dummy else (dummy_text + "\n")
                    )

            llama_inputs.append(
                LlamaModel.generate_llama_input_from_model_inputs(
                    input_list=transcript.to_model_input(), extra_suffix=speech.text
                ).dict()
            )

            previous_speaker_type = speech.speaker_type
            speeches_so_far.append(speech)
        return llama_inputs

    @classmethod
    def convert_all_speeches_for_debater(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, index: int = 0, use_dummy: bool = False
    ) -> list[dict[str, str]]:
        return RowConverter.convert_transcript(
            row=row,
            config=config,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.JUDGE,
            is_debater=True,
            dataset=dataset,
            index=index,
            use_dummy=use_dummy,
        )

    @classmethod
    def convert_all_speeches_for_judge(
        cls, row: DataRow, config: TrainingConfig, dataset: RawDataset, index: int = 0, use_dummy: bool = False
    ) -> list[dict[str, str]]:
        return RowConverter.convert_transcript(
            row=row,
            config=config,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.DEBATER,
            is_debater=False,
            dataset=dataset,
            index=index,
            use_dummy=use_dummy,
        )

    @classmethod
    def convert_row(
        cls,
        row: DataRow,
        target: TrainingTarget,
        config: TrainingConfig,
        dataset: RawDataset,
        index: int = 0,
        use_dummy: bool = False,
    ) -> list[dict[str, str]]:
        if target == TrainingTarget.DEBATER:
            return RowConverter.convert_all_speeches_for_debater(
                row=row, config=config, dataset=dataset, index=index, use_dummy=use_dummy
            )
        elif target == TrainingTarget.JUDGE:
            return RowConverter.convert_all_speeches_for_judge(
                row=row, config=config, dataset=dataset, index=index, use_dummy=use_dummy
            )
        else:
            raise Exception(f"Tried to train on an ineligible training target of {target}. This line should not be reached.")
