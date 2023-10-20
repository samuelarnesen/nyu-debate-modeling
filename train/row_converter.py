from agents.debater import Debater
from agents.judge import Judge
from agents.models.llama_model import LlamaInput, LlamaModel
from agents.prompt import Prompt, PromptParser, PromptTag
from agents.transcript import Transcript
from data.data import DataRow, SpeakerType, SpeechData
import utils.constants as constants

from typing import Any, Callable


class RowConverter:
    @classmethod
    def generate_prompt_from_speech(
        cls, row: DataRow, speech: SpeechData, prompts_file_path: str, prompt_name: str
    ) -> Prompt:
        prompt_config = PromptParser.convert_data_row_to_default_prompt_config(row=row, position=speech.position)
        prompt = PromptParser.parse(prompts_file_path=prompts_file_path, prompt_config=prompt_config, name=prompt_name)
        return prompt

    @classmethod
    def get_speaker_from_speech(cls, speech: SpeechData) -> str:
        return (
            constants.DEFAULT_DEBATER_A_NAME
            if speech.position == 0
            else (constants.DEFAULT_DEBATER_B_NAME if speech.position == 1 else constants.DEFAULT_JUDGE_NAME)
        )

    @classmethod
    def convert_transcript(
        cls,
        row: DataRow,
        prompts_file_path: str,
        prompt_name: str,
        skipping_func: Callable[[SpeechData], bool],
        is_debater: bool,
    ):
        llama_inputs = []

        only_judge_has_spoken = True
        round_one_ongoing = True
        previous_speaker_type = SpeakerType.JUDGE
        speeches_so_far = []
        rounds = 1
        for i, speech in enumerate(row.speeches):
            # we want to skip whatever judgment the judge made before the round started
            if only_judge_has_spoken and speech.speaker_type == SpeakerType.JUDGE:
                continue
            only_judge_has_spoken = False

            if speech.speaker_type == SpeakerType.JUDGE and previous_speaker_type == SpeakerType.DEBATER:
                rounds += 1

            if skipping_func(speech):
                speeches_so_far.append(speech)
                continue

            name = RowConverter.get_speaker_from_speech(speech)
            transcript = Transcript(
                name=name,
                prompt=RowConverter.generate_prompt_from_speech(
                    row=row, speech=speech, prompts_file_path=prompts_file_path, prompt_name=prompt_name
                ),
                speech_format=(
                    Debater.generate_default_speech_format(name=name, num_speeches=rounds, include_scratchpad=False)
                    if is_debater
                    else Judge.generate_default_speech_format(num_speeches=rounds)
                ),
            )

            if rounds > 1:  # this conditional lets us handle the simultaneity of the first round
                for previous_speech in speeches_so_far:
                    transcript.add_speech(
                        speaker=RowConverter.get_speaker_from_speech(speech=previous_speech), content=previous_speech.text
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
        cls, row: DataRow, prompts_file_path: str, prompt_name: str
    ) -> list[dict[str, str]]:
        return RowConverter.convert_transcript(
            row=row,
            prompts_file_path=prompts_file_path,
            prompt_name=prompt_name,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.JUDGE,
            is_debater=True,
        )

    @classmethod
    def convert_all_speeches_for_judge(cls, row: DataRow, prompts_file_path: str, prompt_name: str) -> list[dict[str, str]]:
        return RowConverter.convert_transcript(
            row=row,
            prompts_file_path=prompts_file_path,
            prompt_name=prompt_name,
            skipping_func=lambda speech: speech.speaker_type == SpeakerType.DEBATER,
            is_debater=False,
        )
