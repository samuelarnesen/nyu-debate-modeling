from agents.models.llama_model import LlamaInput, LlamaModel
from agents.prompt import Prompt, PromptParser, PromptTag
from agents.transcript import Transcript


from data.data import DataRow, SpeakerType, SpeechData
import utils.constants as constants


class RowConverter:
    """
    This class exists so that we can convert a human debate round into a row into an actual training dataset.
    Despite the only function that's relevant externally being convert_all_speeches(), I'm keeping this as
    a separate class for now in case we want to construct other datasets such as having only opening speeches,
    only rebuttals, or only judge decisions.
    """

    @classmethod
    def generate_prompt_from_speech(
        cls, row: DataRow, speech: SpeechData, prompts_file_path: str, prompt_name: str
    ) -> Prompt:
        prompt_config = PromptParser.convert_data_row_to_default_prompt_config(row=row, position=speech.position)
        prompt = PromptParser.parse(prompts_file_path=prompts_file_path, prompt_config=prompt_config, name=prompt_name)
        return prompt

    @classmethod
    def get_speaker_from_speech(cls, speech: SpeechData) -> str:
        return constants.DEFAULT_DEBATER_A_NAME if speech.position == 0 else constants.DEFAULT_DEBATER_B_NAME

    @classmethod
    def convert_all_speeches(cls, row: DataRow, prompts_file_path: str, prompt_name: str) -> list[dict[str, str]]:
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
                speeches_so_far.append(speech)
                continue

            transcript = Transcript(
                is_debater=True,
                debater_name=constants.DEFAULT_DEBATER_A_NAME if speech.position == 0 else constants.DEFAULT_DEBATER_B_NAME,
                prompt=RowConverter.generate_prompt_from_speech(
                    row=row, speech=speech, prompts_file_path=prompts_file_path, prompt_name=prompt_name
                ),
                num_speeches=rounds,
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
