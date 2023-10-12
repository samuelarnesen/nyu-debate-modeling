from agents.agent import Agent
from agents.model import Model
from agents.models.offline_model import OfflineModel
from agents.prompt import Prompt
from agents.transcript import Transcript
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from typing import Optional, Union


class Debater(Agent):
    def __init__(self, name: str, prompt: Union[Prompt, list[Prompt]], model: Model, num_speeches: int):
        super().__init__(name=name, is_debater=True, prompt=prompt, model=model, num_speeches=num_speeches)

    def generate(self) -> Optional[list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=450, debater_name=self.name)


class OfflineDebater(Debater):
    def __init__(self, debater: Debater, file_path: str, first_debater_prompt: Prompt):
        super().__init__(
            name=debater.name,
            prompt=debater.prompts,
            model=OfflineModel(
                alias=debater.model.alias, is_debater=debater.is_debater, file_path=file_path, prompt=first_debater_prompt
            ),
            num_speeches=debater.num_speeches,
        )
