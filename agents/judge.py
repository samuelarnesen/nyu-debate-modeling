from agents.agent import Agent
from agents.model import Model
from agents.prompt import Prompt
from agents.transcript import Transcript
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from typing import Optional, Union
import random


class Judge(Agent):
    def __init__(self, name: str, prompt: Union[Prompt, list[Prompt]], model: Model, num_speeches: int):
        super().__init__(name=name, is_debater=False, prompt=prompt, model=model, num_speeches=num_speeches)
        self.logger = LoggerUtils.get_default_logger(__name__)

    def generate(self) -> [list[str]]:
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        return self.model.predict(inputs=model_inputs, max_new_tokens=450, decide=False)

    def judge(self) -> list[bool]:
        def validate_responses(predictions) -> None:
            for prediction in filter(
                lambda x: x not in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME], predictions
            ):
                self.logger.warn("Response of {} was invalid".format(prediction))

        batch_reasoning = self.generate()
        for i, reasoning in enumerate(batch_reasoning):
            super().receive_message(speaker=self.name, content=reasoning, idx=i)

        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        predictions = self.model.predict(inputs=model_inputs, max_new_tokens=15, decide=True)

        validate_responses(predictions)

        return [constants.DEFAULT_DEBATER_A_NAME in prediction for prediction in predictions]
