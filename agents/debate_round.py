from agents.agent import Debater, Judge
from agents.prompt import Prompt, PromptConfig, PromptParser
from data.data import Dataset, SplitType
from utils.logger_utils import LoggerUtils

LOGGER = LoggerUtils.get_default_logger(__name__)


class DebateRound:
    def __init__(self, first_debater: Debater, second_debater: Debater, judge: Judge, dataset: Dataset):
        self.first_debater = first_debater
        self.second_debater = second_debater
        self.judge = judge
        self.debaters = [self.first_debater, self.second_debater]
        self.participants = [self.first_debater, self.second_debater, self.judge]
        self.dataset = dataset

    def reset(self):
        for participant in self.participants:
            participant.reset()

    def run(self, split: SplitType = SplitType.TRAIN, num_speeches=3) -> bool:
        self.first_debater.reset()
        self.second_debater.reset()
        self.judge.reset()

        # TODO: the dataset fetching isn't needed for now but it's good scaffolding
        example = self.dataset.get_example(split=split)
        for speech_num in range(num_speeches):
            for debater in self.debaters:
                response = debater.generate()
                for participant in self.participants:
                    participant.receive_message(speaker=debater.name, content=response)
                LOGGER.debug(f"{debater.name}: {response}\n")

        response, debater_one_wins = self.judge.generate()
        LOGGER.debug(f"{self.judge.name}: {response}\n")
        LOGGER.debug(f"Winner is {'Debater_One' if debater_one_wins else 'Debater_Two'}")

        return debater_one_wins
