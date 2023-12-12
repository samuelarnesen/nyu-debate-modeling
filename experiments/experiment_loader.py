from agents import (
    BoNDebater,
    BoNJudge,
    Debater,
    DebateRound,
    HumanDebater,
    Judge,
    Model,
    ModelType,
    ModelUtils,
    OfflineDebater,
    QuestionMetadata,
)
from data import DatasetType, LoaderUtils, RawDataLoader, RawDataset, SplitType
from prompts import DynamicPromptParser, Prompt, PromptConfig, PromptParser
from utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import random
import yaml

from enum import Enum
from typing import Optional
import itertools


class PromptLoadingConfig(BaseModel):
    file_path: str
    default_prompt_name: str
    dynamic_prompts_file_path: Optional[str]
    dynamic_prompt_name: Optional[str]


class AgentConfig(BaseModel):
    model_type: str
    model_file_path: Optional[str]
    alias: str
    use_scratchpad: bool = False
    override_prompt: Optional[str]
    greedy: bool = True
    is_memorized: bool = False


class AgentsConfig(BaseModel):
    debaters: list[AgentConfig]
    judge: AgentConfig


class DatasetConfig(BaseModel):
    dataset_type: str
    full_dataset_file_path: Optional[str]
    train_file_path: Optional[str]
    val_file_path: Optional[str]
    test_file_path: Optional[str]
    annotations_file_path: Optional[str]
    split_type: Optional[str]


class TopicConfigType(Enum):
    HARD_CODED = 1
    FROM_DATASET = 2


class TopicConfig(BaseModel):
    topic_type: str
    topic: Optional[str]
    positions: Optional[tuple[str, str]]


class OfflineConfig(BaseModel):
    debaters: list[str]
    file_path: str


class HumanConfig(BaseModel):
    debaters: list[str]


class BoNConfig(BaseModel):
    count: int
    prompts: Optional[list[str]]


class ExperimentConfig(BaseModel):
    topic_config: TopicConfig
    word_limit: int
    batch_size: int
    num_speeches: int
    flip: bool = False
    prompt_config: PromptLoadingConfig
    agents: AgentsConfig
    dataset: DatasetConfig
    offline: Optional[OfflineConfig]
    best_of_n: Optional[BoNConfig]
    human: Optional[HumanConfig]
    annotations_classifier_file_path: Optional[str]


class ExperimentLoader:
    @classmethod
    def merge_debate_rounds(cls, debate_rounds: list[DebateRound]) -> DebateRound:
        """Combines the listed debate rounds into one (batched) debate round"""

        def validate() -> None:
            for debate_round in debate_rounds:
                if (
                    debate_rounds[0].first_debater.model != debate_round.first_debater.model
                    or debate_rounds[0].second_debater.model != debate_round.second_debater.model
                    or debate_rounds[0].judge.model != debate_round.judge.model
                    or debate_rounds[0].first_debater.name != debate_round.first_debater.name
                    or debate_rounds[0].second_debater.name != debate_round.second_debater.name
                    or debate_rounds[0].judge.name != debate_round.judge.name
                ):
                    raise Exception("Cannot merge rounds of across models or names")

        validate()
        first_debater_prompts = []
        second_debater_prompts = []
        judge_prompts = []
        metadata_list = []
        for debate_round in debate_rounds:
            for prompt in debate_round.first_debater.prompts:
                first_debater_prompts.append(prompt)
            for prompt in debate_round.second_debater.prompts:
                second_debater_prompts.append(prompt)
            for prompt in debate_round.judge.prompts:
                judge_prompts.append(prompt)
            for metadata in debate_round.metadata:
                metadata_list.append(metadata)

        first_debater = debate_rounds[0].first_debater.copy()
        second_debater = debate_rounds[0].second_debater.copy()
        judge = debate_rounds[0].judge.copy()

        return DebateRound(first_debater=first_debater, second_debater=second_debater, judge=judge, metadata=metadata_list)

    @classmethod
    def create_dataset(cls, experiment: ExperimentConfig) -> RawDataset:
        dataset_config = experiment.dataset
        dataset_type = DatasetType[dataset_config.dataset_type.upper()]
        loader_cls = LoaderUtils.get_loader_type(dataset_type)
        return loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
            annotations_file_path=dataset_config.annotations_file_path,
        )

    @classmethod
    def get_split(cls, experiment: ExperimentConfig) -> SplitType:
        return SplitType[experiment.dataset.split_type.upper()] if experiment.dataset.split_type else SplitType.TRAIN

    @classmethod
    def create_debate_rounds_for_combination(
        cls,
        experiment: ExperimentConfig,
        dataset: RawDataset,
        split_type: SplitType,
        debater_idxs: tuple[int, int],
        count: int,
        model_cache: Optional[dict[str, Model]] = None,
    ) -> tuple[list[DebateRound], dict[str, Model]]:
        """
        Creates a set of debate round for the specific debaters listed in debater_idxs.

        Params:
            experiment: the configuration for the set of debate rounds
            dataset: the dataset from which one draws the questions and positions
            split_type: whether the quesitons/positions should be sampled from the train, val, or test sets
            debater_idxs: which pair of debaters from the experiment config should we be creating debate rounds for
            count: the number of rounds to create
            model_cache: a dictionary mapping a model alias (string) to a model. This is useful so that we do not
                instantiate the same model multiple times if this function is called multiple times in a larger
                tournament (it is not needed if you only invoke the function on one pair of models).

        Returns:
            batched_rounds: a list of debate rounds based on the inputted configuration
            model_cache: a cached set of the models used in these debate rounds (useful if you invoke this
                function again).
        """

        # create logger
        logger = LoggerUtils.get_default_logger(__name__)

        # create debater models
        debater_one_model_type = ModelType[experiment.agents.debaters[debater_idxs[0]].model_type.upper()]
        debater_two_model_type = ModelType[experiment.agents.debaters[debater_idxs[1]].model_type.upper()]
        judge_model_type = ModelType[experiment.agents.judge.model_type.upper()]
        debater_one_model_path = experiment.agents.debaters[debater_idxs[0]].model_file_path
        debater_two_model_path = experiment.agents.debaters[debater_idxs[1]].model_file_path
        judge_model_path = experiment.agents.judge.model_file_path
        logger.debug(f"Instantiating a {debater_one_model_type} from {debater_one_model_path}")

        if not model_cache:
            model_cache = {}

        debater_one_model = (
            ModelUtils.instantiate_model(
                model_type=debater_one_model_type,
                file_path=debater_one_model_path,
                is_debater=True,
                alias=experiment.agents.debaters[debater_idxs[0]].alias,
                greedy=experiment.agents.debaters[debater_idxs[0]].greedy,
            )
            if f"{debater_one_model_type}_{debater_one_model_path}" not in model_cache
            else model_cache[f"{debater_one_model_type}_{debater_one_model_path}"].copy(
                alias=experiment.agents.debaters[debater_idxs[0]].alias, is_debater=True
            )
        )
        model_cache[f"{debater_one_model_type}_{debater_one_model_path}"] = debater_one_model

        debater_two_model = (
            ModelUtils.instantiate_model(
                model_type=debater_two_model_type,
                file_path=debater_two_model_path,
                is_debater=True,
                alias=experiment.agents.debaters[debater_idxs[1]].alias,
                greedy=experiment.agents.debaters[debater_idxs[1]].greedy,
            )
            if f"{debater_two_model_type}_{debater_two_model_path}" not in model_cache
            else model_cache[f"{debater_two_model_type}_{debater_two_model_path}"].copy(
                alias=experiment.agents.debaters[debater_idxs[1]].alias,
                is_debater=True,
                greedy=experiment.agents.debaters[debater_idxs[1]].greedy,
            )
        )
        model_cache[f"{debater_two_model_type}_{debater_two_model_path}"] = debater_two_model

        judge_model = (
            ModelUtils.instantiate_model(
                model_type=judge_model_type,
                file_path=judge_model_path,
                is_debater=False,
                alias=experiment.agents.judge.alias,
            )
            if f"{judge_model_type}_{judge_model_path}" not in model_cache
            else model_cache[f"{judge_model_type}_{judge_model_path}"].copy(
                alias=experiment.agents.judge.alias, is_debater=False
            )
        )
        model_cache[f"{judge_model_type}_{judge_model_path}"] = judge_model

        # create debate rounds
        rounds = []
        topic_config_type = TopicConfigType[experiment.topic_config.topic_type.upper()]
        for i in range(count):
            if topic_config_type == TopicConfigType.FROM_DATASET:
                example = dataset.get_example(idx=i, split=split_type)
                topic = example.question
                position = example.positions[0]
                opponent_position = example.positions[1]
                background_text = example.background_text
                title = example.story_title
                correct_index = example.correct_index
                speeches = example.speeches
            elif topic_config_type == TopicConfigType.HARD_CODED:
                topic = experiment.topic_config.topic
                position = experiment.topic_config.positions[0]
                opponent_position = experiment.topic_config.positions[1]
                background_text = constants.DEFAULT_BACKGROUND_TEXT
                title = ""
                correct_index = None
                speeches = []
            else:
                raise Exception(f"Topic config type {topic_config_type} is not recognized")

            config_a = PromptConfig(
                name=constants.DEFAULT_DEBATER_A_NAME,
                opponent_name=constants.DEFAULT_DEBATER_B_NAME,
                word_limit=experiment.word_limit,
                position=position,
                opponent_position=opponent_position,
                topic=topic,
                background_text=background_text if not experiment.agents.debaters[debater_idxs[0]].is_memorized else title,
            )

            config_b = PromptConfig(
                name=constants.DEFAULT_DEBATER_B_NAME,
                opponent_name=constants.DEFAULT_DEBATER_A_NAME,
                word_limit=experiment.word_limit,
                position=opponent_position,
                opponent_position=position,
                topic=topic,
                background_text=background_text if not experiment.agents.debaters[debater_idxs[1]].is_memorized else title,
            )

            prompt_a = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.agents.debaters[debater_idxs[0]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_a = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.agents.debaters[debater_idxs[1]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_b = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_b,
                name=experiment.agents.debaters[debater_idxs[1]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_b = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.agents.debaters[debater_idxs[0]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_judge = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.prompt_config.default_prompt_name,
            )

            if experiment.prompt_config.dynamic_prompts_file_path and experiment.prompt_config.dynamic_prompt_name:
                prompt_a = DynamicPromptParser.convert_to_dynamic_prompt(
                    dynamic_prompt_file_path=experiment.prompt_config.dynamic_prompts_file_path,
                    prompt=prompt_a,
                    prompt_config=config_a,
                    dataset=dataset,
                    index=i,
                    split=split_type,
                    row=example,
                    dynamic_prompt_name=experiment.prompt_config.dynamic_prompt_name,
                )

                prompt_b = DynamicPromptParser.convert_to_dynamic_prompt(
                    dynamic_prompt_file_path=experiment.prompt_config.dynamic_prompts_file_path,
                    prompt=prompt_b,
                    prompt_config=config_b,
                    dataset=dataset,
                    index=i,
                    split=split_type,
                    row=example,
                    dynamic_prompt_name=experiment.prompt_config.dynamic_prompt_name,
                )

            debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=prompt_a,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.agents.debaters[debater_idxs[0]].use_scratchpad,
            )

            debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=prompt_b,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.agents.debaters[debater_idxs[1]].use_scratchpad,
            )

            judge = Judge(
                name=constants.DEFAULT_JUDGE_NAME,
                prompt=prompt_judge,
                model=judge_model,
                num_speeches=experiment.num_speeches,
            )

            debate_round = DebateRound(
                first_debater=debater_a,
                second_debater=debater_b,
                judge=judge,
                metadata=[
                    QuestionMetadata(
                        first_debater_correct=correct_index == 0,
                        question_idx=i,
                        background_text=background_text,
                        split=split_type,
                    )
                ],
            )

            flipped_debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=flipped_prompt_a,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.agents.debaters[debater_idxs[1]].use_scratchpad,
            )

            flipped_debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=flipped_prompt_b,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.agents.debaters[debater_idxs[0]].use_scratchpad,
            )

            flipped_round = DebateRound(
                first_debater=flipped_debater_a,
                second_debater=flipped_debater_b,
                judge=judge,
                metadata=[
                    QuestionMetadata(
                        first_debater_correct=correct_index == 0,
                        question_idx=i,
                        background_text=background_text,
                        split=split_type,
                    )
                ],
            )

            if experiment.best_of_n:
                if experiment.num_speeches > 1:
                    raise Exception("For now, there can only be 1 speech when doing BoN")

                debater_a_prompts = []
                debater_b_prompts = []
                logger.debug(
                    f"Using {(len(experiment.best_of_n.prompts) if experiment.best_of_n.prompts else 0)} new prompts"
                )
                for prompt_name in experiment.best_of_n.prompts or [experiment.prompt_config.default_prompt_name]:
                    for prompt_list, config in zip([debater_a_prompts, debater_b_prompts], [config_a, config_b]):
                        prompt_list.append(
                            PromptParser.parse(
                                prompts_file_path=experiment.prompt_config.file_path,
                                prompt_config=config,
                                name=prompt_name,
                            )
                        )

                debate_round.set_judge(BoNJudge(judge=debate_round.judge, n=experiment.best_of_n.count, debater_a=True))
                flipped_round.set_judge(BoNJudge(judge=flipped_round.judge, n=experiment.best_of_n.count, debater_a=False))
                debate_round.set_first_debater(
                    BoNDebater(
                        debater=debate_round.first_debater,
                        n=experiment.best_of_n.count,
                        prompts=debater_a_prompts,
                        evaluated=True,
                    )
                )
                flipped_round.set_first_debater(
                    BoNDebater(debater=flipped_round.first_debater, n=experiment.best_of_n.count, evaluated=False)
                )
                debate_round.set_second_debater(
                    BoNDebater(debater=debate_round.second_debater, n=experiment.best_of_n.count, evaluated=False)
                )
                flipped_round.set_second_debater(
                    BoNDebater(
                        debater=flipped_round.second_debater,
                        n=experiment.best_of_n.count,
                        prompts=debater_b_prompts,
                        evaluated=True,
                    )
                )

            if experiment.offline:
                if debate_round.first_debater.model.alias in experiment.offline.debaters:
                    debate_round.set_first_debater(
                        OfflineDebater(
                            debater=debate_round.first_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                            round_idx=(i * 2) if experiment.flip else i,
                        )
                    )
                    flipped_round.set_second_debater(
                        OfflineDebater(
                            debater=flipped_round.second_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                            round_idx=((i * 2) + 1) if experiment.flip else i,
                        )
                    )
                if debate_round.second_debater.model.alias in experiment.offline.debaters:
                    debate_round.set_second_debater(
                        OfflineDebater(
                            debater=debate_round.second_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                            round_idx=(i * 2) if experiment.flip else i,
                        )
                    )
                    flipped_round.set_first_debater(
                        OfflineDebater(
                            debater=flipped_round.first_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                            round_idx=((i * 2) + 1) if experiment.flip else i,
                        )
                    )
            if experiment.human:
                if debate_round.first_debater.model.alias in experiment.human.debaters:
                    debate_round.set_first_debater(HumanDebater(debater=debate_round.first_debater, speeches=speeches))
                    flipped_round.set_first_debater(HumanDebater(debater=flipped_round.first_debater, speeches=speeches))
                if debate_round.second_debater.model.alias in experiment.human.debaters:
                    debate_round.set_second_debater(HumanDebater(debater=debate_round.second_debater, speeches=speeches))
                    flipped_round.set_second_debater(HumanDebater(debater=flipped_round.second_debater, speeches=speeches))

            rounds.append(debate_round)
            if experiment.flip:
                rounds.append(flipped_round)

        if len(rounds) <= 1 or experiment.best_of_n:
            return rounds, model_cache

        # batches the debate rounds for efficient generation
        batched_rounds = []
        current_normal_batch = []
        current_flipped_batch = []
        for i, debate_round in enumerate(rounds):
            if i % 2 == 0:
                current_normal_batch.append(debate_round)
                if len(current_normal_batch) == experiment.batch_size:
                    batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_normal_batch))
                    current_normal_batch = []
            else:
                current_flipped_batch.append(debate_round)
                if len(current_flipped_batch) == experiment.batch_size:
                    batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_flipped_batch))
                    current_flipped_batch = []

        if current_normal_batch:
            batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_normal_batch))
        if current_flipped_batch:
            batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_flipped_batch))

        return batched_rounds, model_cache

    @classmethod
    def get_debater_combinations(cls, experiment: ExperimentConfig):
        """Returns all the combinations of debaters that would need to debate each other in a round robin tournament"""
        all_idxs = [i for i in range(len(experiment.agents.debaters))]
        if len(all_idxs) < 2:
            raise Exception("At least 2 debaters must be defined")

        return [elem for elem in itertools.combinations(all_idxs, r=2)]

    @classmethod
    def generate_debate_rounds(
        cls, experiment_file_path: str, name: str, count: int = 1
    ) -> tuple[list[DebateRound], ExperimentConfig]:
        """
        Generates a list of debate rounds with the given configuration

        Params:
            experiment_file_path: path to the file of the experiment config
            name: the name of the specific config within the broader config file
            count: the number of debate rounds to create

        Returns:
            all_rounds: a list of (batched) debate rounds constructed using the config
            experiment: the configuration used to create the debate rounds
        """
        # create experiment config
        with open(experiment_file_path) as f:
            loaded_yaml = yaml.safe_load(f)
        experiment = ExperimentConfig(**loaded_yaml[name])

        # create dataset
        dataset = ExperimentLoader.create_dataset(experiment)
        split_type = ExperimentLoader.get_split(experiment)

        all_rounds = []
        model_cache = {}
        for combination in ExperimentLoader.get_debater_combinations(experiment=experiment):
            rounds, model_cache = ExperimentLoader.create_debate_rounds_for_combination(
                experiment=experiment,
                dataset=dataset,
                split_type=split_type,
                debater_idxs=combination,
                count=count,
                model_cache=model_cache,
            )
            all_rounds.extend(rounds)

        return all_rounds, experiment
