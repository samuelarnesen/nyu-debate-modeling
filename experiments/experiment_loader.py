from agents import (
    AgentConfig,
    BestOfNDebater,
    Debater,
    DebateRound,
    HumanDebater,
    Judge,
    Model,
    ModelSettings,
    ModelType,
    ModelUtils,
    OfflineModelHelper,
    QuestionMetadata,
    ServedModel,
)
from data import DatasetConfig, DatasetType, LoaderUtils, RawDataLoader, RawDataset, SplitType
from prompts import DynamicPromptParser, Prompt, PromptConfig, PromptLoadingConfig, PromptParser
from utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import random
import yaml

from enum import Enum
from typing import Optional
import itertools


class AgentsConfig(BaseModel):
    debaters: list[AgentConfig]
    judge: AgentConfig


class ExperimentConfig(BaseModel):
    batch_size: int
    num_speeches: int
    flip: bool = False
    prompt_config: PromptLoadingConfig = PromptLoadingConfig()
    agents: AgentsConfig
    dataset: DatasetConfig
    annotations_classifier_file_path: Optional[str]
    enable_self_debate: bool = False
    previous_run_to_replicate_path: Optional[str]


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

        first_debater = debate_rounds[0].first_debater.copy(prompts=first_debater_prompts)
        second_debater = debate_rounds[0].second_debater.copy(prompts=second_debater_prompts)
        judge = debate_rounds[0].judge.copy(prompts=judge_prompts)

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
            supplemental_file_paths=dataset_config.supplemental_file_paths,
        )

    @classmethod
    def get_split(cls, experiment: ExperimentConfig) -> SplitType:
        return SplitType[experiment.dataset.split_type.upper()] if experiment.dataset.split_type else SplitType.TRAIN

    @classmethod
    def get_model_id(cls, model_settings: ModelSettings):
        return f"{model_settings.model_type}_{model_settings.model_file_path}"

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

        debater_one_model_id = ExperimentLoader.get_model_id(experiment.agents.debaters[debater_idxs[0]].model_settings)
        debater_two_model_id = ExperimentLoader.get_model_id(experiment.agents.debaters[debater_idxs[1]].model_settings)
        judge_model_id = ExperimentLoader.get_model_id(experiment.agents.judge.model_settings)

        if not model_cache:
            model_cache = {}

        debater_one_model = (
            ModelUtils.instantiate_model(
                model_settings=experiment.agents.debaters[debater_idxs[0]].model_settings,
                is_debater=True,
            )
            if debater_one_model_id not in model_cache
            else model_cache[debater_one_model_id].copy(
                alias=experiment.agents.debaters[debater_idxs[0]].model_settings.alias, is_debater=True
            )
        )
        if debater_one_model:
            model_cache[debater_one_model_id] = debater_one_model

        debater_two_model = (
            ModelUtils.instantiate_model(
                model_settings=experiment.agents.debaters[debater_idxs[1]].model_settings,
                is_debater=True,
            )
            if debater_two_model_id not in model_cache
            else model_cache[debater_two_model_id].copy(
                alias=experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                is_debater=True,
                nucleus=experiment.agents.debaters[debater_idxs[1]].model_settings.nucleus,
            )
        )
        if debater_two_model:
            model_cache[debater_two_model_id] = debater_two_model

        judge_model = (
            ModelUtils.instantiate_model(
                model_settings=experiment.agents.judge.model_settings,
                is_debater=False,
            )
            if judge_model_id not in model_cache
            else model_cache[judge_model_id].copy(alias=experiment.agents.judge.model_settings.alias, is_debater=False)
        )
        if judge_model:
            model_cache[judge_model_id] = judge_model

        # instantiates offline model helper
        offline_model_helper = None
        first_offline_file_path = experiment.agents.debaters[debater_idxs[0]].model_settings.offline_file_path
        second_offline_file_path = experiment.agents.debaters[debater_idxs[1]].model_settings.offline_file_path
        is_offline = first_offline_file_path or second_offline_file_path or experiment.previous_run_to_replicate_path
        if is_offline:
            if (first_offline_file_path and second_offline_file_path) and (
                first_offline_file_path != second_offline_file_path
            ):
                raise Exception("Offline file paths must be the same")
            path = first_offline_file_path or second_offline_file_path
            if path and experiment.previous_run_to_replicate_path:
                raise Exception("Offline file paths must be the same")
            path = path or experiment.previous_run_to_replicate_path
            offline_model_helper = OfflineModelHelper(file_path_prefix=path, dataset=dataset)

        # create debate rounds
        rounds = []
        for i in range(count):
            if experiment.prompt_config.use_hardcoded_topics:
                topic = experiment.prompt_config.hardcoded_topic_config.topic
                position = experiment.prompt_config.hardcoded_topic_config.positions[0]
                opponent_position = experiment.prompt_config.hardcoded_topic_config.positions[1]
                background_text = constants.DEFAULT_BACKGROUND_TEXT
                title = ""
                correct_index = None
                speeches = []
            else:
                example = (
                    dataset.get_example(idx=i, split=split_type)
                    if not offline_model_helper
                    else offline_model_helper.get_example(idx=i, split_type=split_type)
                )
                topic = example.question
                position = example.positions[0]
                opponent_position = example.positions[1]
                background_text = example.background_text
                title = example.story_title
                correct_index = example.correct_index
                speeches = example.speeches
            debate_identifier = f"{title}_{topic}"

            config_a = PromptConfig(
                name=constants.DEFAULT_DEBATER_A_NAME,
                opponent_name=constants.DEFAULT_DEBATER_B_NAME,
                position=position,
                opponent_position=opponent_position,
                topic=topic,
                background_text=background_text if not experiment.prompt_config.is_memorized else title,
            )

            config_b = PromptConfig(
                name=constants.DEFAULT_DEBATER_B_NAME,
                opponent_name=constants.DEFAULT_DEBATER_A_NAME,
                position=opponent_position,
                opponent_position=position,
                topic=topic,
                background_text=background_text if not experiment.prompt_config.is_memorized else title,
            )

            prompt_a = PromptParser.parse(
                prompt_config=config_a,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[0]].model_settings.override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_a = PromptParser.parse(
                prompt_config=config_a,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[1]].model_settings.override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_b = PromptParser.parse(
                prompt_config=config_b,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[1]].model_settings.override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_b = PromptParser.parse(
                prompt_config=config_b,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[0]].model_settings.override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_judge = PromptParser.parse(
                prompt_config=config_a,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.prompt_config.default_prompt_name,
            )

            if experiment.prompt_config.use_dynamic_prompt:
                prompt_a = DynamicPromptParser.convert_to_dynamic_prompt(
                    prompt=prompt_a,
                    prompt_config=config_a,
                    dataset=dataset,
                    row=example,
                    dynamic_prompt_file_path=experiment.prompt_config.dynamic_prompts_file_path,
                    dynamic_prompt_name=experiment.prompt_config.dynamic_prompt_name,
                )

                prompt_b = DynamicPromptParser.convert_to_dynamic_prompt(
                    prompt=prompt_b,
                    prompt_config=config_b,
                    dataset=dataset,
                    row=example,
                    dynamic_prompt_file_path=experiment.prompt_config.dynamic_prompts_file_path,
                    dynamic_prompt_name=experiment.prompt_config.dynamic_prompt_name,
                )

            debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=prompt_a,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.debaters[debater_idxs[0]].scratchpad,
            )

            debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=prompt_b,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.debaters[debater_idxs[1]].scratchpad,
            )

            judge = Judge(
                name=constants.DEFAULT_JUDGE_NAME,
                prompt=prompt_judge,
                model=judge_model,
                num_speeches=experiment.num_speeches,
                chain_of_thought=experiment.agents.judge.scratchpad.use_scratchpad,
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
                        question=topic,
                        first_debater_answer=position,
                        second_debater_answer=opponent_position,
                        debate_identifier=debate_identifier,
                    )
                ],
            )

            flipped_debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=flipped_prompt_a,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.debaters[debater_idxs[1]].scratchpad,
            )

            flipped_debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=flipped_prompt_b,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.debaters[debater_idxs[0]].scratchpad,
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
                        question=topic,
                        first_debater_answer=opponent_position,
                        second_debater_answer=position,
                        debate_identifier=debate_identifier,
                    )
                ],
            )

            if first_offline_file_path:
                debate_round.first_debater.model = offline_model_helper.create_offline_model(
                    alias=experiment.agents.debaters[debater_idxs[0]].model_settings.alias,
                    debater_name=debate_round.first_debater.name,
                    idx=i,
                    best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                )
                flipped_round.second_debater.model = offline_model_helper.create_offline_model(
                    alias=experiment.agents.debaters[debater_idxs[0]].model_settings.alias,
                    debater_name=debate_round.second_debater.name,
                    idx=i,
                    best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                )
            if second_offline_file_path:
                debate_round.second_debater.model = offline_model_helper.create_offline_model(
                    alias=experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                    debater_name=flipped_round.second_debater.name,
                    idx=i,
                    best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                )
                flipped_round.first_debater.model = offline_model_helper.create_offline_model(
                    alias=experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                    debater_name=flipped_round.first_debater.name,
                    idx=i,
                    best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                )

            if experiment.agents.debaters[debater_idxs[0]].best_of_n and not first_offline_file_path:
                debate_round.set_first_debater(
                    BestOfNDebater(
                        debater=debate_round.first_debater,
                        opposing_debater=debate_round.second_debater,
                        judge=debate_round.judge,
                        best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                    )
                )
                flipped_round.set_second_debater(
                    BestOfNDebater(
                        debater=flipped_round.second_debater,
                        opposing_debater=flipped_round.first_debater,
                        judge=debate_round.judge,
                        best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                    )
                )
            if experiment.agents.debaters[debater_idxs[1]].best_of_n and not second_offline_file_path:
                debate_round.set_second_debater(
                    BestOfNDebater(
                        debater=debate_round.second_debater,
                        opposing_debater=debate_round.first_debater,
                        judge=debate_round.judge,
                        best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                    )
                )
                flipped_round.set_first_debater(
                    BestOfNDebater(
                        debater=flipped_round.first_debater,
                        opposing_debater=flipped_round.second_debater,
                        judge=debate_round.judge,
                        best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                    )
                )

            if experiment.agents.debaters[debater_idxs[0]].model_settings.is_human:
                debate_round.set_first_debater(HumanDebater(debater=debate_round.first_debater, speeches=speeches))
                flipped_round.set_second_debater(HumanDebater(debater=flipped_round.second_debater, speeches=speeches))
            if experiment.agents.debaters[debater_idxs[1]].model_settings.is_human:
                debate_round.set_second_debater(HumanDebater(debater=debate_round.second_debater, speeches=speeches))
                flipped_round.set_first_debater(HumanDebater(debater=flipped_round.first_debater, speeches=speeches))

            rounds.append(debate_round)
            if experiment.flip and flipped_round.first_debater.model.alias != flipped_round.second_debater.model.alias:
                rounds.append(flipped_round)

        if len(rounds) <= 1:
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
        if not experiment.agents or not experiment.agents.debaters or len(experiment.agents.debaters) < 1:
            raise Exception("At least 1 debater must be defined")
        all_idxs = [i for i in range(len(experiment.agents.debaters))] if len(experiment.agents.debaters) > 1 else [0, 0]
        all_debater_idxs = [elem for elem in itertools.combinations(all_idxs, r=2)]
        if experiment.enable_self_debate and len(experiment.agents.debaters) > 1:
            all_debater_idxs += [(idx, idx) for idx in all_idxs]
        return all_debater_idxs

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
        name = name or [key for key in loaded_yaml][0]
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
