from dotenv import load_dotenv
from pydantic import BaseModel

from enum import Enum
from typing import Optional
import argparse
import logging
import os
import sys


class DebateRoundScriptConfig(BaseModel):
    experiment_name: str
    experiment_file_path: str
    transcript_path_prefix: str
    graphs_path_prefix: str
    stats_path_prefix: str
    full_record_path_prefix: str


class ModelRunScriptConfig(BaseModel):
    config_filepath: str
    config_name: str


class TrainType(Enum):
    SFT = 0
    DPO = 1
    PPO = 2
    PRETRAIN = 3
    PROBE = 4
    CUSTOM_KTO = 5


class ScriptUtils:
    @classmethod
    def setup_script(cls):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        load_dotenv()

    @classmethod
    def get_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--local", action="store_true", default=False)
        parser.add_argument("--num_iters", type=int, default=1_000)
        parser.add_argument("--log_level", type=str, default="INFO")
        parser.add_argument("--configuration", type=str, default="")
        parser.add_argument("--test", action="store_true", default=False)  # needed for local testing (optional otherwise)
        parser.add_argument("--load_only", action="store_true", default=False)
        parser.add_argument("--suppress_graphs", action="store_true", default=False)
        parser.add_argument("--local_rank", type=int, default=0)  # needed for multi-GPU training
        parser.add_argument("--start_time", type=str, default="")
        args = parser.parse_args()
        ScriptUtils.set_log_level(args)
        return args

    @classmethod
    def set_log_level(cls, args) -> None:
        os.environ["LOG_LEVEL"] = str(logging.INFO)

        requested = args.log_level.lower()
        specified = None
        if requested == "debug":
            specified = logging.DEBUG
        elif requested == "info":
            specified = logging.INFO
        elif requested == "warn":
            specified = logging.WARN
        elif requested == "error":
            specified = logging.ERROR
        else:
            raise Exception(f"Request log level {requested} is not eligible")

        os.environ["LOG_LEVEL"] = str(specified)

    @classmethod
    def get_debate_round_script_config(cls, args) -> DebateRoundScriptConfig:
        root = os.environ["SRC_ROOT"]
        output_root = os.environ["OUTPUT_ROOT"] if "OUTPUT_ROOT" in os.environ else root
        transcript_path = f"{output_root}outputs/transcripts"
        graphs_path = f"{output_root}outputs/graphs"
        stats_path = f"{output_root}outputs/stats"
        full_record_path = f"{output_root}outputs/runs"
        if args.test:
            experiment_name = args.configuration
            experiment_file_path = f"{root}experiments/configs/test_experiment.yaml"
        else:
            experiment_name = args.configuration
            experiment_file_path = f"{root}experiments/configs/standard_experiment.yaml"
        return DebateRoundScriptConfig(
            experiment_name=experiment_name,
            experiment_file_path=experiment_file_path,
            transcript_path_prefix=transcript_path,
            graphs_path_prefix=graphs_path,
            stats_path_prefix=stats_path,
            full_record_path_prefix=full_record_path,
        )

    @classmethod
    def get_config_filepath(cls, train_type: TrainType) -> str:
        root = os.environ["SRC_ROOT"]
        default_config_dir = "train/configs"
        if train_type == TrainType.SFT:
            return f"{root}/{default_config_dir}/sft_config.yaml"
        elif train_type == TrainType.DPO:
            return f"{root}/{default_config_dir}/dpo_config.yaml"
        elif train_type == TrainType.PPO:
            return f"{root}/{default_config_dir}/ppo_config.yaml"
        elif train_type == TrainType.PRETRAIN:
            return f"{root}/{default_config_dir}/pretrain_config.yaml"
        elif train_type == TrainType.PROBE:
            return f"{root}/{default_config_dir}/probe_config.yaml"
        elif train_type == TrainType.CUSTOM_KTO:
            return f"{root}/{default_config_dir}/custom_kto_config.yaml"
        else:
            raise Exception(f"Train type {train_type} is not recognized")

    @classmethod
    def get_training_run_script_config(cls, args, train_type: TrainType) -> ModelRunScriptConfig:
        return ModelRunScriptConfig(
            config_filepath=ScriptUtils.get_config_filepath(train_type=train_type), config_name=args.configuration
        )
