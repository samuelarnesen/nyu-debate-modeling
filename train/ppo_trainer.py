from __future__ import annotations

from agents import OpenAIModel, RandomModel, SpeechStructure
from data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, LoggerUtils, StringUtils
import utils.constants as constants

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import GenerationConfig
from trl import PPOConfig, PPOTrainer
from tqdm import tqdm
import pandas as pd
import torch

# TODO: remove
import torch.nn.functional as F
import logging
import math

try:
    from utils.flash_attn_utils import (
        replace_attn_with_flash_attn,
        upcast_layer_for_flash_attention,
    )

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


def logprobs_from_logits(logits, labels, gather=True):
    logp = F.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def override_fwd_pass(self, model, queries, responses, model_inputs, return_logits=False, response_masks=None):
    bs = len(queries)
    fbs = self.config.mini_batch_size
    all_logprobs = []
    all_logits = []
    all_masks = []
    all_values = []

    logger = logging.getLogger("inner_fwd_pass_logger")
    log_level = logging.DEBUG
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(stream_handler)

    model.eval()

    for i in range(math.ceil(bs / fbs)):
        input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
        query_batch = queries[i * fbs : (i + 1) * fbs]
        response_batch = responses[i * fbs : (i + 1) * fbs]
        if response_masks is not None:
            response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]

        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        free_memory = torch.cuda.mem_get_info()[0] / 1024 / 1024
        logger.info(f"Forward Pass: Free Memory: {free_memory} MB")
        logger.info(input_kwargs["input_ids"].shape)

        logits, _, values = model(**input_kwargs)

        if self.is_encoder_decoder:
            input_ids = input_kwargs["decoder_input_ids"]
            attention_mask = input_kwargs["decoder_attention_mask"]
        else:
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        masks = torch.zeros_like(attention_mask)
        masks[:, :-1] = attention_mask[:, 1:]

        for j in range(len(query_batch)):
            if self.is_encoder_decoder:
                # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                start = 1
                end = attention_mask[j, :].sum() - 1
            else:
                start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])
                if response_masks is not None:
                    response_masks_batch[j] = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

            masks[j, :start] = 0
            masks[j, end:] = 0
            if response_masks is not None:
                masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

        if return_logits:
            all_logits.append(logits)
        else:
            del logits
        all_values.append(values)
        all_logprobs.append(logprobs)
        all_masks.append(masks)

    return (
        torch.cat(all_logprobs),
        torch.cat(all_logits)[:, :-1] if return_logits else None,
        torch.cat(all_values)[:, :-1],
        torch.cat(all_masks)[:, :-1],
    )


class PPOTrainerWrapper:
    INSTRUCTION_COLUMN = "instruction"
    INPUT_COLUMN = "input"
    QUERY_COLUMN = "query"
    SCORE_COLUMN = "score"
    RESPONSE_COLUMN = "response"
    OPPONENT_QUERY_COLUMN = "opponent_query"
    JUDGE_QUERY_COLUMN = "judge_query"
    INPUT_IDS_COLUMN = "input_ids"
    EXTRA_SUFFIX_COLUMN = "extra_suffix"
    BATCH_SIZE = 1
    DEFAULT_JUDGE_ALIAS = "openai-judge"
    MAX_GENERATION_LENGTH = 300

    def __init__(self, ppo_trainer: PPOTrainer, config: TrainingConfig):
        self.ppo_trainer = ppo_trainer
        # self.reward_model = OpenAIModel(alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, is_debater=False)
        self.reward_model = RandomModel(alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, is_debater=False)
        self.config = config
        self.generation_config = GenerationConfig(
            max_new_tokens=PPOTrainerWrapper.MAX_GENERATION_LENGTH,
            temperature=0.5,
            top_p=0.9,
            num_return_sequences=1,
            repetition_penalty=1.2,
            do_sample=True,
            use_cache=True,
            pad_token_id=self.ppo_trainer.tokenizer.eos_token_id,
        )
        self.logger = LoggerUtils.get_default_logger(__name__)

    # TODO: remove this
    def print_gpu_memory(self, description: str = ""):
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
            free_memory = torch.cuda.mem_get_info()[0] / 1024 / 1024
            self.logger.info(f"{description}: Free Memory: {free_memory} MB")
        else:
            self.logger.info("CUDA is not available.")

    def train(self):
        self.ppo_trainer.model.gradient_checkpointing_enable()

        # NOTE: TODO: This only optimizes Debater_A
        PPOTrainer.batched_forward_pass = override_fwd_pass

        for i, row in tqdm(enumerate(self.ppo_trainer.dataset)):
            row_to_use = f"{row[PPOTrainerWrapper.QUERY_COLUMN]}\n {constants.INSTRUCTION_SUFFIX}"
            query_tensors = self.ppo_trainer.tokenizer(row_to_use, return_tensors="pt").to("cuda")
            opponent_query_tensors = self.ppo_trainer.tokenizer(
                row[PPOTrainerWrapper.OPPONENT_QUERY_COLUMN], return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                response_tensors = self.ppo_trainer.generate(
                    query_tensors.input_ids[0, :], generation_config=self.generation_config
                )

            output_length = response_tensors.shape[1]
            response_mask = torch.zeros(output_length, dtype=torch.int32)
            response_mask[-PPOTrainerWrapper.MAX_GENERATION_LENGTH :] = 1
            self.logger.info(f"Output length is {output_length}")

            """
            with torch.no_grad():

                opponent_response_tensors = self.ppo_trainer.model.generate(
                    **opponent_query_tensors, generation_config=self.generation_config
                )
            """

            reward_tensors = []
            for j in range(len(response_tensors)):
                decoded = StringUtils.clean_string(
                    self.ppo_trainer.tokenizer.decode(
                        response_tensors[j, -PPOTrainerWrapper.MAX_GENERATION_LENGTH :].to("cpu")
                    )
                )

                self.logger.info(decoded)
                """
                opponent_decoded = StringUtils.clean_string(
                    self.ppo_trainer.tokenizer.decode(
                        opponent_response_tensors[i, -PPOTrainerWrapper.MAX_GENERATION_LENGTH :].to("cpu")
                    )
                )
                """
                opponent_decoded = "Vote for Debater_B. Vote for Debater_B"

                judge_input_text = (
                    row[PPOTrainerWrapper.JUDGE_QUERY_COLUMN]
                    .replace(RowConverter.get_dummy_name_for_speaker(constants.DEFAULT_DEBATER_A_NAME), decoded)
                    .replace(RowConverter.get_dummy_name_for_speaker(constants.DEFAULT_DEBATER_B_NAME), opponent_decoded)
                )

                reasoning = self.reward_model.predict(inputs=[[judge_input_text]], max_new_tokens=450)

                reward = self.reward_model.predict(
                    inputs=[[f"{judge_input_text}\n{reasoning}"]],
                    max_new_tokens=15,
                    speech_structure=SpeechStructure.PREFERENCE,
                )

                reward_tensors.append(torch.tensor(float(reward[0])))

            # del opponent_response_tensors # clears memory

            stats = self.ppo_trainer.step(
                queries=[query_tensors.input_ids[0, :].squeeze()],
                responses=[response_tensors[:, -PPOTrainerWrapper.MAX_GENERATION_LENGTH :].squeeze()],
                scores=reward_tensors,
            )
            # self.ppo_trainer.log_stats(stats=stats, batch=batch, rewards=rewards_tensor)

    def save_model(self):
        self.ppo_trainer.save_model(config.logging_and_saving_config.output_dir)

    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset, config: TrainingConfig) -> Dataset:
        def construct_dataframe(inputs: list[dict[str, str]]):
            df = pd.DataFrame(data=inputs)
            df[PPOTrainerWrapper.QUERY_COLUMN] = (
                df[PPOTrainerWrapper.INSTRUCTION_COLUMN] + "\n" + df[PPOTrainerWrapper.INPUT_COLUMN]
            )
            df = df.drop(
                columns=[
                    PPOTrainerWrapper.INSTRUCTION_COLUMN,
                    PPOTrainerWrapper.INPUT_COLUMN,
                    PPOTrainerWrapper.EXTRA_SUFFIX_COLUMN,
                ]
            )
            return df

        debater_input_lists = [
            RowConverter.convert_row(row=row, config=config, target=TrainingTarget.DEBATER, dataset=raw_dataset, index=i)
            for i, row in enumerate(raw_dataset.get_data(split=SplitType.TRAIN))
        ]
        debater_inputs = [item for debater_input_list in debater_input_lists for item in debater_input_list]
        debater_a_inputs = [debater_inputs[i] for i in filter(lambda x: x % 2 == 0, range(len(debater_inputs)))]
        debater_b_inputs = [debater_inputs[i] for i in filter(lambda x: x % 2 == 1, range(len(debater_inputs)))]

        judge_input_lists = [
            RowConverter.convert_row(
                row=row, config=config, target=TrainingTarget.JUDGE, dataset=raw_dataset, index=i, use_dummy=True
            )
            for i, row in enumerate(raw_dataset.get_data(split=SplitType.TRAIN))
        ]
        judge_inputs = [item for judge_input_list in judge_input_lists for item in judge_input_list]

        df = construct_dataframe(inputs=debater_a_inputs)
        opponent_df = construct_dataframe(inputs=debater_b_inputs)
        judge_df = construct_dataframe(inputs=judge_inputs)

        df[PPOTrainerWrapper.OPPONENT_QUERY_COLUMN] = opponent_df[PPOTrainerWrapper.QUERY_COLUMN]
        df[PPOTrainerWrapper.JUDGE_QUERY_COLUMN] = judge_df[PPOTrainerWrapper.QUERY_COLUMN]

        return Dataset.from_pandas(df).shuffle()

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> PPOTrainerWrapper:
        logger = LoggerUtils.get_default_logger(__name__)

        if FLASH_ATTENTION_AVAILABLE:
            logger.info("Starting to use flash attention")
            replace_attn_with_flash_attn()
        else:
            logger.warning("Flash attention will not be used")

        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local, requires_value_head=True)

        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        ppo_config = PPOConfig(
            steps=config.training_hyperparameters.steps,
            learning_rate=config.training_hyperparameters.learning_rate,
            batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            ppo_epochs=1,
            optimize_device_cache=True,
        )

        train_dataset = PPOTrainerWrapper.convert_dataset(
            raw_dataset=raw_dataset,
            config=config,
        )

        ppo_trainer = PPOTrainerWrapper(
            ppo_trainer=PPOTrainer(
                model=model,
                config=ppo_config,
                dataset=train_dataset,
                tokenizer=tokenizer,
            ),
            config=config,
        )

        return ppo_trainer
