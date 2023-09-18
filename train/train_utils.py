from agents.prompt import Prompt, PromptParser, PromptTag
from data.data import DataRow, RawDataset, SplitType
import utils.constants as constants

from pydantic import BaseModel
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import pandas as pd
import torch
import yaml


class LlamaInput(BaseModel):
    instruction: str
    input: str
    output: str


class PromptConfig(BaseModel):
    prompts_file_path: str
    prompt_name: str


class LoggingAndSavingConfig(BaseModel):
    logging_steps: int
    output_dir: str


class TrainingHyperParameterConfig(BaseModel):
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    learning_rate: float
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str


class TrainingConfig(BaseModel):
    model_name: str
    prompt_config: PromptConfig
    logging_and_saving_config: LoggingAndSavingConfig
    training_hyperparameters: TrainingHyperParameterConfig


class TrainUtils:
    @classmethod
    def parse_config(cls, config_name: str, config_filepath: str):
        with open(config_filepath) as f:
            loaded_yaml = yaml.safe_load(f)
        return TrainingConfig(**loaded_yaml[config_name])

    # TODO: Generalize this to more than just the opening statement
    @classmethod
    def convert_row(cls, row: DataRow, prompts_file_path: str, prompt_name: str) -> dict[str, str]:
        prompt_config = PromptParser.convert_data_row_to_default_prompt_config(row=row)
        prompt = PromptParser.parse(prompts_file_path=prompts_file_path, prompt_config=prompt_config, name=prompt_name)
        return LlamaInput(
            instruction="\n".join(
                [prompt.messages[PromptTag.OVERALL_SYSTEM].content, prompt.messages[PromptTag.DEBATER_SYSTEM].content]
            ),
            input=prompt.messages[PromptTag.PRE_OPENING_SPEECH].content,
            output=row.speeches[0],
        ).dict()

    @classmethod
    def format_instruction(cls, llama_dictionary: dict[str, list[str]]) -> str:
        instructions = []
        for instruction_val, input_val, output_val in zip(
            llama_dictionary.get("instruction"), llama_dictionary.get("input"), llama_dictionary.get("output")
        ):
            llama_input = LlamaInput(
                instruction=instruction_val,
                input=input_val,
                output=output_val,
            )
            instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                {constants.INSTRUCTION_PREFIX}
                {llama_input.instruction}
                {constants.INPUT_PREFIX}
                {llama_input.input}
                {constants.RESPONSE_PREFIX}
                {llama_input.output}"""
            instructions.append(instruction)
        return instructions

    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset, prompts_file_path: str, prompt_name: str) -> Dataset:
        llama_inputs = map(
            lambda row: TrainUtils.convert_row(row=row, prompts_file_path=prompts_file_path, prompt_name=prompt_name),
            raw_dataset.get_data(split=SplitType.TRAIN),
        )
        df = pd.DataFrame(data=llama_inputs)
        return Dataset.from_pandas(df)

    @classmethod
    def get_trainer(cls, config: TrainingConfig, raw_dataset: RawDataset, is_local: bool = False) -> SFTTrainer:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # TODO: make configurable
        if not is_local:
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                quantization_config=bnb_config,
                use_cache=False,
                device_map="auto",
            )
            model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                revision="main",
            )

        # make this configurable
        training_args = TrainingArguments(
            output_dir=config.logging_and_saving_config.output_dir,
            num_train_epochs=config.training_hyperparameters.num_train_epochs,
            per_device_train_batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim=config.training_hyperparameters.optim,
            logging_steps=config.logging_and_saving_config.logging_steps,
            save_strategy="epoch",
            learning_rate=config.training_hyperparameters.learning_rate,
            bf16=True,
            tf32=not is_local,
            max_grad_norm=config.training_hyperparameters.max_grad_norm,
            warmup_ratio=config.training_hyperparameters.warmup_ratio,
            lr_scheduler_type=config.training_hyperparameters.lr_scheduler_type,
            disable_tqdm=False,
            use_cpu=is_local,
        )

        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=constants.INSTRUCTION_PREFIX,
            response_template=constants.RESPONSE_PREFIX,
            tokenizer=tokenizer,
        )

        train_dataset = TrainUtils.convert_dataset(
            raw_dataset=raw_dataset,
            prompts_file_path=config.prompt_config.prompts_file_path,
            prompt_name=config.prompt_config.prompt_name,
        )

        return SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config if not is_local else None,
            tokenizer=tokenizer,
            data_collator=collator,
            formatting_func=TrainUtils.format_instruction,
            args=training_args,
        )
