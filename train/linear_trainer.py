from __future__ import annotations

from agents import LLModuleWithLinearProbe, ModelSettings, ModelUtils
from data import JudgingProbeDataRow, QualityJudgingDataset, RawDataset, SplitType
from train.train_utils import TrainingConfig, TrainUtils

from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
import torch.optim as optim
import torch

from typing import Optional, Type


class LinearTrainer:
    """Class for training a linear probe on top of a model"""

    def __init__(self, model: LLModuleWithLinearProbe, dataset: QualityJudgingDataset, config: TrainingConfig):
        self.probe = model.probe
        self.dataset = dataset
        self.config = config

    def train(self):
        optim = optim.AdamW(params=probe_model.parameters(), lr=config.training_hyperparameters.learning_rate)
        for epoch in range(config.training_hyperparameters.num_train_epochs):
            self.train_epoch(optimizer=optimizer)

    def train_epoch(self, optimizer: optim.Optimizer):
        data_length = len(self.dataset.get_data(SplitType.TRAIN))
        for i in range(0, data_length, config.training_hyperparameters.per_device_train_batch_size):
            self.train_batch(
                batch=self.dataset.get_batch(config.training_hyperparameters.per_device_train_batch_size),
                optimizer=optimizer,
            )

    def train_batch(self, batch: list[JudgingProbeDataRow], optimizer: optim.Optimizer):
        loss_func = nn.MSELoss()
        optim.zero_grad()
        pred = self.probe(torch.stack([row.internal_representation for row in batch]))
        target = self.probe(torch.stack([row.target for row in batch]))
        loss = loss_func(pred, target)
        loss.backward()
        optim.step()

    @classmethod
    def get_trainer(cls, config: TrainingConfig, raw_dataset: Optional[RawDataset] = None, **kwargs) -> LinearTrainer:
        if not raw_dataset:
            raw_dataset = TrainUtils.create_dataset(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local)
        return LinearTrainer(model=model, dataset=raw_dataset, config=config)
