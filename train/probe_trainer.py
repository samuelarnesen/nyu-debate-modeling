from __future__ import annotations

from data import JudgingProbeDataRow, QualityJudgingDataset, RawDataset, SplitType
from models import LLModuleWithLinearProbe, ProbeHyperparams
from train.train_utils import TrainingConfig, TrainUtils
from utils import logger_utils

from pydantic import BaseModel
import torch.nn as nn
import torch.optim as optim
import torch

from typing import Optional, Type
import os


class ProbeTrainer:
    """Class for training a linear probe on top of a model"""

    def __init__(self, model: nn.Module, dataset: QualityJudgingDataset, config: TrainingConfig):
        self.probe = model
        self.dataset = dataset
        self.config = config
        self.logger = logger_utils.get_default_logger(__name__)
        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.BCEWithLogitsLoss()

    def train(self):
        for name, param in self.probe.named_parameters():
            self.logger.info(f"Name: {name}, Param: {param}")
        optimizer = optim.AdamW(
            params=self.probe.parameters(), lr=self.config.training_hyperparameters.learning_rate, weight_decay=1e-5
        )
        for epoch in range(self.config.training_hyperparameters.num_train_epochs):
            train_loss = self.train_batch(
                batch=self.dataset.get_data(split=SplitType.TRAIN),
                optimizer=optimizer,
            )
            val_loss = self.validate_batch(batch=self.dataset.get_data(split=SplitType.VAL))
            self.logger.info(val_loss if not self.config.dataset.combine_train_and_val else train_loss)

        for name, param in self.probe.named_parameters():
            self.logger.info(f"Name: {name}, Param: {param}")

    def train_batch(self, batch: list[JudgingProbeDataRow], optimizer: optim.Optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate_batch(self, batch: list[JudgingProbeDataRow]):
        with torch.no_grad():
            self.probe.eval()
            loss = self.get_loss(batch)
            self.probe.train()
        return loss.item()

    def get_loss(self, batch: list[JudgingProbeDataRow]) -> torch.tensor:
        input_tensor = torch.stack([row.internal_representation.float() for row in batch])
        pred = self.probe(input_tensor)
        target = torch.stack([row.target for row in batch])
        return self.loss_func(pred, target)

    def save_model(self):
        if not os.path.exists(self.config.logging_and_saving_config.output_dir):
            os.makedirs(self.config.logging_and_saving_config.output_dir)
        torch.save(self.probe.state_dict(), f"{self.config.logging_and_saving_config.output_dir}/probe.pth")

    @classmethod
    def get_trainer(
        cls, config: TrainingConfig, raw_dataset: Optional[RawDataset] = None, is_local: bool = False, **kwargs
    ) -> ProbeTrainer:
        if not raw_dataset:
            raw_dataset = TrainUtils.create_dataset(config=config)
        probe_hyperparams = ProbeHyperparams(**config.training_hyperparameters.supplemental)
        model = LLModuleWithLinearProbe.instantiate_probe(
            file_path=probe_hyperparams.file_path,
            linear_idxs=probe_hyperparams.linear_idxs,
            hidden_size=probe_hyperparams.hidden_size,
        )
        return ProbeTrainer(model=model, dataset=raw_dataset, config=config)
