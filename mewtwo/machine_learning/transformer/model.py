from dataclasses import dataclass
from typing import Union

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from mewtwo.external_code.dnabert.bert_layers import BertModel
from mewtwo.machine_learning.transformer.dnabert_regressor import DNABERTRegressor
from mewtwo.machine_learning.transformer.regressor_dataset import RegressionDataset
from mewtwo.parsers.parse_dnabert_data import parse_dnabert_data
from mewtwo.machine_learning.data_preparation.calculate_sample_weights import get_sample_weights
from mewtwo.machine_learning.transformer.loss_functions import get_loss_function, WeightedMSELoss, \
    CombinedMSEPearsonLoss, CombinedMSESpearmanLoss
from mewtwo.machine_learning.transformer.config.config_types import FinetuningType, SchedulerType, LossFunctionType
from mewtwo.parsers.parse_model_config import ModelConfig
from mewtwo.machine_learning.transformer.schedulers import WarmupReduceOnPlateau


@dataclass
class Model:
    model: DNABERTRegressor
    train_dataloader: DataLoader
    eval_dataloader: DataLoader
    tokenizer: AutoTokenizer
    optimizer: optim.AdamW
    scheduler: Union[torch.optim.lr_scheduler.LambdaLR, ReduceLROnPlateau, WarmupReduceOnPlateau]
    loss_function: Union[nn.MSELoss, WeightedMSELoss, CombinedMSEPearsonLoss, CombinedMSESpearmanLoss]
    config: ModelConfig

    def save_model_checkpoint(self, out_file):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "config": self.config,


                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict()}, out_file)

    def update_epoch(self, epoch):
        self.config.epochs = epoch

    def train_model(self, device='cpu'):
        self.model.train()  # set to training mode
        self.model.to(device)

        total_loss = 0.0
        for batch in self.train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            if self.config.loss_function_config.type in LossFunctionType.WEIGHTED:
                loss = self.loss_function(outputs, labels, weights)
            else:
                loss = self.loss_function(outputs, labels)

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                if self.config.scheduler_config.type in SchedulerType.WARMUP_SCHEDULERS:
                    self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def evaluate_model(self, device="cpu"):
        self.model.eval()  # Set to evaluation mode
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                weights = batch["weights"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if self.config.loss_function_config.type in LossFunctionType.WEIGHTED:
                    loss = self.loss_function(outputs, labels, weights)
                else:
                    loss = self.loss_function(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.eval_dataloader)

        return avg_loss, all_preds, all_labels


def initialise(finetuning_mode: FinetuningType, dropout, adapter_config):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    base_model = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")

    if finetuning_mode == FinetuningType.LINEAR_HEAD or finetuning_mode == FinetuningType.ADAPTER:
        for param in base_model.parameters():
            param.requires_grad = False

    if finetuning_mode == FinetuningType.ADAPTER:
        assert adapter_config is not None
        model = DNABERTRegressor(base_model, dropout=dropout, use_adapters=True, lora_r=adapter_config.rank,
                                 lora_alpha=adapter_config.alpha,
                                 lora_dropout=adapter_config.dropout)
    else:
        model = DNABERTRegressor(base_model, dropout=dropout, use_adapters=False)

    return model, tokenizer


def prepare_data(input_file: str, tokenizer: AutoTokenizer, shuffle: bool, batch_size: int = 5) \
        -> DataLoader:
    sequences, labels = parse_dnabert_data(input_file)
    sample_weights = get_sample_weights(labels)
    dataset = RegressionDataset(sequences, labels, sample_weights, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def load_model(input_training_data, input_validation_data, config_file=None, model_checkpoint=None):
    checkpoint = None
    load_from_checkpoint = False
    if model_checkpoint is not None:
        if config_file:
            print("Warning: Model checkpoint and config file have been given. Config file is ignored.")

        load_from_checkpoint = True
        checkpoint = torch.load(model_checkpoint)
        model_config = checkpoint["config"]

    elif config_file is not None:
        model_config = ModelConfig.from_file(config_file)

    else:
        raise ValueError("Config file or model checkpoint must be given")

    model, tokenizer = initialise(model_config.finetuning_mode, model_config.hidden_layer_dropout,
                                  model_config.adapter_config)

    train_dataloader = prepare_data(input_training_data, tokenizer, True, batch_size=model_config.batch_size)
    eval_dataloader = prepare_data(input_validation_data, tokenizer, True, batch_size=model_config.batch_size)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = optim.AdamW(model.parameters(), lr=model_config.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        optimizer = optim.AdamW(model.parameters(), lr=model_config.learning_rate)

    if model_config.scheduler_config.type == SchedulerType.COS_ANNEAL_WARMUP:
        warmup_steps = model_config.scheduler_config.warmup_epochs * len(train_dataloader)
        training_steps = model_config.scheduler_config.training_epochs * len(train_dataloader)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
    elif model_config.scheduler_config.type == SchedulerType.REDUCE_ON_PLATEAU:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True,
                                      min_lr=5e-6)
    elif model_config.scheduler_config.type == SchedulerType.REDUCE_ON_PLATEAU_WARMUP:
        warmup_steps = model_config.scheduler_config.warmup_epochs * len(train_dataloader)
        scheduler = WarmupReduceOnPlateau(
            optimizer,
            warmup_steps=warmup_steps,
            plateau_scheduler_kwargs={
                "mode": "min",
                "patience": model_config.scheduler_config.plateau_patience,
                "factor": model_config.scheduler_config.factor,
                "min_lr": 5e-6
            }, load_from_checkpoint=load_from_checkpoint
        )

    else:
        raise ValueError(f"Unknown scheduler type: {model_config.scheduler_config.type.name}")

    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler.get_last_lr()[0]

    loss_function = get_loss_function(model_config.loss_function_config)
    training_model = Model(model, train_dataloader, eval_dataloader, tokenizer, optimizer, scheduler, loss_function,
                           model_config)

    return training_model
