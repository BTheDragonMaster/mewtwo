import argparse
import os
from enum import Enum

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
from mewtwo.machine_learning.transformer.loss_functions import CombinedMSEPearsonLoss, WeightedMSELoss
from mewtwo.machine_learning.transformer.config.config_types import FinetuningType
from mewtwo.parsers.parse_model_config import ModelConfig


def initialise(finetuning_mode: FinetuningType, dropout, adapter_config=None):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    base_model = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")

    if finetuning_mode == FinetuningType.LINEAR_HEAD or finetuning_mode == FinetuningType.ADAPTER:
        for param in base_model.parameters():
            param.requires_grad = False

    if finetuning_mode == FinetuningType.ADAPTER:
        assert adapter_config is not None
        model = DNABERTRegressor(base_model, use_adapters=True, lora_r=adapter_config.rank,
                                 lora_alpha=adapter_config.alpha,
                                 lora_dropout=adapter_config.dropout)
    else:
        model = DNABERTRegressor(base_model, use_adapters=False)

    return model, tokenizer


def load_model(config_file, model_checkpoint=None):
    model_config = ModelConfig.from_file(config_file)
    model = initialise(model_config.finetuning_mode, model_config.adapter_config)


