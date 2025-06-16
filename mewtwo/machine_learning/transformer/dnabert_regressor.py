from typing import Optional

import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


class DNABERTRegressor(nn.Module):
    def __init__(self, base_model, dropout: float = 0.2, use_adapters=False, lora_alpha: int = 16, lora_r: int = 8,
                 lora_dropout: float = 0.1, second_layer_dim: Optional[int] = None):
        super().__init__()
        self.base = base_model

        if use_adapters:
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                     inference_mode=False,
                                     r=lora_r,
                                     lora_alpha=lora_alpha,
                                     lora_dropout=lora_dropout,
                                     target_modules=["attention.self.Wqkv"])
            self.base = get_peft_model(self.base, peft_config)

        hidden_size = self.base.base_model.config.hidden_size

        if second_layer_dim is None:

            self.regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1),
                                           nn.Sigmoid())

        else:
            self.regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, second_layer_dim),
                                           nn.ReLU(), nn.Dropout(dropout), nn.Linear(second_layer_dim, 1),
                                           nn.Sigmoid())

    def forward(self, input_ids, attention_mask=None):
        # Pass through the model
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0]  # CLS token for regression
        predictions = self.regressor(cls_token).squeeze(-1)
        return predictions
