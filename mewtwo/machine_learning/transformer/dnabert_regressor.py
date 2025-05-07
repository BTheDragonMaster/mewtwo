import torch
import sys
from transformers import AutoTokenizer, AutoModel
import os
import torch.nn as nn

import mewtwo.external_code

sys.path.append(os.path.join(mewtwo.external_code.__file__, "DNABERT-2-117M"))

from mewtwo.external_code.dnabert.bert_layers import BertModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")
model.to("cpu")


class DNABERTRegressor(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.1)
        # Regression output: predicting continuous values
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        # Pass through the model
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # Use the CLS token for regression
        logits = self.regressor(self.dropout(cls_token))
        return logits

