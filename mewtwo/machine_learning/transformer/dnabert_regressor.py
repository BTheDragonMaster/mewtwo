import torch.nn as nn


class DNABERTRegressor(nn.Module):
    def __init__(self, base_model, hidden_size=768, dropout: float = 0.2,
                 tuning_mode: str = 'linear_head'):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout)
        if tuning_mode == 'linear_head':

            self.regressor = nn.Sequential(self.dropout, nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, input_ids, attention_mask=None):
        # Pass through the model
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(output).squeeze(-1)
        return logits

