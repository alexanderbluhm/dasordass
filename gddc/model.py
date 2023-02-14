'''
Model for the german das dass classifier (gddc)
'''
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, lang_model: Optional[nn.Module],
                 n_model: Optional[int] = 768, dropout: Optional[float] = 0.1):
        super().__init__()
        self.bert = lang_model
        self.norm = nn.LayerNorm(n_model)
        self.fc1 = nn.Linear(n_model, 128)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.norm(outputs[1])
        x = F.relu(self.drop1(self.fc1(pooled_output)))
        return self.act(self.drop2(self.fc2(x)))
