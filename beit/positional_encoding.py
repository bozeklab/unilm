import math

import torch
from torch import nn

from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, drop_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_rate)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        print(self.pe)
        return self.pe.permute(1, 0, 2)