"""
This defines the encoder-decoder model.
Both the encoder and the decoder are GRU-based.
This makes it easier to reason about the system because we only have one memory vector.
We also get a small improvement in speed.
While we lose some modelling ability, it's probably not significant.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: torch.device) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.embedding(input_).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, device: torch.device) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.embedding(input_).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, self.device)