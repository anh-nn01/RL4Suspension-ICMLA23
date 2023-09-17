import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *

"""************************************
    Q-Network: CRITIC
        function: (st,at) -> Q-value = Q(st, at)
************************************"""
class QNet(nn.Module):
    def __init__(self, input_dim=4+2, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, s, a):
        outs = torch.cat([s, a], dim=-1)
        outs = F.relu(self.hidden1(outs))
        outs = F.relu(self.hidden2(outs))
        outs = self.output(outs)
        # outs = 1000*outs
        return outs

"""************************************
    Policy Network: ACTOR
        function:  st -> at
************************************"""
class PolicyNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, s):
        outs = F.relu(self.hidden1(s))
        outs = F.relu(self.hidden2(outs))
        outs = self.output(outs)
        outs = torch.tanh(outs)    # range [-1, 1]
        # outs = outs * 500   # desire output is [1000,2000]
        return outs