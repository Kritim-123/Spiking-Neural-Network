import torch
from torch import nn

from .forward_snn import ForwardSNN
from .turn_left_snn import TurnLeftSNN
from .turn_right_snn import TurnRightSNN


class ExpertPolicy(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.experts = nn.ModuleList([ForwardSNN(), TurnLeftSNN(), TurnRightSNN()])
        self.selector = nn.Linear(128 * len(self.experts), num_actions)

    def spike_features(self, features):
        rates = [expert.spikes(features).mean(dim=1) for expert in self.experts]
        return torch.cat(rates, dim=1)

    def forward(self, features):
        return self.selector(self.spike_features(features))


def waypoint_selector(num_actions):
    # A hidden layer lets the same bearing mean different actions around obstacles.
    return nn.Sequential(nn.Linear(128 * 3 + 3, 64), nn.ReLU(), nn.Linear(64, num_actions))
