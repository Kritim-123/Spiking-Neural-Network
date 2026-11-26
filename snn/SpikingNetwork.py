import torch
from torch import nn



if __package__:
    from .lif_step import lif_step
else:
    from lif_step import lif_step

class SpikingNetwork(nn.Module):
    def __init__(self, input_size = 32, hidden_size = 128, num_classes = 4):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, num_classes)

    def spikes(self, inputs):
        # inputs: [batch, frames, input_features]

        currents = self.input_layer(inputs)
        mem = torch.zeros_like(currents[:, 0, :])
        spikes_over_time = []

        for current in currents.unbind(dim=1):
            spike, mem = lif_step(current, mem)
            spikes_over_time.append(spike)

        return torch.stack(spikes_over_time, dim=1)

    def forward(self, inputs):
        spike_rates = self.spikes(inputs).mean(dim=1)
        return self.readout(spike_rates)
