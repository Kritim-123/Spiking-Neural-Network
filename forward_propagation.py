import torch
from torch import nn

from lif_step import lif_step

class SpikingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, num_classes)

        def forward(self, inputs):
            # inputs: [batch_size, time_steps, input_size]

            currents = self.input_layer(inputs)

            # currents: [batch_size, time_steps, hidden_size]

            mem = torch.zeros_like(currents[:, 0, :])
            spikes_over_time = []

            for current in currents.unbind(dim=1):
                spike, mem = lif_step(current, mem)
                spikes_over_time.append(spike)

            # One average firign rate per hidden neuron, for each example.
            spike_rates = torch.stack(spikes_over_time, dim=1).mean(dim=1)
            return self.readout(spike_rates)

        