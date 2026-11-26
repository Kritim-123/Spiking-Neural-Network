from .SpikingNetwork import SpikingNetwork


class ActionExpert(SpikingNetwork):
    def __init__(self):
        # Reconstruct 32 CNN features at each frame from 128 spiking neurons.
        super().__init__(input_size=32, hidden_size=128, num_classes=32)

    def forward(self, features):
        return self.readout(self.spikes(features))