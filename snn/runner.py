from SpikingNetwork import SpikingNetwork
import torch
from torch import nn

model = SpikingNetwork(input_size = 4, hidden_size = 128, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
los_fn = nn.CrossEntropyLoss()

inputs = torch.randn(8,20,4) # 8 sequences, 20 stesps, 4 features per step
labels = torch.randint(0,3, (8,)) # 8 labels, one for each sequence

optimizer.zero_grad()
logits = model(inputs)
loss = los_fn(logits, labels)
loss.backward()

print("Input weight gradient:", model.input_layer.weight.grad.norm().item())

optimizer.step()