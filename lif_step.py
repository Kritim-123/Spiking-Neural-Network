import torch

def lif_step(current, mem, beta=0.9, threshold=1.0):
    mem = beta * mem + current

    distance = mem - threshold
    hard_spike = (distance >=0).float()
    soft_spike = torch.sigmoid(5*distance)

    # Forward pass: binary spike. Backward pass: sigmoid gradient

    spike = hard_spike + (soft_spike - soft_spike.detach())

    mem = mem - threshold * hard_spike

    return spike, mem