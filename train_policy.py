from pathlib import Path
import sys
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cnn.cnn import depth_encoder

ROOT = Path(__file__).resolve().parent

# Use the same SNN import setup as camera_to_snn.py

sys.path.insert(0, str(ROOT / "snn"))
from SpikingNetwork import SpikingNetwork

ACTIONS = ("forward", "stop", "turn_left", "turn_right")

VAL_BATCH = "batch_2026-11-24_02"
BATCH_SIZE = 8
EPOCHS = 15

class DepthClips(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index], allow_pickle=False) as clip:
            depth = torch.from_numpy(clip["depth"]).float()
            action = int(clip["action"].item())

            return depth, action


def run_epoch(loader, cnn, snn, device, optimizer=None):
    training = optimizer is not None
    cnn.train(training)
    snn.train(training)

    total_loss = 0.0

    # Rows are the recorded actions. Columns are the predicted actions.
    confusion = torch.zeros(4,4, dtype = torch.long)

    with torch.set_grad_enabled(training):
        for depth, actions in loader:
            batch_size, frames, height, width = depth.shape

            # Combine the clips and frame dimensions for the 2D CNN.

            images = depth.to(device).reshape(batch_size * frames, 1, height, width)
            images = torch.nan_to_num(images, nan=4.0, posinf = 4.0, neginf=0.0)
            images = images.clamp(0.0, 4.0) / 4.0
            images = F.interpolate(images, size = (90, 160), mode = "area")
            targets = actions.to(device)

            if training:
                optimizer.zero_grad()

            # Restore the frame order within each clip for ths NN

            features = cnn(images).reshape(batch_size, frames, 32)
            logits = snn(features)
            loss = F.cross_entropy(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size
            predicted = logits.detach().argmax(dim=1).cpu()
            pairs = len(ACTIONS) * actions + predicted
            confusion += torch.bincount(pairs, minlength = 16).reshape(4,4)

    return total_loss / len(loader.dataset), confusion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root = ROOT / "training_samples" / "train"
    val_root = ROOT / "training_samples" / "val"

    train_files = sorted(train_root.rglob("*.npz"))
    val_files = sorted(val_root.rglob("*.npz"))

    if not val_files:
        val_files = sorted((train_root / VAL_BATCH).rglob("*.npz"))
        val_set = set(val_files)
        train_files = [path for path in train_files if path not in val_set]

    if not train_files or not val_files:
        raise ValueError("Need clips in both the training and validation sets")

    train_loader = DataLoader(DepthClips(train_files), batch_size= BATCH_SIZE, shuffle= True)
    val_loader = DataLoader(DepthClips(val_files), batch_size = BATCH_SIZE)

    cnn = depth_encoder.to(device)
    snn = SpikingNetwork(32, 128, len(ACTIONS)).to(device)

    # One optimizer updates both models.
    optimizer = torch.optim.Adam([*cnn.parameters(), *snn.parameters()], lr=0.001)

    print(f"Device: {device} | Train: {len(train_files)} | Validation: {len(val_files)}")
    best_val_loss = float("inf")

    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = uuid4().hex
    checkpoint = checkpoint_dir / f"vision_policy_{timestamp}_{run_id}.pt"

    for epoch in range(1, EPOCHS+1):
        train_loss, _ = run_epoch(train_loader, cnn, snn, device, optimizer)
        val_loss, counts = run_epoch(val_loader, cnn, snn, device)
        val_accuracy = counts.diag().sum().item() / len (val_files)

    print(
        f"Epoch {epoch:02d} | train loss {train_loss:.3f} |"
        f"val loss {val_loss:.3f} | val accuracy {val_accuracy:.1%}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {"cnn": cnn.state_dict(), "snn": snn.state_dict(), "actions": ACTIONS},
            checkpoint
        )
        scores = ",".join(
            f"{name} {counts[i,i].item()}/{counts[i].sum().item()}"
            for i,name in enumerate(ACTIONS)
        )
        print(f" Saved {checkpoint.name} | correct/total: {scores}")

if __name__ == "__main__":
    main()


