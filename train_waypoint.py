"""Train an action selector from goal-labeled clips with frozen vision and SNNs."""

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cnn.cnn import depth_encoder, preprocess_depth
from record_clips import ACTIONS, FRAME_COUNT, ROOT
from snn.expert_policy import ExpertPolicy, waypoint_selector


def split_waypoint_clips(root):
    groups = defaultdict(list)
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as clip:
            key = sha256(clip["depth"].tobytes()).digest()
        groups[key].append(path)

    if len(groups) < 2:
        raise ValueError("Record at least two different waypoint clips before training.")
    train, val = [], []
    if any(path.is_relative_to(root / "val") for paths in groups.values() for path in paths):
        for paths in groups.values():
            # A duplicate of an explicitly held-out clip must stay held out too.
            destination = val if any(path.is_relative_to(root / "val") for path in paths) else train
            destination.extend(paths)
    else:
        clips = list(groups.values())
        Random(42).shuffle(clips)
        val_size = max(1, round(0.2 * len(clips)))
        val = [path for group in clips[:val_size] for path in group]
        train = [path for group in clips[val_size:] for path in group]
    if not train or not val:
        raise ValueError("Need distinct waypoint clips for training and validation.")
    return train, val


@torch.no_grad()
def encode_examples(paths, cnn, experts, device):
    inputs, labels = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as clip:
            depth = clip["depth"]
            goal = clip["goal_features"]
            actions = clip["action"]
        if (
            depth.shape != (FRAME_COUNT, 180, 320)
            or actions.ndim != 1 or not len(actions)
            or goal.shape != (len(actions), 3) or not np.isfinite(goal).all()
            or not np.issubdtype(actions.dtype, np.integer)
            or np.any(actions < 0) or np.any(actions >= len(ACTIONS))
        ):
            raise ValueError(f"Invalid waypoint clip: {path}")
        images = torch.as_tensor(depth, dtype=torch.float32, device=device).unsqueeze(1)
        features = cnn(preprocess_depth(images)).unsqueeze(0)
        rates = experts.spike_features(features).expand(len(actions), -1)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=device)
        inputs.append(torch.cat((rates, goal), dim=1))
        labels.append(torch.as_tensor(actions, dtype=torch.long, device=device))
    return torch.cat(inputs), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-checkpoint", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "waypoint_samples")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    try:
        train_paths, val_paths = split_waypoint_clips(args.data)
    except ValueError as exc:
        parser.error(str(exc))

    state = torch.load(args.expert_checkpoint, weights_only=True, map_location="cpu")
    if "expert_policy" not in state or tuple(state.get("actions", ())) != ACTIONS:
        parser.error("--expert-checkpoint must be a three-expert checkpoint with the recorded action order")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = deepcopy(depth_encoder).to(device).eval().requires_grad_(False)
    cnn.load_state_dict(state["cnn"])
    experts = ExpertPolicy(num_actions=len(ACTIONS)).to(device).eval().requires_grad_(False)
    experts.load_state_dict(state["expert_policy"])
    train_x, train_y = encode_examples(train_paths, cnn, experts, device)
    val_x, val_y = encode_examples(val_paths, cnn, experts, device)
    print(f"{device} | Train {len(train_paths)} clips / {len(train_y)} labels | "
          f"Validation {len(val_paths)} clips / {len(val_y)} labels")
    print("Action order:", ", ".join(ACTIONS))
    print("Train and validation label counts:",
          torch.bincount(train_y, minlength=len(ACTIONS)).tolist(),
          torch.bincount(val_y, minlength=len(ACTIONS)).tolist())

    selector = waypoint_selector(len(ACTIONS)).to(device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    output = args.output or ROOT / "checkpoints" / f"waypoint_policy_{uuid4().hex}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(selector(inputs), labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = selector(val_x)
            val_loss = F.cross_entropy(logits, val_y).item()
            accuracy = (logits.argmax(dim=1) == val_y).float().mean().item()
        print(f"Epoch {epoch:02d} | Val loss {val_loss:.3f} | Accuracy {accuracy:.1%}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({**state, "waypoint_selector": selector.state_dict(),
                        "waypoint_epoch": epoch}, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
