import argparse
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cnn.cnn import depth_encoder
from snn.expert_policy import ExpertPolicy
from train_policy import ACTIONS, BATCH_SIZE, DepthClips, ROOT, split_clips

EXPERT_EPOCHS = 15
SELECTOR_EPOCHS = 15


@torch.no_grad()
def encode_clips(files, cnn, device):
    loader = DataLoader(DepthClips(files), batch_size=BATCH_SIZE)
    features, labels = [], []

    for depth, actions in loader:
        batch_size, frames, height, width = depth.shape
        images = depth.to(device).reshape(batch_size * frames, 1, height, width)
        images = torch.nan_to_num(images, nan=4.0, posinf=4.0, neginf=0.0)
        images = images.clamp(0.0, 4.0) / 4.0
        images = F.interpolate(images, size=(90, 160), mode="area")

        features.append(cnn(images).reshape(batch_size, frames, 32))
        labels.append(actions.to(device))

    return torch.cat(features), torch.cat(labels)


def train_expert(expert, features, labels):
    own_clips = features[labels == ACTIONS.index(expert.action)]
    if len(own_clips) == 0:
        raise ValueError(f"No training clips for {expert.action}")

    print(f"Training {expert.action} SNN on {len(own_clips)} clips", flush=True)
    loader = DataLoader(own_clips, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)

    for _ in range(EXPERT_EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            reconstruction = expert(batch)
            loss = F.mse_loss(reconstruction, batch)
            loss.backward()
            optimizer.step()

    expert.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-checkpoint", required=True, type=Path)
    parser.add_argument("--preview", action="store_true", help="Watch saved expert checkpoints in MuJoCo")
    parser.add_argument("--preview-scene", type=Path, help="Scene for the MuJoCo preview")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files, val_files, validation_id = split_clips(ROOT / "training_samples")

    checkpoint = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("validation_id") != validation_id:
        parser.error(
            "CNN checkpoint was not trained with the current validation split. "
            "Run train_policy.py and pass its new checkpoint to --cnn-checkpoint."
        )
    cnn = depth_encoder.to(device).eval()
    cnn.load_state_dict(checkpoint["cnn"])
    cnn.requires_grad_(False)

    print(f"Device: {device} | Train: {len(train_files)} | Val: {len(val_files)}")
    train_x, train_y = encode_clips(train_files, cnn, device)
    val_x, val_y = encode_clips(val_files, cnn, device)

    policy = ExpertPolicy(num_actions=len(ACTIONS)).to(device)
    for expert in policy.experts:
        train_expert(expert, train_x, train_y)

    # The experts are frozen. The selector learns from all four action labels.
    with torch.no_grad():
        train_rates = policy.spike_features(train_x)
        val_rates = policy.spike_features(val_x)

    loader = DataLoader(
        TensorDataset(train_rates, train_y), batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.Adam(policy.selector.parameters(), lr=0.001)
    output_dir = ROOT / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"expert_policy_{uuid4().hex}.pt"
    temporary_path = output_path.with_suffix(".tmp")
    best_loss = float("inf")
    preview = None

    for epoch in range(1, SELECTOR_EPOCHS + 1):
        for rates, labels in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(policy.selector(rates), labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = policy.selector(val_rates)
            val_loss = F.cross_entropy(logits, val_y).item()
            correct = logits.argmax(dim=1).eq(val_y)
            accuracy = correct.float().mean().item()

        print(f"Selector epoch {epoch:02d} | val loss {val_loss:.3f} | acc {accuracy:.1%}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "cnn": cnn.state_dict(),
                    "expert_policy": policy.state_dict(),
                    "actions": ACTIONS,
                    "validation_id": validation_id,
                    "selector_epoch": epoch,
                },
                temporary_path,
            )
            temporary_path.replace(output_path)
            if args.preview and preview is None:
                command = [
                    sys.executable, "-m", "go2_dual_view.policy_viewer",
                    "--checkpoint", str(output_path), "--follow-checkpoint",
                ]
                if args.preview_scene is not None:
                    command.extend(["--scene", str(args.preview_scene.resolve())])
                preview = subprocess.Popen(
                    command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            for i, action in enumerate(ACTIONS):
                mask = val_y == i
                print(f"  {action}: {correct[mask].sum().item()}/{mask.sum().item()}")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
