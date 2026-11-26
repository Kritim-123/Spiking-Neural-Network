import argparse
from collections import deque
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk

import mujoco
import torch
import torch.nn.functional as F

from cnn.cnn import depth_encoder
from go2_dual_view.commands import COMMANDS
from go2_dual_view.pose_motion import apply_pose_command
from go2_dual_view.viewer import DualView

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "snn"))
from SpikingNetwork import SpikingNetwork

REVERSE_TICKS = 50
TURN_TICKS = 100

class PolicyView(DualView):
    def __init__(self, window, checkpoint):
        super().__init__(window)
        window.title("Go2 dual view - CNN + SNN")

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.actions = state["actions"]

        self.cnn = depth_encoder
        self.snn = SpikingNetwork()
        self.cnn.load_state_dict(state["cnn"])
        self.snn.load_state_dict(state["snn"])
        self.cnn.eval()
        self.snn.eval()

        # Start in the corridor and keep the scene's original standing pose.
        self.data.qpos[:2] = (2.85, 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.show_room()

        self.depth_renderer = mujoco.Renderer(self.model, height=180, width=320)
        self.depth_renderer.enable_depth_rendering()

        self.features = deque(maxlen=8)
        self.action = "stop"
        self.tick = 0
        self.recovery_ticks = 0

        self.status = ttk.Label(window, text="Collecting 8 frames...")
        self.status.pack(pady=6)
        self.after_id = window.after(0, self.update_simulation)

    @torch.inference_mode()
    def update_action(self):
        self.depth_renderer.update_scene(self.data, camera="front_cam")
        depth = self.depth_renderer.render()

        # Use the same preprocessing as train_policy.py.
        image = torch.from_numpy(depth.copy()).float()[None, None]
        image = torch.nan_to_num(image, nan=4.0, posinf=4.0, neginf=0.0)
        image = image.clamp(0.0, 4.0) / 4.0
        image = F.interpolate(image, size=(90, 160), mode="area")
        self.features.append(self.cnn(image)[0])

        if len(self.features) == 8:
            sequence = torch.stack(tuple(self.features)).unsqueeze(0)
            scores = self.snn(sequence)[0]
            self.action = self.actions[scores.argmax().item()]
            if self.action == "stop":
                self.recovery_ticks = REVERSE_TICKS + TURN_TICKS
            self.status.configure(text=f"SNN action: {self.action}")

    def update_simulation(self):
        start = time.perf_counter()

        # Apply two 20 ms pose updates per screen refresh.
        for _ in range(2):
            if self.recovery_ticks == 0 and self.tick % 12 == 0:
                self.update_action()

            if self.recovery_ticks > TURN_TICKS:
                movement = "reverse"
            elif self.recovery_ticks > 0:
                movement = "turn_right"
            else:
                movement = self.action

            apply_pose_command(self.model, self.data, COMMANDS[movement], dt=0.02)
            self.tick += 1

            if self.recovery_ticks > 0:
                self.status.configure(text=f"SNN: stop | Recovery: {movement}")
                self.recovery_ticks -=1

                if self.recovery_ticks == 0:
                    self.features.clear()
                    self.action = "stop"
                    self.status.configure(text="Collecting 8 fresh frames...")

        self.display(self.global_view, self.global_renderer, self.camera)
        self.display(self.head_view, self.head_renderer, "front_cam")

        elapsed_ms = (time.perf_counter() - start) * 1000
        delay_ms = max(1, round(40 - elapsed_ms))
        self.after_id = self.window.after(delay_ms, self.update_simulation)

    def close(self):
        self.window.after_cancel(self.after_id)
        self.depth_renderer.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    window = tk.Tk()
    PolicyView(window, args.checkpoint)
    window.mainloop()


if __name__ == "__main__":
    main()