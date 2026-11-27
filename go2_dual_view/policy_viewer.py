import argparse
from collections import deque
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch

from cnn.cnn import depth_encoder, preprocess_depth
from go2_dual_view.commands import COMMANDS
from go2_dual_view.pose_motion import apply_pose_command
from go2_dual_view.viewer import DualView
from go2_vision.waypoint import ARRIVAL_RADIUS, waypoint_features

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "snn"))
from SpikingNetwork import SpikingNetwork
from snn.expert_policy import ExpertPolicy, waypoint_selector

REVERSE_TICKS = 50
TURN_TICKS = 100

class PolicyView(DualView):
    def __init__(self, window, checkpoint, scene_path=None, follow_checkpoint=False, waypoint=None):
        super().__init__(window, scene_path, waypoint)
        self.checkpoint = checkpoint
        self.checkpoint_mtime = None
        self.follow_checkpoint = follow_checkpoint
        self.next_checkpoint_poll = 0
        self.cnn = deepcopy(depth_encoder).cpu()
        self.global_panel.configure(text="Top-down view")
        self.head_panel.configure(text="Go2 front camera | RGB and depth")
        self.pose_status = ttk.Label(self.global_panel)
        self.pose_status.pack(pady=(4, 0))

        self.depth_renderer = mujoco.Renderer(self.model, height=180, width=320)
        self.depth_renderer.enable_depth_rendering()
        self.depth_view = ttk.Label(self.head_panel)
        self.depth_view.pack(pady=(6, 0))
        self.depth_status = ttk.Label(self.head_panel)
        self.depth_status.pack(pady=(4, 0))

        self.features = deque(maxlen=8)
        self.start_position = (2.85, 0.0) if scene_path is None else None
        self.status = ttk.Label(window)
        self.status.pack(pady=6)
        self.load_checkpoint()
        self.show_room()
        self.after_id = window.after(0, self.update_simulation)

    def load_checkpoint(self):
        mtime = self.checkpoint.stat().st_mtime_ns
        if mtime == self.checkpoint_mtime:
            return

        state = torch.load(BytesIO(self.checkpoint.read_bytes()), map_location="cpu", weights_only=True)
        self.actions = state["actions"]
        self.goal_selector = None
        if "waypoint_selector" in state:
            if self.waypoint is None:
                raise ValueError("This checkpoint requires --goal X Y.")
            self.goal_selector = waypoint_selector(len(self.actions)).eval()
            self.goal_selector.load_state_dict(state["waypoint_selector"])
        elif self.waypoint is not None:
            raise ValueError("--goal requires a waypoint_policy checkpoint.")

        if "expert_policy" in state:
            self.snn = ExpertPolicy(num_actions=len(self.actions))
            self.snn.load_state_dict(state["expert_policy"])
        else:
            self.snn = SpikingNetwork(num_classes=len(self.actions))
            self.snn.load_state_dict(state["snn"])
        self.cnn.load_state_dict(state["cnn"])
        self.cnn.eval()
        self.snn.eval()
        self.checkpoint_mtime = mtime

        title = f"Go2 | {self.checkpoint.name}"
        if "selector_epoch" in state:
            title += f" | selector epoch {state['selector_epoch']}"
        self.window.title(title)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        if self.start_position is not None:
            self.data.qpos[:2] = self.start_position
        mujoco.mj_forward(self.model, self.data)
        self.features.clear()
        self.action = "stop"
        self.tick = 0
        self.recovery_ticks = 0
        self.status.configure(text="Collecting 8 frames...")

    def read_depth(self):
        self.depth_renderer.update_scene(self.data, camera="front_cam")
        return self.depth_renderer.render()

    def display_depth(self):
        depth = self.read_depth()
        center = depth[depth.shape[0] // 2, depth.shape[1] // 2]
        clipped = np.nan_to_num(depth, nan=4.0, posinf=4.0, neginf=0.0).clip(0.0, 4.0)
        gray = (255 * (1 - clipped / 4.0)).astype(np.uint8)
        image = Image.fromarray(gray).resize((480, 270)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line((232, 135, 248, 135), fill="#36e5d3", width=2)
        draw.line((240, 127, 240, 143), fill="#36e5d3", width=2)
        self.depth_view.frame_image = ImageTk.PhotoImage(image, master=self.window)
        self.depth_view.configure(image=self.depth_view.frame_image)
        distance = f"{center:.2f} m" if np.isfinite(center) else "unavailable"
        self.depth_status.configure(text=f"Center depth: {distance} | Near is bright (0–4 m)")

    @torch.inference_mode()
    def update_action(self):
        depth = self.read_depth()

        # Use the same preprocessing as train_policy.py.
        image = torch.from_numpy(depth.copy()).float()[None, None]
        self.features.append(self.cnn(preprocess_depth(image))[0])

        if len(self.features) == 8:
            sequence = torch.stack(tuple(self.features)).unsqueeze(0)
            if self.goal_selector is None:
                scores = self.snn(sequence)[0]
            else:
                rates = self.snn.spike_features(sequence)
                goal = torch.from_numpy(waypoint_features(self.data, self.waypoint)).unsqueeze(0)
                scores = self.goal_selector(torch.cat((rates, goal), dim=1))[0]
            self.action = self.actions[scores.argmax().item()]
            if self.action == "stop" and self.goal_selector is None:
                self.recovery_ticks = REVERSE_TICKS + TURN_TICKS
            self.status.configure(text=f"SNN action: {self.action}")

    def update_simulation(self):
        start = time.perf_counter()

        if self.follow_checkpoint and start >= self.next_checkpoint_poll:
            self.next_checkpoint_poll = start + 1
            self.load_checkpoint()

        # Apply two 20 ms pose updates per screen refresh.
        for _ in range(2):
            if self.waypoint is not None and np.linalg.norm(self.data.qpos[:2] - self.waypoint) <= ARRIVAL_RADIUS:
                movement = "stop"
                self.status.configure(text="Waypoint reached")
                break
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
        self.display_depth()
        x, y = self.data.qpos[:2]
        pose_text = f"Position: x {x:.2f} m, y {y:.2f} m | Command: {movement}"
        if self.waypoint is not None:
            distance = np.linalg.norm(self.data.qpos[:2] - self.waypoint)
            pose_text += f" | Goal: {distance:.2f} m away"
        self.pose_status.configure(text=pose_text)

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
    parser.add_argument("--scene", type=Path, help="MuJoCo scene to preview")
    parser.add_argument("--follow-checkpoint", action="store_true", help="Reload newer saved weights")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="Waypoint in world meters")
    args = parser.parse_args()
    if args.goal is not None and not np.isfinite(args.goal).all():
        parser.error("--goal requires finite coordinates")

    torch.set_num_threads(2)
    window = tk.Tk()
    PolicyView(window, args.checkpoint, args.scene, args.follow_checkpoint, args.goal)
    window.mainloop()


if __name__ == "__main__":
    main()
