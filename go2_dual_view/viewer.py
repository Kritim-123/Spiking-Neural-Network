"""View the Go2 scene and its head camera side by side."""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import mujoco
from PIL import Image, ImageTk


class DualView:
    def __init__(self, window):
        self.window = window
        scene_path = Path(__file__).resolve().parent.parent / "go2_vision" / "scene_indoor.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        # Match the pose used for the original head-camera demo.
        self.data.qpos[0] = 4.55
        mujoco.mj_forward(self.model, self.data)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.drag_position = None

        panels = ttk.Frame(window, padding=12)
        panels.pack()
        global_panel = ttk.LabelFrame(panels, text="Global view", padding=6)
        global_panel.grid(row=0, column=0, sticky="n", padx=(0, 10))
        head_panel = ttk.LabelFrame(panels, text="Go2 head camera", padding=6)
        head_panel.grid(row=0, column=1, sticky="n")

        self.global_view = ttk.Label(global_panel)
        self.global_view.pack()
        self.head_view = ttk.Label(head_panel)
        self.head_view.pack()
        ttk.Label(global_panel, text="Drag to orbit · Scroll to zoom").pack(pady=5)
        buttons = ttk.Frame(global_panel)
        buttons.pack()
        ttk.Button(buttons, text="Robot and doorway", command=self.show_robot).pack(side="left")
        ttk.Button(buttons, text="Whole room", command=self.show_room).pack(side="left", padx=6)

        self.global_renderer = mujoco.Renderer(self.model, height=432, width=640)
        self.head_renderer = mujoco.Renderer(self.model, height=270, width=480)
        self.global_view.bind("<ButtonPress-1>", self.start_drag)
        self.global_view.bind("<B1-Motion>", self.orbit_camera)
        self.global_view.bind("<MouseWheel>", self.zoom_camera)
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.show_robot()
        self.display(self.head_view, self.head_renderer, "front_cam")

    def display(self, label, renderer, camera):
        renderer.update_scene(self.data, camera=camera)
        label.frame_image = ImageTk.PhotoImage(Image.fromarray(renderer.render()), master=self.window)
        label.configure(image=label.frame_image)

    def show_robot(self):
        self.camera.lookat[:] = [4.75, -0.37, 0.15]
        self.camera.distance = 3.9
        self.camera.azimuth = 90
        self.camera.elevation = -88
        self.display(self.global_view, self.global_renderer, self.camera)

    def show_room(self):
        self.camera.lookat[:] = [2, 0, 0]
        self.camera.distance = 17
        self.camera.azimuth = 90
        self.camera.elevation = -89
        self.display(self.global_view, self.global_renderer, self.camera)

    def start_drag(self, event):
        self.drag_position = (event.x, event.y)

    def orbit_camera(self, event):
        old_x, old_y = self.drag_position
        self.drag_position = (event.x, event.y)
        self.camera.azimuth -= (event.x - old_x) * 0.4
        elevation = self.camera.elevation + (event.y - old_y) * 0.4
        self.camera.elevation = max(-89, min(-10, elevation))
        self.display(self.global_view, self.global_renderer, self.camera)

    def zoom_camera(self, event):
        distance = self.camera.distance * 0.9 ** (event.delta / 120)
        self.camera.distance = max(0.7, min(30, distance))
        self.display(self.global_view, self.global_renderer, self.camera)

    def close(self):
        self.global_renderer.close()
        self.head_renderer.close()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    DualView(root)
    root.mainloop()
