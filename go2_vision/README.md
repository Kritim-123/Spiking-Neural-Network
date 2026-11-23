# Go2 indoor camera

The scene uses the Unitree Go2 model and meshes from MuJoCo Menagerie. The furnished indoor layout comes from the local Go2-Learning project. A partition closes the corridor ahead and the left doorway is filled, leaving the right doorway open. The Go2 base carries a front camera with a 75° vertical field of view, angled 25° to the right to keep the doorway in frame.

From this folder with your Python environment active, run:

```powershell
python -m pip install -r requirements.txt
python capture_camera.py
```

The script places the Go2 at a sample pose near the turn and captures 320 × 180 images. Look in `camera_output` for `front_rgb.png`, `front_depth.png`, and `front_depth.npy`. The NPY contains raw depth in metres for model input. The depth PNG is a preview with nearer surfaces shown brighter.

To inspect the whole scene interactively:

```powershell
python -m mujoco.viewer --mjcf=scene_indoor.xml
```

For live vision, import `capture_frames` from `capture_camera.py` and pass your simulation's current `MjData` and a persistent `mujoco.Renderer` on each camera tick. Your existing locomotion policy can drive the scene.

Go2 source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2


