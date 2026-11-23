"""Save RGB and metric-depth frames from the Go2 torso camera."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def capture_frames(renderer, data):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera="front_cam")
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    return rgb, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("camera_output"))
    args = parser.parse_args()

    scene_path = Path(__file__).resolve().with_name("scene_indoor.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    # Place the robot just before the right-hand doorway for this sample.
    data.qpos[0] = 4.55
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, height=180, width=320) as renderer:
        rgb, depth = capture_frames(renderer, data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output_dir / "front_rgb.png")
    np.save(args.output_dir / "front_depth.npy", depth)

    # Display nearer geometry as brighter. The NPY retains metric depth.
    depth_preview = (255 * (1 - np.clip(depth / 4.0, 0, 1))).astype(np.uint8)
    Image.fromarray(depth_preview).save(args.output_dir / "front_depth.png")
    print(f"Saved RGB {rgb.shape} and depth {depth.shape} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
