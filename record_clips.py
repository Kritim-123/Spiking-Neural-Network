import argparse
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

import mujoco
import numpy as np

from go2_vision.capture_camera import capture_frames
from go2_vision.waypoint import draw_waypoint, waypoint_features

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "go2_vision" / "scene_indoor.xml"
ACTIONS = ("forward", "stop", "turn_left", "turn_right")
KEYS = {"w": 0, "s": 1, "a": 2, "d": 3}
FRAME_COUNT = 8
TRAVEL_METERS = 0.7


def room_bounds(model):
    west, south, east, north = (
        model.geom(name) for name in ("wall_west", "wall_south", "wall_east", "wall_north")
    )
    return (
        np.array([west.pos[0] + west.size[0], south.pos[1] + south.size[1]]),
        np.array([east.pos[0] - east.size[0], north.pos[1] - north.size[1]]),
    )


def random_path(model, data, rng, bounds):
    lower, upper = bounds
    robot_geoms = model.body_rootid[model.geom_bodyid] == model.body("base").id
    obstacle_geoms = ~robot_geoms & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    # Check every camera pose and the midpoint between each pair.
    distances = np.linspace(0.0, TRAVEL_METERS, 2 * FRAME_COUNT - 1)

    for _ in range(300):
        start = rng.uniform(lower, upper)
        yaw = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        positions = start + distances[:, None] * direction
        if np.any(positions[-1] < lower) or np.any(positions[-1] > upper):
            continue

        data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        for position in positions:
            data.qpos[:2] = position
            mujoco.mj_forward(model, data)
            if any(
                (robot_geoms[first] and obstacle_geoms[second])
                or (robot_geoms[second] and obstacle_geoms[first])
                for first, second in data.contact.geom
                if first >= 0 and second >= 0
            ):
                break
        else:
            return positions[::2], yaw

    raise RuntimeError("Could not find a clear path for an eight-frame clip.")


def capture_clip(renderer, model, data, positions):
    frames = []
    for position in positions:
        data.qpos[:2] = position
        mujoco.mj_forward(model, data)
        _, depth = capture_frames(renderer, data)
        frames.append(depth.copy())

    return np.stack(frames).astype(np.float32)


def capture_overhead(renderer, model, data, goal=None):
    camera = "top_down"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = model.stat.center
        camera.distance = 2.5 * model.stat.extent
        camera.azimuth = 90
        camera.elevation = -89

    renderer.update_scene(data, camera=camera)
    scene = renderer.scene
    if scene.ngeom < scene.maxgeom:
        base = data.body("base")
        origin = base.xpos + np.array([0.0, 0.0, 0.27])
        tip = origin + 0.85 * base.xmat.reshape(3, 3)[:, 0]
        arrow = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), origin,
            np.eye(3).ravel(), np.array([0.05, 0.85, 0.75, 1.0], dtype=np.float32),
        )
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.06, origin, tip)
        scene.ngeom += 1

    if goal is not None:
        draw_waypoint(scene, goal)
    return renderer.render()


def label_clip(clip, overhead, number, count, goal=None):
    import matplotlib.pyplot as plt

    choice = None
    with plt.rc_context({"keymap.save": [], "keymap.quit": []}):
        fig = plt.figure(figsize=(14, 5.6))
        grid = fig.add_gridspec(2, 6, width_ratios=(1.25, 1.25, 1, 1, 1, 1))
        overview = fig.add_subplot(grid[:, :2])
        overview.imshow(overhead)
        overview.set_title("Top view at frame 8 | Arrow shows heading")
        overview.axis("off")

        for i, frame in enumerate(clip, start=1):
            row, column = divmod(i - 1, 4)
            ax = fig.add_subplot(grid[row, column + 2])
            ax.imshow(frame, cmap="gray", vmin=0.0, vmax=4.0)
            ax.set_title(f"Frame {i}", weight="bold" if i == FRAME_COUNT else "normal")
            ax.axis("off")

        def select_action(event):
            nonlocal choice
            if event.key in KEYS or event.key == "q":
                choice = event.key
                plt.close(fig)
            elif event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", select_action)
        title = f"Clip {number}/{count} - What should the robot do after frame 8?"
        if goal is not None:
            title += f" | Gold waypoint ({goal[0]:g}, {goal[1]:g})"
        fig.suptitle(title)
        fig.supxlabel("W Forward   S Stop   A Left   D Right   Q Skip   Esc Finish")
        fig.tight_layout()
        plt.show()

    return choice


def predict(clip):
    import torch
    from vision_policy import VisionPolicy

    policy = VisionPolicy()
    weights = torch.load(
        ROOT / "vision_policy.pt", map_location="cpu", weights_only=True
    )
    policy.load_state_dict(weights)
    policy.eval()

    with torch.inference_mode():
        logits = policy(torch.from_numpy(clip).unsqueeze(0))[0]

    print("Action scores:", dict(zip(ACTIONS, logits.tolist())))
    print("Predicted action:", ACTIONS[logits.argmax().item()])


def main():
    parser = argparse.ArgumentParser(description="Collect labeled Go2 depth clips in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--start-x", type=float, default=3.85)
    parser.add_argument("--end-x", type=float, default=4.55)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--random", action="store_true", help="Randomize indoor starting position and heading.")
    parser.add_argument("--count", type=int, default=1, help="Number of clips to preview, including skips.")
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible random clips.")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--goal", nargs=2, type=float, action="append", metavar=("X", "Y"),
                        help="Label a waypoint at (X, Y); repeat to label the same clip for multiple goals")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.count > 1 and (not args.random or args.predict):
        parser.error("--count greater than 1 requires --random without --predict")
    if args.seed is not None and (not args.random or args.seed < 0):
        parser.error("--seed requires --random and a nonnegative integer")
    if args.goal and (args.predict or not np.isfinite(args.goal).all()):
        parser.error("--goal requires finite coordinates and recording mode")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    home = model.key("home").id
    if args.random:
        try:
            bounds = room_bounds(model)
        except KeyError:
            parser.error("--random requires a scene with wall_west, wall_south, wall_east and wall_north")
        rng = np.random.default_rng(args.seed)

    if args.goal:
        try:
            lower, upper = room_bounds(model)
        except KeyError:
            pass  # Custom scenes without named boundary walls can still record fixed poses.
        else:
            if np.any(np.asarray(args.goal) <= lower) or np.any(np.asarray(args.goal) >= upper):
                parser.error("waypoints must be inside the room")

    dataset = "waypoint_samples" if args.goal else "training_samples"
    split_folder = ROOT / dataset / args.split
    batch_folder = None
    saved = 0
    with ExitStack() as renderers:
        renderer = renderers.enter_context(mujoco.Renderer(model, height=180, width=320))
        if not args.predict:
            overhead_renderer = renderers.enter_context(mujoco.Renderer(
                model,
                height=min(400, model.vis.global_.offheight),
                width=min(600, model.vis.global_.offwidth),
            ))
        for number in range(1, args.count + 1):
            mujoco.mj_resetDataKeyframe(model, data, home)
            if args.random:
                positions, yaw = random_path(model, data, rng, bounds)
                x, y = positions[0]
                print(f"Clip {number}: start ({x:.2f}, {y:.2f}) m, heading {np.degrees(yaw):.0f}°")
            else:
                positions = np.column_stack((
                    np.linspace(args.start_x, args.end_x, FRAME_COUNT),
                    np.full(FRAME_COUNT, data.qpos[1] + args.y_offset),
                ))
            clip = capture_clip(renderer, model, data, positions)

            if args.predict:
                predict(clip)
                return

            labels, goals, features = [], [], []
            for goal in args.goal or [None]:
                overhead = capture_overhead(overhead_renderer, model, data, goal)
                choice = label_clip(clip, overhead, number, args.count, goal)
                if choice is None:
                    break
                if choice in KEYS:
                    labels.append(KEYS[choice])
                    if goal is not None:
                        goals.append(goal)
                        features.append(waypoint_features(data, goal))

            if not labels:
                if choice is None:
                    break
                print("Clip skipped.")
                continue

            if batch_folder is None:
                split_folder.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                batch_folder = Path(mkdtemp(prefix=f"batch_{timestamp}_", dir=split_folder))
            path = batch_folder / f"{uuid4().hex}.npz"
            if args.goal:
                np.savez_compressed(
                    path, depth=clip, action=np.asarray(labels, dtype=np.int64),
                    goals=np.asarray(goals, dtype=np.float32), goal_features=np.stack(features),
                )
            else:
                np.savez_compressed(path, depth=clip, action=np.int64(labels[0]))
            saved += 1
            print(f"Saved {', '.join(ACTIONS[action] for action in labels)}: {path}")
            if choice is None:
                break

    print(f"Session complete. Saved {saved} clip(s).")
    if batch_folder is not None:
        print(f"Batch folder: {batch_folder}")


if __name__ == "__main__":
    main()
