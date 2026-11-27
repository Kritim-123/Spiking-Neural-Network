# SNN Go2 training data

Help teach a simulated Unitree Go2 what to do next indoors. The recorder places the robot at different positions and headings, then shows eight frames from its front depth camera. You choose the next action and the recorder saves the frames and your label together. The scene and robot model are included in this repo.

## 1. Set up the project

You'll need Git, Python 3.14, and a desktop session that can open a preview window. You can collect clips without a physical robot, a model checkpoint, or PyTorch.

Clone the repo and open its folder. If you already have a copy, open it and run `git pull` to get the latest recorder.

```sh
git clone https://github.com/Kritim-123/Spiking-Neural-Network.git
cd Spiking-Neural-Network
```

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r go2_vision/requirements.txt matplotlib
```

**macOS or Linux**

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r go2_vision/requirements.txt matplotlib
```

## 2. Open the recorder

Start with one preview to check that the window opens. Press `Q` in the window to skip the test clip.

| Terminal | Preview one clip | Record a batch of up to 50 clips |
| --- | --- | --- |
| Windows PowerShell | `.\.venv\Scripts\python.exe record_clips.py --random --count 1` | `.\.venv\Scripts\python.exe record_clips.py --random --count 50` |
| macOS or Linux | `.venv/bin/python record_clips.py --random --count 1` | `.venv/bin/python record_clips.py --random --count 50` |

Run these commands from the project folder. The preview opens separately from the terminal, so click the preview if the keys don't respond. The recorder counts skipped previews toward `--count`. Start a new run whenever you want to collect more clips.

To record in the workshop layout, add `--scene go2_vision/scene_workshop.xml` to a random recording command. The workshop has staggered openings, crates, and a dead-end service bay. For example, with your Python environment active:

```powershell
python record_clips.py --scene go2_vision/scene_workshop.xml --random --count 50
```

For a larger layout with a loop corridor, offset doorways, furnished rooms, and a blind bay, record in the training center:

```powershell
python record_clips.py --scene go2_vision/scene_training_center.xml --random --count 50
```

For a harder layout with angled racks, narrow gates, blind loading bays, and low pallets, use the warehouse:

```powershell
python record_clips.py --scene go2_vision/scene_warehouse.xml --random --count 50
```

## 3. Label the next action

Look at **frame 8**, the bottom-right depth image, and decide what the robot should do immediately afterward. The overhead panel shows the robot's position at frame 8. Its arrow shows the direction it faces. Use left and right from the robot's point of view.

| Key | Action | When to use it |
| --- | --- | --- |
| `W` | Forward | The path straight ahead is clear. |
| `A` | Turn left | The view shows a clear opening to the left that the robot should take. |
| `D` | Turn right | The view shows a clear opening to the right that the robot should take. |
| `S` | Stop | The robot should stop because it has no clear way forward or around the obstacle. |
| `Q` | Skip | The view is unclear or several actions seem equally reasonable. |
| `Esc` | Finish | End the session and keep everything you've saved so far. |

Use the same labeling rules for each batch. A saved sample is an `.npz` file containing eight depth frames and one action label.

## Record and train waypoint navigation

Set one or more goal coordinates in world meters. Each gold marker appears in the overhead view. Repeat `--goal` to label the same depth clip for different destinations, one prompt at a time. For each prompt, choose the immediate action toward the marked goal. If a wall blocks the direct route, label the next action along a clear detour. Use `S` when the robot has arrived or should hold, and `Q` to skip the current goal.

```powershell
python .\record_clips.py --scene .\go2_vision\scene_training_center.xml --random --count 50 --goal 0 2.7 --goal 0 -2.7
```

Waypoint clips go to `waypoint_samples/train/`. Targets sharing a depth clip stay together in the train/validation split. Use `--split val` when recording an independent validation set. Without separate validation recordings, the trainer holds out 20% of the distinct depth clips.

Train only the new selector using an existing three-expert checkpoint. Replace the example names with the filenames from your runs.

```powershell
python .\train_waypoint.py --expert-checkpoint .\checkpoints\expert_policy_YOUR_RUN.pt
python -m go2_dual_view.policy_viewer --checkpoint .\checkpoints\waypoint_policy_YOUR_RUN.pt --scene .\go2_vision\scene_training_center.xml --goal 0 2.7
```

The viewer displays the distance to the goal and stops within 25 cm. Use the same scene and varied goals to check whether the policy reaches its destination. The selector learns local movement decisions from your labels and can get stuck on routes it has not learned.

## 4. Share the batch

The first saved clip in each run creates a new folder under `training_samples/train/`, such as `batch_2026-11-24_12-00-00_ab12cd34`. The terminal prints its full path on the **Batch folder** line when the run ends.

Zip that entire batch folder and send the ZIP to Kritim. If you record several batches, include each batch folder with its name intact. For clips meant for validation, add `--split val` to the command and keep those folders under `training_samples/val/`.
