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

## 3. Label the next action

Look at **frame 8**, the bottom-right image, and decide what the robot should do immediately afterward. Use left and right from the robot's point of view.

| Key | Action | When to use it |
| --- | --- | --- |
| `W` | Forward | The path straight ahead is clear. |
| `A` | Turn left | The view shows a clear opening to the left that the robot should take. |
| `D` | Turn right | The view shows a clear opening to the right that the robot should take. |
| `S` | Stop | The robot should stop because it has no clear way forward or around the obstacle. |
| `Q` | Skip | The view is unclear or several actions seem equally reasonable. |
| `Esc` | Finish | End the session and keep everything you've saved so far. |

Use the same labeling rules for each batch. A saved sample is an `.npz` file containing eight depth frames and one action label.

## 4. Share the batch

The first saved clip in each run creates a new folder under `training_samples/train/`, such as `batch_2026-11-24_12-00-00_ab12cd34`. The terminal prints its full path on the **Batch folder** line when the run ends.

Zip that entire batch folder and send the ZIP to Kritim. If you record several batches, include each batch folder with its name intact. For clips meant for validation, add `--split val` to the command and keep those folders under `training_samples/val/`.
