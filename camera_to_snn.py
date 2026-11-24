from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from cnn.cnn import depth_encoder
from go2_vision.capture_camera import capture_frames

ROOT = Path(__file__).resolve().parent

# SpikingNetwork.py imports lif_step.py from the same folder.
sys.path.insert(0, str(ROOT / "snn"))
from SpikingNetwork import SpikingNetwork


def main():
    #This scene loads the shared go2_vision/go2.xml
    scene = ROOT / "go2_vision" / "scene_indoor.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    #Capture eight frames at different positions approaching the wall.
    frames = []
    with mujoco.Renderer(model, height = 180, width = 320) as renderer:
        for x in np.linspace(3.85, 4.55, 8):
            data.qpos[0] = x
            mujoco.mj_forward(model, data)
            _, depth = capture_frames(renderer, data)
            frames.append(depth)

            print("raw depth frame:", frames[0].shape, frames[0].dtype)

            #Stack frames and add the single depth channel: [8, 1, 180, 320]
            clip = torch.from_numpy(np.stack(frames)).float().unsqueeze(1)
            clip = torch.nan_to_num(clip, nan=4.0, posinf=4.0, neginf=0.0)
            clip = clip.clamp(0.0, 4.0) / 4.0
            clip = F.interpolate(clip, size = (90,160), mode="area")

            snn = SpikingNetwork(input_size = 32, hidden_size = 128, num_classes = 3)
            depth_encoder.eval()
            snn.eval()

            with torch.inference_mode():
                frame_features = depth_encoder(clip) # [8,32]
                snn_input = frame_features.unsqueeze(0) # [1,8,32]
                logits = snn(snn_input) # [1,3]

                print("CNN input: ", tuple(clip.shape))
                print("CNN output: ", tuple(frame_features.shape))
                print("SNN input: ", tuple(snn_input.shape))
                print("SNN output:", tuple(logits.shape))

if __name__ == "__main__":
    main()