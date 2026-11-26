from math import atan2, cos, sin

import mujoco


def apply_pose_command(model, data, command, dt=0.02):
    forward, sideways, yaw_rate = command

    # Read the heading from the upright Go2's base quaternion (w, x, y, z).
    w, _, _, z = data.qpos[3:7]
    yaw = 2 * atan2(z, w)

    # Convert robot-relative speeds into movement along the scene's axes.
    data.qpos[0] += (forward * cos(yaw) - sideways * sin(yaw)) * dt
    data.qpos[1] += (forward * sin(yaw) + sideways * cos(yaw)) * dt

    half_yaw = (yaw + yaw_rate * dt) / 2
    data.qpos[3:7] = (cos(half_yaw), 0.0, 0.0, sin(half_yaw))

    mujoco.mj_forward(model, data)