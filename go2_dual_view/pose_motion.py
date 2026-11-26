from math import atan2, cos, sin

import mujoco


def apply_pose_command(model, data, command, dt=0.02):
    forward, sideways, yaw_rate = command
    if forward == sideways == yaw_rate == 0:
        return

    previous_pose = data.qpos[:7].copy()

    # Read the heading from the upright Go2's base quaternion (w, x, y, z).
    w, _, _, z = data.qpos[3:7]
    yaw = 2 * atan2(z, w)

    # Convert robot-relative speeds into movement along the scene's axes.
    data.qpos[0] += (forward * cos(yaw) - sideways * sin(yaw)) * dt
    data.qpos[1] += (forward * sin(yaw) + sideways * cos(yaw)) * dt

    half_yaw = (yaw + yaw_rate * dt) / 2
    data.qpos[3:7] = (cos(half_yaw), 0.0, 0.0, sin(half_yaw))

    mujoco.mj_forward(model, data)

    # Pose updates bypass the physics solver, so reject moves into obstacles.
    robot_geoms = model.body_rootid[model.geom_bodyid] == model.body("base").id
    obstacle_geoms = ~robot_geoms & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    # The office doorway's 2 cm sill is a walkable surface in this preview.
    sill = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "office_threshold")
    if sill >= 0:
        obstacle_geoms[sill] = False

    blocked = any(
        (robot_geoms[first] and obstacle_geoms[second])
        or (robot_geoms[second] and obstacle_geoms[first])
        for first, second in data.contact.geom
        if first >= 0 and second >= 0
    )
    if blocked:
        data.qpos[:7] = previous_pose
        mujoco.mj_forward(model, data)
