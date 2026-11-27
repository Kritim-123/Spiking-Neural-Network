from math import atan2, cos, hypot, radians, sin

import numpy as np

ARRIVAL_RADIUS = 0.25


def waypoint_features(data, goal):
    """Return sine and cosine of relative bearing, and distance in meters."""
    dx, dy = np.asarray(goal) - data.qpos[:2]
    distance = hypot(dx, dy)
    rotation = data.body("base").xmat.reshape(3, 3)
    yaw = atan2(rotation[1, 0], rotation[0, 0])
    bearing = atan2(dy, dx) - yaw if distance else 0.0
    return np.array([sin(bearing), cos(bearing), distance], dtype=np.float32)


def draw_waypoint(scene, goal):
    """Draw a goal marker in an overhead render without changing the model."""
    import mujoco

    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.19), np.array([*goal, 0.22]), np.eye(3).ravel(),
            np.array([1.0, 0.63, 0.08, 1.0], dtype=np.float32),
        )
        scene.ngeom += 1
