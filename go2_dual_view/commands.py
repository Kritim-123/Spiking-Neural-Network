# Each tuple is (forward speed in m/s, sideways speed in m/s, yaw rate in rad/s).
COMMANDS = {
    "forward": (0.35, 0.0, 0.0),
    "stop": (-0.15, 0.0, 0.0),
    "turn_left": (0.0, 0.0, 0.4),
    "turn_right": (0.0, 0.0, -0.4),
}