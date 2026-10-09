"""
UnityNavigationEnv
==================

Drop-in replacement for envs/navigation_env.py (MiniWorld) that runs the
simulation in Unity instead.

It has the same Gymnasium interface:
    observation = {"image": uint8 (60, 80, 3), "goal": float32 [distance, angle]}
    action      = [v in [0, 1], omega in [-1, 1]]
    reset(seed=...) -> observation, info
    step(action)    -> observation, reward, terminated, truncated, info

Python is the TCP server. Start Python FIRST, then press Play in Unity.
"""

import socket
import struct

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.unity_planner import UnityGridPlanner


CMD_RESET = 0
CMD_STEP = 1
CMD_CLOSE = 2


class UnityNavigationEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        host="127.0.0.1",
        port=5000,
        max_episode_steps=400,
        check_solvable=True,
        max_reset_attempts=20
    ):
        super().__init__()

        # Informational only - Unity enforces the real limit
        # (NavigationEnvironment.maxEpisodeSteps). Keep them equal.
        self.max_episode_steps = max_episode_steps

        # Layout checking: a seed whose layout has no path is replaced by a
        # deterministic "re-roll" seed, so the same seed always gives the
        # same (solvable) episode.
        self.check_solvable = check_solvable
        self.max_reset_attempts = max_reset_attempts
        # clearance 0.3 = agent radius (0.25) + 5 cm -> near-true shortest path for SPL
        self.planner = UnityGridPlanner(clearance=0.3)
        self.shortest_path_length = None

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(60, 80, 3), dtype=np.uint8
            ),
            "goal": spaces.Box(
                low=np.array([0.0, -np.pi], dtype=np.float32),
                high=np.array([30.0, np.pi], dtype=np.float32),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Latest world state (useful for evaluation / waypoint targets)
        self.agent_pos = None
        self.agent_yaw = None
        self.goal_pos = None
        self.obstacles = None

        # --------------------------------------------------
        # Start server and wait for Unity
        # --------------------------------------------------

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)

        print(f"[UnityEnv] Waiting for Unity on {host}:{port} ... press Play in Unity.")

        self.connection, address = self.server.accept()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print(f"[UnityEnv] Unity connected from {address}")

    # ======================================================
    # Gymnasium API
    # ======================================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))

        for attempt in range(self.max_reset_attempts):

            layout_seed = (int(seed) + attempt * 1_000_003) % (2**31 - 1)

            self._send(struct.pack("<Bi", CMD_RESET, layout_seed))
            observation, _, _, _, info = self._receive()

            self.planner.build(info["obstacles"], info["goal_pos"])
            self.shortest_path_length = self.planner.path_length(info["agent_pos"], info["goal_pos"])

            if not self.check_solvable or np.isfinite(self.shortest_path_length):
                break
        else:
            print(f"[UnityEnv] warning: seed {seed} gave no solvable layout")

        info["layout_seed"] = layout_seed
        info["shortest_path_length"] = self.shortest_path_length

        return observation, info

    def step(self, action):

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        v = float(np.clip(action[0], 0.0, 1.0))
        omega = float(np.clip(action[1], -1.0, 1.0))

        self._send(struct.pack("<Bff", CMD_STEP, v, omega))

        return self._receive()

    def close(self):

        try:
            self._send(struct.pack("<B", CMD_CLOSE))
        except OSError:
            pass

        try:
            self.connection.close()
        finally:
            self.server.close()

    # ======================================================
    # Networking
    # ======================================================

    def _send(self, payload):
        self.connection.sendall(struct.pack("<I", len(payload)) + payload)

    def _receive_exactly(self, num_bytes):

        data = bytearray()

        while len(data) < num_bytes:
            chunk = self.connection.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Unity closed the connection.")
            data.extend(chunk)

        return bytes(data)

    def _receive(self):

        length = struct.unpack("<I", self._receive_exactly(4))[0]
        packet = self._receive_exactly(length)

        offset = 0

        def read(fmt):
            nonlocal offset
            values = struct.unpack_from(fmt, packet, offset)
            offset += struct.calcsize(fmt)
            return values

        width, height, channels, image_length = read("<iiii")

        image = np.frombuffer(
            packet, dtype=np.uint8, count=image_length, offset=offset
        )
        offset += image_length

        # Unity's ReadPixels returns rows bottom-to-top -> flip to top-to-bottom
        image = np.flipud(
            image.reshape(height, width, channels)
        ).copy()

        distance, angle, reward = read("<fff")
        terminated, truncated, success, collision = read("<????")

        agent_x, agent_z, agent_yaw, goal_x, goal_z = read("<fffff")

        (obstacle_count,) = read("<i")
        obstacles = np.array(
            [read("<ffff") for _ in range(obstacle_count)],
            dtype=np.float32
        ).reshape(obstacle_count, 4)

        (step_count,) = read("<i")

        self.agent_pos = np.array([agent_x, agent_z], dtype=np.float32)
        self.agent_yaw = agent_yaw
        self.goal_pos = np.array([goal_x, goal_z], dtype=np.float32)
        self.obstacles = obstacles

        observation = {
            "image": image,
            "goal": np.array([distance, angle], dtype=np.float32)
        }

        info = {
            "success": bool(success),
            "collision": bool(collision),
            "step": step_count,
            "agent_pos": self.agent_pos,
            "agent_yaw": agent_yaw,
            "goal_pos": self.goal_pos,
            "obstacles": obstacles,
            "shortest_path_length": self.shortest_path_length
        }

        return observation, float(reward), bool(terminated), bool(truncated), info
