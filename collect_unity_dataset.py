"""
Step A - collect a waypoint dataset from Unity.

    python collect_unity_dataset.py        (then press Play in Unity)

An expert follows the planner's path (with some random noise so the
data also contains mistakes / recoveries). Every step we save:
    image      uint8  [60, 80, 3]
    goal       float  [distance, angle]
    waypoints  float  [3, 2]   (angle, distance) along the planned path
    episode id

Output: data/unity_waypoint_dataset.npz
"""

import os
import time
import numpy as np

from envs.unity_navigation_env import UnityNavigationEnv
from envs.unity_planner import UnityGridPlanner


NUM_EPISODES = 400
SEED_OFFSET = 10_000          # keep away from SAC train / eval seeds
NOISE_PROBABILITY = 0.25      # chance of a random action at each step
NOISE_HOLD_STEPS = 5          # random actions are held for a few steps
OUTPUT_PATH = os.path.join("data", "unity_waypoint_dataset.npz")


def expert_action(planner, info):
    target = planner.lookahead_point(info["agent_pos"], info["goal_pos"])
    angle, _ = planner.relative_angle_distance(info["agent_pos"], info["agent_yaw"], target)
    omega = float(np.clip(2.0 * angle, -1.0, 1.0))
    v = 0.8 if abs(angle) < 0.5 else 0.15
    return np.array([v, omega], dtype=np.float32)


def main():
    rng = np.random.default_rng(0)
    env = UnityNavigationEnv(port=5000)
    planner = UnityGridPlanner()

    images, goals, waypoints, episode_ids = [], [], [], []
    successes = 0
    start = time.time()

    try:
        for episode in range(NUM_EPISODES):

            observation, info = env.reset(seed=SEED_OFFSET + episode)
            planner.build(info["obstacles"], info["goal_pos"])

            terminated = truncated = False
            noise_steps_left = 0
            noise_action = None

            while True:
                # ---- record current step ----
                images.append(observation["image"])
                goals.append(observation["goal"])
                waypoints.append(
                    planner.waypoints(info["agent_pos"], info["agent_yaw"], info["goal_pos"])
                )
                episode_ids.append(episode)

                if terminated or truncated:
                    break

                # ---- choose action ----
                if noise_steps_left == 0 and rng.random() < NOISE_PROBABILITY:
                    noise_steps_left = NOISE_HOLD_STEPS
                    noise_action = np.array(
                        [rng.uniform(0, 1), rng.uniform(-1, 1)], dtype=np.float32
                    )

                if noise_steps_left > 0:
                    action = noise_action
                    noise_steps_left -= 1
                else:
                    action = expert_action(planner, info)

                observation, reward, terminated, truncated, info = env.step(action)

            successes += int(info["success"])

            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                      f"frames {len(images)} | "
                      f"success {successes / (episode + 1):.2f} | "
                      f"{time.time() - start:.0f}s")
    finally:
        env.close()

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        images=np.stack(images).astype(np.uint8),
        goals=np.stack(goals).astype(np.float32),
        waypoints=np.stack(waypoints).astype(np.float32),
        episode_ids=np.array(episode_ids, dtype=np.int32)
    )
    print(f"\nSaved {len(images)} frames from {NUM_EPISODES} episodes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
