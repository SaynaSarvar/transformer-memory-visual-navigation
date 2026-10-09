"""
End-to-end test of the Unity environment.

1. Run:   python test_unity_env.py
2. Then press Play in Unity.

It runs a few episodes with a simple "turn toward the goal and drive"
controller (no learning). If the agent drives to the goal, then the
image, goal angle sign, actions, reward and reset logic all work.
"""

import time
import numpy as np

from envs.unity_navigation_env import UnityNavigationEnv


def goal_seeking_action(observation):
    distance, angle = observation["goal"]
    omega = float(np.clip(2.0 * angle, -1.0, 1.0))      # + angle -> turn right
    v = 0.8 if abs(angle) < 0.6 else 0.2                 # slow down while turning
    return np.array([v, omega], dtype=np.float32)


def main():
    env = UnityNavigationEnv(port=5000)

    try:
        for episode in range(5):

            observation, info = env.reset(seed=episode)

            if episode == 0:
                image = observation["image"]
                print("Image shape:", image.shape, image.dtype)
                try:
                    from PIL import Image
                    Image.fromarray(image).save("unity_first_frame.png")
                    print("Saved unity_first_frame.png (check it is NOT upside down)")
                except ImportError:
                    print("Pillow not installed - skipping image save.")

            print(f"\nEpisode {episode}: start goal = {observation['goal']}, "
                  f"agent = {info['agent_pos']}, goal = {info['goal_pos']}")

            total_reward = 0.0
            steps = 0
            collisions = 0
            start = time.time()
            terminated = truncated = False

            while not (terminated or truncated):
                action = goal_seeking_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                collisions += int(info["collision"])

            fps = steps / (time.time() - start)
            print(f"  steps={steps}  reward={total_reward:.2f}  "
                  f"success={info['success']}  collisions={collisions}  "
                  f"speed={fps:.0f} steps/s")
    finally:
        env.close()


if __name__ == "__main__":
    main()
