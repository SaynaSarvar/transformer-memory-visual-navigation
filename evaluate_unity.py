"""
Evaluate trained agents and baselines on the SAME fixed Unity layouts.

    python3 evaluate_unity.py                      (then press Play in Unity)
    python3 evaluate_unity.py --runs full no_memory no_pretrain --episodes 100

Methods
-------
  heuristic      : turn toward the goal angle and drive (no vision, no memory, no learning)
  <run name>     : a SAC run from checkpoints_unity_v2/<run name>/ (deterministic policy)
  expert         : follows the map-based planner (uses the true map -> upper bound)

Metrics (per method, over all evaluation episodes)
-------
  Success   : fraction of episodes that reach the goal
  SPL       : Success weighted by Path Length = mean( S * L / max(P, L) )
              L = shortest path length, P = distance actually travelled
  Coll/ep   : mean number of collision steps per episode
  CollEp    : fraction of episodes with at least one collision
  Steps     : mean episode length (successful episodes only)

Outputs
-------
  results/eval_episodes.csv   one row per (method, episode)
  results/eval_summary.csv    one row per method
"""

import os
import csv
import json
import argparse

import numpy as np
import torch

from envs.unity_navigation_env import UnityNavigationEnv
from envs.unity_planner import UnityGridPlanner
from agents.navigation_agent import NavigationAgent
from models.sac_actor import SACActor


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RUNS_DIR = "checkpoints_unity_v2"


# ============================================================
# Policies (all have reset(observation, info) and act(observation, info))
# ============================================================

class HeuristicPolicy:
    name = "heuristic"

    def reset(self, observation, info):
        pass

    def act(self, observation, info):
        distance, angle = observation["goal"]
        omega = float(np.clip(2.0 * angle, -1.0, 1.0))
        v = 0.8 if abs(angle) < 0.6 else 0.2
        return np.array([v, omega], dtype=np.float32)


class ExpertPolicy:
    name = "expert"

    def __init__(self):
        self.planner = UnityGridPlanner()

    def reset(self, observation, info):
        self.planner.build(info["obstacles"], info["goal_pos"])

    def act(self, observation, info):
        target = self.planner.lookahead_point(info["agent_pos"], info["goal_pos"])
        angle, _ = self.planner.relative_angle_distance(
            info["agent_pos"], info["agent_yaw"], target
        )
        omega = float(np.clip(2.0 * angle, -1.0, 1.0))
        v = 0.8 if abs(angle) < 0.5 else 0.15
        return np.array([v, omega], dtype=np.float32)


class SACPolicy:

    def __init__(self, run_name, which="best"):
        self.name = run_name
        run_dir = os.path.join(RUNS_DIR, run_name)

        with open(os.path.join(run_dir, "config.json")) as f:
            config = json.load(f)

        self.agent = NavigationAgent(
            sac_agent=None,
            embedding_dim=config["embedding_dim"],
            memory_length=8,
            device=DEVICE
        )
        self.agent.memory_length = config["memory_length"]

        self.agent.observation_encoder.load_state_dict(
            torch.load(os.path.join(run_dir, "observation_encoder.pt"), map_location=DEVICE)
        )
        self.agent.transformer_memory.load_state_dict(
            torch.load(os.path.join(run_dir, "transformer_memory.pt"), map_location=DEVICE)
        )
        self.agent.observation_encoder.eval()
        self.agent.transformer_memory.eval()

        self.actor = SACActor(state_dim=config["embedding_dim"]).to(DEVICE)
        if which == "best":
            checkpoint = torch.load(os.path.join(run_dir, "sac_best_checkpoint.pt"), map_location=DEVICE)
            self.actor.load_state_dict(checkpoint["actor"])
        else:
            self.actor.load_state_dict(
                torch.load(os.path.join(run_dir, "sac_actor_final.pt"), map_location=DEVICE)
            )
        self.actor.eval()

        print(f"Loaded SAC run '{run_name}' ({which}) | memory {config['memory_length']} "
              f"| pretrained {config['pretrained']}")

    def reset(self, observation, info):
        self.agent.reset_memory()

    @torch.no_grad()
    def act(self, observation, info):
        state, _ = self.agent.get_belief_state(observation)
        mean, _ = self.actor(state)
        squashed = torch.tanh(mean)[0].cpu().numpy()
        return np.array([(squashed[0] + 1.0) / 2.0, squashed[1]], dtype=np.float32)


# ============================================================
# Evaluation loop
# ============================================================

def run_episode(env, policy, seed):
    observation, info = env.reset(seed=seed)
    policy.reset(observation, info)

    shortest = info["shortest_path_length"]
    previous_pos = info["agent_pos"].copy()
    travelled = 0.0
    collisions = 0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        travelled += float(np.linalg.norm(info["agent_pos"] - previous_pos))
        previous_pos = info["agent_pos"].copy()
        collisions += int(info["collision"])
        steps += 1

    success = bool(info["success"])
    spl = float(success) * shortest / max(travelled, shortest, 1e-6) if np.isfinite(shortest) else 0.0

    return {
        "method": policy.name,
        "seed": seed,
        "success": int(success),
        "spl": spl,
        "collisions": collisions,
        "steps": steps,
        "path_length": travelled,
        "shortest_path": shortest
    }


def summarize(rows):
    success = np.array([r["success"] for r in rows])
    successful_steps = [r["steps"] for r in rows if r["success"]]
    return {
        "method": rows[0]["method"],
        "episodes": len(rows),
        "success": success.mean(),
        "spl": np.mean([r["spl"] for r in rows]),
        "collisions_per_episode": np.mean([r["collisions"] for r in rows]),
        "collision_episode_rate": np.mean([r["collisions"] > 0 for r in rows]),
        "steps_successful": np.mean(successful_steps) if successful_steps else float("nan")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=["full", "no_memory", "no_pretrain"],
                        help="SAC run names in checkpoints_unity_v2/ (missing ones are skipped)")
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=50_000,
                        help="evaluation seeds (never used for training or data collection)")
    parser.add_argument("--no-heuristic", action="store_true")
    parser.add_argument("--no-expert", action="store_true")
    args = parser.parse_args()

    policies = []
    if not args.no_heuristic:
        policies.append(HeuristicPolicy())
    for run in args.runs:
        if os.path.exists(os.path.join(RUNS_DIR, run, "config.json")):
            policies.append(SACPolicy(run, args.checkpoint))
        else:
            print(f"Skipping '{run}': no {RUNS_DIR}/{run}/config.json")
    if not args.no_expert:
        policies.append(ExpertPolicy())

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    env = UnityNavigationEnv(port=5000)
    all_rows, summaries = [], []

    try:
        for policy in policies:
            print(f"\nEvaluating {policy.name} on {len(seeds)} episodes ...")
            rows = []
            for k, seed in enumerate(seeds, 1):
                rows.append(run_episode(env, policy, seed))
                if k % 20 == 0:
                    print(f"  {k}/{len(seeds)} | success so far {np.mean([r['success'] for r in rows]):.2f}")
            all_rows += rows
            summaries.append(summarize(rows))
    finally:
        env.close()

    # ---------------- print table ----------------
    print("\n" + "=" * 78)
    print(f"{'Method':<14}{'Success':>9}{'SPL':>8}{'Coll/ep':>9}{'CollEp':>8}{'Steps(succ)':>13}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['method']:<14}{s['success']:>9.3f}{s['spl']:>8.3f}"
              f"{s['collisions_per_episode']:>9.2f}{s['collision_episode_rate']:>8.2f}"
              f"{s['steps_successful']:>13.1f}")
    print("=" * 78)

    # ---------------- save CSVs ----------------
    os.makedirs("results", exist_ok=True)
    with open("results/eval_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open("results/eval_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print("Saved results/eval_episodes.csv and results/eval_summary.csv")


if __name__ == "__main__":
    main()
