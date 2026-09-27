import os
import random
import csv

import numpy as np
import torch

from envs.navigation_env import NavigationEnv
from agents.sac_agent import SACAgent
from agents.navigation_agent import NavigationAgent


# ============================================================
# Configuration
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MEMORY_LENGTH = 8
EMBEDDING_DIM = 256
HIDDEN_DIM = 256

NUM_EVAL_EPISODES = 100
EVAL_START_SEED = 10_000

MAX_EPISODE_STEPS = 200

OBSERVATION_ENCODER_PATH = (
    "minibatch_best_observation_encoder.pt"
)

TRANSFORMER_PATH = (
    "minibatch_best_transformer_memory.pt"
)

# ============================================================
# BEST CHECKPOINT
# ============================================================

BEST_CHECKPOINT_PATH = (
    "checkpoints/sac_best_checkpoint.pt"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Load Representation
# ============================================================

def load_frozen_representation(agent):

    print("\nLoading pretrained representation...")

    observation_encoder_state = torch.load(
        OBSERVATION_ENCODER_PATH,
        map_location=DEVICE
    )

    transformer_state = torch.load(
        TRANSFORMER_PATH,
        map_location=DEVICE
    )

    agent.observation_encoder.load_state_dict(
        observation_encoder_state
    )

    agent.transformer_memory.load_state_dict(
        transformer_state
    )

    agent.observation_encoder.eval()
    agent.transformer_memory.eval()

    for parameter in agent.observation_encoder.parameters():
        parameter.requires_grad = False

    for parameter in agent.transformer_memory.parameters():
        parameter.requires_grad = False

    print("ObservationEncoder loaded.")
    print("TransformerMemory loaded.")
    print("Representation frozen.")


# ============================================================
# Load BEST Actor
# ============================================================

def load_actor(sac_agent):

    print("\nLoading BEST SAC checkpoint...")

    checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
    )

    # --------------------------------------------------------
    # Check checkpoint format
    # --------------------------------------------------------

    if "actor" not in checkpoint:

        raise KeyError(
            "The best checkpoint does not contain "
            "an 'actor' key."
        )

    # --------------------------------------------------------
    # Load actor
    # --------------------------------------------------------

    sac_agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    sac_agent.actor.eval()

    # --------------------------------------------------------
    # Print checkpoint information
    # --------------------------------------------------------

    print("BEST SAC actor loaded.")

    print(
        f"Checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    if "episode" in checkpoint:

        print(
            f"Checkpoint episode: "
            f"{checkpoint['episode']}"
        )

    if "success_rate_50" in checkpoint:

        print(
            f"Training SuccessRate(50): "
            f"{checkpoint['success_rate_50']:.4f}"
        )

    if "mean_reward_50" in checkpoint:

        print(
            f"Training MeanReward(50): "
            f"{checkpoint['mean_reward_50']:.4f}"
        )

    if "alpha" in checkpoint:

        print(
            f"Training Alpha: "
            f"{checkpoint['alpha']:.6f}"
        )


# ============================================================
# Deterministic Action
# ============================================================

@torch.no_grad()
def deterministic_action(actor, state):

    mean, log_std = actor.forward(state)

    squashed_action = torch.tanh(mean)

    velocity = (
        squashed_action[:, 0:1] + 1.0
    ) / 2.0

    angular_velocity = (
        squashed_action[:, 1:2]
    )

    action = torch.cat(
        [
            velocity,
            angular_velocity
        ],
        dim=1
    )

    return action


# ============================================================
# Stochastic Action
# ============================================================

@torch.no_grad()
def stochastic_action(actor, state):

    action, _, _, _ = actor.sample(state)

    return action


# ============================================================
# Get Robot Position
# ============================================================

def get_robot_position(env):

    """
    MiniWorld stores the agent position in env.agent.pos.

    We only use x and z because MiniWorld is effectively
    operating on the ground plane.
    """

    pos = np.asarray(
        env.agent.pos,
        dtype=np.float32
    )

    return np.array(
        [
            pos[0],
            pos[2]
        ],
        dtype=np.float32
    )


# ============================================================
# Calculate Euclidean Path Length
# ============================================================

def calculate_path_length(positions):

    if len(positions) < 2:
        return 0.0

    positions = np.asarray(
        positions,
        dtype=np.float32
    )

    differences = (
        positions[1:] - positions[:-1]
    )

    segment_lengths = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.sum(segment_lengths)
    )


# ============================================================
# Calculate Shortest Path Length
# ============================================================
#
# IMPORTANT: this must be called using the robot's STARTING
# position (i.e. right after env.reset(), before any steps
# are taken). Calling it after the episode loop has finished
# will use the robot's FINAL position instead, which — since
# the robot is at or near the goal by then — produces a
# shortest-path length close to zero and silently deflates
# every SPL computed from it. See calling site in run_episode().
# ============================================================

def calculate_shortest_path_length(env, from_position=None):

    """
    Uses the environment's existing BFS path planner.

    _find_path() already exists in NavigationEnv and returns
    grid/path points from robot to goal.

    from_position: the (x, z) position to plan from. Pass the
    robot's starting position explicitly. If omitted, falls
    back to the robot's CURRENT position (only correct if
    called immediately after reset).
    """

    robot_position = (
        from_position
        if from_position is not None
        else get_robot_position(env)
    )

    goal_position = np.asarray(
        [
            env.goal.pos[0],
            env.goal.pos[2]
        ],
        dtype=np.float32
    )

    try:

        path = env._find_path(
            robot_position,
            goal_position
        )

    except TypeError:

        try:
            path = env._find_path()

        except Exception:
            return np.nan

    except Exception:
        return np.nan

    if path is None:
        return np.nan

    if len(path) < 2:
        return 0.0

    path = np.asarray(
        path,
        dtype=np.float32
    )

    return calculate_path_length(
        path
    )


# ============================================================
# Run One Episode
# ============================================================

@torch.no_grad()
def run_episode(
    env,
    navigation_agent,
    mode,
    seed
):

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=seed
    )

    navigation_agent.reset_memory()

    # --------------------------------------------------------
    # Initial belief state
    # --------------------------------------------------------

    state, _ = (
        navigation_agent.get_belief_state(
            observation
        )
    )

    # --------------------------------------------------------
    # Initial robot position
    # --------------------------------------------------------

    positions = []

    start_position = get_robot_position(env)

    positions.append(
        start_position
    )

    # --------------------------------------------------------
    # Shortest path length — MUST be computed here, from the
    # robot's starting position, before any steps are taken.
    # Computing this after the episode loop would measure the
    # distance from the robot's final (near-goal) position to
    # the goal instead of the true start-to-goal shortest path.
    # --------------------------------------------------------

    shortest_path_length = calculate_shortest_path_length(
        env,
        from_position=start_position
    )

    # --------------------------------------------------------
    # Episode statistics
    # --------------------------------------------------------

    episode_reward = 0.0
    episode_steps = 0

    success = False
    collision_occurred = False

    collision_count = 0

    terminated = False
    truncated = False

    # --------------------------------------------------------
    # Episode loop
    # --------------------------------------------------------

    while not (terminated or truncated):

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        if mode == "stochastic":

            action = stochastic_action(
                navigation_agent.sac_agent.actor,
                state
            )

        elif mode == "deterministic":

            action = deterministic_action(
                navigation_agent.sac_agent.actor,
                state
            )

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        # ----------------------------------------------------
        # Tensor -> NumPy
        # ----------------------------------------------------

        action_np = (
            action.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action_np
        )

        # ----------------------------------------------------
        # Position after action
        # ----------------------------------------------------

        positions.append(
            get_robot_position(env)
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        if info.get(
            "success",
            False
        ):

            success = True

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if info.get(
            "collision",
            False
        ):

            collision_occurred = True

            collision_count += 1

        # ----------------------------------------------------
        # Next belief
        # ----------------------------------------------------

        next_state, _ = (
            navigation_agent.get_belief_state(
                next_observation
            )
        )

        state = next_state

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        episode_reward += reward
        episode_steps += 1

    # ========================================================
    # Path metrics
    # ========================================================

    actual_path_length = (
        calculate_path_length(
            positions
        )
    )

    # NOTE: shortest_path_length was already computed above,
    # right after reset, using the robot's starting position.
    # It is intentionally NOT recomputed here.

    # ========================================================
    # SPL
    # ========================================================

    if (
        success
        and np.isfinite(shortest_path_length)
        and shortest_path_length > 0
    ):

        spl = (
            shortest_path_length
            / max(
                actual_path_length,
                shortest_path_length
            )
        )

    else:

        spl = 0.0

    # ========================================================
    # Episode classifications
    # ========================================================

    timeout = (
        truncated
        and not success
    )

    success_without_collision = (
        success
        and not collision_occurred
    )

    success_with_collision = (
        success
        and collision_occurred
    )

    return {

        "seed": seed,

        "mode": mode,

        "reward": episode_reward,

        "steps": episode_steps,

        "success": success,

        "collision": collision_occurred,

        "collision_count": collision_count,

        "success_without_collision":
            success_without_collision,

        "success_with_collision":
            success_with_collision,

        "timeout": timeout,

        "actual_path_length":
            actual_path_length,

        "shortest_path_length":
            shortest_path_length,

        "spl": spl
    }


# ============================================================
# Confidence Interval
# ============================================================

def success_rate_confidence_interval(
    successes,
    n
):

    if n == 0:

        return (
            float("nan"),
            float("nan")
        )

    p = successes / n

    se = np.sqrt(
        p * (1.0 - p) / n
    )

    lower = max(
        0.0,
        p - 1.96 * se
    )

    upper = min(
        1.0,
        p + 1.96 * se
    )

    return lower, upper


# ============================================================
# Summarize Results
# ============================================================

def summarize_results(
    results,
    mode
):

    results = [
        result
        for result in results
        if result["mode"] == mode
    ]

    n = len(results)

    successes = sum(
        result["success"]
        for result in results
    )

    collisions = sum(
        result["collision"]
        for result in results
    )

    total_collision_events = sum(
        result["collision_count"]
        for result in results
    )

    success_without_collision = sum(
        result["success_without_collision"]
        for result in results
    )

    success_with_collision = sum(
        result["success_with_collision"]
        for result in results
    )

    timeouts = sum(
        result["timeout"]
        for result in results
    )

    rewards = np.array(
        [
            result["reward"]
            for result in results
        ],
        dtype=np.float32
    )

    steps = np.array(
        [
            result["steps"]
            for result in results
        ],
        dtype=np.float32
    )

    actual_path_lengths = np.array(
        [
            result["actual_path_length"]
            for result in results
        ],
        dtype=np.float32
    )

    shortest_path_lengths = np.array(
        [
            result["shortest_path_length"]
            for result in results
            if np.isfinite(
                result["shortest_path_length"]
            )
        ],
        dtype=np.float32
    )

    successful_steps = np.array(
        [
            result["steps"]
            for result in results
            if result["success"]
        ],
        dtype=np.float32
    )

    successful_path_lengths = np.array(
        [
            result["actual_path_length"]
            for result in results
            if result["success"]
        ],
        dtype=np.float32
    )

    successful_spl = np.array(
        [
            result["spl"]
            for result in results
            if result["success"]
        ],
        dtype=np.float32
    )

    all_spl = np.array(
        [
            result["spl"]
            for result in results
        ],
        dtype=np.float32
    )

    success_rate = (
        successes / n
    )

    collision_rate = (
        collisions / n
    )

    success_no_collision_rate = (
        success_without_collision / n
    )

    success_collision_rate = (
        success_with_collision / n
    )

    timeout_rate = (
        timeouts / n
    )

    mean_reward = (
        np.mean(rewards)
    )

    std_reward = (
        np.std(rewards)
    )

    mean_steps = (
        np.mean(steps)
    )

    mean_collision_count = (
        total_collision_events / n
    )

    if len(successful_steps) > 0:

        mean_success_steps = (
            np.mean(successful_steps)
        )

    else:

        mean_success_steps = float("nan")

    if len(successful_path_lengths) > 0:

        mean_success_path_length = (
            np.mean(
                successful_path_lengths
            )
        )

    else:

        mean_success_path_length = float("nan")

    if len(shortest_path_lengths) > 0:

        mean_shortest_path = (
            np.mean(
                shortest_path_lengths
            )
        )

    else:

        mean_shortest_path = float("nan")

    mean_spl = (
        np.mean(all_spl)
    )

    if len(successful_spl) > 0:

        mean_success_spl = (
            np.mean(successful_spl)
        )

    else:

        mean_success_spl = float("nan")

    ci_low, ci_high = (
        success_rate_confidence_interval(
            successes,
            n
        )
    )

    # ========================================================
    # Print
    # ========================================================

    print("\n" + "=" * 70)
    print(
        f"EVALUATION — {mode.upper()}"
    )
    print("=" * 70)

    print(
        f"Number of episodes: "
        f"{n}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.4f} "
        f"({success_rate * 100:.2f}%)"
    )

    print(
        f"95% CI: "
        f"[{ci_low * 100:.2f}%, "
        f"{ci_high * 100:.2f}%]"
    )

    print(
        f"Collision episode rate: "
        f"{collision_rate:.4f} "
        f"({collision_rate * 100:.2f}%)"
    )

    print(
        f"Total collision events: "
        f"{total_collision_events}"
    )

    print(
        f"Mean collision events / episode: "
        f"{mean_collision_count:.2f}"
    )

    print(
        f"Success without collision: "
        f"{success_no_collision_rate:.4f} "
        f"({success_no_collision_rate * 100:.2f}%)"
    )

    print(
        f"Success + collision: "
        f"{success_collision_rate:.4f} "
        f"({success_collision_rate * 100:.2f}%)"
    )

    print(
        f"Timeout rate: "
        f"{timeout_rate:.4f} "
        f"({timeout_rate * 100:.2f}%)"
    )

    print(
        f"Mean reward: "
        f"{mean_reward:.4f}"
    )

    print(
        f"Reward std: "
        f"{std_reward:.4f}"
    )

    print(
        f"Mean episode steps: "
        f"{mean_steps:.2f}"
    )

    print(
        f"Mean steps on successful episodes: "
        f"{mean_success_steps:.2f}"
    )

    print(
        f"Mean actual path length: "
        f"{np.mean(actual_path_lengths):.4f}"
    )

    print(
        f"Mean shortest path length: "
        f"{mean_shortest_path:.4f}"
    )

    print(
        f"Mean successful path length: "
        f"{mean_success_path_length:.4f}"
    )

    print(
        f"SPL: "
        f"{mean_spl:.4f}"
    )

    print(
        f"SPL × 100: "
        f"{mean_spl * 100:.2f}"
    )

    print(
        f"Mean SPL on successful episodes: "
        f"{mean_success_spl:.4f}"
    )

    print(
        f"Successful episodes: "
        f"{successes}/{n}"
    )

    print(
        f"Collision episodes: "
        f"{collisions}/{n}"
    )

    print(
        f"Timeout episodes: "
        f"{timeouts}/{n}"
    )

    print("=" * 70)


# ============================================================
# Save CSV
# ============================================================

def save_results_csv(
    results,
    path="evaluation_results_best_checkpoint.csv"
):

    fieldnames = [
        "seed",
        "mode",
        "reward",
        "steps",
        "success",
        "collision",
        "collision_count",
        "success_without_collision",
        "success_with_collision",
        "timeout",
        "actual_path_length",
        "shortest_path_length",
        "spl"
    ]

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )

    print(
        "\nSaved evaluation results to:"
    )

    print(path)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("BEST SAC CHECKPOINT EVALUATION + SPL")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Episodes per mode: "
        f"{NUM_EVAL_EPISODES}"
    )

    print(
        f"Evaluation seed range: "
        f"{EVAL_START_SEED} - "
        f"{EVAL_START_SEED + NUM_EVAL_EPISODES - 1}"
    )

    print(
        f"Checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        EVAL_START_SEED
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = NavigationEnv()

    # --------------------------------------------------------
    # SAC
    # --------------------------------------------------------

    sac_agent = SACAgent(
        state_dim=EMBEDDING_DIM,
        action_dim=2,
        hidden_dim=HIDDEN_DIM,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Navigation Agent
    # --------------------------------------------------------

    navigation_agent = NavigationAgent(
        sac_agent=sac_agent,
        embedding_dim=EMBEDDING_DIM,
        memory_length=MEMORY_LENGTH,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Load representation
    # --------------------------------------------------------

    load_frozen_representation(
        navigation_agent
    )

    # --------------------------------------------------------
    # Load BEST actor
    # --------------------------------------------------------

    load_actor(
        sac_agent
    )

    # ========================================================
    # Evaluation
    # ========================================================

    all_results = []

    # ========================================================
    # STOCHASTIC
    # ========================================================

    print("\n")
    print("#" * 70)
    print("# STOCHASTIC EVALUATION")
    print("#" * 70)

    for episode in range(
        NUM_EVAL_EPISODES
    ):

        seed = (
            EVAL_START_SEED
            + episode
        )

        result = run_episode(
            env=env,
            navigation_agent=navigation_agent,
            mode="stochastic",
            seed=seed
        )

        all_results.append(
            result
        )

        print(
            f"[Stochastic] "
            f"Episode {episode + 1:03d}/"
            f"{NUM_EVAL_EPISODES} | "
            f"Seed {seed} | "
            f"Reward {result['reward']:8.3f} | "
            f"Steps {result['steps']:3d} | "
            f"Success {result['success']} | "
            f"Collision {result['collision']} | "
            f"Collisions {result['collision_count']} | "
            f"SPL {result['spl']:.3f}"
        )

    # ========================================================
    # DETERMINISTIC
    # ========================================================

    print("\n")
    print("#" * 70)
    print("# DETERMINISTIC EVALUATION")
    print("#" * 70)

    for episode in range(
        NUM_EVAL_EPISODES
    ):

        seed = (
            EVAL_START_SEED
            + episode
        )

        result = run_episode(
            env=env,
            navigation_agent=navigation_agent,
            mode="deterministic",
            seed=seed
        )

        all_results.append(
            result
        )

        print(
            f"[Deterministic] "
            f"Episode {episode + 1:03d}/"
            f"{NUM_EVAL_EPISODES} | "
            f"Seed {seed} | "
            f"Reward {result['reward']:8.3f} | "
            f"Steps {result['steps']:3d} | "
            f"Success {result['success']} | "
            f"Collision {result['collision']} | "
            f"Collisions {result['collision_count']} | "
            f"SPL {result['spl']:.3f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    summarize_results(
        all_results,
        mode="stochastic"
    )

    summarize_results(
        all_results,
        mode="deterministic"
    )

    # ========================================================
    # CSV
    # ========================================================

    save_results_csv(
        all_results,
        path="evaluation_results_best_checkpoint.csv"
    )

    # ========================================================
    # Close
    # ========================================================

    env.close()

    print(
        "\nEvaluation finished."
    )


if __name__ == "__main__":
    main()
