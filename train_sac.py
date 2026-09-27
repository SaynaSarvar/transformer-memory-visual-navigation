import os
import random
import numpy as np
import torch

from envs.navigation_env import NavigationEnv
from agents.sac_agent import SACAgent
from agents.navigation_agent import NavigationAgent
from agents.rollout_collector import RolloutCollector
from utils.replay_buffer import ReplayBuffer


# ============================================================
# Configuration
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEED = 42

MEMORY_LENGTH = 8
EMBEDDING_DIM = 256

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64

WARMUP_TRANSITIONS = 1_000

# Final SAC training
TRAIN_EPISODES = 3000

GAMMA = 0.99
ALPHA = 0.2
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

MAX_EPISODE_STEPS = 200

OBSERVATION_ENCODER_PATH = (
    "minibatch_best_observation_encoder.pt"
)

TRANSFORMER_PATH = (
    "minibatch_best_transformer_memory.pt"
)

CHECKPOINT_DIR = "checkpoints"

# ------------------------------------------------------------
# Held-out validation
#
# These seeds are deliberately chosen well outside the
# training seed range (SEED + 1 ... SEED + TRAIN_EPISODES)
# so the evaluation is a genuine generalization check rather
# than a number correlated with the training stream itself.
# ------------------------------------------------------------

EVAL_EVERY_EPISODES = 100

EVAL_SEEDS = list(range(90000, 90040))


# ============================================================
# Final checkpoints
# ============================================================

ACTOR_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sac_actor_final.pt"
)

CRITIC_1_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sac_critic_1_final.pt"
)

CRITIC_2_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sac_critic_2_final.pt"
)


# ============================================================
# Best checkpoint
# ============================================================

BEST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sac_best_checkpoint.pt"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Load pretrained visual + transformer models
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

    # --------------------------------------------------------
    # Freeze representation
    # --------------------------------------------------------

    agent.observation_encoder.eval()
    agent.transformer_memory.eval()

    for parameter in agent.observation_encoder.parameters():
        parameter.requires_grad = False

    for parameter in agent.transformer_memory.parameters():
        parameter.requires_grad = False

    print("ObservationEncoder loaded and frozen.")
    print("TransformerMemory loaded and frozen.")


# ============================================================
# Held-out deterministic evaluation
#
# Runs the CURRENT policy, deterministically (mean action,
# no sampling noise), over a fixed set of seeds the policy
# never sees during training. This is what checkpoint
# selection should be based on -- the training-time rolling
# SuccessRate(50) is measured on the same correlated stream
# of episodes the policy just trained on, and overestimates
# how well the policy generalizes.
# ============================================================

@torch.no_grad()
def evaluate_deterministic(env, navigation_agent, seeds):

    successes = 0
    rewards = []

    for seed in seeds:

        observation, info = env.reset(seed=seed)

        navigation_agent.reset_memory()

        state, _ = navigation_agent.get_belief_state(
            observation
        )

        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):

            # ------------------------------------------------
            # Deterministic action: use the actor's mean,
            # squashed the same way sample() does, but with
            # no stochastic noise.
            # ------------------------------------------------

            mean, _ = navigation_agent.sac_agent.actor.forward(
                state
            )

            squashed_action = torch.tanh(mean)

            velocity = (
                squashed_action[:, 0:1] + 1.0
            ) / 2.0

            angular_velocity = squashed_action[:, 1:2]

            action = torch.cat(
                [velocity, angular_velocity],
                dim=1
            )

            action = (
                action.squeeze(0).cpu().numpy()
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            state, _ = navigation_agent.get_belief_state(
                observation
            )

            episode_reward += reward

            if info.get("success", False):
                successes += 1

        rewards.append(episode_reward)

    success_rate = successes / len(seeds)
    mean_reward = float(np.mean(rewards))

    return success_rate, mean_reward


# ============================================================
# Save best SAC checkpoint
# ============================================================

def save_best_checkpoint(
    sac_agent,
    episode,
    success_rate_50,
    mean_reward_50
):

    checkpoint = {
        # ----------------------------------------------------
        # Training information
        # ----------------------------------------------------
        "episode": episode,
        "success_rate_50": success_rate_50,
        "mean_reward_50": mean_reward_50,

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------
        "actor": sac_agent.actor.state_dict(),

        # ----------------------------------------------------
        # Critics
        # ----------------------------------------------------
        "critic_1": sac_agent.critic_1.state_dict(),
        "critic_2": sac_agent.critic_2.state_dict(),

        # ----------------------------------------------------
        # Target critics
        # ----------------------------------------------------
        "target_critic_1": (
            sac_agent.target_critic_1.state_dict()
        ),

        "target_critic_2": (
            sac_agent.target_critic_2.state_dict()
        ),

        # ----------------------------------------------------
        # Adaptive entropy temperature
        # ----------------------------------------------------
        "log_alpha": sac_agent.log_alpha.detach().cpu(),

        "alpha": sac_agent.alpha.item(),
    }

    torch.save(
        checkpoint,
        BEST_CHECKPOINT_PATH
    )

    print("\n" + "=" * 70)
    print("NEW BEST CHECKPOINT (held-out eval)")
    print("=" * 70)

    print(
        f"Episode:                {episode}"
    )

    print(
        f"Held-out SuccessRate:   {success_rate_50:.4f}"
    )

    print(
        f"Held-out MeanReward:    {mean_reward_50:.4f}"
    )

    print(
        f"Alpha:                  {sac_agent.alpha.item():.6f}"
    )

    print(
        f"Saved to:               {BEST_CHECKPOINT_PATH}"
    )

    print("=" * 70 + "\n")


# ============================================================
# Main training function
# ============================================================

def main():

    print("=" * 60)
    print("SAC TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Memory length: {MEMORY_LENGTH}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Replay capacity: {REPLAY_CAPACITY}")
    print(f"Warmup transitions: {WARMUP_TRANSITIONS}")
    print(f"Training episodes: {TRAIN_EPISODES}")
    print(f"Held-out eval every: {EVAL_EVERY_EPISODES} episodes")
    print(f"Held-out eval seeds: {len(EVAL_SEEDS)}")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # ========================================================
    # Environment
    # ========================================================

    env = NavigationEnv()

    # ========================================================
    # SAC Agent
    # ========================================================

    sac_agent = SACAgent(
        state_dim=EMBEDDING_DIM,
        action_dim=2,
        hidden_dim=256,
        gamma=GAMMA,
        alpha=ALPHA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        device=DEVICE
    )

    # ========================================================
    # Navigation Agent
    # ========================================================

    navigation_agent = NavigationAgent(
        sac_agent=sac_agent,
        embedding_dim=EMBEDDING_DIM,
        memory_length=MEMORY_LENGTH,
        device=DEVICE
    )

    load_frozen_representation(
        navigation_agent
    )

    # ========================================================
    # Replay Buffer
    # ========================================================

    replay_buffer = ReplayBuffer(
        state_dim=EMBEDDING_DIM,
        action_dim=2,
        capacity=REPLAY_CAPACITY
    )

    # ========================================================
    # Rollout Collector
    # ========================================================

    collector = RolloutCollector(
        env=env,
        agent=navigation_agent,
        replay_buffer=replay_buffer
    )

    print("\nSAC components initialized.")
    print("Starting training...\n")

    # ========================================================
    # Training statistics
    # ========================================================

    total_updates = 0

    episode_rewards = []
    episode_lengths = []

    success_count = 0
    collision_count = 0

    success_history = []

    # ========================================================
    # Best checkpoint statistics (held-out eval based)
    # ========================================================

    best_eval_success_rate = -1.0
    best_eval_mean_reward = -float("inf")

    # ========================================================
    # Training Loop
    # ========================================================

    for episode in range(
        1,
        TRAIN_EPISODES + 1
    ):

        seed = SEED + episode

        # ----------------------------------------------------
        # Collect episode
        # ----------------------------------------------------

        (
            episode_reward,
            episode_steps,
            success,
            collision
        ) = collector.collect_episode(
            seed=seed
        )

        episode_rewards.append(
            episode_reward
        )

        episode_lengths.append(
            episode_steps
        )

        # ----------------------------------------------------
        # Episode statistics
        # ----------------------------------------------------

        if success:
            success_count += 1

        success_history.append(
            success
        )

        if collision:
            collision_count += 1

        # ----------------------------------------------------
        # SAC Updates
        # ----------------------------------------------------

        critic_1_losses = []
        critic_2_losses = []
        actor_losses = []
        alpha_losses = []
        alphas = []

        if len(replay_buffer) >= WARMUP_TRANSITIONS:

            # Approximately one SAC update
            # per environment transition.

            num_updates = episode_steps

            for _ in range(num_updates):

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                ) = replay_buffer.sample(
                    BATCH_SIZE
                )

                # ------------------------------------------------
                # NumPy → Torch
                # ------------------------------------------------

                states = torch.tensor(
                    states,
                    dtype=torch.float32,
                    device=DEVICE
                )

                actions = torch.tensor(
                    actions,
                    dtype=torch.float32,
                    device=DEVICE
                )

                rewards = torch.tensor(
                    rewards,
                    dtype=torch.float32,
                    device=DEVICE
                )

                next_states = torch.tensor(
                    next_states,
                    dtype=torch.float32,
                    device=DEVICE
                )

                dones = torch.tensor(
                    dones,
                    dtype=torch.float32,
                    device=DEVICE
                )

                # ------------------------------------------------
                # SAC update
                # ------------------------------------------------

                losses = sac_agent.update(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                )

                critic_1_losses.append(
                    losses["critic_1_loss"]
                )

                critic_2_losses.append(
                    losses["critic_2_loss"]
                )

                actor_losses.append(
                    losses["actor_loss"]
                )

                alpha_losses.append(
                    losses["alpha_loss"]
                )

                alphas.append(
                    losses["alpha"]
                )

                total_updates += 1

        # ========================================================
        # Loss statistics
        # ========================================================

        if critic_1_losses:

            mean_c1 = np.mean(
                critic_1_losses
            )

            mean_c2 = np.mean(
                critic_2_losses
            )

            mean_actor = np.mean(
                actor_losses
            )

            mean_alpha = np.mean(
                alphas
            )

        else:

            mean_c1 = float("nan")
            mean_c2 = float("nan")
            mean_actor = float("nan")
            mean_alpha = float("nan")

        # ========================================================
        # Training progress (rolling window — informational only,
        # NOT used for checkpoint selection anymore)
        # ========================================================

        recent_rewards = episode_rewards[-50:]
        recent_successes = success_history[-50:]

        recent_success_rate = (
            sum(
                1
                for s in recent_successes
                if s
            )
            / max(
                1,
                len(recent_successes)
            )
        )

        # ========================================================
        # Print progress
        # ========================================================

        print(
            f"Episode {episode:03d} | "
            f"Reward {episode_reward:8.3f} | "
            f"Steps {episode_steps:3d} | "
            f"Success {success} | "
            f"Collision {collision} | "
            f"Buffer {len(replay_buffer):5d} | "
            f"C1 {mean_c1:8.4f} | "
            f"C2 {mean_c2:8.4f} | "
            f"Actor {mean_actor:8.4f} | "
            f"Alpha {mean_alpha:7.4f} | "
            f"SuccessRate(50) "
            f"{recent_success_rate:5.2f}"
        )

        # ========================================================
        # Held-out evaluation + best checkpoint
        #
        # Checkpoint selection is now based on a fixed, held-out
        # set of seeds the policy never trains on -- a genuine
        # generalization check -- instead of the rolling
        # SuccessRate(50) over the training stream itself.
        # ========================================================

        if episode % EVAL_EVERY_EPISODES == 0:

            eval_success_rate, eval_mean_reward = (
                evaluate_deterministic(
                    env,
                    navigation_agent,
                    EVAL_SEEDS
                )
            )

            print(
                f"  [Held-out eval @ ep {episode}] "
                f"SuccessRate {eval_success_rate:.2f} | "
                f"MeanReward {eval_mean_reward:.2f}"
            )

            is_better = (
                eval_success_rate
                > best_eval_success_rate
            )

            # ----------------------------------------------------
            # Tie-breaker:
            # same held-out SuccessRate, higher held-out MeanReward
            # ----------------------------------------------------

            if (
                eval_success_rate
                == best_eval_success_rate
                and eval_mean_reward
                > best_eval_mean_reward
            ):
                is_better = True

            if is_better:

                best_eval_success_rate = (
                    eval_success_rate
                )

                best_eval_mean_reward = (
                    eval_mean_reward
                )

                save_best_checkpoint(
                    sac_agent=sac_agent,
                    episode=episode,
                    success_rate_50=(
                        best_eval_success_rate
                    ),
                    mean_reward_50=(
                        best_eval_mean_reward
                    )
                )

    # ========================================================
    # Final Statistics
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    mean_reward = np.mean(
        episode_rewards
    )

    mean_length = np.mean(
        episode_lengths
    )

    success_rate = (
        success_count
        / TRAIN_EPISODES
    )

    collision_rate = (
        collision_count
        / TRAIN_EPISODES
    )

    print(
        f"Mean episode reward: "
        f"{mean_reward:.4f}"
    )

    print(
        f"Mean episode length: "
        f"{mean_length:.2f}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.4f}"
    )

    print(
        f"Collision rate: "
        f"{collision_rate:.4f}"
    )

    print(
        f"Successful episodes: "
        f"{success_count}/{TRAIN_EPISODES}"
    )

    print(
        f"Collision episodes: "
        f"{collision_count}/{TRAIN_EPISODES}"
    )

    print(
        f"Replay buffer size: "
        f"{len(replay_buffer)}"
    )

    print(
        f"Total SAC updates: "
        f"{total_updates}"
    )

    # ========================================================
    # Best checkpoint summary
    # ========================================================

    print("\n" + "=" * 60)
    print("BEST CHECKPOINT (held-out eval)")
    print("=" * 60)

    print(
        f"Best held-out SuccessRate: "
        f"{best_eval_success_rate:.4f}"
    )

    print(
        f"Best held-out MeanReward: "
        f"{best_eval_mean_reward:.4f}"
    )

    print(
        f"Path: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    # ========================================================
    # Save Final SAC Models
    # ========================================================

    torch.save(
        sac_agent.actor.state_dict(),
        ACTOR_PATH
    )

    torch.save(
        sac_agent.critic_1.state_dict(),
        CRITIC_1_PATH
    )

    torch.save(
        sac_agent.critic_2.state_dict(),
        CRITIC_2_PATH
    )

    print("\nSaved final models:")

    print(
        f"Actor:   {ACTOR_PATH}"
    )

    print(
        f"Critic1: {CRITIC_1_PATH}"
    )

    print(
        f"Critic2: {CRITIC_2_PATH}"
    )

    env.close()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
