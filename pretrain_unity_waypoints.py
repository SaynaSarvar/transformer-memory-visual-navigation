"""
Step B - pretrain ObservationEncoder + TransformerMemory on Unity data
with the waypoint-prediction task (the paper's auxiliary objective).

    python pretrain_unity_waypoints.py          (Unity not needed)

Input : data/unity_waypoint_dataset.npz
Output: unity_best_observation_encoder.pt
        unity_best_transformer_memory.pt
        unity_best_waypoint_head.pt

Sequences are built exactly like NavigationAgent does at run time:
the memory holds the last <= 8 embeddings of the SAME episode and the
last token is the current observation. Early steps of an episode have
shorter memories (1..7), so batches are grouped by memory length.
"""

import numpy as np
import torch

from models.observation_encoder import ObservationEncoder
from models.transformer_memory import TransformerMemory
from models.waypoint_head import WaypointHead
from models.waypoint_loss import WaypointLoss


DATA_PATH = "data/unity_waypoint_dataset.npz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0

MEMORY_LENGTH = 8
EMBEDDING_DIM = 256

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
VALIDATION_FRACTION = 0.15

# Unity distances are up to ~25 m; scale them so the distance MSE
# does not drown the angle loss. Predictions are in units of 10 m.
DISTANCE_SCALE = 10.0

ENCODER_PATH = "unity_best_observation_encoder.pt"
TRANSFORMER_PATH = "unity_best_transformer_memory.pt"
HEAD_PATH = "unity_best_waypoint_head.pt"


# ============================================================
# Data
# ============================================================

def load_data():
    data = np.load(DATA_PATH)
    images = torch.from_numpy(data["images"])                # uint8 [N,60,80,3]
    goals = torch.from_numpy(data["goals"])                  # [N,2]
    waypoints = torch.from_numpy(data["waypoints"]).clone()  # [N,3,2]
    episode_ids = data["episode_ids"]

    waypoints[:, :, 1] /= DISTANCE_SCALE

    # index of the first frame of each frame's episode
    episode_start = np.zeros(len(episode_ids), dtype=np.int64)
    for k in range(1, len(episode_ids)):
        episode_start[k] = k if episode_ids[k] != episode_ids[k - 1] else episode_start[k - 1]

    # memory length available at each frame (1..8)
    lengths = np.minimum(np.arange(len(episode_ids)) - episode_start + 1, MEMORY_LENGTH)

    return images, goals, waypoints, episode_ids, lengths


def split_by_episode(episode_ids, rng):
    episodes = np.unique(episode_ids)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * VALIDATION_FRACTION))
    val_episodes = set(episodes[:n_val].tolist())
    is_val = np.array([e in val_episodes for e in episode_ids])
    return np.where(~is_val)[0], np.where(is_val)[0]


# ============================================================
# Model step
# ============================================================

def batch_loss(indices, lengths, images, goals, waypoints,
               encoder, transformer, head, criterion):
    """Average waypoint loss over a batch of end-frame indices."""

    total_loss = 0.0
    total_angle = 0.0
    total_distance = 0.0

    for L in np.unique(lengths[indices]):
        ends = indices[lengths[indices] == L]                 # [B]
        offsets = np.arange(-L + 1, 1)                        # [L]
        frame_idx = (ends[:, None] + offsets[None, :]).reshape(-1)

        image = images[frame_idx].to(DEVICE).float().permute(0, 3, 1, 2) / 255.0
        goal = goals[frame_idx].to(DEVICE)

        embeddings = encoder(image, goal).view(len(ends), L, -1)       # [B,L,D]
        belief, _ = transformer(embeddings[:, -1, :], embeddings)
        predicted = head(belief)

        target = waypoints[ends].to(DEVICE)
        loss, angle_loss, distance_loss = criterion(predicted, target)

        weight = len(ends) / len(indices)
        total_loss = total_loss + weight * loss
        total_angle += weight * angle_loss.item()
        total_distance += weight * distance_loss.item()

    return total_loss, total_angle, total_distance


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    images, goals, waypoints, episode_ids, lengths = load_data()
    train_idx, val_idx = split_by_episode(episode_ids, rng)
    print(f"Frames: {len(images)} | train {len(train_idx)} | val {len(val_idx)} | device {DEVICE}")

    encoder = ObservationEncoder(visual_dim=128, goal_dim=128, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    transformer = TransformerMemory(embedding_dim=EMBEDDING_DIM, num_heads=4,
                                    memory_length=MEMORY_LENGTH).to(DEVICE)
    head = WaypointHead(input_dim=EMBEDDING_DIM, num_waypoints=3).to(DEVICE)
    criterion = WaypointLoss()

    parameters = list(encoder.parameters()) + list(transformer.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):

        # ---------------- train ----------------
        encoder.train(); transformer.train(); head.train()
        rng.shuffle(train_idx)
        train_losses = []

        for start in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[start:start + BATCH_SIZE]
            loss, _, _ = batch_loss(batch, lengths, images, goals, waypoints,
                                    encoder, transformer, head, criterion)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---------------- validate ----------------
        encoder.eval(); transformer.eval(); head.eval()
        val_loss = val_angle = val_distance = 0.0
        n_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_idx), 256):
                batch = val_idx[start:start + 256]
                loss, a, d = batch_loss(batch, lengths, images, goals, waypoints,
                                        encoder, transformer, head, criterion)
                val_loss += loss.item(); val_angle += a; val_distance += d
                n_batches += 1

        val_loss /= n_batches; val_angle /= n_batches; val_distance /= n_batches

        # mean angle error in degrees: 1 - cos(e) -> e = acos(1 - loss)
        angle_error_deg = np.degrees(np.arccos(np.clip(1 - val_angle, -1, 1)))
        distance_error_m = np.sqrt(val_distance) * DISTANCE_SCALE

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(encoder.state_dict(), ENCODER_PATH)
            torch.save(transformer.state_dict(), TRANSFORMER_PATH)
            torch.save(head.state_dict(), HEAD_PATH)
            marker = "  <- saved best"

        print(f"Epoch {epoch:2d} | train {np.mean(train_losses):.4f} | "
              f"val {val_loss:.4f} (angle err ~{angle_error_deg:.1f} deg, "
              f"dist RMSE ~{distance_error_m:.2f} m){marker}")

    print(f"\nBest validation loss: {best_val:.4f}")
    print(f"Saved: {ENCODER_PATH}, {TRANSFORMER_PATH}, {HEAD_PATH}")


if __name__ == "__main__":
    main()
