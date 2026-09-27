import torch
import torch.nn as nn

from models.visual_encoder import VisualEncoder


class ObservationEncoder(nn.Module):

    def __init__(
        self,
        visual_dim=128,
        goal_dim=128,
        embedding_dim=256
    ):

        super().__init__()

        # ------------------------------
        # Visual encoder
        # ------------------------------

        self.visual_encoder = VisualEncoder(
            feature_dim=visual_dim
        )

        # ------------------------------
        # Goal encoder
        # ------------------------------

        self.goal_encoder = nn.Sequential(

            nn.Linear(
                2,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                goal_dim
            ),

            nn.ReLU()
        )

        # ------------------------------
        # Fusion layer
        # ------------------------------

        self.fusion = nn.Linear(
            visual_dim + goal_dim,
            embedding_dim
        )


    def forward(
        self,
        image,
        goal
    ):

        # Encode image
        visual_features = self.visual_encoder(
            image
        )

        # Encode goal
        goal_features = self.goal_encoder(
            goal
        )

        # Concatenate
        combined_features = torch.cat(
            [
                visual_features,
                goal_features
            ],
            dim=1
        )

        # Project to final embedding
        embedding = self.fusion(
            combined_features
        )

        return embedding