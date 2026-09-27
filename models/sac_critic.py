import torch
import torch.nn as nn


class SACCritic(nn.Module):

    def __init__(
        self,
        state_dim=256,
        action_dim=2,
        hidden_dim=256
    ):
        super().__init__()

        # --------------------------------------------------
        # State + Action
        # --------------------------------------------------

        input_dim = state_dim + action_dim

        # --------------------------------------------------
        # Q-network
        # --------------------------------------------------

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        state,
        action
    ):

        # --------------------------------------------------
        # Concatenate state and action
        # --------------------------------------------------

        state_action = torch.cat(
            [
                state,
                action
            ],
            dim=1
        )

        # --------------------------------------------------
        # Predict Q-value
        # --------------------------------------------------

        q_value = self.network(
            state_action
        )

        return q_value