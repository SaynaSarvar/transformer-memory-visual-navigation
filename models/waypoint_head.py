import torch
import torch.nn as nn


class WaypointHead(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        num_waypoints=3
    ):

        super().__init__()

        self.num_waypoints = num_waypoints

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                num_waypoints * 2
            )
        )

    def forward(self, belief_state):

        x = self.network(
            belief_state
        )

        waypoints = x.view(
            -1,
            self.num_waypoints,
            2
        )

        return waypoints