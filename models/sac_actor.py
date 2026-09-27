import torch
import torch.nn as nn
import torch.nn.functional as F


class SACActor(nn.Module):

    def __init__(
        self,
        state_dim=256,
        action_dim=2,
        hidden_dim=256
    ):
        super().__init__()

        self.action_dim = action_dim

        # --------------------------------------------------
        # Shared feature network
        # --------------------------------------------------

        self.network = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU()
        )

        # --------------------------------------------------
        # Mean of Gaussian distribution
        # --------------------------------------------------

        self.mean_layer = nn.Linear(
            hidden_dim,
            action_dim
        )

        # --------------------------------------------------
        # Log standard deviation
        # --------------------------------------------------

        self.log_std_layer = nn.Linear(
            hidden_dim,
            action_dim
        )

    def forward(self, state):

        # --------------------------------------------------
        # 1. Extract features
        # --------------------------------------------------

        features = self.network(state)

        # --------------------------------------------------
        # 2. Predict mean
        # --------------------------------------------------

        mean = self.mean_layer(features)

        # --------------------------------------------------
        # 3. Predict log standard deviation
        # --------------------------------------------------

        log_std = self.log_std_layer(features)

        # Keep standard deviation numerically stable
        log_std = torch.clamp(
            log_std,
            min=-20,
            max=2
        )

        return mean, log_std

    def sample(self, state):

        # --------------------------------------------------
        # 1. Get Gaussian parameters
        # --------------------------------------------------

        mean, log_std = self.forward(state)

        std = log_std.exp()

        # --------------------------------------------------
        # 2. Create Gaussian distribution
        # --------------------------------------------------

        normal = torch.distributions.Normal(
            mean,
            std
        )

        # --------------------------------------------------
        # 3. Reparameterized sample
        # --------------------------------------------------

        raw_action = normal.rsample()

        # --------------------------------------------------
        # 4. Gaussian log probability
        # --------------------------------------------------

        log_prob = normal.log_prob(
            raw_action
        )

        # --------------------------------------------------
        # 5. Tanh transformation
        # --------------------------------------------------

        squashed_action = torch.tanh(
            raw_action
        )

        # --------------------------------------------------
        # 6. Tanh Jacobian correction
        # --------------------------------------------------

        log_prob -= 2 * (
            torch.log(torch.tensor(2.0))
            - raw_action
            - F.softplus(-2 * raw_action)
        )

        # --------------------------------------------------
        # 7. Convert velocity from [-1,1] to [0,1]
        # --------------------------------------------------

        velocity = (
            squashed_action[:, 0:1] + 1.0
        ) / 2.0

        # --------------------------------------------------
        # 8. Angular velocity remains [-1,1]
        # --------------------------------------------------

        angular_velocity = (
            squashed_action[:, 1:2]
        )

        # --------------------------------------------------
        # 9. Combine action dimensions
        # --------------------------------------------------

        action = torch.cat(
            [
                velocity,
                angular_velocity
            ],
            dim=1
        )

        # --------------------------------------------------
        # 10. Velocity scaling correction
        #
        # v = (squashed_action + 1) / 2 is a linear map with
        # Jacobian 0.5. Change of variables:
        #   log p_Y(y) = log p_X(x) - log|dy/dx|
        #              = log p_X(x) - log(0.5)
        #              = log p_X(x) + log(2)
        # (previously this subtracted log(2), which is the
        # wrong sign and skews entropy for this dimension)
        # --------------------------------------------------

        log_prob[:, 0] += torch.log(
            torch.tensor(2.0)
        )

        # --------------------------------------------------
        # 11. Sum over action dimensions
        # --------------------------------------------------

        log_prob = log_prob.sum(
            dim=1,
            keepdim=True
        )

        return action, log_prob, mean, log_std