import torch
import torch.nn as nn


class WaypointLoss(nn.Module):

    def __init__(
        self,
        angle_weight=1.0,
        distance_weight=1.0
    ):
        super().__init__()

        self.angle_weight = angle_weight
        self.distance_weight = distance_weight

    def forward(
        self,
        predicted_waypoints,
        target_waypoints
    ):

        # ------------------------------------------
        # Separate angle and distance
        # ------------------------------------------

        predicted_angles = predicted_waypoints[:, :, 0]
        predicted_distances = predicted_waypoints[:, :, 1]

        target_angles = target_waypoints[:, :, 0]
        target_distances = target_waypoints[:, :, 1]

        # ------------------------------------------
        # Angle Loss
        # ------------------------------------------

        angle_difference = (
            target_angles
            -
            predicted_angles
        )

        angle_loss = torch.mean(
            1.0 - torch.cos(angle_difference)
        )

        # ------------------------------------------
        # Distance Loss
        # ------------------------------------------

        distance_loss = torch.mean(
            (
                target_distances
                -
                predicted_distances
            ) ** 2
        )

        # ------------------------------------------
        # Total Waypoint Loss
        # ------------------------------------------

        total_loss = (
            self.angle_weight * angle_loss
            +
            self.distance_weight * distance_loss
        )

        return (
            total_loss,
            angle_loss,
            distance_loss
        )