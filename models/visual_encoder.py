import torch
import torch.nn as nn


class VisualEncoder(nn.Module):

    def __init__(self, feature_dim=128):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            )
        )

        self.projection = nn.Linear(
            128,
            feature_dim
        )


    def forward(self, image):

        x = self.cnn(image)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.projection(x)

        return x
