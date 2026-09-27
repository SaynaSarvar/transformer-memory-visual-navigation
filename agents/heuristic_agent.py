import numpy as np


class HeuristicAgent:

    def __init__(
        self,
        angle_threshold=0.2,
        max_forward_steps=15
    ):
        self.angle_threshold = angle_threshold
        self.max_forward_steps = max_forward_steps

        self.previous_distance = None
        self.forward_steps = 0

    def reset(self):
        self.previous_distance = None
        self.forward_steps = 0

    def select_action(self, goal):

        distance = goal[0]
        angle = goal[1]

        # Turn toward the goal
        if angle > self.angle_threshold:

            action = 0
            self.forward_steps = 0

        elif angle < -self.angle_threshold:

            action = 1
            self.forward_steps = 0

        else:

            action = 2
            self.forward_steps += 1

        # Store current distance
        self.previous_distance = distance

        # Prevent endless forward movement
        if self.forward_steps >= self.max_forward_steps:

            if angle >= 0:
                action = 0
            else:
                action = 1

            self.forward_steps = 0

        return action
