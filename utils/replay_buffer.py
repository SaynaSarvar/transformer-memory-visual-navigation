import numpy as np


class ReplayBuffer:

    def __init__(
        self,
        state_dim=256,
        action_dim=2,
        capacity=100000
    ):

        self.capacity = capacity

        self.state_dim = state_dim
        self.action_dim = action_dim

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32
        )

        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (capacity, 1),
            dtype=np.float32
        )

        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32
        )

        self.dones = np.zeros(
            (capacity, 1),
            dtype=np.float32
        )

        # --------------------------------------------------
        # Buffer position
        # --------------------------------------------------

        self.position = 0

        self.size = 0


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        # --------------------------------------------------
        # Store transition
        # --------------------------------------------------

        self.states[self.position] = state

        self.actions[self.position] = action

        self.rewards[self.position] = reward

        self.next_states[self.position] = next_state

        self.dones[self.position] = done

        # --------------------------------------------------
        # Move pointer
        # --------------------------------------------------

        self.position = (
            self.position + 1
        ) % self.capacity

        # --------------------------------------------------
        # Update current size
        # --------------------------------------------------

        self.size = min(
            self.size + 1,
            self.capacity
        )


    def sample(self, batch_size):

        # --------------------------------------------------
        # Random indices
        # --------------------------------------------------

        indices = np.random.randint(
            0,
            self.size,
            size=batch_size
        )

        # --------------------------------------------------
        # Sample batch
        # --------------------------------------------------

        states = self.states[
            indices
        ]

        actions = self.actions[
            indices
        ]

        rewards = self.rewards[
            indices
        ]

        next_states = self.next_states[
            indices
        ]

        dones = self.dones[
            indices
        ]

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )


    def __len__(self):

        return self.size