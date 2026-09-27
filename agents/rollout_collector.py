import numpy as np


class RolloutCollector:

    def __init__(
        self,
        env,
        agent,
        replay_buffer
    ):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

    def collect_episode(self, seed=None):

        observation, info = self.env.reset(
            seed=seed
        )

        self.agent.reset_memory()

        episode_reward = 0.0
        episode_steps = 0

        terminated = False
        truncated = False

        success = False
        collision_occurred = False

        # --------------------------------------------------
        # Initial belief state
        # --------------------------------------------------

        state, _ = self.agent.get_belief_state(
            observation
        )

        # --------------------------------------------------
        # Episode
        # --------------------------------------------------

        while not (terminated or truncated):

            # ----------------------------------------------
            # Select action
            # ----------------------------------------------

            action = self.agent.act_from_state(
                state
            )

            # ----------------------------------------------
            # Environment transition
            # ----------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = self.env.step(action)

            # ----------------------------------------------
            # Track episode status
            # ----------------------------------------------

            if info.get("success", False):
                success = True

            if info.get("collision", False):
                collision_occurred = True

            # ----------------------------------------------
            # Next belief state
            # ----------------------------------------------

            next_state, _ = (
                self.agent.get_belief_state(
                    next_observation
                )
            )

            # ----------------------------------------------
            # Tensor → NumPy
            # ----------------------------------------------

            state_np = (
                state.squeeze(0)
                .cpu()
                .numpy()
            )

            next_state_np = (
                next_state.squeeze(0)
                .cpu()
                .numpy()
            )

            action_np = np.asarray(
                action,
                dtype=np.float32
            )

            reward_np = np.array(
                [reward],
                dtype=np.float32
            )

            # ----------------------------------------------
            # IMPORTANT: only a true terminal state (goal
            # reached / success) should cut off bootstrapping.
            # A time-limit truncation is NOT a real end of
            # the MDP -- the robot didn't fail, the clock
            # just ran out. Treating truncation as terminal
            # biases every value estimate near step 200 of
            # every episode toward zero, which destabilizes
            # training. See: Pardo et al. 2018 "Time Limits
            # in RL".
            # ----------------------------------------------

            done_np = np.array(
                [float(terminated)],
                dtype=np.float32
            )

            # ----------------------------------------------
            # Store transition
            # ----------------------------------------------

            self.replay_buffer.add(
                state_np,
                action_np,
                reward_np,
                next_state_np,
                done_np
            )

            # ----------------------------------------------
            # Move to next state
            # ----------------------------------------------

            state = next_state

            episode_reward += reward
            episode_steps += 1

        return (
            episode_reward,
            episode_steps,
            success,
            collision_occurred
        )