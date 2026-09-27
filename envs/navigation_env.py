import numpy as np

import gymnasium as gym
from gymnasium import spaces

from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box


class NavigationEnv(MiniWorldEnv):

    def __init__(self, render_mode=None):

        super().__init__(
            max_episode_steps=200,
            domain_rand=False,
            render_mode=render_mode
        )

        # ==================================================
        # Observation Space
        # ==================================================

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(60, 80, 3),
                dtype=np.uint8
            ),

            "goal": spaces.Box(
                low=np.array(
                    [0.0, -np.pi],
                    dtype=np.float32
                ),
                high=np.array(
                    [20.0, np.pi],
                    dtype=np.float32
                ),
                dtype=np.float32
            )
        })

        # ==================================================
        # Action Space
        # ==================================================
        # action = [linear_velocity, angular_velocity]
        #
        # v     ∈ [0, 1]
        # omega ∈ [-1, 1]

        self.action_space = spaces.Box(
            low=np.array(
                [0.0, -1.0],
                dtype=np.float32
            ),
            high=np.array(
                [1.0, 1.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.previous_distance = None
        self.step_count = 0

        # Store obstacle positions
        self.obstacle_positions = []

        # Number of collisions so far in the current episode
        self.episode_collision_count = 0

    # ======================================================
    # Random Position Sampling
    # ======================================================

    def _sample_position(
        self,
        min_x,
        max_x,
        min_z,
        max_z,
        existing_positions,
        min_distance
    ):

        for _ in range(100):

            x = self.np_random.uniform(
                min_x,
                max_x
            )

            z = self.np_random.uniform(
                min_z,
                max_z
            )

            candidate = np.array(
                [x, z]
            )

            valid = True

            for position in existing_positions:

                position_2d = np.array(
                    [
                        position[0],
                        position[2]
                    ]
                )

                distance = np.linalg.norm(
                    candidate - position_2d
                )

                if distance < min_distance:
                    valid = False
                    break

            if valid:
                return (
                    x,
                    0,
                    z
                )

        raise RuntimeError(
            "Could not find a valid random position."
        )

    # ======================================================
    # Find Actual BFS Path
    # ======================================================

    def _find_path(
        self,
        start,
        goal
    ):

        # --------------------------------------------------
        # Grid Configuration
        # --------------------------------------------------

        grid_size = 0.25
        min_coord = 0.5
        max_coord = 9.5

        grid_width = int(
            (max_coord - min_coord)
            / grid_size
        ) + 1

        grid_height = grid_width

        # --------------------------------------------------
        # World → Grid
        # --------------------------------------------------

        def world_to_grid(x, z):

            gx = int(
                round(
                    (x - min_coord)
                    / grid_size
                )
            )

            gz = int(
                round(
                    (z - min_coord)
                    / grid_size
                )
            )

            # Keep inside grid

            gx = max(
                0,
                min(gx, grid_width - 1)
            )

            gz = max(
                0,
                min(gz, grid_height - 1)
            )

            return gx, gz

        # --------------------------------------------------
        # Grid → World
        # --------------------------------------------------

        def grid_to_world(gx, gz):

            x = (
                min_coord
                + gx * grid_size
            )

            z = (
                min_coord
                + gz * grid_size
            )

            return x, z

        # --------------------------------------------------
        # Convert Start / Goal
        # --------------------------------------------------

        start_grid = world_to_grid(
            start[0],
            start[1]
        )

        goal_grid = world_to_grid(
            goal[0],
            goal[1]
        )

        # --------------------------------------------------
        # Check Whether Grid Cell Is Blocked
        # --------------------------------------------------

        obstacle_radius = 0.9

        def is_blocked(gx, gz):

            world_x, world_z = grid_to_world(
                gx,
                gz
            )

            for obstacle_position in self.obstacle_positions:

                obstacle_x = obstacle_position[0]
                obstacle_z = obstacle_position[2]

                distance = np.sqrt(
                    (world_x - obstacle_x) ** 2
                    +
                    (world_z - obstacle_z) ** 2
                )

                if distance < obstacle_radius:
                    return True

            return False

        # --------------------------------------------------
        # Validate Start / Goal
        # --------------------------------------------------

        if is_blocked(
            start_grid[0],
            start_grid[1]
        ):
            return None

        if is_blocked(
            goal_grid[0],
            goal_grid[1]
        ):
            return None

        # --------------------------------------------------
        # BFS Initialization
        # --------------------------------------------------

        queue = [
            start_grid
        ]

        parent = {
            start_grid: None
        }

        # --------------------------------------------------
        # Four-Connected Grid
        # --------------------------------------------------

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ]

        # --------------------------------------------------
        # Breadth-First Search
        # --------------------------------------------------

        while queue:

            current = queue.pop(0)

            # Goal reached

            if current == goal_grid:
                break

            current_x, current_z = current

            # Explore neighbors

            for dx, dz in directions:

                neighbor = (
                    current_x + dx,
                    current_z + dz
                )

                neighbor_x, neighbor_z = neighbor

                # ------------------------------------------
                # Boundary Check
                # ------------------------------------------

                if neighbor_x < 0:
                    continue

                if neighbor_x >= grid_width:
                    continue

                if neighbor_z < 0:
                    continue

                if neighbor_z >= grid_height:
                    continue

                # ------------------------------------------
                # Already Visited
                # ------------------------------------------

                if neighbor in parent:
                    continue

                # ------------------------------------------
                # Obstacle Check
                # ------------------------------------------

                if is_blocked(
                    neighbor_x,
                    neighbor_z
                ):
                    continue

                # ------------------------------------------
                # Store Parent
                # ------------------------------------------

                parent[neighbor] = current

                queue.append(
                    neighbor
                )

        # --------------------------------------------------
        # No Path Found
        # --------------------------------------------------

        if goal_grid not in parent:
            return None

        # --------------------------------------------------
        # Reconstruct Path
        # --------------------------------------------------

        path = []

        current = goal_grid

        while current is not None:

            path.append(
                current
            )

            current = parent[current]

        # Goal → ... → Start
        #
        # Reverse:
        #
        # Start → ... → Goal

        path.reverse()

        # --------------------------------------------------
        # Convert Grid Path → World Coordinates
        # --------------------------------------------------

        world_path = []

        for gx, gz in path:

            world_position = grid_to_world(
                gx,
                gz
            )

            world_path.append(
                world_position
            )

        return world_path

    # ======================================================
    # Extract Waypoints From Planned Path
    # ======================================================

    def _get_waypoints(
        self,
        num_waypoints=3
    ):

        # --------------------------------------------------
        # 1. Robot Position
        # --------------------------------------------------

        robot_position = (
            self.agent.pos[0],
            self.agent.pos[2]
        )

        # --------------------------------------------------
        # 2. Goal Position
        # --------------------------------------------------

        goal_position = (
            self.goal.pos[0],
            self.goal.pos[2]
        )

        # --------------------------------------------------
        # 3. Find Path
        # --------------------------------------------------

        path = self._find_path(
            start=robot_position,
            goal=goal_position
        )

        if path is None:
            return None

        # --------------------------------------------------
        # 4. Remove Points Too Close To Robot
        # --------------------------------------------------

        path = [
            point
            for point in path
            if np.linalg.norm(
                np.array(point)
                -
                np.array(robot_position)
            ) > 0.25
        ]

        # --------------------------------------------------
        # 5. Check Path Length
        # --------------------------------------------------

        if len(path) == 0:
            return None

        # --------------------------------------------------
        # 6. Select Waypoints
        # --------------------------------------------------

        waypoint_indices = np.linspace(
            0,
            len(path) - 1,
            num_waypoints,
            dtype=int
        )

        selected_waypoints = [
            path[index]
            for index in waypoint_indices
        ]

        # --------------------------------------------------
        # 7. Robot Orientation
        # --------------------------------------------------

        robot_angle = self.agent.dir

        # --------------------------------------------------
        # 8. Convert Waypoints To Relative
        #    Angle + Distance
        # --------------------------------------------------

        waypoint_targets = []

        for waypoint in selected_waypoints:

            waypoint_x = waypoint[0]
            waypoint_z = waypoint[1]

            # ----------------------------------------------
            # Difference Between Robot And Waypoint
            # ----------------------------------------------

            dx = (
                waypoint_x
                -
                robot_position[0]
            )

            dz = (
                waypoint_z
                -
                robot_position[1]
            )

            # ----------------------------------------------
            # Distance
            # ----------------------------------------------

            distance = np.sqrt(
                dx ** 2
                +
                dz ** 2
            )

            # ----------------------------------------------
            # Global Angle
            # ----------------------------------------------

            waypoint_angle = np.arctan2(
                dz,
                dx
            )

            # ----------------------------------------------
            # Relative Angle
            # ----------------------------------------------

            relative_angle = (
                waypoint_angle
                -
                robot_angle
            )

            # ----------------------------------------------
            # Normalize Angle
            # ----------------------------------------------

            relative_angle = (
                relative_angle
                +
                np.pi
            ) % (
                2 * np.pi
            ) - np.pi

            waypoint_targets.append(
                [
                    relative_angle,
                    distance
                ]
            )

        # --------------------------------------------------
        # 9. Return As NumPy Array
        # --------------------------------------------------

        return np.array(
            waypoint_targets,
            dtype=np.float32
        )

    # ======================================================
    # Environment Generation
    # ======================================================

    def _gen_world(self, seed=None):

        self.seed = seed

        # ==================================================
        # Room
        # ==================================================

        self.add_rect_room(
            min_x=0,
            max_x=10,
            min_z=0,
            max_z=10
        )

        # ==================================================
        # Generate Valid Configuration
        # ==================================================

        for attempt in range(100):

            # --------------------------------------------------
            # Robot
            # --------------------------------------------------

            robot_position = self._sample_position(
                min_x=1.0,
                max_x=3.0,
                min_z=1.0,
                max_z=3.0,
                existing_positions=[],
                min_distance=0.0
            )

            # --------------------------------------------------
            # Goal
            # --------------------------------------------------

            goal_position = self._sample_position(
                min_x=7.0,
                max_x=9.0,
                min_z=7.0,
                max_z=9.0,
                existing_positions=[
                    robot_position
                ],
                min_distance=4.0
            )

            # --------------------------------------------------
            # Obstacles
            # --------------------------------------------------

            obstacle_positions = []

            for _ in range(3):

                obstacle_position = self._sample_position(
                    min_x=2.5,
                    max_x=7.5,
                    min_z=2.5,
                    max_z=7.5,
                    existing_positions=(
                        [
                            robot_position,
                            goal_position
                        ]
                        +
                        obstacle_positions
                    ),
                    min_distance=2.0
                )

                obstacle_positions.append(
                    obstacle_position
                )

            # --------------------------------------------------
            # Temporarily Store Obstacles
            # --------------------------------------------------

            self.obstacle_positions = obstacle_positions

            # --------------------------------------------------
            # Check Whether Path Exists
            # --------------------------------------------------

            path = self._find_path(
                start=(
                    robot_position[0],
                    robot_position[2]
                ),
                goal=(
                    goal_position[0],
                    goal_position[2]
                )
            )

            # --------------------------------------------------
            # Valid Configuration
            # --------------------------------------------------

            if path is not None:
                break

        else:

            raise RuntimeError(
                "Could not generate a solvable environment."
            )

        # ==================================================
        # Create Obstacles
        # ==================================================

        self.obstacles = []

        for position in obstacle_positions:

            obstacle = Box(
                color="grey",
                size=1.3
            )

            self.place_entity(
                obstacle,
                pos=position
            )

            self.obstacles.append(
                obstacle
            )

        # ==================================================
        # Create Goal
        # ==================================================

        self.goal = Box(
            color="red",
            size=0.8
        )

        self.place_entity(
            self.goal,
            pos=goal_position
        )

        # ==================================================
        # Create Robot
        # ==================================================

        self.place_agent(
            pos=robot_position,
            dir=0.0
        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        self.step_count = 0

        # Reset collision counter for the new episode
        self.episode_collision_count = 0

        self.previous_distance = (
            self._get_goal_information()[0]
        )

        observation = {
            "image": obs,
            "goal": self._get_goal_information()
        }

        return observation, info

    # ======================================================
    # Step
    # ======================================================

    def step(self, action):

        # --------------------------------------------------
        # 1. Read Continuous Action
        # --------------------------------------------------

        v = float(
            action[0]
        )

        omega = float(
            action[1]
        )

        # --------------------------------------------------
        # 2. Simulation Time Step
        # --------------------------------------------------

        dt = 0.1

        # --------------------------------------------------
        # 3. Update Robot Orientation
        # --------------------------------------------------

        self.agent.dir += (
            omega * dt
        )

        # Keep direction inside [-pi, pi]

        self.agent.dir = (
            self.agent.dir
            + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        # --------------------------------------------------
        # 4. Calculate Proposed Movement
        # --------------------------------------------------

        dx = (
            v
            * dt
            * np.cos(self.agent.dir)
        )

        dz = (
            v
            * dt
            * np.sin(self.agent.dir)
        )

        old_position = np.array(
            self.agent.pos,
            dtype=np.float32
        )

        proposed_position = (
            old_position.copy()
        )

        proposed_position[0] += dx
        proposed_position[2] += dz

        # --------------------------------------------------
        # 5. Collision Check
        #
        # We keep track of the NEAREST colliding obstacle so
        # that, if a collision happens, we can slide the
        # robot along that obstacle's surface instead of
        # freezing it in place (see step 6 below).
        # --------------------------------------------------

        collision = False
        colliding_obstacle_pos = None
        nearest_distance = float("inf")

        for obstacle in self.obstacles:

            obstacle_position = np.array(
                obstacle.pos,
                dtype=np.float32
            )

            distance = np.linalg.norm(
                proposed_position[[0, 2]]
                -
                obstacle_position[[0, 2]]
            )

            if distance < 1.0:

                collision = True

                if distance < nearest_distance:
                    nearest_distance = distance
                    colliding_obstacle_pos = obstacle_position

        # --------------------------------------------------
        # 6. Move — Slide Along Obstacle On Collision
        #
        # IMPORTANT CHANGE:
        #
        # Previously, a collision completely rejected the
        # proposed movement, freezing the robot in place.
        # If the policy kept choosing a similar heading,
        # this caused the robot to get stuck against an
        # obstacle for the rest of the episode (very high
        # per-episode collision counts, timeouts).
        #
        # Instead, on collision we project the movement onto
        # the tangential direction (perpendicular to the
        # obstacle normal) so the robot slides along the
        # obstacle surface rather than stopping dead.
        # --------------------------------------------------

        if collision:

            to_obstacle = (
                colliding_obstacle_pos[[0, 2]]
                -
                old_position[[0, 2]]
            )

            obstacle_distance = np.linalg.norm(to_obstacle)

            if obstacle_distance > 1e-6:

                normal = to_obstacle / obstacle_distance

                movement = (
                    proposed_position[[0, 2]]
                    -
                    old_position[[0, 2]]
                )

                tangential = (
                    movement
                    -
                    np.dot(movement, normal) * normal
                )

                proposed_position[[0, 2]] = (
                    old_position[[0, 2]] + tangential
                )

        self.agent.pos = tuple(
            proposed_position
        )

        # --------------------------------------------------
        # 7. Get New Goal Information
        # --------------------------------------------------

        goal = self._get_goal_information()

        current_distance = goal[0]

        # --------------------------------------------------
        # 8. Reward
        # --------------------------------------------------

        reward = (
            self.previous_distance
            -
            current_distance
        )

        self.previous_distance = (
            current_distance
        )

        # Small Time Penalty

        reward -= 0.01

        # Collision Penalty

        if collision:
            reward -= 0.1
            self.episode_collision_count += 1

        # --------------------------------------------------
        # 9. Success
        # --------------------------------------------------

        success = (
            current_distance < 0.8
        )

        if success:
            reward += 10.0

        terminated = success

        # --------------------------------------------------
        # 10. Episode Truncation
        #
        # In addition to the normal time-limit truncation,
        # we cut the episode short if the robot has racked
        # up too many collisions -- this is a "stuck episode"
        # circuit breaker so we don't waste rollout steps
        # (and gradient updates) on a robot that is jammed
        # against an obstacle.
        # --------------------------------------------------

        truncated = False

        self.step_count += 1

        if (
            self.step_count
            >= self.max_episode_steps
        ):
            truncated = True

        if self.episode_collision_count >= 20:

            truncated = True

            # Extra penalty for getting stuck
            reward -= 1.0

        # --------------------------------------------------
        # 11. Render Observation
        # --------------------------------------------------

        obs = self.render_obs()

        observation = {
            "image": obs,
            "goal": goal
        }

        info = {
            "success": success,
            "collision": collision
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ======================================================
    # Goal Information
    # ======================================================

    def _get_goal_information(self):

        # --------------------------------------------------
        # Robot Position
        # --------------------------------------------------

        robot_pos = np.array(
            self.agent.pos
        )

        # --------------------------------------------------
        # Goal Position
        # --------------------------------------------------

        goal_pos = np.array(
            self.goal.pos
        )

        # --------------------------------------------------
        # Difference
        # --------------------------------------------------

        dx = (
            goal_pos[0]
            -
            robot_pos[0]
        )

        dz = (
            goal_pos[2]
            -
            robot_pos[2]
        )

        # --------------------------------------------------
        # Distance
        # --------------------------------------------------

        distance = np.sqrt(
            dx ** 2
            +
            dz ** 2
        )

        # --------------------------------------------------
        # Global Goal Angle
        # --------------------------------------------------

        goal_angle = np.arctan2(
            dz,
            dx
        )

        # --------------------------------------------------
        # Robot Orientation
        # --------------------------------------------------

        robot_angle = self.agent.dir

        # --------------------------------------------------
        # Relative Angle
        # --------------------------------------------------

        relative_angle = (
            goal_angle
            -
            robot_angle
        )

        # --------------------------------------------------
        # Normalize Angle
        # --------------------------------------------------

        relative_angle = (
            relative_angle
            + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        return np.array(
            [
                distance,
                relative_angle
            ],
            dtype=np.float32
        )
