"""
Grid path planner for the Unity room + waypoint targets.

Unity version of NavigationEnv._find_path / _get_waypoints.

Uses the layout that Unity sends every step (info["obstacles"],
info["goal_pos"], info["agent_pos"], info["agent_yaw"]).

Angle convention (same as the goal angle Unity sends):
    angle = 0      -> straight ahead
    angle > 0      -> to the RIGHT
    angle in radians, normalized to [-pi, pi]
"""

import heapq
import numpy as np


class UnityGridPlanner:

    def __init__(
        self,
        room_half_size=9.75,     # inner edge of the walls
        cell_size=0.25,
        clearance=0.6            # agent radius (0.25) + safety margin
    ):
        self.room_half_size = room_half_size
        self.cell_size = cell_size
        self.clearance = clearance

        self.n = int(round(2 * room_half_size / cell_size)) + 1
        coords = -room_half_size + np.arange(self.n) * cell_size
        # grid[i, j] -> world (x = coords[i], z = coords[j])
        self.grid_x, self.grid_z = np.meshgrid(coords, coords, indexing="ij")

        self.blocked = None
        self.cost_to_goal = None

    # ======================================================
    # World <-> grid
    # ======================================================

    def to_cell(self, x, z):
        i = int(round((x + self.room_half_size) / self.cell_size))
        j = int(round((z + self.room_half_size) / self.cell_size))
        return (
            min(max(i, 0), self.n - 1),
            min(max(j, 0), self.n - 1)
        )

    def to_world(self, i, j):
        return (
            -self.room_half_size + i * self.cell_size,
            -self.room_half_size + j * self.cell_size
        )

    # ======================================================
    # Build the map once per episode (after reset)
    # ======================================================

    def build(self, obstacles, goal_pos):
        """
        obstacles: array [N, 4] = x, z, size_x, size_z (axis-aligned boxes)
        goal_pos:  array [2]    = x, z
        """
        c = self.clearance
        blocked = np.zeros((self.n, self.n), dtype=bool)

        # Walls
        blocked |= np.abs(self.grid_x) > self.room_half_size - c
        blocked |= np.abs(self.grid_z) > self.room_half_size - c

        # Obstacles (inflated by clearance)
        for ox, oz, sx, sz in obstacles:
            blocked |= (
                (np.abs(self.grid_x - ox) < sx / 2 + c)
                & (np.abs(self.grid_z - oz) < sz / 2 + c)
            )

        self.blocked = blocked
        self.cost_to_goal = self._dijkstra_from(
            self.to_cell(goal_pos[0], goal_pos[1])
        )

    def _neighbors(self, i, j):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n and 0 <= nj < self.n and not self.blocked[ni, nj]:
                    # forbid cutting corners diagonally
                    if di != 0 and dj != 0 and (
                        self.blocked[i + di, j] or self.blocked[i, j + dj]
                    ):
                        continue
                    yield ni, nj, (1.4142 if di and dj else 1.0)

    def _dijkstra_from(self, start):
        cost = np.full((self.n, self.n), np.inf, dtype=np.float32)
        cost[start] = 0.0
        heap = [(0.0, start)]

        while heap:
            d, (i, j) = heapq.heappop(heap)
            if d > cost[i, j]:
                continue
            for ni, nj, step in self._neighbors(i, j):
                nd = d + step
                if nd < cost[ni, nj]:
                    cost[ni, nj] = nd
                    heapq.heappush(heap, (nd, (ni, nj)))

        return cost

    # ======================================================
    # Path from current position
    # ======================================================

    def _line_of_sight(self, a, b):
        steps = int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell_size * 0.5))) + 1
        for t in np.linspace(0.0, 1.0, steps):
            if self.blocked[self.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]:
                return False
        return True

    def path_length(self, agent_pos, goal_pos=None):
        """
        Shortest collision-free path length (m) from agent to goal; inf if unreachable.
        The zig-zag grid path is shortened with line-of-sight checks
        (string pulling), so it is close to the true geometric shortest path.
        """
        cell = self.to_cell(agent_pos[0], agent_pos[1])
        if not np.isfinite(self.cost_to_goal[cell]):
            return float("inf")

        points = [tuple(agent_pos)] + (self.path(agent_pos) or [])
        if goal_pos is not None:
            points.append(tuple(goal_pos))

        length, anchor, k = 0.0, points[0], 1
        while k < len(points):
            # furthest point still visible from the anchor
            j = k
            while j + 1 < len(points) and self._line_of_sight(anchor, points[j + 1]):
                j += 1
            length += float(np.hypot(points[j][0] - anchor[0], points[j][1] - anchor[1]))
            anchor, k = points[j], j + 1
        return length

    def _nearest_free_cell(self, cell):
        if np.isfinite(self.cost_to_goal[cell]):
            return cell
        free = np.argwhere(np.isfinite(self.cost_to_goal))
        if len(free) == 0:
            return None
        d = np.sum((free - np.array(cell)) ** 2, axis=1)
        return tuple(free[np.argmin(d)])

    def path(self, agent_pos):
        """World-space path [(x, z), ...] from agent to goal, or None."""
        cell = self._nearest_free_cell(self.to_cell(agent_pos[0], agent_pos[1]))
        if cell is None:
            return None

        cells = [cell]
        while self.cost_to_goal[cell] > 0:
            best = min(
                self._neighbors(*cell),
                key=lambda n: self.cost_to_goal[n[0], n[1]],
                default=None
            )
            if best is None or self.cost_to_goal[best[0], best[1]] >= self.cost_to_goal[cell]:
                break
            cell = (best[0], best[1])
            cells.append(cell)

        return [self.to_world(i, j) for i, j in cells]

    # ======================================================
    # Waypoint targets  ->  [num_waypoints, 2] = (angle, distance)
    # ======================================================

    @staticmethod
    def relative_angle_distance(agent_pos, agent_yaw, point):
        dx = point[0] - agent_pos[0]
        dz = point[1] - agent_pos[1]
        distance = np.hypot(dx, dz)
        # Unity yaw: 0 = +z, clockwise positive -> bearing = atan2(dx, dz)
        angle = np.arctan2(dx, dz) - agent_yaw
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle, distance

    def waypoints(self, agent_pos, agent_yaw, goal_pos, num_waypoints=3, min_distance=0.5):

        path = self.path(agent_pos)

        if path is None:
            path = [tuple(goal_pos)]

        path = path + [tuple(goal_pos)]

        path = [
            p for p in path
            if np.hypot(p[0] - agent_pos[0], p[1] - agent_pos[1]) > min_distance
        ] or [tuple(goal_pos)]

        indices = np.linspace(0, len(path) - 1, num_waypoints, dtype=int)

        return np.array(
            [self.relative_angle_distance(agent_pos, agent_yaw, path[k]) for k in indices],
            dtype=np.float32
        )

    def lookahead_point(self, agent_pos, goal_pos, lookahead=1.0):
        """Point ~lookahead metres ahead along the path (for the expert)."""
        path = self.path(agent_pos)
        if not path:
            return tuple(goal_pos)
        for p in path:
            if np.hypot(p[0] - agent_pos[0], p[1] - agent_pos[1]) >= lookahead:
                return p
        return tuple(goal_pos)
