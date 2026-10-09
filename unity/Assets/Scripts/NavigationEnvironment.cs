using UnityEngine;

// Unity version of envs/navigation_env.py  (v2: harder layouts)
//
// Every reset:
//   1. Obstacles (boxes AND long walls) are scattered over the whole room,
//      each randomly rotated 0 or 90 degrees, never overlapping and always
//      leaving a gap of at least `obstacleGap` metres between them.
//   2. Agent and goal are placed ANYWHERE in the room (not in fixed corners),
//      away from obstacles and at least `minAgentGoalDistance` apart.
//   3. The agent starts with a random heading.
// Python additionally checks that a path exists (and re-rolls if not).
public class NavigationEnvironment : MonoBehaviour
{
    [Header("Scene References")]
    public AgentController agent;
    public CameraObservation cameraObservation;
    public Transform goal;
    public Transform[] obstacles;

    [Header("Episode")]
    public float stepDeltaTime = 0.1f;
    public int maxEpisodeSteps = 400;
    public float successDistance = 1.0f;
    public int maxCollisionsPerEpisode = 20;

    [Header("Reward")]
    public float timePenalty = 0.01f;
    public float collisionPenalty = 0.3f;   // was 0.1 - discourages scraping along obstacles
    public float successReward = 10f;
    public float stuckPenalty = 2f;

    [Header("Random Layout")]
    public float spawnHalfSize = 8.5f;          // agent / goal x,z in [-8.5, 8.5]
    public float obstacleHalfSize = 8.0f;       // obstacle centres in [-8, 8]
    public float roomInnerHalfSize = 9.75f;     // inner face of the outer walls
    public float minAgentGoalDistance = 10f;
    public float obstacleGap = 1.2f;            // free space between obstacles (> agent width)
    public float spawnClearance = 1.0f;         // agent/goal distance from any obstacle edge
    public bool randomObstacleRotation = true;
    public bool randomStartHeading = true;

    [Header("Debug (read only)")]
    public int stepCount;
    public int collisionCount;
    public float lastReward;
    public float distanceToGoal;
    public float angleToGoalDegrees;

    private float previousDistance;

    // footprint of each obstacle after placement: centre (x,z) + half extents
    private Vector2[] obstacleCentres;
    private Vector2[] obstacleHalfExtents;

    public struct StepResult
    {
        public byte[] image;
        public float distance;
        public float angle;
        public float reward;
        public bool terminated;
        public bool truncated;
        public bool success;
        public bool collision;
    }

    // ------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------
    public StepResult ResetEpisode(int seed)
    {
        System.Random rng = new System.Random(seed);

        PlaceObstacles(rng);

        Vector2 agentXZ = SampleFreePoint(rng, null, 0f);
        Vector2 goalXZ = SampleFreePoint(rng, agentXZ, minAgentGoalDistance);

        Vector3 g = goal.position;
        goal.position = new Vector3(goalXZ.x, g.y, goalXZ.y);

        float yaw = randomStartHeading ? (float)(rng.NextDouble() * 360.0) : 0f;
        agent.Teleport(new Vector3(agentXZ.x, agent.transform.position.y, agentXZ.y), yaw);

        stepCount = 0;
        collisionCount = 0;
        lastReward = 0f;

        ComputeGoal(out float distance, out float angle);
        previousDistance = distance;

        return BuildResult(distance, angle, 0f, false, false, false, false);
    }

    // ------------------------------------------------------------
    // Step
    // ------------------------------------------------------------
    public StepResult Step(float v, float omega)
    {
        bool collision = agent.Move(v, omega, stepDeltaTime);

        ComputeGoal(out float distance, out float angle);

        float reward = previousDistance - distance;
        previousDistance = distance;
        reward -= timePenalty;

        if (collision)
        {
            reward -= collisionPenalty;
            collisionCount++;
        }

        bool success = distance < successDistance;
        if (success)
        {
            reward += successReward;
        }

        stepCount++;

        bool truncated = false;
        if (!success && stepCount >= maxEpisodeSteps)
        {
            truncated = true;
        }
        if (!success && collisionCount >= maxCollisionsPerEpisode)
        {
            truncated = true;
            reward -= stuckPenalty;
        }

        lastReward = reward;
        return BuildResult(distance, angle, reward, success, truncated, success, collision);
    }

    // ------------------------------------------------------------
    // Layout generation
    // ------------------------------------------------------------
    private void PlaceObstacles(System.Random rng)
    {
        int n = obstacles.Length;
        obstacleCentres = new Vector2[n];
        obstacleHalfExtents = new Vector2[n];
        int placed = 0;

        for (int i = 0; i < n; i++)
        {
            Transform o = obstacles[i];

            bool rotate = randomObstacleRotation && rng.NextDouble() < 0.5;
            Vector3 s = o.localScale;                       // obstacles are unit cubes
            Vector2 half = rotate ? new Vector2(s.z, s.x) * 0.5f : new Vector2(s.x, s.z) * 0.5f;

            // keep the whole obstacle inside the room
            float limitX = Mathf.Min(obstacleHalfSize, roomInnerHalfSize - half.x - 0.1f);
            float limitZ = Mathf.Min(obstacleHalfSize, roomInnerHalfSize - half.y - 0.1f);

            bool ok = false;
            Vector2 c = Vector2.zero;

            for (int attempt = 0; attempt < 300 && !ok; attempt++)
            {
                c = new Vector2(
                    Mathf.Lerp(-limitX, limitX, (float)rng.NextDouble()),
                    Mathf.Lerp(-limitZ, limitZ, (float)rng.NextDouble())
                );

                ok = true;
                for (int j = 0; j < placed; j++)
                {
                    bool overlapX = Mathf.Abs(c.x - obstacleCentres[j].x) < half.x + obstacleHalfExtents[j].x + obstacleGap;
                    bool overlapZ = Mathf.Abs(c.y - obstacleCentres[j].y) < half.y + obstacleHalfExtents[j].y + obstacleGap;
                    if (overlapX && overlapZ) { ok = false; break; }
                }
            }

            if (!ok)
            {
                // No room left: park it far below the floor so it plays no part.
                o.position = new Vector3(0f, -50f, 0f);
                obstacleCentres[i] = new Vector2(1000f, 1000f);
                obstacleHalfExtents[i] = Vector2.zero;
                placed++;
                continue;
            }

            o.rotation = Quaternion.Euler(0f, rotate ? 90f : 0f, 0f);
            o.position = new Vector3(c.x, Mathf.Max(o.position.y, s.y * 0.5f), c.y);

            obstacleCentres[i] = c;
            obstacleHalfExtents[i] = half;
            placed++;
        }

        Physics.SyncTransforms();
    }

    private Vector2 SampleFreePoint(System.Random rng, Vector2? awayFrom, float minDistance)
    {
        Vector2 best = Vector2.zero;
        float bestScore = -1f;

        for (int attempt = 0; attempt < 500; attempt++)
        {
            Vector2 p = new Vector2(
                Mathf.Lerp(-spawnHalfSize, spawnHalfSize, (float)rng.NextDouble()),
                Mathf.Lerp(-spawnHalfSize, spawnHalfSize, (float)rng.NextDouble())
            );

            if (DistanceToNearestObstacle(p) < spawnClearance)
            {
                continue;
            }

            float d = awayFrom.HasValue ? Vector2.Distance(p, awayFrom.Value) : float.MaxValue;
            if (d >= minDistance)
            {
                return p;
            }

            if (d > bestScore) { bestScore = d; best = p; }   // fallback: farthest free point
        }

        return best;
    }

    private float DistanceToNearestObstacle(Vector2 p)
    {
        float nearest = float.MaxValue;
        for (int i = 0; i < obstacleCentres.Length; i++)
        {
            float dx = Mathf.Max(Mathf.Abs(p.x - obstacleCentres[i].x) - obstacleHalfExtents[i].x, 0f);
            float dz = Mathf.Max(Mathf.Abs(p.y - obstacleCentres[i].y) - obstacleHalfExtents[i].y, 0f);
            nearest = Mathf.Min(nearest, Mathf.Sqrt(dx * dx + dz * dz));
        }
        return nearest;
    }

    // Footprints for PythonClient (x, z, sizeX, sizeZ) - already rotation-aware.
    public Vector4 GetObstacleFootprint(int i)
    {
        if (obstacleCentres == null)
        {
            Vector3 p = obstacles[i].position;
            Vector3 s = obstacles[i].lossyScale;
            return new Vector4(p.x, p.z, s.x, s.z);
        }
        return new Vector4(obstacleCentres[i].x, obstacleCentres[i].y,
                           obstacleHalfExtents[i].x * 2f, obstacleHalfExtents[i].y * 2f);
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------
    private void ComputeGoal(out float distance, out float angle)
    {
        Vector3 toGoal = goal.position - agent.transform.position;
        toGoal.y = 0f;

        distance = toGoal.magnitude;

        Vector3 local = agent.transform.InverseTransformDirection(toGoal);
        angle = Mathf.Atan2(local.x, local.z);

        distanceToGoal = distance;
        angleToGoalDegrees = angle * Mathf.Rad2Deg;
    }

    private StepResult BuildResult(float distance, float angle, float reward,
                                   bool terminated, bool truncated, bool success, bool collision)
    {
        return new StepResult
        {
            image = cameraObservation.CaptureRGB(),
            distance = distance,
            angle = angle,
            reward = reward,
            terminated = terminated,
            truncated = truncated,
            success = success,
            collision = collision
        };
    }
}
