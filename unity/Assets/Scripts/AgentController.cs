using UnityEngine;

// Moves the agent like the MiniWorld robot:
//   v     in [0, 1]  -> forward speed  (v * maxLinearVelocity m/s)
//   omega in [-1, 1] -> turning speed  (omega * maxAngularVelocity rad/s)
// Positive omega turns RIGHT (clockwise seen from above). The goal angle sent to
// Python is also positive when the goal is on the RIGHT, so a positive angle is
// reduced by a positive omega - the same convention as your MiniWorld env.
//
// Collisions are handled by casting the capsule forward against the
// "Obstacle" layer and sliding along whatever it hits (same idea as the
// tangential slide in navigation_env.py). The Rigidbody must be KINEMATIC.
[RequireComponent(typeof(CapsuleCollider))]
public class AgentController : MonoBehaviour
{
    [Header("Movement")]
    public float maxLinearVelocity = 2f;   // m/s when v = 1
    public float maxAngularVelocity = 1f;  // rad/s when omega = +-1

    [Header("Collision")]
    public LayerMask obstacleMask;         // set to "Obstacle" in the Inspector
    public float skinWidth = 0.02f;

    [Header("Keyboard Debug (turn OFF for training)")]
    public bool keyboardControl = false;

    private CapsuleCollider capsule;

    void Awake()
    {
        capsule = GetComponent<CapsuleCollider>();
    }

    void Update()
    {
        if (!keyboardControl)
        {
            return;
        }

        float v = Input.GetAxis("Vertical");
        float omega = Input.GetAxis("Horizontal");
        Move(v, omega, Time.deltaTime);
    }

    // Applies one action for dt seconds. Returns true if the agent hit something.
    public bool Move(float v, float omega, float dt)
    {
        v = Mathf.Clamp(v, -1f, 1f);
        omega = Mathf.Clamp(omega, -1f, 1f);

        // 1. Rotate
        float yawDegrees = omega * maxAngularVelocity * dt * Mathf.Rad2Deg;
        transform.Rotate(0f, yawDegrees, 0f, Space.World);

        // 2. Translate (with collision + slide)
        Vector3 move = transform.forward * (v * maxLinearVelocity * dt);
        move.y = 0f;

        return MoveWithCollision(move);
    }

    // Instantly places the agent (used by episode reset).
    public void Teleport(Vector3 position, float yawDegrees)
    {
        transform.SetPositionAndRotation(position, Quaternion.Euler(0f, yawDegrees, 0f));
        Physics.SyncTransforms();
    }

    private bool MoveWithCollision(Vector3 move)
    {
        Physics.SyncTransforms();
        bool collided = false;

        // At most 2 passes: move until contact, then slide along the surface.
        for (int pass = 0; pass < 2; pass++)
        {
            float distance = move.magnitude;
            if (distance < 1e-5f)
            {
                break;
            }

            Vector3 direction = move / distance;
            GetCapsulePoints(out Vector3 p1, out Vector3 p2, out float radius);

            bool hitSomething = Physics.CapsuleCast(
                p1, p2, radius, direction,
                out RaycastHit hit,
                distance + skinWidth,
                obstacleMask,
                QueryTriggerInteraction.Ignore
            );

            if (!hitSomething)
            {
                transform.position += move;
                break;
            }

            collided = true;

            float allowed = Mathf.Max(0f, hit.distance - skinWidth);
            transform.position += direction * allowed;
            Physics.SyncTransforms();

            // Remove the part of the movement that pushes into the surface
            Vector3 remaining = move - direction * allowed;
            Vector3 normal = hit.normal;
            normal.y = 0f;

            if (normal.sqrMagnitude < 1e-6f)
            {
                break;
            }

            normal.Normalize();
            move = remaining - Vector3.Dot(remaining, normal) * normal;
        }

        Physics.SyncTransforms();
        return collided;
    }

    private void GetCapsulePoints(out Vector3 p1, out Vector3 p2, out float radius)
    {
        Vector3 scale = transform.lossyScale;
        radius = capsule.radius * Mathf.Max(Mathf.Abs(scale.x), Mathf.Abs(scale.z)) * 0.98f;
        float halfHeight = Mathf.Max(capsule.height * Mathf.Abs(scale.y) * 0.5f, radius);
        Vector3 center = transform.TransformPoint(capsule.center);
        float offset = halfHeight - radius;

        p1 = center + Vector3.up * offset;
        p2 = center - Vector3.up * offset;
    }
}
