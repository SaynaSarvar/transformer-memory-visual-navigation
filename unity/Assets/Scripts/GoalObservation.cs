using UnityEngine;

public class GoalObservation : MonoBehaviour
{
    [Header("Goal Reference")]
    public Transform goal;

    [Header("Observation")]
    public float distanceToGoal;
    public float angleToGoal;

    void Update()
    {
        CalculateGoalObservation();
    }

    void CalculateGoalObservation()
    {
        if (goal == null)
        {
            return;
        }

        // Vector from agent to goal
        Vector3 directionToGoal = goal.position - transform.position;

        // Ignore height difference
        directionToGoal.y = 0f;

        // Distance to goal
        distanceToGoal = directionToGoal.magnitude;

        // Direction relative to agent
        Vector3 localDirection =
            transform.InverseTransformDirection(directionToGoal);

        // Angle between agent's forward direction and goal
        angleToGoal =
            Mathf.Atan2(localDirection.x, localDirection.z)
            * Mathf.Rad2Deg;
    }
}