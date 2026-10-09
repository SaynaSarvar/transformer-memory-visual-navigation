using UnityEngine;

public class AgentObservation : MonoBehaviour
{
    [Header("References")]
    public CameraObservation cameraObservation;
    public GoalObservation goalObservation;

    [Header("Current Observation")]
    public float distanceToGoal;
    public float angleToGoal;

    void Update()
{
    CollectObservation();

    if (Input.GetKeyDown(KeyCode.O))
    {
        TestObservationPacket();
    }
}

    void CollectObservation()
    {
        if (goalObservation != null)
        {
            distanceToGoal = goalObservation.distanceToGoal;

            // Convert degrees to radians
            angleToGoal =
                goalObservation.angleToGoal * Mathf.Deg2Rad;
        }
    }

    public ObservationPacket GetObservation()
{
    byte[] imageData = null;

    if (cameraObservation != null)
    {
        imageData = cameraObservation.CaptureRGB();
    }

    return new ObservationPacket(
        imageData,
        80,
        60,
        3,
        distanceToGoal,
        angleToGoal
    );
}

    public Texture2D GetImage()
    {
        if (cameraObservation == null)
        {
            return null;
        }

        return cameraObservation.CaptureImage();
    }

    public float GetDistanceToGoal()
    {
        return distanceToGoal;
    }

    public float GetAngleToGoal()
    {
        return angleToGoal;
    }
    void TestObservationPacket()
{
    ObservationPacket observation = GetObservation();

    if (observation == null)
    {
        Debug.LogError("Observation Packet is null.");
        return;
    }

    if (observation.image == null)
    {
        Debug.LogError("Observation image is null.");
        return;
    }

    Debug.Log(
        "Observation Packet created successfully!"
    );

    Debug.Log(
        "Image size: " +
        observation.imageWidth +
        " x " +
        observation.imageHeight +
        " x " +
        observation.imageChannels
    );

    Debug.Log(
        "RGB byte count: " +
        observation.image.Length
    );

    Debug.Log(
        "Distance: " +
        observation.distanceToGoal
    );

    Debug.Log(
        "Angle (radians): " +
        observation.angleToGoal
    );
}
}