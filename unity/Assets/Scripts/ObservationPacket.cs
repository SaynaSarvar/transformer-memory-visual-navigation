using UnityEngine;

[System.Serializable]
public class ObservationPacket
{
    // Raw RGB image data
    public byte[] image;

    // Image dimensions
    public int imageWidth;
    public int imageHeight;
    public int imageChannels;

    // Goal observation
    public float distanceToGoal;
    public float angleToGoal;

    public ObservationPacket(
        byte[] image,
        int imageWidth,
        int imageHeight,
        int imageChannels,
        float distanceToGoal,
        float angleToGoal
    )
    {
        this.image = image;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.imageChannels = imageChannels;

        this.distanceToGoal = distanceToGoal;
        this.angleToGoal = angleToGoal;
    }
}