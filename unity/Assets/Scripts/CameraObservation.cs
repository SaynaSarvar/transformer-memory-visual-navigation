using UnityEngine;

public class CameraObservation : MonoBehaviour
{
    [Header("Image Settings")]
    public int imageWidth = 80;
    public int imageHeight = 60;

    private Camera agentCamera;
    private RenderTexture renderTexture;
    private Texture2D texture;

    void Start()
    {
        agentCamera = GetComponent<Camera>();

        renderTexture = new RenderTexture(
            imageWidth,
            imageHeight,
            24
        );

        texture = new Texture2D(
            imageWidth,
            imageHeight,
            TextureFormat.RGB24,
            false
        );

        Debug.Log("CameraObservation initialized.");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            Texture2D capturedImage = CaptureImage();

            if (capturedImage != null)
            {
                Debug.Log(
                    "Image captured: " +
                    capturedImage.width +
                    " x " +
                    capturedImage.height
                );

                byte[] imageBytes = capturedImage.EncodeToPNG();

                string filePath =
                    Application.dataPath +
                    "/captured_observation.png";

                System.IO.File.WriteAllBytes(
                    filePath,
                    imageBytes
                );

                Debug.Log(
                    "Image saved to: " + filePath
                );
            }
        }
    }

    public Texture2D CaptureImage()
    {
        if (agentCamera == null)
        {
            return null;
        }

        RenderTexture previousTarget = agentCamera.targetTexture;

        agentCamera.targetTexture = renderTexture;

        agentCamera.Render();

        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = renderTexture;

        texture.ReadPixels(
            new Rect(
                0,
                0,
                imageWidth,
                imageHeight
            ),
            0,
            0
        );

        texture.Apply();

        RenderTexture.active = previousActive;
        agentCamera.targetTexture = previousTarget;

        return texture;
    }
    public byte[] CaptureRGB()
{
    Texture2D capturedImage = CaptureImage();

    if (capturedImage == null)
    {
        return null;
    }

    // Get raw RGB24 pixel data
    byte[] rgbData = capturedImage.GetRawTextureData();

    return rgbData;
}
}