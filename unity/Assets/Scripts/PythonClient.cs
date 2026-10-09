using System;
using System.IO;
using System.Net.Sockets;
using UnityEngine;

// TCP bridge between Unity and Python.
// Python is the SERVER (it starts first and waits). Unity connects and
// then simply obeys commands, one per frame:
//
//   Python -> Unity   [uint32 length][payload]
//     RESET : byte 0, int32 seed
//     STEP  : byte 1, float32 v, float32 omega
//     CLOSE : byte 2
//
//   Unity -> Python   [uint32 length][payload]
//     int32 width, int32 height, int32 channels, int32 imageLength, bytes image
//     float32 distance, float32 angle(rad), float32 reward
//     byte terminated, byte truncated, byte success, byte collision
//     float32 agentX, float32 agentZ, float32 agentYaw(rad), float32 goalX, float32 goalZ
//     int32 obstacleCount, then per obstacle: float32 x, z, sizeX, sizeZ
//     int32 stepCount
//
// Everything is little-endian (BinaryWriter default).
public class PythonClient : MonoBehaviour
{
    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 5000;
    public float reconnectInterval = 1f;
    public int receiveTimeoutMs = 60000;

    [Header("Environment")]
    public NavigationEnvironment environment;

    private const byte CMD_RESET = 0;
    private const byte CMD_STEP = 1;
    private const byte CMD_CLOSE = 2;

    private TcpClient client;
    private NetworkStream stream;
    private float nextConnectTime;
    private bool loggedWaiting;

    void Start()
    {
        // Keep running and don't cap FPS - more steps per second.
        Application.runInBackground = true;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
    }

    void Update()
    {
        if (stream == null)
        {
            if (Time.unscaledTime >= nextConnectTime)
            {
                nextConnectTime = Time.unscaledTime + reconnectInterval;
                TryConnect();
            }
            return;
        }

        try
        {
            HandleOneCommand();
        }
        catch (Exception error)
        {
            Debug.LogWarning("Python connection closed: " + error.Message);
            Disconnect();
        }
    }

    // ------------------------------------------------------------
    // Connection
    // ------------------------------------------------------------
    private void TryConnect()
    {
        try
        {
            client = new TcpClient();
            client.NoDelay = true;
            client.ReceiveTimeout = receiveTimeoutMs;
            client.Connect(host, port);
            stream = client.GetStream();
            loggedWaiting = false;
            Debug.Log("Connected to Python server at " + host + ":" + port);
        }
        catch (Exception)
        {
            client?.Close();
            client = null;
            stream = null;

            if (!loggedWaiting)
            {
                Debug.Log("Waiting for Python server on " + host + ":" + port + " ...");
                loggedWaiting = true;
            }
        }
    }

    private void Disconnect()
    {
        stream?.Close();
        client?.Close();
        stream = null;
        client = null;
    }

    void OnApplicationQuit()
    {
        Disconnect();
    }

    // ------------------------------------------------------------
    // Command handling
    // ------------------------------------------------------------
    private void HandleOneCommand()
    {
        int length = BitConverter.ToInt32(ReceiveExactly(4), 0);
        byte[] payload = ReceiveExactly(length);

        using (BinaryReader reader = new BinaryReader(new MemoryStream(payload)))
        {
            byte command = reader.ReadByte();

            if (command == CMD_RESET)
            {
                int seed = reader.ReadInt32();
                SendResult(environment.ResetEpisode(seed));
            }
            else if (command == CMD_STEP)
            {
                float v = reader.ReadSingle();
                float omega = reader.ReadSingle();
                SendResult(environment.Step(v, omega));
            }
            else if (command == CMD_CLOSE)
            {
                Debug.Log("Python asked to close the connection.");
                Disconnect();
            }
            else
            {
                throw new Exception("Unknown command: " + command);
            }
        }
    }

    private void SendResult(NavigationEnvironment.StepResult result)
    {
        CameraObservation cam = environment.cameraObservation;
        Transform agentT = environment.agent.transform;
        Transform goalT = environment.goal;

        using (MemoryStream memory = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(memory))
        {
            writer.Write(cam.imageWidth);
            writer.Write(cam.imageHeight);
            writer.Write(3);
            writer.Write(result.image.Length);
            writer.Write(result.image);

            writer.Write(result.distance);
            writer.Write(result.angle);
            writer.Write(result.reward);

            writer.Write(result.terminated);
            writer.Write(result.truncated);
            writer.Write(result.success);
            writer.Write(result.collision);

            writer.Write(agentT.position.x);
            writer.Write(agentT.position.z);
            writer.Write(agentT.eulerAngles.y * Mathf.Deg2Rad);
            writer.Write(goalT.position.x);
            writer.Write(goalT.position.z);

            Transform[] obstacles = environment.obstacles;
            writer.Write(obstacles.Length);
            for (int i = 0; i < obstacles.Length; i++)
            {
                // centre x, z and footprint size x, z (rotation already applied)
                Vector4 f = environment.GetObstacleFootprint(i);
                writer.Write(f.x);
                writer.Write(f.y);
                writer.Write(f.z);
                writer.Write(f.w);
            }

            writer.Write(environment.stepCount);

            byte[] packet = memory.ToArray();
            byte[] lengthBytes = BitConverter.GetBytes(packet.Length);

            stream.Write(lengthBytes, 0, 4);
            stream.Write(packet, 0, packet.Length);
            stream.Flush();
        }
    }

    private byte[] ReceiveExactly(int numBytes)
    {
        byte[] data = new byte[numBytes];
        int total = 0;

        while (total < numBytes)
        {
            int received = stream.Read(data, total, numBytes - total);
            if (received == 0)
            {
                throw new Exception("Connection closed while receiving data.");
            }
            total += received;
        }

        return data;
    }
}
