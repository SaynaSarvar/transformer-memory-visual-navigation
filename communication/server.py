import socket
import struct


HOST = "127.0.0.1"
PORT = 5000


def receive_exactly(connection, num_bytes):
    data = bytearray()

    while len(data) < num_bytes:

        chunk = connection.recv(
            num_bytes - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "Connection closed while receiving data."
            )

        data.extend(chunk)

    return bytes(data)


def receive_packet(connection):

    # -------------------------
    # Receive packet length
    # -------------------------

    length_data = receive_exactly(
        connection,
        4
    )

    packet_length = struct.unpack(
        "<I",
        length_data
    )[0]

    # -------------------------
    # Receive packet
    # -------------------------

    packet = receive_exactly(
        connection,
        packet_length
    )

    return packet


def decode_observation(packet):

    offset = 0

    # -------------------------
    # Image information
    # -------------------------

    image_width = struct.unpack_from(
        "<i",
        packet,
        offset
    )[0]

    offset += 4

    image_height = struct.unpack_from(
        "<i",
        packet,
        offset
    )[0]

    offset += 4

    image_channels = struct.unpack_from(
        "<i",
        packet,
        offset
    )[0]

    offset += 4

    image_length = struct.unpack_from(
        "<i",
        packet,
        offset
    )[0]

    offset += 4

    # -------------------------
    # Image
    # -------------------------

    image_bytes = packet[
        offset:
        offset + image_length
    ]

    offset += image_length

    # -------------------------
    # Goal information
    # -------------------------

    distance_to_goal = struct.unpack_from(
        "<f",
        packet,
        offset
    )[0]

    offset += 4

    angle_to_goal = struct.unpack_from(
        "<f",
        packet,
        offset
    )[0]

    offset += 4

    return (
        image_width,
        image_height,
        image_channels,
        image_bytes,
        distance_to_goal,
        angle_to_goal
    )


def send_packet(connection, packet):

    length_data = struct.pack(
        "<I",
        len(packet)
    )

    connection.sendall(
        length_data + packet
    )


def start_server():

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind(
        (HOST, PORT)
    )

    server.listen(1)

    print(
        f"Server listening on "
        f"{HOST}:{PORT}"
    )

    connection, address = server.accept()

    print(
        f"Unity connected from {address}"
    )

    try:

        while True:

            # Receive observation
            packet = receive_packet(
                connection
            )

            print()
            print(
                "========== Observation =========="
            )

            print(
                "Packet size:",
                len(packet)
            )

            (
                image_width,
                image_height,
                image_channels,
                image_bytes,
                distance_to_goal,
                angle_to_goal
            ) = decode_observation(
                packet
            )

            print(
                "Image:",
                image_width,
                "x",
                image_height,
                "x",
                image_channels
            )

            print(
                "Image bytes:",
                len(image_bytes)
            )

            print(
                "Distance:",
                distance_to_goal
            )

            print(
                "Angle:",
                angle_to_goal
            )

            print(
                "================================="
            )
            # -------------------------
            # Temporary test action
            # -------------------------

            linear_velocity = 0.5
            angular_velocity = 0.0

            action = struct.pack(
                "<ff",
                linear_velocity,
                angular_velocity
            )

            send_packet(
                connection,
                action
            )

            print(
                "Action sent:",
                linear_velocity,
                angular_velocity
            )

    except ConnectionError as error:

        print(
            f"Connection closed: {error}"
        )

    finally:

        connection.close()
        server.close()


if __name__ == "__main__":
    start_server()
