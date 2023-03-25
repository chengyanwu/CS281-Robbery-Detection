import socket
import pickle
import struct
import cv2
import time

server_ip = '128.111.180.167'  # Replace with your server's IP address
server_port = 8089

# Create a socket and bind to the IP and port
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print ('Socket created')
server_socket.bind((server_ip, server_port))
print('Socket bind complete')
server_socket.listen(10)
print('Socket now listening')

# Accept incoming connections
conn, addr = server_socket.accept()

frames_received = 0
frames_list = []

# Video parameters
video_codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = 30
video_size = (640, 480)
video_out = cv2.VideoWriter('output.avi', video_codec, video_fps, video_size)

while True:
    # Receive the message size
    message_size = struct.unpack("Q", conn.recv(8))[0]

    # Receive the serialized frame
    data = b""
    while len(data) < message_size:
        data += conn.recv(4096)

    # Deserialize the frame
    frame = pickle.loads(data)
    frames_received += 1

    # Save the frame to the list
    frames_list.append(frame)

    # When 30 frames are received, save them as a video
    if frames_received == 30:
        for frame in frames_list:
            video_out.write(frame)
        frames_list = []
        frames_received = 0

        # If you want to stop after the first video, uncomment the following lines:
        # video_out.release()
        # break

conn.close()
server_socket.close()
