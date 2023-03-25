# GENERAL LIBRARIES
import socket
import math
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import argparse
from datetime import datetime
# MACHINE LEARNING LIBRARIES
import numpy as np
import tensorflow as tf

# CUSTOM LIBRARIES

from utils.tools import read_yaml, Logger
from utils.trainer import Trainer
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches
from utils.data import load_mpose, load_kinetics, random_flip, random_noise, one_hot
from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger
import sys
sys.path.append(os.path.abspath('../AcT'))
# sys.path.append(os.path.abspath('../bin'))

# Use OpenCV to capture webcam video stream
import cv2

# crime detection
from typing import Any, Dict, List, Tuple
from sklearn import preprocessing

import json as json

config = read_yaml('/home/homesecurity/CS281-Robbery-Detection/AcT/utils/config.yaml')
model_size = config['MODEL_SIZE']
n_heads = config[model_size]['N_HEADS']
n_layers = config[model_size]['N_LAYERS']
embed_dim = config[model_size]['EMBED_DIM']
dropout = config[model_size]['DROPOUT']
mlp_head_size = config[model_size]['MLP']
activation = tf.nn.gelu
d_model = 64 * n_heads
d_ff = d_model * 4
pos_emb = config['POS_EMB']

# print('Configuration: ')
# print(config)

now = datetime.now()
logger = Logger(config['LOG_DIR']+now.strftime("%y%m%d%H%M%S")+'.txt')


# SET GPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[config['GPU']], 'GPU')
tf.config.experimental.set_memory_growth(gpus[config['GPU']], True)

# Helper Functions
def build_act(transformer):
    inputs = tf.keras.layers.Input(shape=(config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'], 
                                            config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS']))
    x = tf.keras.layers.Dense(d_model)(inputs)
    x = PatchClassEmbedding(d_model, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'], 
                            pos_emb=None)(x)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
    x = tf.keras.layers.Dense(mlp_head_size)(x)
    outputs = tf.keras.layers.Dense(config[config['DATASET']]['CLASSES'])(x)
    return tf.keras.models.Model(inputs, outputs)


# build model
# trainer = Trainer(config, logger)
# trainer.get_model()
# model = trainer.return_model()
file_path = sys.argv[1]
print(f'filepath: {file_path}')

transformer = TransformerEncoder(d_model, n_heads, d_ff, dropout, activation, n_layers)
model = build_act(transformer)
model.load_weights('/home/homesecurity/CS281-Robbery-Detection/AcT/bin/AcT_large_1_2.h5')
# model = tf.keras.models.load_model('AcT_pretrained_weights/AcT_base_1_0.h5')

print("---Model Summary-----")

model.summary()
# print(model.trainable_variables)

print("-----------------------")
# model = trainer.model
# cap = cv2.VideoCapture(0)

# print('-------Loading Keypoints----')
kp_seq: List[List[List[float]]] = []

with open(file_path, "r") as f:
    data: List[Dict[str, Any]] = json.load(f)

# keypoint/s are intiazlied per file
keypoints: List[List[float]] = []
keypoint: List[float] = []

for d in data:
    for i, kp in enumerate(d["keypoints"]):
        if (i + 1) % 3 != 0:
            keypoint.append(kp)

        if (len(keypoint) == 136):
            # 1D vector : (272,)
            keypoint_with_v = []
            for j in range(68):
                keypoint_with_v.append(keypoint[2*j])
                keypoint_with_v.append(keypoint[2*j+1])

            # keypoints will be appended 30 times
            # keypoints: 2D vector (30, 272)
            keypoints.append(keypoint_with_v)
            keypoint = []
            break

        if (len(keypoints) == 30):
            # we only be append (30,272) to kq_seq
            kp_seq.append(keypoints)
            keypoints = []
 
n: int= len(kp_seq)

new_kp_seq: List[List[List[float]]] = [[]]
extended_kp_seq: List[List[float]] = kp_seq[0]
for i in range(1, n):
    extended_kp_seq.extend(kp_seq[i])
# append the rest of keypoints that haven't been appended to kp_seq
# size of keypoints must < 36
extended_kp_seq.extend(keypoints)

# print(f'extended kp_seq size: {np.asarray(extended_kp_seq).shape}')

total_number_of_frames: int = len(extended_kp_seq)
# print("total_number_of_frames: ", total_number_of_frames)

multiplier: float = total_number_of_frames / 30
for i in range(30):
    idx = int(i * multiplier)
    new_kp_seq[0].append(extended_kp_seq[int(idx)])

# print(f'new kp_seq size: {np.asarray(new_kp_seq).shape}')


# print('input size: ', len(kp_seq))
# print('inner input size 1: ', len(kp_seq[0]))
# print('inner input size 2: ', len(kp_seq[0][0]))

# model.eval()
# print(model.predict(np.asarray(new_kp_seq)))
# start_time = time.monotonic()

# with torch.no_grad
# print(np.argmax(tf.nn.softmax(model.predict(np.asarray(new_kp_seq)), axis=-1), axis=1))
# end_time = time.monotonic()
# end_time = time.perf_counter()

# print(model.predict(np.asarray(new_kp_seq)))
# print(np.argmax(tf.nn.softmax(model.predict(np.asarray(new_kp_seq)), axis=-1), axis=1) )




# HOST = '127.0.0.1'  # The remote host (in this case, the same machine)
# PORT = 5000        # The same port as used by the server

# # Define the variable you want to send
# # for demo
# steal_result = np.argmax(tf.nn.softmax(model.predict(np.asarray(new_kp_seq)), axis=-1), axis=1)[0]
# print('ZM steal result', steal_result)

# # Create a socket object
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Connect the socket to the port where the receiver is listening
# sock.connect((HOST, PORT))

# # Convert the variable to a string and send it over the socket
# sock.sendall(str(steal_result).encode())

# # Close the socket
# sock.close()

start_time = time.perf_counter()
if np.argmax(tf.nn.softmax(model.predict(np.asarray(new_kp_seq)), axis=-1), axis=1) == 0:
    
    with open("result.txt", "w") as f:
    # Write the variable to the file
        f.write('0')
    
    end_time = time.perf_counter()

    print("Action: Normal")
else:
    with open("result.txt", "w") as f:
    # Write the variable to the file
        f.write('1')
    
    end_time = time.perf_counter()
    print("Action: Steal")

print(f"inference time: {end_time-start_time} s.")

print('-----------------')







# while True:
#     # Read frame from camera
#     ret, image_np = cap.read()
#     if image_np is None:
#         break

#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#     image_np_expanded = np.expand_dims(image_np, axis=0)

#     # Things to try:
#     # Flip horizontally
#     # image_np = np.fliplr(image_np).copy()

#     # Convert image to grayscale
#     # image_np = np.tile(
#     #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

#     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

#     label_id_offset = 1
#     image_np_with_detections = image_np.copy()

#     # Display output
#     cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
