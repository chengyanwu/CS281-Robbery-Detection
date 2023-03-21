# GENERAL LIBRARIES
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple
import sys
import json as json
from tqdm import tqdm

# MACHINE LEARNING LIBRARIES
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import torch

# AcT LIBRARIES
from AcT.utils.tools import read_yaml, Logger
from AcT.utils.trainer import Trainer
from AcT.utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches
from AcT.utils.data import load_mpose, load_kinetics, random_flip, random_noise, one_hot
from AcT.utils.tools import CustomSchedule, CosineSchedule
from AcT.utils.tools import Logger

# AlphaPose LIBRARIES
from AlphaPose.detector.apis import get_detector
from AlphaPose.trackers.tracker_api import Tracker
from AlphaPose.trackers.tracker_cfg import cfg as tcfg
from AlphaPose.trackers import track
from AlphaPose.alphapose.models import builder
from AlphaPose.alphapose.utils.config import update_config
from AlphaPose.alphapose.utils.detector import DetectionLoader
from AlphaPose.alphapose.utils.file_detector import FileDetectionLoader
from AlphaPose.alphapose.utils.transforms import flip, flip_heatmap
from AlphaPose.alphapose.utils.vis import getTime
from AlphaPose.alphapose.utils.webcam_detector import WebCamDetectionLoader
from AlphaPose.alphapose.utils.writer import DataWriter

# OpenCV for camera streaming
import cv2

# Parsing user input
parser = argparse.ArgumentParser(description='Crime Detection Demo')
parser.add_argument('--AP_cfg', type=str, required=True,
                    help='Alphapose configure file name')
parser.add_argument('--AP_CKPT', type=str, required=True,
                    help='Alphapose model file name')
parser.add_argument('--AP_detector', type=str, required=True,
                    help='Alphapose detector name')
parser.add_argument('--AcT_cfg', type=str, required=True,
                    help='AcT configure file name')
parser.add_argument('--AcT_CKPT', type=str, required=True,
                    help='AcT weight file name')
parser.add_argument('--video', dest='video',
                    help='video-name', default="")

parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')

args = parser.parse_args()

# Reading AlphaPose Configuration
AP_cfg = update_config(args.AP_cfg)

# Reading AcT Configuration
AcT_cfg = read_yaml(args.AcT_cfg)
model_size = AcT_cfg['MODEL_SIZE']
n_heads = AcT_cfg[model_size]['N_HEADS']
n_layers = AcT_cfg[model_size]['N_LAYERS']
embed_dim = AcT_cfg[model_size]['EMBED_DIM']
dropout = AcT_cfg[model_size]['DROPOUT']
mlp_head_size = AcT_cfg[model_size]['MLP']
activation = tf.nn.gelu
d_model = 64 * n_heads
d_ff = d_model * 4
pos_emb = AcT_cfg['POS_EMB']

# print('Configuration: ')
# print(config)

# Helper Functions
def build_act():
    transformer = TransformerEncoder(d_model, n_heads, d_ff, dropout, activation, n_layers)

    inputs = tf.keras.layers.Input(shape=(AcT_cfg[AcT_cfg['DATASET']]['FRAMES'] // AcT_cfg['SUBSAMPLE'], 
                                            AcT_cfg[AcT_cfg['DATASET']]['KEYPOINTS'] * AcT_cfg['CHANNELS']))
    x = tf.keras.layers.Dense(d_model)(inputs)
    x = PatchClassEmbedding(d_model, AcT_cfg[AcT_cfg['DATASET']]['FRAMES'] // AcT_cfg['SUBSAMPLE'], 
                            pos_emb=None)(x)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
    x = tf.keras.layers.Dense(mlp_head_size)(x)
    outputs = tf.keras.layers.Dense(AcT_cfg[AcT_cfg['DATASET']]['CLASSES'])(x)
    return tf.keras.models.Model(inputs, outputs)

def set_GPU():
    # SET GPU 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[AcT_cfg['GPU']], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[AcT_cfg['GPU']], True)

def load_AP_Detector():
    # Check Input
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            mode = 'video'
            input_source = videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # TODO: 1. alphapose FileDetectionLoader to read video file and run yolo
    #       2. load alphapose pose model
    if mode == 'video':
        # get_detector loads Yolo
        det_loader = DetectionLoader(input_source, get_detector(args), AP_cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()
    
    # Load Pose Model
    pose_model = builder.build_sppe(AP_cfg.MODEL, preset_cfg=AP_cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.AP_CKPT,))
    pose_model.load_state_dict(torch.load(args.AP_CKPT, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(AP_cfg.DATASET.TRAIN)
    # load pose model to gpu
    pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    pose_model.eval()

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

# set up GPU
set_GPU()

# Built Pose Estimation Model From AlphaPose
load_AP_Detector()

# Built AcT
AcT_model = build_act()
AcT_model.load_weights(args.AcT_CKPT)


# print("---Model Summary-----")

# model.summary()
# # print(model.trainable_variables) 

# print("-----------------------")
# # model = trainer.model
# # cap = cv2.VideoCapture(0)

# print('-------Loading Keypoints----')
# path = sys.a
# with open(path) as f:
#     data = np.array(json.loads(f.read()))
# # print(data[0]['keypoints'])

# kp_seq = []
# keypoints = []
# keypoint = []

# for d in data:
#     for i, kp in enumerate(d["keypoints"]):
#         if (i + 1) % 3 != 0:
#             keypoint.append(kp)

#         if (len(keypoint) == 136):
#             # 1D vector : (272,)
#             keypoint_with_v = []
#             for j in range(68):
#                 if (j != 0):
#                     keypoint_with_v.append(keypoint[2*j])
#                     keypoint_with_v.append(keypoint[2*j+1])
#                     keypoint_with_v.append(
#                         keypoint[2*j]-keypoint[2*j-2])
#                     keypoint_with_v.append(
#                         keypoint[2*j+1]-keypoint[2*j-1])
#                 else:
#                     keypoint_with_v.append(keypoint[2*j])
#                     keypoint_with_v.append(keypoint[2*j+1])
#                     keypoint_with_v.append(0)
#                     keypoint_with_v.append(0)

#             # keypoints will be appended 30 times
#             # keypoints: 2D vector (30, 272)
#             keypoints.append(keypoint_with_v)
#             keypoint = []
#             break

#         if (len(keypoints) == 30):
#             # we only be append (30,272) to kq_seq
#             kp_seq.append(keypoints)
#             keypoints = []
# # for k in kp_seq:
# #     print(k)
# #     print("--------------------")
# kp_seq = np.asarray(kp_seq)
# n = len(kp_seq)
# print('input size: ', len(kp_seq))

# # model.eval()
# print(model.predict(tf.concat(kp_seq, axis=0)))


# print('-----------------')







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
