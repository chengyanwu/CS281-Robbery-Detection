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
import time

# sys.path.append('/home/homesecurity/CS281-Robbery-Detection/AlphaPose')
# sys.path.insert(0,'/home/homesecurity/CS281-Robbery-Detection/AlphaPose')

# sys.path.append(os.path.abspath('AlphaPose'))
# print(sys.path)

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
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')

args = parser.parse_args()
args.posebatch = args.posebatch * len(args.gpus)


# Reading AlphaPose Configuration
AP_cfg = update_config(args.AP_cfg)
print('------Alphapose Configuration----')
print(AP_cfg)
print('--------------------------------')

# Reading AcT Configuration
AcT_cfg = read_yaml(args.AcT_cfg)
print('------AcT Configuration----')
print(AcT_cfg)
print('--------------------------------')
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

    # fpr tensorflow (AcT)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[AcT_cfg['GPU']], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[AcT_cfg['GPU']], True)

    # for pytorch (AP)
    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")

    return gpus

def load_AP_Detector():
    # Check Input
    print(len(args.video))
    print('-------')
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            mode = 'video'
            input_source = videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')
    else:
        mode = 'none'
        input_source = 'none'

    # TODO: 1. alphapose FileDetectionLoader to read video file and run yolo
    #       2. load alphapose pose model
    if mode == 'video':
        # get_detector loads Yolo
        det_loader = DetectionLoader(input_source, get_detector(args), AP_cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(input_source, get_detector(args), AP_cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()
    
    # Load Pose Model
    pose_model = builder.build_sppe(AP_cfg.MODEL, preset_cfg=AP_cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.AP_CKPT,))
    pose_model.load_state_dict(torch.load(args.AP_CKPT, map_location=args.device))
    # pose_dataset = builder.retrieve_dataset(AP_cfg.DATASET.TRAIN)
    # load pose model to gpu
    pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    pose_model.eval()

    return pose_model, det_loader, im_names_desc

# set up GPU
gpus = set_GPU()

# Built Pose Estimation Model From AlphaPose
pose_model, det_loader, im_names_desc = load_AP_Detector()
data_len = det_loader.length
im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
writer = DataWriter(AP_cfg, args, save_video=False, queueSize=args.qsize).start()

runtime_profile = {
    'dt': [],
    'pt': [],
    'pn': []
}

try:
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name)
                continue
            if args.profile:
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
            # Pose Estimation
            inps = inps.to(args.device)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % args.posebatch:
                leftover = 1
            num_batches = datalen // args.posebatch + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * args.posebatch:min((j + 1) * args.posebatch, datalen)]
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            if args.profile:
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)
            if args.profile:
                ckpt_time, pose_time = getTime(ckpt_time)
            hm = hm.cpu()
            writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
            if args.profile:
                ckpt_time, post_time = getTime(ckpt_time)
            if args.profile:
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
                'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )
    print('===========================> Finish Model Running.')
    while(writer.running()):
        time.sleep(1)
        print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
    writer.stop()
    det_loader.stop()

except Exception as e:
    print(repr(e))
    print('An error as above occurs when processing the images, please check it')
    pass
except KeyboardInterrupt:
    print('===========================> Finish Model Running.')
    # Thread won't be killed when press Ctrl+C
    if args.sp:
        det_loader.terminate()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
    else:
        # subprocesses are killed, manually clear queues

        det_loader.terminate()
        writer.terminate()
        writer.clear_queues()
        det_loader.clear_queues()

# Built AcT
# AcT_model = build_act()
# AcT_model.load_weights(args.AcT_CKPT)



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
