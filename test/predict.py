# GENERAL LIBRARIES
import socket
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
# import cv2
# from result import return_result



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
from utils.tools import read_yaml, Logger
from utils.trainer import Trainer
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches
from utils.data import load_mpose, load_kinetics, random_flip, random_noise, one_hot
from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger

# AlphaPose LIBRARIES
from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

# OpenCV for camera streaming
import cv2


# HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
# PORT = 5000        # Port to listen on (non-privileged ports are > 1023)

# # Create a socket object
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Bind the socket to a specific address and port
# sock.bind((HOST, PORT))

# # Listen for incoming connections (only 1 at a time)
# sock.listen(1)

# # Wait for a connection
# conn, addr = sock.accept()

# # Receive the data
# data = conn.recv(1024)

# # Convert the received data back to the original variable type
# variable_received = int(data.decode())

# # Close the connection and the socket
# conn.close()
# sock.close()

# # Use the received variable
# print("The received variable is:", variable_received)

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

# dummy arguments
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--profile', default=True, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--showbox', default=True, action='store_true',
                    help='visualize human bbox')

args = parser.parse_args()
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = False
args.pose_flow = False

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


global video_path
global video_result
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
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[AcT_cfg['GPU']], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[AcT_cfg['GPU']], True)

    # for pytorch (AP)
    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    gpus = args.gpus

    return gpus
def loop():
    n = 0
    while True:
        yield n
        n += 1

# def load_AP_Detector():
#     # Check Input
#     print(len(args.video))
#     print('-------')
#     if len(args.video):
#         if os.path.isfile(args.video):
#             videofile = args.video
#             mode = 'video'
#             input_source = videofile
#         else:
#             raise IOError('Error: --video must refer to a video file, not directory.')
#     else:
#         mode = 'none'
#         input_source = 'none'

#     # TODO: 1. alphapose FileDetectionLoader to read video file and run yolo
#     #       2. load alphapose pose model
#     if mode == 'video':
#         # get_detector loads Yolo
#         det_loader = DetectionLoader(input_source, get_detector(args), AP_cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
#         # det_worker = det_loader.start()
#     else:
#         det_loader = DetectionLoader(input_source, get_detector(args), AP_cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
#         # det_worker = det_loader.start()
    
#     # Load Pose Model
#     pose_model = builder.build_sppe(AP_cfg.MODEL, preset_cfg=AP_cfg.DATA_PRESET)
#     print('Loading pose model from %s...' % (args.AP_CKPT,))
#     pose_model.load_state_dict(torch.load(args.AP_CKPT, map_location=args.device))
#     # pose_dataset = builder.retrieve_dataset(AP_cfg.DATASET.TRAIN)
#     # load pose model to gpu
#     pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
#     pose_model.eval()

#     return pose_model, det_loader
if __name__ == "__main__":
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

    # set up GPU
    gpus = set_GPU()

    # Built Pose Estimation Model From AlphaPose
    # pose_model, det_loader = load_AP_Detector()
    # Check Input
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            mode = 'video'
            input_source = videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')
    else:
        mode = 'webcam'
        input_source = "http://localhost:3000"

    # TODO: 1. alphapose FileDetectionLoader to read video file and run yolo
    #       2. load alphapose pose model
    if mode == 'video':
        # get_detector loads Yolo
        det_loader = DetectionLoader(input_source, get_detector(args), AP_cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()
    else:
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), AP_cfg, args)
        det_worker = det_loader.start()

    # Load Pose Model
    pose_model = builder.build_sppe(AP_cfg.MODEL, preset_cfg=AP_cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.AP_CKPT,))
    pose_model.load_state_dict(torch.load(args.AP_CKPT, map_location=args.device))
    # pose_dataset = builder.retrieve_dataset(AP_cfg.DATASET.TRAIN)
    # load pose model to gpu
    pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    pose_model.eval()

    # # Load AcT Model
    # AcT_model = build_act()
    # AcT_model.load_weights(args.AcT_CKPT)

    # print("---AcT Model Summary-----")

    # AcT_model.summary()

    # print("-----------------------")

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    writer = DataWriter(AP_cfg, args, save_video=False, queueSize=args.qsize).start()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    if args.save_video and mode != 'image':
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
        if mode == 'video':
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(AP_cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
    else:
        writer = DataWriter(AP_cfg, args, save_video=False, queueSize=queueSize).start()

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    
    kp_seq: List[List[List[float]]] = []

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
            video_path = writer.get_save_path()
        writer.stop()
        det_loader.stop()
        
        with open("result.txt", "r") as f:
            video_result = f.read()
        os.remove("result.txt")
        # Open video file
        video = cv2.VideoCapture(video_path)

        # Define label parameters
        if video_result == '0':
            text = 'Normal'
        else:
            text = 'Theft'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)  # White color in BGR format

        # Get video dimensions
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define label position
        label_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = width - label_size[0] - 10  # 10 pixels from the right edge
        y = label_size[1] + 10  # 10 pixels from the top edge

        # Define video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

        # Loop through video frames
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Add label to frame
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

            # Save frame to output video
            output_video.write(frame)
            
        # Release resources
        video.release()
        output_video.release()


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
