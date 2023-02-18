# GENERAL LIBRARIES
import os
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

# Use OpenCV to capture webcam video stream
import cv2

    
config = read_yaml('utils/config.yaml')
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

print('Configuration: ')
print(config)

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

transformer = TransformerEncoder(d_model, n_heads, d_ff, dropout, activation, n_layers)
model = build_act(transformer)
model.load_weights('AcT_pretrained_weights/AcT_micro_1_0.h5')
# model = tf.keras.models.load_model('AcT_pretrained_weights/AcT_base_1_0.h5')

print("---Model Summary-----")

model.summary()
# print(model.trainable_variables) 

print("-----------------------")
# model = trainer.model
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
