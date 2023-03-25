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

# LOAD CONFIG 
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--config', default='utils/config.yaml', type=str, help='Config path', required=False)    
parser.add_argument('--benchmark','-b', action='store_true', help='Run a benchmark') 
parser.add_argument('--search','-s', action='store_true', help='Run a random search')
    
args = parser.parse_args()
config = read_yaml(args.config)

for entry in ['MODEL_DIR','RESULTS_DIR','LOG_DIR']:
    if not os.path.exists(config[entry]):
        os.mkdir(config[entry])

now = datetime.now()
logger = Logger(config['LOG_DIR']+now.strftime("%y%m%d%H%M%S")+'.txt')


# SET GPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[config['GPU']], 'GPU')
tf.config.experimental.set_memory_growth(gpus[config['GPU']], True)


# SET TRAINER
trainer = Trainer(config, logger)

if args.benchmark:
    # RUN BENCHMARK
    trainer.do_benchmark()

elif args.search:
    # RUN RANDOM SEARCH
    trainer.do_random_search()
    
else:
    print('Nothing to do! Specify one of the following arguments:')
    print('\t --benchmark [-b]: run a benchmark')
    print('\t --search [-s]: run a random search')

# import sys
# import tensorflow.keras
# import pandas as pd
# import sklearn as sk
# import scipy as sp
# import tensorflow as tf
# import platform
# print(f"Python Platform: {platform.platform()}")
# print(f"Tensor Flow Version: {tf.__version__}")
# print(f"Keras Version: {tensorflow.keras.__version__}")
# print()
# print(f"Python {sys.version}")
# print(f"Pandas {pd.__version__}")
# print(f"Scikit-Learn {sk.__version__}")
# print(f"SciPy {sp.__version__}")
# gpu = len(tf.config.list_physical_devices('GPU'))>0
# print("GPU is", "available" if gpu else "NOT AVAILABLE")