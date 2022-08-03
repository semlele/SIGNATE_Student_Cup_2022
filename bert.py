import collections
import os
import random

import matplotlib.pyplot as plt
#import nlp
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer

# seeds
SEED = 46
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    current_device = torch.cuda.current_device()
    print("Device:", torch.cuda.get_device_name(current_device))
    
    
# config
data_dir = os.path.join(os.environ["HOME"], "Workspace/learning/signate/SIGNATE_Student_Cup_2020/data")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = os.path.join(data_dir, "train.csv")
TEST_FILE = os.path.join(data_dir, "test.csv")
MODELS_DIR = "./models/"
MODEL_NAME = 'bert-base-uncased'
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 128
NUM_CLASSES = 4
EPOCHS = 5
NUM_SPLITS = 5