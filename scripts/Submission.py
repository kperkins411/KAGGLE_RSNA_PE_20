
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import Dataset,DataLoader
import cv2
import albumentations as albu
import functools
import glob
from albumentations.pytorch.transforms import ToTensorV2
# from tqdm.auto import tqdm
import gc

#images to eval
PATH = '../input/rsna-str-pulmonary-embolism-detection'
PATH_TEST=PATH+'/test/'


