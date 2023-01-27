import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader

import sys, os
sys.path.append('..')

from generator import *
from discriminator import *
from trainer import *
from hiragana_dataset import make_input3

path = '../maru.jpg'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

sns.heatmap(image)
plt.show()