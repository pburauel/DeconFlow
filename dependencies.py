import argparse 



import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as functional
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
from torchvision import datasets, transforms

from torch.distributions import Normal
from torch.nn.functional import softplus
import torch.distributions as D
import torch.nn as nn

import importlib
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import pickle
import time
import datetime

import itertools
import gc
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

from statsmodels.distributions.empirical_distribution import ECDF


# for application
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import glob
from scipy.stats import shapiro


import statsmodels.api as sm


from tqdm import tqdm


from ray import tune
from ray import train as ray_train
from ray.train import RunConfig, ScalingConfig
from ray.tune import Tuner, TuneConfig
from ray.train import Checkpoint
from ray.tune.analysis import ExperimentAnalysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


