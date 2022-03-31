# Standard libs
import os, re, shutil, sys, time, datetime, logging, pickle, json, socket
import random, math, gc, copy
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import multiprocessing, threading
import numpy as np
from collections import deque
from collections import OrderedDict
import collections
import numba

# PyTorch libs
import torch
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels
from torch.utils.data.sampler import WeightedRandomSampler
from torch_baidu_ctc import CTCLoss

# libs from FLBench
from argParser import args
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
#from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model

from utils.nlp import *
print('c-1')
from utils.speech import SPEECH
# for voice
print('c0')
from utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio
print('c1')
from utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT
print('c2...')
from utils.speech import BackgroundNoiseDataset
print('cc...')

from helper.clientSampler import clientSampler
from utils.yogi import YoGi

