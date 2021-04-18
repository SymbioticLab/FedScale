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

if args.task == 'nlp':
    from utils.nlp import *
elif args.task == 'speech':
    from utils.speech import SPEECH
    from utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
    from utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
    from utils.speech import BackgroundNoiseDataset

from helper.clientSampler import clientSampler
from utils.yogi import YoGi

# shared functions of aggregator and clients
# initiate for nlp
tokenizer = None

if args.task == 'nlp' or args.task == 'text_clf':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

modelDir = os.path.join(args.log_path, args.model)
modelPath = modelDir+'/'+str(args.model)+'.pth.tar' if args.model_path is None else args.model_path

def init_dataset():
    global tokenizer

    outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                }

    logging.info("====Initialize the model")

    if args.task == 'nlp':
        # we should train from scratch
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
        model = AutoModelWithLMHead.from_config(config)
    elif args.task == 'text_clf':
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
        config.num_labels = outputClass[args.data_set]
        # config.output_attentions = False
        # config.output_hidden_states = False
        from transformers import AlbertForSequenceClassification

        model = AlbertForSequenceClassification(config)

    elif args.task == 'tag-one-sample':
        # Load LR model for tag prediction
        model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
    elif args.task == 'speech':
        if args.model == 'mobilenet':
            from utils.resnet_speech import mobilenet_v2
            model = mobilenet_v2(num_classes=outputClass[args.data_set], inchannels=1)
        elif args.model == "resnet18":
            from utils.resnet_speech import resnet18
            model = resnet18(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet34":
            from utils.resnet_speech import resnet34
            model = resnet34(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet50":
            from utils.resnet_speech import resnet50
            model = resnet50(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet101":
            from utils.resnet_speech import resnet101
            model = resnet101(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet152":
            from utils.resnet_speech import resnet152
            model = resnet152(num_classes=outputClass[args.data_set], in_channels=1)
        else:
            # Should not reach here
            logging.info('Model must be resnet or mobilenet')
            sys.exit(-1)

    elif args.task == 'voice':
        from utils.voice_model import DeepSpeech, supported_rnns

        # Initialise new model training
        with open(args.labels_path) as label_file:
            labels = json.load(label_file)

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[args.rnn_type.lower()],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)
    else:
        model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    if args.load_model:
        try:
            with open(modelPath, 'rb') as fin:
                model = pickle.load(fin)

            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            sys.exit(-1)

    train_dataset, test_dataset = [], []

    # Load data if the machine acts as clients
    if args.this_rank != 0:

        if args.data_set == 'Mnist':
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                          transform=test_transform)

        elif args.data_set == 'cifar10':
            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                            transform=test_transform)

        elif args.data_set == "imagenet":
            train_transform, test_transform = get_data_transform('imagenet')
            train_dataset = datasets.ImageNet(args.data_dir, split='train', download=False, transform=train_transform)
            test_dataset = datasets.ImageNet(args.data_dir, split='val', download=False, transform=test_transform)

        elif args.data_set == 'emnist':
            test_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
            train_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

        elif args.data_set == 'femnist':
            from utils.femnist import FEMNIST

            train_transform, test_transform = get_data_transform('mnist')
            train_dataset = FEMNIST(args.data_dir, train=True, transform=train_transform)
            test_dataset = FEMNIST(args.data_dir, train=False, transform=test_transform)

        elif args.data_set == 'openImg':
            from utils.openImg import OPENIMG

            transformer_ns = 'openImg' if args.model != 'inception_v3' else 'openImgInception'
            train_transform, test_transform = get_data_transform(transformer_ns)
            train_dataset = OPENIMG(args.data_dir, train=True, transform=train_transform)
            test_dataset = OPENIMG(args.data_dir, train=False, transform=test_transform)

        elif args.data_set == 'blog':
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
            test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        elif args.data_set == 'stackoverflow':
            from utils.stackoverflow import stackoverflow

            train_dataset = stackoverflow(args.data_dir, train=True)
            test_dataset = stackoverflow(args.data_dir, train=False)

        elif args.data_set == 'yelp':
            import utils.dataloaders as fl_loader

            train_dataset = fl_loader.TextSentimentDataset(args.data_dir, train=True, tokenizer=tokenizer, max_len=args.clf_block_size)
            test_dataset = fl_loader.TextSentimentDataset(args.data_dir, train=False, tokenizer=tokenizer, max_len=args.clf_block_size)

        elif args.data_set == 'google_speech':
            bkg = '_background_noise_'
            data_aug_transform = transforms.Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
            bg_dataset = BackgroundNoiseDataset(os.path.join(args.data_dir, bkg), data_aug_transform)
            add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
            train_dataset = SPEECH(args.data_dir, train= True,
                                    transform=transforms.Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))
            valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
            test_dataset = SPEECH(args.data_dir, train=False,
                                    transform=transforms.Compose([LoadAudio(),
                                             FixAudioLength(),
                                             valid_feature_transform]))
        elif args.data_set == 'common_voice':
            from utils.voice_data_loader import SpectrogramDataset
            train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                           manifest_filepath=args.train_manifest,
                                           labels=model.labels,
                                           normalize=True,
                                           speed_volume_perturb=args.speed_volume_perturb,
                                           spec_augment=args.spec_augment,
                                           data_mapfile=args.data_mapfile)
            test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                          manifest_filepath=args.test_manifest,
                                          labels=model.labels,
                                          normalize=True,
                                          speed_volume_perturb=False,
                                          spec_augment=False)
        else:
            print('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog', 'stackoverflow', 'speech', 'yelp']))
            sys.exit(-1)

    return model, train_dataset, test_dataset
