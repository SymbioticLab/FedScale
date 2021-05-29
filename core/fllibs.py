# Standard libs
import os, re, shutil, sys, time, datetime, logging, pickle, json, socket
import random, math, gc, copy
from collections import OrderedDict
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import multiprocessing, threading
import numpy as np
import collections
import numba
import numpy

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

# libs from FLBench
from argParser import args
from utils.utils_data import get_data_transform
from utils.utils_model import test_model
from utils.divide_data import select_dataset, DataPartitioner

if args.task == 'nlp':
    from utils.nlp import mask_tokens, load_and_cache_examples
    from transformers import (
        AdamW,
        AutoConfig,
        AutoTokenizer,
        MobileBertForPreTraining,
    )
elif args.task == 'speech':
    from utils.speech import SPEECH
    from utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
    from utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
    from utils.speech import BackgroundNoiseDataset
elif args.task == 'detection':
    import pickle
    from utils.rcnn.lib.roi_data_layer.roidb import combined_roidb
    from utils.rcnn.lib.datasets.factory import get_imdb
    from utils.rcnn.lib.datasets.pascal_voc import readClass
    from utils.rcnn.lib.roi_data_layer.roibatchLoader import roibatchLoader
    from utils.rcnn.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
    from utils.rcnn.lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
        adjust_learning_rate, save_checkpoint, clip_gradient
    from utils.rcnn.lib.model.faster_rcnn.resnet import resnet
    from utils.rcnn.lib.model.rpn.bbox_transform import clip_boxes
    from utils.rcnn.lib.model.roi_layers import nms
    from utils.rcnn.lib.model.rpn.bbox_transform import bbox_transform_inv
elif args.task == 'voice':
    from torch_baidu_ctc import CTCLoss

from client_manager import clientManager
from utils.yogi import YoGi

# shared functions of aggregator and clients
# initiate for nlp
tokenizer = None
os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port


outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist' : 1010
            }

def init_model():
    global tokenizer

    logging.info("Initializing the model ...")

    if args.task == 'nlp':
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, args.model_name+'-config.json'))
        model = AutoModelWithLMHead.from_config(config)
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name, do_lower_case=True)

        # model_name = 'google/mobilebert-uncased'
        # config = AutoConfig.from_pretrained(model_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # model = MobileBertForPreTraining.from_pretrained(model_name)
        # model = AutoModelWithLMHead.from_config(config)

    elif args.task == 'text_clf':
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
        config.num_labels = outputClass[args.data_set]
        from transformers import AlbertForSequenceClassification

        model = AlbertForSequenceClassification(config)

    elif args.task == 'tag-one-sample':
        # Load LR model for tag prediction
        model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
    elif args.task == 'speech':
        if args.model == 'mobilenet':
            from utils.resnet_speech import mobilenet_v2
            model = mobilenet_v2(num_classes=outputClass[args.data_set])
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
    elif args.task == 'detection':
        #np.random.seed(cfg.RNG_SEED)
        cfg_from_file(args.cfg_file)
        cfg_from_list(['DATA_DIR', args.data_dir])
        model = resnet(readClass(os.path.join(args.data_dir, "class.txt")), 50, pretrained=True, class_agnostic=False,pretrained_model=args.backbone)
        model.create_architecture()
        return model
    else:
        model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    return model


def init_dataset():

    if args.task == "detection":
        if not os.path.exists(args.data_cache):
            imdb_name = "voc_2007_trainval"
            imdbval_name = "voc_2007_test"
            imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name, ['DATA_DIR', args.data_dir], sizes=args.train_size_file)
            train_dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, imdb._image_index_temp,  training=True)
            imdb_, roidb_, ratio_list_, ratio_index_ = combined_roidb(imdbval_name, ['DATA_DIR', args.data_dir], sizes=args.test_size_file, training=False)
            imdb_.competition_mode(on=True)
            test_dataset = roibatchLoader(roidb_, ratio_list_, ratio_index_, 1, imdb_.num_classes, imdb_._image_index_temp, training=False, normalize = False)
            with open(args.data_cache, 'wb') as f:
                pickle.dump(train_dataset, f, -1)
                pickle.dump(test_dataset, f, -1)
        else:
            with open(args.data_cache, 'rb') as f:
                train_dataset = pickle.load(f)
                test_dataset = pickle.load(f)
    else:

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
            from utils.openimage import OpenImage

            train_transform, test_transform = get_data_transform('openImg')
            train_dataset = OpenImage(args.data_dir, dataset='train', transform=train_transform)
            test_dataset = OpenImage(args.data_dir, dataset='test', transform=test_transform)

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
            train_dataset = SPEECH(args.data_dir, dataset= 'train',
                                    transform=transforms.Compose([LoadAudio(),
                                            data_aug_transform,
                                            add_bg_noise,
                                            train_feature_transform]))
            valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
            test_dataset = SPEECH(args.data_dir, dataset='test',
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

    return train_dataset, test_dataset
