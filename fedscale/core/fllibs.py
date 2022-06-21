# Standard libs
import os, re, shutil, sys, time, datetime, logging, pickle, json, socket
import random, math, gc, copy
from collections import OrderedDict
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import multiprocessing, threading
import numpy as np
import collections
import numpy

# libs from fedscale
from fedscale.core.arg_parser import args
from fedscale.dataloaders.utils_data import get_data_transform
from fedscale.core.utils.model_test_module import test_model
from fedscale.dataloaders.divide_data import select_dataset, DataPartitioner
from fedscale.core.client_manager import clientManager
from fedscale.core.utils.yogi import YoGi
from fedscale.core.optimizer import ServerOptimizer

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

tokenizer = None
if args.task == 'nlp' or args.task == 'text_clf':
    from fedscale.dataloaders.nlp import mask_tokens, load_and_cache_examples
    from transformers import (
        AdamW,
        AutoConfig,
        AlbertTokenizer,
        AutoTokenizer,
        MobileBertForPreTraining,
        AutoModelWithLMHead
    )
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
elif args.task == 'speech':
    import numba
    from fedscale.dataloaders.speech import SPEECH
    from fedscale.dataloaders.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
    from fedscale.dataloaders.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
    from fedscale.dataloaders.speech import BackgroundNoiseDataset
elif args.task == 'detection':
    import pickle
    from fedscale.dataloaders.rcnn.lib.roi_data_layer.roidb import combined_roidb
    from fedscale.dataloaders.rcnn.lib.datasets.factory import get_imdb
    from fedscale.dataloaders.rcnn.lib.datasets.pascal_voc import readClass
    from fedscale.dataloaders.rcnn.lib.roi_data_layer.roibatchLoader import roibatchLoader
    from fedscale.dataloaders.rcnn.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
    from fedscale.dataloaders.rcnn.lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
        adjust_learning_rate, save_checkpoint, clip_gradient
    from fedscale.dataloaders.rcnn.lib.model.faster_rcnn.resnet import resnet
    from fedscale.dataloaders.rcnn.lib.model.rpn.bbox_transform import clip_boxes
    from fedscale.dataloaders.rcnn.lib.model.roi_layers import nms
    from fedscale.dataloaders.rcnn.lib.model.rpn.bbox_transform import bbox_transform_inv
elif args.task == 'voice':
    from torch_baidu_ctc import CTCLoss
elif args.task == 'rl':
    import gym
    from fedscale.dataloaders.dqn import *

# shared functions of aggregator and clients
# initiate for nlp

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port


outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,'amazon':5,
                'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist' : 1010
            }

def init_model():
    global tokenizer

    logging.info("Initializing the model ...")

    if args.task == 'nlp':
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, args.model+'-config.json'))
        model = AutoModelWithLMHead.from_config(config)
        tokenizer = AlbertTokenizer.from_pretrained(args.model, do_lower_case=True)

        # model_name = 'google/mobilebert-uncased'
        # config = AutoConfig.from_pretrained(model_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # model = MobileBertForPreTraining.from_pretrained(model_name)
        # model = AutoModelWithLMHead.from_config(config)

    elif args.task == 'text_clf':

        if args.model == 'albert':
            from transformers import AlbertForSequenceClassification
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(os.path.join(args.log_path, 'albert-small-config.json'))
            config.num_labels = outputClass[args.data_set]
            model = AlbertForSequenceClassification(config)
        elif args.model == 'lr':
            from fedscale.core.utils.models import  LogisticRegression
            model = LogisticRegression(300, outputClass[args.data_set])


    elif args.task == 'tag-one-sample':
        # Load LR model for tag prediction
        model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
    elif args.task == 'speech':
        if args.model == 'mobilenet':
            from fedscale.core.utils.resnet_speech import mobilenet_v2
            model = mobilenet_v2(num_classes=outputClass[args.data_set])
        elif args.model == "resnet18":
            from fedscale.core.utils.resnet_speech import resnet18
            model = resnet18(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet34":
            from fedscale.core.utils.resnet_speech import resnet34
            model = resnet34(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet50":
            from fedscale.core.utils.resnet_speech import resnet50
            model = resnet50(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet101":
            from fedscale.core.utils.resnet_speech import resnet101
            model = resnet101(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet152":
            from fedscale.core.utils.resnet_speech import resnet152
            model = resnet152(num_classes=outputClass[args.data_set], in_channels=1)
        else:
            # Should not reach here
            logging.info('Model must be resnet or mobilenet')
            sys.exit(-1)

    elif args.task == 'voice':
        from fedscale.core.utils.voice_model import DeepSpeech, supported_rnns

        # Initialise new model training
        with open(os.path.join(args.data_dir, "labels.json")) as label_file:
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
        cfg_from_file(args.cfg_file)
        cfg_from_list(['DATA_DIR', args.data_dir])
        model = resnet(readClass(os.path.join(args.data_dir, "class.txt")), 50, pretrained=True, class_agnostic=False,pretrained_model=args.backbone)
        model.create_architecture()
        return model
    elif args.task == 'rl':
        model = DQN(args).target_net
    else:
        if args.model == "lr":
            from fedscale.core.utils.models import LogisticRegression
            model = LogisticRegression(args.input_dim, outputClass[args.data_set])
        elif args.model == 'svm':
            from fedscale.core.utils.models import LinearSVM
            model = LinearSVM(args.input_dim, outputClass[args.data_set])
        else:
            model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    return model


def init_dataset():

    if args.task == "detection":
        if not os.path.exists(args.data_cache):
            imdb_name = "voc_2007_trainval"
            imdbval_name = "voc_2007_test"
            imdb, roidb, ratio_list, ratio_index = combined_roidb(
                imdb_name, ['DATA_DIR', args.data_dir], sizes=args.train_size_file)
            train_dataset = roibatchLoader(
                roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, imdb._image_index_temp,  training=True)
            imdb_, roidb_, ratio_list_, ratio_index_ = combined_roidb(
                imdbval_name, ['DATA_DIR', args.data_dir], sizes=args.test_size_file, training=False)
            imdb_.competition_mode(on=True)
            test_dataset = roibatchLoader(roidb_, ratio_list_, ratio_index_, 1, 
                imdb_.num_classes, imdb_._image_index_temp, training=False, normalize = False)
            with open(args.data_cache, 'wb') as f:
                pickle.dump(train_dataset, f, -1)
                pickle.dump(test_dataset, f, -1)
        else:
            with open(args.data_cache, 'rb') as f:
                train_dataset = pickle.load(f)
                test_dataset = pickle.load(f)
    elif args.task == "rl":
        train_dataset = test_dataset = RLData(args)
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
            from fedscale.dataloaders.femnist import FEMNIST

            train_transform, test_transform = get_data_transform('mnist')
            train_dataset = FEMNIST(args.data_dir, dataset='train', transform=train_transform)
            test_dataset = FEMNIST(args.data_dir, dataset='test', transform=test_transform)

        elif args.data_set == 'openImg':
            from fedscale.dataloaders.openimage import OpenImage

            train_transform, test_transform = get_data_transform('openImg')
            train_dataset = OpenImage(args.data_dir, dataset='train', transform=train_transform)
            test_dataset = OpenImage(args.data_dir, dataset='test', transform=test_transform)

        elif args.data_set == 'blog':
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
            test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        elif args.data_set == 'stackoverflow':
            from fedscale.dataloaders.stackoverflow import stackoverflow

            train_dataset = stackoverflow(args.data_dir, train=True)
            test_dataset = stackoverflow(args.data_dir, train=False)

        elif args.data_set == 'amazon':
            if args.model == 'albert':
                import fedscale.dataloaders.amazon as fl_loader
                train_dataset = fl_loader.AmazonReview_loader(args.data_dir, train=True, tokenizer=tokenizer, max_len=args.clf_block_size  )
                test_dataset = fl_loader.AmazonReview_loader(args.data_dir, train=False, tokenizer=tokenizer, max_len=args.clf_block_size )

            elif args.model == 'lr':
                import fedscale.dataloaders.word2vec as fl_loader
                train_dataset = fl_loader.AmazonReview_word2vec(args.data_dir, args.embedding_file, train=True)
                test_dataset = fl_loader.AmazonReview_word2vec( args.data_dir, args.embedding_file, train=False)

        elif args.data_set == 'yelp':
            import fedscale.dataloaders.yelp as fl_loader

            train_dataset = fl_loader.TextSentimentDataset(args.data_dir, train=True, tokenizer=tokenizer, max_len=args.clf_block_size)
            test_dataset = fl_loader.TextSentimentDataset(args.data_dir, train=False, tokenizer=tokenizer, max_len=args.clf_block_size)

        elif args.data_set == 'google_speech':
            bkg = '_background_noise_'
            data_aug_transform = transforms.Compose(
                [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
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
            from fedscale.dataloaders.voice_data_loader import SpectrogramDataset
            train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                        data_dir=args.data_dir,
                                        labels=model.labels,
                                        train=True,
                                        normalize=True,
                                        speed_volume_perturb=args.speed_volume_perturb,
                                        spec_augment=args.spec_augment,
                                        data_mapfile=args.data_mapfile)
            test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                        data_dir=args.data_dir,
                                        labels=model.labels,
                                        train=False,
                                        normalize=True,
                                        speed_volume_perturb=False,
                                        spec_augment=False)
        else:
            logging.info('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog', 'stackoverflow', 'speech', 'yelp']))
            sys.exit(-1)

    return train_dataset, test_dataset
