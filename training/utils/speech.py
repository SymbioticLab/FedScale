from __future__ import print_function
import warnings
import os
import os.path
import numpy as np
import torch
import codecs
import string
import time
import numba
import librosa
from torch.utils.data import Dataset

CLASSES = ['up', 'two', 'sheila', 'zero', 'yes', 'five', 'one', 'happy', 'marvin', 'no', 'go', 'seven', 'eight', 'tree', 'stop', 'down', 'forward', 'learn', 'house', 'three', 'six', 'backward', 'dog', 'cat', 'wow', 'left', 'off', 'on', 'four', 'visual', 'nine', 'bird', 'right', 'follow', 'bed']


class SPEECH():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    training_file = 'train'
    test_file = 'test'
    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, classes=CLASSES):



        self.train = train  # training set or test set
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.classMapping = {classes[i]: i for i in range(len(classes))}

        if self.train:
            self.data_file = self.training_file
        else:
            self.data_file = self.test_file

        #if not self._check_exists():
        #    raise RuntimeError('Dataset not found.' +
        #                       ' You have to download it')

        # # load class information
        # with open(os.path.join(self.processed_folder, 'classTags'), 'r') as fin:
        #     self.classes = [tag.strip() for tag in fin.readlines()]

        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data, self.targets = self.load_file(self.path)
        #print(self.data[:10])
        #print(self.targets[:10])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.data[index], int(self.targets[index])
        data_dir = os.path.join(self.root, 'train') if self.train else os.path.join(self.root, 'test')
        data = {'path': os.path.join(data_dir, path), 'target': target}


        if self.transform is not None:
            data = self.transform(data)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        #logging.info('====== data input shape is =====')
        #logging.info(data['input'].shape)
        return data['input'], data['target']

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_file(self, path):
        rawData, rawTags = [], []

        audioFiles = os.scandir(path)

        clientMap = {}
        for idx, audio in enumerate(audioFiles):
            audio = audio.name
            classTag = audio.split('_')[0]
            if classTag in self.classMapping:
                rawData.append(audio)
                rawTags.append(self.classMapping[classTag])
        return rawData, rawTags


class BackgroundNoiseDataset():
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data

