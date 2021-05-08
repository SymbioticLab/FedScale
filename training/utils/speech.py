from __future__ import print_function
import warnings
import os
import numpy as np
import numba
import librosa
import csv

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

    def __init__(self, root, dataset='train', transform=None, target_transform=None, classes=CLASSES):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.classMapping = {classes[i]: i for i in range(len(classes))}
        self.data_file = dataset # 'train', 'test', 'validation'


        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data, self.targets = self.load_file(self.path)

        self.data_dir =  os.path.join(self.root, self.data_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.data[index], int(self.targets[index])
        data = {'path': os.path.join(self.data_dir, path), 'target': target}

        if self.transform is not None:
            data = self.transform(data)

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

    def load_meta_data(self, path):
        data_to_label = {}
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    data_to_label[row[1]] = self.classMapping[row[-2]]
                line_count += 1

        return data_to_label

    def load_file(self, path):
        rawData, rawTags = [], []
        # load meta file to get labels
        classMapping = self.load_meta_data(os.path.join(self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))

        for imgFile in list(classMapping.keys()):
            rawData.append(imgFile)
            rawTags.append(classMapping[imgFile])

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

