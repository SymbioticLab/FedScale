from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
import time

class OPENIMG():
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

    def __init__(self, root, train=True, transform=None, target_transform=None, imgview=False):
        
        self.train = train  # training set or test set
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data_file = self.training_file
        else:
            self.data_file = self.test_file

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You have to download it')

        # load class information
        with open(os.path.join(self.processed_folder, 'classTags'), 'r') as fin:
            self.classes = [tag.strip() for tag in fin.readlines()]

        self.classMapping = self.class_to_idx
        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data, self.targets = self.load_file(self.path)

        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.path, imgName))
        
        # avoid channel error
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

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
        stime = time.time()
        rawImg, rawTags = [], []

        imgFiles = os.scandir(path)
        #imgFiles = [f for f in os.listdir(path)]# if os.path.isfile(os.path.join(path, f)) and '.jpg' in f]

        for imgFile in imgFiles:
            imgFile = imgFile.name
            classTag = imgFile.replace('.jpg', '').split('__')[1]
            if classTag in self.classMapping:
                rawImg.append(imgFile)
                rawTags.append(self.classMapping[classTag])

        dtime = time.time() - stime
        print(dtime)
        return rawImg, rawTags
