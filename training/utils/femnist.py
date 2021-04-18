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
import pickle

class FEMNIST():
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


        # load data and targets
        self.raw_data, self.data, self.targets = self.load_file(self.root)
        #self.mapping = {idx:file for idx, file in enumerate(raw_data)}

        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_path, target = self.raw_data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.root, img_path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        print(img.shape, target)
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


    def load_file(self, path):
        if self.train:
            with open(os.path.join(path, 'train_img_to_path'), 'rb') as f:
                rawImg = pickle.load(f)
                rawPath = pickle.load(f)

            with open(os.path.join(path, 'train_img_to_target'), 'rb') as f:
                rawTags = pickle.load(f)
        else:
            with open(os.path.join(path, 'test_img_to_path'), 'rb') as f:
                rawImg = pickle.load(f)
                rawPath = pickle.load(f)

            with open(os.path.join(path, 'test_img_to_target'), 'rb') as f:
                rawTags = pickle.load(f)

        return rawImg, rawPath, rawTags

