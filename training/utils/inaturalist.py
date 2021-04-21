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

class INATURALIST():
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

    def __init__(self, root, train=True, transform=None, target_transform=None, imgview=False, max_class=1e10):
        
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

        # # load class information
        # with open(os.path.join(self.processed_folder, 'classTags'), 'r') as fin:
        #     self.classes = [tag.strip() for tag in fin.readlines()]

        with open(os.path.join(self.processed_folder, 'train.txt'), 'r') as fin:
            self.training_data = [tag.strip() for tag in fin.readlines()]

        with open(os.path.join(self.processed_folder, 'val.txt'), 'r') as fin:
            self.testing_data = [tag.strip() for tag in fin.readlines()]

        self.data, self.targets = self.load_file()

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
        img = Image.open(os.path.join(self.processed_folder, imgName))
        
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

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_file(self):
        stime = time.time()
        rawImg, rawTags = [], []

        # imgFiles = os.scandir(path)
        
        if self.train:
            for imgFile in self.training_data:
                rawImg.append(imgFile)
                rawTags.append(int(imgFile.replace('.jpg', '').split('/')[2]))
        
        else:
            for imgFile in self.testing_data:
                rawImg.append(imgFile)
                rawTags.append(int(imgFile.replace('.jpg', '').split('/')[2]))

        dtime = time.time() - stime
        print(dtime)
        return rawImg, rawTags
