from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import csv


class OpenImage():
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

    def __init__(self, root, dataset='train', transform=None, target_transform=None,
                 imgview=False, client_mapping_file=None, num_clt=1e10, noniid=0):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_file = dataset  # 'train', 'test', 'validation'
        self.client_mapping_file = client_mapping_file
        self.num_clt = num_clt
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You have to download it')

        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data_to_clientID = {}
        self.data, self.targets = self.load_file(self.path)
        self.imgview = imgview
        self.noniid = noniid

    def __getitem__(self, index):
        """
        Args:
            id_clt (int): Index, client ID

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # index , clientID = id_clt
        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.path, imgName))
        # avoid channel error
        if img.mode != 'RGB':
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

        print("Checking data path:", os.path.join(self.processed_folder, self.data_file))
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path):
        datas, labels = [], []
        unique_clientIds = set()
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    unique_clientIds.add(row[0])
                    self.data_to_clientID[len(datas)] = row[0]
                    if len(unique_clientIds) > self.num_clt:
                        break
                    datas.append(row[1])
                    labels.append(int(row[-1]))

                line_count += 1
        return datas, labels

    def load_file(self, path):
        # load meta file to get labels
        # datas, labels = self.load_meta_data(os.path.join(self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))
        if self.client_mapping_file is not None:
            datas, labels = self.load_meta_data(self.client_mapping_file)
        else:
            datas, labels = self.load_meta_data(
                os.path.join(self.processed_folder, 'client_data_mapping', self.data_file + '.csv'))

        return datas, labels


