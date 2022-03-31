from __future__ import print_function
import warnings
import os
import os.path
import torch
import time
import pickle
import h5py as h5
import torch.nn.functional as F
#import logging

class stackoverflow():
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
    MAX_SEQ_LEN = 20000

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

    def __init__(self, root, train=True):
        self.train = train  # training set or test set
        self.root = root

        self.train_file = 'stackoverflow_train.h5'
        self.test_file = 'stackoverflow_test.h5'
        self.train = train

        self.vocab_tokens_size = 10000
        self.vocab_tags_size = 500

        # load data and targets
        self.raw_data, self.raw_targets, self.dict = self.load_file(self.root, self.train)
        # temp_raw_data, temp_raw_targets, temp_dict = self.load_file(self.root, self.train)

        # self.raw_data, self.raw_targets, self.dict = [], [], {}

        # count = 0
        # # we only take the 1-tag samples
        # for idx in range(len(temp_raw_data)):
        #     if len(temp_raw_targets[idx]) == 1:
        #         self.dict[count] = temp_dict[idx]
        #         self.raw_data.append(temp_raw_data[idx])
        #         self.raw_targets.append(temp_raw_targets[idx])

        #         count += 1

        # # dump the file
        # file_name = self.train_file if self.train else self.test_file

        # # check whether we have generated the cache file before
        # cache_path = os.path.join(self.root, file_name + '_cache_1')
        # with open(cache_path, 'wb') as fout:
        #     pickle.dump(self.raw_data, fout)
        #     pickle.dump(self.raw_targets, fout)
        #     pickle.dump(self.dict, fout)

        if not self.train:
            self.raw_data = self.raw_data[:100000]
            self.raw_targets = self.raw_targets[:100000]
        else:
            self.raw_data = self.raw_data[:10000000]
            self.raw_targets = self.raw_targets[:10000000]

        # we can't enumerate the raw data, thus generating artificial data to cheat the divide_data_loader
        self.data = [-1, len(self.dict)]
        self.targets = [-1, len(self.dict)]

    def __getitem__(self, index):
        """
        Args:xx
            index (int): Index

        Returns:
            tuple: (text, tags)
        """

        # Lookup tensor

        tokens = self.raw_data[index]
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = F.one_hot(tokens, self.vocab_tokens_size).float()
        tokens = tokens.mean(0)

        tags = self.raw_targets[index][0]
        # tags = torch.tensor(tags, dtype=torch.long)
        # tags = F.one_hot(tags, self.vocab_tags_size).float()
        # tags = tags.sum(0)

        return tokens, tags

    def __mapping_dict__(self):
        return self.dict

    def __len__(self):
        return len(self.raw_data)

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

    def create_tag_vocab(self, vocab_size, path):
        """Creates vocab from `vocab_size` most common tags in Stackoverflow."""
        tags_file = "vocab_tags.txt"
        with open(os.path.join(path,tags_file), 'rb') as f:
            tags = pickle.load(f)
        return tags[:vocab_size]


    def create_token_vocab(self, vocab_size, path):
        """Creates vocab from `vocab_size` most common words in Stackoverflow."""
        tokens_file = "vocab_tokens.txt"
        with open(os.path.join(path, tokens_file), 'rb') as f:
            tokens = pickle.load(f)
        return tokens[:vocab_size]

    def load_file(self, path, is_train):
        file_name = self.train_file if self.train else self.test_file

        # check whether we have generated the cache file before
        cache_path = os.path.join(path, file_name + '_cache_1')

        text, target_tags = [], []
        mapping_dict = {}

        if os.path.exists(cache_path):
            print("====Load {} from cache".format(file_name))
            # dump the cache
            with open(cache_path, 'rb') as fin:
                text = pickle.load(fin)
                target_tags = pickle.load(fin)
                mapping_dict = pickle.load(fin)
        else:
            print("====Load {} from scratch".format(file_name))
            # Mapping from sample id to target tag
            # First, get the token and tag dict
            vocab_tokens = self.create_token_vocab(self.vocab_tokens_size, path)
            vocab_tags = self.create_tag_vocab(self.vocab_tags_size, path)

            vocab_tokens_dict = {k: v for v, k in enumerate(vocab_tokens)}
            vocab_tags_dict = {k: v for v, k in enumerate(vocab_tags)}

            # Load the traning data
            if self.train:
                train_file = h5.File(os.path.join(path, self.train_file), "r")
            else:
                train_file = h5.File(os.path.join(path, self.test_file), "r")
            print(self.train)

            count = 0
            clientCount = 0
            client_list = list(train_file['examples'])
            start_time = time.time()

            for clientId, client in enumerate(client_list):
                tags_list = list(train_file['examples'][client]['tags'])
                tokens_list = list(train_file['examples'][client]['tokens'])
                title_list = list(train_file['examples'][client]['title'])

                for tags, tokens, title in zip(tags_list, tokens_list, title_list):
                    tags_list = [vocab_tags_dict[s] for s in tags.decode("utf-8").split('|') if s in vocab_tags_dict]
                    if not tags_list:
                        continue

                    tokens_list = [vocab_tokens_dict[s] for s in (tokens.decode("utf-8").split()+title.decode("utf-8").split()) if s in vocab_tokens_dict]
                    if not tokens_list:
                        continue

                    mapping_dict[count] = clientId
                    text.append(tokens_list)
                    target_tags.append(tags_list)

                    count += 1

                clientCount += 1

                num_of_remains = len(client_list) - clientId
                #print("====In loading data, remains {} clients, may take {} sec".format(num_of_remains, (time.time() - start_time)/clientCount * num_of_remains))
                # logging.info("====In loading  data, remains {} clients".format(num_of_remains)

                if clientId % 5000 == 0:
                    # dump the cache
                    with open(cache_path, 'wb') as fout:
                        pickle.dump(text, fout)
                        pickle.dump(target_tags, fout)
                        pickle.dump(mapping_dict, fout)

                    #print("====Dump for {} clients".format(clientId))

            # dump the cache
            with open(cache_path, 'wb') as fout:
                pickle.dump(text, fout)
                pickle.dump(target_tags, fout)
                pickle.dump(mapping_dict, fout)

        return text, target_tags, mapping_dict

# sp = stackoverflow('./', )
