# -*- coding: utf-8 -*-
from random import Random
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import random, csv
from argParser import args


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        last_client_id = -1
        client_stat = {}
        category_dict = {}
        numOfLabels = self.getNumOfLabels()

        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = -1

            for row in csv_reader:
                if line_count == -1:
                    logging.info(f'Column names are {", ".join(row)}')
                else:
                    client_id, sample_name, sample_category = int(row[0]), row[1], row[2]

                    if client_id != last_client_id:
                        self.partitions.append([])
                        client_stat[client_id] = [0] * num_of_labels

                    if sample_category not in category_dict:
                        category_dict[sample_category] = len(category_dict)

                    self.partitions[-1].append(line_count)
                    client_stat[client_id][category_dict[sample_category]] += 1

                line_count += 1


    def partition_data_helper(self, num_clients, data_map_file=None):

        # read mapping file to partition trace
        if data_map_file is not None:
            self.trace_partition(data_map_file)
        else:
            self.uniform_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        # random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition, istest):
        resultIndex = self.partitions[partition]

        exeuteLength = -1 if not istest else int(len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)


    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None):
    """Load data given client Id"""
    partition = partition.use(rank - 1, isTest)
    timeOut = 0 if isTest else 60
    dropLast = False if isTest else True

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=args.num_loaders, drop_last=dropLast, timeout=timeOut, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=args.num_loaders, drop_last=dropLast, timeout=timeOut)



