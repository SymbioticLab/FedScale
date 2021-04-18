# -*- coding: utf-8 -*-
from random import Random
#from core.dataloader import DataLoader
from torch.utils.data import DataLoader
import numpy as np
from math import *
import logging
from scipy import stats
import numpy as np
from pyemd import emd
from collections import OrderedDict
import time
import pickle, random
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

    # len(sizes) is the number of workers
    # sequential 1-> random 2->zipf 3-> identical
    def __init__(self, data, numOfClass=0, seed=10, splitConfFile=None, isTest=False, dataMapFile=None):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)
        self.data = data
        self.labels = self.data.targets
        self.is_trace = False
        self.dataMapFile = None
        self.args = args
        self.isTest = isTest

        np.random.seed(seed)

        stime = time.time()
        #logging.info("====Start to initiate DataPartitioner")

        self.targets = OrderedDict()
        self.indexToLabel = {}
        self.totalSamples = 0
        self.data_len = len(self.data)
        self.task = args.task
        self.skip_partition = True if self.data.targets[0] is -1 or args.skip_partition is True else False

        if self.skip_partition:
            logging.info("====Warning: skip_partition is True")

        if self.skip_partition:
            pass

        elif splitConfFile is None:
            # categarize the samples
            for index, label in enumerate(self.labels):
                if label not in self.targets:
                    self.targets[label] = []
                self.targets[label].append(index)
                self.indexToLabel[index] = label

            self.totalSamples += len(self.data)
        else:
            # each row denotes the number of samples in this class
            with open(splitConfFile, 'r') as fin:
                labelSamples = [int(x.strip()) for x in fin.readlines()]

            # categarize the samples
            baseIndex = 0
            for label, _samples in enumerate(labelSamples):
                for k in range(_samples):
                    self.indexToLabel[baseIndex + k] = label
                self.targets[label] = [baseIndex + k for k in range(_samples)]
                self.totalSamples += _samples
                baseIndex += _samples

        if dataMapFile is not None:
            self.dataMapFile = dataMapFile
            self.is_trace = True

        self.numOfLabels = max(len(self.targets.keys()), numOfClass)
        self.workerDistance = []
        self.classPerWorker = None

        logging.info("====Initiating DataPartitioner takes {} s\n".format(time.time() - stime))

    def getTargets(self):
        tempTarget = self.targets.copy()

        for key in tempTarget:
            self.rng.shuffle(tempTarget[key])

        return tempTarget

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    # Calculates JSD between pairs of distribution
    def js_distance(self, x, y):
        m = (x + y)/2
        js = 0.5 * stats.entropy(x, m) + 0.5 * stats.entropy(y, m)
        return js

    # Caculates Jensen-Shannon Divergence for each worker
    def get_JSD(self, dataDistr, tempClassPerWorker, sizes):
        for worker in range(len(sizes)):
            # tempDataSize = sum(tempClassPerWorker[worker])
            # if tempDataSize == 0:
            #     continue
            # tempDistr =np.array([c / float(tempDataSize) for c in tempClassPerWorker[worker]])
            self.workerDistance.append(0)#self.js_distance(dataDistr, tempDistr))

    # Generates a distance matrix for EMD
    def generate_distance_matrix(self, size):
        return np.logical_xor(1, np.identity(size)) * 1.0

    # Caculates Earth Mover's Distance for each worker
    def get_EMD(self, dataDistr, tempClassPerWorker, sizes):
        dist_matrix = self.generate_distance_matrix_v2(len(dataDistr))
        for worker in range(len(sizes)):
            tempDataSize = sum(tempClassPerWorker[worker])
            if tempDataSize == 0:
                continue
            tempDistr =np.array([c / float(tempDataSize) for c in tempClassPerWorker[worker]])
            self.workerDistance.append(emd(dataDistr, tempDistr, dist_matrix))

    def loadFilterInfo(self):
        # load data-to-client mapping
        indicesToRm = []

        try:
            dataToClient = OrderedDict()

            with open(self.args.data_mapfile, 'rb') as db:
                dataToClient = pickle.load(db)

            clientNumSamples = {}
            sampleIdToClient = []

            # data share the same index with labels
            for index, _sample in enumerate(self.data.data):
                sample = _sample.split('__')[0]
                clientId = dataToClient[sample]

                if clientId not in clientNumSamples:
                    clientNumSamples[clientId] = 0

                clientNumSamples[clientId] += 1
                sampleIdToClient.append(clientId)

            for index, clientId in enumerate(sampleIdToClient):
                if clientNumSamples[clientId] < self.args.filter_less:
                    indicesToRm.append(index)

        except Exception as e:
            logging.info("====Failed to generate indicesToRm, because of {}".format(e))
            #pass

        return indicesToRm

    def loadFilterInfoNLP(self):
        indices = []
        base = 0

        for idx, sample in enumerate(self.data.slice_index):
            if sample < args.filter_less:
                indices = indices + [base+i for i in range(sample)]
            base += sample

        return indices

    def loadFilterInfoBase(self):
        indices = []

        try:
            for client in self.data.client_mapping:
                if len(self.data.client_mapping[client]) < args.filter_less or len(self.data.client_mapping[client]) > args.filter_more:
                    indices += self.data.client_mapping[client]

                    # remove the metadata
                    for idx in self.data.client_mapping[client]:
                        self.data[idx] = None

        except Exception as e:
            pass

        return indices

    def partitionTraceCV(self, dataToClient):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = self.numOfLabels

        # data share the same index with labels
        for index, sample in enumerate(self.data.data):
            sample = sample.split('__')[0]
            clientId = dataToClient[sample]
            labelId = self.labels[index]

            if clientId not in clientToData:
                clientToData[clientId] = []
                clientNumSamples[clientId] = [0] * numOfLabels

            clientToData[clientId].append(index)
            clientNumSamples[clientId][labelId] += 1

        numOfClients = len(clientToData.keys())
        self.classPerWorker = np.zeros([numOfClients, numOfLabels])

        for clientId in range(numOfClients):
            self.classPerWorker[clientId] = clientNumSamples[clientId]
            self.rng.shuffle(clientToData[clientId])
            self.partitions.append(clientToData[clientId])

        overallNumSamples = np.asarray(self.classPerWorker.sum(axis=0)).reshape(-1)
        totalNumOfSamples = self.classPerWorker.sum()

        self.get_JSD(overallNumSamples/float(totalNumOfSamples), self.classPerWorker, [0] * numOfClients)

    def partitionTraceSpeech(self, dataToClient):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = 35

        # data share the same index with labels

        for index, sample in enumerate(self.data.data):
            clientId = dataToClient[sample]
            labelId = self.labels[index]

            if clientId not in clientToData:
                clientToData[clientId] = []
                clientNumSamples[clientId] = [0] * numOfLabels

            clientToData[clientId].append(index)
            clientNumSamples[clientId][labelId] += 1

        numOfClients = len(clientToData.keys())
        self.classPerWorker = np.zeros([numOfClients, numOfLabels])

        for clientId in range(numOfClients):
            #logging.info(clientId)
            self.classPerWorker[clientId] = clientNumSamples[clientId]
            self.rng.shuffle(clientToData[clientId])
            self.partitions.append(clientToData[clientId])

        overallNumSamples = np.asarray(self.classPerWorker.sum(axis=0)).reshape(-1)
        totalNumOfSamples = self.classPerWorker.sum()

        self.get_JSD(overallNumSamples/float(totalNumOfSamples), self.classPerWorker, [0] * numOfClients)

    def partitionTraceNLP(self):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = 1
        base = 0
        numOfClients = 0

        numOfLabels = self.args.num_class
        for index, cId in enumerate(self.data.dict.keys()):
            clientId = cId
            labelId = self.data.targets[index]

            if clientId not in clientToData:
                clientToData[clientId] = []
                clientNumSamples[clientId] = [0] * numOfLabels
            clientToData[clientId].append(index)

        numOfClients = len(self.clientToData)

    def partitionTraceBase(self):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = self.args.num_class

        clientToData = self.data.client_mapping
        for clientId in clientToData:
            clientNumSamples[clientId] = [1] * numOfLabels

        numOfClients = len(clientToData)
        self.classPerWorker = np.zeros([numOfClients+1, numOfLabels])

        for clientId in range(numOfClients):
            self.classPerWorker[clientId] = clientNumSamples[clientId]
            self.rng.shuffle(clientToData[clientId])
            self.partitions.append(clientToData[clientId])

            # if len(clientToData[clientId]) < args.filter_less or len(clientToData[clientId]) > args.filter_more:
            #     # mask the raw data
            #     for idx in clientToData[clientId]:
            #         self.data[idx] = None

        overallNumSamples = np.asarray(self.classPerWorker.sum(axis=0)).reshape(-1)
        totalNumOfSamples = self.classPerWorker.sum()

        self.get_JSD(overallNumSamples/float(totalNumOfSamples), self.classPerWorker, [0] * numOfClients)

    def partitionDataByDefault(self, sizes, sequential, ratioOfClassWorker, filter_class, _args):
        if self.is_trace and not self.args.enforce_random:
            # use the real trace, thus no need to partition
            if self.task == 'speech' or self.task == 'cv':
                dataToClient = OrderedDict()

                with open(self.dataMapFile, 'rb') as db:
                    dataToClient = pickle.load(db)

                if self.task == 'speech':
                    self.partitionTraceSpeech(dataToClient=dataToClient)
                else:
                    self.partitionTraceCV(dataToClient=dataToClient)
            else:
                self.partitionTraceBase()
        else:
            self.partitionData(sizes=sizes, sequential=sequential,
                               ratioOfClassWorker=ratioOfClassWorker,
                               filter_class=filter_class, args=_args)

    def partitionData(self, sizes=None, sequential=0, ratioOfClassWorker=None, filter_class=0, args = None):
        targets = self.getTargets()
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()

        usedSamples = 100000

        keyDir = {key:int(key) for i, key in enumerate(targets.keys())}
        keyLength = [0] * numOfLabels

        if not self.skip_partition:
            for key in keyDir.keys():
                keyLength[keyDir[key]] = len(targets[key])

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

        # random partition
        if sequential == 0:
            logging.info("========= Start of Random Partition =========\n")

            # may need to filter ...
            indicesToRm = set()
            indexes = None
            if self.args.filter_less != 0 and self.isTest is False:
                if self.task == 'cv':
                    indicesToRm = set(self.loadFilterInfo())
                else:
                    indicesToRm = set(self.loadFilterInfoBase())

                indexes = [x for x in range(0, data_len) if x not in indicesToRm]
                # we need to remove those with less than certain number of samples
                logging.info("====Try to remove clients w/ less than {} samples, and remove {} samples".format(self.args.filter_less, len(indicesToRm)))
            else:
                indexes = [x for x in range(data_len)]

            self.rng.shuffle(indexes)
            realDataLen = len(indexes)

            for ratio in sizes:
                part_len = int(ratio * realDataLen)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

            if not self.skip_partition:
                for id, partition in enumerate(self.partitions):
                    for index in partition:
                        tempClassPerWorker[id][self.indexToLabel[index]] += 1
        else:
            logging.info('========= Start of Class/Worker =========\n')

            if ratioOfClassWorker is None:
                # random distribution
                if sequential == 1:
                    ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels)
                # zipf distribution
                elif sequential == 2:
                    ratioOfClassWorker = np.random.zipf(args['param'], [len(sizes), numOfLabels])
                    logging.info("==== Load Zipf Distribution ====\n {} \n".format(repr(ratioOfClassWorker)))
                    ratioOfClassWorker = ratioOfClassWorker.astype(np.float32)
                else:
                    ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)

            if filter_class > 0:
                for w in range(len(sizes)):
                    # randomly filter classes by forcing zero samples
                    wrandom = self.rng.sample(range(numOfLabels), filter_class)
                    for wr in wrandom:
                        ratioOfClassWorker[w][wr] = 0.001

            # normalize the ratios
            if sequential == 1 or sequential == 3:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
                for worker in range(len(sizes)):
                    ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :]/float(sumRatiosPerClass[worker])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(floor(usedSamples * ratioOfClassWorker[worker][keyDir[c]]), keyLength[keyDir[c]])
                        self.rng.shuffle(targets[c])
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength

                    self.rng.shuffle(self.partitions[-1])
            elif sequential == 2:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=0)
                for c in targets.keys():
                    ratioOfClassWorker[:, keyDir[c]] = ratioOfClassWorker[:, keyDir[c]]/float(sumRatiosPerClass[keyDir[c]])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(int(math.ceil(keyLength[keyDir[c]] * ratioOfClassWorker[worker][keyDir[c]])), len(targets[c]))
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                    self.rng.shuffle(self.partitions[-1])

            elif sequential == 4:
                # load data from given config file
                clientGivenSamples = {}
                with open(args['clientSampleConf'], 'r') as fin:
                    for clientId, line in enumerate(fin.readlines()):
                        clientGivenSamples[clientId] = [int(x) for x in line.strip().split()]

                # split the data
                for clientId in range(len(clientGivenSamples.keys())):
                    self.partitions.append([])

                    for c in list(targets.keys()):
                        takeLength = clientGivenSamples[clientId][c]
                        if clientGivenSamples[clientId][c] > targets[c]:
                            logging.info("========== Failed to allocate {} samples for class {} to client {}, actual quota is {}"\
                                .format(clientGivenSamples[clientId][c], c, clientId, targets[c]))
                            takeLength = targets[c]

                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                self.rng.shuffle(self.partitions[-1])

        # concatenate ClassPerWorker
        if self.classPerWorker is None:
            self.classPerWorker = tempClassPerWorker
        else:
            self.classPerWorker = np.concatenate((self.classPerWorker, tempClassPerWorker), axis=0)

        # Calculates statistical distances
        totalDataSize = max(sum(keyLength), 1)
        # Overall data distribution
        dataDistr = np.array([key / float(totalDataSize) for key in keyLength])
        self.get_JSD(dataDistr, tempClassPerWorker, sizes)

        logging.info("Raw class per worker is : " + repr(tempClassPerWorker) + '\n')
        logging.info('========= End of Class/Worker =========\n')

    def log_selection(self):

        # totalLabels = [0 for i in range(len(self.classPerWorker[0]))]
        # logging.info("====Total # of workers is :{}, w/ {} labels, {}, {}".format(len(self.classPerWorker), len(self.classPerWorker[0]), len(self.partitions), len(self.workerDistance)))

        # for index, row in enumerate(self.classPerWorker):
        #     rowStr = ''
        #     numSamples = 0
        #     for i, label in enumerate(self.classPerWorker[index]):
        #         rowStr += '\t'+str(int(label))
        #         totalLabels[i] += label
        #         numSamples += label

        #     logging.info(str(index) + ':\t' + rowStr + '\n' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index]))+ '\nDistance: ' + str(self.workerDistance[index])+ '\n')
        #     logging.info("=====================================\n")

        # logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        # logging.info("=====================================\n")

        # remove unused variables

        self.classPerWorker = None
        self.numOfLabels = None
        pass

    def use(self, partition, istest, is_rank, fractional):
        _partition = partition
        resultIndex = []

        if is_rank == -1:
            resultIndex = self.partitions[_partition]
        else:
            for i in range(len(self.partitions)):
                if i % self.args.total_worker == is_rank:
                    resultIndex += self.partitions[i]

        exeuteLength = -1 if istest == False or fractional == False else int(len(resultIndex) * args.test_ratio)

        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        #logging.info("====Data length for client {} is {}".format(partition, len(resultIndex)))
        return Partition(self.data, resultIndex)

    def getDistance(self):
        return self.workerDistance

    def getSize(self):
        # return the size of samples
        return [len(partition) for partition in self.partitions]

def partition_dataset(partitioner, workers, partitionRatio=[], sequential=0, ratioOfClassWorker=None, filter_class=0, arg={'param': 1.95}):
    """ Partitioning Data """
    stime = time.time()
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    if len(partitionRatio) > 0:
        partition_sizes = partitionRatio

    partitioner.partitionDataByDefault(sizes=partition_sizes, sequential=sequential, ratioOfClassWorker=ratioOfClassWorker,filter_class=filter_class, _args=arg)
    #logging.info("====Partitioning data takes {} s\n".format(time.time() - stime()))

def select_dataset(rank: int, partition: DataPartitioner, batch_size: int, isTest=False, is_rank=0, fractional=True, collate_fn=None):

    partition = partition.use(rank - 1, isTest, is_rank-1, fractional)
    timeOut = 0 if isTest else 60
    numOfThreads = args.num_loaders #int(min(args.num_loaders, len(partition)/(batch_size+1)))
    dropLast = False if isTest else True

    if collate_fn is None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=numOfThreads, drop_last=dropLast, timeout=timeOut)#, worker_init_fn=np.random.seed(12))
    else:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=numOfThreads, drop_last=dropLast, timeout=timeOut, collate_fn=collate_fn)#, worker_init_fn=np.random.seed(12))

